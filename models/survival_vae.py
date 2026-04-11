import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

# ============================================================================
# 1. DATASET LOADER (Separates X from Survival Targets)
# ============================================================================
class ClinicalDataset(Dataset):
    def __init__(self, data_path, train_idx, mask_path=None, target_cols=['duration', 'event']):
        """
        Args:
            data_path: Path to .csv file.
            mask_path: Optional path to mask .csv file.
            target_cols: List of column names to treat as Survival Targets (not inputs).
        """
        self.raw_df = pd.read_csv(data_path)
        
        # Encode 'Sex' if present so it isn't dropped
        if 'Sex' in self.raw_df.columns:
            self.raw_df['Sex'] = self.raw_df['Sex'].map({'F': 0, 'M': 1, 'Female': 0, 'Male': 1})
        
        # Helper: Try to find survival columns (generic)
        if 'duration' in self.raw_df.columns and 'event' in self.raw_df.columns:
            target_cols = ['duration', 'event']
        elif 'Time' in self.raw_df.columns and 'Event' in self.raw_df.columns:
            target_cols = ['Time', 'Event']
        elif 'Survival_in_days' in self.raw_df.columns and 'Status' in self.raw_df.columns:
            target_cols = ['Survival_in_days', 'Status']
        else:
            raise KeyError("Could not find survival columns (duration/event or Time/Event)")
            
        self.target_cols = target_cols  # Store for later use
        self.targets = self.raw_df[target_cols].values.astype(np.float32)
        
        # Drop non-numeric columns AND ID columns
        id_cols = ['hadm_id', 'subject_id', 'stay_id', 'icustay_id', 'patient_id']
        cols_to_drop = [col for col in id_cols if col in self.raw_df.columns]
        
        feature_df_numeric = self.raw_df.drop(columns=target_cols + cols_to_drop).select_dtypes(include=[np.number])
        
        # Store ID columns for later reconstruction
        self.id_data = self.raw_df[cols_to_drop] if cols_to_drop else None
            
        self.data = feature_df_numeric.values.astype(np.float32)
        self.feature_columns = feature_df_numeric.columns # Save for reconstruction
        
        # 2. Handle Mask
        if mask_path and os.path.exists(mask_path):
            self.mask = (~np.isnan(self.data)).astype(np.float32)
        else:
            self.mask = (~np.isnan(self.data)).astype(np.float32)
            
        # 3. Fix Scaler: Fit on train rows only, transform all
        self.min_val = np.nanmin(self.data[train_idx], axis=0)
        self.max_val = np.nanmax(self.data[train_idx], axis=0)
        self.scale = self.max_val - self.min_val
        self.scale[self.scale < 1e-6] = 1.0 # Avoid division by zero
        
        self.data = (self.data - self.min_val) / self.scale
        
        # 4. Fill NaNs with 0.5 (middle of [0,1] range after scaling) for better initialization
        self.data_filled = np.nan_to_num(self.data, nan=0.5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'x': torch.FloatTensor(self.data_filled[idx]),
            'm': torch.FloatTensor(self.mask[idx]),
            't': torch.FloatTensor([self.targets[idx, 0]]), # Duration
            'e': torch.FloatTensor([self.targets[idx, 1]]), # Event
            'idx': idx
        }

    def inverse_transform(self, scaled_data):
        return scaled_data * self.scale + self.min_val

# ============================================================================
# 2. COX PARTIAL LIKELIHOOD LOSS
# ============================================================================
def cox_ph_loss(risk_scores, events):
    """
    Computes Negative Log Partial Likelihood for Cox Proportional Hazards.
    Assumes inputs are ALREADY SORTED by Time (Descending)!
    """
    # Risk scores exp(h(x))
    risk_scores = torch.clamp(risk_scores, min=-10, max=10)
    risk_exp = torch.exp(risk_scores)
    
    # Cumulative Sum
    risk_cumsum = torch.cumsum(risk_exp, dim=0)
    
    # Log of risk set sums
    log_risk_cumsum = torch.log(risk_cumsum + 1e-8)
    
    # Negative L for minimization
    loss_vector = (risk_scores - log_risk_cumsum) * events
    loss = -torch.sum(loss_vector) / (torch.sum(events) + 1e-8) # Normalize
    
    return loss

# ============================================================================
# 3. VAE + SURVIVAL NETWORK ARCHITECTURE
# ============================================================================
class SurvivalNetwork(nn.Module):
    """Simple MLP predicting Risk Score from Latent Z"""
    def __init__(self, latent_dim):
        super(SurvivalNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1) # Output: Log Hazard (Scalar)
        )
        
    def forward(self, z):
        return self.net(z)

class SurvivalVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(SurvivalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 64), # Input: Data + Mask
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid() # Force output to [0,1] since we scaled data!
        )
        
        # --- Survival Network ---
        self.surv_net = SurvivalNetwork(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, m):
        # Ensure 2D tensors
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if m.dim() == 1:
            m = m.unsqueeze(0)
            
        # 1. Encode
        inputs = torch.cat([x, m], dim=1)
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # 2. Sample Z
        z = self.reparameterize(mu, logvar)
        
        # 3. Decode (Reconstruct X)
        recon_x = self.decoder(z)
        
        # 4. Predict Risk (Estimate T)
        risk_score = self.surv_net(z)
        
        return recon_x, mu, logvar, risk_score
