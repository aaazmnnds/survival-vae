"""
Survival-VAE: Variational Autoencoder for MNAR Imputation + Survival Analysis
=============================================================================
This script defines the Deep Generative Model used to solve the "Threshold-Interaction MNAR" problem.
It transforms a standard VAE into a "Survival-VAE" by adding a Cox Proportional Hazards layer
to the latent space, ensuring that the learned embeddings are NOT JUST for reconstruction,
but also for PROGNOSIS.

Key Components:
1. ClinicalDataset: Handles Features (X) vs Targets (T, E)
2. SurvivalVAE: Encoder -> Latent -> Decoder + SurvivalNetwork
3. CoxPHLoss: Implementation of Negative Log Partial Likelihood
4. Training Loop: Optimizes L_total = L_recon + beta * L_kl + gamma * L_surv

Author: Research Assistant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import numpy as np
import pandas as pd
import os
import argparse
import json

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
            
        print(f"DEBUG: Using survival targets: {target_cols}") # Debug log
        self.target_cols = target_cols  # Store for later use
        self.targets = self.raw_df[target_cols].values.astype(np.float32)
        
        # Drop non-numeric columns AND ID columns
        id_cols = ['hadm_id', 'subject_id', 'stay_id', 'icustay_id', 'patient_id']
        cols_to_drop = [col for col in id_cols if col in self.raw_df.columns]
        
        feature_df_numeric = self.raw_df.drop(columns=target_cols + cols_to_drop).select_dtypes(include=[np.number])
        dropped_cols = set(self.raw_df.columns) - set(feature_df_numeric.columns) - set(target_cols)
        if dropped_cols:
            print(f"DEBUG: Dropped non-feature columns: {dropped_cols}")
        
        # Store ID columns for later reconstruction
        self.id_data = self.raw_df[cols_to_drop] if cols_to_drop else None
            
        self.data = feature_df_numeric.values.astype(np.float32)
        self.feature_columns = feature_df_numeric.columns # Save for reconstruction
        
        # 2. Handle Mask
        if mask_path and os.path.exists(mask_path):
            mask_df = pd.read_csv(mask_path)
            # We ONLY use the mask file to determine which columns to keep/drop
            if set(target_cols).issubset(mask_df.columns):
                mask_df = mask_df.drop(columns=target_cols)
            mask_df = mask_df[feature_df_numeric.columns]
            
            # CRITICAL FIX: The mask for VAE training MUST be based on ACTUAL DATA availability.
            # Natural Missing values (NaN in input) must be treated as missing (0), NOT observed.
            # The mask file only tracks Artificially Removed values, so using it blindly 
            # treats Natural Missing as "Observed" (which breaks imputation).
            self.mask = (~np.isnan(self.data)).astype(np.float32)
        else:
            self.mask = (~np.isnan(self.data)).astype(np.float32)
            
        # 3. Fix Scaler: Fit on train rows only, transform all
        self.min_val = np.nanmin(self.data[train_idx], axis=0)
        self.max_val = np.nanmax(self.data[train_idx], axis=0)
        self.scale = self.max_val - self.min_val
        self.scale[self.scale < 1e-6] = 1.0 # Avoid division by zero
        
        self.data = (self.data - self.min_val) / self.scale
        
        # 4. Fill NaNs for Input (Zero Imputation for initial pass)
        # Note: After scaling, 0.0 is meaningful?
        # Usually for VAE, we want missing values to not affect conv?
        # But we use linear layers.
        # Zero filling is acceptable if 0 is within range [0, 1].
        # Since we subtracted min, 0.0 corresponds to min_val.
        # Ideally we fill with 0.5 or mean (which is roughly 0.5 after scaling)?
        # For simplicity, keep nan_to_num(0).
        # Fill NaNs with 0.5 (middle of [0,1] range after scaling) for better initialization
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
    Args:
        risk_scores (Tensor): Predicted risk (log-hazard), shape [B, 1]
        events (Tensor): Event indicators (1=Dead, 0=Censored), shape [B, 1]
    Note: Assumes inputs are ALREADY SORTED by Time (Descending)!
    """
    # Risk scores exp(h(x))
    # Stability Check: Clip risk scores to avoid overflow
    risk_scores = torch.clamp(risk_scores, min=-10, max=10)
    risk_exp = torch.exp(risk_scores)
    
    # Cumulative Sum (reversed because sorted descending)
    # cumsum[i] = sum_{j=0 to i} exp(h(j)) -> Risk Set R(t_i)
    # Since we sorted descending, R(t_i) includes all indices 0..i
    risk_cumsum = torch.cumsum(risk_exp, dim=0)
    
    # Log of risk set sums
    log_risk_cumsum = torch.log(risk_cumsum + 1e-8)
    
    # We only care about instances where Event=1
    # L = sum( h_i - log(sum_{j in R} exp(h_j)) )
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
        # Ensure 2D tensors (handle edge cases)
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
        risk_score = self.surv_net(z) # Can also use mu for deterministic risk
        
        return recon_x, mu, logvar, risk_score

# ============================================================================
# 4. IMPUTATION FUNCTION
# ============================================================================
def impute_dataset(model, data_path, train_idx, mask_path, output_path):
    """
    Generate imputed dataset using trained model.
    """
    print(f"  Generating final imputation...")
    dataset = ClinicalDataset(data_path, train_idx=train_idx, mask_path=mask_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    model.eval()
    imputed_values = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['x']
            m = batch['m']
            
            # Forward pass
            recon_x, _, _, _ = model(x, m)
            
            # Combine: Keep observed values, use reconstructions for missing
            # imputed = observed * mask + reconstructed * (1 - mask)
            imputed_x_scaled = x * m + recon_x * (1 - m)
            
            # Inverse Transform!
            imputed_x = dataset.inverse_transform(imputed_x_scaled.numpy())
            
            imputed_values.append(imputed_x)
    
    # Reconstruct DataFrame
    imputed_array = np.vstack(imputed_values)
    imputed_df = pd.DataFrame(imputed_array, columns=dataset.feature_columns)
    
    # Add back ID columns (if they exist)
    if dataset.id_data is not None:
        for col in dataset.id_data.columns:
            imputed_df.insert(0, col, dataset.id_data[col].values)
    
    # Add back survival targets (using dynamic column names)
    imputed_df[dataset.target_cols[0]] = dataset.targets[:, 0]
    imputed_df[dataset.target_cols[1]] = dataset.targets[:, 1]
    
    # Save
    imputed_df.to_csv(output_path, index=False)
    print(f"Imputed dataset saved to: {output_path}")
    print(f"Shape: {imputed_df.shape}")
    print(f"Missing values: {imputed_df.isnull().sum().sum()}")

# ============================================================================
# 5. TRAINING LOOP
# ============================================================================
def train_model(dataset_path, train_idx, mask_path=None, epochs=50, latent_dim=10, beta=1.0, gamma=1.0, lr=1e-3, binary_feature_indices=None):
    """
    beta: Weight for KL Divergence (default 1.0)
    gamma: Weight for Survival Loss (default 1.0)
    """
    dataset = ClinicalDataset(dataset_path, train_idx=train_idx, mask_path=mask_path)
    sampler = SubsetRandomSampler(train_idx)
    dataloader = DataLoader(dataset, batch_size=512, sampler=sampler)
    
    input_dim = dataset.data.shape[1]
    
    model = SurvivalVAE(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting Training (Beta={beta}, Gamma={gamma})...")
    model.train()
    
    for epoch in range(epochs):
        total_recon = 0
        total_surv = 0
        
        for batch in dataloader:
            x = batch['x']
            m = batch['m']
            t = batch['t']
            e = batch['e']
            
            # --- IMPT: SORT BATCH BY TIME FOR COX LOSS ---
            # Cox Loss assumes descending order of Time
            sorted_indices = torch.argsort(t, dim=0, descending=True).reshape(-1)
            x = x[sorted_indices]
            m = m[sorted_indices]
            t = t[sorted_indices]
            e = e[sorted_indices]
            
            # Ensure 2D shapes (in case batch_size=1)
            if x.dim() == 1: x = x.unsqueeze(0)
            if m.dim() == 1: m = m.unsqueeze(0)
            if t.dim() == 1: t = t.unsqueeze(1) # shape [B, 1]
            if e.dim() == 1: e = e.unsqueeze(1) # shape [B, 1]
            
            optimizer.zero_grad()
            
            # Forward
            recon_x, mu, logvar, risk = model(x, m)
            
            # 1. Reconstruction Loss (MSE on OBSERVED values only)
            # CRITICAL: We multiply by 'm' so we don't train on missing values (which are filled with placeholders)
            # Using reduction='none' gives element-wise errors
            element_wise_loss = F.mse_loss(recon_x, x, reduction='none')
            masked_loss = element_wise_loss * m
            
            # Sum over features, Mean over batch
            # We normalize by total observed values? Or just standard sum(dim=1).mean()?
            # Standard VAE is usually sum(dim=1).mean() with 0s for missing
            recon_loss = masked_loss.sum(dim=1).mean()
            
            # 2. KL Loss (Sum over latent dims, Mean over batch)
            # Matches standard VAE ELBO formulation
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            
            # 3. Survival Loss (Cox)
            surv_loss = cox_ph_loss(risk, e)
            
            # Total Loss (using beta parameter instead of hardcoded 0.01)
            loss = recon_loss + beta * kl_loss + gamma * surv_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_recon += recon_loss.item()
            total_surv += surv_loss.item()
            
        if (epoch+1) % 5 == 0:
            avg_recon = total_recon / len(dataloader)
            avg_surv = total_surv / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} | Recon: {avg_recon:.4f} | Surv: {avg_surv:.4f}")
        
    print("Training Complete.")
    return model

def run_survival_vae_pipeline(dataset_name, severity, fold_idx):
    print(f"Starting Survival-VAE Imputation - Fold {fold_idx}...")
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    POSSIBLE_DATA_DIRS = [
        './datasets',
        'datasets',
        '../datasets',
        '/home/azman/VAE/Survival-VAE_study',
        '/Users/azmannads/VAE 2/Survival-VAE_study',
        '/Users/azmannads/Documents/Research collections/Research 2025/datasets',
        '.'
    ]

    # Optuna results are now standardized within the study folder structure

    def find_dir(possibilities, default_name):
        for p in possibilities:
            if os.path.exists(p):
                return p
        return default_name

    data_dir = find_dir(POSSIBLE_DATA_DIRS, 'datasets')
    optuna_dir = os.path.join(data_dir, 'final', 'optuna_results_final')
    
    # Update output path structure to include fold index
    output_dir = os.path.join(data_dir, 'final', 'imputation_results_final', f'fold_{fold_idx}')
    os.makedirs(output_dir, exist_ok=True)
    
    mnar_path = os.path.join(data_dir, f'{dataset_name}_mnar_{severity}.csv')
    mask_path = os.path.join(data_dir, f'{dataset_name}_mask_{severity}.csv')
    
    # Optuna Path Logic: Current Fold -> Fold 0 -> Root Fallback
    optuna_file = os.path.join(optuna_dir, f'fold_{fold_idx}', f'{dataset_name}_{severity}_optuna_survival_vae_results.csv')
    if not os.path.exists(optuna_file):
        # Fallback to fold_0 (Methodology: Optimize on fold 0 only)
        optuna_file = os.path.join(optuna_dir, 'fold_0', f'{dataset_name}_{severity}_optuna_survival_vae_results.csv')
    
    if not os.path.exists(optuna_file):
        # Fallback to root for backward compatibility
        optuna_file = os.path.join(optuna_dir, f'{dataset_name}_{severity}_optuna_survival_vae_results.csv')
    
    output_path = os.path.join(output_dir, f'{dataset_name}_{severity}_survival_vae_imputed.csv')
    
    if not os.path.exists(mnar_path):
        print(f"Error: File not found: {mnar_path}")
        return

    print(f"\nProcessing {dataset_name} | {severity} | Fold {fold_idx}")
    
    # 0. Load Split Indices
    splits_path = os.path.join(data_dir, f'cv_splits_{dataset_name}.json')
    if not os.path.exists(splits_path):
        splits_path = os.path.join(os.path.dirname(mnar_path), f'cv_splits_{dataset_name}.json')
        
    with open(splits_path, 'r') as f:
        cv_splits = json.load(f)
    fold_key = f'fold_{fold_idx + 1}'
    train_idx = np.array(cv_splits[fold_key]['train'])
    val_idx   = np.array(cv_splits[fold_key]['val'])
    
    # 1. Load Optimized Params
    print(f"  Searching for Optuna results at: {optuna_file}")
    if os.path.exists(optuna_file):
        df_optuna = pd.read_csv(optuna_file)
        params = df_optuna.iloc[0]
        beta = float(params['beta'])
        gamma = float(params.get('gamma', 1.0))
        latent_dim = int(params.get('latent_dim', 10))
        lr = float(params['learning_rate'])
        epochs = 100
        print(f"  Loaded Optimized: Beta={beta:.4f}, Gamma={gamma:.4f}, Latent={latent_dim}, LR={lr:.6f}, Epochs={epochs}")
    else:
        beta, gamma, latent_dim, lr, epochs = 1.0, 1.0, 10, 1e-3, 100
        print(f"  Warning: Optuna file not found. Using defaults.")
    
    # 2. Train model
    temp_df = pd.read_csv(mnar_path, nrows=0)
    drop_cols = ['subject_id', 'hadm_id', 'stay_id', 'patient_id', 'Survival_in_days', 'Status', 'duration', 'event', 'Time', 'Event']
    feature_cols = [c for c in temp_df.columns if c not in drop_cols]
    
    KNOWN_BINARY_FEATURES = [
        'Hormone_Tx', 'Radiotherapy', 'Chemotherapy', 'ER_Positive',
        'Sex', 'CCI_MI', 'CCI_CHF', 'CCI_PVD', 'CCI_Stroke',
        'CCI_Renal', 'CCI_Liver', 'CCI_Cancer'
    ]
    binary_feature_indices = [
        i for i, col in enumerate(feature_cols)
        if col in KNOWN_BINARY_FEATURES
    ]
    
    model = train_model(mnar_path, train_idx, mask_path, epochs=epochs, latent_dim=latent_dim, beta=beta, gamma=gamma, lr=lr, binary_feature_indices=binary_feature_indices)
    model.to(device)
    
    # 3. Generate imputed dataset
    impute_dataset(model.cpu(), mnar_path, train_idx, mask_path, output_path)

    print(f"\n{'='*60}")
    print(f"Survival-VAE Complete for {dataset_name} {severity} Fold {fold_idx}!")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Survival-VAE Imputation for a specific fold.')
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4], help='CV fold index (0-4)')
    parser.add_argument('--dataset', type=str, required=True, choices=['metabric', 'mimic'], help='Dataset name')
    parser.add_argument('--scenario', type=str, required=True, choices=['light', 'moderate', 'severe'], help='Imputation scenario')
    args = parser.parse_args()
    
    run_survival_vae_pipeline(args.dataset, args.scenario, args.fold)
