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
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os

# ============================================================================
# 1. DATASET LOADER (Separates X from Survival Targets)
# ============================================================================
class ClinicalDataset(Dataset):
    def __init__(self, data_path, mask_path=None, target_cols=['duration', 'event']):
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
            
        # Helper: Try to find survival columns (generic)
        self.target_cols = target_cols  # Store for later use
        self.targets = self.raw_df[target_cols].values.astype(np.float32)
        
        # Drop non-numeric columns AND ID columns
        id_cols = ['hadm_id', 'subject_id', 'stay_id', 'icustay_id', 'patient_id']
        cols_to_drop = [col for col in id_cols if col in self.raw_df.columns]
        
        feature_df_numeric = self.raw_df.drop(columns=target_cols + cols_to_drop).select_dtypes(include=[np.number])
        dropped_cols = set(self.raw_df.columns) - set(feature_df_numeric.columns) - set(target_cols)
        feature_df_numeric = self.raw_df.drop(columns=target_cols + cols_to_drop).select_dtypes(include=[np.number])
        
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
            
        # 3. Manual Min-Max Scaling (Robust to NaNs)
        self.min_val = np.nanmin(self.data, axis=0)
        self.max_val = np.nanmax(self.data, axis=0)
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
    loss = -torch.sum(loss_vector) / (torch.sum(events) + 1e-6) # Normalize
    
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
def impute_dataset(model, data_path, mask_path, output_path):
    """
    Generate imputed dataset using trained model.
    Args:
        model: Trained SurvivalVAE model
        data_path: Path to MNAR data CSV
        mask_path: Path to mask CSV
        output_path: Where to save imputed CSV
    """
    print(f"Imputing dataset: {data_path}")
    dataset = ClinicalDataset(data_path, mask_path)
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
def train_model(dataset_path, mask_path=None, epochs=50, beta=1.0, gamma=1.0, latent_dim=10, lr=1e-3):
    """
    beta: Weight for KL Divergence (default 1.0)
    gamma: Weight for Survival Loss (default 1.0)
    """
    print(f"Loading data from {dataset_path}...")
    dataset = ClinicalDataset(dataset_path, mask_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
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
            sorted_indices = torch.argsort(t, dim=0, descending=True).reshape(-1)
            x = x[sorted_indices]
            m = m[sorted_indices]
            t = t[sorted_indices]
            e = e[sorted_indices]
            
            if x.dim() == 1: x = x.unsqueeze(0)
            if m.dim() == 1: m = m.unsqueeze(0)
            if t.dim() == 1: t = t.unsqueeze(1) 
            if e.dim() == 1: e = e.unsqueeze(1) 
            
            optimizer.zero_grad()
            recon_x, mu, logvar, risk = model(x, m)
            
            # 1. Reconstruction Loss (MSE on OBSERVED values only) - NORMALIZED to per-patient
            mse = F.mse_loss(recon_x, x, reduction='none')
            num_obs = m.sum()
            avg_mse = (mse * m).sum() / (num_obs + 1e-8)
            recon_loss = avg_mse * model.input_dim
            
            # 2. KL Divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            
            # 3. Survival Loss (Cox)
            surv_loss = cox_ph_loss(risk, e)
            
            # Total Loss
            loss = recon_loss + beta * kl_loss + gamma * surv_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_recon += recon_loss.item()
            total_surv += surv_loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Recon: {recon_loss.item():.4f} | KL: {(beta*kl_loss).item():.4f} | Surv: {(gamma*surv_loss).item():.4f}")
        
    print("Training Complete.")
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Survival-VAE Imputation')
    parser.add_argument('--data', type=str, help='Path to MNAR data CSV')
    parser.add_argument('--mask', type=str, help='Path to mask CSV')
    parser.add_argument('--output', type=str, help='Path to save imputed CSV')
    parser.add_argument('--model_save', type=str, default='survival_vae.pth', help='Path to save model')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--beta', type=float, default=1.0, help='KL divergence loss weight')
    parser.add_argument('--gamma', type=float, default=1.0, help='Survival loss weight')
    parser.add_argument('--dataset', type=str, choices=['metabric', 'mimic', 'both'], default='both',
                        help='Which dataset to process: metabric, mimic, or both')
    parser.add_argument('--json_params', type=str, help='Path to JSON file with optimized hyperparameters')
    args = parser.parse_args()
    
    # Load optimized params if provided
    opt_params = {}
    if args.json_params and os.path.exists(args.json_params):
        import json
        with open(args.json_params, 'r') as f:
            opt_params = json.load(f)
        print(f"Loaded optimized parameters from {args.json_params}")
        args.beta = opt_params.get('beta', args.beta)
        args.gamma = opt_params.get('gamma', args.gamma)
        args.latent_dim = opt_params.get('latent_dim', 10)
        args.lr = opt_params.get('lr', 1e-3)
    else:
        args.latent_dim = 10
        args.lr = 1e-3
    
    # Default paths if not provided
    POSSIBLE_DATA_DIRS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets')),  # Relative path
    ]

    DATA_DIR = None
    for path in POSSIBLE_DATA_DIRS:
        if os.path.exists(path):
            DATA_DIR = path
            break
            
    if DATA_DIR is None:
        # Fallback to current relative if nothing found, though likely to fail later if not present
        DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets'))
        print(f"Warning: Could not find datasets in expected locations. Defaulting to: {DATA_DIR}")
    else:
        print(f"Using datasets at: {DATA_DIR}")
    
    if args.data is None:
        # Run on ALL Datasets and ALL Scenarios
        print(f"No arguments provided. Running pipeline for {args.dataset.upper()}...")
        
        # Select datasets based on argument
        if args.dataset == 'both':
            datasets = ['metabric', 'mimic']
        else:
            datasets = [args.dataset]
        scenarios = ['light', 'moderate', 'severe']
        
        # Set random seeds for reproducibility
        import random
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        for dataset_name in datasets:
            print(f"\n{'-'*60}")
            print(f"PROCESSING DATASET: {dataset_name.upper()}")
            print(f"{'-'*60}")
            
            for severity in scenarios:
                mnar_path = os.path.join(DATA_DIR, f'{dataset_name}_mnar_{severity}.csv')
                mask_path = os.path.join(DATA_DIR, f'{dataset_name}_mask_{severity}.csv')
                output_path = os.path.join(DATA_DIR, f'{dataset_name}_imputed_survival_vae_{severity}.csv')
                model_save_path = f'survival_vae_{dataset_name}_{severity}.pth'
                
                if os.path.exists(mnar_path):
                    print(f"\n{'-'*40}")
                    print(f"Scenario: {severity.upper()}")
                    print(f"Input: {mnar_path}")
                    print(f"Output: {output_path}")
                    print(f"{'-'*40}")
                    
                    # 1. Train model
                    model = train_model(mnar_path, mask_path, epochs=args.epochs, beta=args.beta, gamma=args.gamma, latent_dim=args.latent_dim, lr=args.lr)
                    
                    # 2. Save model
                    torch.save(model.state_dict(), model_save_path)
                    print(f"\nModel saved to: {model_save_path}")
                    
                    # 3. Generate imputed dataset
                    impute_dataset(model, mnar_path, mask_path, output_path)
                else:
                    print(f"\nSkipping {dataset_name.upper()} {severity.upper()} - Data not found: {mnar_path}")
        
        print(f"\n{'-'*60}")
        print(f"All Datasets and Scenarios Processed!")
        print(f"{'-'*60}")
        
    else:
        # Run on single provided file
        mnar_path = args.data
        mask_path = args.mask
        output_path = args.output
    
        if os.path.exists(mnar_path):
            print(f"-" * 60)
            print(f"Survival-VAE Imputation Pipeline (Single Run)")
            print(f"-" * 60)
            print(f"Input Data: {mnar_path}")
            print(f"Mask File: {mask_path}")
            print(f"Output: {output_path}")
            print(f"Epochs: {args.epochs}, Beta: {args.beta}, Gamma: {args.gamma}")
            print(f"=" * 60)
            
            # 1. Train model
            model = train_model(mnar_path, mask_path, epochs=args.epochs, beta=args.beta, gamma=args.gamma, latent_dim=args.latent_dim, lr=args.lr)
            
            # 2. Save model
            torch.save(model.state_dict(), args.model_save)
            print(f"\nModel saved to: {args.model_save}")
            
            # 3. Generate imputed dataset
            impute_dataset(model, mnar_path, mask_path, output_path)
            
            print(f"\n{'-'*60}")
            print(f"Pipeline Complete!")
            print(f"{'-'*60}")
        else:
            print(f"Error: Data not found at {mnar_path}")
            print("Run 'simulate_metabric_mnar.py' or 'simulate_mimic_mnar.py' first.")
