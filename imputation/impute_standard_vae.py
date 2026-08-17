"""
Standard VAE Imputation Script
==============================
Implements a standard Variational Autoencoder (VAE) for imputation.
Following Survival-VAE structure but:
1. NO survival network (no risk prediction).
2. Loss: L_total = L_recon + beta * L_KL (NO L_Cox).
3. T and E are included as input features for better imputation.

Strategy: Single Imputation (Deep Learning Standard)
Output: {dataset}_imputed_standard_vae_{scenario}.csv
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
from sklearn.preprocessing import MinMaxScaler

# ============================================================================
# 1. DATASET LOADER (Includes T, E in Features)
# ============================================================================
class ClinicalDataset(Dataset):
    def __init__(self, data_path, train_idx, target_cols=['duration', 'event']):
        self.raw_df = pd.read_csv(data_path)
        
        if 'Sex' in self.raw_df.columns:
            self.raw_df['Sex'] = self.raw_df['Sex'].map({'F': 0, 'M': 1, 'Female': 0, 'Male': 1})
        
        # Identify survival columns
        if 'Survival_in_days' in self.raw_df.columns:
            self.target_cols = ['Survival_in_days', 'Status']
        elif 'duration' in self.raw_df.columns:
            self.target_cols = ['duration', 'event']
        else:
            self.target_cols = ['Time', 'Event']
            
        # Identify IDs
        id_cols = ['hadm_id', 'subject_id', 'stay_id', 'icustay_id', 'patient_id', 'admittime', 'dischtime']
        self.cols_to_drop = [col for col in id_cols if col in self.raw_df.columns]
        
        # Features (Biological)
        feature_df = self.raw_df.drop(columns=self.target_cols + self.cols_to_drop).select_dtypes(include=[np.number])
        self.feature_columns = feature_df.columns
        
        # 3. Fix Scaler: Fit on train rows only, transform all (NaN-safe)
        X_raw = feature_df.values.astype(np.float32)
        x_min = np.nanmin(X_raw[train_idx], axis=0)
        x_max = np.nanmax(X_raw[train_idx], axis=0)
        x_scale = x_max - x_min
        x_scale[x_scale < 1e-6] = 1.0
        X_scaled = (X_raw - x_min) / x_scale
        self.x_min, self.x_scale = x_min, x_scale
        
        # Final augmented feature matrix
        self.data_all = X_scaled
        self.mask = (~np.isnan(self.data_all)).astype(np.float32)
        self.data_filled = np.nan_to_num(self.data_all, nan=0.5) # Initialize with middle
        
        # Store original targets and IDs
        self.targets_orig = self.raw_df[self.target_cols].copy()
        self.ids_orig = self.raw_df[self.cols_to_drop].copy() if self.cols_to_drop else None

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, idx):
        return {
            'x': torch.FloatTensor(self.data_filled[idx]),
            'm': torch.FloatTensor(self.mask[idx])
        }
        
    def inverse_transform(self, scaled_features):
        return scaled_features * self.x_scale + self.x_min

# ============================================================================
# 2. STANDARD VAE ARCHITECTURE
# ============================================================================
class StandardVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(StandardVAE, self).__init__()
        
        self.input_dim = input_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, m):
        inputs = torch.cat([x, m], dim=1)
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

# ============================================================================
# 3. TRAINING AND IMPUTATION
# ============================================================================
def run_standard_vae_imputation(dataset_name, severity, fold_idx):
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    data_dir = os.environ.get('SVAE_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets'))
    optuna_dir = os.path.join(data_dir, 'final', 'optuna_results_final')
    
    # Update output path structure to include fold index
    output_dir = os.path.join(data_dir, 'final', 'imputation_results_final', f'fold_{fold_idx}')
    os.makedirs(output_dir, exist_ok=True)
    
    mnar_path = os.path.join(data_dir, f'{dataset_name}_mnar_{severity}.csv')
    
    # Optuna Path Logic: Current Fold -> Fold 0 -> Root Fallback
    optuna_file = os.path.join(optuna_dir, f'fold_{fold_idx}', f'{dataset_name}_{severity}_optuna_standard_vae_results.csv')
    if not os.path.exists(optuna_file):
        # Fallback to fold_0 (Methodology: Optimize on fold 0 only)
        optuna_file = os.path.join(optuna_dir, 'fold_0', f'{dataset_name}_{severity}_optuna_standard_vae_results.csv')
        
    if not os.path.exists(optuna_file):
        # Fallback to root for backward compatibility
        optuna_file = os.path.join(optuna_dir, f'{dataset_name}_{severity}_optuna_standard_vae_results.csv')
    
    output_path = os.path.join(output_dir, f'{dataset_name}_{severity}_standard_vae_imputed.csv')
    
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
    
    # 2. Setup Data
    dataset = ClinicalDataset(mnar_path, train_idx=train_idx)
    sampler = SubsetRandomSampler(train_idx)
    dataloader = DataLoader(dataset, batch_size=512, sampler=sampler)
    
    # 1. Load Optimized Params
    if os.path.exists(optuna_file):
        df_optuna = pd.read_csv(optuna_file)
        params = df_optuna.iloc[0]
        beta = float(params['beta'])
        latent_dim = int(params['latent_dim'])
        lr = float(params['learning_rate'])
        epochs = 100  # Fixed epochs (not optimized in Optuna)
        print(f"  Loaded Optimized: Beta={beta:.6f}, Latent={latent_dim}, LR={lr:.6f}, Epochs={epochs}")
    else:
        beta, latent_dim, lr, epochs = 1.0, 10, 1e-3, 100
        print(f"  Warning: Optuna file not found. Using defaults.")

    # 2. Setup Data
    # (Moved up to load train_idx first)
    input_dim = dataset.data_all.shape[1]
    
    model = StandardVAE(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    KNOWN_BINARY_FEATURES = [
        'Hormone_Tx', 'Radiotherapy', 'Chemotherapy', 'ER_Positive',
        'Sex', 'CCI_MI', 'CCI_CHF', 'CCI_PVD', 'CCI_Stroke',
        'CCI_Renal', 'CCI_Liver', 'CCI_Cancer'
    ]
    binary_feature_indices = [
        i for i, col in enumerate(dataset.feature_columns)
        if col in KNOWN_BINARY_FEATURES
    ]

    # 3. Train
    print(f"  Training Standard VAE ({epochs} epochs)... ")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x, m = batch['x'].to(device), batch['m'].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x, m)
            
            # Recon Loss (Observed Only)
            num_features = x.shape[1]
            binary_cols = binary_feature_indices if binary_feature_indices is not None else []
            cont_cols = [i for i in range(num_features) if i not in binary_cols]
            total_loss_sum = 0.0
            if len(cont_cols) > 0:
                mse_loss = F.mse_loss(recon_x[:, cont_cols], x[:, cont_cols], reduction='none')
                masked_mse = mse_loss * m[:, cont_cols]
                total_loss_sum += masked_mse.sum()
            if len(binary_cols) > 0:
                bce_loss = F.binary_cross_entropy(recon_x[:, binary_cols], x[:, binary_cols], reduction='none')
                masked_bce = bce_loss * m[:, binary_cols]
                total_loss_sum += masked_bce.sum()
            total_observed = m.sum() + 1e-8
            recon_loss = total_loss_sum / total_observed
            
            # KL Divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            
            loss = recon_loss + beta * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 50 == 0 or (epoch+1) == epochs:
            print(f"    Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.6f}")
            
    # 4. Impute
    print(f"  Generating Imputation...")
    model.eval()
    with torch.no_grad():
        x_all = torch.FloatTensor(dataset.data_filled).to(device)
        m_all = torch.FloatTensor(dataset.mask).to(device)
        recon_x, _, _ = model(x_all, m_all)
        
        imputed_norm = (x_all * m_all + recon_x * (1 - m_all)).cpu().numpy()
        
        # Inverse Transform (Features Only)
        X_imputed = dataset.inverse_transform(imputed_norm)
        
        imputed_df = pd.DataFrame(X_imputed, columns=dataset.feature_columns)
        
        # Add back original targets and IDs
        for col in dataset.target_cols:
            imputed_df[col] = dataset.targets_orig[col].values
        
        if dataset.ids_orig is not None:
            for col in dataset.cols_to_drop:
                if col in dataset.ids_orig.columns:
                    imputed_df.insert(0, col, dataset.ids_orig[col].values)
                    
        imputed_df.to_csv(output_path, index=False)
        print(f"  Saved: {os.path.basename(output_path)}")

    print(f"\n{'='*60}")
    print(f"Standard VAE Complete for {dataset_name} {severity} Fold {fold_idx}!")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Standard VAE imputation for a specific fold.")
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4], help="CV fold index (0-4)")
    parser.add_argument("--dataset", type=str, required=True, choices=["metabric", "mimic"], help="Dataset name")
    parser.add_argument("--scenario", type=str, required=True, choices=["light", "moderate", "severe"], help="Imputation scenario")
    args = parser.parse_args()
    
    run_standard_vae_imputation(args.dataset, args.scenario, args.fold)
