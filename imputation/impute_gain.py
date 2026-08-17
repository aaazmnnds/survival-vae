"""
GAIN Imputation Script (Generative Adversarial Imputation Nets)
==============================================================
Implements GAIN (Yoon et al., 2018) using PyTorch.
Architecture:
- Generator: Observes X_obs, Mask, Noise -> Imputes X_miss
- Discriminator: Distinguishes Real Observed components from Imputed components
- Hint Mechanism: Provides partial information about Mask to Discriminator

Strategy: Single Imputation (Deep Learning Standard)
Output: 6 total .csv files (2 Datasets * 3 Scenarios)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
import os
import argparse
import json

# Configuration
DATA_DIR = os.environ.get('SVAE_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets'))
# RESULTS_DIR will be set dynamically per fold

# Training Hyperparameters
BATCH_SIZE = 512 # Increased for speed
DEFAULT_EPOCHS = 200

# ============================================================================
# GAIN MODELS
# ============================================================================
class NetG(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super(NetG, self).__init__()
        self.fc1 = nn.Linear(dim * 2, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim) 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        inputs = torch.cat([x, m], dim=1)
        out = self.relu(self.fc1(inputs))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out)) 
        return out

class NetD(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super(NetD, self).__init__()
        self.fc1 = nn.Linear(dim * 2, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim) 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        inputs = torch.cat([x, h], dim=1)
        out = self.relu(self.fc1(inputs))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        return out

# ============================================================================
# UTILS
# ============================================================================
def sample_binary(m, n, p=0.5):
    A = np.random.uniform(0., 1., size=[m, n])
    B = A > p
    C = 1. * B
    return C

def run_gain_imputation(dataset_name, severity, fold_idx):
    print(f"Starting GAIN Imputation (PyTorch) - Fold {fold_idx}...")
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)
    
    print(f"\n{'#'*60}")
    print(f"PROCESSING: {dataset_name.upper()} | {severity.upper()} | FOLD {fold_idx}")
    print(f"{'#'*60}")
    
    mnar_path = os.path.join(DATA_DIR, f'{dataset_name}_mnar_{severity}.csv')
    
    # Update output path structure to include fold index
    output_dir = os.path.join(DATA_DIR, 'final', 'imputation_results_final', f'fold_{fold_idx}')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{dataset_name}_{severity}_gain_imputed.csv')
    
    if not os.path.exists(mnar_path):
        print(f"Error: File not found: {mnar_path}")
        return
        
    print(f"Input: {mnar_path}")
    
    # 0. Load Split Indices
    splits_path = os.path.join(DATA_DIR, f'cv_splits_{dataset_name}.json')
    if not os.path.exists(splits_path):
        # Try local directory if not in DATA_DIR
        splits_path = os.path.join(os.path.dirname(mnar_path), f'cv_splits_{dataset_name}.json')
        
    with open(splits_path, 'r') as f:
        cv_splits = json.load(f)
    fold_key = f'fold_{fold_idx + 1}'
    train_idx = np.array(cv_splits[fold_key]['train'])
    val_idx   = np.array(cv_splits[fold_key]['val'])
    
    # 1. Load Data
    df = pd.read_csv(mnar_path)
            
    # 2. Load Optimized Hyperparameters
    optuna_dir = os.path.join(DATA_DIR, 'final', 'optuna_results_final')
    
    # Optuna Path Logic: Current Fold -> Fold 0 -> Root Fallback
    optuna_file = os.path.join(optuna_dir, f'fold_{fold_idx}', f'{dataset_name}_{severity}_optuna_gain_results.csv')
    if not os.path.exists(optuna_file):
        # Fallback to fold_0 (Methodology: Optimize on fold 0 only)
        optuna_file = os.path.join(optuna_dir, 'fold_0', f'{dataset_name}_{severity}_optuna_gain_results.csv')
        
    if not os.path.exists(optuna_file):
        # Fallback to root for backward compatibility
        optuna_file = os.path.join(optuna_dir, f'{dataset_name}_{severity}_optuna_gain_results.csv')
    if os.path.exists(optuna_file):
        df_optuna = pd.read_csv(optuna_file)
        best_params = df_optuna.iloc[0]
        lr = float(best_params['learning_rate'])
        alpha = float(best_params['alpha'])
        hint_rate = float(best_params['hint_rate'])
        hidden_dim = int(best_params['hidden_dim'])
        epochs = DEFAULT_EPOCHS
        print(f"  Loaded Optimized: lr={lr:.6f}, alpha={alpha:.2f}, hint={hint_rate:.2f}, hidden={hidden_dim}, epochs={epochs}")
    else:
        lr = 0.001
        alpha = 100
        hint_rate = 0.9
        hidden_dim = 64
        epochs = DEFAULT_EPOCHS
        print(f"  Warning: Optuna file not found. Using default parameters.")
    
    # Encode 'Sex' if present
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'F': 0, 'M': 1, 'Female': 0, 'Male': 1})
    
    # Identify target and ID columns
    if 'Survival_in_days' in df.columns:
        target_cols = ['Survival_in_days', 'Status']
    elif 'duration' in df.columns:
        target_cols = ['duration', 'event']
    else:
        target_cols = ['Time', 'Event']

    id_cols = ['hadm_id', 'subject_id', 'stay_id', 'icustay_id', 'patient_id', 'admittime', 'dischtime']
    cols_to_drop = [col for col in id_cols if col in df.columns]
    
    # 3. Preprocessing (Scale Features AND Targets) (NaN-safe)
    feature_df_mnar = df.drop(columns=cols_to_drop + target_cols).select_dtypes(include=[np.number])
    feature_cols = feature_df_mnar.columns
    
    X_raw = feature_df_mnar.values.astype(np.float32)
    x_min = np.nanmin(X_raw[train_idx], axis=0)
    x_max = np.nanmax(X_raw[train_idx], axis=0)
    x_scale = x_max - x_min
    x_scale[x_scale < 1e-6] = 1.0
    X_scaled = (X_raw - x_min) / x_scale
    
    # Combine for Imputation
    X_all_scaled = X_scaled
    
    # Mask and data tensors
    mask = 1. - np.isnan(X_all_scaled)
    data_filled = np.nan_to_num(X_all_scaled, nan=0.)
    
    dim = X_all_scaled.shape[1]
    
    # 4. Models & Optimizers
    netG = NetG(dim, hidden_dim=hidden_dim).to(device)
    netD = NetD(dim, hidden_dim=hidden_dim).to(device)
    
    optG = optim.Adam(netG.parameters(), lr=lr)
    optD = optim.Adam(netD.parameters(), lr=lr)
    
    dataset_full = torch.FloatTensor(data_filled).to(device)
    mask_full = torch.FloatTensor(mask).to(device)
    
    dataset = TensorDataset(dataset_full, mask_full)
    sampler = SubsetRandomSampler(train_idx)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
    
    KNOWN_BINARY_FEATURES = [
        'Hormone_Tx', 'Radiotherapy', 'Chemotherapy', 'ER_Positive',
        'Sex', 'CCI_MI', 'CCI_CHF', 'CCI_PVD', 'CCI_Stroke',
        'CCI_Renal', 'CCI_Liver', 'CCI_Cancer'
    ]
    binary_feature_indices = [
        i for i, col in enumerate(feature_cols)
        if col in KNOWN_BINARY_FEATURES
    ]

    # 5. Training Loop
    print(f"  Training GAIN ({epochs} epochs)... ")
    
    for epoch in range(epochs):
        for X_mb, M_mb in dataloader:
            
            # Hint Mechanism
            H_mb_binary = sample_binary(X_mb.shape[0], dim, hint_rate)
            H_mb = M_mb * torch.FloatTensor(H_mb_binary).to(device) + 0.5 * (1 - torch.FloatTensor(H_mb_binary).to(device))
            
            # Random Noise
            Z_mb = torch.FloatTensor(np.random.uniform(0., 0.1, size=[X_mb.shape[0], dim])).to(device)
            
            # --- Train Discriminator ---
            optD.zero_grad()
            G_sample = netG(X_mb + (1 - M_mb) * Z_mb, M_mb)
            X_hat = M_mb * X_mb + (1 - M_mb) * G_sample
            D_prob = netD(X_hat, H_mb)
            D_loss = -torch.mean(M_mb * torch.log(D_prob + 1e-8) + (1-M_mb) * torch.log(1. - D_prob + 1e-8))
            D_loss.backward()
            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
            optD.step()
            
            # --- Train Generator ---
            optG.zero_grad()
            G_sample = netG(X_mb + (1 - M_mb) * Z_mb, M_mb)
            X_hat = M_mb * X_mb + (1 - M_mb) * G_sample
            D_prob = netD(X_hat, H_mb)
            G_loss_temp = -torch.mean((1-M_mb) * torch.log(D_prob + 1e-8))
            num_features = X_mb.shape[1]
            binary_cols = binary_feature_indices if binary_feature_indices is not None else []
            cont_cols = [i for i in range(num_features) if i not in binary_cols]
            total_loss_sum = 0.0
            if len(cont_cols) > 0:
                mse_loss = F.mse_loss(G_sample[:, cont_cols], X_mb[:, cont_cols], reduction='none')
                masked_mse = mse_loss * M_mb[:, cont_cols]
                total_loss_sum += masked_mse.sum()
            if len(binary_cols) > 0:
                bce_loss = F.binary_cross_entropy(G_sample[:, binary_cols], X_mb[:, binary_cols], reduction='none')
                masked_bce = bce_loss * M_mb[:, binary_cols]
                total_loss_sum += masked_bce.sum()
            total_observed = M_mb.sum() + 1e-8
            MSE_loss = total_loss_sum / total_observed
            G_loss = G_loss_temp + alpha * MSE_loss
            G_loss.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            optG.step()
            
        if (epoch+1) % 50 == 0 or (epoch+1) == epochs:
            print(f"    Epoch {epoch+1} | D_loss: {D_loss.item():.4f} | G_loss: {G_loss.item():.4f}")
    
    # 6. Final Imputation
    print(f"  Generating final imputation...")
    netG.eval()
    with torch.no_grad():
        G_sample = netG(dataset_full, mask_full)
        X_imputed_norm = (mask_full * dataset_full + (1 - mask_full) * G_sample).cpu().numpy()
        
    # Inverse Transform (Features Only)
    X_imputed = X_imputed_norm * x_scale + x_min
    
    # Create DataFrame
    imputed_df = pd.DataFrame(X_imputed, columns=feature_cols)
    
    # Add back original targets and IDs
    for col in target_cols:
        imputed_df[col] = df[col].values
    
    if id_cols:
        for col in id_cols:
            if col in df.columns:
                imputed_df.insert(0, col, df[col].values)
    
    # Save
    imputed_df.to_csv(output_path, index=False)
    print(f"  Saved: {os.path.basename(output_path)}")

    print(f"\n{'='*60}")
    print(f"GAIN Pipeline Complete for {dataset_name} {severity} Fold {fold_idx}!")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GAIN imputation for a specific fold.")
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4], help="CV fold index (0-4)")
    parser.add_argument("--dataset", type=str, required=True, choices=["metabric", "mimic"], help="Dataset name")
    parser.add_argument("--scenario", type=str, required=True, choices=["light", "moderate", "severe"], help="Imputation scenario")
    args = parser.parse_args()
    
    run_gain_imputation(args.dataset, args.scenario, args.fold)
