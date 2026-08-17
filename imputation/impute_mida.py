"""
MIDA Imputation Script (Multiple Imputation Denoising Autoencoder)
==================================================================
Implements MIDA (Gondara & Wang, 2018) algorithm using PyTorch.
Architecture:
- Denoising Autoencoder: Encoder -> Latent -> Decoder
- Input: Data (NaNs=0) + Missing Indicator Mask
- Loss: RMSE on Observed values (Reconstruction Error)

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

# Default Hyperparameters
BATCH_SIZE = 512 # Increased for GPU speed
DEFAULT_EPOCHS = 100

# ============================================================================
# MIDA MODEL
# ============================================================================
class MIDA(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(MIDA, self).__init__()
        
        # Encoder: Input = Data + Mask (Dimension * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, latent_dim * 2),
            nn.Tanh(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.Tanh(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()
        )
        
        # Decoder: Latent -> Data
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.Tanh(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.Tanh(),
            nn.Linear(latent_dim * 2, input_dim),
            nn.Sigmoid() # Output normalized [0,1]
        )

    def forward(self, x, m):
        inputs = torch.cat([x, m], dim=1)
        z = self.encoder(inputs)
        out = self.decoder(z)
        return out

def run_mida_imputation(dataset_name, severity, fold_idx):
    print(f"Starting MIDA Imputation (PyTorch) - Fold {fold_idx}...")
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
    output_path = os.path.join(output_dir, f'{dataset_name}_{severity}_mida_imputed.csv')
    
    if not os.path.exists(mnar_path):
        print(f"Error: File not found: {mnar_path}")
        return
        
    print(f"Input: {mnar_path}")
    
    # 0. Load Split Indices
    splits_path = os.path.join(DATA_DIR, f'cv_splits_{dataset_name}.json')
    if not os.path.exists(splits_path):
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
    optuna_file = os.path.join(optuna_dir, f'fold_{fold_idx}', f'{dataset_name}_{severity}_optuna_mida_results.csv')
    if not os.path.exists(optuna_file):
        # Fallback to fold_0 (Methodology: Optimize on fold 0 only)
        optuna_file = os.path.join(optuna_dir, 'fold_0', f'{dataset_name}_{severity}_optuna_mida_results.csv')
        
    if not os.path.exists(optuna_file):
        # Fallback to root for backward compatibility
        optuna_file = os.path.join(optuna_dir, f'{dataset_name}_{severity}_optuna_mida_results.csv')
    if os.path.exists(optuna_file):
        df_optuna = pd.read_csv(optuna_file)
        best_params = df_optuna.iloc[0]
        lr = float(best_params['learning_rate'])
        latent_dim_offset = int(best_params['latent_dim_offset'])
        epochs = int(best_params['epochs'])
        print(f"  Loaded Optimized: lr={lr}, offset={latent_dim_offset}, epochs={epochs}")
    else:
        lr = 0.001
        latent_dim_offset = 5
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
    
    # 3. Preprocessing (Scale Features AND Targets)
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
    
    # Create Mask (1=Observed, 0=Missing)
    mask = 1. - np.isnan(X_all_scaled)
    data_filled = np.nan_to_num(X_all_scaled, nan=0.)
    
    dim = X_all_scaled.shape[1]
    latent_dim = dim + latent_dim_offset
    
    input_tensor = torch.FloatTensor(data_filled).to(device)
    mask_tensor = torch.FloatTensor(mask).to(device)
    
    # Training — train fold only using sampler
    dataset = TensorDataset(input_tensor, mask_tensor)
    sampler = SubsetRandomSampler(train_idx)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
    
    # 4. Model & Optimizer
    model = MIDA(input_dim=dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='none')
    
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
    print(f"  Training MIDA ({epochs} epochs)... ")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for x, m in dataloader:
            optimizer.zero_grad()
            recon_x = model(x, m)
            
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
            loss = total_loss_sum / total_observed
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 50 == 0 or (epoch+1) == epochs:
            avg_loss = total_loss / len(dataloader)
            print(f"    Epoch {epoch+1} | Loss: {avg_loss:.6f}")
    
    # 6. Final Imputation
    print(f"  Generating final imputation...")
    model.eval()
    with torch.no_grad():
        recon_x = model(input_tensor, mask_tensor)
        X_imputed_norm = (mask_tensor * input_tensor + (1 - mask_tensor) * recon_x).cpu().numpy()
        
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
    print(f"MIDA Pipeline Complete for {dataset_name} {severity} Fold {fold_idx}!")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MIDA imputation for a specific fold.")
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4], help="CV fold index (0-4)")
    parser.add_argument("--dataset", type=str, required=True, choices=["metabric", "mimic"], help="Dataset name")
    parser.add_argument("--scenario", type=str, required=True, choices=["light", "moderate", "severe"], help="Imputation scenario")
    args = parser.parse_args()
    
    run_mida_imputation(args.dataset, args.scenario, args.fold)
