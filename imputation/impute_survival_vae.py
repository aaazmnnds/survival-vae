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

import sys
import os
# Add repository root to path for cross-folder discovery
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.survival_vae import ClinicalDataset, SurvivalVAE, cox_ph_loss

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
def train_model(dataset_path, train_idx, mask_path=None, epochs=50, latent_dim=10, beta=1.0, gamma=1.0, lr=1e-3):
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
        '/Users/azmannads/VAE/Survival-VAE_study',
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
    model = train_model(mnar_path, train_idx, mask_path, epochs=epochs, latent_dim=latent_dim, beta=beta, gamma=gamma, lr=lr)
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
