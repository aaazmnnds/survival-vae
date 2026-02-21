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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'datasets')
DATASETS = ['metabric', 'mimic']
SCENARIOS = ['light', 'moderate', 'severe']

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 200  # GAIN needs more epochs
HINT_RATE = 0.9
ALPHA = 100 # Hyperparameter for Reconstruction Loss

# ============================================================================
# GAIN MODELS
# ============================================================================
class NetG(nn.Module):
    def __init__(self, dim):
        super(NetG, self).__init__()
        self.fc1 = nn.Linear(dim * 2, 64) # Input: Data + Mask
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, dim) # Output: Imputed Vector
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        # Concatenate Data and Mask
        inputs = torch.cat([x, m], dim=1)
        out = self.relu(self.fc1(inputs))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out)) # Output normalized [0,1]
        return out

class NetD(nn.Module):
    def __init__(self, dim):
        super(NetD, self).__init__()
        self.fc1 = nn.Linear(dim * 2, 64) # Input: Components + Hint
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, dim) # Output: Probability Mask
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        # Concatenate Data and Hint
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

def run_gain_imputation():
    print(f"Starting GAIN Imputation (PyTorch)...")
    
    # Set random seeds for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    for dataset_name in DATASETS:
        print(f"\n{'#'*60}")
        print(f"PROCESSING DATASET: {dataset_name.upper()}")
        print(f"{'#'*60}")
        
        for severity in SCENARIOS:
            mnar_path = os.path.join(DATA_DIR, f'{dataset_name}_mnar_{severity}.csv')
            output_path = os.path.join(DATA_DIR, f'{dataset_name}_imputed_gain_{severity}.csv')
            
            if not os.path.exists(mnar_path):
                print(f"Skipping {severity.upper()} - File not found: {mnar_path}")
                continue
                
            print(f"\nScenario: {severity.upper()}")
            print(f"Input: {mnar_path}")
            
            # 1. Load Data
            df = pd.read_csv(mnar_path)
            
            # Encode 'Sex' column if present
            if 'Sex' in df.columns:
                df['Sex'] = df['Sex'].map({'F': 0, 'M': 1, 'Female': 0, 'Male': 1})
            
            # Drop non-numeric columns
            df_numeric = df.select_dtypes(include=[np.number])
            
            # PREVENT LEAKAGE: Drop Survival Targets
            target_cols = ['Survival_in_days', 'Status', 'time', 'event', 'duration', 'death_event', 'Time', 'Event']
            cols_to_drop = [c for c in target_cols if c in df_numeric.columns]
            
            df_targets = df_numeric[cols_to_drop].copy()
            df_features = df_numeric.drop(columns=cols_to_drop)
            
            print(f"  Dropping targets to prevent leakage: {cols_to_drop}")
            
            data = df_features.values
            
            # 2. Preprocessing (GAIN requires [0,1] normalization)
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Create Mask (1=Observed, 0=Missing)
            mask = 1. - np.isnan(data_scaled)
            
            # Fill NaNs with 0 for Input
            data_filled = np.nan_to_num(data_scaled, nan=0.)
            
            dim = data.shape[1]
            
            # 3. Models & Optimizers
            netG = NetG(dim)
            netD = NetD(dim)
            
            optG = optim.Adam(netG.parameters(), lr=0.001)
            optD = optim.Adam(netD.parameters(), lr=0.001)
            
            # 4. Training Loop
            print(f"  Training GAIN ({EPOCHS} epochs)...")
            
            for epoch in range(EPOCHS):
                # Batching (Simplified: Full batch for small data, or random blocks)
                # Using random indices for stochasticity
                idx = np.random.permutation(len(data))
                
                # Mini-batch loop
                for i in range(0, len(data), BATCH_SIZE):
                    end = min(i + BATCH_SIZE, len(data))
                    batch_idx = idx[i:end]
                    
                    X_mb = torch.FloatTensor(data_filled[batch_idx])
                    M_mb = torch.FloatTensor(mask[batch_idx])
                    
                    # Hint Matrix
                    H_mb_binary = sample_binary(len(batch_idx), dim, HINT_RATE)
                    H_mb = M_mb * torch.FloatTensor(H_mb_binary) + 0.5 * (1 - torch.FloatTensor(H_mb_binary))
                    
                    # Noise for Generator
                    Z_mb = torch.FloatTensor(np.random.uniform(0., 0.1, size=[len(batch_idx), dim]))
                    
                    # --- Train Discriminator ---
                    optD.zero_grad()
                    G_sample = netG(X_mb + M_mb * Z_mb, M_mb) # Noise only on observed? No, typically Z + X
                    # Standard GAIN combine:
                    X_hat = M_mb * X_mb + (1 - M_mb) * G_sample
                    
                    D_prob = netD(X_hat, H_mb)
                    
                    # D Loss: Maximize log(D(real)) + log(1-D(fake))
                    # Here we want D to predict M correctly
                    # L_D = -mean( M * log(D) + (1-M) * log(1-D) )
                    D_loss = -torch.mean(M_mb * torch.log(D_prob + 1e-8) + (1-M_mb) * torch.log(1. - D_prob + 1e-8))
                    
                    D_loss.backward()
                    optD.step()
                    
                    # --- Train Generator ---
                    optG.zero_grad()
                    G_sample = netG(X_mb + M_mb * Z_mb, M_mb)
                    X_hat = M_mb * X_mb + (1 - M_mb) * G_sample
                    D_prob = netD(X_hat, H_mb)
                    
                    # G Loss 1: Fool Discriminator (Maximize log(D_prob) where M=0)
                    G_loss_temp = -torch.mean((1-M_mb) * torch.log(D_prob + 1e-8))
                    
                    # G Loss 2: Reconstruction MSE (on Observed Data)
                    MSE_loss = torch.mean((M_mb * X_mb - M_mb * G_sample)**2) / torch.mean(M_mb)
                    
                    G_loss = G_loss_temp + ALPHA * MSE_loss
                    
                    G_loss.backward()
                    optG.step()
                    
                if (epoch+1) % 50 == 0:
                    print(f"    Epoch {epoch+1} | D_loss: {D_loss.item():.4f} | G_loss: {G_loss.item():.4f}")
            
            # 5. Final Imputation
            print(f"  Generating final imputation...")
            X_tensor = torch.FloatTensor(data_filled)
            M_tensor = torch.FloatTensor(mask)
            
            with torch.no_grad():
                G_sample = netG(X_tensor, M_tensor) # Use noise=0 or mean noise for inference
                
                # Combine: Observed + Generated(Missing)
                # Note: GAIN outputs are normalized [0,1]
                X_imputed_norm = M_tensor * X_tensor + (1 - M_tensor) * G_sample
                
            # Inverse Transform to Original Scale
            X_imputed = scaler.inverse_transform(X_imputed_norm.numpy())
            
            # Create DataFrame
            imputed_df = pd.DataFrame(X_imputed, columns=df_features.columns)
            
            # Add back targets (unmodified)
            for col in df_targets.columns:
                imputed_df[col] = df_targets[col].values
            
            # Save
            imputed_df.to_csv(output_path, index=False)
            print(f"  Saved: {os.path.basename(output_path)}")

    print(f"\n{'='*60}")
    print(f"GAIN Pipeline Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_gain_imputation()
