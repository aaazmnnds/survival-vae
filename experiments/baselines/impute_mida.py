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
EPOCHS = 100
THETA = 7 # Scaling factor for overtraining (optional, simplified here to standard DAE)

# ============================================================================
# MIDA MODEL
# ============================================================================
class MIDA(nn.Module):
    def __init__(self, input_dim, latent_dim=None):
        super(MIDA, self).__init__()
        
        if latent_dim is None:
            latent_dim = input_dim + 5 # Heuristic: Input + Overhead
            
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

def run_mida_imputation():
    print(f"Starting MIDA Imputation (PyTorch)...")
    
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
            output_path = os.path.join(DATA_DIR, f'{dataset_name}_imputed_mida_{severity}.csv')
            
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
            
            # 2. Preprocessing (MIDA requires [0,1] normalization for sigmoid)
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Create Mask (1=Observed, 0=Missing)
            mask = 1. - np.isnan(data_scaled)
            
            # Fill NaNs with 0 for Input
            data_filled = np.nan_to_num(data_scaled, nan=0.)
            
            dim = data.shape[1]
            input_tensor = torch.FloatTensor(data_filled)
            mask_tensor = torch.FloatTensor(mask)
            
            dataset = TensorDataset(input_tensor, mask_tensor)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            # 3. Model & Optimizer
            model = MIDA(input_dim=dim)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss(reduction='none') # We need element-wise to apply mask
            
            # 4. Training Loop
            print(f"  Training MIDA ({EPOCHS} epochs)...")
            model.train()
            
            for epoch in range(EPOCHS):
                total_loss = 0
                for x_batch, m_batch in dataloader:
                    optimizer.zero_grad()
                    
                    # Reconstruction
                    recon_x = model(x_batch, m_batch)
                    
                    # Loss only on OBSERVED values
                    # L = sum( (x - recon)^2 * m ) / sum(m)
                    loss = (loss_fn(recon_x, x_batch) * m_batch).sum() / (m_batch.sum() + 1e-8)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch+1) % 20 == 0:
                    avg_loss = total_loss / len(dataloader)
                    print(f"    Epoch {epoch+1} | Loss: {avg_loss:.4f}")
            
            # 5. Final Imputation
            print(f"  Generating final imputation...")
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(data_filled)
                M_tensor = torch.FloatTensor(mask)
                
                recon_x = model(X_tensor, M_tensor)
                
                # Combine: Observed + Generated(Missing)
                # Note: MIDA outputs are normalized [0,1]
                X_imputed_norm = M_tensor * X_tensor + (1 - M_tensor) * recon_x
                
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
    print(f"MIDA Pipeline Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_mida_imputation()
