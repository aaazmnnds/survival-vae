"""
MIDA Imputation Script (Portable Version)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os

class MIDA(nn.Module):
    def __init__(self, input_dim, latent_dim=None):
        super(MIDA, self).__init__()
        if latent_dim is None: latent_dim = input_dim + 5
        self.encoder = nn.Sequential(nn.Linear(input_dim * 2, latent_dim * 2), nn.Tanh(), nn.Linear(latent_dim * 2, latent_dim * 2), nn.Tanh(), nn.Linear(latent_dim * 2, latent_dim), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, latent_dim * 2), nn.Tanh(), nn.Linear(latent_dim * 2, latent_dim * 2), nn.Tanh(), nn.Linear(latent_dim * 2, input_dim), nn.Sigmoid())

    def forward(self, x, m):
        return self.decoder(self.encoder(torch.cat([x, m], dim=1)))

def run_mida_imputation():
    POSSIBLE_DATA_DIRS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')),
        '/Users/azmannads/Documents/Research collections/Research 2025/datasets',
        '/Users/azmannads/VAE/datasets',
    ]
    DATA_DIR = next((p for p in POSSIBLE_DATA_DIRS if os.path.exists(p)), POSSIBLE_DATA_DIRS[0])
    
    DATASETS, SCENARIOS = ['metabric', 'mimic'], ['light', 'moderate', 'severe']
    BATCH_SIZE, EPOCHS = 64, 100

    for dataset_name in DATASETS:
        for severity in SCENARIOS:
            mnar_path = os.path.join(DATA_DIR, f'{dataset_name}_mnar_{severity}.csv')
            if not os.path.exists(mnar_path): continue
                
            df = pd.read_csv(mnar_path)
            if 'Sex' in df.columns: df['Sex'] = df['Sex'].map({'F': 0, 'M': 1, 'Female': 0, 'Male': 1})
            df_numeric = df.select_dtypes(include=[np.number])
            target_cols = ['Survival_in_days', 'Status', 'time', 'event', 'duration', 'death_event', 'Time', 'Event']
            cols_to_drop = [c for c in target_cols if c in df_numeric.columns]
            df_targets, df_features = df_numeric[cols_to_drop].copy(), df_numeric.drop(columns=cols_to_drop)
            
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(df_features.values)
            mask, data_filled = 1. - np.isnan(data_scaled), np.nan_to_num(data_scaled, nan=0.)
            
            model = MIDA(input_dim=df_features.shape[1])
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss(reduction='none')
            
            dataloader = DataLoader(TensorDataset(torch.FloatTensor(data_filled), torch.FloatTensor(mask)), batch_size=BATCH_SIZE, shuffle=True)
            model.train()
            for epoch in range(EPOCHS):
                for x_batch, m_batch in dataloader:
                    optimizer.zero_grad()
                    loss = (loss_fn(model(x_batch, m_batch), x_batch) * m_batch).sum() / (m_batch.sum() + 1e-8)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            with torch.no_grad():
                X_tensor, M_tensor = torch.FloatTensor(data_filled), torch.FloatTensor(mask)
                X_imputed_norm = M_tensor * X_tensor + (1 - M_tensor) * model(X_tensor, M_tensor)
            
            imputed_df = pd.DataFrame(scaler.inverse_transform(X_imputed_norm.numpy()), columns=df_features.columns)
            for col in df_targets.columns: imputed_df[col] = df_targets[col].values
            imputed_df.to_csv(os.path.join(DATA_DIR, f'{dataset_name}_imputed_mida_{severity}.csv'), index=False)

if __name__ == "__main__":
    run_mida_imputation()
