"""
GAIN Imputation Script (Portable Version)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os

class NetG(nn.Module):
    def __init__(self, dim):
        super(NetG, self).__init__()
        self.fc1 = nn.Linear(dim * 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        inputs = torch.cat([x, m], dim=1)
        out = self.relu(self.fc1(inputs))
        out = self.relu(self.fc2(out))
        return self.sigmoid(self.fc3(out))

class NetD(nn.Module):
    def __init__(self, dim):
        super(NetD, self).__init__()
        self.fc1 = nn.Linear(dim * 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        inputs = torch.cat([x, h], dim=1)
        out = self.relu(self.fc1(inputs))
        out = self.relu(self.fc2(out))
        return self.sigmoid(self.fc3(out))

def sample_binary(m, n, p=0.5):
    return (np.random.uniform(0., 1., size=[m, n]) > p).astype(float)

def run_gain_imputation():
    POSSIBLE_DATA_DIRS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')),
        '/Users/azmannads/Documents/Research collections/Research 2025/datasets',
        '/Users/azmannads/VAE/datasets',
    ]
    DATA_DIR = next((p for p in POSSIBLE_DATA_DIRS if os.path.exists(p)), POSSIBLE_DATA_DIRS[0])
    
    DATASETS = ['metabric', 'mimic']
    SCENARIOS = ['light', 'moderate', 'severe']
    BATCH_SIZE, EPOCHS, HINT_RATE, ALPHA = 64, 200, 0.9, 100

    for dataset_name in DATASETS:
        for severity in SCENARIOS:
            mnar_path = os.path.join(DATA_DIR, f'{dataset_name}_mnar_{severity}.csv')
            output_path = os.path.join(DATA_DIR, f'{dataset_name}_imputed_gain_{severity}.csv')
            if not os.path.exists(mnar_path): continue
                
            df = pd.read_csv(mnar_path)
            if 'Sex' in df.columns:
                df['Sex'] = df['Sex'].map({'F': 0, 'M': 1, 'Female': 0, 'Male': 1})
            
            df_numeric = df.select_dtypes(include=[np.number])
            target_cols = ['Survival_in_days', 'Status', 'time', 'event', 'duration', 'death_event', 'Time', 'Event']
            cols_to_drop = [c for c in target_cols if c in df_numeric.columns]
            df_targets = df_numeric[cols_to_drop].copy()
            df_features = df_numeric.drop(columns=cols_to_drop)
            
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(df_features.values)
            mask = 1. - np.isnan(data_scaled)
            data_filled = np.nan_to_num(data_scaled, nan=0.)
            dim = df_features.shape[1]
            
            netG, netD = NetG(dim), NetD(dim)
            optG, optD = optim.Adam(netG.parameters(), lr=0.001), optim.Adam(netD.parameters(), lr=0.001)
            
            for epoch in range(EPOCHS):
                idx = np.random.permutation(len(data_filled))
                for i in range(0, len(data_filled), BATCH_SIZE):
                    end = min(i + BATCH_SIZE, len(data_filled))
                    batch_idx = idx[i:end]
                    X_mb, M_mb = torch.FloatTensor(data_filled[batch_idx]), torch.FloatTensor(mask[batch_idx])
                    H_mb_binary = sample_binary(len(batch_idx), dim, HINT_RATE)
                    H_mb = M_mb * torch.FloatTensor(H_mb_binary) + 0.5 * (1 - torch.FloatTensor(H_mb_binary))
                    Z_mb = torch.FloatTensor(np.random.uniform(0., 0.1, size=[len(batch_idx), dim]))
                    
                    optD.zero_grad()
                    G_sample = netG(X_mb + M_mb * Z_mb, M_mb)
                    X_hat = M_mb * X_mb + (1 - M_mb) * G_sample
                    D_prob = netD(X_hat, H_mb)
                    D_loss = -torch.mean(M_mb * torch.log(D_prob + 1e-8) + (1-M_mb) * torch.log(1. - D_prob + 1e-8))
                    D_loss.backward()
                    optD.step()
                    
                    optG.zero_grad()
                    G_sample = netG(X_mb + M_mb * Z_mb, M_mb)
                    X_hat = M_mb * X_mb + (1 - M_mb) * G_sample
                    D_prob = netD(X_hat, H_mb)
                    G_loss = -torch.mean((1-M_mb) * torch.log(D_prob + 1e-8)) + ALPHA * (torch.mean((M_mb * X_mb - M_mb * G_sample)**2) / torch.mean(M_mb))
                    G_loss.backward()
                    optG.step()
            
            with torch.no_grad():
                X_imputed_norm = torch.FloatTensor(mask) * torch.FloatTensor(data_filled) + (1 - torch.FloatTensor(mask)) * netG(torch.FloatTensor(data_filled), torch.FloatTensor(mask))
            
            imputed_df = pd.DataFrame(scaler.inverse_transform(X_imputed_norm.numpy()), columns=df_features.columns)
            for col in df_targets.columns: imputed_df[col] = df_targets[col].values
            imputed_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    run_gain_imputation()
