"""
DeepHit Training Script (Portable Version)
Optimized for Mac Mini M1 (MPS)
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import itertools
from pathlib import Path

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class DeepHit(nn.Module):
    def __init__(self, input_dim, num_durations, hidden_dims=[64, 32], dropout=0.2):
        super(DeepHit, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.shared = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, num_durations)
    
    def forward(self, x):
        features = self.shared(x)
        logits = self.output(features)
        return F.softmax(logits, dim=1)

def deephit_loss(pred_pmf, y_time, y_event, alpha=0.1):
    mask_event = (y_event == 1).float()
    idx = y_time.long().unsqueeze(1)
    max_idx = pred_pmf.shape[1] - 1
    idx = torch.clamp(idx, 0, max_idx)
    probs = torch.gather(pred_pmf, 1, idx).squeeze()
    nll = -torch.log(probs + 1e-8) * mask_event
    nll_loss = nll.sum() / (mask_event.sum() + 1e-8)
    return nll_loss

def get_c_index_discrete(pred_pmf, times, events):
    cif = torch.cumsum(pred_pmf, dim=1).cpu().detach().numpy()
    surv = 1.0 - cif
    risk_score = -np.sum(surv, axis=1)
    from sksurv.metrics import concordance_index_censored
    try:
        c_index = concordance_index_censored(events.astype(bool), times, -risk_score)[0]
    except:
        c_index = 0.5
    return c_index

def run_tuning_deephit(X_train, y_time_train, y_event_train, X_val, y_time_val, y_event_val, search_space, num_durations):
    best_config = None
    best_c_index = -1
    keys = search_space.keys()
    combinations = list(itertools.product(*search_space.values()))
    
    Xt = torch.FloatTensor(X_train).to(device)
    Yt_t = torch.FloatTensor(y_time_train).to(device)
    Yt_e = torch.FloatTensor(y_event_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    train_ds = TensorDataset(Xt, Yt_t, Yt_e)
    
    for i, values in enumerate(combinations):
        config = dict(zip(keys, values))
        loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
        model = DeepHit(X_train.shape[1], num_durations, config['hidden_dims'], config['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        
        best_run_val = -1
        patience = 5
        ctr = 0
        for epoch in range(20):
            model.train()
            for bx, bt, be in loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = deephit_loss(out, bt, be, config.get('alpha', 0.1))
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                out_val = model(Xv)
                val_ci = get_c_index_discrete(out_val, y_time_val, y_event_val)
            if val_ci > best_run_val:
                best_run_val = val_ci
                ctr = 0
            else:
                ctr += 1
            if ctr >= patience: break
        
        if best_run_val > best_c_index:
            best_c_index = best_run_val
            best_config = config
    return best_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--m_imp', type=int, default=None)
    parser.add_argument('--num_durations', type=int, default=50)
    args = parser.parse_args()

    POSSIBLE_DATA_DIRS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')),
        '/Users/azmannads/Documents/Research collections/Research 2025/datasets',
        '/Users/azmannads/VAE/datasets',
    ]
    DATA_DIR = next((p for p in POSSIBLE_DATA_DIRS if os.path.exists(p)), POSSIBLE_DATA_DIRS[0])

    if args.method == 'mice' and args.m_imp is not None:
        filename = f"{args.dataset}_imputed_{args.method}_{args.scenario}_{args.m_imp}.csv"
    else:
        filename = f"{args.dataset}_imputed_{args.method}_{args.scenario}.csv"
        
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath): return

    df = pd.read_csv(filepath)
    time_col = next((c for c in ['duration', 'survival_time', 'time'] if c in df.columns), None)
    event_col = next((c for c in ['event', 'death_event', 'Event', 'Status'] if c in df.columns), None)
    
    times = df[time_col].values
    events = df[event_col].values
    bins = np.linspace(times.min(), times.max(), args.num_durations)
    y_time_discrete = np.clip(np.digitize(times, bins) - 1, 0, args.num_durations-1)
    
    feature_cols = [c for c in df.columns if c not in [time_col, event_col]]
    X = df[feature_cols].values.astype(np.float32)
    
    X_train, X_test, yt_train, yt_test, ye_train, ye_test, raw_t_train, raw_t_test = train_test_split(
        X, y_time_discrete, events, times, test_size=0.2, random_state=42
    )
    X_train, X_val, yt_train, yt_val, ye_train, ye_val, _, _ = train_test_split(
        X_train, yt_train, ye_train, raw_t_train, test_size=0.1, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    search_space = {
        'hidden_dims': [[64, 32], [128, 64], [32, 32]],
        'dropout': [0.1, 0.3],
        'lr': [1e-3, 5e-4],
        'batch_size': [64],
        'alpha': [0.1, 0.5]
    }
    
    best_config = run_tuning_deephit(X_train, yt_train, ye_train, X_val, yt_val, ye_val, search_space, args.num_durations)
    
    model = DeepHit(X_train.shape[1], args.num_durations, best_config['hidden_dims']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_config['lr'])
    
    Xt = torch.FloatTensor(X_train).to(device)
    Yt_t = torch.FloatTensor(yt_train).to(device)
    Yt_e = torch.FloatTensor(ye_train).to(device)
    ds = TensorDataset(Xt, Yt_t, Yt_e)
    loader = DataLoader(ds, batch_size=best_config['batch_size'], shuffle=True)
    
    for epoch in range(100):
        model.train()
        for bx, bt, be in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = deephit_loss(out, bt, be, best_config.get('alpha', 0.1))
            loss.backward()
            optimizer.step()
            
    model_dir = os.path.join(DATA_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    fname = f"deephit_tuned_{args.dataset}_{args.scenario}_{args.method}"
    if args.m_imp: fname += f"_{args.m_imp}"
    torch.save(model.state_dict(), os.path.join(model_dir, fname + ".pt"))

if __name__ == "__main__":
    main()
