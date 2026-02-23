"""
DeepSurv Training Script (Portable Version)
Optimized for Mac Mini M1 (MPS)
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import itertools
from pathlib import Path

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class DeepSurv(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2):
        super(DeepSurv, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def cox_partial_likelihood_loss(log_h, events):
    times = events[:, 0]
    event_indicators = events[:, 1]
    sorted_idx = torch.argsort(times, descending=True)
    log_h_sorted = log_h[sorted_idx].squeeze()
    events_sorted = event_indicators[sorted_idx]
    risk_scores = torch.exp(log_h_sorted)
    risk_set_sum = torch.cumsum(risk_scores, dim=0)
    log_risk_set = torch.log(risk_set_sum + 1e-8)
    loss_per_event = (log_h_sorted - log_risk_set) * events_sorted
    num_events = torch.sum(events_sorted)
    if num_events == 0:
        return torch.tensor(0.0, device=log_h.device, requires_grad=True)
    return -torch.sum(loss_per_event) / num_events

def calculate_c_index(predictions, times, events):
    n = len(predictions)
    concordant = 0
    permissible = 0
    for i in range(n):
        if events[i] == 0: continue
        for j in range(n):
            if times[i] < times[j]:
                permissible += 1
                if predictions[i] > predictions[j]: concordant += 1
                elif predictions[i] == predictions[j]: concordant += 0.5
    return concordant / permissible if permissible > 0 else 0.5

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = cox_partial_likelihood_loss(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_preds, all_times, all_events = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            out = model(X_batch.to(device))
            all_preds.extend(out.cpu().numpy().flatten())
            all_times.extend(y_batch[:, 0].numpy())
            all_events.extend(y_batch[:, 1].numpy())
    return calculate_c_index(np.array(all_preds), np.array(all_times), np.array(all_events))

def run_tuning(X_train, y_train, X_val, y_val, search_space):
    best_config, best_c_index = None, -1
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    keys = search_space.keys()
    combinations = list(itertools.product(*search_space.values()))
    
    for values in combinations:
        config = dict(zip(keys, values))
        batch_size = config.get('batch_size', 64)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        model = DeepSurv(X_train.shape[1], config['hidden_dims'], config['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
        
        best_val_ci_run = -1
        patience, ctr = 5, 0
        for epoch in range(30):
            train_one_epoch(model, loader, optimizer)
            val_ci = evaluate(model, val_loader)
            if val_ci > best_val_ci_run:
                best_val_ci_run = val_ci
                ctr = 0
            else:
                ctr += 1
            if ctr >= patience: break
        if best_val_ci_run > best_c_index:
            best_c_index = best_val_ci_run
            best_config = config
    return best_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--m_imp', type=int, default=None)
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
    
    time_col = next((c for c in ['duration', 'survival_time', 'time'] if c in df.columns), 'duration')
    event_col = next((c for c in ['event', 'death_event'] if c in df.columns), 'event')
    
    X = df[[c for c in df.columns if c not in [time_col, event_col]]].values.astype(np.float32)
    y = np.stack([df[time_col].values, df[event_col].values], axis=1).astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y[:, 1])
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train[:, 1])
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    search_space = {'hidden_dims': [[32, 32], [64, 32]], 'dropout': [0.1, 0.3], 'lr': [0.001], 'batch_size': [64]}
    best_config = run_tuning(X_train, y_train, X_val, y_val, search_space)
    
    model = DeepSurv(X_train.shape[1], best_config['hidden_dims'], best_config['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_config['lr'])
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=best_config['batch_size'], shuffle=True)
    
    best_state, best_ci, ctr = None, -1, 0
    for epoch in range(100):
        train_one_epoch(model, loader, optimizer)
        val_ci = evaluate(model, DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=64))
        if val_ci > best_ci:
            best_ci, best_state, ctr = val_ci, model.state_dict(), 0
        elif ctr >= 10: break
        else: ctr += 1
                
    model_dir = os.path.join(DATA_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    save_name = f"deepsurv_tuned_{args.dataset}_{args.scenario}_{args.method}" + (f"_{args.m_imp}" if args.m_imp else "") + ".pt"
    torch.save({'state_dict': best_state, 'config': best_config, 'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_}, os.path.join(model_dir, save_name))

if __name__ == "__main__":
    main()
