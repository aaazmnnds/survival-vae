"""
DeepSurv Training Script with Hyperparameter Tuning
Optimized for Mac Mini M1 (MPS)

Usage:
    python train_deepsurv_tuned.py --dataset metabric --scenario light --method mice --m_imp 1
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
import time
import json

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS backend")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

class DeepSurv(nn.Module):
    """Deep Cox Proportional Hazards Network"""
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
    """Cox partial likelihood loss (Breslow approximation for ties)"""
    # Sort by risk scores (already done via dataloader in training usually, but explicit here)
    # Actually, standard implementation requires sorting by time. 
    # Here we assume the batch isn't sorted, so we do it locally or just approximation.
    # For correctness in small batches, let's just use the full batch formulation without sorting requirement if possible,
    # OR explicit sort. Explicit sort is safer.
    
    # Needs: predictions (log_h), event indicators (events[:, 1]), observed times (events[:, 0])
    times = events[:, 0]
    event_indicators = events[:, 1]
    
    # Sort decreasing by time
    sorted_idx = torch.argsort(times, descending=True)
    log_h_sorted = log_h[sorted_idx].squeeze()
    events_sorted = event_indicators[sorted_idx]
    
    # Risk = exp(log_hazard)
    risk_scores = torch.exp(log_h_sorted)
    
    # Cumulative risk (risk set)
    risk_set_sum = torch.cumsum(risk_scores, dim=0)
    
    # Log partial likelihood
    # PL = sum_{i: E_i=1} (log_h_i - log(sum_{j in R_i} exp(log_h_j)))
    log_risk_set = torch.log(risk_set_sum + 1e-8)
    
    # Difference
    diff = log_h_sorted - log_risk_set
    
    # Mask for events only
    loss_per_event = diff * events_sorted
    
    # Mean negative log likelihood
    num_events = torch.sum(events_sorted)
    if num_events == 0:
        return torch.tensor(0.0, device=log_h.device, requires_grad=True)
        
    return -torch.sum(loss_per_event) / num_events

def calculate_c_index(predictions, times, events):
    """Concordance Index (C-Index)"""
    n = len(predictions)
    concordant = 0
    permissible = 0
    
    # O(N^2) naive implementation - okay for validation sets of reasonable size
    for i in range(n):
        if events[i] == 0: continue
        for j in range(n):
            if times[i] < times[j]: # Time i < Time j (i died first)
                permissible += 1
                if predictions[i] > predictions[j]: # Higher hazard for i
                    concordant += 1
                elif predictions[i] == predictions[j]:
                    concordant += 0.5
                    
    return concordant / permissible if permissible > 0 else 0.5

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = cox_partial_likelihood_loss(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            all_preds.extend(out.cpu().numpy().flatten())
            all_times.extend(y_batch[:, 0].numpy())
            all_events.extend(y_batch[:, 1].numpy())
            
    c_index = calculate_c_index(np.array(all_preds), np.array(all_times), np.array(all_events))
    return c_index

def run_tuning(X_train, y_train, X_val, y_val, search_space):
    """Run grid search for hyperparameters"""
    best_config = None
    best_c_index = -1
    best_model_state = None
    
    # Create datasets efficiently
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    keys = search_space.keys()
    combinations = list(itertools.product(*search_space.values()))
    
    print(f"Starting Hyperparameter Tuning ({len(combinations)} combinations)...")
    
    for i, values in enumerate(combinations):
        config = dict(zip(keys, values))
        
        # Batch size fix
        batch_size = config.get('batch_size', 64)
        lr = config.get('lr', 0.001)
        hidden = config['hidden_dims']
        dropout = config['dropout']
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        model = DeepSurv(X_train.shape[1], hidden, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Train (Simple loop for tuning)
        max_epochs = 30 # Reduced epochs for tuning speed
        patience = 5
        best_val_ci_run = -1
        patience_counter = 0
        
        for epoch in range(max_epochs):
            train_one_epoch(model, train_loader, optimizer)
            val_ci = evaluate(model, val_loader)
            
            if val_ci > best_val_ci_run:
                best_val_ci_run = val_ci
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        print(f"[{i+1}/{len(combinations)}] Config: {config} -> Val C-Index: {best_val_ci_run:.4f}")
        
        if best_val_ci_run > best_c_index:
            best_c_index = best_val_ci_run
            best_config = config
            best_model_state = model.state_dict()
            
    print(f"\nBest Config Found: {best_config} (C-Index: {best_c_index:.4f})")
    return best_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--m_imp', type=int, default=None, help="MICE imputation index (1-5)")
    args = parser.parse_args()
    
    # --- File Path Logic ---
    base_dir = "/Users/azmannads/Downloads/Research 2025/datasets"
    if args.method == 'mice' and args.m_imp is not None:
        filename = f"{args.dataset}_imputed_{args.method}_{args.scenario}_{args.m_imp}.csv"
    else:
        filename = f"{args.dataset}_imputed_{args.method}_{args.scenario}.csv"
        
    filepath = os.path.join(base_dir, filename)
    
    if not os.path.exists(filepath):
        # Try alternate naming pattern (sometimes method comes after dataset)
        filename_alt = f"{args.dataset}_imputed_{args.method}_{args.scenario}.csv"
        filepath_alt = os.path.join(base_dir, filename_alt)
        if os.path.exists(filepath_alt):
            filepath = filepath_alt
            print(f"Found file at alternate path: {filepath}")
        else:
            print(f"Error: File not found at {filepath}")
            return

    print(f"Loading data: {filepath}")
    df = pd.read_csv(filepath)
    
    # --- Column Handling ---
    # Based on file inspection, both datasets use 'duration' and 'event'
    if 'duration' in df.columns:
        time_col = 'duration'
    elif 'survival_time' in df.columns: 
        time_col = 'survival_time'
    elif 'time' in df.columns:
        time_col = 'time'
    else:
        raise ValueError(f"Could not find time column. Available: {df.columns.tolist()}")

    if 'event' in df.columns:
        event_col = 'event'
    elif 'death_event' in df.columns:
        event_col = 'death_event'
    else:
        raise ValueError(f"Could not find event column. Available: {df.columns.tolist()}")
        
    print(f"Columns: Time='{time_col}', Event='{event_col}'")
    
    # Preprocessing
    feature_cols = [c for c in df.columns if c not in [time_col, event_col]]
    X = df[feature_cols].values.astype(np.float32)
    times = df[time_col].values.astype(np.float32)
    events = df[event_col].values.astype(np.float32)
    
    # Split
    X_train, X_test, t_train, t_test, e_train, e_test = train_test_split(
        X, times, events, test_size=0.2, random_state=42, stratify=events
    )
    X_train, X_val, t_train, t_val, e_train, e_val = train_test_split(
        X_train, t_train, e_train, test_size=0.1, random_state=42, stratify=e_train
    )
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Prepare targets [time, event]
    y_train = np.stack([t_train, e_train], axis=1)
    y_val = np.stack([t_val, e_val], axis=1)
    
    # --- Hyperparameter Tuning ---
    search_space = {
        'hidden_dims': [[32, 32], [64, 32], [128, 64, 32]],
        'dropout': [0.1, 0.3],
        'lr': [0.001, 0.0005],
        'batch_size': [64]
    }
    
    best_config = run_tuning(X_train, y_train, X_val, y_val, search_space)
    
    # --- Retrain Best Model ---
    print("\nRetraining best model on full training set...")
    model = DeepSurv(X_train.shape[1], best_config['hidden_dims'], best_config['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_config['lr'])
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(train_ds, batch_size=best_config['batch_size'], shuffle=True)
    
    # Train for longer (100 epochs or early stop)
    best_model_state = None
    best_val_ci = -1
    patience = 10
    counter = 0
    
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    val_loader = DataLoader(val_ds, batch_size=best_config['batch_size'])
    
    for epoch in range(100):
        train_loss = train_one_epoch(model, loader, optimizer)
        val_ci = evaluate(model, val_loader)
        
        if val_ci > best_val_ci:
            best_val_ci = val_ci
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
    # Save Model
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    if args.m_imp is not None:
        save_name = f"deepsurv_tuned_{args.dataset}_{args.scenario}_{args.method}_{args.m_imp}.pt"
    else:
        save_name = f"deepsurv_tuned_{args.dataset}_{args.scenario}_{args.method}.pt"
        
    save_path = os.path.join(model_dir, save_name)
    torch.save({
        'state_dict': best_model_state,
        'config': best_config,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'feature_cols': feature_cols
    }, save_path)
    
    print(f"Saved tuned model to {save_path}")

if __name__ == "__main__":
    main()
