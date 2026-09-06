import os
import argparse
import json
import time
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler

# Silence all scikit-learn and convergence warnings to prevent log explosion
warnings.filterwarnings("ignore")

# Configuration
POSSIBLE_DATA_DIRS = [
    './datasets',
    'datasets',
    '../datasets',
    '/home/azman/VAE/Survival-VAE_study',
    '/Users/azmannads/VAE 2/Survival-VAE_study',
    '/Users/azmannads/Documents/Research collections/Research 2025/datasets',
    '.'
]

DATA_DIR = None
for p in POSSIBLE_DATA_DIRS:
    if os.path.exists(p):
        DATA_DIR = p
        break

if DATA_DIR is None:
    DATA_DIR = 'datasets'
    print(f"Warning: Could not find datasets directory. Defaulting to: {DATA_DIR}")

def find_truth_file(dataset_name):
    if dataset_name == 'metabric':
        filenames = ['metabric_processed.csv', 'metabric.csv']
    else:
        filenames = ['mimic_sepsis_highdim.csv', 'mimic_processed.csv', 'mimic.csv']
    
    search_dirs = [
        DATA_DIR,
        os.path.join(DATA_DIR, 'final'),
        os.path.join(DATA_DIR, 'raw'),
        os.path.dirname(DATA_DIR),
        os.path.join(os.path.dirname(DATA_DIR), 'final'),
        os.path.join(os.path.dirname(DATA_DIR), 'raw'),
        os.path.join(os.path.dirname(DATA_DIR), 'data'),
        '.',
        './final',
        '../datasets',
        '../datasets/final'
    ]
    
    for d in search_dirs:
        for f in filenames:
            full_path = os.path.join(d, f)
            if os.path.exists(full_path):
                return full_path
    return None

SPLIT_FILES = {
    'metabric': 'cv_splits_metabric.json',
    'mimic': 'cv_splits_mimic.json'
}

DATASETS = ['metabric', 'mimic']
SCENARIOS = ['light', 'moderate', 'severe']

def load_split_indices(dataset_name, fold_idx=0):
    split_file = os.path.join(DATA_DIR, SPLIT_FILES[dataset_name])
    with open(split_file, 'r') as f:
        splits = json.load(f)
    fold_key = f"fold_{fold_idx+1}"
    return splits[fold_key]['train'], splits[fold_key]['val']

class NetG(nn.Module):
    def __init__(self, dim, hidden_dim):
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
        return self.sigmoid(self.fc3(out))

class NetD(nn.Module):
    def __init__(self, dim, hidden_dim):
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
        return self.sigmoid(self.fc3(out))

def sample_binary(m, n, p=0.5):
    return (np.random.uniform(0., 1., size=[m, n]) > p).astype(float)

def objective(trial, dataset_data):
    try:
        # 1. Hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        alpha = trial.suggest_float("alpha", 10, 200)
        hint_rate = trial.suggest_float("hint_rate", 0.5, 0.95)
        hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
        batch_size = 512
        epochs = 100
        
        # Unpack pre-loaded data
        (X_train, M_train, X_input_all, M_input_all, X_val_truth_scaled, mask_val_art, dim_all, feature_cols, binary_feature_indices) = dataset_data
        
        # 3. Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        netG = NetG(dim_all, hidden_dim).to(device)
        netD = NetD(dim_all, hidden_dim).to(device)
        optG = optim.Adam(netG.parameters(), lr=lr)
        optD = optim.Adam(netD.parameters(), lr=lr)
        
        # 4. Training
        for epoch in range(epochs):
            sample_idx = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch_idx = sample_idx[i:i+batch_size]
                X_mb = torch.FloatTensor(X_train[batch_idx]).to(device)
                M_mb = torch.FloatTensor(M_train[batch_idx]).to(device)
                
                H_mb_binary = torch.FloatTensor(sample_binary(len(batch_idx), dim_all, hint_rate)).to(device)
                H_mb = M_mb * H_mb_binary + 0.5 * (1 - H_mb_binary)
                Z_mb = torch.FloatTensor(np.random.uniform(0., 0.01, size=[len(batch_idx), dim_all])).to(device)
                
                optD.zero_grad()
                G_sample = netG(X_mb * M_mb + Z_mb * (1-M_mb), M_mb)
                X_hat = X_mb * M_mb + G_sample * (1-M_mb)
                D_prob = netD(X_hat, H_mb)
                D_loss = -torch.mean(M_mb * torch.log(D_prob + 1e-8) + (1-M_mb) * torch.log(1. - D_prob + 1e-8))
                D_loss.backward()
                optD.step()
                
                optG.zero_grad()
                G_sample = netG(X_mb * M_mb + Z_mb * (1-M_mb), M_mb)
                X_hat = X_mb * M_mb + G_sample * (1-M_mb)
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
                optG.step()
                
        # 5. Evaluate
        netG.eval()
        with torch.no_grad():
            X_val_mb = torch.FloatTensor(X_input_all).to(device)
            M_val_mb = torch.FloatTensor(M_input_all).to(device)
            G_val = netG(X_val_mb, M_val_mb).cpu().numpy()
            
            G_val_feat = G_val[:, :len(feature_cols)]
            X_val_feat = X_input_all[:, :len(feature_cols)]
            M_val_feat = M_input_all[:, :len(feature_cols)]
            X_val_imputed = M_val_feat * X_val_feat + (1 - M_val_feat) * G_val_feat
            
            artificial_idx = mask_val_art == 1
            if np.sum(artificial_idx) == 0: return 1.0
            
            mse = (X_val_imputed[artificial_idx] - X_val_truth_scaled[artificial_idx])**2
            rmse = np.sqrt(np.mean(mse))
            
        return rmse if not np.isnan(rmse) else 1.0
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 1.0

def save_callback(study, trial, output_path):
    """Callback to save results after every trial (checkpointing)."""
    df_results = study.trials_dataframe()
    if df_results.empty:
        return
        
    cols = ['number'] + [c for c in df_results.columns if c.startswith('params_')] + ['value']
    df_results = df_results[cols].sort_values('value')
    df_results.columns = [c.replace('params_', '') if c.startswith('params_') else c for c in df_results.columns]
    df_results = df_results.rename(columns={'number': 'trial', 'value': 'rmse_norm'})
    
    df_results.to_csv(output_path, index=False)
    print(f" [Checkpoint] Saved trial {trial.number} results to {output_path}")

def prepare_data(dataset_name, scenario, fold_idx):
    """Load and preprocess data once."""
    mnar_path = os.path.join(DATA_DIR, f'{dataset_name}_mnar_{scenario}.csv')
    truth_path = find_truth_file(dataset_name)
    mask_path = os.path.join(DATA_DIR, f'{dataset_name}_mask_{scenario}.csv')
    
    if not truth_path or not os.path.exists(truth_path):
        raise FileNotFoundError(f"Truth file for {dataset_name} not found in {DATA_DIR}")
        
    df_mnar = pd.read_csv(mnar_path)
    df_truth = pd.read_csv(truth_path)
    df_mask = pd.read_csv(mask_path)
    
    if 'Sex' in df_mnar.columns:
        for df in [df_mnar, df_truth]:
            df['Sex'] = df['Sex'].map({'F': 0, 'M': 1, 'Female': 0, 'Male': 1})
    
    target_cols = ['Survival_in_days', 'Status'] if 'Survival_in_days' in df_mnar.columns else \
                  (['duration', 'event'] if 'duration' in df_mnar.columns else ['Time', 'Event'])

    id_cols = ['hadm_id', 'subject_id', 'stay_id', 'icustay_id', 'patient_id', 'admittime', 'dischtime']
    cols_to_drop = [col for col in id_cols if col in df_mnar.columns]
    
    feature_df_mnar = df_mnar.drop(columns=cols_to_drop + target_cols).select_dtypes(include=[np.number])
    feature_cols = feature_df_mnar.columns
    
    X_mnar = feature_df_mnar.values
    X_truth = df_truth[feature_cols].values
    mask_art = df_mask[feature_cols].values.astype(bool).astype(float)
    T_E_mnar = df_mnar[target_cols].values
    
    train_idx, val_idx = load_split_indices(dataset_name, fold_idx=fold_idx)
    scaler_x = MinMaxScaler()
    X_mnar_scaled = np.zeros_like(X_mnar, dtype=float)
    X_mnar_scaled[train_idx] = scaler_x.fit_transform(X_mnar[train_idx])
    X_mnar_scaled[val_idx] = scaler_x.transform(X_mnar[val_idx])
    X_truth_scaled = scaler_x.transform(X_truth)
    
    M_train_obs = 1. - np.isnan(X_mnar_scaled)
    X_input_filled = np.nan_to_num(X_mnar_scaled, nan=0.5)
    
    scaler_te = MinMaxScaler()
    TE_mnar_scaled = np.zeros_like(T_E_mnar, dtype=float)
    TE_mnar_scaled[train_idx] = scaler_te.fit_transform(T_E_mnar[train_idx])
    TE_mnar_scaled[val_idx] = scaler_te.transform(T_E_mnar[val_idx])
    
    X_input_all = np.concatenate([X_input_filled, TE_mnar_scaled], axis=1)
    M_input_all = np.concatenate([M_train_obs, np.ones_like(TE_mnar_scaled)], axis=1)
    
    X_train, M_train = X_input_all[train_idx], M_input_all[train_idx]
    X_val_input_all, M_val_input_all = X_input_all[val_idx], M_input_all[val_idx]
    X_val_truth_scaled = X_truth_scaled[val_idx]
    mask_val_art = mask_art[val_idx]
    
    return (X_train, M_train, X_val_input_all, M_val_input_all, X_val_truth_scaled, mask_val_art, X_input_all.shape[1], feature_cols, None)

def run_study(dataset, scenario, n_trials=100, fold_idx=0):
    study_name = f"gain_{dataset}_{scenario}_fold{fold_idx}{suffix_str}"
    os.makedirs(f"final/optuna_results_final/fold_{fold_idx}", exist_ok=True)
    output_path = f"final/optuna_results_final/fold_{fold_idx}/{dataset}_{scenario}_optuna_gain_results{suffix_str}.csv"
    
    # Pre-load data once
    print(f"Loading data for {dataset}...")
    dataset_data = prepare_data(dataset, scenario, fold_idx)
    
    # Use SQLite for persistence to allow skipping/resuming

    # Use SQLite for persistence to allow skipping/resuming
    study_name = f"gain_{dataset}_{scenario}_fold{fold_idx}{suffix_str}"
    storage_name = f"sqlite:///optuna_gain_fold{fold_idx}{suffix_str}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True
    )
    
    # Check how many trials are already complete
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    trials_to_run = max(0, n_trials - completed_trials)
    
    if trials_to_run > 0:
        print(f"  Resuming study: {completed_trials} trials done, {trials_to_run} remaining...")
        study.optimize(
            lambda t: objective(t, dataset_data), 
            n_trials=trials_to_run,
            callbacks=[lambda s, t: save_callback(s, t, output_path)]
        )
    else:
        print(f"  Study {study_name} already complete with {completed_trials} trials. Skipping.")
    
    print(f"Completed study: {study_name}. Final results in {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--dataset", type=str, choices=DATASETS, help="Run only for this dataset")
    parser.add_argument("--scenario", type=str, choices=SCENARIOS, help="Run only for this scenario")
    parser.add_argument("--fold", type=int, default=0, choices=[0,1,2,3,4], help="CV fold index (0-4)")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix for output files")
    args = parser.parse_args()
    suffix_str = f"_{args.suffix}" if args.suffix else ""
    
    target_datasets = [args.dataset] if args.dataset else DATASETS
    target_scenarios = [args.scenario] if args.scenario else SCENARIOS
    
    for d in target_datasets:
        for s in target_scenarios:
            print(f"Optimizing GAIN for {d} - {s} (Trials: {args.trials})...")
            run_study(d, s, n_trials=args.trials, fold_idx=args.fold)
