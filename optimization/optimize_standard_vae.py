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

class StandardVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(StandardVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, m):
        h = self.encoder(torch.cat([x, m], dim=1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def objective(trial, dataset_data):
    try:
        # 1. Hyperparameters
        beta = trial.suggest_float("beta", 1e-4, 1e-1, log=True)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        latent_dim = trial.suggest_int("latent_dim", 8, 32)
        epochs = trial.suggest_int("epochs", 50, 150)
        batch_size = 512
        
        # Unpack pre-loaded data
        (X_in_all, M_in_all, X_truth_scaled, mask_art, train_idx, val_idx, feature_cols, binary_feature_indices) = dataset_data
        
        dim_all = X_in_all.shape[1]
        X_train, M_train = X_in_all[train_idx], M_in_all[train_idx]
        
        X_val_truth_scaled = X_truth_scaled[val_idx]
        mask_val_art = mask_art[val_idx]
        
        # 3. Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StandardVAE(dim_all, latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(M_train))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        # 4. Training
        model.train()
        for epoch in range(epochs):
            for x_mb, m_mb in train_loader:
                x_mb, m_mb = x_mb.to(device), m_mb.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(x_mb, m_mb)
                num_features = x_mb.shape[1]
                binary_cols = binary_feature_indices if binary_feature_indices is not None else []
                cont_cols = [i for i in range(num_features) if i not in binary_cols]
                total_loss_sum = 0.0
                if len(cont_cols) > 0:
                    mse_loss = F.mse_loss(recon[:, cont_cols], x_mb[:, cont_cols], reduction='none')
                    masked_mse = mse_loss * m_mb[:, cont_cols]
                    total_loss_sum += masked_mse.sum()
                if len(binary_cols) > 0:
                    bce_loss = F.binary_cross_entropy(recon[:, binary_cols], x_mb[:, binary_cols], reduction='none')
                    masked_bce = bce_loss * m_mb[:, binary_cols]
                    total_loss_sum += masked_bce.sum()
                total_observed = m_mb.sum() + 1e-8
                recon_loss = total_loss_sum / total_observed
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                loss = recon_loss + beta * kl_loss
                loss.backward()
                optimizer.step()
                
        # 5. Evaluate
        model.eval()
        with torch.no_grad():
            X_val_torch = torch.FloatTensor(X_in_all[val_idx]).to(device)
            M_val_torch = torch.FloatTensor(M_in_all[val_idx]).to(device)
            recon_val, _, _ = model(X_val_torch, M_val_torch)
            recon_val = recon_val.cpu().numpy()
            
            recon_feat = recon_val[:, :len(feature_cols)]
            X_val_feat = X_in_all[val_idx, :len(feature_cols)]
            M_val_feat = M_in_all[val_idx, :len(feature_cols)]
            X_val_imputed = M_val_feat * X_val_feat + (1 - M_val_feat) * recon_feat
            
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
    
    if 'Survival_in_days' in df_mnar.columns:
        target_cols = ['Survival_in_days', 'Status']
    elif 'duration' in df_mnar.columns:
        target_cols = ['duration', 'event']
    else:
        target_cols = ['Time', 'Event']

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
    
    M_obs = 1. - np.isnan(X_mnar_scaled)
    X_filled = np.nan_to_num(X_mnar_scaled, nan=0.5)
    
    scaler_te = MinMaxScaler()
    TE_mnar_scaled = np.zeros_like(T_E_mnar, dtype=float)
    TE_mnar_scaled[train_idx] = scaler_te.fit_transform(T_E_mnar[train_idx])
    TE_mnar_scaled[val_idx] = scaler_te.transform(T_E_mnar[val_idx])
    
    X_in_all = np.concatenate([X_filled, TE_mnar_scaled], axis=1)
    M_in_all = np.concatenate([M_obs, np.ones_like(TE_mnar_scaled)], axis=1)
    
    KNOWN_BINARY_FEATURES = [
        'Hormone_Tx', 'Radiotherapy', 'Chemotherapy', 'ER_Positive',
        'Sex', 'CCI_MI', 'CCI_CHF', 'CCI_PVD', 'CCI_Stroke',
        'CCI_Renal', 'CCI_Liver', 'CCI_Cancer'
    ]
    binary_feature_indices = [
        i for i, col in enumerate(feature_cols)
        if col in KNOWN_BINARY_FEATURES
    ]
    return (X_in_all, M_in_all, X_truth_scaled, mask_art, train_idx, val_idx, feature_cols, binary_feature_indices)

def run_study(dataset, scenario, n_trials=100, fold_idx=0):
    study_name = f"standard_vae_{dataset}_{scenario}_fold{fold_idx}{suffix_str}"
    os.makedirs(f"final/optuna_results_final/fold_{fold_idx}", exist_ok=True)
    output_path = f"final/optuna_results_final/fold_{fold_idx}/{dataset}_{scenario}_optuna_standard_vae_results{suffix_str}.csv"
    
    # Pre-load data once per run_study
    print(f"Loading data for {dataset}...")
    dataset_data = prepare_data(dataset, scenario, fold_idx)
    
    # Use SQLite for persistence to allow skipping/resuming

    # Use SQLite for persistence to allow skipping/resuming
    study_name = f"standard_vae_{dataset}_{scenario}_fold{fold_idx}{suffix_str}"
    storage_name = f"sqlite:///optuna_standard_vae_fold{fold_idx}{suffix_str}.db"
    
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
    parser.add_argument("--trials", type=int, default=100) # Reduced default trials for stability
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
            print(f"Optimizing Standard VAE for {d} - {s} (Trials: {args.trials})...")
            run_study(d, s, n_trials=args.trials, fold_idx=args.fold)
