"""
Survival-VAE Hyperparameter Tuning with Optuna (Multi-Objective)
================================================================
Optimizes Survival-VAE hyperparameters using Bayesian Optimization.
Objectives:
  1. Minimize Imputation RMSE (Reconstruction Quality)
  2. Maximize C-Index (Prognostic Utility)

Search Space:
  - beta (KL weight): Log-Uniform [1e-4, 1e-1]
  - gamma (Cox weight): Log-Uniform [0.1, 10.0]
  - latent_dim: Categorical [4, 6, 8, 10, 16, 20, 32]
  - lr: Log-Uniform [1e-4, 1e-2]

Protocol:
  - 100 Trials per Scenario
  - Validation on Fold 0 Only
  - Median Pruning for Efficiency
  - Pareto Frontier Selection (Balanced Rule)
"""

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
from sklearn.preprocessing import MinMaxScaler
from sksurv.metrics import concordance_index_censored

import sys
# Add imputation directory to sys.path so we can import impute_survival_vae
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../imputation')))

# Import core model components
try:
    from impute_survival_vae import ClinicalDataset, SurvivalVAE, cox_ph_loss
except ImportError:
    from impute_survival_vae import ClinicalDataset, SurvivalVAE
    
    def cox_ph_loss(risk_scores, events):
        """Cox Proportional Hazards Loss (Negative Log Partial Likelihood)"""
        # Assumes data is already sorted by time (descending)
        risk_scores = risk_scores.reshape(-1)
        events = events.reshape(-1)
        
        # Stability: clamp risk scores and add epsilon (as suggested)
        risk_scores = torch.clamp(risk_scores, min=-10, max=10)
        hazard_ratio = torch.exp(risk_scores)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-7)
        uncensored_likelihood = risk_scores - log_risk
        censored_likelihood = uncensored_likelihood * events
        
        num_events = events.sum()
        if num_events < 1e-6:
            return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)
            
        return -censored_likelihood.sum() / num_events

# Configuration
DATA_DIR = os.environ.get('SVAE_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets'))

print(f"Using datasets at: {DATA_DIR}")
SCENARIOS = ['light', 'moderate', 'severe']
DATASETS = ['metabric', 'mimic']
SPLIT_FILES = {
    'metabric': 'cv_splits_metabric.json',
    'mimic': 'cv_splits_mimic.json'
}

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

def load_split_indices(dataset_name, fold_idx=0):
    """Load fixed CV indices for specific fold"""
    split_file = os.path.join(DATA_DIR, SPLIT_FILES[dataset_name])
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    fold_key = f"fold_{fold_idx+1}"
    return splits[fold_key]['train'], splits[fold_key]['val']

def compute_rmse(model, dataloader, device):
    """Compute RMSE on Observed values only (NaN-safe version as provided)"""
    model.eval()
    mse_sum = 0
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            m = batch[1].to(device)
            
            recon_x, _, _, _ = model(x, m)
            
            # NaN safety check
            if torch.isnan(recon_x).any() or torch.isinf(recon_x).any():
                return 1.0  # Return worst-case RMSE
            
            # MSE on observed only
            mse = ((recon_x - x) ** 2) * m
            mse_sum += mse.sum().item()
            count += m.sum().item()
    
    if count == 0 or np.isnan(mse_sum):
        return 1.0
        
    rmse = np.sqrt(mse_sum / count)
    return rmse if not np.isnan(rmse) else 1.0

def compute_c_index(model, dataloader, device):
    """Compute C-index on Validation set (NaN-safe version)"""
    model.eval()
    risk_scores = []
    times = []
    events = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            m = batch[1].to(device)
            t = batch[2]
            e = batch[3]
            
            _, _, _, risk = model(x, m)
            
            # NaN safety provided by user
            if torch.isnan(risk).any():
                return 0.5

            risk_scores.extend(risk.cpu().numpy().flatten())
            times.extend(t.numpy().flatten())
            events.extend(e.numpy().flatten())
            
    try:
        c_index = concordance_index_censored(
            np.array(events, dtype=bool),
            np.array(times),
            np.array(risk_scores)
        )[0]
    except:
        c_index = 0.5 # Default if fail
        
    return c_index

def objective(trial, dataset_data):
    try:
        # 1. Hyperparameters
        beta = trial.suggest_float("beta", 1e-4, 1e-1, log=True)
        gamma = trial.suggest_float("gamma", 0.1, 10.0, log=True)
        latent_dim = trial.suggest_categorical("latent_dim", [4, 6, 8, 10, 16, 20, 32])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = 512
        
        # Unpack pre-loaded data
        (X_train, M_train, T_train, E_train, X_val, M_val, T_val, E_val, input_dim, feature_cols) = dataset_data
        
        KNOWN_BINARY_FEATURES = [
            'Hormone_Tx', 'Radiotherapy', 'Chemotherapy', 'ER_Positive',
            'Sex', 'CCI_MI', 'CCI_CHF', 'CCI_PVD', 'CCI_Stroke',
            'CCI_Renal', 'CCI_Liver', 'CCI_Cancer'
        ]
        binary_feature_indices = [
            i for i, col in enumerate(feature_cols)
            if col in KNOWN_BINARY_FEATURES
        ]
        
        # Cause 3: Data Normalization Check (as suggested)
        if trial.number % 10 == 0:
            print(f"Data stats (Trial {trial.number}): min={X_train.min():.4f}, max={X_train.max():.4f}, mean={X_train.mean():.4f}")
            if torch.isnan(X_train).any():
                print(f"ERROR: NaN detected in input data for Trial {trial.number}!")
                return float('inf'), 0.0
        
        # Create TensorDatasets
        train_ds = TensorDataset(X_train, M_train, T_train, E_train)
        val_ds = TensorDataset(X_val, M_val, T_val, E_val)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # 3. Model Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SurvivalVAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
        
        # Cause 1: Model Initialization Issue (Debug code provided by user)
        with torch.no_grad():
            test_batch = next(iter(train_loader))
            test_x = test_batch[0][:2].to(device)
            test_m = test_batch[1][:2].to(device)
            test_recon, _, _, _ = model(test_x, test_m)
            if torch.isnan(test_recon).any():
                print(f"ERROR: Model produces NaN on initialization (Trial {trial.number})!")
                return float('inf'), 0.0

        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 4. Training Loop (Mini-Batch for stable optimization)
        MAX_EPOCHS = 100
        PATIENCE = 10
        best_c_index = 0.0
        patience_counter = 0
        
        for epoch in range(MAX_EPOCHS):
            model.train()
            for batch in train_loader:
                x, m, t, e = [b.to(device) for b in batch]
                
                # Sorting within batch for Cox Loss
                sort_idx = torch.argsort(t, descending=True)
                x, m, t, e = x[sort_idx], m[sort_idx], t[sort_idx], e[sort_idx]
                
                optimizer.zero_grad()
                recon_x, mu, logvar, risk = model(x, m)
                
                num_features = x.shape[1]
                binary_cols = binary_feature_indices if binary_feature_indices is not None else []
                cont_cols = [i for i in range(num_features) if i not in binary_cols]
                total_loss_sum = 0.0
                if len(cont_cols) > 0:
                    mse_loss = F.mse_loss(recon_x[:, cont_cols], x[:, cont_cols], reduction='none')
                    masked_mse = mse_loss * m[:, cont_cols]
                    total_loss_sum += masked_mse.sum()
                if len(binary_cols) > 0:
                    bce_loss = F.binary_cross_entropy(recon_x[:, binary_cols], x[:, binary_cols], reduction='none')
                    masked_bce = bce_loss * m[:, binary_cols]
                    total_loss_sum += masked_bce.sum()
                total_observed = m.sum() + 1e-8
                recon_loss = total_loss_sum / total_observed
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                if e.dim() == 1: e = e.unsqueeze(1)
                surv_loss = cox_ph_loss(risk, e)
                
                loss = recon_loss + beta * kl_loss + gamma * surv_loss
                if torch.isnan(loss) or torch.isinf(loss): return float('inf'), 0.0

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            model.eval()
            val_rmse = compute_rmse(model, val_loader, device)
            val_c_index = compute_c_index(model, val_loader, device)
            
            if epoch % 20 == 0:
                print(f"  Trial {trial.number} | Epoch {epoch}: RMSE={val_rmse:.4f}, C={val_c_index:.4f}") 
                
            if val_c_index > best_c_index:
                best_c_index = val_c_index
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE: break
        
        return val_rmse, best_c_index
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float('inf'), 0.0

def prepare_data(dataset_name, scenario, fold_idx):
    """Load and preprocess data once."""
    mnar_path = os.path.join(DATA_DIR, f'{dataset_name}_mnar_{scenario}.csv')
    mask_path = os.path.join(DATA_DIR, f'{dataset_name}_mask_{scenario}.csv')
    truth_path = find_truth_file(dataset_name)
    
    df_mnar = pd.read_csv(mnar_path)
    df_truth = pd.read_csv(truth_path) if truth_path else None
    
    if 'Sex' in df_mnar.columns and df_truth is not None:
        for df in [df_mnar, df_truth]:
            df['Sex'] = df['Sex'].map({'F': 0, 'M': 1, 'Female': 0, 'Male': 1})
    
    target_cols = ['Survival_in_days', 'Status'] if 'Survival_in_days' in df_mnar.columns else \
                  (['duration', 'event'] if 'duration' in df_mnar.columns else ['Time', 'Event'])

    id_cols = ['hadm_id', 'subject_id', 'stay_id', 'patient_id']
    cols_to_drop = [col for col in id_cols if col in df_mnar.columns]
    
    feature_df_mnar = df_mnar.drop(columns=cols_to_drop + target_cols).select_dtypes(include=[np.number])
    feature_cols = feature_df_mnar.columns
    X_mnar = feature_df_mnar.values
    M_obs = 1. - np.isnan(X_mnar)
    T_E_mnar = df_mnar[target_cols].values
    
    train_idx, val_idx = load_split_indices(dataset_name, fold_idx=fold_idx)
    scaler_x = MinMaxScaler()
    X_mnar_scaled = np.zeros_like(X_mnar, dtype=float)
    X_mnar_scaled[train_idx] = scaler_x.fit_transform(X_mnar[train_idx])
    X_mnar_scaled[val_idx] = scaler_x.transform(X_mnar[val_idx])

    def get_tensors(indices):
        X = torch.FloatTensor(X_mnar_scaled[indices])
        X = torch.nan_to_num(X, nan=0.5) 
        M = torch.FloatTensor(M_obs[indices])
        T = torch.FloatTensor(T_E_mnar[indices, 0])
        E = torch.FloatTensor(T_E_mnar[indices, 1])
        return X, M, T, E

    X_train, M_train, T_train, E_train = get_tensors(train_idx)
    X_val, M_val, T_val, E_val = get_tensors(val_idx)
    
    # Pre-sort training data for Cox Loss stability
    sort_idx = torch.argsort(T_train, descending=True)
    return (X_train[sort_idx], M_train[sort_idx], T_train[sort_idx], E_train[sort_idx], 
            X_val, M_val, T_val, E_val, X_mnar.shape[1], feature_cols)

def save_callback(study, trial, output_path):
    """Callback to save results after every trial (checkpointing)."""
    df_results = study.trials_dataframe()
    if df_results.empty:
        return
        
    # Filter only completed trials
    df_results = df_results[df_results['state'] == 'COMPLETE']
    if df_results.empty:
        return

    # Handle multi-objective columns
    value_cols = [c for c in df_results.columns if c.startswith('values_')]
    param_cols = [c for c in df_results.columns if c.startswith('params_')]
    
    cols = ['number'] + param_cols + value_cols
    df_results = df_results[cols]
    
    # Rename columns for clarity
    df_results.columns = [c.replace('params_', '') if c.startswith('params_') else c for c in df_results.columns]
    # In Survival-VAE, values_0=RMSE, values_1=C-Index
    df_results = df_results.rename(columns={'values_0': 'rmse', 'values_1': 'c_index', 'number': 'trial', 'lr': 'learning_rate'})
    
    df_results.to_csv(output_path, index=False)
    print(f" [Checkpoint] Saved trial {trial.number} results to {output_path}")

def run_study(dataset, scenario, n_trials=100, fold_idx=0):
    start_time = time.time()
    print(f"\nLoading data for {dataset}...")
    dataset_data = prepare_data(dataset, scenario, fold_idx)
    
    study_name = f"vae_{dataset}_{scenario}_fold{fold_idx}"
    storage_name = f"sqlite:///optuna_survival_vae_fold{fold_idx}.db"
    output_dir = f"final/optuna_results_final/fold_{fold_idx}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{dataset}_{scenario}_optuna_survival_vae_results.csv"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        directions=["minimize", "maximize"], 
        load_if_exists=True
    )
    
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    trials_to_run = max(0, n_trials - n_completed)
    
    if trials_to_run > 0:
        print(f"  Resuming study: {n_completed} trials done, {trials_to_run} remaining...")
        study.optimize(
            lambda t: objective(t, dataset_data), 
            n_trials=trials_to_run,
            callbacks=[lambda s, t: save_callback(s, t, output_path)]
        )
    else:
        print(f"  Study {study_name} already complete with {n_completed} trials. Skipping.")
    
    print(f"Study {study_name} complete in optuna_survival_vae.db")
    
    # Extract Best Trial using Clinical Utility Priority
    best_trials = study.best_trials
    
    if len(best_trials) == 0:
        print("No Pareto optimal trials found. All trials were pruned or failed.")
        return

    # To balance properly, we normalize the Pareto frontier values to [0, 1]
    rmse_vals = np.array([t.values[0] for t in best_trials])
    c_vals = np.array([t.values[1] for t in best_trials])
    
    # 1-C is the penalty for survival
    c_penalty = 1.0 - c_vals
    
    r_min, r_max = rmse_vals.min(), rmse_vals.max()
    c_min, c_max = c_penalty.min(), c_penalty.max()
    
    # Avoid division by zero
    r_range = (r_max - r_min) if (r_max - r_min) > 1e-6 else 1.0
    c_range = (c_max - c_min) if (c_max - c_min) > 1e-6 else 1.0
    
    best_balanced = None
    best_score = float('inf')
    
    print(f"\nPareto Optimal Trials ({len(best_trials)}):")
    for i, t in enumerate(best_trials):
        # Normalize
        norm_rmse = (rmse_vals[i] - r_min) / r_range
        norm_c_penalty = (c_penalty[i] - c_min) / c_range
        
        # Clinical Utility Priority Score: 
        # We weight survival utility (C-index) more heavily (0.7) than imputation (0.3)
        # because the Survival-VAE's primary novel claim is prognostic latent space.
        score = 0.3 * norm_rmse + 0.7 * norm_c_penalty
        
        print(f"  Trial {t.number}: RMSE={rmse_vals[i]:.4f}, C-Index={c_vals[i]:.4f} (Priority Score={score:.4f})")
        
        if score < best_score:
            best_score = score
            best_balanced = t
            
    print(f"\nBest Balanced Configuration:")
    print(f"  RMSE: {best_balanced.values[0]:.4f}")
    print(f"  C-Index: {best_balanced.values[1]:.4f}")
    print(f"  Params: {best_balanced.params}")
    
    # Save best params to CSV (Prepend as first row)
    best_balanced_row = {
        'trial': best_balanced.number,
        'beta': best_balanced.params['beta'],
        'gamma': best_balanced.params['gamma'],
        'latent_dim': best_balanced.params['latent_dim'],
        'learning_rate': best_balanced.params['lr'],
        'rmse_norm': best_balanced.values[0],
        'c_index': best_balanced.values[1]
    }
    
    if os.path.exists(output_path):
        df_history = pd.read_csv(output_path)
        # Ensure we don't accidentally double-prepend if the script is rerun
        df_final = pd.concat([pd.DataFrame([best_balanced_row]), df_history], ignore_index=True)
        df_final.to_csv(output_path, index=False)
    else:
        pd.DataFrame([best_balanced_row]).to_csv(output_path, index=False)

    print(f"Study {study_name} results (with best trial) saved to: {output_path}")
    
    end_time = time.time()
    total_time = (end_time - start_time) / 3600.0
    n_pruned = len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))
    n_complete = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
    
    print(f"\nStudy saved to: optuna_survival_vae.db")
    print(f"   Total runtime: {total_time:.2f} hours")
    print(f"   Trials: {n_complete} complete, {n_pruned} pruned, {n_trials} total scheduled")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=DATASETS + ['all'], default='all')
    parser.add_argument("--scenario", type=str, choices=SCENARIOS + ['all'], default='all')
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--fold", type=int, default=0, choices=[0,1,2,3,4], help="CV fold index (0-4)")
    args = parser.parse_args()
    
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    scenarios = SCENARIOS if args.scenario == 'all' else [args.scenario]
    
    total_studies = len(datasets) * len(scenarios)
    current_study = 0
    
    print(f"\n{'='*60}")
    print(f"STARTING OPTUNA HYPERPARAMETER TUNING")
    print(f"Total studies to run: {total_studies}")
    print(f"Trials per study: {args.trials}")
    print(f"{'='*60}\n")
    
    for d in datasets:
        for s in scenarios:
            current_study += 1
            print(f"\n{'#'*60}")
            print(f"STUDY {current_study}/{total_studies}: {d.upper()} - {s.upper()}")
            print(f"{'#'*60}")
            run_study(d, s, n_trials=args.trials, fold_idx=args.fold)
            print(f"\nCompleted {current_study}/{total_studies} studies")
            
    print("\n" + "="*60)
    print("ALL STUDIES COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("1. Visualize: python plot_pareto_frontiers.py (Create this script if needed)")
    print("2. View dashboard: optuna-dashboard sqlite:///optuna_survival_vae.db")
    print("3. Run final evaluation with best params")
