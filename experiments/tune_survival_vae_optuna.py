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
POSSIBLE_DATA_DIRS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets'))  # Relative path
]

DATA_DIR = None
for path in POSSIBLE_DATA_DIRS:
    if os.path.exists(path):
        DATA_DIR = path
        break

if DATA_DIR is None:
    raise FileNotFoundError(f"Could not find datasets directory. Checked: {POSSIBLE_DATA_DIRS}")

print(f"Using datasets at: {DATA_DIR}")
SCENARIOS = ['light', 'moderate', 'severe']
DATASETS = ['metabric', 'mimic']
SPLIT_FILES = {
    'metabric': 'cv_splits_metabric.json',
    'mimic': 'cv_splits_mimic.json'
}

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

def objective(trial, dataset_name, scenario):
    try:
        # 1. Hyperparameters
        beta = trial.suggest_float("beta", 1e-4, 1e-1, log=True)
        gamma = trial.suggest_float("gamma", 0.1, 10.0, log=True)
        latent_dim = trial.suggest_categorical("latent_dim", [4, 6, 8, 10, 16, 20, 32])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = 64
        
        # 2. Data Loading (Fold 0)
        mnar_path = os.path.join(DATA_DIR, f'{dataset_name}_mnar_{scenario}.csv')
        mask_path = os.path.join(DATA_DIR, f'{dataset_name}_mask_{scenario}.csv')
        
        # Load full dataset using ClinicalDataset class (handles preprocessing)
        full_dataset = ClinicalDataset(mnar_path, mask_path)
        
        # Get indices
        train_idx, val_idx = load_split_indices(dataset_name, fold_idx=0)
        
        # Extract data tensors from ClinicalDataset
        # We need to extract them to implement the Pre-Sorting Optimization
        # ClinicalDataset has .data (X), .mask (M), .targets (T, E)
        
        # Helper to extract tensors for indices
        def get_tensors(indices):
            # FIXED: Use scaled data with 0.5 padding for missingness
            # consistent with evaluate_survival_vae_cv.py
            X = torch.FloatTensor(full_dataset.data[indices])
            X = torch.nan_to_num(X, nan=0.5) 
                
            M = torch.FloatTensor(full_dataset.mask[indices])
            T = torch.FloatTensor(full_dataset.targets[indices, 0])
            E = torch.FloatTensor(full_dataset.targets[indices, 1])
            return X, M, T, E

        X_train, M_train, T_train, E_train = get_tensors(train_idx)
        X_val, M_val, T_val, E_val = get_tensors(val_idx)
        
        # Cause 3: Data Normalization Check (as suggested)
        if trial.number % 10 == 0:
            print(f"Data stats (Trial {trial.number}): min={X_train.min():.4f}, max={X_train.max():.4f}, mean={X_train.mean():.4f}")
            if torch.isnan(X_train).any():
                print(f"ERROR: NaN detected in input data for Trial {trial.number}!")
                return float('inf'), 0.0
        
        # CRITICAL OPTIMIZATION: Sort Training Data by Time (Descending) ONCE
        # This replaces sorting inside the training loop
        sort_idx = torch.argsort(T_train, descending=True)
        X_train = X_train[sort_idx]
        M_train = M_train[sort_idx]
        T_train = T_train[sort_idx]
        E_train = E_train[sort_idx]
        
        # Create TensorDatasets
        # Note: shuffle=False for train_loader to maintain sorted order for Cox Loss
        train_ds = TensorDataset(X_train, M_train, T_train, E_train)
        val_ds = TensorDataset(X_val, M_val, T_val, E_val)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # 3. Model Setup
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        model = SurvivalVAE(input_dim=full_dataset.data.shape[1], latent_dim=latent_dim).to(device)
        
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
            epoch_loss = 0
            for batch in train_loader:
                x = batch[0].to(device)
                m = batch[1].to(device)
                t = batch[2].to(device)
                e = batch[3].to(device)
                
                # Pre-sorted sorting (within batch if needed, but T_train was pre-sorted)
                # Keep sorting for batch-level safety
                sort_idx = torch.argsort(t, descending=True)
                x, m, t, e = x[sort_idx], m[sort_idx], t[sort_idx], e[sort_idx]
                
                optimizer.zero_grad()
                recon_x, mu, logvar, risk = model(x, m)
                
                # 1. Normalized Recon Loss (Per-patient Scale)
                mse = F.mse_loss(recon_x, x, reduction='none')
                num_obs = m.sum()
                avg_mse = (mse * m).sum() / (num_obs + 1e-8)
                recon_loss = avg_mse * x.shape[1] 
                
                # 2. KL Loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                
                # 3. Surv Loss
                if e.dim() == 1: e = e.unsqueeze(1)
                surv_loss = cox_ph_loss(risk, e)
                
                loss = recon_loss + beta * kl_loss + gamma * surv_loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    return float('inf'), 0.0

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Validation (using compute logic)
            model.eval()
            val_rmse = compute_rmse(model, val_loader, device)
            val_c_index = compute_c_index(model, val_loader, device)
            
            # Progress Logging
            # Log progress for every 20 epochs for ALL trials so progress is visible
            if epoch % 20 == 0:
                print(f"  Trial {trial.number} | Epoch {epoch}: RMSE={val_rmse:.4f}, C={val_c_index:.4f}") 
            
            # Pruning is NOT supported in multi-objective optimization
            # trial.report(val_c_index, epoch)
            # if trial.should_prune():
            #     raise optuna.TrialPruned()
                
            # Early Stopping on C-Index
            if val_c_index > best_c_index:
                best_c_index = val_c_index
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break
        
        # Final log for successful trial
        print(f"Trial {trial.number} finished: RMSE={val_rmse:.4f}, C-Index={best_c_index:.4f}")
        return val_rmse, best_c_index

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        # Return worst possible values to avoid crashing study
        return float('inf'), 0.0

def run_study(dataset, scenario, n_trials=100):
    start_time = time.time()
    print(f"\n{'-'*60}")
    print(f"Running Optuna Study: {dataset.upper()} - {scenario.upper()}")
    print(f"{'-'*60}")
    
    study_name = f"vae_{dataset}_{scenario}"
    storage_name = "sqlite:///optuna_survival_vae.db"
    
    # Multi-objective: Minimize RMSE, Maximize C-Index
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        directions=["minimize", "maximize"], 
        # pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=10), # Not supported
        load_if_exists=True
    )
    
    study.optimize(lambda t: objective(t, dataset, scenario), n_trials=n_trials)
    
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
    
    # Save best params to JSON
    result = {
        "dataset": dataset,
        "scenario": scenario,
        "best_params": best_balanced.params,
        "best_metrics": {
            "rmse": best_balanced.values[0],
            "c_index": best_balanced.values[1]
        }
    }
    
    os.makedirs("optuna_results", exist_ok=True)
    with open(f"optuna_results/best_{dataset}_{scenario}.json", "w") as f:
        json.dump(result, f, indent=4)
        
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
            run_study(d, s, n_trials=args.trials)
            print(f"\nCompleted {current_study}/{total_studies} studies")
            
    print("\n" + "="*60)
    print("ALL STUDIES COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("1. Visualize: python plot_pareto_frontiers.py (Create this script if needed)")
    print("2. View dashboard: optuna-dashboard sqlite:///optuna_survival_vae.db")
    print("3. Run final evaluation with best params")
