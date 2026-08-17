"""
MICE Imputation Script (Rubin's Rules: m=5)
===========================================
Implements Multiple Imputation by Chained Equations (MICE) for:
1. Two Datasets: METABRIC (Stage A), MIMIC-IV (Stage B)
2. Three Scenarios: Light, Moderate, Severe
3. Five Imputations (m=5): Generates 5 stochastic datasets per scenario

Output: 30 total .csv files (2 * 3 * 5)
"""

import pandas as pd
import numpy as np
import os
import argparse
import json
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import MinMaxScaler

# Configuration
POSSIBLE_DATA_DIRS = [
    './datasets',
    'datasets',
    '../datasets',
    '/home/azman/VAE/Survival-VAE_study',
    '/Users/azmannads/VAE/Survival-VAE_study',
    '/Users/azmannads/Documents/Research collections/Research 2025/datasets',
    '.'
]

# Optuna results are now standardized within the study folder structure

def find_dir(possibilities, default_name):
    for p in possibilities:
        if os.path.exists(p):
            return p
    return default_name

DATA_DIR = find_dir(POSSIBLE_DATA_DIRS, 'datasets')
# RESULTS_DIR will be set dynamically per fold

M_IMPUTATIONS = 5

def run_mice_imputation(dataset_name, severity, fold_idx):
    print(f"Starting MICE Imputation (Rubin's Rules: m={M_IMPUTATIONS}) - Fold {fold_idx}...")
    
    print(f"\n{'#'*60}")
    print(f"PROCESSING: {dataset_name.upper()} | {severity.upper()} | FOLD {fold_idx}")
    print(f"{'#'*60}")
    
    mnar_path = os.path.join(DATA_DIR, f'{dataset_name}_mnar_{severity}.csv')
    
    if not os.path.exists(mnar_path):
        print(f"Error: File not found: {mnar_path}")
        return
        
    print(f"Input: {mnar_path}")
    
    # 0. Load Split Indices
    splits_path = os.path.join(DATA_DIR, f'cv_splits_{dataset_name}.json')
    if not os.path.exists(splits_path):
        splits_path = os.path.join(os.path.dirname(mnar_path), f'cv_splits_{dataset_name}.json')
        
    with open(splits_path, 'r') as f:
        cv_splits = json.load(f)
    fold_key = f'fold_{fold_idx + 1}'
    train_idx = np.array(cv_splits[fold_key]['train'])
    val_idx   = np.array(cv_splits[fold_key]['val'])
    
    # Load MNAR Data
    df_mnar = pd.read_csv(mnar_path)
    
    # 1. Load Optimized Hyperparameters
    optuna_dir = os.path.join(DATA_DIR, 'final', 'optuna_results_final')
    
    # Optuna Path Logic: Current Fold -> Fold 0 -> Root Fallback
    optuna_file = os.path.join(optuna_dir, f'fold_{fold_idx}', f'{dataset_name}_{severity}_optuna_mice_results.csv')
    if not os.path.exists(optuna_file):
        # Fallback to fold_0 (Methodology: Optimize on fold 0 only)
        optuna_file = os.path.join(optuna_dir, 'fold_0', f'{dataset_name}_{severity}_optuna_mice_results.csv')
        
    if not os.path.exists(optuna_file):
        # Fallback to root for backward compatibility
        optuna_file = os.path.join(optuna_dir, f'{dataset_name}_{severity}_optuna_mice_results.csv')
    if os.path.exists(optuna_file):
        df_optuna = pd.read_csv(optuna_file)
        best_params = df_optuna.iloc[0]
        max_iter = int(best_params['max_iter'])
        print(f"  Loaded Optimized max_iter: {max_iter}")
    else:
        max_iter = 10
        print(f"  Warning: Optuna file not found. Using default max_iter: {max_iter}")
    
    # Encode 'Sex'
    if 'Sex' in df_mnar.columns:
        df_mnar['Sex'] = df_mnar['Sex'].map({'F': 0, 'M': 1, 'Female': 0, 'Male': 1})
    
    # Identify target and ID columns
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
    
    # 3. Fix Scaler: Fit on train rows only, transform all (NaN-safe)
    X_raw = feature_df_mnar.values.astype(np.float32)
    x_min = np.nanmin(X_raw[train_idx], axis=0)
    x_max = np.nanmax(X_raw[train_idx], axis=0)
    x_scale = x_max - x_min
    x_scale[x_scale < 1e-6] = 1.0
    X_scaled = (X_raw - x_min) / x_scale
    
    # Combine for Imputation
    X_input = X_scaled
    
    df_targets_orig = df_mnar[target_cols].copy()
    df_ids = df_mnar[cols_to_drop].copy() if cols_to_drop else None
    
    import warnings
    output_dir = os.path.join(DATA_DIR, 'final', 'imputation_results_final', f'fold_{fold_idx}')
    os.makedirs(output_dir, exist_ok=True)
    
    for m in range(1, M_IMPUTATIONS + 1):
        # Use random_state=m for each of the 5 independent imputations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=max_iter,
                random_state=m,
                sample_posterior=True
            )
            # Correct: fit on train only, transform all
            imputer.fit(X_input[train_idx])
            imputed_array_all = imputer.transform(X_input)
        
        # Extract and Inverse Transform (manual)
        X_imputed_clipped = np.clip(imputed_array_all, 0, 1)
        X_imputed = X_imputed_clipped * x_scale + x_min
        imputed_df = pd.DataFrame(X_imputed, columns=feature_cols)
        
        # Add back targets and IDs
        for col in target_cols:
            imputed_df[col] = df_targets_orig[col].values
        
        if df_ids is not None:
            for col in id_cols:
                if col in df_ids.columns:
                    imputed_df.insert(0, col, df_ids[col].values)
        
        # Save each as a separate file
        output_filename = f'{dataset_name}_{severity}_mice_imputed_m{m}.csv'
        output_path = os.path.join(output_dir, output_filename)
        imputed_df.to_csv(output_path, index=False)
        print(f"  [m={m}] Saved: {output_filename}")

    print(f"\n{'#'*60}")
    print(f"MICE Pipeline Complete for {dataset_name} {severity} Fold {fold_idx}!")
    print(f"{'#'*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MICE imputation for a specific fold.")
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4], help="CV fold index (0-4)")
    parser.add_argument("--dataset", type=str, required=True, choices=["metabric", "mimic"], help="Dataset name")
    parser.add_argument("--scenario", type=str, required=True, choices=["light", "moderate", "severe"], help="Imputation scenario")
    args = parser.parse_args()
    
    run_mice_imputation(args.dataset, args.scenario, args.fold)
