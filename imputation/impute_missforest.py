"""
missForest Imputation Script
============================
Implements 'missForest' (Stekhoven & Buhlmann, 2012).
Uses GPU (cuml) for MIMIC and CPU (sklearn) for METABRIC.
Strategy: Single Imputation (Standard for ML Benchmarks)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import argparse
import json
import warnings

# Configuration
DATA_DIR = os.environ.get('SVAE_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets'))
# RESULTS_DIR will be set dynamically per fold

# Default missForest settings
DEFAULT_MAX_ITER = 10
DEFAULT_N_TREES = 100

def run_missforest_imputation(dataset_name, severity, fold_idx):
    print(f"Starting missForest Imputation - Fold {fold_idx}...")
    
    # Try to check if cuml is available for GPU
    try:
        import cuml
        HAS_CUML = True
    except ImportError:
        HAS_CUML = False
        print("Warning: cuml (RAPIDS) not found. GPU acceleration disabled.")

    print(f"\n{'#'*60}")
    print(f"PROCESSING: {dataset_name.upper()} | {severity.upper()} | FOLD {fold_idx}")
    print(f"{'#'*60}")
    
    mnar_path = os.path.join(DATA_DIR, f'{dataset_name}_mnar_{severity}.csv')
    
    # Update output path structure to include fold index
    output_dir = os.path.join(DATA_DIR, 'final', 'imputation_results_final', f'fold_{fold_idx}')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{dataset_name}_{severity}_missforest_imputed.csv')
    
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
    
    # Load Data
    df_mnar = pd.read_csv(mnar_path)
    
    # 1. Load Optimized Hyperparameters
    optuna_dir = os.path.join(DATA_DIR, 'final', 'optuna_results_final')
    
    # Optuna Path Logic: Current Fold -> Fold 0 -> Root Fallback
    optuna_file = os.path.join(optuna_dir, f'fold_{fold_idx}', f'{dataset_name}_{severity}_optuna_missforest_results.csv')
    if not os.path.exists(optuna_file):
        # Fallback to fold_0 (Methodology: Optimize on fold 0 only)
        optuna_file = os.path.join(optuna_dir, 'fold_0', f'{dataset_name}_{severity}_optuna_missforest_results.csv')
        
    if not os.path.exists(optuna_file):
        # Fallback to root for backward compatibility
        optuna_file = os.path.join(optuna_dir, f'{dataset_name}_{severity}_optuna_missforest_results.csv')
    if os.path.exists(optuna_file):
        df_optuna = pd.read_csv(optuna_file)
        best_params = df_optuna.iloc[0]
        n_estimators = int(best_params['n_estimators'])
        max_depth = best_params['max_depth']
        try:
            max_depth = int(float(max_depth)) if not pd.isna(max_depth) else None
        except:
            max_depth = None
        max_iter = int(best_params['max_iter'])
        print(f"  Loaded Optimized: Trees={n_estimators}, Depth={max_depth}, Iter={max_iter}")
    else:
        n_estimators = DEFAULT_N_TREES
        max_depth = None
        max_iter = DEFAULT_MAX_ITER
        print(f"  Warning: Optuna file not found. Using default parameters.")
    
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
            
    # Use GPU (cuml) for MIMIC if available
    use_gpu = (dataset_name == 'mimic' and HAS_CUML)
    
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import RandomForestRegressor

    if use_gpu:
        print(f"  Running GPU missForest (Trees={n_estimators}, Depth={max_depth})...")
        from cuml.ensemble import RandomForestRegressor as cumlRF
        estimator = cumlRF(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth else 16,
            random_state=42
        )
    else:
        print(f"  Running CPU missForest (Trees={n_estimators}, Depth={max_depth})...")
        estimator = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            random_state=42
        )
        # Correct: fit on train only, transform all
        imputer.fit(X_input[train_idx])
        imputed_array_all = imputer.transform(X_input)
    
    # Inverse transform (manual) and build DF
    X_imputed = imputed_array_all * x_scale + x_min
    imputed_df = pd.DataFrame(X_imputed, columns=feature_cols)
    
    for col in target_cols:
        imputed_df[col] = df_mnar[target_cols][col].values
    
    if id_cols:
        for col in id_cols:
            if col in df_mnar.columns:
                imputed_df.insert(0, col, df_mnar[col].values)
    
    imputed_df.to_csv(output_path, index=False)
    print(f"  Saved: {os.path.basename(output_path)}")

    print(f"\n{'='*60}")
    print(f"missForest Pipeline Complete for {dataset_name} {severity} Fold {fold_idx}!")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run missForest imputation for a specific fold.")
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4], help="CV fold index (0-4)")
    parser.add_argument("--dataset", type=str, required=True, choices=["metabric", "mimic"], help="Dataset name")
    parser.add_argument("--scenario", type=str, required=True, choices=["light", "moderate", "severe"], help="Imputation scenario")
    args = parser.parse_args()
    
    run_missforest_imputation(args.dataset, args.scenario, args.fold)
