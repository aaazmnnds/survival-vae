import os
import argparse
import json
import time
import optuna
import numpy as np
import pandas as pd
import warnings
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler

# Try to import cuml for GPU acceleration
try:
    from cuml.ensemble import RandomForestRegressor as cuRF
    HAS_CUML = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor as cuRF
    HAS_CUML = False
    print("Warning: cuML not found. Falling back to sklearn (CPU).")

# Silence warnings
warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = os.environ.get('SVAE_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets'))

def find_truth_file(dataset_name):
    if dataset_name == 'metabric':
        filenames = ['metabric_processed.csv', 'metabric.csv']
    else:
        filenames = ['mimic_sepsis_highdim.csv', 'mimic_processed.csv', 'mimic.csv']
    
    search_dirs = [DATA_DIR, os.path.join(DATA_DIR, 'final'), '.']
    for d in search_dirs:
        for f in filenames:
            full_path = os.path.join(d, f)
            if os.path.exists(full_path):
                return full_path
    return None

SPLIT_FILES = {'metabric': 'cv_splits_metabric.json', 'mimic': 'cv_splits_mimic.json'}

def objective(trial, dataset_data):
    try:
        n_estimators = trial.suggest_int("n_estimators", 50, 100) # Slightly reduced for speed
        max_depth = trial.suggest_categorical("max_depth", [5, 10, 20])
        max_iter = trial.suggest_int("max_iter", 5, 10)
        
        # Unpack pre-loaded data
        (X_train, X_val_input, X_val_truth_scaled, mask_val_art, n_feat) = dataset_data
        
        # IterativeImputer with cuML RandomForest
        rf_params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": 42}
        estimator = cuRF(**rf_params) if HAS_CUML else cuRF(**rf_params, n_jobs=8)
            
        imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=42, verbose=0)
        imputer.fit(X_train)
        X_val_imputed_all = imputer.transform(X_val_input)
        
        X_val_imputed = X_val_imputed_all[:, :n_feat]
        idx = mask_val_art == 1
        rmse = np.sqrt(np.mean((X_val_imputed[idx] - X_val_truth_scaled[idx])**2))
        
        return rmse if not np.isnan(rmse) else 1.0
    except Exception as e:
        print(f"Trial failed: {e}")
        return 1.0

def save_callback(study, trial, output_path):
    df = study.trials_dataframe()
    if not df.empty:
        cols = ['number'] + [c for c in df.columns if c.startswith('params_')] + ['value']
        df = df[cols].sort_values('value')
        df.columns = [c.replace('params_', '') if c.startswith('params_') else c for c in df.columns]
        df.rename(columns={'number': 'trial', 'value': 'rmse_norm'}).to_csv(output_path, index=False)
        print(f" [Checkpoint] Saved trial {trial.number} to {output_path}")

def prepare_data(dataset_name, scenario, fold_idx):
    """Load and preprocess data once."""
    mnar_path = os.path.join(DATA_DIR, f'{dataset_name}_mnar_{scenario}.csv')
    truth_path = find_truth_file(dataset_name)
    mask_path = os.path.join(DATA_DIR, f'{dataset_name}_mask_{scenario}.csv')
    
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
    X_mnar = feature_df_mnar.values.astype(np.float64)
    X_truth = df_truth[feature_cols].values.astype(np.float64)
    mask_artificial = df_mask[feature_cols].values.astype(np.float64)
    T_E_mnar = df_mnar[target_cols].values.astype(np.float64)
    
    split_file = os.path.join(DATA_DIR, SPLIT_FILES[dataset_name])
    with open(split_file, 'r') as f: splits = json.load(f)
    fold_key = f"fold_{fold_idx+1}"
    train_idx, val_idx = splits[fold_key]['train'], splits[fold_key]['val']
    
    scaler_x = MinMaxScaler()
    X_mnar_scaled = np.zeros_like(X_mnar, dtype=float)
    X_mnar_scaled[train_idx] = scaler_x.fit_transform(X_mnar[train_idx])
    X_mnar_scaled[val_idx] = scaler_x.transform(X_mnar[val_idx])
    X_truth_scaled = scaler_x.transform(X_truth)
    
    scaler_te = MinMaxScaler()
    TE_mnar_scaled = np.zeros_like(T_E_mnar, dtype=float)
    TE_mnar_scaled[train_idx] = scaler_te.fit_transform(T_E_mnar[train_idx])
    TE_mnar_scaled[val_idx] = scaler_te.transform(T_E_mnar[val_idx])
    
    X_input = np.concatenate([X_mnar_scaled, TE_mnar_scaled], axis=1)
    return (X_input[train_idx], X_input[val_idx], X_truth_scaled[val_idx], mask_artificial[val_idx], len(feature_cols))

def run_study(dataset, scenario, n_trials=100, fold_idx=0):
    os.makedirs(f"final/optuna_results_final/fold_{fold_idx}", exist_ok=True)
    output_path = f"final/optuna_results_final/fold_{fold_idx}/{dataset}_{scenario}_optuna_missforest_results.csv"
    print(f"Loading data for missForest {dataset}-{scenario}...")
    dataset_data = prepare_data(dataset, scenario, fold_idx)
    
    # Use SQLite for persistence to allow skipping/resuming

    # Use SQLite for persistence to allow skipping/resuming
    study_name = f"missforest_{dataset}_{scenario}_fold{fold_idx}"
    storage_name = f"sqlite:///optuna_missforest_fold{fold_idx}.db"
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--dataset", type=str, choices=['metabric', 'mimic'])
    parser.add_argument("--scenario", type=str, choices=['light', 'moderate', 'severe'])
    parser.add_argument("--fold", type=int, default=0, choices=[0,1,2,3,4], help="CV fold index (0-4)")
    args = parser.parse_args()
    
    datasets = [args.dataset] if args.dataset else ['metabric', 'mimic']
    scenarios = [args.scenario] if args.scenario else ['light', 'moderate', 'severe']
    
    for d in datasets:
        for s in scenarios:
            print(f"Optimizing missForest for {d}-{s} - Fold {args.fold}...")
            run_study(d, s, n_trials=args.trials, fold_idx=args.fold)
