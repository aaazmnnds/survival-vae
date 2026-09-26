"""
RSF Hyperparameter Optimization with Optuna
============================================
Tunes Random Survival Forest hyperparameters for each
dataset x scenario x imputation method combination.
Objective: Maximize C-index on validation fold.
"""

import os
import argparse
import json
import optuna
import numpy as np
import pandas as pd
import warnings
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

def find_dir(possibilities, default_name):
    for p in possibilities:
        if os.path.exists(p):
            return p
    return default_name

DATA_DIR = find_dir(POSSIBLE_DATA_DIRS, 'datasets')

DATASETS = ['metabric', 'mimic']
SCENARIOS = ['light', 'moderate', 'severe']
METHODS = ['survival_vae', 'standard_vae', 'gain', 'mida', 'mice', 'missforest']
SPLIT_FILES = {
    'metabric': 'cv_splits_metabric.json',
    'mimic': 'cv_splits_mimic.json'
}
N_TRIALS = 50


def load_splits(dataset_name):
    path = os.path.join(DATA_DIR, SPLIT_FILES[dataset_name])
    with open(path, 'r') as f:
        return json.load(f)


def format_y_sksurv(times, events):
    return np.array(
        [(bool(e), t) for e, t in zip(events, times)],
        dtype=[('Status', '?'), ('Survival_in_days', '<f8')]
    )


def load_imputed_data(dataset, scenario, method, fold_idx, suffix_str=''):
    """Load imputed CSV and return X, T, E arrays."""
    imp_dir = os.path.join(
        DATA_DIR, 'final', f'imputation_results_final{suffix_str}', f'fold_{fold_idx}'
    )

    if method == 'mice':
        # Average across 5 imputations
        dfs = []
        for m in range(1, 6):
            path = os.path.join(
                imp_dir, f'{dataset}_{scenario}_mice_imputed_m{m}.csv'
            )
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
        if not dfs:
            return None, None, None, None
        # Average numeric columns only, preserve non-numeric from first imputation
        df_concat = pd.concat(dfs, axis=0).reset_index(drop=True)
        numeric_cols = df_concat.select_dtypes(include=[np.number]).columns
        df_numeric = df_concat[numeric_cols].groupby(level=0).mean().reset_index(drop=True)
        df_non_numeric = dfs[0].select_dtypes(exclude=[np.number]).reset_index(drop=True)
        df = pd.concat([df_non_numeric, df_numeric], axis=1)
    else:
        path = os.path.join(
            imp_dir, f'{dataset}_{scenario}_{method}_imputed.csv'
        )
        if not os.path.exists(path):
            return None, None, None, None
        df = pd.read_csv(path)

    # Identify target and ID columns
    if 'Survival_in_days' in df.columns:
        t_col, e_col = 'Survival_in_days', 'Status'
    elif 'duration' in df.columns:
        t_col, e_col = 'duration', 'event'
    else:
        t_col, e_col = 'Time', 'Event'

    id_cols = ['hadm_id', 'subject_id', 'stay_id', 'icustay_id',
               'patient_id', 'admittime', 'dischtime']
    cols_to_drop = [c for c in id_cols if c in df.columns]

    feature_df = df.drop(columns=[t_col, e_col] + cols_to_drop).select_dtypes(include=[np.number])

    # Encode Sex if present
    if 'Sex' in df.columns and df['Sex'].dtype == object:
        feature_df['Sex'] = df['Sex'].map({'F': 0, 'M': 1, 'Female': 0, 'Male': 1})

    X = feature_df.values.astype(np.float32)
    T = df[t_col].values.astype(np.float64)
    E = df[e_col].values.astype(bool)

    # Clip non-positive times
    T = np.clip(T, 1e-5, None)

    return X, T, E, feature_df.columns.tolist()


def prepare_fold_data(dataset, scenario, method, fold_idx, suffix_str=''):
    """Load data and split into train/val using CV splits."""
    splits = load_splits(dataset)
    fold_key = f'fold_{fold_idx + 1}'
    train_idx = np.array(splits[fold_key]['train'])
    val_idx = np.array(splits[fold_key]['val'])

    X, T, E, feat_cols = load_imputed_data(dataset, scenario, method, fold_idx, suffix_str=suffix_str)
    if X is None:
        return None

    # NaN-safe scaler fit on train only
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = X.copy().astype(np.float64)
    X_train_raw = np.nan_to_num(X[train_idx].astype(np.float64), nan=0.0)
    X_scaled[train_idx] = scaler.fit_transform(X_train_raw)
    X_scaled[val_idx] = np.nan_to_num(
        scaler.transform(X[val_idx].astype(np.float64)), nan=0.0)

    X_train = X_scaled[train_idx]
    X_val = X_scaled[val_idx]
    y_train = format_y_sksurv(T[train_idx], E[train_idx])
    y_val = format_y_sksurv(T[val_idx], E[val_idx])

    # Subsample for Optuna efficiency on large datasets (MIMIC only)
    # Final training uses all data - this only affects hyperparameter search
    if dataset == 'mimic' and len(X_train) > 3000:
        rng = np.random.RandomState(42)
        sub = rng.choice(len(X_train), 3000, replace=False)
        X_train = X_train[sub]
        y_train = y_train[sub]

    return X_train, X_val, y_train, y_val


def objective(trial, data):
    X_train, X_val, y_train, y_val = data

    n_estimators = trial.suggest_int('n_estimators', 50, 150)
    min_samples_split = trial.suggest_int('min_samples_split', 5, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 20)

    try:
        rsf = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        rsf.fit(X_train, y_train)
        risk = rsf.predict(X_val)
        c_index = concordance_index_censored(
            y_val['Status'], y_val['Survival_in_days'], risk
        )[0]
        return c_index
    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        return 0.5


def save_callback(study, trial, output_path):
    df = study.trials_dataframe()
    if df.empty:
        return
    cols = ['number'] + [c for c in df.columns if c.startswith('params_')] + ['value']
    df = df[cols].sort_values('value', ascending=False)
    df.columns = [c.replace('params_', '') if c.startswith('params_') else c
                  for c in df.columns]
    df = df.rename(columns={'number': 'trial', 'value': 'c_index'})
    df.to_csv(output_path, index=False)


def run_study(dataset, scenario, method, fold_idx, suffix_str=''):
    print(f"  Optimizing RSF: {dataset} | {scenario} | {method} | fold {fold_idx}")

    data = prepare_fold_data(dataset, scenario, method, fold_idx, suffix_str=suffix_str)
    if data is None:
        print(f"  Skipping - imputed file not found")
        return

    output_dir = os.path.join(
        DATA_DIR, 'final', f'survival_results{suffix_str}', 'optuna', f'fold_{fold_idx}'
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f'{dataset}_{scenario}_{method}_optuna_rsf_results.csv'
    )

    suffix_db = f"_{suffix_str}" if suffix_str else ""
    study_name = f"rsf_{dataset}_{scenario}_{method}_fold{fold_idx}{suffix_db}"
    storage_name = f"sqlite:///optuna_rsf_fold{fold_idx}{suffix_db}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        load_if_exists=True
    )

    completed = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    trials_to_run = max(0, N_TRIALS - completed)

    if trials_to_run > 0:
        print(f"    {completed} done, {trials_to_run} remaining...")
        study.optimize(
            lambda t: objective(t, data),
            n_trials=trials_to_run,
            callbacks=[lambda s, t: save_callback(s, t, output_path)]
        )
    else:
        print(f"    Already complete ({completed} trials). Skipping.")

    # Save final best result
    save_callback(study, None, output_path)
    best = study.best_trial
    print(f"    Best C-index: {best.value:.4f} | Params: {best.params}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, choices=[0,1,2,3,4])
    parser.add_argument('--dataset', type=str, default='all',
                        choices=DATASETS + ['all'])
    parser.add_argument('--scenario', type=str, default='all',
                        choices=SCENARIOS + ['all'])
    parser.add_argument('--method', type=str, default='all',
                        choices=METHODS + ['all'])
    parser.add_argument('--trials', type=int, help='Override N_TRIALS')
    parser.add_argument('--suffix', type=str, default='', help='Optional suffix for input/output dirs')
    args = parser.parse_args()

    if args.trials is not None:
        N_TRIALS = args.trials

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    scenarios = SCENARIOS if args.scenario == 'all' else [args.scenario]
    methods = METHODS if args.method == 'all' else [args.method]

    total = len(datasets) * len(scenarios) * len(methods)
    current = 0

    print(f"\nRSF Optuna | Fold {args.fold} | {total} combinations")
    print('=' * 60)

    for d in datasets:
        for s in scenarios:
            for m in methods:
                current += 1
                print(f"\n[{current}/{total}]")
                run_study(d, s, m, args.fold, suffix_str=args.suffix)

    print(f"\n{'='*60}")
    print(f"RSF Optuna complete for fold {args.fold}")
