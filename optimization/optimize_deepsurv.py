"""
DeepSurv Hyperparameter Optimization with Optuna
=================================================
Tunes DeepSurv (CoxPH) hyperparameters using pycox for each
dataset x scenario x imputation method combination.
Objective: Maximize C-index on validation fold.
Library: pycox 0.3.0 (as per manuscript Table D.2)
"""

import os
import argparse
import json
import optuna
import numpy as np
import pandas as pd
import warnings
import torchtuples as tt
from pycox.models import CoxPH
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = os.environ.get('SVAE_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets'))
DATASETS = ['metabric', 'mimic']
SCENARIOS = ['light', 'moderate', 'severe']
METHODS = ['survival_vae', 'standard_vae', 'gain', 'mida', 'mice', 'missforest']
SPLIT_FILES = {'metabric': 'cv_splits_metabric.json', 'mimic': 'cv_splits_mimic.json'}
N_TRIALS = 50


def load_splits(dataset_name):
    with open(os.path.join(DATA_DIR, SPLIT_FILES[dataset_name])) as f:
        return json.load(f)


def format_y_sksurv(times, events):
    return np.array([(bool(e), t) for e, t in zip(events, times)],
                    dtype=[('Status', '?'), ('Survival_in_days', '<f8')])


def load_imputed_data(dataset, scenario, method, fold_idx):
    imp_dir = os.path.join(DATA_DIR, 'final', 'imputation_results_final', f'fold_{fold_idx}')
    if method == 'mice':
        dfs = []
        for m in range(1, 6):
            p = os.path.join(imp_dir, f'{dataset}_{scenario}_mice_imputed_m{m}.csv')
            if os.path.exists(p):
                dfs.append(pd.read_csv(p))
        if not dfs:
            return None, None, None, None
        # Average numeric columns only, preserve non-numeric from first imputation
        df_concat = pd.concat(dfs, axis=0).reset_index(drop=True)
        numeric_cols = df_concat.select_dtypes(include=[np.number]).columns
        df_numeric = df_concat[numeric_cols].groupby(level=0).mean().reset_index(drop=True)
        df_non_numeric = dfs[0].select_dtypes(exclude=[np.number]).reset_index(drop=True)
        df = pd.concat([df_non_numeric, df_numeric], axis=1)
    else:
        p = os.path.join(imp_dir, f'{dataset}_{scenario}_{method}_imputed.csv')
        if not os.path.exists(p):
            return None, None, None, None
        df = pd.read_csv(p)

    t_col = 'Survival_in_days' if 'Survival_in_days' in df.columns else ('duration' if 'duration' in df.columns else 'Time')
    e_col = 'Status' if 'Status' in df.columns else ('event' if 'event' in df.columns else 'Event')
    id_cols = ['hadm_id', 'subject_id', 'stay_id', 'icustay_id', 'patient_id', 'admittime', 'dischtime']
    cols_to_drop = [c for c in id_cols if c in df.columns]
    feature_df = df.drop(columns=[t_col, e_col] + cols_to_drop).select_dtypes(include=[np.number])

    X = feature_df.values.astype(np.float32)
    T = np.clip(df[t_col].values.astype(np.float64), 1e-5, None)
    E = df[e_col].values.astype(np.float32)
    return X, T, E, feature_df.columns.tolist()


def prepare_fold_data(dataset, scenario, method, fold_idx):
    splits = load_splits(dataset)
    fold_key = f'fold_{fold_idx + 1}'
    train_idx = np.array(splits[fold_key]['train'])
    val_idx = np.array(splits[fold_key]['val'])

    X, T, E, _ = load_imputed_data(dataset, scenario, method, fold_idx)
    if X is None:
        return None

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = X.copy().astype(np.float64)
    X_train_raw = np.nan_to_num(X[train_idx].astype(np.float64), nan=0.0)
    X_scaled[train_idx] = scaler.fit_transform(X_train_raw)
    X_scaled[val_idx] = np.nan_to_num(
        scaler.transform(X[val_idx].astype(np.float64)), nan=0.0)

    return (X_scaled[train_idx].astype(np.float32), X_scaled[val_idx].astype(np.float32),
            T[train_idx].astype(np.float32), E[train_idx].astype(np.float32),
            T[val_idx].astype(np.float32), E[val_idx].astype(np.float32))


def objective(trial, data):
    X_train, X_val, T_train, E_train, T_val, E_val = data
    num_layers = trial.suggest_int('num_layers', 1, 3)
    hidden_size = trial.suggest_int('hidden_size', 16, 128)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = 512

    try:
        net = tt.practical.MLPVanilla(
            X_train.shape[1], [hidden_size] * num_layers, 1,
            batch_norm=True, dropout=dropout
        )
        model = CoxPH(net, tt.optim.Adam(lr=lr))
        model.fit(
            X_train, (T_train, E_train),
            batch_size=batch_size, epochs=100,
            callbacks=[tt.callbacks.EarlyStopping(patience=10)],
            val_data=(X_val, (T_val, E_val)),
            verbose=False
        )
        _ = model.compute_baseline_hazards()
        risk_val = model.predict(X_val).flatten()
        y_val_sk = format_y_sksurv(T_val, E_val.astype(bool))
        return concordance_index_censored(
            y_val_sk['Status'], y_val_sk['Survival_in_days'], risk_val
        )[0]
    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        return 0.5


def save_callback(study, trial, output_path):
    df = study.trials_dataframe()
    if df.empty:
        return
    cols = ['number'] + [c for c in df.columns if c.startswith('params_')] + ['value']
    df = df[cols].sort_values('value', ascending=False)
    df.columns = [c.replace('params_', '') if c.startswith('params_') else c for c in df.columns]
    df.rename(columns={'number': 'trial', 'value': 'c_index'}).to_csv(output_path, index=False)


def run_study(dataset, scenario, method, fold_idx):
    print(f"  Optimizing DeepSurv: {dataset} | {scenario} | {method} | fold {fold_idx}")
    data = prepare_fold_data(dataset, scenario, method, fold_idx)
    if data is None:
        print("  Skipping - imputed file not found")
        return

    output_dir = os.path.join(DATA_DIR, 'final', 'survival_results', 'optuna', f'fold_{fold_idx}')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{dataset}_{scenario}_{method}_optuna_deepsurv_results.csv')

    study = optuna.create_study(
        study_name=f"deepsurv_{dataset}_{scenario}_{method}_fold{fold_idx}",
        storage=f"sqlite:///optuna_deepsurv_fold{fold_idx}.db",
        direction='maximize', load_if_exists=True
    )
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    trials_to_run = max(0, N_TRIALS - completed)

    if trials_to_run > 0:
        print(f"    {completed} done, {trials_to_run} remaining...")
        study.optimize(lambda t: objective(t, data), n_trials=trials_to_run,
                       callbacks=[lambda s, t: save_callback(s, t, output_path)])
    else:
        print(f"    Already complete ({completed} trials). Skipping.")

    save_callback(study, None, output_path)
    print(f"    Best C-index: {study.best_trial.value:.4f} | Params: {study.best_trial.params}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, choices=[0,1,2,3,4])
    parser.add_argument('--dataset', type=str, default='all', choices=DATASETS + ['all'])
    parser.add_argument('--scenario', type=str, default='all', choices=SCENARIOS + ['all'])
    parser.add_argument('--method', type=str, default='all', choices=METHODS + ['all'])
    parser.add_argument('--trials', type=int, help='Override N_TRIALS')
    args = parser.parse_args()

    if args.trials is not None:
        N_TRIALS = args.trials

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    scenarios = SCENARIOS if args.scenario == 'all' else [args.scenario]
    methods = METHODS if args.method == 'all' else [args.method]
    total = len(datasets) * len(scenarios) * len(methods)
    current = 0

    print(f"\nDeepSurv Optuna | Fold {args.fold} | {total} combinations")
    for d in datasets:
        for s in scenarios:
            for m in methods:
                current += 1
                print(f"\n[{current}/{total}]")
                run_study(d, s, m, args.fold)

    print(f"\nDeepSurv Optuna complete for fold {args.fold}")
