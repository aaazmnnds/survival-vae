"""
Survival Model Training and Evaluation (5-Fold CV)
===================================================
Trains RSF, XGBoost, DeepSurv, DeepHit on imputed data per fold.
Loads best hyperparameters from Optuna CSVs.
Metrics: C-index, IBS, iAUC (t25/t50/t75), AUC(t) at t_med.
Fully aligned with manuscript specifications.

Key fixes vs previous script:
- pycox for DeepSurv and DeepHit (no custom PyTorch bugs)
- Scaler fit on train_idx only (no leakage)
- iAUC quartiles from uncensored event times only
- t_med from uncensored event times only
- Loads Optuna hyperparameters
- All 6 imputation methods including survival_vae and mice
- Uses explicit val_idx from CV splits
"""

import os
import argparse
import json
import warnings
import time
import numpy as np
import pandas as pd
from scipy import stats
from datetime import timedelta

import torch
import torchtuples as tt
from pycox.models import CoxPH, DeepHitSingle

from sklearn.preprocessing import MinMaxScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score,
    cumulative_dynamic_auc
)
from sksurv.functions import StepFunction
import xgboost as xgb

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
POSSIBLE_DATA_DIRS = [
    './datasets', 'datasets', '../datasets',
    '/home/azman/VAE/Survival-VAE_study',
    '/Users/azmannads/VAE/Survival-VAE_study',
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# DATA LOADING
# ============================================================================
def load_splits(dataset_name):
    with open(os.path.join(DATA_DIR, SPLIT_FILES[dataset_name])) as f:
        return json.load(f)


def format_y_sksurv(times, events):
    return np.array(
        [(bool(e), t) for e, t in zip(events, times)],
        dtype=[('Status', '?'), ('Survival_in_days', '<f8')]
    )


def load_imputed_data(dataset, scenario, method, fold_idx):
    """Load imputed CSV. MICE: average across 5 files."""
    imp_dir = os.path.join(
        DATA_DIR, 'final', 'imputation_results_final', f'fold_{fold_idx}'
    )

    if method == 'mice':
        dfs = []
        for m in range(1, 6):
            p = os.path.join(imp_dir, f'{dataset}_{scenario}_mice_imputed_m{m}.csv')
            if os.path.exists(p):
                dfs.append(pd.read_csv(p))
        if not dfs:
            return None
        # Average numeric columns only, preserve non-numeric from first imputation
        df_concat = pd.concat(dfs, axis=0).reset_index(drop=True)
        numeric_cols = df_concat.select_dtypes(include=[np.number]).columns
        df_numeric = df_concat[numeric_cols].groupby(level=0).mean().reset_index(drop=True)
        df_non_numeric = dfs[0].select_dtypes(exclude=[np.number]).reset_index(drop=True)
        df = pd.concat([df_non_numeric, df_numeric], axis=1)
    else:
        p = os.path.join(imp_dir, f'{dataset}_{scenario}_{method}_imputed.csv')
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p)

    return df


def prepare_arrays(df, train_idx, val_idx, test_idx):
    """Extract X, T, E and scale features. Returns train/val/test splits."""
    # Identify columns
    if 'Survival_in_days' in df.columns:
        t_col, e_col = 'Survival_in_days', 'Status'
    elif 'duration' in df.columns:
        t_col, e_col = 'duration', 'event'
    else:
        t_col, e_col = 'Time', 'Event'

    id_cols = ['hadm_id', 'subject_id', 'stay_id', 'icustay_id',
               'patient_id', 'admittime', 'dischtime']
    cols_to_drop = [c for c in id_cols if c in df.columns]

    feature_df = df.drop(
        columns=[t_col, e_col] + cols_to_drop
    ).select_dtypes(include=[np.number])

    X = feature_df.values.astype(np.float32)
    T = np.clip(df[t_col].values.astype(np.float64), 1e-5, None)
    E = df[e_col].values.astype(bool)

    # Scaler fit on train_idx only - no leakage
    # NaN-safe: fill NaNs before scaling (imputed data should have none,
    # but guards against MICE averaging edge cases or zero-variance features)
    scaler = MinMaxScaler()
    X_scaled = X.copy().astype(np.float64)
    X_train_raw = np.nan_to_num(X[train_idx].astype(np.float64), nan=0.0)
    X_scaled[train_idx] = scaler.fit_transform(X_train_raw)
    X_scaled[val_idx] = np.nan_to_num(
        scaler.transform(X[val_idx].astype(np.float64)), nan=0.0)
    X_scaled[test_idx] = np.nan_to_num(
        scaler.transform(X[test_idx].astype(np.float64)), nan=0.0)

    X_train = X_scaled[train_idx].astype(np.float32)
    X_val = X_scaled[val_idx].astype(np.float32)
    X_test = X_scaled[test_idx].astype(np.float32)

    T_train, E_train = T[train_idx], E[train_idx]
    T_val, E_val = T[val_idx], E[val_idx]
    T_test, E_test = T[test_idx], E[test_idx]

    y_train = format_y_sksurv(T_train, E_train)
    y_val = format_y_sksurv(T_val, E_val)
    y_test = format_y_sksurv(T_test, E_test)

    return (X_train, X_val, X_test,
            T_train, E_train, T_val, E_val, T_test, E_test,
            y_train, y_val, y_test)


def get_eval_times(T_train, E_train):
    """
    Compute evaluation time points from UNCENSORED training event times.
    t25, t50, t75 for iAUC.
    t_med (median) for AUC(t).
    Per manuscript: quartiles and median of UNCENSORED patients only.
    """
    event_times = T_train[E_train]
    if len(event_times) == 0:
        event_times = T_train  # fallback
    t25 = np.percentile(event_times, 25)
    t50 = np.percentile(event_times, 50)
    t75 = np.percentile(event_times, 75)
    t_med = np.median(event_times)
    return np.array([t25, t50, t75]), t_med


def filter_test_for_ibs(X_test, y_test, T_test, y_train):
    """
    Filter test samples with T > max training time.
    Required for IBS and time-dependent AUC.
    """
    max_train_t = y_train['Survival_in_days'].max()
    valid = y_test['Survival_in_days'] < max_train_t
    if valid.sum() == 0:
        return None, None, None
    return X_test[valid], y_test[valid], T_test[valid]


# ============================================================================
# OPTUNA PARAM LOADING
# ============================================================================
def load_best_params(dataset, scenario, method, fold_idx, model_name):
    """Load best hyperparameters from Optuna CSV."""
    name_map = {
        'xgb': 'xgboost',
        'rsf': 'rsf',
        'deepsurv': 'deepsurv',
        'deephit': 'deephit'
    }
    csv_name = name_map.get(model_name, model_name)
    optuna_dir = os.path.join(
        DATA_DIR, 'final', 'survival_results', 'optuna', 'fold_0'
    )
    csv_path = os.path.join(
        optuna_dir,
        f'{dataset}_{scenario}_{method}_optuna_{csv_name}_results.csv'
    )
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


# ============================================================================
# BRESLOW ESTIMATOR (for XGBoost survival functions)
# ============================================================================
class BreslowEstimator:
    def __init__(self):
        self.cum_baseline_hazard = None
        self.times = None
        self.offset = 0.0

    def fit(self, risk_scores_raw, times, events):
        self.offset = np.max(risk_scores_raw)
        risk_scores = np.exp(risk_scores_raw - self.offset)
        order = np.argsort(times)
        risk_scores = risk_scores[order]
        times = times[order]
        events = events[order]
        unique_times, unique_indices = np.unique(times, return_index=True)
        risk_sum_reverse = np.cumsum(risk_scores[::-1])[::-1]
        risk_sets = risk_sum_reverse[unique_indices]
        cum_hazard, current_cum = [], 0.0
        for i in range(len(unique_times)):
            start = unique_indices[i]
            end = unique_indices[i+1] if i+1 < len(unique_indices) else len(times)
            d_i = np.sum(events[start:end])
            hazard_i = d_i / risk_sets[i] if risk_sets[i] > 0 else 0
            current_cum += hazard_i
            cum_hazard.append(current_cum)
        self.times = unique_times
        self.cum_baseline_hazard = np.array(cum_hazard)
        return self

    def get_survival_function(self, risk_scores_raw_new):
        risk_new = np.exp(risk_scores_raw_new - self.offset)
        baseline_surv = np.exp(-self.cum_baseline_hazard)
        x_step = np.concatenate(([0.0], self.times))
        funcs = []
        for r in risk_new:
            y_step = np.concatenate(([1.0], np.power(baseline_surv, r)))
            funcs.append(StepFunction(x_step, y_step))
        return funcs


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================
def train_eval_rsf(data, params, eval_times, t_med):
    X_train, X_val, X_test, T_train, E_train, T_val, E_val, T_test, E_test, y_train, y_val, y_test = data

    # Load params or use defaults
    n_estimators = int(params.get('n_estimators', 100)) if params else 100
    min_samples_split = int(params.get('min_samples_split', 10)) if params else 10
    min_samples_leaf = int(params.get('min_samples_leaf', 15)) if params else 15

    # Train on train+val combined for final evaluation
    X_trainval = np.vstack([X_train, X_val])
    T_trainval = np.concatenate([T_train, T_val])
    E_trainval = np.concatenate([E_train, E_val])
    y_trainval = format_y_sksurv(T_trainval, E_trainval)

    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42, n_jobs=16
    )
    rsf.fit(X_trainval, y_trainval)

    # C-index on full test set
    risk = rsf.predict(X_test)
    c_index = concordance_index_censored(
        y_test['Status'], y_test['Survival_in_days'], risk
    )[0]

    # Filter test for IBS/AUC
    X_test_f, y_test_f, T_test_f = filter_test_for_ibs(X_test, y_test, T_test, y_trainval)
    if X_test_f is None:
        return c_index, np.nan, np.nan, np.nan

    # Clip eval_times to valid range for IBS/AUC
    max_test_t = y_test_f['Survival_in_days'].max()
    eval_times_clipped = eval_times[eval_times < max_test_t]
    if len(eval_times_clipped) == 0:
        return c_index, np.nan, np.nan, np.nan

    # IBS
    surv_funcs = rsf.predict_survival_function(X_test_f)
    surv_probs = np.row_stack([fn(eval_times_clipped) for fn in surv_funcs])
    ibs = integrated_brier_score(y_trainval, y_test_f, surv_probs, eval_times_clipped)

    # iAUC (mean over t25/t50/t75)
    risk_f = rsf.predict(X_test_f)
    auc_vals = cumulative_dynamic_auc(y_trainval, y_test_f, risk_f, eval_times_clipped)[0]
    iauc = np.mean(auc_vals)

    # AUC at t_med
    t_med_arr = np.array([t_med])
    if t_med <= y_test_f['Survival_in_days'].max():
        auc_tmed = cumulative_dynamic_auc(y_trainval, y_test_f, risk_f, t_med_arr)[0][0]
    else:
        auc_tmed = np.nan

    return c_index, ibs, iauc, auc_tmed


def train_eval_xgboost(data, params, eval_times, t_med):
    X_train, X_val, X_test, T_train, E_train, T_val, E_val, T_test, E_test, y_train, y_val, y_test = data

    n_estimators = int(params.get('n_estimators', 100)) if params else 100
    learning_rate = float(params.get('learning_rate', 0.05)) if params else 0.05
    max_depth = int(params.get('max_depth', 3)) if params else 3
    subsample = float(params.get('subsample', 1.0)) if params else 1.0
    colsample_bytree = float(params.get('colsample_bytree', 1.0)) if params else 1.0

    X_trainval = np.vstack([X_train, X_val])
    T_trainval = np.concatenate([T_train, T_val])
    E_trainval = np.concatenate([E_train, E_val])
    y_trainval = format_y_sksurv(T_trainval, E_trainval)

    dtrain = xgb.DMatrix(X_trainval)
    y_xgb = np.where(E_trainval, T_trainval, -T_trainval)
    dtrain.set_label(y_xgb)

    xgb_params = {
        'objective': 'survival:cox',
        'verbosity': 0,
        'eta': learning_rate,
        'max_depth': max_depth,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'seed': 42,
        'tree_method': 'hist',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'min_child_weight': 5,
        'gamma': 1
    }
    bst = xgb.train(xgb_params, dtrain, num_boost_round=n_estimators)

    risk = bst.predict(xgb.DMatrix(X_test), output_margin=True)
    c_index = concordance_index_censored(
        y_test['Status'], y_test['Survival_in_days'], risk
    )[0]

    X_test_f, y_test_f, T_test_f = filter_test_for_ibs(X_test, y_test, T_test, y_trainval)
    if X_test_f is None:
        return c_index, np.nan, np.nan, np.nan

    risk_f = bst.predict(xgb.DMatrix(X_test_f), output_margin=True)
    risk_train = bst.predict(dtrain, output_margin=True)
    breslow = BreslowEstimator().fit(risk_train, T_trainval, E_trainval)
    surv_funcs = breslow.get_survival_function(risk_f)

    # Clip eval_times to valid range for IBS/AUC
    max_test_t = y_test_f['Survival_in_days'].max()
    eval_times_clipped = eval_times[eval_times < max_test_t]
    if len(eval_times_clipped) == 0:
        return c_index, np.nan, np.nan, np.nan

    surv_probs = np.row_stack([fn(eval_times_clipped) for fn in surv_funcs])
    ibs = integrated_brier_score(y_trainval, y_test_f, surv_probs, eval_times_clipped)

    auc_vals = cumulative_dynamic_auc(y_trainval, y_test_f, risk_f, eval_times_clipped)[0]
    iauc = np.mean(auc_vals)

    t_med_arr = np.array([t_med])
    if t_med <= y_test_f['Survival_in_days'].max():
        auc_tmed = cumulative_dynamic_auc(y_trainval, y_test_f, risk_f, t_med_arr)[0][0]
    else:
        auc_tmed = np.nan

    return c_index, ibs, iauc, auc_tmed


def train_eval_deepsurv(data, params, eval_times, t_med):
    X_train, X_val, X_test, T_train, E_train, T_val, E_val, T_test, E_test, y_train, y_val, y_test = data

    num_layers = int(params.get('num_layers', 2)) if params else 2
    hidden_size = int(params.get('hidden_size', 32)) if params else 32
    dropout = float(params.get('dropout', 0.3)) if params else 0.3
    lr = float(params.get('lr', 0.001)) if params else 0.001
    batch_size = int(params.get('batch_size', 64)) if params else 64

    X_trainval = np.vstack([X_train, X_val])
    T_trainval = np.concatenate([T_train, T_val]).astype(np.float32)
    E_trainval = np.concatenate([E_train, E_val]).astype(np.float32)
    y_trainval = format_y_sksurv(T_trainval, E_trainval)

    try:
        net = tt.practical.MLPVanilla(
            X_trainval.shape[1], [hidden_size] * num_layers, 1,
            batch_norm=True, dropout=dropout
        )
        model = CoxPH(net, tt.optim.Adam(lr=lr))
        # Train on train only with val for early stopping
        model.fit(
            X_train.astype(np.float32),
            (T_train.astype(np.float32), E_train.astype(np.float32)),
            batch_size=batch_size, epochs=100,
            callbacks=[tt.callbacks.EarlyStopping(patience=10), tt.callbacks.TerminateOnNaN()],
            val_data=(X_val.astype(np.float32),
                      (T_val.astype(np.float32), E_val.astype(np.float32))),
            verbose=False
        )
        _ = model.compute_baseline_hazards()

        # C-index on full test
        risk = model.predict(X_test.astype(np.float32)).flatten()
        c_index = concordance_index_censored(
            y_test['Status'], y_test['Survival_in_days'], risk
        )[0]

        X_test_f, y_test_f, T_test_f = filter_test_for_ibs(X_test, y_test, T_test, y_train)
        if X_test_f is None:
            return c_index, np.nan, np.nan, np.nan

        # Survival functions via Breslow
        risk_f = model.predict(X_test_f.astype(np.float32)).flatten()
        risk_train = model.predict(X_train.astype(np.float32)).flatten()
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        cox_breslow = CoxPHSurvivalAnalysis(ties='breslow')
        cox_breslow.fit(risk_train.reshape(-1, 1), y_train)
        surv_funcs = cox_breslow.predict_survival_function(risk_f.reshape(-1, 1))

        # Clip eval_times to valid range for IBS/AUC
        max_test_t = y_test_f['Survival_in_days'].max()
        eval_times_clipped = eval_times[eval_times < max_test_t]
        if len(eval_times_clipped) == 0:
            return c_index, np.nan, np.nan, np.nan

        surv_probs = np.row_stack([fn(eval_times_clipped) for fn in surv_funcs])
        ibs = integrated_brier_score(y_train, y_test_f, surv_probs, eval_times_clipped)

        auc_vals = cumulative_dynamic_auc(y_train, y_test_f, risk_f, eval_times_clipped)[0]
        iauc = np.mean(auc_vals)

        t_med_arr = np.array([t_med])
        if t_med <= y_test_f['Survival_in_days'].max():
            auc_tmed = cumulative_dynamic_auc(y_train, y_test_f, risk_f, t_med_arr)[0][0]
        else:
            auc_tmed = np.nan

        return c_index, ibs, iauc, auc_tmed

    except Exception as e:
        print(f"    DeepSurv failed: {e}")
        return np.nan, np.nan, np.nan, np.nan


def train_eval_deephit(data, params, eval_times, t_med):
    X_train, X_val, X_test, T_train, E_train, T_val, E_val, T_test, E_test, y_train, y_val, y_test = data

    num_layers = int(params.get('num_layers', 2)) if params else 2
    hidden_size = int(params.get('hidden_size', 64)) if params else 64
    dropout = float(params.get('dropout', 0.3)) if params else 0.3
    lr = float(params.get('lr', 0.001)) if params else 0.001
    num_durations = int(params.get('num_durations', 10)) if params else 10
    alpha = float(params.get('alpha', 0.2)) if params else 0.2
    batch_size = int(params.get('batch_size', 64)) if params else 64

    try:
        labtrans = DeepHitSingle.label_transform(num_durations)
        y_train_dt = labtrans.fit_transform(
            T_train.astype(np.float32), E_train.astype(np.float32)
        )
        y_val_dt = labtrans.transform(
            T_val.astype(np.float32), E_val.astype(np.float32)
        )

        net = tt.practical.MLPVanilla(
            X_train.shape[1], [hidden_size] * num_layers,
            labtrans.out_features, batch_norm=True, dropout=dropout
        )
        model = DeepHitSingle(
            net, tt.optim.Adam(lr=lr),
            alpha=alpha, sigma=0.1,
            duration_index=labtrans.cuts
        )
        model.fit(
            X_train.astype(np.float32), y_train_dt,
            batch_size=batch_size, epochs=100,
            callbacks=[tt.callbacks.EarlyStopping(patience=10), tt.callbacks.TerminateOnNaN()],
            val_data=(X_val.astype(np.float32), y_val_dt),
            verbose=False
        )

        # C-index on full test
        surv_test = model.predict_surv_df(X_test.astype(np.float32))
        risk = -surv_test.values.sum(0)
        c_index = concordance_index_censored(
            y_test['Status'], y_test['Survival_in_days'], risk
        )[0]

        X_test_f, y_test_f, T_test_f = filter_test_for_ibs(X_test, y_test, T_test, y_train)
        if X_test_f is None:
            return c_index, np.nan, np.nan, np.nan

        surv_f = model.predict_surv_df(X_test_f.astype(np.float32))

        # IBS - interpolate survival at eval_times
        surv_times = surv_f.index.values
        surv_matrix = surv_f.values.T  # shape: (n_test, n_times)

        def interp_surv(surv_row, query_times):
            return np.array([
                np.interp(t, surv_times, surv_row, left=1.0, right=0.0)
                for t in query_times
            ])

        # Clip eval_times to valid range for IBS/AUC
        max_test_t = y_test_f['Survival_in_days'].max()
        eval_times_clipped = eval_times[eval_times < max_test_t]
        if len(eval_times_clipped) == 0:
            return c_index, np.nan, np.nan, np.nan

        surv_probs = np.row_stack([
            interp_surv(surv_matrix[i], eval_times_clipped)
            for i in range(len(surv_matrix))
        ])
        ibs = integrated_brier_score(y_train, y_test_f, surv_probs, eval_times_clipped)

        # iAUC - risk = -sum(S)
        risk_f = -surv_f.values.sum(0)
        auc_vals = cumulative_dynamic_auc(y_train, y_test_f, risk_f, eval_times_clipped)[0]
        iauc = np.mean(auc_vals)

        t_med_arr = np.array([t_med])
        if t_med <= y_test_f['Survival_in_days'].max():
            auc_tmed = cumulative_dynamic_auc(y_train, y_test_f, risk_f, t_med_arr)[0][0]
        else:
            auc_tmed = np.nan

        return c_index, ibs, iauc, auc_tmed

    except Exception as e:
        print(f"    DeepHit failed: {e}")
        return np.nan, np.nan, np.nan, np.nan


# ============================================================================
# SUMMARY GENERATION
# ============================================================================
def calculate_95_ci(values):
    arr = np.array(values)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = np.mean(arr)
    sd = np.std(arr)
    sem = stats.sem(arr)
    ci = sem * 2.776  # t_{0.025, 4} for n=5 folds
    return mean, sd, mean - ci, mean + ci


def generate_summary(results_df, output_path):
    rows = []
    for (ds, sc, meth, mod), g in results_df.groupby(
        ['Dataset', 'Scenario', 'Method', 'Model']
    ):
        c_m, c_sd, c_lo, c_hi = calculate_95_ci(g['C_Index'])
        ibs_m, ibs_sd, _, _ = calculate_95_ci(g['IBS'])
        ia_m, ia_sd, _, _ = calculate_95_ci(g['iAUC'])
        q_m, q_sd, q_lo, q_hi = calculate_95_ci(g['AUC_tmed'])
        rows.append({
            'Dataset': ds, 'Scenario': sc, 'Method': meth, 'Model': mod,
            'C_Index_Mean': round(c_m, 4), 'C_Index_SD': round(c_sd, 4),
            'IBS_Mean': round(ibs_m, 4), 'IBS_SD': round(ibs_sd, 4),
            'iAUC_Mean': round(ia_m, 4), 'iAUC_SD': round(ia_sd, 4),
            'AUC_tmed_Mean': round(q_m, 4), 'AUC_tmed_SD': round(q_sd, 4),
            'AUC_tmed_95CI': f"{q_m:.4f} ({q_lo:.4f}-{q_hi:.4f})"
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[Done] Summary saved to {output_path}")


# ============================================================================
# MAIN EVALUATION
# ============================================================================
def run_evaluation(fold_idx, dataset_filter='all', scenario_filter='all',
                   method_filter='all'):
    results = []
    out_dir = os.path.join(DATA_DIR, 'final', 'survival_results', f'fold_{fold_idx}')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'results_survival_fold.csv')
    sum_file = os.path.join(out_dir, 'results_survival_summary.csv')

    datasets = DATASETS if dataset_filter == 'all' else [dataset_filter]
    scenarios = SCENARIOS if scenario_filter == 'all' else [scenario_filter]
    methods = METHODS if method_filter == 'all' else [method_filter]

    splits_cache = {}

    for dataset in datasets:
        if dataset not in splits_cache:
            splits_cache[dataset] = load_splits(dataset)
        splits = splits_cache[dataset]

        fold_key = f'fold_{fold_idx + 1}'
        train_idx = np.array(splits[fold_key]['train'])
        val_idx = np.array(splits[fold_key]['val'])
        test_idx = np.array(splits[fold_key]['test'])

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()} | Fold {fold_idx}")
        print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

        for scenario in scenarios:
            for method in methods:
                print(f"\n  Scenario: {scenario} | Method: {method}")

                df = load_imputed_data(dataset, scenario, method, fold_idx)
                if df is None:
                    print(f"    Skipping - imputed file not found")
                    continue

                try:
                    arrays = prepare_arrays(df, train_idx, val_idx, test_idx)
                except Exception as e:
                    print(f"    Data prep failed: {e}")
                    continue

                X_train = arrays[0]
                T_train, E_train = arrays[3], arrays[4]
                y_train = arrays[9]

                # Evaluation times from UNCENSORED training event times
                eval_times, t_med = get_eval_times(T_train, E_train)
                print(f"    t25={eval_times[0]:.1f} t50={eval_times[1]:.1f} "
                      f"t75={eval_times[2]:.1f} t_med={t_med:.1f}")

                model_fns = {
                    'rsf': train_eval_rsf,
                    'xgb': train_eval_xgboost,
                    'deepsurv': train_eval_deepsurv,
                    'deephit': train_eval_deephit,
                }

                for model_name, train_fn in model_fns.items():
                    t_start = time.time()
                    print(f"    {model_name.upper()}...", end=' ', flush=True)

                    params = load_best_params(
                        dataset, scenario, method, fold_idx, model_name
                    )
                    if params is None:
                        print(f"(no Optuna params - using defaults)")
                    else:
                        print(f"(Optuna params loaded)", end=' ')

                    try:
                        c, ibs, iauc, auc_tmed = train_fn(
                            arrays, params, eval_times, t_med
                        )
                        elapsed = time.time() - t_start
                        print(f"C={c:.4f} IBS={ibs:.4f} "
                              f"iAUC={iauc:.4f} AUC_tmed={auc_tmed:.4f} "
                              f"({elapsed:.1f}s)")
                        results.append({
                            'Dataset': dataset.upper(),
                            'Scenario': scenario,
                            'Method': method,
                            'Model': model_name,
                            'Fold': fold_idx,
                            'C_Index': c,
                            'IBS': ibs,
                            'iAUC': iauc,
                            'AUC_tmed': auc_tmed
                        })
                    except Exception as e:
                        print(f"FAILED: {e}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Save per-fold results
    if results:
        df_results = pd.DataFrame(results)

        # Incremental save - merge with existing
        if os.path.exists(out_file):
            df_existing = pd.read_csv(out_file)
            df_results = pd.concat([df_existing, df_results]).drop_duplicates(
                subset=['Dataset', 'Scenario', 'Method', 'Model', 'Fold'],
                keep='last'
            ).reset_index(drop=True)

        df_results.to_csv(out_file, index=False)
        print(f"\n[Done] Results saved to {out_file}")
        generate_summary(df_results, sum_file)
    else:
        print("\nNo results generated.")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and evaluate survival models on imputed data.'
    )
    parser.add_argument('--fold', type=str, required=True, 
                        help='Fold index (0-4) or "all" to run all 5 folds.')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=DATASETS + ['all'])
    parser.add_argument('--scenario', type=str, default='all',
                        choices=SCENARIOS + ['all'])
    parser.add_argument('--method', type=str, default='all',
                        choices=METHODS + ['all'])
    args = parser.parse_args()

    if args.fold.lower() == 'all':
        folds = [0, 1, 2, 3, 4]
    else:
        folds = [int(args.fold)]

    for f in folds:
        print(f"\n" + "="*60)
        print(f"Survival Model Evaluation | Fold {f}")
        print(f"Data dir: {DATA_DIR}")
        print(f"Device: {device}")
        print("="*60)

        run_evaluation(
            fold_idx=f,
            dataset_filter=args.dataset,
            scenario_filter=args.scenario,
            method_filter=args.method
        )
