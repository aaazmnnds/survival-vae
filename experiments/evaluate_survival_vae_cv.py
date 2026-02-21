"""
Robust 5-Fold CV Evaluation for Survival-VAE
============================================
Performs strictly separated training/imputation/evaluation per fold
to avoid data leakage from global scaling or imputation.

Pipeline:
1. Load Data (Truth, MNAR, Mask) and Splits.
2. Loop Folds:
    a. Split into Train/Val/Test.
    b. Fit Scaler on Train, Transform Val/Test.
    c. Train Survival-VAE on Train/Val (using optimized params).
    d. Impute Test set (using model trained on Train).
    e. Impute Train set (for downstream ML).
    f. Level 1 Eval: RMSE (Test set masked values).
    g. Level 2 Eval: Train RSF/XGB on Imputed Train -> Eval on Imputed Test.
3. Aggregate and Report.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, integrated_brier_score, cumulative_dynamic_auc
from sksurv.functions import StepFunction
import xgboost as xgb
import json
import os
import warnings
import time
import random
import argparse

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For MPS (Mac)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass # Older torch versions might not have this
    os.environ['PYTHONHASHSEED'] = str(seed)


warnings.filterwarnings('ignore')

# Configuration
# --- Path Resolution ---
def find_dir(name, search_paths):
    for path in search_paths:
        target = os.path.join(path, name)
        if os.path.exists(target):
            return target
        # Also check if the path itself is the directory if it ends with the name
        if os.path.basename(path.rstrip('/')) == name:
            return path
    return None

POSSIBLE_ROOTS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
]

DATA_DIR = find_dir('datasets', POSSIBLE_ROOTS) or os.path.join(POSSIBLE_ROOTS[0], 'datasets')
OPTUNA_DIR = find_dir('optuna_results', POSSIBLE_ROOTS) or os.path.join(POSSIBLE_ROOTS[0], 'optuna_results')
RESULTS_DIR = os.path.join(os.path.dirname(DATA_DIR), 'results_cv')

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

print(f"Data Directory: {DATA_DIR}")
print(f"Optuna Results: {OPTUNA_DIR}")
print(f"Results Output: {RESULTS_DIR}")

DATASETS = ['metabric', 'mimic']
SCENARIOS = ['light', 'moderate', 'severe']
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==============================================================================
# Model Definitions
# ==============================================================================

class BreslowEstimator:
    """Estimates baseline cumulative hazard and survival functions with stability offset"""
    def __init__(self):
        self.cum_baseline_hazard = None
        self.times = None
        self.offset = 0.0

    def fit(self, risk_scores_raw, times, events):
        """Fit baseline hazard from training data with stability offset"""
        self.offset = np.max(risk_scores_raw)
        risk_scores = np.exp(risk_scores_raw - self.offset) # Stability fix
        
        order = np.argsort(times)
        risk_scores = risk_scores[order]
        times = times[order]
        events = events[order]

        unique_times, unique_indices = np.unique(times, return_index=True)
        cum_hazard = []
        current_cum = 0.0
        
        # Calculate risk sets (reverse cumsum of exp(risk))
        risk_sum_reverse = np.cumsum(risk_scores[::-1])[::-1]
        risk_sets = risk_sum_reverse[unique_indices]
        
        for i, t in enumerate(unique_times):
            start = unique_indices[i]
            end = unique_indices[i+1] if i + 1 < len(unique_indices) else len(times)
            d_i = np.sum(events[start:end])
            risk_set_i = risk_sets[i]
            
            hazard_i = d_i / risk_set_i if risk_set_i > 0 else 0
            current_cum += hazard_i
            cum_hazard.append(current_cum)
            
        self.times = unique_times
        self.cum_baseline_hazard = np.array(cum_hazard)
        return self

    def get_survival_function(self, risk_scores_raw_new):
        """Returns list of StepFunction objects for survival probabilities (using offset)"""
        if self.cum_baseline_hazard is None:
            raise ValueError("Must call fit() first")
            
        risk_new = np.exp(risk_scores_raw_new - self.offset) # Stability fix
        
        baseline_surv = np.exp(-self.cum_baseline_hazard)
        x_step = np.concatenate(([0.0], self.times))
        
        funcs = []
        for r in risk_new:
            surv_probs = np.power(baseline_surv, r)
            y_step = np.concatenate(([1.0], surv_probs))
            funcs.append(StepFunction(x_step, y_step))
            
        return funcs

# ==============================================================================
# 1. MODEL DEFINITIONS (Copied for reproducibility)
# ==============================================================================

class SurvivalNetwork(nn.Module):
    def __init__(self, latent_dim):
        super(SurvivalNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1) # Log Hazard
        )
        
    def forward(self, z):
        return self.net(z)

class DeepSurv(nn.Module):
    """DeepSurv: Cox Proportional Hazards Deep Neural Network"""
    def __init__(self, input_dim, hidden_dims=[32, 32], dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class DeepHit(nn.Module):
    """DeepHit: Deep Learning for Survival Analysis with Competing Risks"""
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3, num_durations=10):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h
        self.shared = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, num_durations)
        
    def forward(self, x):
        return self.output(self.shared(x))

class SurvivalVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(SurvivalVAE, self).__init__()
        self.input_dim = input_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 64), # Data + Mask
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid() # Scale [0,1]
        )
        
        # Survival
        self.surv_net = SurvivalNetwork(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, m):
        if x.dim() == 1: x = x.unsqueeze(0)
        if m.dim() == 1: m = m.unsqueeze(0)
            
        inputs = torch.cat([x, m], dim=1)
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        risk_score = self.surv_net(z)
        return recon_x, mu, logvar, risk_score

def cox_ph_loss(log_h, events):
    """
    Cox Partial Likelihood Loss (Robust)
    Matches train_all_models_cv1.py
    """
    # Sort by time descending (Ref script does this explicitly)
    # We must assume inputs are (risk_scores, events) ALREADY sorted by time descending.
    
    # Risk
    risk = torch.exp(log_h)
    
    # Cumulative Risk (Risk Set Sum)
    # Since t is decreasing: t_0 >= t_1 >= ... >= t_N.
    # RiskSetSum[i] = sum(exp(h[i...N])). Wait, ref script uses:
    # risk_cumsum = torch.cumsum(risk, 0) where risk is sorted by -time.
    # If sorted by -time, index 0 is largest time.
    # subjects surviving at time t_i are those with t_j >= t_i.
    # If sorted descending, those are indices j <= i.
    # So cumsum(risk) is correct.
    
    risk_cumsum = torch.cumsum(risk, dim=0)
    log_risk_cumsum = torch.log(risk_cumsum + 1e-8)
    
    loss_vector = (log_h - log_risk_cumsum) * events
    
    num_events = torch.sum(events)
    if num_events < 1e-6:
        return torch.tensor(0.0, device=log_h.device, requires_grad=True)
        
    return -torch.sum(loss_vector) / num_events

def deephit_loss(logits, y_time, y_event):
    """
    Robust DeepHit Loss (NLL for Events + Log-Survival for Censored)
    Matches train_all_models_cv1.py
    """
    pmf = F.softmax(logits, dim=1)
    
    # 1. Event Loss (maximize PMF at observed bin)
    # Using cross_entropy on logits is numerically stable
    ce_loss = F.cross_entropy(logits, y_time.long(), reduction='none')
    
    # 2. Censored Loss (maximize Survival at observed bin)
    # S(t) = 1 - CDF(t)
    cdf = torch.cumsum(pmf, dim=1)
    # Stability fix: clamp to handle precision issues where cdf might slightly exceed 1.0
    surv_prob = torch.clamp(1.0 - cdf, min=1e-8)
    
    # Gather survival prob at the observed bin index
    idx = y_time.long().unsqueeze(1)
    surv_at_bin = surv_prob.gather(1, idx).squeeze()
    censored_loss = -torch.log(surv_at_bin) # Epsilon already in clamp
    
    # Combine based on event indicator
    loss = (ce_loss * y_event) + (censored_loss * (1 - y_event))
    return loss.mean()

# ==============================================================================
# 2. UTILITIES
# ==============================================================================

def load_data(dataset, scenario):
    """Load Truth, MNAR, Mask, and Splits"""
    # Filenames
    if dataset == 'metabric':
        truth_file = 'metabric_processed.csv'
    else:
        truth_file = 'final/mimic_sepsis_highdim.csv'
        
    mnar_file = f'{dataset}_mnar_{scenario}.csv'
    mask_file = f'{dataset}_mask_{scenario}.csv'
    split_file = f'cv_splits_{dataset}.json'
    
    # Paths
    truth_path = os.path.join(DATA_DIR, truth_file)
    mnar_path = os.path.join(DATA_DIR, mnar_file)
    mask_path = os.path.join(DATA_DIR, mask_file)
    split_path = os.path.join(DATA_DIR, split_file)
    
    # Load
    df_truth = pd.read_csv(truth_path)
    df_mnar = pd.read_csv(mnar_path)
    df_mask = pd.read_csv(mask_path)
    
    with open(split_path, 'r') as f:
        splits = json.load(f)
        
    return df_truth, df_mnar, df_mask, splits

def preprocess_df(df):
    """Basic preprocessing: Drop IDs, Encode Sex, Identify Survival Cols"""
    # 1. Encode Sex
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'F': 0, 'M': 1, 'Female': 0, 'Male': 1})
        
    # 2. Identify Survival Cols
    surv_cols = ['duration', 'event'] # Standardized
    if 'Time' in df.columns: surv_cols = ['Time', 'Event']
    elif 'Survival_in_days' in df.columns: surv_cols = ['Survival_in_days', 'Status']
    
    t_col, e_col = surv_cols
    
    # 3. Drop IDs
    id_cols = ['hadm_id', 'subject_id', 'stay_id', 'icustay_id', 'patient_id']
    cols_to_drop = [c for c in id_cols if c in df.columns]
    
    # Feature columns (Numeric only)
    feature_cols = [c for c in df.columns if c not in surv_cols + cols_to_drop and pd.api.types.is_numeric_dtype(df[c])]
    
    return df, feature_cols, t_col, e_col

def format_y_sksurv(t, e):
    return np.array([(bool(ei), ti) for ei, ti in zip(e, t)], 
                    dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

# ==============================================================================
# 3. TRAINING & EVALUATION
# ==============================================================================

def train_vae(X_train, M_train, T_train, E_train, input_dim, params):
    """Train Survival-VAE on one fold"""
    # Hyperparams
    beta = params.get('beta', 1.0)
    gamma = params.get('gamma', 1.0)
    latent_dim = int(params.get('latent_dim', 10))
    lr = params.get('lr', 1e-3)
    epochs = 100 # Fixed max epochs
    
    print(f"    [VAE] Hyperparams: beta={beta:.6f}, gamma={gamma:.4f}, latent={latent_dim}, lr={lr:.5f}")
    
    model = SurvivalVAE(input_dim, latent_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Prepare Data
    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    M_t = torch.tensor(M_train, dtype=torch.float32).to(DEVICE)
    T_t = torch.tensor(T_train, dtype=torch.float32).to(DEVICE)
    E_t = torch.tensor(E_train, dtype=torch.float32).to(DEVICE)
    
    dataset = TensorDataset(X_t, M_t, T_t, E_t)
    
    # Seeding for DataLoader
    g = torch.Generator()
    g.manual_seed(42)
    
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=g)
    
    model.train()
    for epoch in range(epochs):
        epoch_recon, epoch_kl, epoch_surv = 0, 0, 0
        for bx, bm, bt, be in loader:
            optimizer.zero_grad()
            
            # Sort by time desc for Cox Loss
            sort_idx = torch.argsort(bt, descending=True)
            bx, bm, bt, be = bx[sort_idx], bm[sort_idx], bt[sort_idx], be[sort_idx]
            
            if be.dim() == 1: be = be.unsqueeze(1)
            
            recon, mu, logvar, risk = model(bx, bm)
            
            # 1. Recon Loss (MSE on Observed) - NORMALIZED to per-patient scale
            # (Total MSE / num_observed) * input_dim
            mse = F.mse_loss(recon, bx, reduction='none')
            num_observed = bm.sum()
            if num_observed > 0:
                avg_pixel_mse = (mse * bm).sum() / num_observed
                recon_loss = avg_pixel_mse * model.input_dim
            else:
                recon_loss = torch.tensor(0.0, device=DEVICE)
            
            # 2. KL Loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            
            # 3. Surv Loss (Cox)
            surv_loss = cox_ph_loss(risk, be)
            
            loss = recon_loss + beta * kl_loss + gamma * surv_loss
            loss.backward()
            
            # Stability: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_recon += recon_loss.item()
            epoch_kl += (beta * kl_loss).item()
            epoch_surv += (gamma * surv_loss).item()
            
        if (epoch + 1) % 50 == 0:
            avg_r = epoch_recon / len(loader)
            avg_k = epoch_kl / len(loader)
            avg_s = epoch_surv / len(loader)
            print(f"    [VAE] Epoch {epoch+1}/{epochs} | Recon: {avg_r:.4f} | KL: {avg_k:.4f} | Surv: {avg_s:.4f}")
            
    return model

def impute_data(model, X, M):
    """Impute data using trained model"""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        M_t = torch.tensor(M, dtype=torch.float32).to(DEVICE)
        
        recon, _, _, _ = model(X_t, M_t)
        
        # Impute: Observed * Mask + Recon * (1-Mask)
        # Note: Input X already has 0 or mean at missing locations.
        # Ideally, we want to replace missing spots with Recon.
        # M is 1 for Observed, 0 for Missing.
        
        imputed = X_t * M_t + recon * (1 - M_t)
        
    return imputed.cpu().numpy()

def evaluate_rmse(X_imputed, X_truth, Mask_Simulation):
    """Calculate RMSE only on simulated missing values"""
    # Mask_Simulation: 1=Observed, 0=Missing/Removed
    # We want to check error where Mask_Simulation == 0 AND Truth is available (not NaN)
    
    # Ensure inputs are numpy arrays
    X_imputed = np.array(X_imputed)
    X_truth = np.array(X_truth)
    Mask_Simulation = np.array(Mask_Simulation)
    
    # Valid indices: Simulation says missing (we imputed it) AND Ground Truth exists
    missing_indices = (Mask_Simulation == 0) & (~np.isnan(X_truth))
    
    if not np.any(missing_indices):
        return 0.0
    
    mse = mean_squared_error(X_truth[missing_indices], X_imputed[missing_indices])
    return np.sqrt(mse)

def evaluate_mae(X_imputed, X_truth, Mask_Simulation):
    """Calculate MAE only on simulated missing values"""
    X_imputed = np.array(X_imputed)
    X_truth = np.array(X_truth)
    Mask_Simulation = np.array(Mask_Simulation)
    
    missing_indices = (Mask_Simulation == 0) & (~np.isnan(X_truth))
    
    if not np.any(missing_indices):
        return 0.0
    
    mae = np.mean(np.abs(X_truth[missing_indices] - X_imputed[missing_indices]))
    return mae

def run_ml_evaluation(X_train, y_train, X_test, y_test, opt_params):
    """Train RSF/XGB and evaluate"""
    # opt_params is already the best_params dict from the main loop
    best_params = opt_params

    # --- Scaling ---
    # Fail-safe: Detect if inputs are already bad
    if not np.isfinite(X_train).all():
        print(f"  Warning: X_train contains Non-finite values before Scaling! NaNs: {np.isnan(X_train).sum()}, Infs: {np.isinf(X_train).sum()}")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Switch to MinMaxScaler to match baseline protocol
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Final check
    if not np.isfinite(X_train).all():
        print("  Warning: X_train contains Non-finite values AFTER Scaling!")
        X_train = np.nan_to_num(X_train, nan=0.0)
    
    results = {}
    
    # Format y for scikit-survival
    # y = [Event, Time]
    y_train_struc = format_y_sksurv(y_train[:, 1], y_train[:, 0]) 
    y_test_struc = format_y_sksurv(y_test[:, 1], y_test[:, 0])
    
    # --- Define Evaluation Times (Matches Reference) ---
    # Use quartiles of training set for IBS integration
    # IBS requires at least 2 time points
    q25 = np.percentile(y_train[:, 1], 25)
    q50 = np.percentile(y_train[:, 1], 50)
    q75 = np.percentile(y_train[:, 1], 75)
    times = np.array([q25, q50, q75])
    
    # Clip to max training time (minus epsilon) to avoid "time must be smaller than largest observed time point"
    max_train_time = y_train[:, 1].max()
    times = np.clip(times, a_min=1e-5, a_max=max_train_time - 1e-5)
    times = np.unique(times) # Ensure unique

    # For IBS: Filter test set to only include times < max training time
    # This is required by sksurv for censoring distribution estimation (IPCW)
    valid_test_idx = y_test[:, 1] < max_train_time
    if valid_test_idx.sum() > 0:
        y_test_ibs = y_test[valid_test_idx]
        X_test_ibs = X_test[valid_test_idx]
        y_test_struc_ibs = format_y_sksurv(y_test_ibs[:, 1], y_test_ibs[:, 0])
    else:
        y_test_ibs = None
        X_test_ibs = None
        y_test_struc_ibs = None
    
    # --- RSF ---
    try:
        rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=42)
        rsf.fit(X_train, y_train_struc)
        c_index = rsf.score(X_test, y_test_struc)
        results['RSF_C_Index'] = c_index
        
        # IBS
        try:
            if y_test_struc_ibs is not None:
                surv_funcs = rsf.predict_survival_function(X_test_ibs)
                
                # Evaluate StepFunctions at specific times
                surv_probs = np.row_stack([fn(times) for fn in surv_funcs])
                
                ibs = integrated_brier_score(y_train_struc, y_test_struc_ibs, surv_probs, times)
                results['RSF_IBS'] = ibs
            else:
                results['RSF_IBS'] = np.nan
            
            # Integrated AUC & Median AUC (50th percentile)
            # Use 'risk' scores (higher = shorter survival = higher risk)
            if y_test_struc_ibs is not None:
                risk_scores_ibs = rsf.predict(X_test_ibs)
                auc_scores, mean_auc = cumulative_dynamic_auc(y_train_struc, y_test_struc_ibs, risk_scores_ibs, times)
                results['RSF_IntegratedAUC'] = mean_auc
                results['RSF_AUC_Median'] = auc_scores[1]
            else:
                results['RSF_IntegratedAUC'] = np.nan
                results['RSF_AUC_Median'] = np.nan
            
        except Exception as e:
            # print(f"  Error in RSF IBS/AUC: {e}")
            results['RSF_IBS'] = np.nan
            results['RSF_IntegratedAUC'] = np.nan
            results['RSF_AUC_Median'] = np.nan
    except Exception as e:
        print(f"  Error in RSF: {e}")
        print(f"  X_train has NaNs: {np.isnan(X_train).any()}")
        results['RSF_C_Index'] = np.nan
        
    # --- XGB ---
    try:
        # XGB Labels: +Time if Event, -Time if Censored
        y_xgb_train = np.where(y_train[:, 0], y_train[:, 1], -y_train[:, 1])
        y_xgb_test = np.where(y_test[:, 0], y_test[:, 1], -y_test[:, 1])
        
        dtrain = xgb.DMatrix(X_train, label=y_xgb_train)
        dtest = xgb.DMatrix(X_test, label=y_xgb_test)
        
        # STABLE XGBoost Parameters from Reference
        # eta=0.05, max_depth=3 prevents overflows
        params = {
            'objective': 'survival:cox',
            'tree_method': 'hist',
            'verbosity': 0,
            'eta': 0.05,
            'max_depth': 3,
            'random_state': 42
        }
        bst = xgb.train(params, dtrain, num_boost_round=100)
        
        risk = bst.predict(dtest, output_margin=True)
        c_index_xgb = concordance_index_censored(y_test_struc['Status'], y_test_struc['Survival_in_days'], risk)[0]
        results['XGB_C_Index'] = c_index_xgb
        
        # XGB IBS using Breslow
        try:
            breslow_xgb = BreslowEstimator()
            # Fit on Training Risk (Use Log-Risk/Margin for stability in Breslow)
            log_risk_train = bst.predict(dtrain, output_margin=True)
            breslow_xgb.fit(log_risk_train, y_train[:, 1], y_train[:, 0])
            
            # Predict Survival Function on Test Risk (Use Margin)
            xgb_surv_funcs = breslow_xgb.get_survival_function(risk) # risk is MARGIN from line 563
            
            # IBS: Use filtered test set
            if y_test_struc_ibs is not None:
                risk_xgb_ibs = bst.predict(xgb.DMatrix(X_test_ibs), output_margin=True)
                survs_xgb_ibs = breslow_xgb.get_survival_function(risk_xgb_ibs)
                surv_probs_xgb = np.row_stack([fn(times) for fn in survs_xgb_ibs])
                
                ibs_xgb = integrated_brier_score(y_train_struc, y_test_struc_ibs, surv_probs_xgb, times)
                results['XGB_IBS'] = ibs_xgb
                
                # Integrated AUC & Median AUC
                auc_scores_xgb, mean_auc_xgb = cumulative_dynamic_auc(y_train_struc, y_test_struc_ibs, risk_xgb_ibs, times)
                results['XGB_IntegratedAUC'] = mean_auc_xgb
                results['XGB_AUC_Median'] = auc_scores_xgb[1]
            else:
                results['XGB_IBS'] = np.nan
                results['XGB_IntegratedAUC'] = np.nan
                results['XGB_AUC_Median'] = np.nan
            
        except Exception as e:
            print(f"  Error in XGB IBS/AUC: {e}")
            results['XGB_IBS'] = np.nan
            results['XGB_IntegratedAUC'] = np.nan
            results['XGB_AUC_Median'] = np.nan
            
    except Exception as e:
        results['XGB_C_Index'] = np.nan
        results['XGB_IBS'] = np.nan
        results['XGB_IntegratedAUC'] = np.nan
        results['XGB_AUC_Median'] = np.nan
        
    # --- DeepSurv (Full Batch to match Reference) ---
    try:
        # NOTE: y_train, y_test have [Event, Time]
        y_event_t = torch.tensor(y_train[:, 0], dtype=torch.float32).to(DEVICE)
        y_time_t = torch.tensor(y_train[:, 1], dtype=torch.float32).to(DEVICE)
        X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        X_te = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

        ds_model = DeepSurv(X_train.shape[1], hidden_dims=[32, 32], dropout=0.3).to(DEVICE)
        optimizer = optim.Adam(ds_model.parameters(), lr=best_params.get('lr', 1e-3))
        
        ds_model.train()
        for epoch in range(100): # Reference uses 100 epochs, no batches
             optimizer.zero_grad()
             
             # Full Batch Forward
             log_h = ds_model(X_t).squeeze()
             
             # Sort whole dataset by time desc (Standard Cox)
             # Note: T_max at index 0, T_min at index N
             sorted_idx = torch.argsort(y_time_t, descending=True)
             log_h_sorted = log_h[sorted_idx]
             events_sorted = y_event_t[sorted_idx]
             
             loss = cox_ph_loss(log_h_sorted, events_sorted)
             
             if torch.isnan(loss) or torch.isinf(loss):
                 print(f"    [DeepSurv] Loss is {loss.item()} at epoch {epoch}!")
                 # Check if inputs are bad
                 if torch.isnan(log_h).any(): print("      log_h contains NaNs")
                 if torch.isnan(X_t).any(): print("      X_t contains NaNs")
                 break
                 
             loss.backward()
             torch.nn.utils.clip_grad_norm_(ds_model.parameters(), 1.0) # Clip Gradients
             optimizer.step()
                 
        ds_model.eval()
        with torch.no_grad():
            # Output of ds_model is log_h. 
            log_h_pred = ds_model(X_te).cpu().numpy().flatten()
            risk_pred = np.exp(log_h_pred)
            
        # C-Index
        # sksurv.concordance_index_censored expects risk (larger = shorter survival)
        c_index_ds = concordance_index_censored(y_test_struc['Status'], y_test_struc['Survival_in_days'], risk_pred)[0]
        results['DeepSurv_C_Index'] = c_index_ds
        
        # IBS / AUC (Need Survival Function)
        try:
            breslow_ds = BreslowEstimator()
            
            # Risk on Train (Use Log-Hazard)
            with torch.no_grad():
                log_h_train_ds = ds_model(X_t).cpu().numpy().flatten()
                
            breslow_ds.fit(log_h_train_ds, y_train[:, 1], y_train[:, 0])
            
            if y_test_struc_ibs is not None:
                # Predict Survival Functions
                with torch.no_grad():
                    log_h_pred_ibs = ds_model(torch.tensor(X_test_ibs, dtype=torch.float32).to(DEVICE)).cpu().numpy().flatten()
                    risk_pred_ibs = np.exp(log_h_pred_ibs)
                
                ds_surv_funcs = breslow_ds.get_survival_function(log_h_pred_ibs)
                surv_probs_ds = np.row_stack([fn(times) for fn in ds_surv_funcs])
                
                results['DeepSurv_IBS'] = integrated_brier_score(y_train_struc, y_test_struc_ibs, surv_probs_ds, times)
                
                # AUC
                auc_scores_ds, mean_auc_ds = cumulative_dynamic_auc(y_train_struc, y_test_struc_ibs, risk_pred_ibs, times)
                results['DeepSurv_IntegratedAUC'] = mean_auc_ds
                results['DeepSurv_AUC_Median'] = auc_scores_ds[1]
            else:
                results['DeepSurv_IBS'] = np.nan
                results['DeepSurv_IntegratedAUC'] = np.nan
                results['DeepSurv_AUC_Median'] = np.nan
            
        except Exception as e:
            print(f"  Error in DeepSurv IBS/AUC: {e}")
            results['DeepSurv_IBS'] = np.nan
            results['DeepSurv_IntegratedAUC'] = np.nan
            results['DeepSurv_AUC_Median'] = np.nan

    except Exception as e:
        print(f"  Error in DeepSurv: {e}")
        results['DeepSurv_C_Index'] = np.nan
        results['DeepSurv_IBS'] = np.nan
        results['DeepSurv_IntegratedAUC'] = np.nan
        results['DeepSurv_AUC_Median'] = np.nan
        
    # --- DeepHit (Full Batch) ---
    try:
        # Prepare Data (Discretized Time)
        num_durations = 10 # Reference train_all_models_cv1.py uses 10
        
        # Discretize based on reference logic in train_all_models_cv1.py
        t_train_raw = y_train[:, 1]
        max_t = t_train_raw.max()
        bins = np.linspace(0, max_t + 1e-5, num_durations + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2 # Fix scope of bin_centers
        
        def discretize(t, bins):
            idxs = np.digitize(t, bins[1:], right=True)
            return np.clip(idxs, 0, num_durations - 1)
            
        y_disc_train = discretize(y_train[:, 1], bins)
        
        X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        yt_t = torch.tensor(y_disc_train, dtype=torch.long).to(DEVICE)
        ye_t = torch.tensor(y_train[:, 0], dtype=torch.float32).to(DEVICE)
        
        dh_model = DeepHit(X_train.shape[1], hidden_dims=[64, 32], dropout=0.3, num_durations=num_durations).to(DEVICE)
        optimizer = optim.Adam(dh_model.parameters(), lr=best_params.get('lr', 1e-3))
        
        dh_model.train()
        for epoch in range(100):
             optimizer.zero_grad()
             
             # Defensive: Check for NaNs in input
             if torch.isnan(X_t).any():
                  print("    [DeepHit] X_t contains NaNs right before forward!")
                  break
                  
             out = dh_model(X_t)
             
             loss = deephit_loss(out, yt_t, ye_t)
             
             if torch.isnan(loss) or torch.isinf(loss):
                 print(f"    [DeepHit] Loss is {loss.item()} at epoch {epoch}!")
                 # Check weights
                 for name, param in dh_model.named_parameters():
                      if torch.isnan(param).any(): print(f"      Param {name} has NaNs")
                 break
             
             loss.backward()
             torch.nn.utils.clip_grad_norm_(dh_model.parameters(), 1.0) # Clip Gradients
             optimizer.step()
                
        dh_model.eval()
        X_te = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            # Evaluation on FULL test set for C-Index
            logits_te = dh_model(X_te)
            pmf_te = F.softmax(logits_te, dim=1).cpu().numpy()
        
        cdf_te = np.cumsum(pmf_te, axis=1)
        # Risk score = -integral(S(t)) ~ -sum(S(t))
        risk_score_dh = -np.sum(1.0 - cdf_te, axis=1)
        c_index_dh = concordance_index_censored(y_test_struc['Status'], y_test_struc['Survival_in_days'], risk_score_dh)[0]
        results['DeepHit_C_Index'] = c_index_dh

        if y_test_struc_ibs is not None:
            # Mask pmf for IBS set
            pmf_ibs = pmf_te[valid_test_idx]
            cdf_ibs = np.cumsum(pmf_ibs, axis=1)
            
            surv_probs_dh = []
            for i in range(len(pmf_ibs)):
                 s_probs = 1.0 - cdf_ibs[i]
                 
                 # Ensure StepFunction style logic (Matches reference deephit_predict_survival)
                 x_surv = np.concatenate(([0.0], bin_centers))
                 y_surv = np.concatenate(([1.0], s_probs))
                 
                 # Clip times to evaluation range to avoid extrapolation
                 s_t = np.interp(times, x_surv, y_surv)
                 surv_probs_dh.append(s_t)
            surv_probs_dh = np.array(surv_probs_dh)
            
            results['DeepHit_IBS'] = integrated_brier_score(y_train_struc, y_test_struc_ibs, surv_probs_dh, times)
            
            # AUC on IBS-capable set
            risk_score_dh_ibs = risk_score_dh[valid_test_idx]
            auc_scores_dh, mean_auc_dh = cumulative_dynamic_auc(y_train_struc, y_test_struc_ibs, risk_score_dh_ibs, times)
            results['DeepHit_IntegratedAUC'] = mean_auc_dh
            results['DeepHit_AUC_Median'] = auc_scores_dh[1]
        else:
            results['DeepHit_IBS'] = np.nan
            results['DeepHit_IntegratedAUC'] = np.nan
            results['DeepHit_AUC_Median'] = np.nan

    except Exception as e:
        print(f"  Error in DeepHit: {e}")
        results['DeepHit_C_Index'] = np.nan
        results['DeepHit_IBS'] = np.nan
        results['DeepHit_IntegratedAUC'] = np.nan
        results['DeepHit_AUC_Median'] = np.nan
        
    return results

# ==============================================================================
# 4. MAIN LOOP
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=DATASETS + ['all'], default='all')
    parser.add_argument("--scenario", type=str, choices=SCENARIOS + ['all'], default='all')
    args = parser.parse_args()
    
    set_seed(42)
    final_results = []
    
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    scenarios = SCENARIOS if args.scenario == 'all' else [args.scenario]
    
    for dataset in datasets:
        for scenario in scenarios:
            print(f"\nProcessing {dataset.upper()} - {scenario.upper()}")
            
            # 1. Load Data
            df_truth, df_mnar, df_mask, splits = load_data(dataset, scenario)
            
            # 2. Preprocess (Encode Sex, etc.)
            df_truth, feats, t_col, e_col = preprocess_df(df_truth)
            df_mnar, _, _, _ = preprocess_df(df_mnar) # Assume structure aligned
            df_mask = df_mask[feats] # Align columns
            
            X_truth_full = df_truth[feats].values
            X_mnar_full = df_mnar[feats].values
            M_full = (~np.isnan(X_mnar_full)).astype(float) # Derived mask 1=Obs, 0=Miss
            Y_full = df_truth[[e_col, t_col]].values # Event, Time

            # --- Global normalization scaler (from full truth, for RMSE_Normalized) ---
            # This ensures normalized metrics are comparable across scenarios regardless of
            # MNAR severity (per-fold scaler shrinks under heavy missingness, inflating norm RMSE)
            global_min = np.nanmin(X_truth_full, axis=0)
            global_max = np.nanmax(X_truth_full, axis=0)
            global_range = global_max - global_min
            global_range[global_range == 0] = 1.0
            
            # 3. Load Optimized Params
            param_file = os.path.join(OPTUNA_DIR, f'best_{dataset}_{scenario}.json')
            if os.path.exists(param_file):
                with open(param_file, 'r') as f:
                    opt_data = json.load(f)
                # FIX: Extract nested 'best_params'
                opt_params = opt_data.get('best_params', {})
                print(f"  Loaded params: {opt_params}")
            else:
                print(f"  Warning: Params not found, using defaults.")
                opt_params = {}
                
            # 4. Loop Folds
            fold_metrics = []
            
            fold_keys = [k for k in splits.keys() if 'fold_' in k]
            fold_keys.sort(key=lambda x: int(x.split('_')[1]))
            
            for f_idx, fold_key in enumerate(fold_keys):
                print(f"  Fold {f_idx+1}...", end='', flush=True)
                
                # Indices
                idx_train = splits[fold_key]['train']
                idx_val = splits[fold_key]['val']
                idx_test = splits[fold_key]['test']
                
                # Combine Train+Val for final training (strict CV mandates this, or just Train)
                # Typically valid to use Train+Val for final model, eval on Test
                idx_fit = idx_train + idx_val
                
                # Split Data
                X_train = X_mnar_full[idx_fit]
                # Mask for VAE (Tracks ALL missingness: Natural + Artificial)
                M_train = (~np.isnan(X_train)).astype(np.float32)
                
                Y_train = Y_full[idx_fit]
                
                X_test = X_mnar_full[idx_test]
                M_test = (~np.isnan(X_test)).astype(np.float32) # Mask for VAE (Tracks ALL missingness: Natural + Artificial)
                Y_test = Y_full[idx_test]
                Mask_test_sim = df_mask.iloc[idx_test].values # Simulation mask for RMSE
                X_truth_test = X_truth_full[idx_test]

                # FILTER: Remove non-positive times (RSF requires time > 0)
                # Apply to Train
                valid_train = Y_train[:, 1] > 0
                X_train = X_train[valid_train]
                M_train = M_train[valid_train]
                Y_train = Y_train[valid_train]
                
                # Apply to Test
                valid_test = Y_test[:, 1] > 0
                X_test = X_test[valid_test]
                M_test = M_test[valid_test]
                Y_test = Y_test[valid_test]
                Mask_test_sim = Mask_test_sim[valid_test]
                X_truth_test = X_truth_test[valid_test]
                
                # Scale (Fit on Train, Transform Test)
                scaler = MinMaxScaler()
                
                # Careful: Fit only on OBSERVED data? 
                # Partial fit not supported by MinMaxScaler.
                # Standard practice: Fit min/max on available entries avoiding NaNs.
                # X_train has NaNs.
                
                # Custom MinMax ignoring NaNs
                train_min = np.nanmin(X_train, axis=0)
                train_max = np.nanmax(X_train, axis=0)
                
                # Handle columns that are All-NaN
                train_min = np.nan_to_num(train_min, nan=0.0)
                train_max = np.nan_to_num(train_max, nan=1.0)
                
                tune_range = train_max - train_min
                tune_range[tune_range == 0] = 1.0 # Avoid div/0
                
                # Double check for NaNs in range (should be gone)
                tune_range = np.nan_to_num(tune_range, nan=1.0)
                
                X_train_sc = (X_train - train_min) / tune_range
                X_test_sc = (X_test - train_min) / tune_range
                
                # Fill NaNs with 0.5 (middle of [0,1]) for neutral VAE input padding
                X_train_in = np.nan_to_num(X_train_sc, nan=0.5)
                X_test_in = np.nan_to_num(X_test_sc, nan=0.5)
                
                # Train VAE
                input_dim = X_train.shape[1]
                vae = train_vae(X_train_in, M_train, Y_train[:,1], Y_train[:,0], input_dim, opt_params)
                
                # Impute
                X_test_imp_sc = impute_data(vae, X_test_in, M_test)
                X_train_imp_sc = impute_data(vae, X_train_in, M_train)
                
                # Descale (for RMSE matches original scale usually)
                X_test_imp = X_test_imp_sc * tune_range + train_min
                X_train_imp = X_train_imp_sc * tune_range + train_min

                # ASSERTION: Ensure all NaNs (natural + artificial) are gone
                if np.isnan(X_train_imp).any() or np.isnan(X_test_imp).any():
                    print(f"  CRITICAL ERROR: NaNs remain after imputation in fold {f_idx+1}!")
                    X_train_imp = np.nan_to_num(X_train_imp, nan=0.0)
                    X_test_imp = np.nan_to_num(X_test_imp, nan=0.0)
                
                # Level 1: RMSE
                rmse = evaluate_rmse(X_test_imp, X_truth_test, M_test)

                # Normalized RMSE: use global truth scaler (not per-fold MNAR scaler)
                # This avoids inflated norm values under severe missingness where per-fold
                # min/max is estimated from fewer observed values.
                X_test_imp_global_sc = (X_test_imp - global_min) / global_range
                X_truth_test_global_sc = (X_truth_test - global_min) / global_range
                rmse_norm = evaluate_rmse(X_test_imp_global_sc, X_truth_test_global_sc, M_test)

                # Level 1: MAE
                mae = evaluate_mae(X_test_imp, X_truth_test, M_test)
                mae_norm = evaluate_mae(X_test_imp_global_sc, X_truth_test_global_sc, M_test)
                
                # Level 2: Downstream ML (Train on Imputed Train, Eval on Imputed Test)
                ml_res = run_ml_evaluation(X_train_imp, Y_train, X_test_imp, Y_test, opt_params)
                
                metrics = {
                    'Dataset': dataset,
                    'Scenario': scenario,
                    'Fold': f_idx + 1,
                    'RMSE': rmse,
                    'RMSE_Normalized': rmse_norm,
                    'MAE': mae,
                    'MAE_Normalized': mae_norm,
                    **ml_res
                }
                fold_metrics.append(metrics)
                final_results.append(metrics)
                
                print(f" Done. RMSE={rmse:.4f}, MAE={mae:.4f}, C-Index={ml_res.get('RSF_C_Index', np.nan):.4f}, IBS={ml_res.get('RSF_IBS', np.nan):.4f}, XGB-C={ml_res.get('XGB_C_Index', np.nan):.4f}, XGB-IBS={ml_res.get('XGB_IBS', np.nan):.4f}")
                
            # Summarize Scenario
            df_fold = pd.DataFrame(fold_metrics)
            print(f"  > Avg RMSE: {df_fold['RMSE'].mean():.4f} ± {df_fold['RMSE'].std():.4f}")
            print(f"  > Avg MAE:  {df_fold['MAE'].mean():.4f} ± {df_fold['MAE'].std():.4f}")
            print(f"  > Avg RSF C-Index: {df_fold['RSF_C_Index'].mean():.4f} ± {df_fold['RSF_C_Index'].std():.4f}")
            print(f"  > Avg RSF IBS:     {df_fold['RSF_IBS'].mean():.4f} ± {df_fold['RSF_IBS'].std():.4f}")
            print(f"  > Avg RSF IntAUC:  {df_fold['RSF_IntegratedAUC'].mean():.4f} ± {df_fold['RSF_IntegratedAUC'].std():.4f}")
            print(f"  > Avg RSF MedAUC:  {df_fold['RSF_AUC_Median'].mean():.4f} ± {df_fold['RSF_AUC_Median'].std():.4f}")
            print(f"  > Avg XGB C-Index: {df_fold['XGB_C_Index'].mean():.4f} ± {df_fold['XGB_C_Index'].std():.4f}")
            print(f"  > Avg XGB IBS:     {df_fold['XGB_IBS'].mean():.4f} ± {df_fold['XGB_IBS'].std():.4f}")
            print(f"  > Avg XGB IntAUC:  {df_fold['XGB_IntegratedAUC'].mean():.4f} ± {df_fold['XGB_IntegratedAUC'].std():.4f}")
            print(f"  > Avg XGB MedAUC:  {df_fold['XGB_AUC_Median'].mean():.4f} ± {df_fold['XGB_AUC_Median'].std():.4f}")
            
            print(f"  > Avg DeepSurv C:  {df_fold['DeepSurv_C_Index'].mean():.4f} ± {df_fold['DeepSurv_C_Index'].std():.4f}")
            print(f"  > Avg DeepSurv IBS:{df_fold['DeepSurv_IBS'].mean():.4f} ± {df_fold['DeepSurv_IBS'].std():.4f}")
            print(f"  > Avg DeepHit C:   {df_fold['DeepHit_C_Index'].mean():.4f} ± {df_fold['DeepHit_C_Index'].std():.4f}")
            print(f"  > Avg DeepHit IBS: {df_fold['DeepHit_IBS'].mean():.4f} ± {df_fold['DeepHit_IBS'].std():.4f}")

    # Save
    df_final = pd.DataFrame(final_results)
    out_path = os.path.join(RESULTS_DIR, 'final_evaluation_results.csv')
    df_final.to_csv(out_path, index=False)
    print(f"\nSaved final results to {out_path}")

if __name__ == "__main__":
    main()
