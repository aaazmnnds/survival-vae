"""
Imputation Metrics Calculation (Portable Version)
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

def calculate_metrics_on_indices(truth_df, imputed_df, mask_df, indices):
    truth_sub = truth_df.iloc[indices]
    imp_sub = imputed_df.iloc[indices]
    mask_sub = mask_df.iloc[indices]
    all_y_true, all_y_imp = [], []
    
    feature_cols = [c for c in truth_sub.columns if pd.api.types.is_numeric_dtype(truth_sub[c]) 
                   and c not in ['duration', 'event', 'Time', 'Event', 'Survival_in_days', 'Status', 'subject_id', 'hadm_id', 'stay_id']]

    for col in feature_cols:
        if col not in imp_sub.columns or col not in mask_sub.columns: continue
        mask_vals = mask_sub[col].astype(bool) if mask_sub[col].dtype != object else mask_sub[col].astype(str).str.lower() == 'true'
        target_mask = (truth_sub[col].notna()) & (mask_vals == True)
        if target_mask.sum() == 0: continue
        all_y_true.extend(truth_sub.loc[target_mask, col].values)
        all_y_imp.extend(imp_sub.loc[target_mask, col].values)
        
    if not all_y_true: return np.nan, np.nan, 0
    return np.sqrt(mean_squared_error(all_y_true, all_y_imp)), mean_absolute_error(all_y_true, all_y_imp), len(all_y_true)

def run_evaluation():
    POSSIBLE_DATA_DIRS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')),
        '/Users/azmannads/Documents/Research collections/Research 2025/datasets',
        '/Users/azmannads/VAE/datasets',
    ]
    DATA_DIR = next((p for p in POSSIBLE_DATA_DIRS if os.path.exists(p)), POSSIBLE_DATA_DIRS[0])
    
    DATASETS = {'METABRIC': {'truth': 'metabric_processed.csv', 'mask_prefix': 'metabric_mask', 'cv_splits': 'cv_splits_metabric.json'},
                'MIMIC': {'truth': 'final/mimic_sepsis_highdim.csv', 'mask_prefix': 'mimic_mask', 'cv_splits': 'cv_splits_mimic.json'}}
    
    results = []
    for dataset_name, config in DATASETS.items():
        truth_path = os.path.join(DATA_DIR, config['truth'])
        if not os.path.exists(truth_path): continue
        df_truth = pd.read_csv(truth_path)
        
        # Global normalization for metrics
        cols = [c for c in df_truth.columns if pd.api.types.is_numeric_dtype(df_truth[c]) and c not in ['duration', 'event', 'Time', 'Event', 'Survival_in_days', 'Status']]
        d_min, d_max = df_truth[cols].min(), df_truth[cols].max()
        d_range = (d_max - d_min).replace(0, 1)
        df_truth_norm = df_truth.copy()
        for c in cols: df_truth_norm[c] = (df_truth[c] - d_min[c]) / d_range[c]
        
        with open(os.path.join(DATA_DIR, config['cv_splits']), 'r') as f:
            cv_data = json.load(f)
            
        for scenario in ['light', 'moderate', 'severe']:
            df_mask = pd.read_csv(os.path.join(DATA_DIR, f"{config['mask_prefix']}_{scenario}.csv"))
            for method in ['survival_vae', 'mice', 'missforest', 'gain', 'mida']:
                # Handle MICE m=5
                if method == 'mice':
                    for m in range(1, 6):
                        imp_path = os.path.join(DATA_DIR, f"{dataset_name.lower()}_imputed_mice_{scenario}_{m}.csv")
                        if not os.path.exists(imp_path): continue
                        df_imp = pd.read_csv(imp_path)
                        for c in cols: df_imp[c] = (df_imp[c] - d_min[c]) / d_range[c]
                        for f_idx in range(5):
                            test_idx = cv_data[f'fold_{f_idx+1}']['test']
                            rmse, mae, n = calculate_metrics_on_indices(df_truth_norm, df_imp, df_mask, test_idx)
                            results.append({'Dataset': dataset_name, 'Scenario': scenario, 'Method': 'mice', 'Fold': f_idx, 'RMSE': rmse, 'MAE': mae})
                else:
                    imp_path = os.path.join(DATA_DIR, f"{dataset_name.lower()}_imputed_{method}_{scenario}.csv")
                    if not os.path.exists(imp_path): continue
                    df_imp = pd.read_csv(imp_path)
                    for c in cols: df_imp[c] = (df_imp[c] - d_min[c]) / d_range[c]
                    for f_idx in range(5):
                        test_idx = cv_data[f'fold_{f_idx+1}']['test']
                        rmse, mae, n = calculate_metrics_on_indices(df_truth_norm, df_imp, df_mask, test_idx)
                        results.append({'Dataset': dataset_name, 'Scenario': scenario, 'Method': method, 'Fold': f_idx, 'RMSE': rmse, 'MAE': mae})
                        
    if results:
        pd.DataFrame(results).to_csv(os.path.join(DATA_DIR, 'results_imputation_folds.csv'), index=False)
        print("Evaluation complete.")

if __name__ == "__main__":
    run_evaluation()
