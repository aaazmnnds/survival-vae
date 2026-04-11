"""
Calculates normalized RMSE and MAE on held-out test fold artificially masked entries per CV fold.
"""

import pandas as pd
import numpy as np
import os
import json
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse

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

def find_dir(possibilities, default_name):
    for p in possibilities:
        if os.path.exists(p):
            return p
    return default_name

# Global path setup (will be finalized in main with argparse items)
DATA_DIR = find_dir(POSSIBLE_DATA_DIRS, 'datasets')

# Dataset Config
DATASETS = {
    'METABRIC': {
        'truth': 'metabric_processed.csv',
        'mask_prefix': 'metabric_mask',
        'cv_splits': 'cv_splits_metabric.json'
    },
    'MIMIC': {
        'truth': 'mimic_sepsis_highdim.csv', 
        'mask_prefix': 'mimic_mask',
        'cv_splits': 'cv_splits_mimic.json'
    }
}

SCENARIOS = ['light', 'moderate', 'severe']
METHODS = ['standard_vae', 'mice', 'missforest', 'gain', 'mida', 'survival_vae']

def find_truth_file(dataset_name, base_dir):
    if dataset_name.upper() == 'METABRIC':
        candidates = [
            os.path.join(base_dir, 'metabric_processed.csv'),
            os.path.join(base_dir, 'metabric.csv')
        ]
    elif dataset_name.upper() == 'MIMIC':
        candidates = [
            os.path.join(base_dir, 'final/mimic_sepsis_highdim.csv'),
            os.path.join(base_dir, 'mimic_sepsis_highdim.csv'),
            os.path.join(base_dir, 'mimic_processed.csv')
        ]
    else:
        return None
    
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_cv_splits(split_file):
    with open(os.path.join(DATA_DIR, split_file), 'r') as f:
        return json.load(f)

def verify_fold_data(dataset_name, cv_data, ground_truth_df):
    """Verify that fold indices are valid and data is consistent"""
    print(f"\n[VERIFICATION] {dataset_name}")
    print("-"*60)
    
    n_samples = len(ground_truth_df)
    print(f"Total samples in dataset: {n_samples}")
    
    total_test_samples = 0
    all_test_indices = set()
    
    # Map fold_1..5 to 0..4
    for i in range(1, 6):
        fold_key = f'fold_{i}'
        if fold_key not in cv_data:
            print(f"  Warning: Missing {fold_key} in CV data")
            continue
            
        test_indices = cv_data[fold_key]['test']
        
        # Check indices are in range
        if max(test_indices) >= n_samples:
             print(f"  Warning: Fold {fold_key}: indices exceed dataset size! Max index: {max(test_indices)}")
        
        # Check for overlap
        current_indices_set = set(test_indices)
        overlap = all_test_indices & current_indices_set
        if len(overlap) > 0:
            print(f"  Warning: Fold {fold_key}: {len(overlap)} overlapping indices with previous folds!")
            
        all_test_indices.update(current_indices_set)
        total_test_samples += len(test_indices)
        print(f"  Fold {i-1} ({fold_key}): {len(test_indices)} test samples [DONE]")
    
    print(f"  Total test samples across folds: {total_test_samples}")
    print(f"  Expected (N): {n_samples}")
    if total_test_samples != n_samples:
         print(f"  Warning: Total test samples ({total_test_samples}) != Dataset size ({n_samples})")
    print()

def calculate_metrics_on_indices(truth_df, imputed_df, mask_df, indices):
    """Calculates RMSE and MAE on specific indices (masked values only)."""
    
    # Extract subsets
    truth_sub = truth_df.iloc[indices]
    imp_sub = imputed_df.iloc[indices]
    mask_sub = mask_df.iloc[indices]
    
    all_y_true = []
    all_y_imp = []
    
    common_cols = [c for c in truth_sub.columns if c in imp_sub.columns and c in mask_sub.columns]
    
    for col in common_cols:
        if not pd.api.types.is_numeric_dtype(truth_sub[col]):
            continue
            
        # Handle Boolean strings or 0/1 in mask
        if mask_sub[col].dtype == object:
            mask_vals = mask_sub[col].astype(str).str.lower() == 'true'
        else:
            mask_vals = mask_sub[col].astype(bool)
            
        # Target: Observed in Truth AND Missing in Mask (artificially removed)
        # Note: imputation logic usually keeps observed values, but we check mask just in case
        target_mask = (truth_sub[col].notna()) & (mask_vals == True)
        
        if target_mask.sum() == 0:
            continue
            
        y_true = truth_sub.loc[target_mask, col].values
        y_imp = imp_sub.loc[target_mask, col].values
        
        all_y_true.extend(y_true)
        all_y_imp.extend(y_imp)
        
    if len(all_y_true) > 0:
        rmse = np.sqrt(mean_squared_error(all_y_true, all_y_imp))
        mae = mean_absolute_error(all_y_true, all_y_imp)
        n_masked = len(all_y_true)
    else:
        rmse, mae, n_masked = np.nan, np.nan, 0
        
    return rmse, mae, n_masked

def calculate_mice_metrics_for_fold(fold_test_indices, imputed_dfs_list, mask_df, truth_df):
    """
    Calculate MICE metrics for one fold using all 5 imputations.
    Averages metrics across the 5 imputations (Rubin's rules for point estimates).
    """
    rmse_list = []
    mae_list = []
    total_masked = 0
    
    for imputed_df in imputed_dfs_list:
        rmse, mae, n_masked = calculate_metrics_on_indices(truth_df, imputed_df, mask_df, fold_test_indices)
        
        if not np.isnan(rmse):
            rmse_list.append(rmse)
            mae_list.append(mae)
            total_masked = n_masked # Should be same for all imputations
            
    if not rmse_list:
        return np.nan, np.nan, 0
        
    # Average across imputations
    avg_rmse = np.mean(rmse_list)
    avg_mae = np.mean(mae_list)
    
    return avg_rmse, avg_mae, total_masked

def generate_summary_table(results_df, output_path):
    """Generate summary statistics (mean ± std across folds)"""
    summary = results_df.groupby(['Dataset', 'Scenario', 'Method']).agg({
        'RMSE': ['mean', 'std'],
        'MAE': ['mean', 'std']
    }).round(4)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.to_csv(output_path)
    print(f"Saved summary statistics to {output_path}")
    return summary

def main():
    parser = argparse.ArgumentParser(description='Calculate imputation metrics for a specific fold.')
    parser.add_argument('--fold', type=int, required=True, choices=[0,1,2,3,4], help='CV fold index (0-4)')
    args = parser.parse_args()
    
    # Finalize paths
    RESULTS_DIR = os.path.join(DATA_DIR, 'final', 'imputation_metrics', f'fold_{args.fold}')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    OUTPUT_FILE = os.path.join(RESULTS_DIR, 'results_imputation_folds.csv')
    SUMMARY_FILE = os.path.join(RESULTS_DIR, 'results_imputation_summary.csv')
    
    results = []
    
    # Import Scaler
    from sklearn.preprocessing import MinMaxScaler
    
    for dataset_name_key, config in DATASETS.items():
        dataset_slug = dataset_name_key.lower() # metabric
        dataset_display = dataset_name_key # METABRIC
        
        print(f"Processing {dataset_display} Dataset")
        print("="*60)
        
        # Load Ground Truth
        truth_path = find_truth_file(dataset_display, DATA_DIR)
        
        if not truth_path or not os.path.exists(truth_path):
            print(f"  Error: Ground truth not found for {dataset_display} in {DATA_DIR}")
            continue
        print(f"  Using truth file: {truth_path}")
        df_truth = pd.read_csv(truth_path)
        
        # --- NORMALIZATION LOGIC ---
        # We explicitly normalize specific columns for MIMIC to [0,1] to allow comparison with METABRIC
        # METABRIC is assumed to be already processed/normalized, but we can re-normalize to be safe/consistent?
        # User requested: "Normalize to [0,1] using same MinMaxScaler from training" implies using per-fold scaling?
        # But for reporting imputation error on the whole dataset, a global scaler on the Truth makes sense.
        # This keeps the "Ground Truth" range as [0,1].
        
        feature_cols = [c for c in df_truth.columns if pd.api.types.is_numeric_dtype(df_truth[c]) 
                       and c not in ['duration', 'event', 'Time', 'Event', 'Survival_in_days', 'Status', 'subject_id', 'hadm_id', 'stay_id', 'admittime', 'dischtime']]
        
        # Load CV Splits
        cv_data = load_cv_splits(config['cv_splits'])
        
        # 5. Fix Scaler: Fit on train rows only (NaN-safe)
        fold_key = f'fold_{args.fold + 1}'
        train_idx = np.array(cv_data[fold_key]['train'])
        
        raw_vals = df_truth[feature_cols].values
        data_min = np.nanmin(raw_vals[train_idx], axis=0)
        data_max = np.nanmax(raw_vals[train_idx], axis=0)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0
        
        data_min_s = pd.Series(data_min, index=feature_cols)
        data_range_s = pd.Series(data_range, index=feature_cols)
        
        def normalize_df(df_target):
            df_norm = df_target.copy()
            for col in feature_cols:
                if col in df_target.columns:
                     df_norm[col] = (df_target[col] - data_min_s[col]) / data_range_s[col]
            return df_norm
        
        # Normalize Ground Truth
        print(f"  [ normalization ] Normalizing {dataset_display} ground truth to [0,1] scale based on training range...")
        df_truth = normalize_df(df_truth)
        
        # We need a dummy verification call. We don't have imputed data yet inside loop.
        # But we can verify CV against truth.
        verify_fold_data(dataset_display, cv_data, df_truth)
        
        for scenario in SCENARIOS:
            print(f"  Scenario: {scenario}")
            print("-" * 40)
            
            # Load Mask
            mask_path = os.path.join(DATA_DIR, f"{config['mask_prefix']}_{scenario}.csv")
            if not os.path.exists(mask_path):
                print(f"    Mask not found: {mask_path}")
                continue
            df_mask = pd.read_csv(mask_path)
            
            for method in METHODS:
                print(f"    Method: {method}")
                
                # Load Imputed Data
                if method == 'mice':
                    imputed_dfs = []
                    for m in range(1, 6):
                        imp_path = os.path.join(DATA_DIR, 'final', 'imputation_results_final', f'fold_{args.fold}', f"{dataset_slug}_{scenario}_mice_imputed_m{m}.csv")
                        if os.path.exists(imp_path):
                            print(f"      Loading MICE m={m}: {os.path.basename(imp_path)}")
                            df_temp = pd.read_csv(imp_path)
                            if dataset_display in ['MIMIC', 'METABRIC']:
                                df_temp = normalize_df(df_temp)
                            imputed_dfs.append(df_temp)
                    
                    if len(imputed_dfs) < 5:
                         print(f"      Warning: Found only {len(imputed_dfs)}/5 MICE files for fold {args.fold}")
                    
                    if not imputed_dfs:
                        continue
                        
                else: # Single imputation
                    imp_path = os.path.join(DATA_DIR, 'final', 'imputation_results_final', f'fold_{args.fold}', f"{dataset_slug}_{scenario}_{method}_imputed.csv")
                    
                    if not os.path.exists(imp_path):
                        print(f"      File not found: {imp_path}")
                        continue
                        
                    print(f"      Loading: {os.path.basename(imp_path)}")
                    df_imp = pd.read_csv(imp_path)
                    if dataset_display in ['MIMIC', 'METABRIC']:
                        df_imp = normalize_df(df_imp)
                
                # Calculate for the specific fold only
                test_idx = cv_data[fold_key]['test']
                
                if method == 'mice':
                    rmse, mae, n_masked = calculate_mice_metrics_for_fold(
                        test_idx, imputed_dfs, df_mask, df_truth
                    )
                    mode_str = "(averaged over 5 imputations)"
                else:
                    rmse, mae, n_masked = calculate_metrics_on_indices(
                        df_truth, df_imp, df_mask, test_idx
                    )
                    mode_str = f"({n_masked:,} masked values)"
                
                if not np.isnan(rmse):
                    print(f"      Fold {args.fold}: RMSE={rmse:.4f}, MAE={mae:.4f} {mode_str}")
                    results.append({
                        'Dataset': dataset_display,
                        'Scenario': scenario,
                        'Method': method,
                        'Fold': args.fold,
                        'RMSE': rmse,
                        'MAE': mae
                    })
                

    # Save outputs
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved fold-level results to {OUTPUT_FILE}")
        
        generate_summary_table(df_results, SUMMARY_FILE)
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
