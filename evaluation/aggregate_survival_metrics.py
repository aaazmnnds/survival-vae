"""
Aggregate Survival Metrics Across 5 Folds
==========================================
Reads per-fold C-index/Brier CSVs and produces:
1. Main table: Mean +/- SD per combination
2. LaTeX tables for manuscript (Clinical Utility)
"""

import os
import argparse
import numpy as np
import pandas as pd

# Configuration
POSSIBLE_DATA_DIRS = [
    os.environ.get('SVAE_RESULTS_DIR', ''),
    '/Users/nazu.ds/Documents/Research Collections/Research 2025/survival-vae',
    './datasets', 'datasets', '../datasets',
    '/home/azman/VAE/Survival-VAE_study',
    '/Users/azmannads/VAE/Survival-VAE_study',
    '/Users/azmannads/Documents/Research collections/Research 2025/datasets',
    '.'
]

def find_dir(possibilities, default_name):
    for p in possibilities:
        if p and os.path.exists(p): return p
    return default_name

DATA_DIR = find_dir(POSSIBLE_DATA_DIRS, 'datasets')

DATASETS = ['METABRIC', 'MIMIC']
SCENARIOS = ['light', 'moderate', 'severe']
METHODS = ['survival_vae', 'standard_vae', 'mice', 'missforest', 'gain', 'mida']
MODELS = ['rsf', 'xgb', 'deepsurv', 'deephit']

def calculate_mean_sd(values):
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0: return np.nan, np.nan
    return np.mean(arr), np.std(arr)

def load_all_folds():
    all_dfs = []
    for fold_idx in range(5):
        path = os.path.join(DATA_DIR, 'final', 'survival_results', f'fold_{fold_idx}', 'results_survival_fold.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_dfs.append(df)
            print(f"  Loaded fold {fold_idx}: {len(df)} rows")
    if not all_dfs:
        raise FileNotFoundError("No survival metric files found.")
    return pd.concat(all_dfs, ignore_index=True)

def generate_main_table(df, output_path):
    rows = []
    # Group by all dimensions
    groups = df.groupby(['Dataset', 'Scenario', 'Method', 'Model'])
    for (dataset, scenario, method, model), g in groups:
        c_mean, c_sd = calculate_mean_sd(g['C_Index'])
        ibs_mean, ibs_sd = calculate_mean_sd(g['IBS'])
        iauc_mean, iauc_sd = calculate_mean_sd(g['iAUC'])
        auctmed_mean, auctmed_sd = calculate_mean_sd(g['AUC_tmed'])
        iauc_mean, iauc_sd = calculate_mean_sd(g['iAUC'])
        auctmed_mean, auctmed_sd = calculate_mean_sd(g['AUC_tmed'])
        
        rows.append({
            'Dataset': dataset, 'Scenario': scenario, 'Method': method, 'Model': model,
            'C_Index': f"{c_mean:.4f} ({c_sd:.4f})" if not np.isnan(c_mean) else 'N/A',
            'C_Mean': round(c_mean, 4), 'C_SD': round(c_sd, 4),
            'IBS': f"{ibs_mean:.4f} ({ibs_sd:.4f})" if not np.isnan(ibs_mean) else 'N/A',
            'IBS_Mean': round(ibs_mean, 4), 'IBS_SD': round(ibs_sd, 4),
            'iAUC': f"{iauc_mean:.4f} ({iauc_sd:.4f})" if not np.isnan(iauc_mean) else 'N/A',
            'iAUC_Mean': round(iauc_mean, 4), 'iAUC_SD': round(iauc_sd, 4),
            'AUC_tmed': f"{auctmed_mean:.4f} ({auctmed_sd:.4f})" if not np.isnan(auctmed_mean) else 'N/A',
            'AUC_tmed_Mean': round(auctmed_mean, 4), 'AUC_tmed_SD': round(auctmed_sd, 4),
            'iAUC': f"{iauc_mean:.4f} ({iauc_sd:.4f})" if not np.isnan(iauc_mean) else 'N/A',
            'iAUC_Mean': round(iauc_mean, 4), 'iAUC_SD': round(iauc_sd, 4),
            'AUC_tmed': f"{auctmed_mean:.4f} ({auctmed_sd:.4f})" if not np.isnan(auctmed_mean) else 'N/A',
            'AUC_tmed_Mean': round(auctmed_mean, 4), 'AUC_tmed_SD': round(auctmed_sd, 4),
        })
    
    res_df = pd.DataFrame(rows)
    res_df.to_csv(output_path, index=False)
    print(f"Main survival table saved: {output_path}")
    return res_df

def generate_latex_utility_table(df, dataset, model, output_path):
    """Generates a table for a specific Model+Dataset across all scenarios and methods."""
    target_df = df[(df['Dataset'] == dataset.upper()) & (df['Model'] == model.lower())]
    if target_df.empty: return

    method_order = ['survival_vae', 'standard_vae', 'mice', 'missforest', 'gain', 'mida']
    method_display = {'survival_vae': 'Survival-VAE', 'standard_vae': 'Standard VAE', 'mice': 'MICE', 'missforest': 'missForest', 'gain': 'GAIN', 'mida': 'MIDA'}

    lines = [f"% Clinical Utility: {model} on {dataset}", "\\begin{tabular}{lcccccc}", "\\hline", 
             "\\textbf{Scenario} & \\textbf{Survival-VAE} & \\textbf{Standard VAE} & \\textbf{MICE} & \\textbf{missForest} & \\textbf{GAIN} & \\textbf{MIDA} \\\\", 
             "\\hline"]

    for sc in SCENARIOS:
        sc_df = target_df[target_df['Scenario'] == sc]
        row_cells = [sc.capitalize()]
        
        # Find best for this scenario
        means = []
        for m in method_order:
            m_row = sc_df[sc_df['Method'] == m]
            means.append(m_row['C_Mean'].values[0] if not m_row.empty else -1)
        best_idx = np.argmax(means) if any(m != -1 for m in means) else -1
        
        for i, m in enumerate(method_order):
            m_row = sc_df[sc_df['Method'] == m]
            if m_row.empty: 
                row_cells.append("--")
            else:
                val = f"{m_row['C_Mean'].values[0]:.3f} ({m_row['C_SD'].values[0]:.3f})"
                if i == best_idx: val = f"\\textbf{{{val}}}"
                row_cells.append(val)
        lines.append(" & ".join(row_cells) + " \\\\")
    
    lines += ["\\hline", "\\end{tabular}"]
    with open(output_path, 'w') as f: f.write("\n".join(lines))
    print(f"LaTeX table saved: {output_path}")

def main():
    output_dir = os.path.join(DATA_DIR, 'final', 'survival_results', 'aggregated')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading survival metrics...")
    df = load_all_folds()
    
    main_path = os.path.join(output_dir, 'results_clinical_utility_main.csv')
    main_df = generate_main_table(df, main_path)
    
    print("\nGenerating LaTeX Utility Tables...")
    for ds in ['METABRIC', 'MIMIC']:
        for mod in ['rsf', 'xgb', 'deepsurv', 'deephit']:
            path = os.path.join(output_dir, f'latex_utility_{ds}_{mod}.tex')
            generate_latex_utility_table(main_df, ds, mod, path)
    
    print("\nAggregation complete.")

if __name__ == "__main__":
    main()
