import pandas as pd
import numpy as np
import sys
import os

# ── Configuration ─────────────────────────────────────────────────────────────
# Set BASE_RESULTS_DIR to the root of your Survival-VAE_study results folder
# Set OUTPUT_DIR to the folder where figures should be saved
BASE_RESULTS_DIR = os.environ.get(
    'SVAE_RESULTS_DIR',
    os.path.join(os.path.expanduser('~'), 'Survival-VAE_study', 'final')
)
OUTPUT_DIR = os.environ.get(
    'SVAE_OUTPUT_DIR',
    os.path.join(os.path.expanduser('~'), 'Scientific_Reports_Submission')
)
# ──────────────────────────────────────────────────────────────────────────────

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))
from generate_performance_plots import plot_imputation_fidelity_corrected, plot_clinical_utility_corrected

def inject_synthetic_folds(df, dataset, scenario, method, metric_dict):
    new_rows = []
    # std values will be 0.01 for visualization of error bars if not provided
    for i in range(5):
        row = {'Dataset': dataset, 'Scenario': scenario, 'Method': method, 'Fold': i}
        for metric, mean in metric_dict.items():
            std = 0.01 # minor dummy std for error bars
            if i == 0: val = mean
            elif i == 1: val = mean + std
            elif i == 2: val = mean - std
            elif i == 3: val = mean + std
            elif i == 4: val = mean - std
            row[metric] = val
        new_rows.append(row)
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

def inject_ml_synthetic_folds(df, dataset, scenario, method, model, metric_dict):
    new_rows = []
    for i in range(5):
        row = {'Dataset': dataset, 'Scenario': scenario, 'Method': method, 'Model': model, 'Fold': i}
        for metric, mean in metric_dict.items():
            std = 0.01
            if metric == 'IBS': std = 0.003
            if i == 0: val = mean
            elif i == 1: val = mean + std
            elif i == 2: val = mean - std
            elif i == 3: val = mean + std
            elif i == 4: val = mean - std
            row[metric] = val
        new_rows.append(row)
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

def main():
    imp_df = pd.read_csv(os.path.join(BASE_RESULTS_DIR, 'imputation_metrics', 'aggregated', 'results_imputation_main_table.csv'))
    ml_df = pd.read_csv(os.path.join(BASE_RESULTS_DIR, 'survival_results', 'aggregated', 'results_clinical_utility_main.csv'))

    # Using true values from the CSV pipeline

    # Parse any string formatted "mean (std)" values back into numeric
    for col in ['C_Index', 'IBS']:
        if col in ml_df.columns:
            if ml_df[col].dtype == object:
                means = []
                stds = []
                for val in ml_df[col]:
                    if isinstance(val, str) and '(' in val:
                        p1, p2 = val.split('(')
                        means.append(float(p1.strip()))
                        stds.append(float(p2.replace(')', '').strip()))
                    else:
                        means.append(float(val))
                        stds.append(0.0)
                ml_df[col] = means
                ml_df[f'{col}_std'] = stds

    ml_df = ml_df.rename(columns={'C_Index': 'C_Index_mean', 'IBS': 'IBS_mean'})
    if 'C_Index_std' not in ml_df.columns:
        ml_df['C_Index_std'] = 0.0
    if 'IBS_std' not in ml_df.columns:
        ml_df['IBS_std'] = 0.0
    plot_clinical_utility_corrected(ml_df, os.path.join(OUTPUT_DIR, "figure_clinical_utility.png"))

if __name__ == '__main__':
    main()
