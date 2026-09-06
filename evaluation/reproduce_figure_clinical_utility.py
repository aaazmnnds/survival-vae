import pandas as pd
import numpy as np
import sys
import os

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_RESULTS_DIR = os.environ.get('SVAE_RESULTS_DIR', os.path.join(os.path.expanduser('~'), 'Survival-VAE_study', 'final'))
OUTPUT_DIR = os.environ.get('SVAE_OUTPUT_DIR', os.path.join(os.path.expanduser('~'), 'Scientific_Reports_Submission_2'))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))
from generate_performance_plots import plot_clinical_utility_corrected

def main():
    csv_path = os.path.join(BASE_RESULTS_DIR, 'survival_results', 'aggregated', 'results_clinical_utility_main.csv')
    df_long = pd.read_csv(csv_path)

    # Pivot to wide format
    df_wide = df_long.pivot(index=['Dataset', 'Scenario', 'Method', 'Model'], columns='Metric', values=['Mean', 'SD'])
    
    # Flatten columns
    df_wide.columns = [f'{metric}_{stat.lower()}' for stat, metric in df_wide.columns]
    df_wide = df_wide.reset_index()

    # Rename to match what plot_clinical_utility_corrected expects
    # It expects: 'C_Index_mean', 'C_Index_std', 'IBS_mean', 'IBS_std'
    rename_map = {
        'C_Index_sd': 'C_Index_std',
        'IBS_sd': 'IBS_std'
    }
    df_wide = df_wide.rename(columns=rename_map)

    plot_clinical_utility_corrected(df_wide, os.path.join(OUTPUT_DIR, "figure_clinical_utility.png"))

if __name__ == '__main__':
    main()
