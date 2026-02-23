"""
Imputation Fidelity Plotting (Figure 2) - Portable Version
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def create_figure2():
    POSSIBLE_DATA_DIRS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')),
        '/Users/azmannads/Documents/Research collections/Research 2025/datasets',
        '/Users/azmannads/VAE/datasets',
    ]
    DATA_DIR = next((p for p in POSSIBLE_DATA_DIRS if os.path.exists(p)), POSSIBLE_DATA_DIRS[0])
    
    summary_path = os.path.join(DATA_DIR, 'results_imputation_summary.csv')
    if not os.path.exists(summary_path):
        # Try finding the fold-level results and aggregating if summary is missing
        fold_results_path = os.path.join(DATA_DIR, 'results_imputation_folds.csv')
        if not os.path.exists(fold_results_path):
            print("No results found to plot.")
            return
        df_raw = pd.read_csv(fold_results_path)
        df = df_raw.groupby(['Dataset', 'Scenario', 'Method']).agg({'RMSE': ['mean', 'std'], 'MAE': ['mean', 'std']}).reset_index()
        df.columns = ['Dataset', 'Scenario', 'Method', 'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAE_std']
    else:
        df = pd.read_csv(summary_path)

    df['Scenario'] = df['Scenario'].str.capitalize()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'survival_vae': '#E69F00', 'mice': '#0072B2', 'missforest': '#56B4E9', 'gain': '#F0E442', 'mida': '#009E73'}
    method_labels = {'survival_vae': 'S-VAE', 'mice': 'MICE', 'missforest': 'MissForest', 'gain': 'GAIN', 'mida': 'MIDA'}
    methods = ['survival_vae', 'mice', 'missforest', 'gain', 'mida']
    scenarios = ['Light', 'Moderate', 'Severe']
    width = 0.15

    for row_idx, metric in enumerate(['MAE', 'RMSE']):
        for col_idx, dataset in enumerate(['METABRIC', 'MIMIC']):
            ax = axes[row_idx, col_idx]
            data_subset = df[df['Dataset'].str.upper() == dataset.upper()]
            x = np.arange(len(scenarios))
            for i, method in enumerate(methods):
                m_data = data_subset[data_subset['Method'] == method]
                means = [m_data[m_data['Scenario'] == s][f'{metric}_mean'].values[0] if not m_data[m_data['Scenario'] == s].empty else 0 for s in scenarios]
                stds = [m_data[m_data['Scenario'] == s][f'{metric}_std'].values[0] if not m_data[m_data['Scenario'] == s].empty else 0 for s in scenarios]
                ax.bar(x + i*width, means, width, yerr=stds, capsize=3, label=method_labels[method] if (row_idx==0 and col_idx==0) else "", color=colors[method], hatch='//' if method == 'survival_vae' else None, edgecolor='black' if method == 'survival_vae' else None)
            ax.set_title(f'{dataset} - {metric}', fontsize=12, fontweight='bold')
            ax.set_xticks(x + width * 2)
            ax.set_xticklabels(scenarios)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            if col_idx == 0: ax.set_ylabel(f'{metric} (normalized)', fontsize=10, fontweight='bold')

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=12)
    plt.tight_layout()
    plt.savefig('figure2_imputation_fidelity.png', dpi=300, bbox_inches='tight')
    print("Figure 2 saved.")

if __name__ == "__main__":
    create_figure2()
