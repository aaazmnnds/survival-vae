import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
BASE_RESULTS_DIR = os.environ.get('SVAE_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'final'))
OUTPUT_DIR = os.environ.get('SVAE_OUTPUT_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(BASE_RESULTS_DIR, 'imputation_metrics', 'aggregated', 'results_imputation_main_table.csv')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'figure_imputation_fidelity.png')

# Load data
df = pd.read_csv(CSV_PATH)

# Sort and normalize Method names for consistent ordering
method_order = ['survival_vae', 'standard_vae', 'mice', 'missforest', 'gain', 'mida']
method_labels = ['Survival-VAE', 'Standard VAE', 'MICE', 'missForest', 'GAIN', 'MIDA']
scenario_order = ['light', 'moderate', 'severe']
scenario_labels = ['Light', 'Moderate', 'Severe']

# Style configuration
# Survival-VAE in orange with diagonal hatching, others distinct
colors = {
    'survival_vae': '#E69F00',  # Orange
    'standard_vae': '#56B4E9',  # SkyBlue
    'mice': '#999999',         # Grey
    'missforest': '#CC79A7',    # Reddish Purple
    'gain': '#009E73',         # Blueish Green
    'mida': '#D55E00'          # Vermillion
}
hatches = {
    'survival_vae': '//',
    'standard_vae': '',
    'mice': '',
    'missforest': '',
    'gain': '',
    'mida': ''
}

def plot_imputation_fidelity():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    
    # 2x2 Layout: (RMSE top, MAE bottom), (METABRIC left, MIMIC right)
    # axes[0,0]: METABRIC RMSE
    # axes[0,1]: MIMIC RMSE
    # axes[1,0]: METABRIC MAE
    # axes[1,1]: MIMIC MAE
    
    config = [
        # (row, col, dataset, metric, title)
        (0, 0, 'METABRIC', 'RMSE', 'METABRIC - RMSE'),
        (0, 1, 'MIMIC', 'RMSE', 'MIMIC-IV - RMSE'),
        (1, 0, 'METABRIC', 'MAE', 'METABRIC - MAE'),
        (1, 1, 'MIMIC', 'MAE', 'MIMIC-IV - MAE'),
    ]
    
    x = np.arange(len(scenario_labels))
    width = 0.14  # Adjust bar width
    
    for row, col, dataset, metric, title in config:
        ax = axes[row, col]
        ds_df = df[df['Dataset'] == dataset]
        
        for i, method in enumerate(method_order):
            means = []
            stds = []
            for scenario in scenario_order:
                row_data = ds_df[(ds_df['Scenario'] == scenario) & (ds_df['Method'] == method)]
                if not row_data.empty:
                    means.append(row_data[f'{metric}_Mean'].values[0])
                    stds.append(row_data[f'{metric}_SD'].values[0])
                else:
                    means.append(0)
                    stds.append(0)
            
            offset = (i - len(method_order)/2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, label=method_labels[i],
                   color=colors[method], hatch=hatches[method], 
                   capsize=3, edgecolor='black', alpha=0.85)
        
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels, fontsize=12)
        # ax.set_ylim(0, 1.0) # Removed fixed [0,1] scale to allow better visibility
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
    # Add a single legend for the entire figure
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03),
               ncol=len(method_order), fontsize=12, frameon=False)
    
    # Save the figure
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    plot_imputation_fidelity()
