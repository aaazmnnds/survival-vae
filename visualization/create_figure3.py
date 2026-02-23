"""
Clinical Utility Plotting (Figure 3) - Portable Version
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

def create_figure3():
    POSSIBLE_DATA_DIRS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')),
        '/Users/azmannads/Documents/Research collections/Research 2025/datasets',
        '/Users/azmannads/VAE/datasets',
    ]
    DATA_DIR = next((p for p in POSSIBLE_DATA_DIRS if os.path.exists(p)), POSSIBLE_DATA_DIRS[0])
    
    ml_path = os.path.join(DATA_DIR, 'results_ml_level2.csv')
    if not os.path.exists(ml_path):
        print(f"Results not found at {ml_path}")
        return
        
    df = pd.read_csv(ml_path)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    scenarios = ['light', 'moderate', 'severe']
    methods = ['survival_vae', 'missforest', 'mida', 'gain', 'mice']
    method_labels = ['S-VAE', 'missForest', 'MIDA', 'GAIN', 'MICE']
    colors = {'survival_vae': '#E69F00', 'missforest': '#56B4E9', 'mida': '#009E73', 'gain': '#F0E442', 'mice': '#0072B2'}
    patterns = {'survival_vae': '///', 'missforest': '...', 'mida': '', 'gain': '|||', 'mice': '---'}
    
    plot_configs = [
        (axes[0, 0], 'metabric', 'C_Index', 'METABRIC: C-index', 'Higher'),
        (axes[0, 1], 'mimic', 'C_Index', 'MIMIC-IV: C-index', 'Higher'),
        (axes[1, 0], 'metabric', 'IBS', 'METABRIC: IBS', 'Lower'),
        (axes[1, 1], 'mimic', 'IBS', 'MIMIC-IV: IBS', 'Lower')
    ]
    
    for ax, dataset, metric, title, direction in plot_configs:
        data = df[(df['Dataset'].str.lower() == dataset) & (df['Model'] == 'rsf')]
        x = np.arange(len(scenarios))
        width = 0.15
        for i, method in enumerate(methods):
            m_data = data[data['Method'] == method]
            means = [m_data[m_data['Scenario'] == s][metric].mean() if not m_data[m_data['Scenario'] == s].empty else 0 for s in scenarios]
            stds = [m_data[m_data['Scenario'] == s][metric].std() if not m_data[m_data['Scenario'] == s].empty else 0 for s in scenarios]
            ax.bar(x + i*width, means, width, yerr=stds, capsize=3, label=method_labels[i], color=colors[method], hatch=patterns[method], edgecolor='black')
        
        ax.set_ylabel(f'{metric} ({direction} is Better)', fontsize=14)
        ax.set_title(title, fontweight='bold', fontsize=16)
        ax.set_xticks(x + 2*width)
        ax.set_xticklabels(['Light', 'Moderate', 'Severe'])
        ax.grid(axis='y', alpha=0.3)
        
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('figure3_clinical_utility.png', dpi=300, bbox_inches='tight')
    print("Figure 3 saved.")

if __name__ == "__main__":
    create_figure3()
