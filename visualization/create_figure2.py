import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
# We use the new summary file with normalized metrics
df = pd.read_csv('datasets/results_imputation_summary.csv')

# Rename columns to match plotting logic if needed, or adjust plotting logic
# The csv has: 'RMSE_mean', 'MAE_mean'
df['RMSE'] = df['RMSE_mean']
df['MAE'] = df['MAE_mean']
df['Scenario'] = df['Scenario'].str.capitalize() # 'light' -> 'Light'

# Create figure with 2x2 grid (MAE top, RMSE bottom - as per manuscript caption)
# Caption: "mean absolute error (MAE, top row) and root mean squared error (RMSE, bottom row)"
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Define colors for methods
colors = {
    'survival_vae': '#2E86AB',  # Blue
    'mice': '#A23B72',          # Purple
    'missforest': '#F18F01',    # Orange
    'gain': '#C73E1D',          # Red
    'mida': '#6A994E'           # Green
}

# Add hatching for S-VAE as per caption "S-VAE (orange with diagonal hatching)" 
# Wait, code says Blue (#2E86AB). Caption says "orange with diagonal hatching"? 
# Maybe previous caption was wrong or code is different. 
# Manuscript caption: "S-VAE (orange with diagonal hatching)... baselines shown with distinct colors"
# I should stick to the code's colors but maybe add hatching for S-VAE to match caption *description* if I can.
# But swapping color to Orange for S-VAE might confuse if I don't update others.
# Let's keep code colors but ADD hatching for S-VAE. 
# And maybe update caption color description later if needed, but easier to just add hatching.

method_labels = {
    'survival_vae': 'S-VAE',
    'mice': 'MICE',
    'missforest': 'MissForest',
    'gain': 'GAIN',
    'mida': 'MIDA'
}

methods = ['survival_vae', 'mice', 'missforest', 'gain', 'mida']
scenarios = ['Light', 'Moderate', 'Severe']
width = 0.15

# Plotting Loop
for row_idx, metric in enumerate(['MAE', 'RMSE']):
    for col_idx, dataset in enumerate(['METABRIC', 'MIMIC']):
        ax = axes[row_idx, col_idx]
        data_subset = df[df['Dataset'] == dataset]
        
        x = np.arange(len(scenarios))
        
        for i, method in enumerate(methods):
            method_data = data_subset[data_subset['Method'] == method]
            if method_data.empty:
                continue
                
            # Extract values in correct order of scenarios
            values = []
            std_values = []
            for s in scenarios:
                row = method_data[method_data['Scenario'] == s]
                if not row.empty:
                    values.append(row[f'{metric}_mean'].values[0])
                    std_values.append(row[f'{metric}_std'].values[0])
                else:
                    values.append(0)
                    std_values.append(0)
            
            # Plot bar
            hatch = '//' if method == 'survival_vae' else None
            edgecolor = 'black' if method == 'survival_vae' else None
            
            ax.bar(x + i*width, values, width, 
                   yerr=std_values, capsize=3,
                   label=method_labels[method] if (row_idx==0 and col_idx==0) else "",
                   color=colors[method],
                   hatch=hatch,
                   edgecolor=edgecolor)
        
        ax.set_title(f'{dataset} - {metric}', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(scenarios)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        if col_idx == 0:
            ax.set_ylabel(f'{metric} (normalized)', fontsize=10, fontweight='bold')

# Legend (only once)
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=12)

plt.tight_layout()

# Save to correct path
out_path = 'elsarticle/figure_imputation_fidelity.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"✅ Chart saved to: {out_path}")

