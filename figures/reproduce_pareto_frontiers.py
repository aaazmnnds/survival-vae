import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
DATA_DIR = os.environ.get(
    'SVAE_RESULTS_DIR',
    '/Users/nazu.ds/Documents/Research Collections/Research 2025/survival-vae/final'
)
OPTUNA_DIR = os.path.join(DATA_DIR, 'optuna_results_final', 'fold_0')

OUTPUT_DIR = os.environ.get(
    'SVAE_OUTPUT_DIR',
    '/Users/nazu.ds/Documents/Research Collections/Research 2025/Scientific_Reports_Submission'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = ['metabric', 'mimic']
SCENARIOS = ['light', 'moderate', 'severe']

def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

def process_and_plot():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.25, bottom=0.18, left=0.12)
    
    legend_handles = []
    legend_labels = []
    
    idx = 0
    for r, ds in enumerate(DATASETS):
        for c, sc in enumerate(SCENARIOS):
            ax = axes[r, c]
            filename = f"{ds}_{sc}_optuna_survival_vae_results.csv"
            path = os.path.join(OPTUNA_DIR, filename)
            
            if not os.path.exists(path):
                ax.set_visible(False)
                continue
                
            df = pd.read_csv(path)
            df['target_rmse'] = df['rmse'].fillna(df['rmse_norm'])
            df = df.dropna(subset=['target_rmse', 'c_index'])
            
            costs = df[['target_rmse', 'c_index']].values
            pareto_costs = costs.copy()
            pareto_costs[:, 1] = -pareto_costs[:, 1]
            
            efficient_mask = is_pareto_efficient(pareto_costs)
            pareto_df = df[efficient_mask].sort_values(by='target_rmse')
            
            # Re-calculate best trial logic
            r_min, r_max = pareto_df['target_rmse'].min(), pareto_df['target_rmse'].max()
            c_penalty = 1.0 - pareto_df['c_index'].values
            cp_min, cp_max = c_penalty.min(), c_penalty.max()
            r_range = (r_max - r_min) if (r_max - r_min) > 1e-6 else 1.0
            cp_range = (cp_max - cp_min) if (cp_max - cp_min) > 1e-6 else 1.0
            pareto_df['norm_rmse'] = (pareto_df['target_rmse'] - r_min) / r_range
            pareto_df['norm_cp'] = (c_penalty - cp_min) / cp_range
            pareto_df['score'] = 0.3 * pareto_df['norm_rmse'] + 0.7 * pareto_df['norm_cp']
            best_trial = pareto_df.loc[pareto_df['score'].idxmin()]
            
            # Plot
            h1 = ax.scatter(df['target_rmse'], df['c_index'], c='silver', alpha=0.3, s=20, label='All Trials')
            h2 = ax.scatter(pareto_df['target_rmse'], pareto_df['c_index'], c='red', alpha=0.8, s=30, label='Pareto Frontier')
            h3 = ax.scatter(best_trial['target_rmse'], best_trial['c_index'], 
                        marker='*', s=200, c='gold', edgecolors='black', 
                        label='Selected Balanced Configuration', zorder=10)
            
            if len(legend_handles) == 0:
                legend_handles = [h1, h2, h3]
                legend_labels = ['All Trials', 'Pareto Frontier', 'Selected Balanced Configuration']
            
            # Subplot titles
            ax.set_title(f"{ds.upper()} - {sc.capitalize()}", fontsize=12, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)
            
    # Add shared axis labels
    fig.supxlabel('RMSE (Lower is Better)', fontsize=16, y=0.1)
    fig.supylabel('C-index (Higher is Better)', fontsize=16, x=0.05)
    
    # Add a single common legend
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=3, 
               frameon=True, fontsize=12, bbox_to_anchor=(0.5, 0.02))
    
    combined_path = os.path.join(OUTPUT_DIR, "pareto_frontiers_combined.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated combined Pareto plot with shared axis labels: {combined_path}")

if __name__ == "__main__":
    process_and_plot()
