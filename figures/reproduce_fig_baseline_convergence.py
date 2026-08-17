import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import numpy as np

# Configuration
DATA_DIR = os.environ.get('SVAE_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'final'))
OPTUNA_DIR = os.path.join(DATA_DIR, 'optuna_results_final', 'fold_0')
OUTPUT_DIR = os.environ.get('SVAE_OUTPUT_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

METHODS = ['mice', 'missforest', 'gain', 'mida', 'standard_vae']
DATASETS = ['metabric', 'mimic']
SCENARIOS = ['light', 'moderate', 'severe']

METHOD_NAMES = {
    'mice': 'MICE',
    'missforest': 'missForest',
    'gain': 'GAIN',
    'mida': 'MIDA',
    'standard_vae': 'Standard VAE'
}

def process_baselines_split():
    # Split by dataset to generate two separate 3x5 grids
    for ds in DATASETS:
        ds_display = "METABRIC" if ds == 'metabric' else "MIMIC-IV"
        fig, axes = plt.subplots(3, 5, figsize=(18, 12))
        plt.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.15, top=0.90)
        
        legend_handles = []
        legend_labels = []
        
        for row_idx, sc in enumerate(SCENARIOS):
            sc_label = sc.capitalize()
            for col_idx, method in enumerate(METHODS):
                ax = axes[row_idx, col_idx]
                
                filename = f"{ds}_{sc}_optuna_{method}_results.csv"
                path = os.path.join(OPTUNA_DIR, filename)
                
                if not os.path.exists(path):
                    ax.set_visible(False)
                    continue
                
                df = pd.read_csv(path)
                df_sorted = df.sort_values(by='trial')
                trials = df_sorted['trial'].values
                rmse_vals = df_sorted['rmse_norm'].values
                running_min = np.minimum.accumulate(rmse_vals)
                
                h1 = ax.plot(trials, rmse_vals, 'o', color='silver', markersize=3, alpha=0.4, label='Trial RMSE')[0]
                h2 = ax.plot(trials, running_min, 'r-', linewidth=2, label='Best RMSE')[0]
                
                if len(legend_handles) == 0:
                    legend_handles = [h1, h2]
                    legend_labels = ['Trial RMSE', 'Running Best RMSE']
                
                # Title only for top row
                if row_idx == 0:
                    ax.set_title(METHOD_NAMES[method], fontsize=14, fontweight='bold', pad=15)
                
                # Row label (scenario) for first column
                if col_idx == 0:
                    ax.set_ylabel(f"{sc_label}", fontsize=12, fontweight='bold', labelpad=10)
                
                # 3 decimal places on y-axis
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
                
                ax.set_xlabel('Trial' if row_idx == 2 else '', fontsize=10)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.tick_params(labelsize=9)
        
        # Shared legend for the figure
        fig.legend(legend_handles, legend_labels, loc='lower center', ncol=2, 
                   frameon=True, fontsize=14, bbox_to_anchor=(0.5, 0.05))
        
        # Master title for the figure
        fig.suptitle(f"Optimization Convergence Trajectories: {ds_display}", fontsize=18, fontweight='bold', y=0.96)
        
        output_filename = f"baseline_convergence_{ds}.png"
        save_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated split convergence plot: {save_path}")

if __name__ == "__main__":
    process_baselines_split()
