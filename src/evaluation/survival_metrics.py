"""
Survival Metrics Table Generation (Portable Version)
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

def generate_latex_table():
    POSSIBLE_DATA_DIRS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')),
        '/Users/azmannads/Documents/Research collections/Research 2025/datasets',
        '/Users/azmannads/VAE/datasets',
    ]
    DATA_DIR = next((p for p in POSSIBLE_DATA_DIRS if os.path.exists(p)), POSSIBLE_DATA_DIRS[0])
    
    results_path = os.path.join(DATA_DIR, 'results_ml_level2.csv')
    if not os.path.exists(results_path):
        print(f"Results not found at {results_path}")
        return
        
    df = pd.read_csv(results_path)
    method_order = ['survival_vae', 'mice', 'missforest', 'gain', 'mida']
    method_labels = {'survival_vae': 'S-VAE', 'mice': 'MICE', 'missforest': 'missForest', 'gain': 'GAIN', 'mida': 'MIDA'}
    
    print("\\begin{table}[ht]\n\\centering\n\\caption{Time-Dependent AUC (t-d AUC) at Median Survival Time. Values are Mean (95\\% CI).}\n\\label{tab:td_auc}\n\\resizebox{\\textwidth}{!}{%}\n\\begin{tabular}{llcccc}\n\\hline\n\\textbf{Dataset} & \\textbf{Method} & \\textbf{RSF} & \\textbf{XGB} & \\textbf{DeepSurv} & \\textbf{DeepHit} \\\\\n\\hline")
    
    for dataset in ['metabric', 'mimic']:
        dataset_label = "METABRIC" if dataset == 'metabric' else "MIMIC-IV"
        print(f"\\multirow{{5}}{{*}}{{{dataset_label}}}")
        for method in method_order:
            row_str = f" & {method_labels[method]}"
            for model in ['rsf', 'xgb', 'ds', 'dh']:
                subset = df[(df['Dataset'].str.lower() == dataset) & (df['Method'] == method) & (df['Model'] == model) & (df['Scenario'] == 'severe')]
                if subset.empty:
                    val_str = "N/A"
                else:
                    vals = subset['AUC'].values
                    mean = np.mean(vals)
                    ci = stats.sem(vals) * stats.t.ppf(0.975, len(vals)-1)
                    val_str = f"{mean:.3f} ({mean-ci:.3f}--{mean+ci:.3f})"
                row_str += f" & {val_str}"
            print(row_str + " \\\\")
        print("\\hline")
    print("\\end{tabular}%\n}\n\\end{table}")

if __name__ == "__main__":
    generate_latex_table()
