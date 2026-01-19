import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats
import os

# Plotting Configuration
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Arial',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3,
})

def validate_data_for_plotting(df):
    """Verify data structure is correct"""
    # Check for required columns
    required_cols_wide = ['Dataset', 'Scenario', 'Method', 'Fold', 'RMSE', 'MAE']
    required_cols_ml = ['Dataset', 'Scenario', 'Method', 'C_Index', 'IBS']
    
    if all(col in df.columns for col in required_cols_wide):
        print("Validating Imputation Folds Data (Wide Format)...")
    elif all(col in df.columns for col in required_cols_ml):
        print("Validating Clinical Utility Data...")
    else:
        print("⚠️  WARNING: partial or unknown column structure. Columns:", df.columns.tolist())
    
    scenarios = df['Scenario'].unique()
    methods = df['Method'].unique()
    datasets = df['Dataset'].unique()
    
    print("Data validation:")
    print(f"  Datasets: {datasets}")
    print(f"  Scenarios: {scenarios}")
    print(f"  Methods: {methods}")
    print("✓ Validation complete\n")

def get_scenario_labels(dataset):
    """Return labels with exact percentages for the dataset"""
    if dataset == 'METABRIC':
        return ['Light\n(15.73%)', 'Moderate\n(31.30%)', 'Severe\n(51.07%)']
    elif dataset == 'MIMIC':
        return ['Light\n(40.10%)', 'Moderate\n(54.86%)', 'Severe\n(68.36%)']
    else:
        return ['Light', 'Moderate', 'Severe']

def plot_imputation_fidelity_corrected(imp_df, filename="elsarticle/figure_imputation_fidelity.png"):
    """Create corrected grouped bar plots for imputation metrics"""
    
    # Setup
    fig, axes = plt.subplots(2, 2, figsize=(16, 12)) # increased height slightly
    scenarios = ['light', 'moderate', 'severe'] 
    methods = ['survival_vae', 'missforest', 'mida', 'gain', 'mice']
    method_labels = ['S-VAE', 'missForest', 'MIDA', 'GAIN', 'MICE']
    
    # Colors (Wong colorblind-safe palette)
    colors = {
        'survival_vae': '#E69F00',   # Orange
        'missforest': '#56B4E9',      # Sky blue  
        'mida': '#009E73',            # Bluish green
        'gain': '#F0E442',            # Yellow
        'mice': '#0072B2'             # Blue
    }
    
    patterns = {
        'survival_vae': '///',
        'missforest': '...',
        'mida': '',  # Solid
        'gain': '|||',
        'mice': '---'
    }
    
    # Process each subplot
    plot_configs = [
        (axes[0, 0], 'METABRIC', 'RMSE', 'METABRIC: RMSE'),
        (axes[0, 1], 'MIMIC', 'RMSE', 'MIMIC-IV: RMSE'),
        (axes[1, 0], 'METABRIC', 'MAE', 'METABRIC: MAE'),
        (axes[1, 1], 'MIMIC', 'MAE', 'MIMIC-IV: MAE')
    ]
    
    for ax, dataset, metric, title in plot_configs:
        data = imp_df[(imp_df['Dataset'] == dataset)].copy()
        
        # Setup x positions
        x = np.arange(len(scenarios)) 
        width = 0.15 
        
        # Plot each method
        for i, method in enumerate(methods):
            method_data = data[data['Method'] == method]
            
            means = []
            stds = []
            for scenario in scenarios:
                scenario_data = method_data[method_data['Scenario'] == scenario]
                if len(scenario_data) > 0:
                    val_col = metric
                    means.append(scenario_data[val_col].mean())
                    stds.append(scenario_data[val_col].std())
                else:
                    means.append(0)
                    stds.append(0)
            
            bars = ax.bar(x + i * width, 
                         means, 
                         width,
                         yerr=stds,
                         capsize=3,
                         label=method_labels[i],
                         color=colors[method],
                         hatch=patterns[method],
                         edgecolor='black',
                         linewidth=1.2,
                         error_kw={'linewidth': 1.5, 'elinewidth': 1.5})
            
            # Significance Stars
            if method != 'survival_vae':
                 svae_data = data[data['Method'] == 'survival_vae']
                 for s_idx, scenario in enumerate(scenarios):
                     svae_vals = svae_data[svae_data['Scenario'] == scenario][metric].values
                     curr_vals = method_data[method_data['Scenario'] == scenario][metric].values
                     
                     if len(svae_vals) == 5 and len(curr_vals) == 5:
                         t_stat, p_val = stats.ttest_rel(svae_vals, curr_vals)
                         if p_val < 0.05:
                             bar_x = x[s_idx] + i * width
                             bar_h = means[s_idx] + stds[s_idx]
                             mark = '*'
                             if p_val < 0.001: mark = '***'
                             elif p_val < 0.01: mark = '**'
                             
                             y_limit = ax.get_ylim()[1]
                             ax.text(bar_x, bar_h + y_limit*0.02, mark, 
                                     ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)

        # Formatting
        ax.set_ylabel(f'{metric} (Lower is Better)', fontsize=14)
        ax.set_title(title, fontweight='bold', fontsize=16)
        
        # Center ticks
        ax.set_xticks(x + 2 * width)  
        ax.set_xticklabels(get_scenario_labels(dataset)) # Use specific labels
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Legend for Imputation Fidelity: Same top-center style as Clinical Utility
        # Only create legend once
        if ax == axes[0, 0]:
            handles, labels = ax.get_legend_handles_labels()
            # We don't want a legend inside the axes anymore, so we skip ax.legend() here
            # Instead we add it to the figure at the end
            
    # Add single figure-level legend
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=5, fontsize=12, frameon=True)
    
    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved corrected figure: {filename}")

def plot_clinical_utility_corrected(ml_df, filename="elsarticle/figure_clinical_utility.png"):
    """Create corrected grouped bar plots for clinical metrics"""
    
    # Setup
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    scenarios = ['light', 'moderate', 'severe']
    methods = ['survival_vae', 'missforest', 'mida', 'gain', 'mice']
    method_labels = ['S-VAE', 'missForest', 'MIDA', 'GAIN', 'MICE']
    
    # Colors
    colors = {
        'survival_vae': '#E69F00',
        'missforest': '#56B4E9',
        'mida': '#009E73',
        'gain': '#F0E442',
        'mice': '#0072B2'
    }
    
    patterns = {
        'survival_vae': '///',
        'missforest': '...',
        'mida': '',
        'gain': '|||',
        'mice': '---'
    }
    
    name_map = {'survival_vae': 'survival_vae', 'mice': 'mice', 
                'missforest': 'missforest', 'gain': 'gain', 'mida': 'mida'}
    
    # Process each subplot
    plot_configs = [
        (axes[0, 0], 'metabric', 'C_Index', 'METABRIC: C-index', 'Higher'), 
        (axes[0, 1], 'mimic', 'C_Index', 'MIMIC-IV: C-index', 'Higher'),
        (axes[1, 0], 'metabric', 'IBS', 'METABRIC: IBS', 'Lower'),
        (axes[1, 1], 'mimic', 'IBS', 'MIMIC-IV: IBS', 'Lower')
    ]
    
    for ax, dataset, metric, title, direction in plot_configs:
        data = ml_df[(ml_df['Dataset'].str.lower() == dataset) &
                     (ml_df['Model'] == 'rsf')]
        
        x = np.arange(len(scenarios))
        width = 0.15
        
        # Plot each method
        for i, method in enumerate(methods):
            method_data = data[data['Method'] == method]
            
            means = []
            stds = [] 
            for scenario in scenarios:
                scenario_data = method_data[method_data['Scenario'] == scenario]
                if len(scenario_data) > 0:
                    val_col = metric
                    means.append(scenario_data[val_col].mean())
                    stds.append(scenario_data[val_col].std())
                else:
                    means.append(0)
                    stds.append(0)
            
            bars = ax.bar(x + i * width,
                         means,
                         width,
                         yerr=stds,
                         capsize=3,
                         label=method_labels[i],
                         color=colors[method],
                         hatch=patterns[method],
                         edgecolor='black',
                         linewidth=1.2,
                         error_kw={'linewidth': 1.5, 'elinewidth': 1.5})
            
            # Use strict y-limits for C-Index
            # Use strict y-limits for C-Index
            if metric == 'C_Index':
                if dataset == 'metabric':
                    ax.set_ylim(0.5, 0.67) # Requested limit
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                elif dataset == 'mimic':
                    ax.set_ylim(0.60, 0.90) # Requested limit
            
            # Significance Stars
            if method != 'survival_vae':
                 svae_data = data[data['Method'] == 'survival_vae']
                 for s_idx, scenario in enumerate(scenarios):
                     svae_vals = svae_data[svae_data['Scenario'] == scenario][metric].values
                     curr_vals = method_data[method_data['Scenario'] == scenario][metric].values
                     
                     if len(svae_vals) >= 2 and len(curr_vals) >= 2:
                         t_stat, p_val = stats.ttest_rel(svae_vals, curr_vals)
                         if p_val < 0.05:
                             bar_x = x[s_idx] + i * width
                             bar_h = means[s_idx] + stds[s_idx]
                             mark = '*'
                             if p_val < 0.001: mark = '***'
                             elif p_val < 0.01: mark = '**'
                             
                             y_limit = ax.get_ylim()[1]
                             y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
                             # Place slightly above bar
                             ax.text(bar_x, bar_h + y_span*0.02, mark, 
                                     ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)

        # Formatting
        ylabel = f'{metric.replace("_", "-")} ({direction} is Better)'
        if metric == 'C_Index': ylabel = f'C-index ({direction} is Better)' # explicit fix
        
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontweight='bold', fontsize=16)
        ax.set_xticks(x + 2 * width)
        ax.set_xticklabels(get_scenario_labels(dataset.upper())) # Use specific labels
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        
    # Legend for Clinical Utility: "Move legend upper and make it one line"
    # Create single legend for the whole figure at the top
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=5, fontsize=12, frameon=True)
    
    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved corrected figure: {filename}")

def plot_auc_corrected(ml_df, filename="elsarticle/figure_auc_time_dependent.png"):
    """Create corrected grouped bar plots for Time-Dependent AUC comparing methods"""
    
    # Setup
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    scenarios = ['light', 'moderate', 'severe']
    methods = ['survival_vae', 'missforest', 'mida', 'gain', 'mice']
    method_labels = ['S-VAE', 'missForest', 'MIDA', 'GAIN', 'MICE']
    
    colors = {
        'survival_vae': '#E69F00',
        'missforest': '#56B4E9',
        'mida': '#009E73',
        'gain': '#F0E442',
        'mice': '#0072B2'
    }
    
    patterns = {
        'survival_vae': '///',
        'missforest': '...',
        'mida': '',
        'gain': '|||',
        'mice': '---'
    }
    
    name_map = {'survival_vae': 'survival_vae', 'mice': 'mice', 
                'missforest': 'missforest', 'gain': 'gain', 'mida': 'mida'}

    # Filter for XGBoost which generally performs best, consistent with table
    ml_df = ml_df[ml_df['Model'].isin(['xgb', 'XGBoost'])].copy()
    
    plot_configs = [
        (axes[0], 'metabric', 'AUC', 'METABRIC: Time-Dependent AUC (XGBoost)', 'METABRIC'),
        (axes[1], 'mimic', 'AUC', 'MIMIC-IV: Time-Dependent AUC (XGBoost)', 'MIMIC')
    ]
    
    for ax, dataset, metric, title, dataset_display in plot_configs:
        data = ml_df[ml_df['Dataset'] == dataset].copy()
        
        x = np.arange(len(scenarios)) 
        width = 0.15 
        
        for i, method in enumerate(methods):
            method_key = name_map.get(method, method)
            method_data = data[data['Method'] == method_key]
            
            means = []
            stds = []
            for scenario in scenarios:
                scenario_data = method_data[method_data['Scenario'] == scenario]
                if len(scenario_data) > 0:
                    means.append(scenario_data[metric].mean())
                    stds.append(scenario_data[metric].std())
                else:
                    means.append(0)
                    stds.append(0)
            
            # Plot bars
            bars = ax.bar(x + i * width, means, width, 
                         yerr=stds, 
                         capsize=3,
                         label=method_labels[i], 
                         color=colors[method], 
                         hatch=patterns[method],
                         edgecolor='black', 
                         linewidth=1.2, 
                         error_kw={'linewidth': 1.5, 'elinewidth': 1.5})
            
            # Add significance stars (IMPROVEMENT 1)
            if method != 'survival_vae':
                svae_data = data[data['Method'] == 'survival_vae']
                for s_idx, scenario in enumerate(scenarios):
                    svae_vals = svae_data[svae_data['Scenario'] == scenario][metric].values
                    curr_vals = method_data[method_data['Scenario'] == scenario][metric].values
                    
                    if len(svae_vals) >= 2 and len(curr_vals) >= 2:
                        t_stat, p_val = stats.ttest_rel(svae_vals, curr_vals)
                        if p_val < 0.05:
                            bar_x = x[s_idx] + i * width
                            bar_h = means[s_idx] + stds[s_idx]
                            mark = '*'
                            if p_val < 0.001: mark = '***'
                            elif p_val < 0.01: mark = '**'
                            
                            y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
                            ax.text(bar_x, bar_h + y_span*0.02, mark,
                                   ha='center', va='bottom', fontsize=8, 
                                   fontweight='bold', rotation=90)

        # IMPROVEMENT 2: Better Y-axis range
        if dataset == 'metabric':
            ax.set_ylim(0.60, 0.68)  # Tighter range shows detail
        else:  # mimic
            ax.set_ylim(0.66, 0.72)  # Adjusted for MIMIC data
        
        # IMPROVEMENT 3: Clearer axis labels
        ax.set_ylabel('AUC(t) at Median Survival Time\n(Higher is Better)', fontsize=14)
        ax.set_title(title, fontweight='bold', fontsize=16)
        ax.set_xlabel('MNAR Severity', fontsize=14, fontweight='bold')
        
        ax.set_xticks(x + 2 * width)
        ax.set_xticklabels(get_scenario_labels(dataset_display))
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        
    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=5, fontsize=12, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved improved AUC figure (bar plot): {filename}")

if __name__ == "__main__":
    print("Generating Corrected Performance Plots...")
    
    # Load fold-level imputation data
    imp_fold_df = pd.read_csv('datasets/results_imputation_folds.csv')
    validate_data_for_plotting(imp_fold_df)
    plot_imputation_fidelity_corrected(imp_fold_df)
    
    # Load clinical utility data
    ml_df = pd.read_csv('datasets/results_ml_level2.csv')
    plot_clinical_utility_corrected(ml_df)
    plot_auc_corrected(ml_df)
