"""
Aggregate Survival Model Results Across 5 Folds
================================================
Reads per-fold CSVs from all 5 folds and produces:
1. Main table: Mean +/- SD per dataset x scenario x method x model
2. AUC(t) table: Mean (95% CI) averaged across scenarios per dataset x method x model

Fully aligned with manuscript reporting format.
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

# Configuration
POSSIBLE_DATA_DIRS = [
    '/Users/nazu.ds/Documents/Research Collections/Research 2025/survival-vae',
    './datasets', 'datasets', '../datasets',
    '/Users/nazu.ds/Documents/Research Collections/Research 2025/',
    '/home/azman/VAE/Survival-VAE_study',
    '/Users/azmannads/VAE/Survival-VAE_study',
    '/Users/azmannads/Documents/Research collections/Research 2025/datasets',
    '.'
]

def find_dir(possibilities, default_name):
    for p in possibilities:
        if os.path.exists(p):
            return p
    return default_name

DATA_DIR = find_dir(POSSIBLE_DATA_DIRS, 'datasets')

DATASETS = ['METABRIC', 'MIMIC']
SCENARIOS = ['light', 'moderate', 'severe']
METHODS = ['survival_vae', 'standard_vae', 'gain', 'mida', 'mice', 'missforest']
MODELS = ['rsf', 'xgb', 'deepsurv', 'deephit']


def calculate_mean_sd(values):
    """Mean +/- SD across folds."""
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return np.nan, np.nan
    return np.mean(arr), np.std(arr)


def calculate_mean_95ci(values):
    """Mean with 95% CI using t-distribution (df=n-1=4 for n=5 folds)."""
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    mean = np.mean(arr)
    sem = stats.sem(arr)
    # t_{0.025, 4} = 2.776 for 5 folds
    ci = sem * 2.776
    return mean, mean - ci, mean + ci


def load_all_folds():
    """Load per-fold result CSVs from all 5 folds and concatenate."""
    all_dfs = []
    for fold_idx in range(5):
        fold_path = os.path.join(
            DATA_DIR, 'final', 'survival_results',
            f'fold_{fold_idx}', 'results_survival_fold.csv'
        )
        if os.path.exists(fold_path):
            df = pd.read_csv(fold_path)
            all_dfs.append(df)
            print(f"  Loaded fold {fold_idx}: {len(df)} rows")
        else:
            print(f"  [WARNING]  Fold {fold_idx} not found: {fold_path}")

    if not all_dfs:
        raise FileNotFoundError("No fold result files found.")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  Total rows loaded: {len(combined)}")
    print(f"  Folds present: {sorted(combined['Fold'].unique())}")
    return combined


def generate_main_table(df, output_path):
    """
    Main results table: Mean +/- SD across 5 folds.
    Grouped by Dataset x Scenario x Method x Model.
    Metrics: C-index, IBS, iAUC.
    """
    rows = []
    for (dataset, scenario, method, model), g in df.groupby(
        ['Dataset', 'Scenario', 'Method', 'Model']
    ):
        c_mean, c_sd = calculate_mean_sd(g['C_Index'])
        ibs_mean, ibs_sd = calculate_mean_sd(g['IBS'])
        ia_mean, ia_sd = calculate_mean_sd(g['iAUC'])

        rows.append({
            'Dataset': dataset,
            'Scenario': scenario,
            'Method': method,
            'Model': model,
            'C_Index': f"{c_mean:.4f} ({c_sd:.4f})" if not np.isnan(c_mean) else 'N/A',
            'C_Index_Mean': round(c_mean, 4),
            'C_Index_SD': round(c_sd, 4),
            'IBS': f"{ibs_mean:.4f} ({ibs_sd:.4f})" if not np.isnan(ibs_mean) else 'N/A',
            'IBS_Mean': round(ibs_mean, 4),
            'IBS_SD': round(ibs_sd, 4),
            'iAUC': f"{ia_mean:.4f} ({ia_sd:.4f})" if not np.isnan(ia_mean) else 'N/A',
            'iAUC_Mean': round(ia_mean, 4),
            'iAUC_SD': round(ia_sd, 4),
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)
    print(f"[Done] Main table saved: {output_path}")
    return result_df


def generate_auc_tmed_table(df, output_path):
    """
    AUC(t) at t_med table: Mean (95% CI) averaged across scenarios.
    Per manuscript: mean with 95% CI averaging across Light/Moderate/Severe.
    Grouped by Dataset x Method x Model.
    """
    rows = []

    for (dataset, method, model), g in df.groupby(['Dataset', 'Method', 'Model']):
        # Average AUC_tmed across all scenarios and folds
        auc_vals = g['AUC_tmed'].dropna().values
        mean, lo, hi = calculate_mean_95ci(auc_vals)

        rows.append({
            'Dataset': dataset,
            'Method': method,
            'Model': model,
            'AUC_tmed_Mean': round(mean, 4) if not np.isnan(mean) else np.nan,
            'AUC_tmed_95CI_Lo': round(lo, 4) if not np.isnan(lo) else np.nan,
            'AUC_tmed_95CI_Hi': round(hi, 4) if not np.isnan(hi) else np.nan,
            'AUC_tmed': f"{mean:.4f} ({lo:.4f}--{hi:.4f})" if not np.isnan(mean) else 'N/A',
            'N_values': len(auc_vals),
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)
    print(f"[Done] AUC(t) table saved: {output_path}")
    return result_df


def generate_latex_main_table(main_df, dataset_name, output_path):
    """
    Generate LaTeX-ready string for the main C-index/IBS table.
    Matches manuscript table format.
    """
    ds_df = main_df[main_df['Dataset'] == dataset_name.upper()]
    if ds_df.empty:
        print(f"  No data for {dataset_name}")
        return

    lines = []
    lines.append(f"% {dataset_name.upper()} Results Table")
    lines.append("\\begin{tabular}{lllcccccc}")
    lines.append("\\hline")
    lines.append("\\textbf{Scenario} & \\textbf{Model} & \\textbf{Metric} & "
                 "\\textbf{Survival-VAE} & \\textbf{Standard VAE} & "
                 "\\textbf{MICE} & \\textbf{missForest} & "
                 "\\textbf{GAIN} & \\textbf{MIDA} \\\\")
    lines.append("\\hline")

    method_order = ['survival_vae', 'standard_vae', 'mice', 'missforest', 'gain', 'mida']
    model_display = {'rsf': 'RSF', 'xgb': 'XGBoost', 'deepsurv': 'DeepSurv', 'deephit': 'DeepHit'}

    for scenario in SCENARIOS:
        sc_df = ds_df[ds_df['Scenario'] == scenario]
        first_scenario = True

        for model in MODELS:
            mod_df = sc_df[sc_df['Model'] == model]
            if mod_df.empty:
                continue

            for metric, col_mean, col_sd in [('C-index', 'C_Index_Mean', 'C_Index_SD'),
                                              ('IBS', 'IBS_Mean', 'IBS_SD')]:
                row_parts = []
                best_val = None
                best_idx = -1

                # Find best value
                vals = []
                for m in method_order:
                    m_row = mod_df[mod_df['Method'] == m]
                    if m_row.empty:
                        vals.append(np.nan)
                    else:
                        vals.append(m_row[col_mean].values[0])

                if np.isnan(vals).all():
                    best_idx = -1
                else:
                    if metric == 'C-index':
                        best_idx = int(np.nanargmax(vals))
                    else:
                        best_idx = int(np.nanargmin(vals))

                for i, m in enumerate(method_order):
                    m_row = mod_df[mod_df['Method'] == m]
                    if m_row.empty:
                        row_parts.append('--')
                    else:
                        mean_v = m_row[col_mean].values[0]
                        sd_v = m_row[col_sd].values[0]
                        if np.isnan(mean_v):
                            cell = '--'
                        else:
                            cell = f"{mean_v:.3f} ({sd_v:.3f})"
                            if i == best_idx:
                                cell = f"\\textbf{{{cell}}}"
                        row_parts.append(cell)

                sc_label = scenario.capitalize() if first_scenario and metric == 'C-index' else ''
                mod_label = f"\\multirow{{2}}{{*}}{{{model_display[model]}}}" if metric == 'C-index' else ''
                first_scenario = False

                line = f"{sc_label} & {mod_label} & {metric} & " + " & ".join(row_parts) + " \\\\"
                lines.append(line)

            lines.append("\\cmidrule{1-9}")

    lines.append("\\hline")
    lines.append("\\end{tabular}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"[Done] LaTeX table saved: {output_path}")


def print_summary(main_df):
    """Print a quick summary to terminal."""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    for dataset in main_df['Dataset'].unique():
        print(f"\n{dataset}")
        ds_df = main_df[main_df['Dataset'] == dataset]

        for scenario in SCENARIOS:
            sc_df = ds_df[ds_df['Scenario'] == scenario]
            if sc_df.empty:
                continue
            print(f"\n  {scenario.capitalize()}:")

            for model in MODELS:
                mod_df = sc_df[sc_df['Model'] == model]
                if mod_df.empty:
                    continue
                print(f"    {model.upper():10s}", end='')
                for method in ['survival_vae', 'standard_vae', 'mice',
                               'missforest', 'gain', 'mida']:
                    m_row = mod_df[mod_df['Method'] == method]
                    if m_row.empty:
                        print(f"  {'--':>12}", end='')
                    else:
                        c = m_row['C_Index_Mean'].values[0]
                        print(f"  {c:.4f}", end='')
                print()


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate survival model results across 5 folds.'
    )
    parser.add_argument('--latex', action='store_true',
                        help='Also generate LaTeX tables')
    args = parser.parse_args()

    output_dir = os.path.join(DATA_DIR, 'final', 'survival_results', 'aggregated')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from: {DATA_DIR}")
    print(f"Output dir: {output_dir}\n")

    # Load all fold results
    print("Loading per-fold results...")
    df = load_all_folds()

    # Verify coverage
    n_folds = df['Fold'].nunique()
    print(f"\n  Datasets: {df['Dataset'].unique()}")
    print(f"  Scenarios: {df['Scenario'].unique()}")
    print(f"  Methods: {df['Method'].unique()}")
    print(f"  Models: {df['Model'].unique()}")
    print(f"  Folds: {n_folds}")

    if n_folds < 5:
        print(f"\n  [WARNING]  Warning: Only {n_folds}/5 folds available. "
              f"Results will be partial.")

    # Generate main table (C-index, IBS, iAUC - mean +/- SD)
    print("\nGenerating main results table...")
    main_path = os.path.join(output_dir, 'results_main_table.csv')
    main_df = generate_main_table(df, main_path)

    # Generate AUC(t) table (mean 95% CI averaged across scenarios)
    print("\nGenerating AUC(t) at t_med table...")
    auc_path = os.path.join(output_dir, 'results_auc_tmed_table.csv')
    generate_auc_tmed_table(df, auc_path)

    # Save raw combined data
    raw_path = os.path.join(output_dir, 'results_all_folds_combined.csv')
    df.to_csv(raw_path, index=False)
    print(f"[Done] Raw combined data saved: {raw_path}")

    # LaTeX tables
    if args.latex:
        print("\nGenerating LaTeX tables...")
        for ds in ['metabric', 'mimic']:
            latex_path = os.path.join(output_dir, f'latex_table_{ds}.tex')
            generate_latex_main_table(main_df, ds, latex_path)

    # Print terminal summary
    print_summary(main_df)

    print(f"\n{'='*60}")
    print(f"Aggregation complete. Files saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
