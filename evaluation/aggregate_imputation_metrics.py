"""
Aggregate Imputation Metrics Across 5 Folds
============================================
Reads per-fold RMSE/MAE CSVs from all 5 folds and produces:
1. Main table: Mean +/- SD per dataset x scenario x method
2. Fully aligned with manuscript Table format (imputation fidelity)
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

# Configuration
DATA_DIR = os.environ.get('SVAE_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets'))

DATASETS = ['METABRIC', 'MIMIC']
SCENARIOS = ['light', 'moderate', 'severe']
METHODS = ['survival_vae', 'standard_vae', 'mice', 'missforest', 'gain', 'mida']


def calculate_mean_sd(values):
    """Mean +/- SD across folds."""
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return np.nan, np.nan
    return np.mean(arr), np.std(arr)


def load_all_folds():
    """Load per-fold imputation metric CSVs from all 5 folds."""
    all_dfs = []
    for fold_idx in range(5):
        fold_path = os.path.join(
            DATA_DIR, 'final', 'imputation_metrics',
            f'fold_{fold_idx}', 'results_imputation_folds.csv'
        )
        if os.path.exists(fold_path):
            df = pd.read_csv(fold_path)
            all_dfs.append(df)
            print(f"  Loaded fold {fold_idx}: {len(df)} rows")
        else:
            print(f"  Warning: Fold {fold_idx} not found: {fold_path}")

    if not all_dfs:
        raise FileNotFoundError("No fold imputation metric files found.")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  Total rows loaded: {len(combined)}")
    print(f"  Folds present: {sorted(combined['Fold'].unique())}")
    return combined


def generate_main_table(df, output_path):
    """
    Main imputation table: Mean +/- SD across 5 folds.
    Grouped by Dataset x Scenario x Method.
    Metrics: RMSE, MAE.
    Matches manuscript Table format.
    """
    rows = []
    for (dataset, scenario, method), g in df.groupby(
        ['Dataset', 'Scenario', 'Method']
    ):
        rmse_mean, rmse_sd = calculate_mean_sd(g['RMSE'])
        mae_mean, mae_sd = calculate_mean_sd(g['MAE'])

        rows.append({
            'Dataset': dataset,
            'Scenario': scenario,
            'Method': method,
            'RMSE': f"{rmse_mean:.4f} ({rmse_sd:.4f})" if not np.isnan(rmse_mean) else 'N/A',
            'RMSE_Mean': round(rmse_mean, 4),
            'RMSE_SD': round(rmse_sd, 4),
            'MAE': f"{mae_mean:.4f} ({mae_sd:.4f})" if not np.isnan(mae_mean) else 'N/A',
            'MAE_Mean': round(mae_mean, 4),
            'MAE_SD': round(mae_sd, 4),
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)
    print(f"Main imputation table saved: {output_path}")
    return result_df


def generate_latex_table(main_df, dataset_name, output_path):
    """
    Generate LaTeX-ready imputation fidelity table.
    Matches manuscript Table format with bold best values.
    """
    ds_df = main_df[main_df['Dataset'] == dataset_name.upper()]
    if ds_df.empty:
        print(f"  No data for {dataset_name}")
        return

    method_order = ['survival_vae', 'standard_vae', 'mice', 'missforest', 'gain', 'mida']
    method_display = {
        'survival_vae': 'Survival-VAE',
        'standard_vae': 'Standard VAE',
        'mice': 'MICE',
        'missforest': 'missForest',
        'gain': 'GAIN',
        'mida': 'MIDA'
    }

    lines = []
    lines.append(f"% {dataset_name.upper()} Imputation Fidelity Table")
    lines.append("\\begin{tabular}{llccccccc}")
    lines.append("\\hline")
    lines.append(
        "\\textbf{Scenario} & \\textbf{Metric} & "
        "\\textbf{Survival-VAE} & \\textbf{Standard VAE} & "
        "\\textbf{MICE} & \\textbf{missForest} & "
        "\\textbf{GAIN} & \\textbf{MIDA} \\\\"
    )
    lines.append("\\hline")

    for scenario in SCENARIOS:
        sc_df = ds_df[ds_df['Scenario'] == scenario]

        for metric, col_mean, col_sd in [
            ('RMSE', 'RMSE_Mean', 'RMSE_SD'),
            ('MAE', 'MAE_Mean', 'MAE_SD')
        ]:
            vals = []
            for m in method_order:
                m_row = sc_df[sc_df['Method'] == m]
                if m_row.empty:
                    vals.append(np.nan)
                else:
                    vals.append(m_row[col_mean].values[0])

            # Find best (lowest RMSE/MAE)
            if np.isnan(vals).all():
                best_idx = -1
            else:
                best_idx = int(np.nanargmin(vals))

            row_parts = []
            for i, m in enumerate(method_order):
                m_row = sc_df[sc_df['Method'] == m]
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

            sc_label = scenario.capitalize() if metric == 'RMSE' else ''
            line = f"{sc_label} & {metric} & " + " & ".join(row_parts) + " \\\\"
            lines.append(line)

        lines.append("\\cmidrule{1-8}")

    lines.append("\\hline")
    lines.append("\\end{tabular}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"LaTeX table saved: {output_path}")


def print_summary(main_df):
    """Print terminal summary."""
    print("\n" + "="*60)
    print("IMPUTATION METRICS SUMMARY (Mean +/- SD)")
    print("="*60)

    for dataset in main_df['Dataset'].unique():
        print(f"\n{dataset}")
        ds_df = main_df[main_df['Dataset'] == dataset]

        for scenario in SCENARIOS:
            sc_df = ds_df[ds_df['Scenario'] == scenario]
            if sc_df.empty:
                continue
            print(f"\n  {scenario.capitalize()}:")
            print(f"  {'Method':<15} {'RMSE':>12} {'MAE':>12}")
            print(f"  {'-'*40}")

            for method in METHODS:
                m_row = sc_df[sc_df['Method'] == method]
                if m_row.empty:
                    print(f"  {method:<15} {'--':>12} {'--':>12}")
                else:
                    rmse = m_row['RMSE_Mean'].values[0]
                    mae = m_row['MAE_Mean'].values[0]
                    print(f"  {method:<15} {rmse:>12.4f} {mae:>12.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate imputation metrics across 5 folds.'
    )
    parser.add_argument('--latex', action='store_true',
                        help='Also generate LaTeX tables')
    args = parser.parse_args()

    output_dir = os.path.join(DATA_DIR, 'final', 'imputation_metrics', 'aggregated')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from: {DATA_DIR}")
    print(f"Output dir: {output_dir}\n")

    # Load all fold results
    print("Loading per-fold imputation metrics...")
    df = load_all_folds()

    # Verify coverage
    n_folds = df['Fold'].nunique()
    print(f"\n  Datasets: {df['Dataset'].unique()}")
    print(f"  Scenarios: {df['Scenario'].unique()}")
    print(f"  Methods: {df['Method'].unique()}")
    print(f"  Folds: {n_folds}")

    if n_folds < 5:
        print(f"\n  Warning: Only {n_folds}/5 folds available. Results will be partial.")

    # Generate main table
    print("\nGenerating main imputation table...")
    main_path = os.path.join(output_dir, 'results_imputation_main_table.csv')
    main_df = generate_main_table(df, main_path)

    # Save raw combined data
    raw_path = os.path.join(output_dir, 'results_imputation_all_folds_combined.csv')
    df.to_csv(raw_path, index=False)
    print(f"Raw combined data saved: {raw_path}")

    # LaTeX tables
    if args.latex:
        print("\nGenerating LaTeX tables...")
        for ds in ['metabric', 'mimic']:
            latex_path = os.path.join(output_dir, f'latex_imputation_table_{ds}.tex')
            generate_latex_table(main_df, ds, latex_path)

    # Print terminal summary
    print_summary(main_df)

    print(f"\n{'='*60}")
    print(f"Aggregation complete. Files saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
