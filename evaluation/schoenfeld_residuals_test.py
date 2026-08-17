"""
schoenfeld_residuals_test.py
Tests the proportional hazards assumption using Schoenfeld residuals
on METABRIC and MIMIC-IV datasets.
"""
import os
import numpy as np
import pandas as pd
from pycox.datasets import metabric
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

SVAE_RESULTS_DIR = os.environ.get('SVAE_RESULTS_DIR', '../datasets')
SVAE_OUTPUT_DIR = os.environ.get('SVAE_OUTPUT_DIR', '../output')

def run_schoenfeld_test(df, dataset_name):
    print(f"\n{'='*60}")
    print(f"Schoenfeld Residuals Test: {dataset_name}")
    print(f"{'='*60}")
    os.makedirs(SVAE_OUTPUT_DIR, exist_ok=True)
    feature_cols = [col for col in df.columns if col not in ['duration', 'event']]
    df = df[feature_cols + ['duration', 'event']].copy()
    df = df.select_dtypes(include=[np.number])
    feature_cols = [col for col in df.columns if col not in ['duration', 'event']]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    print(f"Fitting CoxPH model on {len(feature_cols)} features...")
    cph = CoxPHFitter(penalizer=0.01)
    try:
        cph.fit(df, duration_col='duration', event_col='event')
    except Exception as e:
        print(f"Convergence failed: {e}. Retrying with penalizer=0.1...")
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df, duration_col='duration', event_col='event')
    print("Running Schoenfeld residuals test (time_transform='rank')...")
    results = proportional_hazard_test(cph, df, time_transform='rank')
    summary_df = results.summary
    n_total = len(summary_df)
    n_violated = (summary_df['p'] <= 0.05).sum()
    n_holds = n_total - n_violated
    print(f"\nGlobal summary: {n_holds}/{n_total} covariates satisfy PH (p > 0.05)")
    print(f"               {n_violated}/{n_total} covariates violate PH (p <= 0.05)")
    print(f"\n{'Feature':<30} | {'Chi-square':<12} | {'p-value':<12} | {'Result'}")
    print("-" * 75)
    for feature, row in summary_df.iterrows():
        chi2 = row['test_statistic']
        p_val = row['p']
        status = "PH holds" if p_val > 0.05 else "PH VIOLATED"
        print(f"{str(feature):<30} | {chi2:<12.4f} | {p_val:<12.4e} | {status}")
    violated_covariates = summary_df[summary_df['p'] <= 0.05].index.tolist()
    if violated_covariates:
        print(f"\nCovariates violating PH: {', '.join(map(str, violated_covariates))}")
    else:
        print("\nNo covariates violate PH at p <= 0.05.")
    output_path = os.path.join(SVAE_OUTPUT_DIR,
                               f'schoenfeld_results_{dataset_name.lower()}.csv')
    summary_df.to_csv(output_path)
    print(f"\nResults saved to: {output_path}")
    return summary_df, n_violated, n_total, violated_covariates

def main():
    print("Loading METABRIC dataset...")
    df_metabric = metabric.read_df()
    # Map pycox generic feature names to actual METABRIC clinical names
    metabric_feature_map = {
        'x0': 'MKI67', 'x1': 'EGFR', 'x2': 'PGR', 'x3': 'ERBB2',
        'x4': 'Hormone_Tx', 'x5': 'Radiotherapy', 'x6': 'Chemotherapy',
        'x7': 'ER_Positive', 'x8': 'Age'
    }
    df_metabric = df_metabric.rename(columns=metabric_feature_map)
    run_schoenfeld_test(df_metabric, "METABRIC")
    mimic_path = os.path.join(SVAE_RESULTS_DIR, 'mimic_sepsis_highdim.csv')
    print(f"\nLoading MIMIC-IV dataset from {mimic_path}...")
    if not os.path.exists(mimic_path):
        print(f"Warning: MIMIC-IV dataset not found at {mimic_path}. Skipping.")
    else:
        df_mimic = pd.read_csv(mimic_path)
        cols_to_drop = ['subject_id', 'hadm_id', 'stay_id',
                        'competing_event', 'cause_specific_event_1']
        df_mimic = df_mimic.drop(
            columns=[c for c in cols_to_drop if c in df_mimic.columns])
        run_schoenfeld_test(df_mimic, "MIMIC")

if __name__ == "__main__":
    main()
