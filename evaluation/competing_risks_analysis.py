"""
competing_risks_analysis.py
Performs a competing risks sensitivity analysis on the MIMIC-IV dataset.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from lifelines import AalenJohansenFitter

SVAE_RESULTS_DIR = os.environ.get('SVAE_RESULTS_DIR', '../datasets')
SVAE_OUTPUT_DIR = os.environ.get('SVAE_OUTPUT_DIR', '../output')

def main():
    data_path = os.path.join(SVAE_RESULTS_DIR, 'mimic_sepsis_highdim.csv')
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    df['competing_event'] = df['event'].replace({0: 2})

    print("Fitting Aalen-Johansen estimator for CIF...")
    ajf_death = AalenJohansenFitter(calculate_variance=True)
    ajf_discharge = AalenJohansenFitter(calculate_variance=True)
    ajf_death.fit(df['duration'], df['competing_event'], event_of_interest=1)
    ajf_discharge.fit(df['duration'], df['competing_event'], event_of_interest=2)

    time_points = [7, 14, 30]
    print("\nCIF Estimates at Key Time Points:")
    print("-" * 60)
    print(f"{'Time (Days)':<15} | {'Death (Event 1)':<20} | {'Discharge (Event 2)':<20}")
    print("-" * 60)
    for t in time_points:
        cif_death = ajf_death.predict(t)
        cif_discharge = ajf_discharge.predict(t)
        print(f"{t:<15} | {float(cif_death):.4f}               | {float(cif_discharge):.4f}")
    print("-" * 60)

    plt.figure(figsize=(10, 6))
    ajf_death.plot(label='In-hospital Death (Event 1)')
    ajf_discharge.plot(label='Discharge Alive (Event 2)')
    plt.title('Cumulative Incidence Functions — Competing Events (MIMIC-IV)')
    plt.xlabel('Time (Days)')
    plt.ylabel('Cumulative Incidence')
    plt.xlim(0, 30)
    plt.grid(True, alpha=0.3)
    os.makedirs(SVAE_OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(SVAE_OUTPUT_DIR, 'figure_competing_risks_cif.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nCIF plot saved to {plot_path}")

    print("\nFitting Cause-Specific Cox Model for In-hospital Death...")
    df['cause_specific_event_1'] = (df['competing_event'] == 1).astype(bool)

    possible_features = ['Lactate', 'Age', 'Creatinine', 'BUN', 'Bicarbonate',
                         'pH', 'BaseExcess', 'AnionGap', 'Albumin', 'Bilirubin_Total']
    key_features = [col for col in possible_features if col in df.columns]
    if not key_features:
        key_features = df.select_dtypes(include=[np.number]).drop(
            columns=['event', 'duration', 'competing_event', 'cause_specific_event_1'],
            errors='ignore'
        ).columns[:10].tolist()

    valid_idx = df["duration"] > 0
    X = df.loc[valid_idx, key_features].fillna(df[key_features].median())
    y = Surv.from_dataframe("cause_specific_event_1", "duration", df[valid_idx])
    cox = CoxPHSurvivalAnalysis()
    cox.fit(X, y)

    print("\nCause-Specific Hazard Ratios (Event 1: In-hospital Death):")
    print("-" * 50)
    print(f"{'Feature':<25} | {'Hazard Ratio':<20}")
    print("-" * 50)
    hr = np.exp(cox.coef_)
    for feature, ratio in zip(key_features, hr):
        print(f"{feature:<25} | {ratio:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main()
