"""
MNAR Simulation Script (Stage B: MIMIC-IV)
Simulates 'Threshold-Interaction' Missingness on MIMIC-IV using Semi-Synthetic Masking.
"""

import pandas as pd
import numpy as np
import os

# Configuration
POSSIBLE_DATA_DIRS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')),
    '/Users/azmannads/Documents/Research collections/Research 2025/datasets',
    '/Users/azmannads/VAE/datasets',
]

DATA_DIR = next((p for p in POSSIBLE_DATA_DIRS if os.path.exists(p)), POSSIBLE_DATA_DIRS[0])
INPUT_FILE = os.path.join(DATA_DIR, 'final/mimic_sepsis_highdim.csv')
OUTPUT_DIR = DATA_DIR

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_mimic_mnar():
    print("Starting MNAR Simulation (MIMIC-IV)...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        return

    df_real = pd.read_csv(INPUT_FILE)
    print(f"Loaded Data: {df_real.shape}")
    
    median_age = df_real['Age'].median()
    lactate_vals = df_real['Lactate'].copy()
    lactate_norm = (lactate_vals.fillna(lactate_vals.mean()) - lactate_vals.mean()) / lactate_vals.std()
    
    target_vars = [
        'Albumin', 'Bilirubin_Total', 'pH', 'Lactate', 'ALT', 'AST', 'AnionGap', 
        'BUN', 'BaseExcess', 'Bicarbonate', 'Calcium', 'Chloride', 'Creatinine', 
        'Glucose', 'INR', 'PT', 'PTT', 'Potassium', 'Sodium', 'WBC', 'pCO2', 'pO2',
        'mean_DBP', 'mean_HR', 'mean_MAP', 'mean_O2Sat', 'mean_RR', 'mean_SBP', 'mean_Temp',
        'min_DBP', 'min_HR', 'min_MAP', 'min_O2Sat', 'min_RR', 'min_SBP', 'min_Temp',
        'max_DBP', 'max_HR', 'max_MAP', 'max_O2Sat', 'max_RR', 'max_SBP', 'max_Temp'
    ]
    
    scenarios = {
        'Light':    {'alpha': 1.0, 'beta': -0.7},
        'Moderate': {'alpha': 3.0, 'beta': 1.0},
        'Severe':   {'alpha': 5.0, 'beta': 5.0}
    }
    
    np.random.seed(42)

    for name, params in scenarios.items():
        print(f"\nProcessing Scenario: {name}")
        df_scenario = df_real.copy()
        mask_artificial = pd.DataFrame(False, index=df_real.index, columns=df_real.columns)
        
        for var in target_vars:
            if var not in df_real.columns: continue
            observed_indices = df_real[var].notna()
            age_indicator = (df_real['Age'] > median_age).astype(float)
            interaction_term = age_indicator * lactate_norm
            prob_missing = sigmoid(params['alpha'] * interaction_term + params['beta'])
            to_mask = (np.random.rand(len(df_real)) < prob_missing) & observed_indices
            df_scenario.loc[to_mask, var] = np.nan
            mask_artificial.loc[to_mask, var] = True
            
        mnar_file = os.path.join(OUTPUT_DIR, f'mimic_mnar_{name.lower()}.csv')
        mask_file = os.path.join(OUTPUT_DIR, f'mimic_mask_{name.lower()}.csv')
        df_scenario.to_csv(mnar_file, index=False)
        mask_artificial.to_csv(mask_file, index=False)
        print(f"  Saved: {mnar_file}")

if __name__ == "__main__":
    simulate_mimic_mnar()
