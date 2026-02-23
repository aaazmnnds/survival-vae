"""
MNAR Simulation Script (Stage A: METABRIC)
Simulates 'Threshold-Interaction' Missingness on the complete METABRIC dataset.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Configuration
POSSIBLE_DATA_DIRS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')),
    '/Users/azmannads/Documents/Research collections/Research 2025/datasets',
    '/Users/azmannads/VAE/datasets',
]

DATA_DIR = next((p for p in POSSIBLE_DATA_DIRS if os.path.exists(p)), POSSIBLE_DATA_DIRS[0])
INPUT_FILE = os.path.join(DATA_DIR, 'metabric_processed.csv')
OUTPUT_DIR = DATA_DIR

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_mnar():
    print("Starting MNAR Simulation (METABRIC)...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        return

    df_complete = pd.read_csv(INPUT_FILE)
    print(f"Loaded Data: {df_complete.shape}")
    
    median_age = df_complete['Age'].median()
    mki67_norm = (df_complete['MKI67'] - df_complete['MKI67'].mean()) / df_complete['MKI67'].std()
    target_vars = ['EGFR', 'PGR', 'ERBB2', 'Hormone_Tx', 'Radiotherapy', 'Chemotherapy', 'ER_Positive']
    
    scenarios = {
        'Light':    {'alpha': 2.5, 'beta': -1.0},
        'Moderate': {'alpha': 3.5, 'beta': 0.5},
        'Severe':   {'alpha': 5.0, 'beta': 3.0}
    }
    
    np.random.seed(42)

    for name, params in scenarios.items():
        print(f"\nProcessing Scenario: {name}")
        df_scenario = df_complete.copy()
        mask_scenario = pd.DataFrame(False, index=df_complete.index, columns=df_complete.columns)
        
        for var in target_vars:
            age_indicator = (df_complete['Age'] > median_age).astype(float)
            interaction_term = age_indicator * mki67_norm
            prob_missing = sigmoid(params['alpha'] * interaction_term + params['beta'])
            missing_indices = np.random.rand(len(df_complete)) < prob_missing
            df_scenario.loc[missing_indices, var] = np.nan
            mask_scenario.loc[missing_indices, var] = True
            
        mnar_file = os.path.join(OUTPUT_DIR, f'metabric_mnar_{name.lower()}.csv')
        mask_file = os.path.join(OUTPUT_DIR, f'metabric_mask_{name.lower()}.csv')
        df_scenario.to_csv(mnar_file, index=False)
        mask_scenario.to_csv(mask_file, index=False)
        print(f"  Saved: {mnar_file}")

if __name__ == "__main__":
    simulate_mnar()
