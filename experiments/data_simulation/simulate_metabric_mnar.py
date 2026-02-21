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
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'datasets'),
    os.path.join(os.path.dirname(__file__), '..', '..', 'datasets'),  # User's Mac Mini path
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets'))  # Relative path
]

DATA_DIR = None
for path in POSSIBLE_DATA_DIRS:
    if os.path.exists(path):
        DATA_DIR = path
        break

if DATA_DIR is None:
    # Default fallback
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'datasets')
    print(f"Warning: Could not find datasets. Defaulting to: {DATA_DIR}")

INPUT_FILE = os.path.join(DATA_DIR, 'metabric_processed.csv')
OUTPUT_DIR = DATA_DIR

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_mnar():
    print("Starting MNAR Simulation (METABRIC)...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        return

    # 1. Load Data
    df_complete = pd.read_csv(INPUT_FILE)
    print(f"Loaded Data: {df_complete.shape}")
    
    # 2. MNAR Parameters
    median_age = df_complete['Age'].median()
    mki67_norm = (df_complete['MKI67'] - df_complete['MKI67'].mean()) / df_complete['MKI67'].std()
    target_vars = ['EGFR', 'PGR', 'ERBB2', 'Hormone_Tx', 'Radiotherapy', 'Chemotherapy', 'ER_Positive']
    
    # Scenarios: Light (~20%), Moderate (~35%), Severe (>50%) - Total dataset missingness
    scenarios = {
        'Light':    {'alpha': 2.5, 'beta': -1.0},
        'Moderate': {'alpha': 3.5, 'beta': 0.5},
        'Severe':   {'alpha': 5.0, 'beta': 3.0}
    }
    
    np.random.seed(42)

    # 3. Simulate Scenarios
    for name, params in scenarios.items():
        print(f"\nProcessing Scenario: {name} (alpha={params['alpha']}, beta={params['beta']})")
        
        df_scenario = df_complete.copy()
        mask_scenario = pd.DataFrame(False, index=df_complete.index, columns=df_complete.columns)
        
        for var in target_vars:
            # Interaction: (Age > Median) * Tumor_Size
            age_indicator = (df_complete['Age'] > median_age).astype(float)
            interaction_term = age_indicator * mki67_norm
            
            # Probability calculation
            prob_missing = sigmoid(params['alpha'] * interaction_term + params['beta'])
            
            # Apply missingness
            missing_indices = np.random.rand(len(df_complete)) < prob_missing
            df_scenario.loc[missing_indices, var] = np.nan
            mask_scenario.loc[missing_indices, var] = True
            
            print(f"  {var}: Removed {missing_indices.sum()} values ({missing_indices.sum()/len(df_complete):.1%})")

        # Save outputs
        mnar_file = os.path.join(OUTPUT_DIR, f'metabric_mnar_{name.lower()}.csv')
        mask_file = os.path.join(OUTPUT_DIR, f'metabric_mask_{name.lower()}.csv')
        
        df_scenario.to_csv(mnar_file, index=False)
        mask_scenario.to_csv(mask_file, index=False)
        print(f"  Saved: {mnar_file}")

if __name__ == "__main__":
    simulate_mnar()
