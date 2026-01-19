"""
MNAR Simulation Script (Stage B: MIMIC-IV)
Simulates 'Threshold-Interaction' Missingness on MIMIC-IV using Semi-Synthetic Masking.
"""

import pandas as pd
import numpy as np
import os

# Configuration
INPUT_FILE = '/Users/azmannads/Downloads/Research 2025/datasets/final/mimic_sepsis_highdim.csv'
OUTPUT_DIR = '/Users/azmannads/Downloads/Research 2025/datasets'

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_mimic_mnar():
    print("Starting MNAR Simulation (MIMIC-IV Semi-Synthetic)...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        return

    # 1. Load Data (Real-World Data with Natural Missingness)
    df_real = pd.read_csv(INPUT_FILE)
    print(f"Loaded Data: {df_real.shape}")
    
    # Calc Baseline Missingness
    base_missing = df_real.isna().sum().sum()
    base_pct = (base_missing / df_real.size) * 100
    print(f"Baseline Missingness: {base_pct:.2f}%")
    
    # 2. MNAR Parameters (Clinical Logic)
    # Mechanism: Age x Lactate interaction -> Missingness in Labs
    median_age = df_real['Age'].median()
    
    # Lactate is Sepsis Severity Proxy (Normalize Observed Values)
    lactate_vals = df_real['Lactate'].copy()
    lactate_mean = lactate_vals.mean()
    lactate_std = lactate_vals.std()
    
    # Fill NaN Lactate with mean just for probability calculation
    lactate_filled = lactate_vals.fillna(lactate_mean)
    lactate_norm = (lactate_filled - lactate_mean) / lactate_std
    
    # Target Variables: Expanded Key Sepsis Labs + Vitals (Aggressive for 70%)
    target_vars = [
        # Labs
        'Albumin', 'Bilirubin_Total', 'pH', 'Lactate', 'ALT', 'AST', 'AnionGap', 
        'BUN', 'BaseExcess', 'Bicarbonate', 'Calcium', 'Chloride', 'Creatinine', 
        'Glucose', 'INR', 'PT', 'PTT', 'Potassium', 'Sodium', 'WBC', 'pCO2', 'pO2',
        # Vitals (Mean)
        'mean_DBP', 'mean_HR', 'mean_MAP', 'mean_O2Sat', 'mean_RR', 'mean_SBP', 'mean_Temp',
        # Vitals (Min/Max - often missing if not continuous monitoring)
        'min_DBP', 'min_HR', 'min_MAP', 'min_O2Sat', 'min_RR', 'min_SBP', 'min_Temp',
        'max_DBP', 'max_HR', 'max_MAP', 'max_O2Sat', 'max_RR', 'max_SBP', 'max_Temp'
    ]
    
    # Scenarios (Tuned for ~40%, ~55%, ~70% total missingness)
    scenarios = {
        'Light':    {'alpha': 1.0, 'beta': -0.7},  # Target ~40% (Baseline 25% + 15%)
        'Moderate': {'alpha': 3.0, 'beta': 1.0},   # Target ~55%
        'Severe':   {'alpha': 5.0, 'beta': 5.0}    # Target ~70% (Saturation territory)
    }
    
    np.random.seed(42)

    # 3. Simulate Scenarios
    for name, params in scenarios.items():
        print(f"\nProcessing Scenario: {name} (alpha={params['alpha']}, beta={params['beta']})")
        
        df_scenario = df_real.copy()
        
        # Mask Matrix: Track ONLY Artificially Removed Values (Ground Truth)
        # 1 = Artificially Removed, 0 = Originally Observed OR Naturally Missing
        mask_artificial = pd.DataFrame(False, index=df_real.index, columns=df_real.columns)
        
        for var in target_vars:
            if var not in df_real.columns:
                continue
                
            # Check which values are currently observed
            observed_indices = df_real[var].notna()
            
            # Interaction: (Age > Median) * Lactate
            age_indicator = (df_real['Age'] > median_age).astype(float)
            interaction_term = age_indicator * lactate_norm
            
            # Probability calculation
            prob_missing = sigmoid(params['alpha'] * interaction_term + params['beta'])
            
            # Sample missingness
            random_draw = np.random.rand(len(df_real))
            should_be_missing = random_draw < prob_missing
            
            # CRITICAL: Only apply mask if value was originally observed
            # We cannot "remove" a value that is already NaN
            to_mask = should_be_missing & observed_indices
            
            # Apply
            df_scenario.loc[to_mask, var] = np.nan
            mask_artificial.loc[to_mask, var] = True
            
        # Calc Total Missingness
        total_cells = df_scenario.size
        total_missing = df_scenario.isna().sum().sum()
        missing_pct = (total_missing / total_cells) * 100
        print(f"  Total Dataset Missingness: {missing_pct:.2f}%")

        # Save outputs
        mnar_file = os.path.join(OUTPUT_DIR, f'mimic_mnar_{name.lower()}.csv')
        mask_file = os.path.join(OUTPUT_DIR, f'mimic_mask_{name.lower()}.csv')
        
        df_scenario.to_csv(mnar_file, index=False)
        mask_artificial.to_csv(mask_file, index=False)
        print(f"  Saved: {mnar_file}")

if __name__ == "__main__":
    simulate_mimic_mnar()
