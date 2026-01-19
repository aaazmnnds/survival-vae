"""
MICE Imputation Script (Rubin's Rules: m=5)
===========================================
Implements Multiple Imputation by Chained Equations (MICE) for:
1. Two Datasets: METABRIC (Stage A), MIMIC-IV (Stage B)
2. Three Scenarios: Light, Moderate, Severe
3. Five Imputations (m=5): Generates 5 stochastic datasets per scenario

Output: 30 total .csv files (2 * 3 * 5)
"""

import pandas as pd
import numpy as np
import os
from sklearn.experimental import enable_iterative_imputer  # Explicitly enable!
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# Configuration
DATA_DIR = '/Users/azmannads/Downloads/Research 2025/datasets'
DATASETS = ['metabric', 'mimic']
SCENARIOS = ['light', 'moderate', 'severe']
M_IMPUTATIONS = 5

def run_mice_imputation():
    print(f"Starting MICE Imputation (Rubin's Rules: m={M_IMPUTATIONS})...")
    
    for dataset_name in DATASETS:
        print(f"\n{'#'*60}")
        print(f"PROCESSING DATASET: {dataset_name.upper()}")
        print(f"{'#'*60}")
        
        for severity in SCENARIOS:
            mnar_path = os.path.join(DATA_DIR, f'{dataset_name}_mnar_{severity}.csv')
            
            if not os.path.exists(mnar_path):
                print(f"Skipping {severity.upper()} - File not found: {mnar_path}")
                continue
                
            print(f"\nScenario: {severity.upper()}")
            print(f"Input: {mnar_path}")
            
            # Load MNAR Data
            df_mnar = pd.read_csv(mnar_path)
            
            # Encode 'Sex' column if present
            if 'Sex' in df_mnar.columns:
                df_mnar['Sex'] = df_mnar['Sex'].map({'F': 0, 'M': 1, 'Female': 0, 'Male': 1})
            
            # Drop non-numeric columns (e.g. timestamps)
            df_numeric = df_mnar.select_dtypes(include=[np.number])
            
            # PREVENT LEAKAGE: Drop Survival Targets
            target_cols = ['Survival_in_days', 'Status', 'time', 'event', 'duration', 'death_event', 'Time', 'Event']
            cols_to_drop = [c for c in target_cols if c in df_numeric.columns]
            
            df_targets = df_numeric[cols_to_drop].copy()
            df_features = df_numeric.drop(columns=cols_to_drop)
            
            print(f"  Dropping targets to prevent leakage: {cols_to_drop}")
            
            # --- CRITICAL FOR MIMIC (Stage B) ---
            # For MIMIC, we have Natural NaNs and Artificial NaNs.
            # MICE will impute BOTH.
            # RMSE calculation later knows which ones are Artificial (using the Mask file).
            
            # Loop m=5 times
            for m in range(1, M_IMPUTATIONS + 1):
                # Use different random seed for each imputation to ensure stochasticity
                seed = 42 + m
                
                # Configure MICE (BayesianRidge is standard for continuous data)
                imputer = IterativeImputer(
                    estimator=BayesianRidge(),
                    max_iter=10,
                    random_state=seed,
                    sample_posterior=True # Important for proper multiple imputation variance!
                )
                
                # Perform Imputation
                # Note: MICE can take a while for large datasets
                imputed_array = imputer.fit_transform(df_features)
                imputed_df = pd.DataFrame(imputed_array, columns=df_features.columns)
                
                # Add back targets (unmodified)
                for col in df_targets.columns:
                    imputed_df[col] = df_targets[col].values
                
                # Save Output
                output_filename = f'{dataset_name}_imputed_mice_{severity}_{m}.csv'
                output_path = os.path.join(DATA_DIR, output_filename)
                
                imputed_df.to_csv(output_path, index=False)
                print(f"  [m={m}] Saved: {output_filename}")

    print(f"\n{'='*60}")
    print(f"MICE Pipeline Complete! (Generated {len(DATASETS)*len(SCENARIOS)*M_IMPUTATIONS} files)")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_mice_imputation()
