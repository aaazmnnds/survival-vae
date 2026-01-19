"""
missForest Imputation Script
============================
Implements 'missForest' (Stekhoven & Buhlmann, 2012) using Scikit-Learn.
Strategy: Single Imputation (Standard for ML Benchmarks)
Method: IterativeImputer with RandomForestRegressor

Output: 6 total .csv files (2 Datasets * 3 Scenarios)
"""

import pandas as pd
import numpy as np
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Configuration
DATA_DIR = '/Users/azmannads/Downloads/Research 2025/datasets'
DATASETS = ['metabric', 'mimic']
SCENARIOS = ['light', 'moderate', 'severe']

# missForest settings
MAX_ITER = 10
N_TREES = 100

def run_missforest_imputation():
    print(f"Starting missForest Imputation...")
    
    for dataset_name in DATASETS:
        print(f"\n{'#'*60}")
        print(f"PROCESSING DATASET: {dataset_name.upper()}")
        print(f"{'#'*60}")
        
        for severity in SCENARIOS:
            mnar_path = os.path.join(DATA_DIR, f'{dataset_name}_mnar_{severity}.csv')
            output_path = os.path.join(DATA_DIR, f'{dataset_name}_imputed_missforest_{severity}.csv')
            
            if not os.path.exists(mnar_path):
                print(f"Skipping {severity.upper()} - File not found: {mnar_path}")
                continue
                
            print(f"\nScenario: {severity.upper()}")
            print(f"Input: {mnar_path}")
            
            # Load Data
            df_mnar = pd.read_csv(mnar_path)
            
            # Encode 'Sex' column if present
            if 'Sex' in df_mnar.columns:
                df_mnar['Sex'] = df_mnar['Sex'].map({'F': 0, 'M': 1, 'Female': 0, 'Male': 1})
            
            # Drop non-numeric columns
            df_numeric = df_mnar.select_dtypes(include=[np.number])
            
            # PREVENT LEAKAGE: Drop Survival Targets
            # We must not use the outcome to impute the features!
            target_cols = ['Survival_in_days', 'Status', 'time', 'event', 'duration', 'death_event', 'Time', 'Event']
            cols_to_drop = [c for c in target_cols if c in df_numeric.columns]
            
            # Store dropped targets to add back later (if needed, though separate file is better)
            # Actually, standard pipeline expects X (imputed). Y is separate usually.
            # But here we overwrite the file. So we should probably keep Y but NOT use it for imputation.
            
            df_targets = df_numeric[cols_to_drop].copy()
            df_features = df_numeric.drop(columns=cols_to_drop)
            
            print(f"  Dropping targets to prevent leakage: {cols_to_drop}")
            
            # Configure missForest
            # Note: We use RandomForestRegressor for all columns. 
            # In a mixed type dataset, one would normally handle categoricals differently,
            # but our processed datasets are pre-encoded/continuous.
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=N_TREES, 
                    n_jobs=4,  # Use 4 cores (safe for 16GB RAM)
                    random_state=42
                ),
                max_iter=MAX_ITER,
                random_state=42
            )
            
            # Perform Imputation
            print(f"  Running missForest (Trees={N_TREES})...")
            imputed_array = imputer.fit_transform(df_features)
            imputed_df = pd.DataFrame(imputed_array, columns=df_features.columns)
            
            # Add back targets (unmodified)
            for col in df_targets.columns:
                imputed_df[col] = df_targets[col].values
            
            # Save Output
            imputed_df.to_csv(output_path, index=False)
            print(f"  Saved: {os.path.basename(output_path)}")

    print(f"\n{'='*60}")
    print(f"missForest Pipeline Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_missforest_imputation()
