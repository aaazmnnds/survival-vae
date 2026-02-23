"""
MICE Imputation Script (Portable Version)
"""

import pandas as pd
import numpy as np
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

def run_mice_imputation():
    POSSIBLE_DATA_DIRS = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')),
        '/Users/azmannads/Documents/Research collections/Research 2025/datasets',
        '/Users/azmannads/VAE/datasets',
    ]
    DATA_DIR = next((p for p in POSSIBLE_DATA_DIRS if os.path.exists(p)), POSSIBLE_DATA_DIRS[0])
    
    DATASETS = ['metabric', 'mimic']
    SCENARIOS = ['light', 'moderate', 'severe']
    M_IMPUTATIONS = 5

    for dataset_name in DATASETS:
        for severity in SCENARIOS:
            mnar_path = os.path.join(DATA_DIR, f'{dataset_name}_mnar_{severity}.csv')
            if not os.path.exists(mnar_path): continue
                
            df_mnar = pd.read_csv(mnar_path)
            if 'Sex' in df_mnar.columns:
                df_mnar['Sex'] = df_mnar['Sex'].map({'F': 0, 'M': 1, 'Female': 0, 'Male': 1})
            
            df_numeric = df_mnar.select_dtypes(include=[np.number])
            target_cols = ['Survival_in_days', 'Status', 'time', 'event', 'duration', 'death_event', 'Time', 'Event']
            cols_to_drop = [c for c in target_cols if c in df_numeric.columns]
            df_targets = df_numeric[cols_to_drop].copy()
            df_features = df_numeric.drop(columns=cols_to_drop)
            
            for m in range(1, M_IMPUTATIONS + 1):
                imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42+m, sample_posterior=True)
                imputed_df = pd.DataFrame(imputer.fit_transform(df_features), columns=df_features.columns)
                for col in df_targets.columns: imputed_df[col] = df_targets[col].values
                
                output_path = os.path.join(DATA_DIR, f'{dataset_name}_imputed_mice_{severity}_{m}.csv')
                imputed_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    run_mice_imputation()
