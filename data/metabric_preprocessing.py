"""
METABRIC Extraction Script
Downloads the METABRIC dataset for Survival Analysis Benchmarking.
"""

import pandas as pd
import numpy as np
import os
import requests
from pathlib import Path

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets')
METABRIC_URL = "https://raw.githubusercontent.com/havakv/pycox/master/pycox/datasets/data/metabric.csv"
TARGET_FILE = os.path.join(OUTPUT_DIR, 'metabric_processed.csv')

def extract_metabric():
    print("Starting METABRIC extraction...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Attempt download via pycox or fallback to direct URL
    try:
        from pycox.datasets import metabric
        print("Downloading via pycox library...")
        df = metabric.read_df()
    except ImportError:
        print("PyCox not found. Downloading from source URL...")
        try:
            response = requests.get(METABRIC_URL)
            response.raise_for_status()
            with open(TARGET_FILE, 'wb') as f:
                f.write(response.content)
            df = pd.read_csv(TARGET_FILE)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return

    # Rename columns to standardized names
    # x0-x3: Gene Expressions, x4-x7: Clinical Binaries, x8: Age
    rename_map = {
        'x0': 'MKI67', 'x1': 'EGFR', 'x2': 'PGR', 'x3': 'ERBB2',
        'x4': 'Hormone_Tx', 'x5': 'Radiotherapy', 'x6': 'Chemotherapy', 
        'x7': 'ER_Positive', 'x8': 'Age'
    }
    df = df.rename(columns=rename_map)

    # Save processed file
    df.to_csv(TARGET_FILE, index=False)
    
    print("-" * 30)
    print("METABRIC Summary:")
    print(f"Patients: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print(f"Event Rate: {df['event'].mean():.2%}")
    print(f"Saved to: {TARGET_FILE}")
    print("-" * 30)

if __name__ == "__main__":
    extract_metabric()
