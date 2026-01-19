"""
MIMIC-IV Sepsis Cohort Extraction Script (High-Dimensional)
Extracts a sepsis cohort (P > 50 features) with labs, vitals, and comorbidities.
"""

import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# --- Configuration ---
MIMIC_PATH = '/Users/azmannads/Downloads/Dr. Zhang/mimic-iv-2.2/'
OUTPUT_PATH = '/Users/azmannads/Downloads/Research 2025/datasets'
INTERMEDIATE_PATH = os.path.join(OUTPUT_PATH, 'intermediate')
FINAL_PATH = os.path.join(OUTPUT_PATH, 'final')

# Ensure directories exist
for p in [INTERMEDIATE_PATH, FINAL_PATH]:
    Path(p).mkdir(parents=True, exist_ok=True)

# Parameters
MIN_AGE = 18
SEPSIS_CODES = ['A40', 'A41', 'R65', '995', '785'] # ICD10 + ICD9 prefixes

# --- Item Configuration ---
LAB_ITEMS = {
    'Albumin': [50862], 'AnionGap': [50868], 'Bicarbonate': [50882], 'BUN': [51006],
    'Calcium': [50893, 50804], 'Chloride': [50902, 50806], 'Creatinine': [50912],
    'Glucose': [50931, 50809], 'Sodium': [50983, 50824], 'Potassium': [50971, 50822],
    'Lactate': [50813], 'Hemoglobin': [51222], 'Hematocrit': [51221], 'Platelets': [51265],
    'WBC': [51301, 51300], 'Lymphocytes': [51244, 51245], 'Neutrophils': [51256],
    'INR': [51237], 'PT': [51274], 'PTT': [51275], 'ALT': [50861], 'AST': [50878],
    'Bilirubin_Total': [50885], 'pH': [50820], 'pO2': [50821], 'pCO2': [50818], 'BaseExcess': [50802]
}

VITAL_ITEMS = {
    'HR': [220045], 'RR': [220210, 224690], 'SBP': [220179, 220050],
    'DBP': [220180, 220051], 'MAP': [220052, 220181, 225312],
    'Temp': [223762, 223761], 'O2Sat': [220277]
}

def verify_files():
    required = [
        'hosp/patients.csv.gz', 'hosp/admissions.csv.gz', 'hosp/diagnoses_icd.csv.gz',
        'hosp/labevents.csv.gz', 'icu/icustays.csv.gz', 'icu/chartevents.csv.gz'
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(MIMIC_PATH, f))]
    if missing:
        print(f"Missing files: {missing}")
        return False
    return True

def extract_base_cohort():
    print("Step 1: Extracting Sepsis Cohort...")
    
    # Load base tables
    patients = pd.read_csv(f'{MIMIC_PATH}hosp/patients.csv.gz', compression='gzip', usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year'])
    admissions = pd.read_csv(f'{MIMIC_PATH}hosp/admissions.csv.gz', compression='gzip', usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime', 'hospital_expire_flag'])
    diagnoses = pd.read_csv(f'{MIMIC_PATH}hosp/diagnoses_icd.csv.gz', compression='gzip', usecols=['hadm_id', 'icd_code'])
    
    # Identify Sepsis Admissions
    sepsis_hadms = diagnoses[diagnoses['icd_code'].str.startswith(tuple(SEPSIS_CODES), na=False)]['hadm_id'].unique()
    cohort = admissions[admissions['hadm_id'].isin(sepsis_hadms)].merge(patients, on='subject_id', how='left')
    
    # Calculate Age & Filter
    cohort['admittime'] = pd.to_datetime(cohort['admittime'])
    cohort['age_approx'] = cohort['admittime'].dt.year - cohort['anchor_year'] + cohort['anchor_age']
    cohort = cohort[cohort['age_approx'] >= MIN_AGE].copy()
    
    # Process Survival Columns
    cohort['dischtime'] = pd.to_datetime(cohort['dischtime'])
    cohort['duration'] = (cohort['dischtime'] - cohort['admittime']).dt.total_seconds() / (24 * 3600)
    cohort['event'] = cohort['hospital_expire_flag']
    cohort = cohort.rename(columns={'age_approx': 'Age', 'gender': 'Sex'})
    
    # Save Base
    cols = ['subject_id', 'hadm_id', 'Age', 'Sex', 'admittime', 'dischtime', 'duration', 'event']
    cohort[cols].to_csv(os.path.join(INTERMEDIATE_PATH, 'cohort_base.csv'), index=False)
    print(f"  Cohort size: {len(cohort)} patients")

def extract_labs():
    print("Step 2: Extracting Labs...")
    cohort = pd.read_csv(os.path.join(INTERMEDIATE_PATH, 'cohort_base.csv'))
    hadm_ids = cohort['hadm_id'].values
    lab_map = {item: name for name, items in LAB_ITEMS.items() for item in items}
    all_lab_ids = list(lab_map.keys())
    
    labs_list = []
    for chunk in pd.read_csv(f'{MIMIC_PATH}hosp/labevents.csv.gz', compression='gzip', chunksize=1000000, usecols=['hadm_id', 'itemid', 'valuenum']):
        chunk = chunk[chunk['hadm_id'].isin(hadm_ids) & chunk['itemid'].isin(all_lab_ids) & chunk['valuenum'].notna()]
        if not chunk.empty:
            labs_list.append(chunk)

    if labs_list:
        labs = pd.concat(labs_list)
        labs['lab_name'] = labs['itemid'].map(lab_map)
        # Aggregate: Take first value per admission
        labs_wide = labs.pivot_table(index='hadm_id', columns='lab_name', values='valuenum', aggfunc='first').reset_index()
        cohort = cohort.merge(labs_wide, on='hadm_id', how='left')
        
    cohort.to_csv(os.path.join(INTERMEDIATE_PATH, 'cohort_with_labs.csv'), index=False)

def extract_vitals():
    print("Step 3: Extracting Vitals...")
    cohort = pd.read_csv(os.path.join(INTERMEDIATE_PATH, 'cohort_with_labs.csv'))
    icustays = pd.read_csv(f'{MIMIC_PATH}icu/icustays.csv.gz', compression='gzip', usecols=['hadm_id', 'stay_id'])
    cohort = cohort.merge(icustays, on='hadm_id', how='left')
    stay_ids = cohort['stay_id'].dropna().unique()
    
    vital_map = {item: name for name, items in VITAL_ITEMS.items() for item in items}
    all_vital_ids = list(vital_map.keys())
    
    vitals_list = []
    for chunk in pd.read_csv(f'{MIMIC_PATH}icu/chartevents.csv.gz', compression='gzip', chunksize=1000000, usecols=['stay_id', 'itemid', 'valuenum']):
        chunk = chunk[chunk['stay_id'].isin(stay_ids) & chunk['itemid'].isin(all_vital_ids) & chunk['valuenum'].notna()]
        if not chunk.empty:
            vitals_list.append(chunk)

    if vitals_list:
        vitals = pd.concat(vitals_list)
        vitals['vital_name'] = vitals['itemid'].map(vital_map)
        # Convert Temp F to C logic
        vitals.loc[vitals['itemid'] == 223761, 'valuenum'] = (vitals.loc[vitals['itemid'] == 223761, 'valuenum'] - 32) * 5/9
        
        # Aggregate Mean/Min/Max
        vitals_agg = vitals.groupby(['stay_id', 'vital_name'])['valuenum'].agg(['mean', 'min', 'max']).unstack()
        vitals_agg.columns = [f'{stat}_{var}' for stat, var in vitals_agg.columns]
        vitals_agg = vitals_agg.reset_index()
        
        # Merge back (handling multiple stays by taking first)
        cohort = cohort.merge(vitals_agg, on='stay_id', how='left').groupby('hadm_id').first().reset_index()

    cohort.drop(columns=['stay_id'], errors='ignore').to_csv(os.path.join(INTERMEDIATE_PATH, 'cohort_with_vitals.csv'), index=False)

def extract_secondary():
    print("Step 4: Extracting Output & Comorbidities...")
    cohort = pd.read_csv(os.path.join(INTERMEDIATE_PATH, 'cohort_with_vitals.csv'))
    
    # Simplified Charlson Comorbidities Mapping (ICD10 prefixes)
    cci_map = {
        'CCI_MI': ['I21', 'I22', 'I252'], 'CCI_CHF': ['I50', 'I42'], 'CCI_PVD': ['I71', 'I790'],
        'CCI_Stroke': ['I60', 'I61', 'I62', 'I63', 'I64'], 'CCI_Renal': ['N18', 'N19'],
        'CCI_Liver': ['K70', 'K71', 'K72', 'K73', 'K74'], 'CCI_Cancer': ['C0', 'C1', 'C2', 'C3', 'C4']
    }
    
    diagnoses = pd.read_csv(f'{MIMIC_PATH}hosp/diagnoses_icd.csv.gz', compression='gzip')
    diag_sub = diagnoses[diagnoses['hadm_id'].isin(cohort['hadm_id'])]
    
    for name, prefixes in cci_map.items():
        pos_hadms = diag_sub[diag_sub['icd_code'].str.startswith(tuple(prefixes), na=False)]['hadm_id'].unique()
        cohort[name] = cohort['hadm_id'].isin(pos_hadms).astype(int)

    final_file = os.path.join(FINAL_PATH, 'mimic_sepsis_highdim.csv')
    cohort.to_csv(final_file, index=False)
    print(f"Final Dataset Saved: {final_file}")
    print(f"Dimensions: {cohort.shape}")

if __name__ == "__main__":
    if verify_files():
        extract_base_cohort()
        extract_labs()
        extract_vitals()
        extract_secondary()
