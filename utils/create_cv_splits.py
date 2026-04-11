"""
Create CV Splits JSON Files
============================
Creates fixed 5-fold stratified cross-validation splits for METABRIC and MIMIC-IV.
Stratified on event column to preserve event rate across folds.
Split ratio: 72% train / 8% val / 20% test (per fold).

Specifically:
- 20% of total = test fold (held out completely), stratified
- 80% remaining split into 90% train (72% total) / 10% val (8% total), stratified

Run ONCE before any pipeline scripts.
Output: cv_splits_metabric.json, cv_splits_mimic.json
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import StratifiedKFold, train_test_split

POSSIBLE_DATA_DIRS = [
    './datasets',
    'datasets',
    '../datasets',
    '/home/azman/VAE/Survival-VAE_study',
    '/Users/azmannads/VAE/Survival-VAE_study',
    '/Users/azmannads/Documents/Research collections/Research 2025/datasets',
    '.'
]

def find_dir(possibilities):
    for p in possibilities:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find datasets directory.")

DATA_DIR = find_dir(POSSIBLE_DATA_DIRS)

DATASETS = {
    'metabric': {
        'file': 'metabric_processed.csv',
        'event_col': 'event'
    },
    'mimic': {
        'file': 'mimic_sepsis_highdim.csv',
        'event_col': 'event'
    }
}

RANDOM_STATE = 42
N_SPLITS = 5

for dataset_name, config in DATASETS.items():
    path = os.path.join(DATA_DIR, config['file'])

    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        continue

    df = pd.read_csv(path)
    print(f"\n{dataset_name.upper()}: {df.shape}")

    if config['event_col'] not in df.columns:
        print(f"Error: Event column '{config['event_col']}' not found.")
        print(f"Available columns: {list(df.columns)}")
        continue

    n = len(df)
    y = df[config['event_col']].values.astype(int)
    indices = np.arange(n)

    event_rate = y.mean()
    print(f"  Overall event rate: {event_rate:.3f} ({y.sum()} events / {n} total)")

    # Stratified outer split: trainval vs test
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    splits = {}
    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(indices, y)):
        # Stratified inner split: train vs val
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=0.10,
            random_state=RANDOM_STATE,
            stratify=y[trainval_idx]
        )

        splits[f'fold_{fold_idx + 1}'] = {
            'train': sorted(train_idx.tolist()),
            'val':   sorted(val_idx.tolist()),
            'test':  sorted(test_idx.tolist())
        }

        total = len(train_idx) + len(val_idx) + len(test_idx)
        train_er = y[train_idx].mean()
        val_er   = y[val_idx].mean()
        test_er  = y[test_idx].mean()

        print(f"  Fold {fold_idx + 1}: "
              f"train={len(train_idx)} (er={train_er:.3f}), "
              f"val={len(val_idx)} (er={val_er:.3f}), "
              f"test={len(test_idx)} (er={test_er:.3f}), "
              f"total={total}")

        # Verify no overlap and full coverage
        assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap!"
        assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"
        assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap!"
        assert total == n, f"Missing indices! {total} != {n}"

    output_path = os.path.join(DATA_DIR, f'cv_splits_{dataset_name}.json')
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"  Saved: {output_path}")

print("\nDone.")
print("\nExpected sizes:")
print("  METABRIC (N=1,904): train~1,371, val~152, test~381 per fold")
print("  MIMIC-IV (N=32,065): train~23,086, val~2,565, test~6,414 per fold")
print("\nVerify event rates (er) are consistent across folds before proceeding.")
