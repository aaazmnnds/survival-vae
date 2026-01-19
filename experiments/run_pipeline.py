"""
Unified Survival Model Training Pipeline (Tuned)
Trains DeepSurv and DeepHit models with Hyperparameter Tuning for all datasets/scenarios.
"""

import os
import subprocess
import time
import argparse

# Configuration
DATASETS = ['metabric', 'mimic']
SCENARIOS = ['light', 'moderate', 'severe']
METHODS = ['mice', 'missforest', 'gain', 'mida', 'vae']
MICE_IMPUTATIONS = 5 # Number of MICE datasets

def run_command(cmd):
    """Run a shell command and print output"""
    print(f"\nRunning: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=DATASETS + ['all'], default='all')
    parser.add_argument('--scenario', type=str, choices=SCENARIOS + ['all'], default='all')
    parser.add_argument('--model', type=str, choices=['deepsurv', 'deephit', 'all'], default='all')
    args = parser.parse_args()

    # Filter based on args
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    scenarios = SCENARIOS if args.scenario == 'all' else [args.scenario]
    models = ['deepsurv', 'deephit'] if args.model == 'all' else [args.model]

    start_time = time.time()
    
    for dataset in datasets:
        for scenario in scenarios:
            for method in METHODS:
                
                # Determine iterations (MICE has 5, others 1)
                iterations = range(1, MICE_IMPUTATIONS + 1) if method == 'mice' else [None]
                
                for m_imp in iterations:
                    print(f"\n{'='*60}")
                    print(f"Processing: {dataset} | {scenario} | {method} " + (f"(m={m_imp})" if m_imp else ""))
                    print(f"{'='*60}")
                    
                    # Common args
                    base_args = f"--dataset {dataset} --scenario {scenario} --method {method}"
                    if m_imp:
                        base_args += f" --m_imp {m_imp}"
                    
                    # Train Models
                    for model in models:
                        script = f"train_{model}_tuned.py"
                        print(f"Training {model}...")
                        success = run_command(f"python {script} {base_args}")
                        
                        if not success:
                            print(f"❌ Failed to train {model} for {dataset}/{scenario}/{method}")
                        else:
                            print(f"✅ Successfully trained {model}")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Pipeline Complete! Total Time: {elapsed/3600:.2f} hours")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
