"""
Unified Survival Model Training Pipeline (Portable Version)
Trains DeepSurv and DeepHit models for all datasets/scenarios.
"""

import os
import subprocess
import time
import argparse

# Configuration
DATASETS = ['metabric', 'mimic']
SCENARIOS = ['light', 'moderate', 'severe']
METHODS = ['mice', 'missforest', 'gain', 'mida', 'survival_vae']
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
                    print(f"\nProcessing: {dataset} | {scenario} | {method} " + (f"(m={m_imp})" if m_imp else ""))
                    
                    # Common args
                    base_args = f"--dataset {dataset} --scenario {scenario} --method {method}"
                    if m_imp:
                        base_args += f" --m_imp {m_imp}"
                    
                    # Train Models
                    for model in models:
                        # Scripts now located in src/models/
                        script_path = os.path.join("src", "models", f"{model}.py")
                        print(f"Training {model} using {script_path}...")
                        success = run_command(f"python {script_path} {base_args}")
                        
                        if not success:
                            print(f"❌ Failed to train {model}")
                        else:
                            print(f"✅ Successfully trained {model}")

    elapsed = time.time() - start_time
    print(f"\nPipeline Complete! Total Time: {elapsed/3600:.2f} hours")

if __name__ == "__main__":
    main()
