import argparse
import subprocess
import time
import sys
import os
from datetime import timedelta

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def run_script(script_path, name, index, total, fold, dataset, scenario, method, trials):
    header = f"[{index}/{total}] Running: {name} | Dataset: {dataset} | Scenario: {scenario} | Method: {method} | Fold: {fold}"
    print("\n" + "="*80)
    print(header)
    print("="*80 + "\n")

    cmd = [
        "python3", script_path,
        "--fold", str(fold),
        "--dataset", dataset,
        "--scenario", scenario,
        "--method", method
    ]
    if trials is not None:
        cmd += ["--trials", str(trials)]

    start_time = time.time()
    try:
        # Stream stdout and stderr in real-time
        process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
        
        elapsed = time.time() - start_time
        clean_name = name.replace(" Optuna", "")
        if process.returncode == 0:
            print(f"\n[OK] {clean_name} completed in {format_time(elapsed)}")
        else:
            print(f"\n[FAILED] {clean_name} failed with return code {process.returncode} after {format_time(elapsed)}")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[FAILED] Error running {name}: {str(e)}")
    
    return elapsed

def main():
    parser = argparse.ArgumentParser(description="Run all Survival Model Optuna optimization scripts in sequence.")
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4], help="CV fold index (0-4)")
    parser.add_argument("--dataset", type=str, required=True, choices=["metabric", "mimic", "all"], help="Dataset name or 'all'")
    parser.add_argument("--scenario", type=str, required=True, choices=["light", "moderate", "severe", "all"], help="Imputation scenario or 'all'")
    parser.add_argument("--method", type=str, default="all", choices=["survival_vae", "standard_vae", "gain", "mida", "mice", "missforest", "all"], help="Imputation method or 'all'")
    parser.add_argument("--trials", type=int, help="Override number of Optuna trials per method (default is 50)")
    args = parser.parse_args()

    # Define target sets
    target_datasets = ["metabric", "mimic"] if args.dataset == "all" else [args.dataset]
    target_scenarios = ["light", "moderate", "severe"] if args.scenario == "all" else [args.scenario]
    target_methods = ["survival_vae", "standard_vae", "gain", "mida", "mice", "missforest"] if args.method == "all" else [args.method]
    
    # Dynamically find the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define survival optimization scripts
    scripts = [
        (os.path.join(script_dir, "optimize_rsf.py"), "RSF Optuna"),
        (os.path.join(script_dir, "optimize_xgboost.py"), "XGBoost Optuna"),
        (os.path.join(script_dir, "optimize_deepsurv.py"), "DeepSurv Optuna"),
        (os.path.join(script_dir, "optimize_deephit.py"), "DeepHit Optuna"),
    ]

    total_runs = len(target_datasets) * len(target_scenarios) * len(target_methods) * len(scripts)
    current_run = 0
    total_start_time = time.time()
    
    print("\n" + "="*80)
    print(f"STARTING SURVIVAL OPTIMIZATION PIPELINE | Fold: {args.fold}")
    print(f"Combinations: {len(target_datasets)} datasets x {len(target_scenarios)} scenarios x {len(target_methods)} methods x {len(scripts)} models")
    print(f"Total optimized runs to perform: {len(target_datasets) * len(target_scenarios) * len(target_methods) * len(scripts)}")
    print("="*80)

    for d in target_datasets:
        for s in target_scenarios:
            for m in target_methods:
                print(f"\n\n>>> BEGINNING COMBINATION: {d.upper()} | {s.upper()} | {m.upper()} <<<")
                for path, name in scripts:
                    current_run += 1
                    run_script(path, name, current_run, total_runs, args.fold, d, s, m, args.trials)

    total_elapsed = time.time() - total_start_time
    
    print("\n" + "="*80)
    print(f"ALL {total_runs} SURVIVAL RUNS COMPLETE | Fold: {args.fold}")
    print(f"Total pipeline time: {format_time(total_elapsed)}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
