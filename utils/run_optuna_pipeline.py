import argparse
import subprocess
import time
import sys
import os

def format_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(minutes)}m {int(seconds)}s"

def run_script(script_path, name, index, total, fold, dataset, scenario, trials):
    header = f"[{index}/{total}] Running: {name} | Dataset: {dataset} | Scenario: {scenario} | Fold: {fold}"
    print("\n" + "="*80)
    print(header)
    print("="*80 + "\n")

    cmd = [
        "python3", script_path,
        "--fold", str(fold),
        "--dataset", dataset,
        "--scenario", scenario,
        "--trials", str(trials)
    ]

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
    parser = argparse.ArgumentParser(description="Run all Optuna optimization scripts in sequence.")
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4], help="CV fold index (0-4)")
    parser.add_argument("--dataset", type=str, required=True, choices=["metabric", "mimic", "all"], help="Dataset name or 'all'")
    parser.add_argument("--scenario", type=str, required=True, choices=["light", "moderate", "severe", "all"], help="Imputation scenario or 'all'")
    parser.add_argument("--trials", type=int, help="Override number of Optuna trials per method")
    args = parser.parse_args()

    # Define target sets
    target_datasets = ["metabric", "mimic"] if args.dataset == "all" else [args.dataset]
    target_scenarios = ["light", "moderate", "severe"] if args.scenario == "all" else [args.scenario]
    
    # Dynamically find the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default trials: 100 for VAEs/Baselines, 50 for missForest
    def get_trials(default_val):
        return args.trials if args.trials is not None else default_val

    scripts = [
        (os.path.join(script_dir, "optimize_survival_vae.py"), "Survival-VAE Optuna", get_trials(100)),
        (os.path.join(script_dir, "optimize_standard_vae.py"), "Standard VAE Optuna", get_trials(100)),
        (os.path.join(script_dir, "optimize_gain.py"), "GAIN Optuna", get_trials(100)),
        (os.path.join(script_dir, "optimize_mida.py"), "MIDA Optuna", get_trials(100)),
        (os.path.join(script_dir, "optimize_mice.py"), "MICE Optuna", get_trials(100)),
        (os.path.join(script_dir, "optimize_missforest.py"), "missForest Optuna", get_trials(50)),
    ]

    total_runs = len(target_datasets) * len(target_scenarios) * len(scripts)
    current_run = 0
    total_start_time = time.time()
    
    print("\n" + "="*80)
    print(f"STARTING FULL OPTUNA PIPELINE | Fold: {args.fold}")
    print(f"Combinations: {len(target_datasets)} datasets x {len(target_scenarios)} scenarios x {len(scripts)} methods")
    print(f"Total optimized runs to perform: {total_runs}")
    print("="*80)

    for d in target_datasets:
        for s in target_scenarios:
            print(f"\n\n>>> BEGINNING COMBINATION: {d.upper()} | {s.upper()} <<<")
            for path, name, trials in scripts:
                current_run += 1
                run_script(path, name, current_run, total_runs, args.fold, d, s, trials)

    total_elapsed = time.time() - total_start_time
    
    print("\n" + "="*80)
    print(f"ALL {total_runs} RUNS COMPLETE | Fold: {args.fold}")
    print(f"Total pipeline time: {format_time(total_elapsed)}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
