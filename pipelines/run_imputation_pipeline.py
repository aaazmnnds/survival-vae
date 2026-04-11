import os
import subprocess
import time
import argparse
import sys
from datetime import timedelta

def run_script(script_name, args):
    """Runs a python script with the given arguments and tracks execution time."""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    
    cmd = [
        "python3", script_path,
        "--fold", str(args.fold),
        "--dataset", args.dataset,
        "--scenario", args.scenario
    ]
    
    print("=" * 60)
    print(f"[{args.current_step}/{args.total_steps}] Running: {script_name} | Dataset: {args.dataset} | Scenario: {args.scenario} | Fold: {args.fold}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run the process and stream output to terminal
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()
    
    elapsed_time = time.time() - start_time
    duration_str = str(timedelta(seconds=int(elapsed_time)))
    
    if process.returncode == 0:
        print(f"\n[OK] {script_name} completed in {duration_str}")
    else:
        print(f"\n[FAILED] {script_name} failed with return code {process.returncode} (Time: {duration_str})")
    
    return elapsed_time

def main():
    parser = argparse.ArgumentParser(description="Automated Imputation Pipeline")
    parser.add_argument("--fold", type=str, required=True, help="CV fold index (0-4) or 'all'")
    parser.add_argument("--dataset", type=str, required=True, choices=["metabric", "mimic", "all"], help="Dataset name or 'all'")
    parser.add_argument("--scenario", type=str, required=True, choices=["light", "moderate", "severe", "all"], help="Imputation scenario or 'all'")
    parser.add_argument("--method", type=str, default="all", choices=["survival_vae", "standard_vae", "gain", "mida", "mice", "missforest", "all"], help="Method name or 'all'")
    
    args = parser.parse_args()
    
    if args.fold == "all":
        target_folds = [0, 1, 2, 3, 4]
    else:
        target_folds = [int(args.fold)]
        
    target_datasets = ["metabric", "mimic"] if args.dataset == "all" else [args.dataset]
    target_scenarios = ["light", "moderate", "severe"] if args.scenario == "all" else [args.scenario]
    
    method_map = {
        "survival_vae": "impute_survival_vae.py",
        "standard_vae": "impute_standard_vae.py",
        "gain": "impute_gain.py",
        "mida": "impute_mida.py",
        "mice": "impute_mice.py",
        "missforest": "impute_missforest.py"
    }
    
    if args.method == "all":
        target_methods = list(method_map.keys())
    else:
        target_methods = [args.method]
    
    total_steps = len(target_folds) * len(target_datasets) * len(target_scenarios) * len(target_methods)
    current_step = 0
    total_pipeline_start = time.time()
    
    print("\n" + "=" * 80)
    print(f"STARTING FULL IMPUTATION PIPELINE | Fold(s): {args.fold}")
    print(f"Combinations: {len(target_folds)} folds x {len(target_datasets)} datasets x {len(target_scenarios)} scenarios x {len(target_methods)} methods")
    print(f"Total imputation runs to perform: {total_steps}")
    print("=" * 80)
    
    for f_idx in target_folds:
        print(f"\n>>>>>>> PROCESSING FOLD {f_idx} <<<<<<<")
        for d in target_datasets:
            for s in target_scenarios:
                print(f"\n>>> BEGINNING COMBINATION: {d.upper()} | {s.upper()} | FOLD {f_idx} <<<")
                for m in target_methods:
                    current_step += 1
                    script_name = method_map[m]
                    
                    # Create a temporary namespace to pass to run_script
                    class StepArgs: pass
                    step_args = StepArgs()
                    step_args.fold = f_idx
                    step_args.dataset = d
                    step_args.scenario = s
                    step_args.current_step = current_step
                    step_args.total_steps = total_steps
                    
                    run_script(script_name, step_args)
        
    total_pipeline_time = time.time() - total_pipeline_start
    total_duration_str = str(timedelta(seconds=int(total_pipeline_time)))
    
    print("\n" + "=" * 80)
    print(f"ALL IMPUTATION RUNS COMPLETE | Fold(s): {args.fold}")
    print(f"Total pipeline time: {total_duration_str}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
