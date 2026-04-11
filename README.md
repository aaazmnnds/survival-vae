# Survival-VAE: Imputation and Survival Analysis for MNAR Clinical Data

Official implementation for the study on Survival-VAE, a deep generative framework for handling informative missingness (Missing Not At Random) in clinical survival data.

## Overview
Survival-VAE addresses the critical challenge of informative missingness in high-stakes clinical informatics. Unlike standard imputation methods that assume missingness is random (MAR), Survival-VAE is specifically designed for Missing Not At Random (MNAR) conditions, where the probability of a value being missing depends on the unobserved value itself or other clinical indicators. 

The framework integrates variational autoencoders (VAE) with survival-aware network outputs to learn latent representations that preserve both the feature distribution and the prognostic signal of the data. This repository provides a formal benchmark for MNAR clinical data generation and a reproducible pipeline for hyperparameter optimization, model training, and performance evaluation.

## Authors and Affiliations
* **Azman Nads** - Lead Researcher, Survival-VAE Development.
* **Phil Bayram & Aupke et al.** - Collaborative Contributors, Domain-Adaptive Frameworks and Reliability Metrics.

## Repository Structure

```text
├── pipelines/              # Master pipeline scripts (primary entry points)
│   ├── run_imputation_pipeline.py
│   ├── run_optuna_pipeline.py
│   └── run_survival_optimization_pipeline.py
├── imputation/             # Imputation implementation for all methods
│   ├── impute_gain.py
│   ├── impute_mice.py
│   ├── impute_mida.py
│   ├── impute_missforest.py
│   ├── impute_standard_vae.py
│   └── impute_survival_vae.py
├── optimization/           # Hyperparameter tuning using Optuna
│   ├── optimize_deephit.py
│   ├── optimize_deepsurv.py
│   ├── optimize_gain.py
│   ├── optimize_mice.py
│   ├── optimize_mida.py
│   ├── optimize_missforest.py
│   ├── optimize_rsf.py
│   ├── optimize_standard_vae.py
│   ├── optimize_survival_vae.py
│   └── optimize_xgboost.py
├── survival/               # Survival model training and evaluation
│   └── train_survival_models.py
├── evaluation/             # Metrics calculation and results aggregation
│   ├── calculate_imputation_metrics.py
│   ├── aggregate_imputation_metrics.py
│   ├── aggregate_survival_metrics.py
│   └── aggregate_survival_results.py
├── figures/                # Scripts to reproduce manuscript figures
│   ├── reproduce_fig_baseline_convergence.py
│   ├── reproduce_figure_imputation_fidelity.py
│   └── reproduce_pareto_frontiers.py
├── simulation/             # MNAR missingness simulation scripts
│   ├── metabric_mnar.py
│   └── mimic_mnar.py
├── utils/                  # Utility scripts
│   └── create_cv_splits.py
├── requirements.txt        # Dependency versions
└── LICENSE
```

## Usage

### 1. Environment Setup
Install the required dependencies using the provided requirements file:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Generate the 5-fold cross-validation splits for the METABRIC and MIMIC-IV datasets:
```bash
python3 utils/create_cv_splits.py
```

### 3. Hyperparameter Optimization
Tuning is performed using Optuna on Fold 0 to find optimal parameters for all imputation and survival models:
```bash
python3 pipelines/run_optuna_pipeline.py --fold 0 --dataset all --scenario all
```

### 4. Running the Imputation Pipeline
Execute the full imputation study across all folders and missingness scenarios (Light, Moderate, Severe):
```bash
python3 pipelines/run_imputation_pipeline.py --fold all --dataset all --scenario all
```

### 5. Prognostic Performance Evaluation
Train and evaluate downstream survival models (RSF, XGBoost, DeepSurv, DeepHit) on the imputed data:
```bash
python3 survival/train_survival_models.py --fold all --dataset all --scenario all
```

### 6. Results Aggregation
Generate the final manuscript tables and LaTeX code:
```bash
python3 evaluation/aggregate_survival_results.py --latex
```

## Data Availability
This study utilizes two high-dimensional clinical datasets:
* **METABRIC**: The Molecular Taxonomy of Breast Cancer International Consortium dataset, containing genomic and clinical profiles of breast cancer patients.
* **MIMIC-IV**: The Medical Information Mart for Intensive Care IV dataset, providing longitudinal EHR data for sepsis cohorts (requires PhysioNet credentialed access).

## Requirements
* torch==2.10.0
* scikit-survival==0.27.0
* xgboost==3.2.0
* optuna==4.7.0
* pycox==0.3.0
* scikit-learn==1.3.2
* numpy==1.26.4
* pandas==2.2.2

## Citation
Citation details are currently pending publication.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
The authors would like to thank the developers of PyTorch, Optuna, and the Scikit-Survival community for their invaluable tools.
