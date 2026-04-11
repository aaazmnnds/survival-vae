# Survival-aware variational autoencoders for handling threshold-interaction missing not at random data in clinical prognosis

Official implementation of the **Survival-VAE** framework for handling missing not at random (MNAR) conditions in clinical survival datasets.

## Overview
Survival-VAE is a variational autoencoder (VAE) architecture that integrates Cox proportional hazards loss to preserve prognostic signals during the imputation of missing data. This repository addresses the challenge of **threshold-interaction MNAR**, a clinically realistic missingness mechanism where diagnostic test ordering depends on non-linear interactions across decision thresholds (e.g., Age x Biomarker). 

By optimizing for both reconstruction fidelity and survival supervision, Survival-VAE ensures that imputed clinical features remain useful for downstream prognostic modeling. The framework is validated on the METABRIC (breast cancer) and MIMIC-IV (sepsis) datasets, demonstrating superior performance in preserving survival information compared to standard VAE, MICE, missForest, GAIN, and MIDA.

## Authors and Affiliations
* **Azman Nads** [1,2] - azmannads@msutawi-tawi.edu.ph
* **Daniel Andrade** [1] - andrade@hiroshima-u.ac.jp

[1] Informatics and Data Science Program, Graduate School of Advanced Science and Engineering, Hiroshima University, Higashihiroshima, Hiroshima, Japan
[2] Department of Statistics, College of Mathematical Sciences, Mindanao State University Tawi-Tawi College of Technology and Oceanography, Bongao, Tawi-Tawi, Philippines

## Repository Structure

```text
├── pipelines/              # Master pipeline scripts (primary entry points)
│   ├── run_imputation_pipeline.py             # Automates imputation for all methods/folds
│   ├── run_optuna_pipeline.py                 # Optimizes imputation hyperparameters
│   └── run_survival_optimization_pipeline.py  # Optimizes downstream survival model parameters
├── imputation/             # Imputation implementation for all methods
│   ├── impute_gain.py
│   ├── impute_mice.py
│   ├── impute_mida.py
│   ├── impute_missforest.py
│   ├── impute_standard_vae.py
│   └── impute_survival_vae.py
├── optimization/           # Hyperparameter tuning logic using Optuna
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
│   ├── calculate_imputation_metrics.py        # Per-fold RMSE/MAE calculation
│   ├── aggregate_imputation_metrics.py        # Multi-fold imputation fidelity aggregation
│   ├── aggregate_survival_metrics.py          # Multi-fold clinical utility aggregation
│   └── aggregate_survival_results.py          # Final manuscript table generation
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
Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Generate the cross-validation splits for the study:
```bash
python3 utils/create_cv_splits.py
```

### 3. Hyperparameter Optimization (Optuna)
Optimization is performed on Fold 0 to find the best parameters for both imputation and survival models.

**Tune Imputation Models:**
```bash
python3 pipelines/run_optuna_pipeline.py --fold 0 --dataset all --scenario all
```

**Tune Downstream Survival Models:**
```bash
python3 pipelines/run_survival_optimization_pipeline.py --fold 0 --dataset all --scenario all
```

### 4. Running the Imputation Study
Execute the full imputation pipeline across all 5 folds and missingness scenarios (Light, Moderate, Severe):
```bash
python3 pipelines/run_imputation_pipeline.py --fold all --dataset all --scenario all
```

### 5. Imputation Fidelity Evaluation
Calculate and aggregate RMSE/MAE metrics across all folds.

**Per-Fold Metrics:**
```bash
python3 evaluation/calculate_imputation_metrics.py --fold 0
```

**Aggregate Fidelity Table:**
```bash
python3 evaluation/aggregate_imputation_metrics.py --latex
```

### 6. Prognostic Performance Evaluation (Survival)
Train and evaluate survival models (RSF, XGBoost, etc.) on the imputed data across all folds.

**Execution:**
```bash
python3 survival/train_survival_models.py --fold all --dataset all --scenario all
```

**Aggregate Clinical Utility Tables:**
```bash
python3 evaluation/aggregate_survival_metrics.py
python3 evaluation/aggregate_survival_results.py --latex
```

## Data Availability
This study utilizes the following clinical datasets:
* **METABRIC**: The Molecular Taxonomy of Breast Cancer International Consortium dataset (available via OncoMX/cBioPortal).
* **MIMIC-IV**: The Medical Information Mart for Intensive Care IV dataset for sepsis cohorts (requires PhysioNet credentialed access).

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
Nads, A., & Andrade, D. (2026). Survival-aware variational autoencoders for handling threshold-interaction missing not at random data in clinical prognosis. (Publication pending).

## License
MIT License. See LICENSE file for details.

## Acknowledgments
This work was supported by the Department of Science and Technology–Science Education Institute (DOST-SEI) of the Philippines through its Foreign Graduate Scholarship Program. The authors gratefully acknowledge the Informatics and Data Science Program A1-427 at Hiroshima University for providing computational resources. We acknowledge the METABRIC consortium and the PhysioNet community for making their data publicly available for research purposes.

