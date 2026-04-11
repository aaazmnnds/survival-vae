# Survival-aware variational autoencoders for handling threshold-interaction missing not at random data in clinical prognosis

Official implementation of the **Survival-VAE** framework for handled missing not at random (MNAR) conditions in clinical survival datasets.

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
Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Generate the cross-validation splits for the study:
```bash
python3 utils/create_cv_splits.py
```

### 3. Hyperparameter Optimization
Run the Optuna optimization pipeline (performed on Fold 0):
```bash
python3 pipelines/run_optuna_pipeline.py --fold 0 --dataset all --scenario all
```

### 4. Running the Imputation Pipeline
Execute the full imputation study across all conditions:
```bash
python3 pipelines/run_imputation_pipeline.py --fold all --dataset all --scenario all
```

### 5. Prognostic Performance Evaluation
Train and evaluate survival models on the imputed data:
```bash
python3 survival/train_survival_models.py --fold all --dataset all --scenario all
```

### 6. Results Aggregation
Generate final manuscript tables and LaTeX output:
```bash
python3 evaluation/aggregate_survival_results.py --latex
```

## Data Availability
This study utilizes the following datasets:
* **METABRIC**: Molecular Taxonomy of Breast Cancer International Consortium (OncoMX/cBioPortal).
* **MIMIC-IV**: Medical Information Mart for Intensive Care IV (PhysioNet credentialed access).

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
Nads, A., & Andrade, D. (2025). Survival-aware variational autoencoders for handling threshold-interaction missing not at random data in clinical prognosis. (Publication pending).

## License
MIT License. See LICENSE file for details.

## Acknowledgments
We thank Hiroshima University and Mindanao State University Tawi-Tawi for supporting this research.
