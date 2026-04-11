# Survival-VAE

**Survival-aware variational autoencoders for handling threshold-interaction
missing not at random data in clinical prognosis**

Azman Nads¹², Daniel Andrade¹

¹ Informatics and Data Science Program, Graduate School of Advanced Science
and Engineering, Hiroshima University, Japan
² Department of Statistics, College of Mathematical Sciences, MSU Tawi-Tawi
College of Technology and Oceanography, Philippines

---

## Overview

This repository contains the implementation of Survival-VAE, a variational
autoencoder that integrates Cox proportional hazards loss for imputing
threshold-interaction missing not at random (MNAR) data in clinical survival
analysis. The method is evaluated against five baseline imputation methods
(MICE, missForest, GAIN, MIDA, Standard VAE) on METABRIC (breast cancer,
N=1,904) and MIMIC-IV (sepsis, N=32,065) under light, moderate, and severe
missingness scenarios.

---

## Repository Structure

```text
.
├── imputation/             # Imputation pipeline for all methods
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
├── survival/               # Survival model training
│   └── train_survival_models.py
├── evaluation/             # Metrics calculation and results aggregation
│   ├── calculate_imputation_metrics.py
│   ├── aggregate_imputation_metrics.py
│   ├── aggregate_survival_metrics.py
│   ├── aggregate_survival_results.py
│   ├── reproduce_fig_baseline_convergence.py
│   ├── reproduce_figure_imputation_fidelity.py
│   └── reproduce_pareto_frontiers.py
├── simulation/             # MNAR missingness simulation scripts
│   ├── metabric_mnar.py
│   └── mimic_mnar.py
└── utils/                  # Utility and pipeline runner scripts
    ├── create_cv_splits.py
    ├── run_imputation_pipeline.py
    ├── run_optuna_pipeline.py
    └── run_survival_optimization_pipeline.py
```

---

## Usage

### 1. Simulate MNAR Missingness
Generate missingness masks based on the threshold-interaction MNAR mechanism.
```bash
python simulation/metabric_mnar.py
python simulation/mimic_mnar.py
```

### 2. Hyperparameter Optimization
Use Optuna to find the best hyperparameters for imputation and survival models.
```bash
python optimization/optimize_survival_vae.py --dataset metabric --scenario light
python optimization/optimize_xgboost.py --dataset mimic --scenario moderate
```

### 3. Imputation
Impute the missing values using the optimized configurations.
```bash
python imputation/impute_survival_vae.py --dataset metabric --scenario light --fold 0
python imputation/impute_mice.py --dataset mimic --scenario severe --fold 0
```

### 4. Train Survival Models
Train downstream prognostic models on the imputed datasets.
```bash
python survival/train_survival_models.py --dataset metabric --scenario light
```

### 5. Evaluate Results
Compute imputation fidelity metrics and aggregate survival performance (C-index).
```bash
# Calculate metrics for a specific fold
python evaluation/calculate_imputation_metrics.py --fold 0

# Aggregate results across all folds and scenarios
python evaluation/aggregate_survival_results.py
```

---

## Data Availability

**METABRIC:** Publicly available via cBioPortal
https://www.cbioportal.org/study/summary?id=brca_metabric

**MIMIC-IV v2.2:** Publicly available via PhysioNet following completion
of required CITI training and execution of a data use agreement.
https://physionet.org/content/mimiciv/

---

## Requirements

Python 3.11/3.12
PyTorch 2.10.0
scikit-survival 0.27.0
xgboost 3.2.0
optuna 4.7.0
pycox 0.3.0
scikit-learn 1.3.2
numpy 1.26.4
pandas 2.2.2

---

## Citation

[To be added upon acceptance]

---

## License

MIT License

---

## Acknowledgments

Supported by the Department of Science and Technology – Science Education
Institute (DOST-SEI) of the Philippines through its Foreign Graduate
Scholarship Program. Computational resources provided by the Informatics
and Data Science Program A1-427, Hiroshima University.
