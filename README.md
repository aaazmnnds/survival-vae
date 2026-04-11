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
└── utils/                  # Utility scripts
    └── create_cv_splits.py
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
Use the automated pipeline to tune hyperparameters for all methods (imputation and survival models) via Optuna. **Note: In this study, optimization is performed using Fold 0 only.**
```bash
# Optimize all methods for a specific dataset and scenario (Fold 0 ONLY)
python pipelines/run_optuna_pipeline.py --dataset metabric --scenario light --fold 0
```
Atomic scripts for specific methods are also available in `optimization/`.

### 3. Imputation
Run the automated imputation pipeline once optimization is complete. **This step should be performed for all 5 folds (0-4).**
```bash
# Impute using all methods for a specific dataset/scenario (Example: Fold 0)
python pipelines/run_imputation_pipeline.py --dataset metabric --scenario all --fold 0

# To run for all folds, iterate through --fold 0 to 4
```
Individual imputation scripts are also available in `imputation/`.

### 4. Train Survival Models
Train downstream prognostic models using the automated survival optimization pipeline. **Perform this for all 5 folds (0-4).**
```bash
# Train on a specific fold
python pipelines/run_survival_optimization_pipeline.py --dataset metabric --scenario light --fold 0
```
Alternatively, use `survival/train_survival_models.py` directly for individual model training.

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
