# MNAR-SVAE

## 1. Overview
MNAR-SVAE is a variational autoencoder integrating Cox proportional hazards loss for clinical survival analysis under threshold-interaction MNAR conditions. We validated the model on METABRIC ($N=1,904$) and MIMIC-IV ($N=25,439$, first hospital admission, first 24-hour vitals). The manuscript is under review at Scientific Reports.

## 2. Repository Structure
```text
survival-vae/
├── datasets/          # CV split JSON files (data not included)
├── evaluation/        # Aggregation and figure reproduction scripts
├── imputation/        # Imputation model implementations
├── optimization/      # Optuna hyperparameter search scripts
├── pipelines/         # Pipeline wrapper scripts
├── simulation/        # MNAR simulation scripts
├── survival/          # Downstream survival model training and evaluation
├── utils/             # Data extraction and preprocessing utilities
├── requirements.txt
└── README.md
```

## 3. Requirements
```bash
pip install -r requirements.txt
```
Key packages: `torch==2.10.0`, `scikit-survival==0.27.0`, `xgboost==3.2.0`, `optuna==4.7.0`, `pycox==0.3.0`

## 4. Data
### METABRIC
Accessed via `pycox.datasets.metabric` (https://github.com/havakv/pycox).

### MIMIC-IV v2.2
Available through PhysioNet (https://physionet.org/content/mimiciv/2.2/) following CITI training and data use agreement execution. After downloading, set the MIMIC path in `utils/extract_mimic_data.py`.

## 5. Pipeline
Step 1: Generate CV splits
```bash
python3 utils/create_cv_splits.py
```

Step 2: Hyperparameter optimization (fold 0 only)
```bash
python3 pipelines/run_optuna_pipeline.py --fold 0 --dataset all --scenario all
```

Step 3: Imputation (all folds)
```bash
python3 pipelines/run_imputation_pipeline.py --fold all --dataset all --scenario all --method all
```

Step 4: Survival model optimization (fold 0 only)
```bash
python3 pipelines/run_survival_optimization_pipeline.py --fold 0 --dataset all --scenario all --method all
```

Step 5: Survival model evaluation (all folds)
```bash
python3 survival/train_survival_models.py --fold all --dataset all --scenario all --method all
```

Step 6: Aggregate results
```bash
python3 evaluation/aggregate_imputation_metrics.py
python3 evaluation/aggregate_survival_metrics.py
```

Step 7: Reproduce figures
```bash
# Optional: set environment variables for custom paths
export SVAE_RESULTS_DIR=/path/to/Survival-VAE_study/final
export SVAE_OUTPUT_DIR=/path/to/output/figures

python3 evaluation/reproduce_figure_imputation_fidelity.py
python3 evaluation/reproduce_figure_clinical_utility.py
```

## 6. MNAR Simulation
The threshold-interaction MNAR mechanism masks target variables with probability $P(M_{ij}=1 \mid X) = \sigma(\alpha \cdot [\mathbb{I}(X_{i,A} > \tau) \cdot X_{i,B}] + \lambda)$. For METABRIC, the interaction variable MKI67 is also a masked target variable, making the mechanism genuinely MNAR. For MIMIC-IV, lactate drives its own missingness. Simulation scripts are in `simulation/`.

## 7. Citation
```text
Nads, A., & Andrade, D. (under review). Survival-supervised variational autoencoders
for robust clinical prognosis under missing not at random conditions.
Scientific Reports.
```

## 8. License
MIT License
