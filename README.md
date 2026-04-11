# Survival-VAE: Imputation and Survival Analysis for MNAR Clinical Data

Official implementation for the study on **Survival-VAE**, a deep generative framework for handling informative missingness (Missing Not At Random) in clinical survival data.

## Authors
- **Azman Nads** (Primary Investigator)
- **Collaboration Team**

## Repository Structure

```text
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
├── utils/                  # Utility and pipeline runners
│   ├── create_cv_splits.py
│   ├── run_imputation_pipeline.py
│   ├── run_optuna_pipeline.py
│   └── run_survival_optimization_pipeline.py
├── requirements.txt        # Dependency versions
└── LICENSE
```

---

## 🚀 Usage

### 1. Data Preparation
First, generate the 5-fold cross-validation splits for the datasets:
```bash
python3 utils/create_cv_splits.py
```

### 2. Hyperparameter Optimization (Optuna)
Optimize hyperparameters for the imputation and survival models (performed on Fold 0):
```bash
python3 utils/run_optuna_pipeline.py --fold 0 --dataset all --scenario all
```

### 3. Imputation Pipeline
Once optimized, run the full imputation pipeline across all 5 folds:
```bash
python3 utils/run_imputation_pipeline.py --fold all --dataset all --scenario all
```

### 4. Survival Evaluation
Train and evaluate prognostic models (RSF, XGBoost, etc.) on the imputed datasets:
```bash
python3 survival/train_survival_models.py --fold all --dataset all --scenario all
```

### 5. Results Aggregation
Aggregate the results and generate LaTeX tables:
```bash
python3 evaluation/aggregate_survival_results.py --latex
```

---

## 📊 Data Availability
This study utilizes the following datasets:
- **METABRIC**: Molecular Taxonomy of Breast Cancer International Consortium (public).
- **MIMIC-IV**: Medical Information Mart for Intensive Care IV (requires PhysioNet credentialed access).

## 🛠 Requirements
Install dependencies using pip:
```bash
pip install -r requirements.txt
```
Key dependencies include `torch`, `scikit-survival`, `xgboost`, `optuna`, and `pycox`.

## 📜 Citation
*Citation placeholder - publication pending.*

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
We thank the open-source contributors of PyTorch, Optuna, and Scikit-Survival.
