# Survival-VAE: Survival-aware Variational Autoencoders for MNAR Data

Official implementation of "Survival-aware variational autoencoders for handling threshold-interaction missing not at random data in clinical prognosis: a two-stage validation study"

**Submitted to:** Journal of Biomedical Informatics (JBI)  
**Authors:** Azman Nads, Daniel Andrade

---

## Overview

Survival-VAE is a deep learning method that handles missing not at random (MNAR) data in clinical survival analysis by integrating survival-aware loss functions into a variational autoencoder framework.

**Key Features:**
- Addresses threshold-interaction MNAR mechanisms
- Integrates Cox partial likelihood loss for survival-aware imputation
- Validated on METABRIC (cancer) and MIMIC-IV (ICU) datasets
- Outperforms MICE, missForest, GAIN, and MIDA baselines

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.12+ (supports MPS for Mac M1/M2)
- scikit-learn 1.0+
- scikit-survival 0.17+
- pandas, numpy, seaborn

### Setup
```bash
# Clone repository
git clone https://github.com/aaazmnnds/survival-vae.git
cd survival-vae

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start
```python
from src.models.survival_vae import train_model, impute_dataset

# Train model (example on small dataset)
model = train_model(
    dataset_path="data/sample_mnar.csv",
    mask_path="data/sample_mask.csv",
    epochs=50,
    beta=0.0002,
    gamma=0.3
)

# Impute missing values
impute_dataset(model, "data/sample_mnar.csv", "data/sample_mask.csv", "output.csv")
```

---

## Datasets

### METABRIC
- **Access:** [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric)
- **Size:** 1,904 patients
- **Features:** 9 clinical/genomic features

### MIMIC-IV
- **Access:** [PhysioNet](https://physionet.org/content/mimiciv/2.2/)
- **Version:** v2.2
- **Size:** 10,733 ICU patients
- **Features:** 54 clinical features

**Note:** Due to data use agreements, raw datasets are not included. Researchers must obtain access through official sources.

---

## Reproducing Paper Results

### 1. Simulate MNAR Data
```bash
python src/missingness/metabric_mnar.py
python src/missingness/mimic_mnar.py
```

### 2. Run Imputation
```bash
# Survival-VAE
python src/models/survival_vae.py

# Baselines
python src/models/mice_baseline.py
python src/models/missforest_baseline.py
```

### 3. Evaluate & Visualize
```bash
python src/evaluation/imputation_metrics.py
python visualization/create_figure2.py
```

---

## Repository Structure
```
survival-vae/
├── src/models/          # Survival-VAE and baseline implementations
├── src/missingness/     # MNAR simulation
├── src/evaluation/      # Evaluation metrics
├── experiments/         # Scripts to reproduce paper results
└── visualization/       # Figure generation scripts
```

---

## Hyperparameters

Optimized hyperparameters using Optuna (as reported in the paper):

| Parameter | METABRIC | MIMIC-IV | Description |
|-----------|----------|----------|-------------|
| Latent dimension | 4 | 16 | Dimension of latent space |
| β (KL-weight) | 2e-4 | 2e-4 | KL divergence weight |
| γ (Cox-weight) | 0.27 | 0.48 | Survival loss weight |
| Learning Rate | 0.007 | 0.002 | Adam optimizer LR |
| Batch size | 64 | 64 | Training batch size |

---

## Citation

If you use this code in your research, please cite:
```bibtex
@article{nads2026survival,
  title={Survival-aware variational autoencoders for handling threshold-interaction missing not at random data in clinical prognosis: a two-stage validation study},
  author={Nads, Azman and Andrade, Daniel},
  journal={Submitted to Journal of Biomedical Informatics},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Contact

**Azman Nads**  
Mindanao State University  
Email: azmannads@msutawi-tawi.edu.ph

**Daniel Andrade**  
Hiroshima University  
Email: andrade@hiroshima-u.ac.jp

---

## Acknowledgments

This work was supported by the Department of Science and Technology–Science Education Institute (DOST-SEI) of the Philippines through its Foreign Graduate Scholarship Program. The authors gratefully acknowledge the Informatics and Data Science Program A1-427 at Hiroshima University for providing computational resources. We acknowledge the METABRIC consortium and the PhysioNet community for making their data publicly available for research purposes. We also acknowledge the use of large language models as a writing assistant for manuscript preparation.
