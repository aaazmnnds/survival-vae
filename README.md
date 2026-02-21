# Survival-VAE: Survival-aware Variational Autoencoders for MNAR Data

Official implementation of "Survival-aware variational autoencoders for handling threshold-interaction missing not at random data in clinical prognosis: a two-stage validation study"

**Published in:** Artificial Intelligence in Medicine  
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
- PyTorch 1.10+
- scikit-learn 1.0+
- scikit-survival 0.17+

### Setup
```bash
# Clone repository
git clone https://github.com/[username]/survival-vae.git
cd survival-vae

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

---

## Quick Start
```python
from src.models.survival_vae import SurvivalVAE

# Initialize model
model = SurvivalVAE(
    input_dim=57,
    latent_dim=10,
    beta=0.01,
    gamma=1.0
)

# Train model
model.fit(X_train, time_train, event_train)

# Impute missing values
X_imputed = model.impute(X_test)
```

---

## Datasets

### METABRIC
- **Access:** [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric)
- **Size:** 1,904 patients
- **Features:** 9 clinical/genomic features
- **Ethics:** Publicly available for research

### MIMIC-IV
- **Access:** [PhysioNet](https://physionet.org/content/mimiciv/2.2/)
- **Version:** v2.2
- **Size:** 10,733 ICU patients
- **Features:** 57 clinical features
- **Requirements:** CITI training certificate required

**Note:** Due to data use agreements, raw datasets are not included. Researchers must obtain access through official sources.

---

## Reproducing Paper Results

### METABRIC Experiments (Stage A)
```bash
python experiments/run_metabric.py \
    --missingness light \
    --method survival_vae \
    --cv_folds 5
```

### MIMIC-IV Experiments (Stage B)
```bash
python experiments/run_mimic.py \
    --missingness severe \
    --method survival_vae \
    --cv_folds 5
```

### Generate All Figures
```bash
python visualization/create_figure2.py  # RMSE comparison
python visualization/create_figure3.py  # MAE comparison
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

Key hyperparameters from the paper (Appendix Table A.2):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Latent dimension | 10 | Dimension of latent space |
| β | 0.01 | KL divergence weight |
| γ | 1.0 | Cox loss weight |
| Training epochs | 50 | Number of training iterations |
| Batch size | 64 | Training batch size |

See `experiments/config.yaml` for complete settings.

---

## Citation

If you use this code in your research, please cite:
```bibtex
@article{nads2025survival,
  title={Survival-aware variational autoencoders for handling threshold-interaction missing not at random data in clinical prognosis: a two-stage validation study},
  author={Nads, Azman and Andrade, Daniel},
  journal={Artificial Intelligence in Medicine},
  year={2025},
  publisher={Elsevier}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Contact

**Azman Nads**  
Department of Statistics, Mindanao State University  
Email: azmannads@msutawi-tawi.edu.ph

**Daniel Andrade**  
Hiroshima University  
Email: andrade@hiroshima-u.ac.jp

---

## Acknowledgments

This research was supported by the DOST-SEI Foreign Graduate Scholarship Program. We acknowledge the use of Claude (Anthropic) as a writing assistant for manuscript preparation.
