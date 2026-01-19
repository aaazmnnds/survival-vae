# Dataset Access Instructions

## METABRIC Dataset

**Source:** cBioPortal  
**URL:** https://www.cbioportal.org/study/summary?id=brca_metabric  
**Access:** Free registration required  
**Citation:** Curtis et al. (2012) Nature

**After obtaining access:**
1. Download clinical and genomic data
2. Place in `data/metabric/` directory
3. Run preprocessing: `python data/metabric_preprocessing.py`

---

## MIMIC-IV Dataset

**Source:** PhysioNet  
**URL:** https://physionet.org/content/mimiciv/2.2/  
**Version:** v2.2  
**Access Requirements:**
- Complete CITI "Data or Specimens Only Research" course
- Sign data use agreement
- Obtain approval from your institution

**After obtaining access:**
1. Download MIMIC-IV v2.2
2. Place in `data/mimic_iv/` directory
3. Run preprocessing: `python data/mimic_preprocessing.py`

---

## Important Notes

⚠️ **Privacy:** We cannot provide raw datasets due to data use agreements  
⚠️ **Ethics:** Ensure you have appropriate IRB approval  
⚠️ **Citation:** Please cite original data sources in your work
