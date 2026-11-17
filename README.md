# RAPID-ER: Decoupling Discrimination from Calibration for Estrogen Receptor Phenotyping in Breast Cancer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

Official implementation and reproducible analysis pipeline for:
> **Decoupling Discrimination from Calibration: A Platform-Resilient, Uncertainty-Aware Transcriptomic Biomarker for Estrogen Receptor Phenotyping in Breast Cancer**[cite: 17]  
> Gautam Kunniyur Ramchander¹\*[cite: 17]  
> ¹Independent Bioinformatics Research Laboratory[cite: 17]  
> \*Corresponding Author: `gautamramchander7@gmail.com`[cite: 17]

---

## Overview

**RAPID-ER** addresses the **discrimination-calibration paradox** in computational oncology—where transcriptomic classifiers achieve high rank discrimination ($\text{AUROC} > 0.95$) yet output severely miscalibrated risk probabilities under cross-cohort and cross-platform domain shift[cite: 17].

Across **4,218 audited primary breast cancer profiles** spanning RNA-seq and microarray technologies, RAPID-ER pairs **100-iteration repeated Elastic-Net stability selection** (300 consensus genes) with a regularized **Stability-Selected Multilayer Perceptron (SS-MLP)**, **label-free within-sample fractional rank normalization**, and **confidence-gated selective abstention**[cite: 17].

### Key Benchmark Results
| Cohort | Platform | Transfer Strategy | Model | AUROC [95% CI] | AUPRC [95% CI] | Sens. | Spec. | Brier | ECE |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TCGA-BRCA** ($N=1,044$)[cite: 17] | RNA-seq (STAR)[cite: 17] | 5-Fold Internal CV[cite: 17] | **SS-MLP**[cite: 17] | 0.9560 [0.938, 0.970][cite: 17] | 0.9851 [0.977, 0.991][cite: 17] | 0.9207[cite: 17] | 0.9367[cite: 17] | 0.0675[cite: 17] | 0.0869[cite: 17] |
| **SCAN-B / GSE96058** ($N=3,073$)[cite: 17] | RNA-seq (FPKM)[cite: 17] | Strategy A (Frozen)[cite: 17] | *ESR1*-Only[cite: 17] | 0.9604 [0.944, 0.973][cite: 17] | 0.9947 [0.992, 0.997][cite: 17] | 0.5844[cite: 17] | 0.9710[cite: 17] | 0.2628[cite: 17] | 0.4290[cite: 17] |
| **SCAN-B / GSE96058** ($N=3,073$)[cite: 17] | RNA-seq (FPKM)[cite: 17] | Strategy A (Frozen)[cite: 17] | **SS-MLP**[cite: 17] | 0.9500 [0.933, 0.965][cite: 17] | 0.9928 [0.989, 0.995][cite: 17] | 0.9774[cite: 17] | 0.7303[cite: 17] | 0.0396[cite: 17] | 0.0396[cite: 17] |
| **GSE42568** ($N=101$)[cite: 17] | Microarray (GPL570)[cite: 17] | Strategy A (Frozen)[cite: 17] | *ESR1*-Only[cite: 17] | 0.7682 [0.667, 0.853][cite: 17] | 0.8667 [0.786, 0.929][cite: 17] | 0.0000[cite: 17] | 1.0000[cite: 17] | 0.6188[cite: 17] | 0.6313[cite: 17] |
| **GSE42568** ($N=101$)[cite: 17] | Microarray (GPL570)[cite: 17] | Strategy B (Rank)[cite: 17] | **SS-MLP**[cite: 17] | 0.7943 [0.701, 0.874][cite: 17] | 0.8773 [0.792, 0.942][cite: 17] | 0.9851[cite: 17] | 0.0000[cite: 17] | 0.3303[cite: 17] | 0.3362[cite: 17] |

* **Downstream Redundancy:** In silico *ESR1* exclusion prior to stability selection preserves diagnostic fidelity ($\text{AUROC} = 0.9608$, $\text{AUPRC} = 0.9863$) via downstream luminal co-regulators (*GATA3*, *GREB1*, *TFF1*, *CA12*, *AGR3*)[cite: 17].
* **Selective Clinical Triage:** Confidence abstention at $\tau = 0.80$ autonomously triages **99.6%** of external SCAN-B specimens at a **3.92%** error rate (dropping to **2.06%** error at $\tau = 0.95$)[cite: 17].

---

## Data Availability

All datasets used in this study are publicly available de-identified records[cite: 17]:
* **TCGA-BRCA ($N = 1,044$):** [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/projects/TCGA-BRCA) (STAR raw counts, Release 32.0)[cite: 17]
* **SCAN-B / GSE96058 ($N = 3,073$):** [NCBI GEO GSE96058](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058) ($\log_2(\text{FPKM}+0.1)$)[cite: 17]
* **GSE42568 ($N = 101$):** [NCBI GEO GSE42568](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42568) (Affymetrix HG-U133 Plus 2.0, $\log_2$ GC-RMA)[cite: 17]

---

## Citation

```bibtex
@article{ramchander2026rapider,
  title   = {Decoupling Discrimination from Calibration: A Platform-Resilient, Uncertainty-Aware Transcriptomic Biomarker for Estrogen Receptor Phenotyping in Breast Cancer},
  author  = {Ramchander, Gautam Kunniyur},
  year    = {2026},
  note    = {Independent Bioinformatics Research Laboratory. Code: [https://github.com/gautam453/RAPID-ER](https://github.com/gautam453/RAPID-ER)}
}
