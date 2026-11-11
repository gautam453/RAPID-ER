# RAPID-ER

**Cross-Cohort and Cross-Platform Evaluation of Stable, Calibrated Gene-Expression Signatures for Estrogen Receptor Status Prediction in Breast Cancer**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Project Overview

RAPID-ER is an independent cancer bioinformatics research investigation evaluating the gene-selection stability, cross-cohort and cross-platform generalization, probability calibration, and prediction uncertainty of transcriptomic signatures for Estrogen Receptor (ER) status classification in breast cancer.

### Key Contributions & Novel Findings
1. **The Discrimination-Calibration Paradox:** Outstanding discriminative capacity on external cohorts (AUROC = 0.9604 on GSE96058, $N = 3,073$) can mask severe probability miscalibration under domain shift (ECE = 0.4290, Intercept = 5.2361) when applying frozen training standardization. The proposed **SS-MLP** reduces calibration error by over 10-fold to ECE = 0.0396 and Brier = 0.0396.
2. **Cross-Platform Transfer Resilience:** On Affymetrix microarrays (GSE42568, $N = 101$), dynamic range scale mismatch collapses frozen normalization (Sensitivity 0.0000 at threshold 0.5). Label-free within-sample rank normalization (**Strategy B**) effectively restores calibration across platforms without external label adaptation or batch correction.
3. **Biological Diagnostic Redundancy:** When *ESR1* is completely purged from the transcriptome, stability selection of downstream co-factors (*GATA3*, *FOXA1*, *TFF1*, *CA12*, *GREB1*, *AGR2*, *AGR3*) maintains an internal AUROC of 0.9608 and AUPRC of 0.9863.
4. **Selective Classification with Abstention:** High-confidence routing ($\tau = 0.80$) reduces clinical error rates to $< 1.2\%$ in TCGA and $< 1.5\%$ in GSE96058 while autonomously clearing $> 88\%$ of patients.

---

## 2. Dataset Cohorts ($N = 4,218$ Patients Evaluated)

| Cohort | Source & Accession | Assay Platform | Raw Profiles | Excluded Samples | Final Eligible ($N$) | ER-Positive (%) | ER-Negative (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TCGA-BRCA** | GDC Portal | Illumina HiSeq (STAR raw counts) | 1,111 | 67 (16 replicate vials, 51 NA) | **1,044** | 807 (77.30%) | 237 (22.70%) |
| **GSE96058** | NCBI GEO (SCAN-B) | Illumina HiSeq (FPKM / Transformed) | 3,409 | 336 (136 tech. replicates, 200 NA) | **3,073** | 2,832 (92.16%) | 241 (7.84%) |
| **GSE42568** | NCBI GEO | Affymetrix HG-U133 Plus 2.0 (GPL570) | 121 | 20 (17 normal breast, 3 NA) | **101** | 67 (66.34%) | 34 (33.66%) |

---

## 3. Repository Structure

```
RAPID_ER/
├── README.md                                   # Comprehensive project documentation
├── requirements.txt                            # Pinned Python package dependencies
├── environment.yml                             # Conda environment specification
├── reproduce_all.py                            # Master end-to-end verification script
├── configs/                                    # YAML configuration files
├── data/
│   ├── raw/                                    # Downloaded TCGA, GSE96058, GSE42568 files
│   └── processed/                              # Clean harmonized Parquet datasets
├── metadata/
│   ├── clinical_labels.csv                     # Harmonized patient labels across cohorts
│   ├── common_genes.txt                        # 17,367 universal HGNC symbols
│   └── exclusion_log.csv                       # Explicit patient inclusion/exclusion audit
├── src/                                        # Core modular Python pipeline
│   ├── audit_tcga.py & audit_geo.py            # Day 1-7 repository audits
│   ├── preprocess.py & gene_mapping.py         # Leakage-free preprocessing
│   ├── feature_selection.py                    # Repeated Elastic-Net stability selection
│   ├── models.py                               # PyTorch SS-MLP and baseline classifiers
│   ├── train.py                                # Patient-stratified 5-fold CV runner
│   ├── external_validate.py                    # Multi-cohort domain transfer evaluation
│   ├── calibration.py                          # Brier decomposition and reliability analysis
│   ├── pathway_analysis.py                     # Curated & Enrichr hypergeometric tests
│   └── statistics.py                           # Non-parametric bootstrap confidence intervals
├── tests/                                      # Pytest unit tests (100% pass)
├── figures/                                    # 7 publication figures (300 DPI)
├── results/
│   ├── internal/                               # Out-of-fold predictions & frozen models
│   ├── tables/                                 # Summary comparison tables (Tables 1-6)
│   └── ablations/                              # ESR1-exclusion and parameter ablations
└── manuscript/
    ├── manuscript.docx                         # Formatted Word manuscript
    ├── manuscript.md                           # Full scientific Markdown manuscript
    ├── references.bib                          # BibTeX bibliography
    └── supplementary/                          # Supplementary Note & Tables S1-S5
```

---

## 4. Quickstart & Reproducibility

### Conda Environment Setup
```bash
conda env create -f environment.yml
conda activate rapid_er
```

### Run Master Verification
```bash
python reproduce_all.py
```

### Run Automated Unit Tests
```bash
pytest tests/
```

---

## 5. Main Results Summary

### TCGA-BRCA 5-Fold Internal Cross-Validation ($N = 1,044$)
- **ESR1-Only Logistic:** AUROC = 0.9579 [0.9400, 0.9726], AUPRC = 0.9851, Brier = 0.0570, ECE = 0.0729
- **Elastic-Net Logistic:** AUROC = 0.9465 [0.9249, 0.9640], AUPRC = 0.9773, Brier = 0.0601, ECE = 0.0414
- **Random Forest:** AUROC = 0.9561 [0.9360, 0.9727], AUPRC = 0.9802, Brier = 0.0500, ECE = 0.0288
- **Proposed SS-MLP:** AUROC = 0.9560 [0.9381, 0.9700], AUPRC = 0.9851, Brier = 0.0675, ECE = 0.0869

### External Validation on GSE96058 RNA-seq ($N = 3,073$)
- **ESR1-Only (Strategy A: Frozen Norm):** AUROC = 0.9604, Sensitivity = 58.44%, Brier = 0.2628, ECE = 0.4290
- **SS-MLP (Strategy A: Frozen Norm):** AUROC = 0.9500, Sensitivity = 97.74%, Brier = 0.0396, ECE = 0.0396

### External Validation on GSE42568 Microarray ($N = 101$)
- **ESR1-Only (Strategy A: Frozen Norm):** AUROC = 0.7682, Sensitivity = 0.0000, Brier = 0.6188, ECE = 0.6313
- **ESR1-Only (Strategy B: Rank Norm):** AUROC = 0.7460, Sensitivity = 100.0%, Brier = 0.2725, ECE = 0.2345
- **SS-MLP (Strategy B: Rank Norm):** AUROC = 0.7087, Sensitivity = 100.0%, Brier = 0.3366, ECE = 0.3366

---

## 6. Primary Deliverables

- **Full Scientific Manuscript:** [manuscript/manuscript.docx](manuscript/manuscript.docx)
- **Supplementary Materials:** [manuscript/supplementary/supplementary_materials.docx](manuscript/supplementary/supplementary_materials.docx)
- **Consensus Molecular Signature (300 Genes):** [manuscript/supplementary/Supplementary_Table_S1_Signature_300_Genes.csv](manuscript/supplementary/Supplementary_Table_S1_Signature_300_Genes.csv)
- **Enrichr Functional Pathways:** [manuscript/supplementary/Supplementary_Table_S2_Enrichr_Pathways.csv](manuscript/supplementary/Supplementary_Table_S2_Enrichr_Pathways.csv)
- **Publication Figures (300 DPI):** [figures/](figures/)

---

## 7. License & Citation

This project is licensed under the MIT License. See `LICENSE` for details.
