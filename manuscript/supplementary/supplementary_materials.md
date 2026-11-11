# Supplementary Materials

## Cross-Cohort and Cross-Platform Evaluation of Stable, Calibrated Gene-Expression Signatures for Estrogen Receptor Status Prediction in Breast Cancer

**Authors:** Kunniyur Ramchander Gautam et al.  
**Affiliation:** Independent Bioinformatics Research Laboratory  
**Corresponding Contact:** research@rapid-er.bio  
**Date:** November 2026  

---

## Supplementary Note 1: Mathematical & Statistical Formulations

### 1.1 Meinshausen–Bühlmann Stability Selection
High-dimensional transcriptomic datasets ($p \approx 15,200 \gg n \approx 835$ training patients) present severe feature selection variance: standard LASSO and Elastic-Net estimators are notoriously sensitive to small sample perturbations. To isolate genuine biological drivers from noise, we implemented stability selection (Meinshausen & Bühlmann, 2010):

Let $\mathcal{D}_{\text{train}} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ represent the training partition. For $B = 100$ bootstrap subsampling rounds:
1. Draw a random subsample $\mathcal{I}_b \subset \{1, \dots, n\}$ of size $\lfloor 0.80 \cdot n \rfloor$ without replacement.
2. Fit an Elastic-Net regularized logistic regression model on $\mathcal{I}_b$:
   $$\hat{\boldsymbol{\beta}}^{(b)} = \arg\min_{\boldsymbol{\beta}} \left\{ -\frac{1}{|\mathcal{I}_b|} \sum_{i \in \mathcal{I}_b} \left[ y_i \log \sigma(\mathbf{x}_i^\top \boldsymbol{\beta}) + (1 - y_i) \log(1 - \sigma(\mathbf{x}_i^\top \boldsymbol{\beta})) \right] + \lambda \left( \alpha \|\boldsymbol{\beta}\|_1 + \frac{1-\alpha}{2} \|\boldsymbol{\beta}\|_2^2 \right) \right\}$$
   with $\alpha = 0.5$ (equal balance of L1/L2) and $C = 1/\lambda = 0.1$.
3. Compute the empirical selection frequency for each candidate gene $j \in \{1, \dots, p\}$:
   $$\Pi_j = \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\hat{\beta}_j^{(b)} \ne 0)$$
4. Retain all genes satisfying $\Pi_j \ge \pi_{\text{thresh}} = 0.70$, subject to an upper budget cap $K = 300$:
   $$\mathcal{S} = \{j : \Pi_j \ge 0.70\}$$

### 1.2 Probability Calibration and Murphy–Sanders Brier Decomposition
For true binary labels $y_i \in \{0, 1\}$ and predicted probabilities $p_i \in [0, 1]$, the total mean Brier score is decomposed into three mutually orthogonal components:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2 = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$

Grouping predictions into $M = 10$ bins $B_m$ with empirical event frequencies $\bar{y}_m = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$ and mean predicted probabilities $\bar{p}_m = \frac{1}{|B_m|} \sum_{i \in B_m} p_i$, the base rate is $\bar{y} = \frac{1}{N} \sum_{i=1}^N y_i$:
- **Reliability ($\text{REL}$):** Measures probability calibration error (lower is better; 0 indicates perfect calibration):
  $$\text{REL} = \sum_{m=1}^M \frac{|B_m|}{N} (\bar{p}_m - \bar{y}_m)^2$$
- **Resolution ($\text{RES}$):** Measures discriminatory separation across bins (higher is better):
  $$\text{RES} = \sum_{m=1}^M \frac{|B_m|}{N} (\bar{y}_m - \bar{y})^2$$
- **Uncertainty ($\text{UNC}$):** Represents irreducible variance determined strictly by class prevalence:
  $$\text{UNC} = \bar{y}(1 - \bar{y})$$

### 1.3 Expected Calibration Error (ECE)
$$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} |\bar{y}_m - \bar{p}_m|$$

### 1.4 Logistic Calibration Intercept and Slope
The logistic calibration curve regresses true binary outcomes on log-odds:
$$\log\left(\frac{P(y=1)}{1 - P(y=1)}\right) = a + b \cdot \text{logit}(p_i)$$
where $a$ is the calibration intercept (ideal $a = 0$; $a > 0$ denotes systematic underestimation) and $b$ is the calibration slope (ideal $b = 1$; $b < 1$ indicates overconfidence in extreme predictions).

### 1.5 Selective Classification and Uncertainty Quantification
For a confidence threshold $\tau \in [0.50, 0.99]$, prediction confidence is defined as:
$$c_i = \max(p_i, 1 - p_i)$$
The decision rule defines acceptance $A_i$ and selective error rate $\mathcal{E}(\tau)$:
$$A_i = \mathbb{I}(c_i \ge \tau)$$
$$\text{Coverage}(\tau) = \frac{1}{N} \sum_{i=1}^N A_i$$
$$\mathcal{E}(\tau) = \frac{\sum_{i=1}^N A_i \cdot \mathbb{I}(\hat{y}_i \ne y_i)}{\sum_{i=1}^N A_i}$$

---

## Supplementary Note 2: Microarray Probe Collapsing and GC-RMA Characteristics

### 2.1 Affymetrix GPL570 Annotation Mapping
GSE42568 was assayed on the Affymetrix Human Genome U133 Plus 2.0 array (`GPL570`). Probe set identifiers (54,675 total probes) were mapped to HGNC gene symbols using the official GPL570 annotation table.
- Non-specific control probe sets (e.g., `AFFX-...`) were removed.
- Ambiguous probe sets mapping to multiple distinct HGNC symbols (e.g., `LOC...///...`) were purged.
- When multiple probe sets mapped to the identical HGNC symbol, expression values were collapsed using a pre-specified median expression rule across probes within each sample.

### 2.2 Scale Mismatch between Microarray and RNA-seq
Affymetrix GC-RMA preprocessing outputs $\log_2$ background-adjusted, quantile-normalized fluorescence intensities on a continuous scale typically spanning $[2.0, 15.0]$. In contrast, RNA-seq counts normalized via library size scaling ($\log_2(\text{CPM} + 1)$) span $[0.0, 18.0]$ with substantial zero-inflation for unexpressed transcripts. Under Transfer Strategy A (freezing training RNA-seq means and standard deviations), the dynamic range difference causes the linear inputs to shift drastically toward negative values, yielding false-negative classifications at standard decision thresholds. Transfer Strategy B (within-sample fractional rank normalization) completely maps both platforms to a uniform relative scale $[0, 1]$, bypassing platform-specific baseline shifts.

---

## Supplementary Note 3: Computational Environment & Hardware Profile

All experiments were executed on an independent research workstation with the following specifications:
- **Operating System:** Microsoft Windows 11 Home (x86_64)
- **Central Processing Unit (CPU):** Intel / AMD 16-thread multi-core processor
- **System Memory (RAM):** 16.0 GB DDR5
- **Dedicated Graphics Processor:** NVIDIA GeForce RTX 5050 Laptop GPU (8.0 GB VRAM, Blackwell architecture `sm_120`)
- **Software Stack:** Python 3.11.11 in Conda virtual environment `rapid_er`, PyTorch 2.6.0+cu124, scikit-learn 1.6.1, XGBoost 2.1.4, GEOparse 2.0.4, GSEAPy 1.1.5, pandas 2.2.3, numpy 2.2.3.
- **Hardware Architecture Note:** PyTorch standard pre-built binaries (cu124) do not bundle precompiled PTX kernels for the newest Blackwell GPU architecture (`sm_120`). Our codebase incorporates an automated device probe in `src/models.py:SSMLPClassifier._get_device()` that safely falls back to CPU execution. Because SS-MLP is a lightweight 2-hidden-layer network (256-64 units), CPU training requires only ~2.5 seconds per fold on 1,044 samples, ensuring 100% reproducible execution on standard consumer laptops without external compute servers.

---

## Supplementary Tables Summary

### Supplementary Table S1: Complete 300-Gene Consensus Molecular Signature
File: `manuscript/supplementary/Supplementary_Table_S1_Signature_300_Genes.csv`  
Contains all 300 stability-selected genes, empirical selection frequencies (all 100%), mean Elastic-Net coefficient weights, and biological functional descriptions. Key luminal drivers include *ESR1*, *CERS3*, *LYPD1*, *AGR3*, *GREB1*, *TFF1*, *GATA3*, *CA12*, *SLC39A6*, and *AGR2*.

### Supplementary Table S2: Complete Enrichr Pathway Library Results
File: `manuscript/supplementary/Supplementary_Table_S2_Enrichr_Pathways.csv`  
Full tabular output of 2,158 statistically significant terms ($q < 0.05$) across Gene Ontology Biological Processes, KEGG, and Reactome libraries. Leading categories involve Estrogen-Dependent Gene Expression ($p = 2.15 \times 10^{-14}$), Response to Steroid Hormone ($p = 4.88 \times 10^{-12}$), and Luminal Cell Differentiation.

### Supplementary Table S3: Per-Fold Internal Cross-Validation Metrics
File: `manuscript/supplementary/Supplementary_Table_S3_Per_Fold_CV_Metrics.csv`  
Detailed performance metrics (AUROC, AUPRC, Brier score, sample counts, positive/negative ratios) for each of the 5 cross-validation folds across all 8 evaluated machine learning models in TCGA-BRCA ($N = 1,044$).

### Supplementary Table S4: Selective Classification Coverage and Risk Trajectories
File: `manuscript/supplementary/Supplementary_Table_S4_Selective_Classification.csv`  
Empirical patient coverage, accepted sample counts, deferred sample counts, and selective classification error rates across confidence thresholds $\tau \in [0.50, 0.99]$ for TCGA-BRCA and GSE96058.

### Supplementary Table S5: Full Ablation Studies Results
File: `manuscript/supplementary/Supplementary_Table_S5_Ablation_Studies.csv`  
Comprehensive performance and calibration metrics across all tested ablation configurations:
1. *ESR1* Exclusion Ablation (AUROC 0.9608, AUPRC 0.9863, Brier 0.0657)
2. Gene Count Capacity Ablation (50, 100, 200, 300 genes)
3. Dropout Regularization Ablation (0.0/0.0, 0.15/0.10, 0.35/0.25, 0.50/0.40)
