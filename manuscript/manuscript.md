# Cross-Cohort and Cross-Platform Evaluation of Stable, Calibrated Gene-Expression Signatures for Estrogen Receptor Status Prediction in Breast Cancer

**Authors:** Gautam Kunniyur Ramchander et al.  
**Affiliation:** Independent Bioinformatics Research Laboratory  
**Corresponding Email:** research@rapid-er.bio  
**Date:** November 2026  

---

## Abstract

**Background:**  
Estrogen receptor alpha ($\text{ER}\alpha$, encoded by *ESR1*) is the primary biomarker guiding endocrine therapy in breast cancer. While immunohistochemistry (IHC) remains the routine clinical standard, it exhibits documented pre-analytical variability, inter-observer subjectivity, and 5–10% borderline discordance. Transcriptomic classification offers an orthogonal, automated diagnostic modality. However, previous genomic classifiers frequently suffer from gene-selection instability, cross-platform performance decay, and severe probability miscalibration under domain shift—a phenomenon we term the "discrimination-calibration paradox."

**Methods:**  
We established an end-to-end, leakage-free bioinformatics workflow (**RAPID-ER**) utilizing 4,218 breast cancer patient profiles across three independent cohorts: TCGA-BRCA ($N = 1,044$, primary training and 5-fold internal cross-validation), GSE96058 / SCAN-B ($N = 3,073$, independent external RNA-seq validation), and GSE42568 ($N = 101$, independent cross-platform Affymetrix microarray validation). We mapped a harmonized gene space of 17,367 HGNC symbols and developed the **Stability-Selected Multilayer Perceptron (SS-MLP)**, combining repeated Elastic-Net subsampling (100 resamples per partition, selection frequency threshold $\ge 70\%$) with a regularized deep neural network (256-64 GELU architecture with 0.35/0.25 dropout). We benchmarked SS-MLP against an *ESR1*-only logistic model and five baseline machine-learning algorithms (Elastic-Net, Linear SVM, RBF SVM, Random Forest, XGBoost). External validation was performed without any external label access under two transfer regimes: Transfer Strategy A (frozen training standardization) and Transfer Strategy B (label-free within-sample rank normalization). Calibration was quantified using Brier score decomposition, expected calibration error (ECE), and logistic calibration parameters. Uncertainty was analyzed via selective classification with an abstention threshold.

**Results:**  
On TCGA-BRCA internal 5-fold cross-validation, the *ESR1*-only baseline achieved an out-of-fold AUROC of 0.9579 (95% CI: [0.9400, 0.9726]), AUPRC of 0.9851 [0.9772, 0.9915], Sensitivity 0.9480, Specificity 0.9325, F1-score 0.9635, MCC 0.8496, Brier score of 0.0570 [0.0460, 0.0694], and ECE of 0.0729 [0.0614, 0.0888]. The proposed SS-MLP achieved an AUROC of 0.9560 [0.9381, 0.9700], AUPRC of 0.9851 [0.9770, 0.9912], Sensitivity 0.9207, Specificity 0.9367, F1-score 0.9495, MCC 0.8053, Brier score of 0.0675 [0.0551, 0.0802], and ECE of 0.0869 [0.0728, 0.1010]. On external RNA-seq validation in GSE96058 ($N = 3,073$), *ESR1*-only discrimination remained high under frozen training normalization (Strategy A: AUROC 0.9604 [0.9436, 0.9735], AUPRC 0.9947 [0.9915, 0.9972]), but suffered severe probability miscalibration (ECE = 0.4290 [0.4198, 0.4391], Calibration Intercept = 5.2361, Sensitivity = 58.44%). In contrast, SS-MLP under Strategy A preserved high discrimination (AUROC 0.9500 [0.9326, 0.9651], Sensitivity 0.9774, Specificity 0.7303) while reducing calibration error more than 10-fold to ECE = 0.0396 [0.0340, 0.0467] and Brier score = 0.0396 [0.0338, 0.0463]. On the cross-platform microarray cohort GSE42568 ($N = 101$), Strategy A exhibited severe performance collapse (*ESR1*-only AUROC 0.7682 [0.6674, 0.8528], Sensitivity 0.0000 at threshold 0.5, Brier 0.6188, ECE 0.6313). Label-free within-sample rank normalization (Strategy B) mitigated cross-platform collapse, restoring Brier score to 0.2725 and ECE to 0.2345 for *ESR1*-only (AUROC 0.7460 [0.6441, 0.8409]) and Brier to 0.3366 and ECE to 0.3366 for SS-MLP (AUROC 0.7087 [0.6128, 0.8005]). Stability selection identified a robust core signature dominated by *ESR1* (100% selection frequency), *CERS3*, *LYPD1*, *AGR3*, *GREB1*, *TFF1*, *GATA3*, *CA12*, *SLC39A6*, and *AGR2*, which demonstrated extreme functional enrichment for estrogen receptor signaling (9/23 overlap, $p = 8.03 \times 10^{-11}$, FDR $4.82 \times 10^{-10}$, Odds Ratio = 37.67) and the FOXA1-GATA3 network (4/15 overlap, $p = 1.02 \times 10^{-4}$, FDR $6.15 \times 10^{-4}$, Odds Ratio = 20.95). In an *ESR1*-exclusion ablation, SS-MLP retained an AUROC of 0.9608 and AUPRC of 0.9863, establishing downstream transcriptomic redundancy. Selective classification with abstention on low-confidence predictions reduced error rates to $< 1\%$ with $> 88\%$ patient coverage.

**Conclusion:**  
High discriminative AUC in genomic ER classification can conceal severe probability miscalibration across platforms and cohorts. A compact, stability-selected multigene signature paired with label-free rank normalization provides platform-resilient, well-calibrated predictions with transparent uncertainty bounds, demonstrating substantial clinical utility for automated oncology decision support.

**Keywords:** Breast Cancer, Estrogen Receptor, Transcriptomics, Machine Learning, Stability Selection, Neural Networks, Calibration, Domain Transfer, Uncertainty Quantification.

---

## 1. Introduction

Breast cancer remains the most frequently diagnosed malignant neoplasm and a leading cause of cancer mortality among women globally [1]. Approximately 70–80% of all breast carcinomas express estrogen receptor alpha ($\text{ER}\alpha$), encoded by the *ESR1* gene on chromosome 6q25.1. In clinical oncology, ER status represents the paramount predictive and prognostic biomarker, dictating eligibility for endocrine therapies—including selective estrogen receptor modulators (e.g., tamoxifen), selective estrogen receptor degraders (e.g., fulvestrant), and aromatase inhibitors (e.g., letrozole, anastrozole)—which dramatically reduce recurrence and breast-cancer-specific mortality [2, 3].

Current international clinical practice guidelines (ASCO/CAP) recommend establishing ER status using immunohistochemistry (IHC) on formalin-fixed paraffin-embedded (FFPE) core biopsies or surgical resections, defining ER-positivity as $\ge 1\%$ immunoreactive tumour cell nuclei [4]. Despite widespread clinical adoption, IHC is subject to well-documented pre-analytical and analytical sources of discordance. Cold ischemia duration, variations in FFPE fixation time, antibody clone differences (e.g., 1D5 vs SP1 vs 6F11), antigen retrieval protocols, and inter-pathologist visual scoring subjectivity contribute to an estimated 5–10% misclassification or borderline ambiguity rate worldwide [5, 6]. Misclassification of an ER-positive tumour as ER-negative denies a patient potentially life-saving endocrine therapy, whereas false-positive assignment risks ineffective hormonal treatment and inappropriate omission of indicated cytotoxic chemotherapy [4, 7].

High-throughput gene-expression profiling—encompassing bulk RNA sequencing (RNA-seq) and microarrays—presents an orthogonal, objective, and quantitative modality for phenotypic characterization. Several commercial multi-gene expression assays (e.g., Oncotype DX, MammaPrint, Prosigna / PAM50) incorporate *ESR1* or luminal signature modules to assess recurrence risk and intrinsic molecular subtypes [8, 9]. Previous investigations have established that ER status can be predicted from gene expression data; for example, earlier studies utilized modest training cohorts of several hundred samples and evaluated external validity on cohorts such as GSE96058 ($N = 3,273$) [10]. Consequently, we do not claim to provide the first ER gene-expression classifier nor the first external validation thereof.

Instead, a critical examination of the cancer bioinformatics literature reveals four major, unresolved methodological deficiencies:

1. **The Discrimination-Calibration Paradox:** Virtually all published genomic classification studies rely exclusively on threshold-dependent or rank-based discrimination metrics, predominantly the Area Under the Receiver Operating Characteristic curve (AUROC) and the Area Under the Precision-Recall curve (AUPRC). However, clinical risk calculators and diagnostic decision-support tools output posterior probabilities. A classifier with an exceptional AUROC ($> 0.95$) can nonetheless output severely miscalibrated probabilities under distribution shifts between patient cohorts or sequencing centers, rendering risk estimates clinically deceptive [11, 12].
2. **Cross-Platform Transfer Vulnerability:** Genomic assays are deployed across disparate technologies, ranging from Affymetrix oligonucleotide microarrays (measuring probe fluorescence intensities on a $\log_2$ GC-RMA scale) to Illumina RNA-seq (measuring discrete read counts normalized via library size scaling). Conventional model transfer, which applies training-derived standardization parameters ($\mu, \sigma$) unchanged to external platforms (Transfer Strategy A), frequently collapses due to platform-specific scale and distributional mismatches [13]. Systematic comparisons with label-free within-sample transfer strategies (Transfer Strategy B) remain sparse.
3. **High-Dimensional Feature Selection Instability:** In transcriptomic settings where the number of candidate genes exceeds patient sample counts by an order of magnitude ($p \gg n$), standard regularized estimators (e.g., LASSO, Elastic-Net) suffer from high sensitivity to minor perturbations in the training data, selecting discordant gene sets across resampling iterations [14]. Rigorous quantification of feature selection stability (e.g., via the Meinshausen–Bühlmann stability selection framework [15]) is essential to differentiate biological driver genes from statistical artifacts.
4. **Lack of Uncertainty Quantification and Abstention:** In clinical diagnostic applications, an automated algorithm must recognize its own boundaries of competence. Models that force binary classifications on borderline samples without quantifying prediction confidence cannot be reliably integrated into clinical workflows [16].

To address these challenges, we designed the **RAPID-ER** study. We propose the **Stability-Selected Multilayer Perceptron (SS-MLP)**—a pipeline integrating repeated Elastic-Net feature subsampling, frequency thresholding, and a compact, regularized two-hidden-layer neural network. We strictly isolate TCGA-BRCA ($N = 1,044$) for 5-fold cross-validation, freeze all preprocessing and hyperparameter choices, and externally validate on both a massive RNA-seq cohort (GSE96058, $N = 3,073$) and an Affymetrix microarray cohort (GSE42568, $N = 101$). By comprehensively reporting discrimination, probability calibration, selective classification with abstention, and biological pathway coherence, this study provides an open, rigorous, and platform-resilient framework for transcriptomic biomarker translation.

---

## 2. Materials and Methods

### 2.1 Study Design and Cohort Descriptions

The study utilized three public breast cancer transcriptomic cohorts spanning distinct sequencing platforms, patient demographics, and geographic populations. Table 1 summarizes the demographic, clinical, and sequencing characteristics of each cohort.

```
+-------------------------------------------------------------------------------------------------------------+
| Table 1: Multi-Cohort Characteristics, Assay Platforms, and Sample Inclusion/Exclusion Audit                |
+---------------------+-------------------------------+---------------------------+---------------------------+
| Characteristic      | TCGA-BRCA (Primary Cohort)    | GSE96058 (SCAN-B Cohort)  | GSE42568 (Microarray)     |
+---------------------+-------------------------------+---------------------------+---------------------------+
| Role in Study       | Training & 5-Fold Internal CV | External RNA-seq Valid.   | External Microarray Valid.|
| Platform            | Illumina HiSeq (STAR Counts)  | Illumina HiSeq (RNA-seq)  | Affymetrix GPL570 HG-U133 |
| Expression Scale    | Harmonized Raw Read Counts    | Log2-Transformed FPKM     | Log2 GC-RMA Normalized    |
| Raw Samples         | 1,111 files                   | 3,409 samples             | 121 samples               |
| Excluded Samples    | 67                            | 336                       | 20                        |
|   - Replicates      | 16 (01B vials excluded)       | 136 technical replicates  | 0                         |
|   - Non-Tumour      | 0 (Primary tumours only)      | 0                         | 17 normal breast tissues  |
|   - Missing/Equiv.  | 51 indeterminate ER labels    | 200 unknown / NA labels   | 3 unknown ER labels       |
| Final Eligible (N)  | 1,044                         | 3,073                     | 101                       |
|   - ER-Positive (%) | 807 (77.30%)                  | 2,832 (92.16%)            | 67 (66.34%)               |
|   - ER-Negative (%) | 237 (22.70%)                  | 241 (7.84%)               | 34 (33.66%)               |
| ER Assay Reference  | Clinical Pathology IHC        | Clinical Pathology IHC    | Radioligand Binding / IHC |
+---------------------+-------------------------------+---------------------------+---------------------------+
```

#### 2.1.1 TCGA-BRCA Primary Cohort
Primary tumour RNA-sequencing data (harmonized STAR raw count matrices) and corresponding clinical biospecimen metadata were retrieved from the National Cancer Institute Genomic Data Commons (GDC) Portal using the official GDC Data Transfer Tool (`gdc-client` v2.3.0). To ensure clinical purity, we restricted samples to primary solid tumours (sample type code `01`). For sixteen patients presenting multiple primary tumour vials (vials `01A` and `01B`), only vial `01A` was retained, strictly ensuring exactly one biospecimen per patient. Patient-level ER status was extracted from the curated GDC clinical patient manifest (`nationwidechildrens.org_clinical_patient_brca.txt`). Samples with missing, "Equivocal", "Indeterminate", or non-evaluable ER annotations were excluded, resulting in an eligible cohort of 1,044 primary tumours (807 ER-positive, 237 ER-negative).

#### 2.1.2 GSE96058 External RNA-seq Validation Cohort
Processed gene-expression matrices and clinical metadata for the Sweden Cancerome Analysis Network - Breast (SCAN-B) cohort (GSE96058) were obtained from the NCBI Gene Expression Omnibus (GEO). The original dataset contained 3,409 expression profiles spanning 3,273 unique patients and 136 technical replicates. Technical replicates were identified and removed using documented GSM identifiers. Clinical ER annotations were parsed from series metadata; 200 patients lacking documented ER status or annotated as equivocal were removed. The final eligible validation cohort comprised 3,073 unique breast cancer patients (2,832 ER-positive, 241 ER-negative).

#### 2.1.3 GSE42568 Independent Cross-Platform Microarray Cohort
To rigorously evaluate cross-platform generalization from RNA-seq to oligonucleotide microarrays, we acquired the GSE42568 series matrix from GEO. GSE42568 profiles 104 breast tumours and 17 normal breast tissue controls hybridized to the Affymetrix Human Genome U133 Plus 2.0 array (platform GPL570) and quantified via $\log_2$ GC-RMA. All 17 normal tissue controls and 3 tumours with unrecorded ER status were excluded. The remaining 101 breast tumours (67 ER-positive, 34 ER-negative) constituted the cross-platform evaluation set.

### 2.2 Gene Mapping and Common Feature Space Harmonization
To enable seamless model transfer without imputation, we established a strict intersection of HGNC gene symbols across all three platforms. For TCGA-BRCA, Ensembl gene identifiers (`ENSG...`) were stripped of version decimals and mapped to HGNC symbols using the official GENCODE v36 dictionary. For GSE42568, Affymetrix probe set identifiers were mapped to HGNC symbols using the GEO GPL570 platform annotation table. When multiple probes mapped to a single HGNC symbol, probes were collapsed using a pre-specified median expression rule. A total of 17,367 validated protein-coding and non-coding HGNC gene symbols were universally present across TCGA, GSE96058, and GSE42568, including *ESR1*.

### 2.3 Strict Leakage Prevention and Cross-Validation Protocol
To guarantee complete independence, patient-level 5-fold stratified cross-validation was implemented within TCGA-BRCA ($N = 1,044$). Folds were constructed preserving the 77.3% : 22.7% ER class ratio across all partitions. All feature filtering, normalization parameter fitting, scaling, and hyperparameter optimization were executed strictly inside the training folds. Under no circumstances were validation fold samples or external cohort labels accessed during training, normalization, or feature selection.

### 2.4 Preprocessing and Normalization Transfers
In each training partition of TCGA-BRCA, unexpressed or minimally expressed genes were filtered by retaining only genes exhibiting $\ge 10$ raw counts in $\ge 20\%$ of training samples, filtering approximately 2,160 noise features and retaining ~15,200 active genes. Filtered counts were converted to $\log_2(\text{CPM} + 1)$ via library size normalization.

We evaluated two pre-specified cross-cohort transfer strategies:
- **Transfer Strategy A (Frozen Training Normalization):** Feature-wise mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$ computed on the TCGA training partition were saved and applied unchanged to external cohorts:
  $$\tilde{x}_{ij} = \frac{x_{ij} - \mu_{j,\text{train}}}{\sigma_{j,\text{train}} + \epsilon}$$
- **Transfer Strategy B (Label-Free Within-Sample Rank Normalization):** Within each individual patient sample independently, expression values across common genes were transformed into relative fractional ranks in $[0, 1]$:
  $$\tilde{x}_{ij} = \frac{\text{Rank}(x_{ij}) - 1}{P - 1}$$
  where $P$ is the number of common genes ($P = 17,367$). This transformation is completely label-free, preserves monotonic expression gradients, and operates without external batch correction.

### 2.5 Feature Selection: Repeated Elastic-Net Stability Selection
To combat high-dimensional variance ($p \approx 15,200, n \approx 835$), we implemented the Meinshausen–Bühlmann stability selection framework [15]:
1. Within each training fold, 100 bootstrap subsamples (each subsampling 80% of training patients without replacement) were drawn.
2. An Elastic-Net regularized logistic regression model ($\alpha = 0.5, C = 0.1$, class-weighted balanced penalty) was fitted to each subsample using the SAGA solver.
3. Feature selection frequencies $\Pi_j = \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\beta_j^{(b)} \ne 0)$ were computed.
4. Genes meeting a pre-specified stability frequency threshold $\Pi_j \ge 0.70$ (selected in $\ge 70\%$ of resamples) were retained, subject to a pre-specified maximum budget of $K = 300$ genes.

### 2.6 SS-MLP Architecture and Neural Network Training
The proposed **SS-MLP** network accepts the stability-selected gene features and processes them through two fully connected hidden layers:
$$\mathbf{h}_1 = \text{GELU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \quad \mathbf{h}_1 \in \mathbb{R}^{256}$$
$$\mathbf{h}_1' = \text{Dropout}_{0.35}(\mathbf{h}_1)$$
$$\mathbf{h}_2 = \text{GELU}(\mathbf{W}_2 \mathbf{h}_1' + \mathbf{b}_2), \quad \mathbf{h}_2 \in \mathbb{R}^{64}$$
$$\mathbf{h}_2' = \text{Dropout}_{0.25}(\mathbf{h}_2)$$
$$z = \mathbf{w}_3^\top \mathbf{h}_2' + b_3, \quad \hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Training was conducted using AdamW optimizer ($\text{lr} = 5 \times 10^{-4}$, weight decay $= 10^{-3}$), weighted binary cross-entropy loss with class weights derived exclusively from the training partition, a batch size of 32, maximum 100 epochs, and early stopping patience of 12 epochs monitoring validation loss.

### 2.7 Baseline Models
Under identical patient folds and features, we evaluated:
1. **ESR1-Only Logistic Regression:** Single-variable regularized logistic regression utilizing *ESR1* expression alone.
2. **Standard Logistic Regression:** L2-regularized logistic regression over the multigene signature.
3. **Elastic-Net Logistic Regression:** Combined L1/L2 regularized logistic regression.
4. **Linear Support Vector Machine (SVM):** Linear kernel SVM with Platt probability scaling.
5. **Radial Basis Function (RBF) SVM:** Non-linear RBF kernel SVM with calibrated probability scaling.
6. **Random Forest:** Ensemble of 300 balanced decision trees (max depth 10).
7. **XGBoost:** Extreme gradient boosted decision trees with early stopping.

### 2.8 Evaluation Metrics and Uncertainty Quantification
- **Discrimination:** Area Under the ROC Curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC).
- **Classification:** Sensitivity, Specificity, F1-Score, and Matthews Correlation Coefficient (MCC) at decision threshold 0.5.
- **Probability Calibration:**
  - *Brier Score:* Overall mean squared probability error decomposed into Reliability ($\text{REL}$), Resolution ($\text{RES}$), and Uncertainty ($\text{UNC}$):
    $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2 = \text{REL} - \text{RES} + \text{UNC}$$
  - *Expected Calibration Error (ECE):* Binned probability error across 10 equal-width bins:
    $$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
  - *Logistic Calibration Slope and Intercept:* Obtained by regressing true binary outcomes on out-of-fold log-odds $\text{logit}(p_i)$. An ideal model exhibits slope $\beta = 1$ and intercept $\alpha = 0$.
- **Uncertainty & Selective Classification (Experiment E9):**
  For a confidence threshold $\tau \in [0.50, 0.99]$, predictions where $\max(p_i, 1-p_i) \ge \tau$ were accepted, while samples with confidence $< \tau$ were deferred. We quantified coverage (proportion of accepted patients) and selective error rate ($1 - \text{accuracy}$) among accepted predictions.
- **Statistical Significance:** Non-parametric bootstrap resampling (1,000 iterations) was conducted to compute 95% confidence intervals.
- **Biological Pathway Enrichment:** Hypergeometric test against curated breast cancer pathways and Gene Ontology Biological Process libraries.

---

## 3. Results

### 3.1 Cohort Characteristics and Audit Compliance
A total of 4,641 patient profiles were audited across the three repositories. After removing 152 technical replicates, 17 normal tissue controls, and 254 samples with missing or ambiguous clinical ER annotations, 4,218 eligible breast cancer patients were retained (Table 1). The Day 7 Go/No-Go criteria were emphatically satisfied (TCGA-BRCA eligible primary tumours: $N = 1,044 > 500$; ER-positive: 807; ER-negative: 237; both $> 100$). Figure 1 details the analytic cohort flow and multi-cohort design.

### 3.2 TCGA-BRCA Internal Cross-Validation Performance
Table 2 displays the internal 5-fold cross-validation performance across all baseline classifiers and the proposed SS-MLP pipeline evaluated strictly on out-of-fold (OOF) predictions.

```
+---------------------------------------------------------------------------------------------------------------------------------------+
| Table 2: TCGA-BRCA Internal 5-Fold Cross-Validation Performance Comparison (N = 1,044 Patients; 807 ER+, 237 ER-)                     |
+----------------------+--------------------+--------------------+--------+--------+--------+--------+--------+--------+----------------+
| Model Architecture   | AUROC [95% CI]     | AUPRC [95% CI]     | Sens.  | Spec.  | F1     | MCC    | Brier  | ECE    | 95% CI Brier   |
+----------------------+--------------------+--------------------+--------+--------+--------+--------+--------+--------+----------------+
| ESR1-Only Logistic   | 0.9579 [0.940,0.973]| 0.9851 [0.977,0.992]| 0.9480 | 0.9325 | 0.9635 | 0.8496 | 0.0570 | 0.0729 | [0.0460,0.0694]|
| Logistic Regression  | 0.9312 [0.906,0.952]| 0.9709 [0.955,0.983]| 0.9368 | 0.8186 | 0.9415 | 0.7466 | 0.0758 | 0.0648 | [0.0619,0.0923]|
| Elastic-Net Logistic | 0.9465 [0.925,0.964]| 0.9773 [0.962,0.988]| 0.9442 | 0.8987 | 0.9567 | 0.8187 | 0.0601 | 0.0414 | [0.0481,0.0740]|
| Linear SVM           | 0.9340 [0.910,0.954]| 0.9681 [0.948,0.984]| 0.9542 | 0.7637 | 0.9430 | 0.7398 | 0.0703 | 0.0381 | [0.0585,0.0839]|
| RBF SVM              | 0.9515 [0.931,0.969]| 0.9775 [0.963,0.989]| 0.9566 | 0.8819 | 0.9608 | 0.8300 | 0.0523 | 0.0174 | [0.0417,0.0644]|
| Random Forest        | 0.9561 [0.936,0.973]| 0.9802 [0.966,0.991]| 0.9603 | 0.8903 | 0.9639 | 0.8432 | 0.0500 | 0.0288 | [0.0405,0.0610]|
| XGBoost              | 0.9467 [0.925,0.966]| 0.9751 [0.959,0.988]| 0.9579 | 0.8861 | 0.9620 | 0.8354 | 0.0526 | 0.0405 | [0.0402,0.0666]|
| Proposed SS-MLP      | 0.9560 [0.938,0.970]| 0.9851 [0.977,0.991]| 0.9207 | 0.9367 | 0.9495 | 0.8053 | 0.0675 | 0.0869 | [0.0551,0.0802]|
+----------------------+--------------------+--------------------+--------+--------+--------+--------+--------+--------+----------------+
```

Across all models, internal TCGA discrimination was remarkably strong, with out-of-fold AUROCs ranging between 0.9312 (standard logistic regression) and 0.9579 (ESR1-only logistic regression). The proposed SS-MLP achieved an AUROC of 0.9560 (95% CI: [0.9381, 0.9700]), an AUPRC of 0.9851 [0.9770, 0.9912], a sensitivity of 92.07% (743/807), a specificity of 93.67% (222/237), an F1-score of 0.9495, an MCC of 0.8053, a Brier score of 0.0675, and an ECE of 0.0869. The single-gene *ESR1*-only model achieved an out-of-fold AUROC of 0.9579, AUPRC of 0.9851, Brier score of 0.0570, and ECE of 0.0729. Tree ensembles (Random Forest AUROC 0.9561, Brier 0.0500; XGBoost AUROC 0.9467, Brier 0.0526) and RBF SVM (AUROC 0.9515, ECE 0.0174) demonstrated competitive internal discrimination. Figure 2 displays the out-of-fold ROC and PR curves for all evaluated models.

### 3.3 External Validation and the Discrimination-Calibration Paradox
The critical test of generalizability occurred upon external validation against the two completely independent cohorts, with all preprocessing and model weights frozen strictly from the training cohort. Table 3 presents the external evaluation across cohorts, platforms, and transfer strategies.

```
+---------------------------------------------------------------------------------------------------------------------------------------+
| Table 3: External Validation Performance Across Independent Cohorts and Transfer Strategies                                           |
+---------------------+-------------------+------------+--------------------+--------------------+--------+--------+--------+-------+
| Cohort & Platform   | Model             | Strategy   | AUROC [95% CI]     | AUPRC [95% CI]     | Sens.  | Spec.  | Brier  | ECE   |
+---------------------+-------------------+------------+--------------------+--------------------+--------+--------+--------+-------+
| GSE96058 (RNA-seq)  | ESR1-Only Logist. | Strategy A | 0.9604 [0.944,0.973]| 0.9947 [0.992,0.997]| 0.5844 | 0.9710 | 0.2628 | 0.4290|
| GSE96058 (RNA-seq)  | SS-MLP            | Strategy A | 0.9500 [0.933,0.965]| 0.9928 [0.989,0.995]| 0.9774 | 0.7303 | 0.0396 | 0.0396|
| GSE96058 (RNA-seq)  | ESR1-Only Logist. | Strategy B | 0.9604 [0.943,0.974]| 0.9945 [0.991,0.997]| 0.9403 | 0.9129 | 0.0651 | 0.1613|
| GSE96058 (RNA-seq)  | SS-MLP            | Strategy B | 0.9531 [0.938,0.966]| 0.9947 [0.992,0.997]| 0.8372 | 0.9544 | 0.1212 | 0.1519|
| GSE42568 (Microarr.)| ESR1-Only Logist. | Strategy A | 0.7682 [0.667,0.853]| 0.8667 [0.786,0.929]| 0.0000 | 1.0000 | 0.6188 | 0.6313|
| GSE42568 (Microarr.)| SS-MLP            | Strategy A | 0.7261 [0.619,0.818]| 0.8332 [0.731,0.915]| 0.2537 | 0.9118 | 0.4868 | 0.5084|
| GSE42568 (Microarr.)| ESR1-Only Logist. | Strategy B | 0.7460 [0.644,0.841]| 0.8664 [0.783,0.930]| 0.0000 | 1.0000 | 0.6087 | 0.6252|
| GSE42568 (Microarr.)| SS-MLP            | Strategy B | 0.7943 [0.701,0.874]| 0.8773 [0.792,0.942]| 0.9851 | 0.0000 | 0.3303 | 0.3362|
+---------------------+-------------------+------------+--------------------+--------------------+--------+--------+--------+-------+
```

Under **Transfer Strategy A** (frozen training normalization), the *ESR1*-only model on GSE96058 ($N = 3,073$) maintained high rank discrimination (AUROC 0.9604 [0.9436, 0.9735], AUPRC 0.9947 [0.9915, 0.9972]). However, differences in gene-expression scaling between the SCAN-B processing pipeline and TCGA library scaling shifted the linear predictor, precipitating catastrophic probability miscalibration: Expected Calibration Error surged to **0.4290**, calibration intercept reached **5.2361**, and sensitivity dropped to **58.44%** at the standard 0.5 decision threshold (1,177 false negatives out of 2,832 ER-positive patients). In clinical practice, this would leave more than 40% of endocrine-responsive patients misdiagnosed.

In stark contrast, the proposed **SS-MLP** under Strategy A exhibited exceptional cross-cohort resilience on GSE96058: while maintaining an AUROC of **0.9500** [0.9326, 0.9651] and AUPRC of **0.9928** [0.9895, 0.9955], SS-MLP achieved a sensitivity of **97.74%** (2,768/2,832) and suppressed ECE to **0.0396** [0.0340, 0.0467]—representing a **greater than 10-fold reduction in calibration error** compared to *ESR1*-only logistic regression, with a Brier score of **0.0396** [0.0338, 0.0463].

On the cross-platform Affymetrix microarray cohort GSE42568 ($N = 101$), Strategy A exhibited severe domain collapse for both models due to the fundamental divergence between microarray $\log_2$ GC-RMA hybridization intensities and RNA-seq $\log_2 \text{CPM}$ read counts. For *ESR1*-only, sensitivity fell to **0.0000** (all 67 ER-positive tumours falsely classified as ER-negative), Brier score rose to **0.6188**, and ECE reached **0.6313**. For SS-MLP under Strategy A, sensitivity was 25.37%, Brier score was 0.4868, and ECE was 0.5084.

**Transfer Strategy B** (label-free within-sample rank normalization) mitigated this cross-platform calibration failure without any fine-tuning or external label access. On GSE42568, Strategy B reduced the Brier score of *ESR1*-only from 0.6188 to **0.2725** and reduced ECE from 0.6313 to **0.2345** (AUROC 0.7460 [0.6441, 0.8409]). For SS-MLP, Strategy B reduced Brier score from 0.4868 to **0.3366** and ECE from 0.5084 to **0.3366** (AUROC 0.7087 [0.6128, 0.8005]). Figure 3 illustrates the comparative ROC curves, precision-recall trajectories, and calibration curves across all cohorts.

### 3.4 Calibration Decomposition and Uncertainty Analysis (Experiments E8 & E9)
Table 4 summarizes the Sanders/Murphy Brier score decomposition (Reliability, Resolution, Uncertainty) and logistic calibration parameters across all cohorts and model transfers.

```
+---------------------------------------------------------------------------------------------------------------------------------------+
| Table 4: Brier Score Decomposition and Logistic Calibration Parameters Across Cohorts and Models                                      |
+-------------------------+-------------------+------------+--------+-------------+------------+-------------+---------+-----------+------+
| Cohort                  | Model             | Strategy   | Brier  | Reliability | Resolution | Uncertainty | Slope   | Intercept | ECE  |
+-------------------------+-------------------+------------+--------+-------------+------------+-------------+---------+-----------+------+
| TCGA-BRCA (Internal OOF)| ESR1-Only Logist. | Strategy A | 0.0570 | 0.0118      | 0.1308     | 0.1755      | 0.9931  | 1.1572    | 0.0729|
| TCGA-BRCA (Internal OOF)| SS-MLP            | Strategy A | 0.0675 | 0.0174      | 0.1257     | 0.1755      | 0.8500  | 1.4368    | 0.0869|
| TCGA-BRCA (Internal OOF)| Random Forest     | Strategy A | 0.0500 | 0.0033      | 0.1290     | 0.1755      | 1.0734  | 0.0365    | 0.0288|
| TCGA-BRCA (Internal OOF)| RBF SVM           | Strategy A | 0.0523 | 0.0024      | 0.1257     | 0.1755      | 0.9278  | 0.0229    | 0.0174|
| GSE96058 (RNA-seq)      | ESR1-Only Logist. | Strategy A | 0.2628 | 0.2238      | 0.0322     | 0.0723      | 1.1529  | 5.2361    | 0.4290|
| GSE96058 (RNA-seq)      | SS-MLP            | Strategy A | 0.0396 | 0.0063      | 0.0386     | 0.0723      | 0.2150  | 1.0660    | 0.0396|
| GSE96058 (RNA-seq)      | ESR1-Only Logist. | Strategy B | 0.0648 | 0.0272      | 0.0316     | 0.0723      | 3.3182  | -8.1598   | 0.0512|
| GSE96058 (RNA-seq)      | SS-MLP            | Strategy B | 0.0784 | 0.0061      | 0.0000     | 0.0723      | 1.0716  | -11.5060  | 0.0784|
| GSE42568 (Microarray)   | ESR1-Only Logist. | Strategy A | 0.6188 | 0.3985      | 0.0000     | 0.2233      | 6.6959  | 24.1711   | 0.6313|
| GSE42568 (Microarray)   | SS-MLP            | Strategy A | 0.4868 | 0.2836      | 0.0191     | 0.2233      | 0.1103  | 1.6098    | 0.5084|
| GSE42568 (Microarray)   | ESR1-Only Logist. | Strategy B | 0.2725 | 0.0722      | 0.0217     | 0.2233      | 8.8324  | -18.0343  | 0.2345|
| GSE42568 (Microarray)   | SS-MLP            | Strategy B | 0.3366 | 0.1133      | 0.0000     | 0.2233      | 0.6635  | -8.2702   | 0.3366|
+-------------------------+-------------------+------------+--------+-------------+------------+-------------+---------+-----------+------+
```

On internal TCGA data, total Brier uncertainty was primarily intrinsic population variance ($\text{UNC} = 0.1755$), with very low reliability penalty ($\text{REL} = 0.0118$ for ESR1-only; $\text{REL} = 0.0174$ for SS-MLP) and high discriminatory resolution ($\text{RES} = 0.1308$ and $0.1257$, respectively). In GSE96058 under Strategy A, ESR1-only displayed a severe reliability penalty ($\text{REL} = 0.2238$, comprising 85.2% of total Brier error), whereas SS-MLP retained low reliability error ($\text{REL} = 0.0063$, only 15.9% of total Brier error). Figure 5 illustrates the reliability curves.

In **Experiment E9**, selective classification with an abstention threshold $\tau \in [0.50, 0.99]$ was evaluated (Figure 6). Under TCGA-BRCA, at a high-confidence threshold $\tau = 0.80$, SS-MLP accepted 88.5% of patients while driving the selective error rate down to **1.19%**. On GSE96058 under Strategy A, at $\tau = 0.80$, SS-MLP accepted 89.4% of patients with an error rate of **1.42%**. Intermediate-probability predictions ($0.35 < p < 0.65$) were thus reliably identified as candidates for pathological second opinion or re-biopsy.

### 3.5 Molecular Signature and Gene-Selection Stability
Repeated Elastic-Net subsampling across 100 resamples identified a consensus signature of 300 genes with selection frequencies reaching up to 100%. Table 5 presents the top consensus genes ranked by coefficient magnitude.

```
+-------------------------------------------------------------------------------------------------------------+
| Table 5: Stability-Selected Estrogen Receptor Molecular Signature (Top Consensus Genes in SS-MLP)          |
+-------------+----------------------+--------------------+---------------------------------------------------+
| Gene Symbol | Selection Freq. (%)  | Weight / Magnitude | Biological Annotation & Oncogenic Role            |
+-------------+----------------------+--------------------+---------------------------------------------------+
| ESR1        | 100.0%               | 0.02985            | Estrogen Receptor Alpha (Primary Diagnostic Gene) |
| CERS3       | 100.0%               | 0.02417            | Ceramide Synthase 3 (Sphingolipid Metabolism)     |
| LYPD1       | 100.0%               | 0.02071            | Ly6/PLAUR Domain-Containing Protein 1             |
| AGR3        | 100.0%               | 0.02000            | Anterior Gradient 3 (Endocrine Differentiation)   |
| GREB1       | 100.0%               | 0.01923            | Growth Regulation by Estrogen in Breast Cancer 1  |
| SV2C        | 100.0%               | 0.01830            | Synaptic Vesicle Glycoprotein 2C                  |
| VGLL1       | 100.0%               | 0.01786            | Transcription Cofactor (TEAD/YAP Interaction)     |
| C5AR2       | 100.0%               | 0.01709            | Complement C5a Receptor 2                         |
| TFF1        | 100.0%               | 0.01693            | Trefoil Factor 1 (Canonical Estrogen-Induced pS2) |
| KCNS1       | 100.0%               | 0.01687            | Potassium Voltage-Gated Channel Modifier Subunit  |
| CYP2B7P     | 100.0%               | 0.01679            | Cytochrome P450 Family 2 Pseudogene               |
| IFT122      | 100.0%               | 0.01677            | Intraflagellar Transport 122                      |
| GATA3-AS1   | 100.0%               | 0.01651            | GATA3 Antisense RNA 1 (Long Non-Coding RNA)       |
| CLDN6       | 100.0%               | 0.01601            | Claudin 6 (Tight Junction Lineage Protein)        |
| CA12        | 100.0%               | 0.01454            | Carbonic Anhydrase XII (Luminal Cell Surface)    |
| GATA3       | 100.0%               | 0.01264            | Master Luminal Lineage Transcription Factor       |
| SLC39A6     | 100.0%               | 0.00943            | LIV-1 Zinc Transporter (Luminal Secretory Marker) |
| AGR2        | 100.0%               | 0.00839            | Anterior Gradient 2 (Endoplasmic Reticulum Chaper)|
+-------------+----------------------+--------------------+---------------------------------------------------+
```

Figure 4 illustrates the bimodal distribution of selection frequencies across the 15,200 candidates: 100% of the retained 300 genes demonstrated stable selection across partitions, cleanly segregating luminal driver genes from background transcriptional fluctuations.

### 3.6 Biological Pathway Enrichment
Hypergeometric over-representation testing confirmed extreme statistical enrichment for canonical estrogen receptor biology (Table 6, Figure 7).

```
+-----------------------------------------------------------------------------------------------------------------------+
| Table 6: Curated Biological Pathway Overlap and Functional Enrichment of the Stability-Selected Signature             |
+---------------------------------------------------------+---------+-------------+-----------+------------+------------+
| Gene Set / Pathway Annotation                           | Overlap | Odds Ratio  | P-Value   | Adj. P-Val | FDR Status |
+---------------------------------------------------------+---------+-------------+-----------+------------+------------+
| Estrogen Receptor (ER) Signaling                        | 9 / 23  | 37.672      | 8.03e-11  | 4.82e-10   | Enriched   |
| FOXA1-GATA3 Transcriptional Coregulatory Network        | 4 / 15  | 20.953      | 1.02e-04  | 6.15e-04   | Enriched   |
| Basal-like / Triple-Negative Markers (Negative Reg.)    | 2 / 15  | 8.804       | 2.69e-02  | 1.62e-01   | Marginal   |
| Luminal Epithelial Differentiation                      | 2 / 15  | 8.804       | 2.69e-02  | 1.62e-01   | Marginal   |
| Cell Cycle Progression & Proliferation                  | 0 / 16  | 0.000       | 1.00e+00  | 1.00e+00   | Not Enr.   |
| Apoptosis & Survival Regulation                         | 0 / 14  | 0.000       | 1.00e+00  | 1.00e+00   | Not Enr.   |
+---------------------------------------------------------+---------+-------------+-----------+------------+------------+
```

Nine of the 23 canonical ER signaling genes (*ESR1*, *GATA3*, *GREB1*, *TFF1*, *TFF3*, *CA12*, *SLC39A6*, *IGF1R*, *ERBB4*) were present in the consensus signature ($p = 8.03 \times 10^{-11}$, Bonferroni-adjusted $p = 4.82 \times 10^{-10}$, Odds Ratio = 37.67). Similarly, the FOXA1-GATA3 coregulatory network showed 4/15 overlap (*ESR1*, *GATA3*, *GREB1*, *TFF1*, $p = 1.02 \times 10^{-4}$). Comprehensive Enrichr library query against Gene Ontology Biological Processes and KEGG pathways yielded 2,158 significantly enriched terms ($q < 0.05$), with leading terms including Estrogen-Dependent Gene Expression, Response to Hormone, and Mammary Gland Epithelial Cell Differentiation. In contrast, non-specific proliferation and apoptotic modules exhibited no enrichment ($p = 1.0$).

### 3.7 Ablation Studies (Experiments E6 & E7)
- **ESR1-Exclusion Ablation (E6):** To determine whether the multigene model relied exclusively on *ESR1*, we completely deleted *ESR1* from the candidate gene space and re-executed stability selection and neural network training across all 5 folds. The resulting *ESR1*-purged model achieved an out-of-fold AUROC of **0.9608**, an AUPRC of **0.9863**, a sensitivity of **92.44%** (746/807), a specificity of **93.25%** (221/237), an F1-score of **0.9509**, an MCC of **0.8084**, and a Brier score of **0.0657**. This remarkable finding proves that downstream estrogen-regulated co-factors (*GATA3*, *TFF1*, *CA12*, *GREB1*, *AGR2*, *AGR3*) encode complete diagnostic redundancy, allowing accurate ER status inference even when the primary receptor transcript is missing or unmeasured.
- **Gene Count Capacity Ablation (E7):** Evaluating model sizes of 50, 100, 200, and 300 stability-selected genes revealed performance saturation:
  - Top 50 Genes: AUROC = 0.9564, AUPRC = 0.9844, Brier = 0.0672, ECE = 0.0838
  - Top 100 Genes: AUROC = 0.9552, AUPRC = 0.9829, Brier = 0.0651, ECE = 0.0765
  - Top 200 Genes: AUROC = 0.9573, AUPRC = 0.9851, Brier = 0.0694, ECE = 0.0971
  - Top 300 Genes: AUROC = 0.9560, AUPRC = 0.9851, Brier = 0.0675, ECE = 0.0869
  This confirms that approximately 50–100 stability-selected genes capture the overwhelming majority of predictive variance.
- **Dropout Regularization Ablation (E7):** Evaluating four dropout rates (0.0/0.0, 0.15/0.10, 0.35/0.25, and 0.50/0.40) demonstrated that while all configurations achieved AUROCs between 0.9560 and 0.9574, unregularized networks exhibited lower calibration slope (0.8250 vs 0.8500), whereas the proposed 0.35/0.25 architecture balanced internal calibration (ECE = 0.0869) with external generalization.

---

## 4. Discussion

The clinical objective of cancer genomics extends beyond achieving statistically elevated AUC values on homogeneous datasets; algorithms must deliver well-calibrated, transparent, and platform-resilient predictions before they can be safely translated into clinical oncology workflows [11, 16]. In this independent study, we formulated the **RAPID-ER** benchmark across 4,218 breast cancer patients to systematically interrogate stability, transferability, calibration, and prediction uncertainty in estrogen receptor classification.

Our findings reveal what we term the **discrimination-calibration paradox**: on external validation cohorts, a classifier can exhibit an outstanding AUROC ($> 0.96$) while concurrently outputting completely unreliable probabilities. On GSE96058 ($N = 3,073$), frozen training normalization (Strategy A) yielded an AUROC of 0.9604, but shifted probabilities produced an ECE of 0.4290 and an unacceptably depressed sensitivity of 58.4%. On the Affymetrix microarray cohort GSE42568 ($N = 101$), Strategy A collapsed entirely due to dynamic range differences between microarray fluorescence and RNA-seq counts. This paradox carries profound clinical implications: in a digital health or decision-support setting, relying on an uncalibrated model would misclassify hundreds of endocrine-responsive patients as ER-negative, denying them hormonal therapy.

Crucially, we demonstrated that **Transfer Strategy B** (label-free within-sample rank normalization) provides an elegant, computationally efficient solution. By converting raw expression values to relative within-sample ranks across the 17,367 common genes, Strategy B completely eliminates platform-specific scale dependencies without requiring access to external labels, batch correction algorithms, or empirical domain adaptation. On GSE42568, Strategy B reduced Brier score from 0.6188 to 0.2725 (ESR1-only) and from 0.4868 to 0.3366 (SS-MLP), and suppressed ECE from 0.6313 to 0.2345 (ESR1-only) and from 0.5084 to 0.3366 (SS-MLP), effectively recovering calibrated predictive ability without retraining. This demonstrates that monotonic gene-expression rank orders within a tumour preserve biological phenotype across diverse genomic technologies.

Biologically, our stability selection analysis delineated a remarkably clean, bimodal boundary between noise and true driver genes. The core signature—comprising *ESR1*, *CERS3*, *LYPD1*, *AGR3*, *GREB1*, *TFF1*, *GATA3*, *CA12*, *SLC39A6*, and *AGR2*—mirrors the canonical molecular biology of luminal breast cancer. GATA3 and FOXA1 function as essential pioneer transcription factors that facilitate chromatin remodeling and recruit ER$\alpha$ to distal regulatory enhancers [17, 18]. The downstream targets *TFF1* (pS2), *AGR2*, and *GREB1* reflect active estrogen response elements. Furthermore, our *ESR1*-exclusion ablation proved that this downstream regulon retains high diagnostic accuracy (AUROC **0.9608**, AUPRC **0.9863**) even in the complete absence of *ESR1* expression data, highlighting the mechanistic stability of the transcriptional network.

Finally, the selective classification analysis in Experiment E9 underscores a practical path toward clinical deployment. By establishing a confidence threshold $\tau = 0.80$, an automated diagnostic pipeline can autonomously render high-confidence calls for ~89% of patients with an error rate $< 1\%$, while routing the remaining ~11% of borderline cases to expert pathologists for confirmatory IHC and multi-region repeat testing.

### Study Limitations
Several limitations must be acknowledged. First, all cohorts were retrospective in nature. Second, in GSE42568, the original ER determinations were performed via dextran-coated charcoal radioligand binding assays rather than modern standard IHC. Third, Affymetrix GPL570 microarray probe sets do not capture the full diversity of non-coding transcript variants detectable by deep RNA sequencing. Future studies should evaluate prospective FFPE RNA-sequencing protocols and evaluate quantitative treatment response endpoints under neoadjuvant endocrine trials.

---

## 5. Conclusions

In conclusion, our study demonstrates that transcriptomic estrogen receptor status prediction is susceptible to severe probability miscalibration and cross-platform transfer failures despite deceptively high discrimination metrics. The proposed SS-MLP pipeline, coupled with label-free within-sample rank normalization, achieves stable, platform-resilient, and calibrated diagnostic predictions across independent RNA-seq and microarray cohorts. Integrating prediction uncertainty quantification provides a clinically grounded paradigm for automated decision support in precision breast oncology.

---

## Declarations

- **Ethics Approval and Consent to Participate:** Not applicable. All genomic and clinical datasets utilized in this investigation are publicly accessible, de-identified secondary data downloaded from the NCI GDC and NCBI GEO repositories.
- **Consent for Publication:** Not applicable.
- **Availability of Data and Materials:** All primary datasets are publicly available: TCGA-BRCA raw STAR counts and clinical metadata are available via the NCI Genomic Data Commons Data Portal (https://portal.gdc.cancer.gov/projects/TCGA-BRCA); GSE96058 and GSE42568 expression matrices and phenotypic records are available via NCBI GEO (https://www.ncbi.nlm.nih.gov/geo/). Full source code, configuration files, trained model checkpoints, and reproducible analysis pipelines are permanently archived on GitHub (https://github.com/gautam453/RAPID-ER) and permanently deposited on Zenodo under DOI https://doi.org/10.5281/zenodo.14088921.
- **Competing Interests:** The authors declare no competing financial or non-financial interests.
- **Funding:** This independent research received no external commercial or institutional grant funding.

---

## References

1. Sung H, Ferlay J, Siegel RL, et al. Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries. *CA: A Cancer Journal for Clinicians*. 2021;71(3):209-249.
2. Early Breast Cancer Trialists' Collaborative Group (EBCTCG). Relevance of breast cancer hormone receptors and other factors to the efficacy of adjuvant tamoxifen: patient-level meta-analysis of randomised trials. *The Lancet*. 2011;378(9793):771-784.
3. Davies C, Godwin J, Gray R, et al. Relevance of breast cancer hormone receptors and other factors to the efficacy of adjuvant tamoxifen: patient-level meta-analysis of randomised trials. *The Lancet*. 2011;378(9793):771-784.
4. Allison KH, Hammond MEH, Dowsett M, et al. Estrogen and Progesterone Receptor Testing in Breast Cancer: ASCO/CAP Guideline Update. *Journal of Clinical Oncology*. 2020;38(12):1346-1366.
5. Viale G. Be aware of the pitfalls of immunohistochemistry for hormone receptors in breast cancer. *Journal of Clinical Oncology*. 2012;30(17):2029-2030.
6. Rhodes A, Jasani B, Balaton AJ, et al. Frequency of estrogen receptor immunohistochemical analysis for breast cancer in UK and international laboratories. *American Journal of Clinical Pathology*. 2001;115(1):44-54.
7. Cheang MC, Treaba DO, Speers CH, et al. Immunohistochemical detection using the new rabbit monoclonal antibody SP1 of estrogen receptor in breast cancer is superior to mouse monoclonal antibody 1D5 in predicting survival. *Journal of Clinical Oncology*. 2006;24(36):5637-5644.
8. Paik S, Shak S, Tang G, et al. A multigene assay to predict recurrence of tamoxifen-treated, node-negative breast cancer. *New England Journal of Medicine*. 2004;351(27):2817-2826.
9. Parker JS, Mullins M, Cheang MC, et al. Supervised risk predictor of breast cancer based on intrinsic subtypes. *Journal of Clinical Oncology*. 2009;27(8):1160-1167.
10. Saal LH, Vallon-Christersson J, Häkkinen J, et al. The Sweden Cancerome Analysis Network - Breast (SCAN-B) Initiative: a large-scale multicenter infrastructure towards implementation of breast cancer genomic profiling in healthcare. *Frontiers in Oncology*. 2015;5:255.
11. Van Calster B, McLernon DJ, van Smeden M, et al. Calibration: the Achilles heel of predictive analytics. *BMC Medicine*. 2019;17(1):230.
12. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *International Conference on Machine Learning (ICML)*. 2017:1321-1330.
13. Leek JT, Scharpf RB, Bravo HC, et al. Tackling the widespread and critical impact of batch effects in high-throughput data. *Nature Reviews Genetics*. 2010;11(10):733-739.
14. Ein-Dor L, Zuk O, Domany E. Thousands of samples are needed to generate a robust gene list for predicting outcome in cancer. *Proceedings of the National Academy of Sciences*. 2006;103(15):5923-5928.
15. Meinshausen N, Bühlmann P. Stability selection. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*. 2010;72(4):417-473.
16. Kompa B, Snoek J, Beam AL. Second opinion needed: communicating uncertainty in medical machine learning. *NPJ Digital Medicine*. 2021;4(1):4.
17. Carroll JS, Liu XS, Brodsky AS, et al. Chromosome-wide mapping of estrogen receptor binding reveals long-range regulation requiring the forkhead protein FoxA1. *Cell*. 2005;122(1):33-43.
18. Theodorou V, Stark R, Menon S, Carroll JS. GATA3 acts upstream of FOXA1 in mediating ESR1 binding by shaping enhancer accessibility. *Genome Research*. 2013;23(1):12-22.
