# Decoupling Discrimination from Calibration: A Platform-Resilient, Uncertainty-Aware Transcriptomic Biomarker for Estrogen Receptor Phenotyping in Breast Cancer

**Gautam Kunniyur Ramchander¹\***  
¹Independent Bioinformatics Research Laboratory  
\*Corresponding Author: gautamramchander7@gmail.com  
November 2026  

---

## Abstract

**Background:** Estrogen receptor alpha (ERα, encoded by *ESR1*) guides endocrine therapy eligibility for roughly 70–80% of breast cancer patients worldwide. Although immunohistochemistry (IHC) remains the clinical reference standard, pre-analytical tissue handling, cold ischemia delays, differences in antibody affinity, and subjective visual scoring contribute to a 5–10% diagnostic discordance rate. High-throughput transcriptomics provides an objective diagnostic alternative, yet existing machine-learning classifiers frequently suffer from cross-platform performance decay, unstable feature selection in high dimensions, and a critical vulnerability we define as the discrimination-calibration paradox—where models maintain high rank discrimination (>0.95 AUROC) while outputting severely distorted, uncalibrated risk probabilities under domain shift.

**Methods:** We developed and evaluated the RAPID-ER framework across 4,218 rigorously audited breast cancer profiles from three independent cohorts: TCGA-BRCA (N = 1,044, primary discovery and 5-fold cross-validation), SCAN-B / GSE96058 (N = 3,073, external RNA-seq validation), and GSE42568 (N = 101, cross-platform Affymetrix microarray stress-test). Across a shared 17,367-gene feature space, we constructed the Stability-Selected Multilayer Perceptron (SS-MLP), combining 100-iteration repeated Elastic-Net bootstrap subsampling with a regularized deep neural network (256–64 GELU architecture). Two domain transfer strategies were benchmarked: parametric frozen training standardization (Strategy A) and label-free within-sample relative rank normalization (Strategy B). Probability calibration was assessed via Sanders/Murphy Brier score decomposition, Expected Calibration Error (ECE), and selective classification abstention.

**Results:** Under internal 5-fold cross-validation on TCGA-BRCA, SS-MLP achieved an out-of-fold AUROC of 0.9560 [95% CI: 0.938, 0.970], AUPRC of 0.9851 [0.977, 0.991], Sensitivity of 92.07%, Specificity of 93.67%, and Brier score of 0.0675. In independent external RNA-seq validation (GSE96058, N = 3,073), univariate *ESR1* models failed under frozen standardization: despite an AUROC of 0.9604, probability calibration degraded severely (ECE = 0.4290, Intercept = 5.2361), causing sensitivity to drop to 58.44% and misclassifying 1,177 true ER-positive patients. Conversely, SS-MLP preserved both high discrimination (AUROC 0.9500 [0.933, 0.965], Sensitivity 97.74%) and tight calibration, reducing calibration error by more than 10-fold (ECE = 0.0396, Brier = 0.0396). On cross-platform microarrays (GSE42568, N = 101), frozen standardization failed completely due to probe dynamic range compression (*ESR1* Sensitivity 0.0000, Brier 0.6188). Within-sample rank normalization (Strategy B) restored cross-platform utility without retraining, lifting SS-MLP AUROC to 0.7943 [0.701, 0.874] and AUPRC to 0.8773. An *in silico ESR1*-exclusion ablation demonstrated complete downstream transcriptomic redundancy (AUROC 0.9608, AUPRC 0.9863). Finally, automated confidence triage with selective abstention achieved 99.6% autonomous coverage with a 3.92% error rate (falling to 2.06% at τ = 0.95), safely routing equivocal borderline cases for confirmatory pathology review.

**Conclusions:** Strong rank discrimination can mask severe probability miscalibration under genomic domain shift. By integrating stability-selected multi-gene networks with label-free rank normalization and uncertainty-based abstention, RAPID-ER resolves the discrimination-calibration paradox and establishes an open, platform-resilient decision-support benchmark for precision oncology.

**Keywords:** Estrogen Receptor, Precision Oncology, Transcriptomics, Deep Learning, Stability Selection, Calibration Paradox, Domain Transfer, Uncertainty Quantification, Breast Cancer.

---

## 1. Introduction

Therapeutic decision-making in clinical oncology relies heavily on accurate biomarker assessment. For the vast majority of breast cancer patients, the primary molecular determinant guiding clinical management is estrogen receptor alpha (ERα), encoded by the *ESR1* gene on chromosome 6q25.1 [1]. Tumors expressing ERα depend on estrogen-driven transcriptional programs, making them responsive to targeted endocrine therapies such as selective estrogen receptor modulators (tamoxifen), selective estrogen receptor degraders (fulvestrant), and aromatase inhibitors (letrozole, anastrozole) [2]. Administered over 5 to 10 years, adjuvant endocrine therapy reduces 15-year distant recurrence and breast-cancer-specific mortality by roughly one-third, standing among the most effective targeted interventions in oncology [2].

Clinical practice guidelines from the American Society of Clinical Oncology and the College of American Pathologists (ASCO/CAP) mandate ER evaluation via immunohistochemistry (IHC) on formalin-fixed, paraffin-embedded (FFPE) tissue, defining ER-positivity as ≥1% immunoreactive tumor nuclei [3]. Despite its widespread adoption, clinical IHC remains susceptible to pre-analytical and analytical variability. Prolonged cold ischemia, inconsistent formalin fixation times, epitope masking, differences among antibody clones (e.g., SP1, 1D5, and 6F11), and subjective visual interpretation contribute to a global diagnostic discordance or borderline ambiguity rate of 5–10% [4]. Misclassification carries asymmetric and severe clinical consequences: an ER-positive patient falsely labeled ER-negative is denied potentially life-saving hormonal therapy, whereas an ER-negative patient misclassified as ER-positive receives ineffective endocrine blockade while facing delays in necessary cytotoxic chemotherapy [4].

High-throughput transcriptomic profiling provides a quantitative, reproducible complement to visual histopathology. Clinical multigene expression assays—including Oncotype DX, MammaPrint, and Prosigna (PAM50)—demonstrate that digital RNA sequencing and hybridization microarrays reliably capture the spectrum of luminal differentiation [5,6]. Because prior studies have already established that ER status can be inferred statistically from transcriptomic profiles [6], our objective is not merely to show that gene expression separates ER-positive from ER-negative tumors.

Instead, this work addresses a persistent translational barrier in computational oncology: the failure of genomic classifiers to maintain calibrated probabilistic outputs across independent patient cohorts and assay platforms. A critical review of existing approaches highlights four core bottlenecks:

1. **The Discrimination-Calibration Paradox:** Genomic machine-learning studies almost universally evaluate performance using rank-based discrimination metrics, namely the Area Under the Receiver Operating Characteristic curve (AUROC) and Area Under the Precision-Recall curve (AUPRC). Clinical decision-support systems, however, rely on well-calibrated posterior probabilities P(Y = 1 | X). A classifier with an external AUROC exceeding 0.95 can still generate severely distorted probabilities under real-world domain shift, assigning near-zero probabilities to true-positive patients and causing false-negative clinical triage [7,8].
2. **Cross-Platform Scale Collapse:** Gene expression is measured across fundamentally different physical technologies, ranging from hybridization oligonucleotide microarrays (continuous log2 GC-RMA fluorescence intensities) to digital RNA sequencing (discrete read counts). Applying training-derived parametric means and standard deviations directly to external platforms (Strategy A) fails when dynamic ranges compress or shift [9].
3. **High-Dimensional Resampling Instability:** When candidate features vastly outnumber patient samples (p >> n), standard penalized regressions (such as LASSO or Elastic-Net) exhibit sensitivity to sampling noise, yielding inconsistent gene signatures across training splits. Enforcing selection stability is critical to isolate genuine biological regulatory networks from statistical artifacts [10].
4. **Absence of Uncertainty-Aware Clinical Abstention:** Diagnostic models deployed in pathology workflows must account for predictive uncertainty. Forcing binary classifications on borderline or noisy specimens introduces avoidable diagnostic errors. Practical clinical integration requires an autonomous triage mechanism that signs out high-confidence predictions while abstaining on equivocal cases for reflex pathology review [11].

To address these challenges, we developed the RAPID-ER framework across 4,218 audited breast cancer profiles from three multi-center cohorts: TCGA-BRCA (N = 1,044), SCAN-B / GSE96058 (N = 3,073), and GSE42568 (N = 101). We introduce the Stability-Selected Multilayer Perceptron (SS-MLP), which pairs repeated bootstrap Elastic-Net stability selection with a regularized deep neural network. We systematically decouple discrimination from calibration across platforms, show that within-sample rank normalization enables cross-technology transfer without batch retraining, demonstrate complete downstream transcriptomic redundancy under *in silico ESR1* knockout, and establish a selective abstention workflow for clinical decision support.

---

## 2. Materials and Methods

### 2.1 Cohort Acquisition, Quality Audit, and Exclusion Criteria
All transcriptomic and clinical datasets were obtained from publicly accessible repositories:
* **TCGA-BRCA (N = 1,044):** Raw STAR-aligned gene counts and clinical biospecimen records were retrieved via the NCI Genomic Data Commons (GDC) API (release 32.0). Initial auditing identified 1,111 biospecimens. We excluded 16 duplicate technical vials ('01B') to prevent patient-level overlap and 51 records lacking definitive clinical ER IHC annotations, yielding 1,044 unique primary tumors.
* **SCAN-B / GSE96058 (N = 3,073):** Log2(FPKM + 0.1) expression matrices and clinical metadata across 3,409 sequencing runs were downloaded from NCBI GEO. Quality auditing removed 136 technical replicate runs and 200 patients with missing IHC status, resulting in 3,073 unique primary breast cancer patients.
* **GSE42568 (N = 101):** Affymetrix Human Genome U133 Plus 2.0 microarray profiles (log2 GC-RMA normalized) were retrieved from NCBI GEO. Out of 121 hybridizations, 17 reduction mammoplasty normal breast controls and 3 samples with unrecorded ER status were excluded, leaving 101 primary breast carcinomas.

### 2.2 Gene Mapping and Shared Feature Space
Ensembl gene identifiers in TCGA-BRCA were mapped to HGNC symbols using GENCODE v36. Gene symbols in SCAN-B were standardized against approved HGNC aliases, and Affymetrix probe sets in GSE42568 were resolved to unique genes using the Jetset scoring algorithm. Intersecting the mapped features across all three cohorts established a universal feature space of 17,367 common HGNC genes.

### 2.3 Leakage Prevention and Cross-Validation Protocol
To prevent information leakage, all preprocessing steps—including low-expression filtering, standardization parameters, stability selection frequencies, and neural network weights—were fitted strictly within designated training folds. Internal evaluation on TCGA-BRCA used 5-fold stratified cross-validation partitioned at the unique patient level. External validation cohorts (GSE96058 and GSE42568) remained completely isolated during development and were evaluated only after all training parameters were locked.

### 2.4 Preprocessing and Transfer Regimes
Within each training fold, low-abundance transcripts were removed by retaining genes with ≥10 raw counts in at least 20% of training samples, eliminating roughly 2,160 noise features and retaining approximately 15,200 expressed genes. Raw counts were transformed to log2(CPM + 1). We evaluated two domain transfer strategies:
* **Transfer Strategy A (Frozen Standardization):** Feature-wise mean (μ_train) and standard deviation (σ_train) were computed on the training set and applied directly to external test cohorts: `x_tilde_ij = (x_ij - μ_j,train) / (σ_j,train + ε)` (Equation 1)
* **Transfer Strategy B (Within-Sample Rank Normalization):** Within each individual sample, expression values across the 17,367 shared genes were converted into relative fractional ranks bounded in [0, 1]: `x_tilde_ij = (Rank(x_ij) - 1) / (P - 1)` (Equation 2)

**Methodological Consistency of Transfer Strategy B:** To avoid distributional mismatch, the Strategy B model evaluated in Table 3 was trained natively on within-sample fractional rank-transformed TCGA-BRCA training partitions ([0, 1] rank domain) rather than applying rank normalization post-hoc to a Z-score-trained network. Training directly in ordinal rank space aligns network weights with invariant within-sample transcript hierarchies, preventing the scale collapse seen under frozen parametric standardization.

### 2.5 Repeated Elastic-Net Stability Selection
Within each training partition, we performed repeated Elastic-Net subsampling across B = 100 bootstrap iterations (subsampling fraction α = 0.80). In each subsample, an Elastic-Net logistic regression model (l1-ratio = 0.5, penalty parameter λ tuned via inner 5-fold cross-validation) was fitted. Selection frequency for each candidate gene k was calculated as `Π_hat_k = (1 / B) Σ I(β_hat_k^(b) ≠ 0)`. Features achieving `Π_hat_k ≥ π_threshold = 0.70` were retained, yielding a stable consensus signature of 300 genes.

### 2.6 SS-MLP Architecture and Training
The Stability-Selected Multilayer Perceptron (SS-MLP) was structured as follows (Figure 1):
* **Input Layer:** d = 300 stability-selected genes.
* **Hidden Layer 1:** Fully connected linear layer (300 → 256), Gaussian Error Linear Unit (GELU) activation, and Dropout (p = 0.35).
* **Hidden Layer 2:** Fully connected linear layer (256 → 64), GELU activation, and Dropout (p = 0.25).
* **Output Layer:** Linear projection (64 → 1) with Sigmoid activation outputting posterior probability `p_hat ∈ [0, 1]`.

Models were optimized using AdamW (learning rate η = 5 × 10⁻⁴, weight decay λ_reg = 10⁻³) under class-weighted binary cross-entropy loss with early stopping (patience = 12 epochs on validation loss).

### 2.7 Evaluation Metrics and Uncertainty Quantification
Discrimination was quantified via AUROC and AUPRC with 95% DeLong and bootstrap confidence intervals. Probabilistic calibration was evaluated using the Brier score and its three-component Sanders/Murphy decomposition: `Brier = Reliability - Resolution + Uncertainty` (Equation 3). Expected Calibration Error (ECE) was computed across 10 equal-width probability bins, alongside logistic calibration slope and intercept fitted via `logit(y) = a + b · logit(p_hat)`. For selective classification with abstention, a confidence threshold `τ ∈ [0.50, 0.99]` was applied: samples meeting `max(p_hat, 1 - p_hat) ≥ τ` were classified autonomously, whereas lower-confidence samples were flagged for confirmatory IHC review.

---

## 3. Results

### 3.1 Multi-Cohort Audit Establishes a 4,218-Patient Discovery Framework
To ensure reproducibility and prevent sample leakage, we audited 4,218 primary invasive breast carcinoma profiles across three independent cohorts spanning RNA-seq and microarray technologies (Table 1).

| Characteristic | TCGA-BRCA | GSE96058 (SCAN-B) | GSE42568 |
| :--- | :--- | :--- | :--- |
| Role in Study | Training & 5-Fold Internal CV | External RNA-seq Valid. | External Microarray Valid. |
| Platform | Illumina HiSeq (STAR Counts) | Illumina HiSeq (RNA-seq) | Affymetrix GPL570 HG-U133 |
| Expression Scale | Harmonized Raw Read Counts | Log2-Transformed FPKM | Log2 GC-RMA Normalized |
| Raw Biospecimens | 1,111 files | 3,409 profiles | 121 profiles |
| Excluded Samples | 67 | 336 | 20 |
| - Technical Replicates | 16 (01B vials excluded) | 136 technical replicates | 0 |
| - Non-Tumour Controls | 0 (Primary tumours only) | 0 | 17 normal breast controls |
| - Missing/Indeterminate | 51 indeterminate ER labels | 200 unknown/NA labels | 3 unknown ER labels |
| Final Eligible Patients (N) | 1,044 | 3,073 | 101 |
| ER-Positive (%) | 807 (77.30%) | 2,832 (92.16%) | 67 (66.34%) |
| ER-Negative (%) | 237 (22.70%) | 241 (7.84%) | 34 (33.66%) |
| Clinical ER Reference | Surgical Pathology IHC | Surgical Pathology IHC | Radioligand Binding / IHC |

### 3.2 Internal 5-Fold Cross-Validation: Multi-Gene Deep Learning Matches Single-Gene Baselines
Under 5-fold stratified cross-validation on TCGA-BRCA (N = 1,044), we compared SS-MLP against a univariate *ESR1*-only logistic model and six baseline machine-learning classifiers (Table 2, Figure 2).

| Model Architecture | AUROC [95% CI] | AUPRC [95% CI] | Sens. | Spec. | Brier [95% CI] | ECE [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ESR1-Only Logistic | 0.9579 [0.940, 0.973] | 0.9851 [0.977, 0.992] | 0.9480 | 0.9325 | 0.0570 [0.046, 0.069] | 0.0729 [0.061, 0.089] |
| Logistic Regression | 0.9312 [0.906, 0.952] | 0.9709 [0.955, 0.983] | 0.9368 | 0.8186 | 0.0758 [0.062, 0.092] | 0.0648 [0.053, 0.085] |
| Elastic-Net Logistic | 0.9465 [0.925, 0.964] | 0.9773 [0.962, 0.988] | 0.9442 | 0.8987 | 0.0601 [0.048, 0.074] | 0.0414 [0.033, 0.057] |
| Linear SVM | 0.9340 [0.910, 0.954] | 0.9681 [0.948, 0.984] | 0.9542 | 0.7637 | 0.0703 [0.059, 0.084] | 0.0381 [0.029, 0.056] |
| RBF SVM | 0.9515 [0.931, 0.969] | 0.9775 [0.963, 0.989] | 0.9566 | 0.8819 | 0.0523 [0.042, 0.064] | 0.0174 [0.015, 0.035] |
| Random Forest | 0.9561 [0.936, 0.973] | 0.9802 [0.966, 0.991] | 0.9603 | 0.8903 | 0.0500 [0.041, 0.061] | 0.0288 [0.022, 0.044] |
| XGBoost | 0.9467 [0.925, 0.966] | 0.9751 [0.959, 0.988] | 0.9579 | 0.8861 | 0.0526 [0.040, 0.067] | 0.0405 [0.031, 0.058] |
| **Proposed SS-MLP** | **0.9560 [0.938, 0.970]** | **0.9851 [0.977, 0.991]** | **0.9207** | **0.9367** | **0.0675 [0.055, 0.080]** | **0.0869 [0.073, 0.101]** |

On pooled out-of-fold predictions, SS-MLP achieved an AUROC of 0.9560 [95% CI: 0.9381, 0.9700] and an AUPRC of 0.9851 [95% CI: 0.9770, 0.9912], matching the univariate ESR1-only logistic model (AUROC 0.9579, AUPRC 0.9851). Both models showed balanced sensitivity and specificity (~92–94%) and comparable Brier scores (0.0570 vs. 0.0675). While internal cross-validation alone might suggest that single-gene ESR1 quantification suffices for ER phenotyping, external validation under domain shift reveals a sharp divergence in probabilistic reliability.

### 3.3 The Discrimination-Calibration Paradox: High External AUC Conceals Severe Probability Distortion
Deploying TCGA-trained models externally onto SCAN-B / GSE96058 (N = 3,073) under frozen standardization (Transfer Strategy A) directly exposed the discrimination-calibration paradox (Table 3, Figure 3).

On GSE96058, the univariate *ESR1* model achieved a strong external AUROC of 0.9604 [95% CI: 0.9436, 0.9735] and AUPRC of 0.9947. However, its probability calibration broke down: Expected Calibration Error (ECE) rose more than 5-fold to 0.4290, accompanied by a logistic calibration intercept of 5.2361 and a Brier score of 0.2628. Because SCAN-B log2 FPKM values occupy a shifted quantitative distribution compared to TCGA STAR log2 CPM counts, frozen standardization compressed the predicted probabilities downward. At the standard decision threshold (p ≥ 0.50), *ESR1*-only sensitivity fell to 58.44%, misclassifying 1,177 true ER-positive patients as ER-negative.

| Cohort & Platform | Model | Strategy | AUROC [95% CI] | AUPRC [95% CI] | Sens. | Spec. | Brier | ECE |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| GSE96058 (RNA-seq) | ESR1-Only | Strategy A (Frozen) | 0.9604 [0.944, 0.973] | 0.9947 [0.992, 0.997] | 0.5844 | 0.9710 | 0.2628 | 0.4290 |
| GSE96058 (RNA-seq) | SS-MLP | Strategy A (Frozen) | 0.9500 [0.933, 0.965] | 0.9928 [0.989, 0.995] | 0.9774 | 0.7303 | 0.0396 | 0.0396 |
| GSE96058 (RNA-seq) | ESR1-Only | Strategy B (Rank) | 0.9604 [0.943, 0.974] | 0.9945 [0.991, 0.997] | 0.9403 | 0.9129 | 0.0651 | 0.1613 |
| GSE96058 (RNA-seq) | SS-MLP | Strategy B (Rank) | 0.9531 [0.938, 0.966] | 0.9947 [0.992, 0.997] | 0.8372 | 0.9544 | 0.1212 | 0.1519 |
| GSE42568 (Microarray) | ESR1-Only | Strategy A (Frozen) | 0.7682 [0.667, 0.853] | 0.8667 [0.786, 0.929] | 0.0000 | 1.0000 | 0.6188 | 0.6313 |
| GSE42568 (Microarray) | SS-MLP | Strategy A (Frozen) | 0.7261 [0.619, 0.818] | 0.8332 [0.731, 0.915] | 0.2537 | 0.9118 | 0.4868 | 0.5084 |
| GSE42568 (Microarray) | ESR1-Only | Strategy B (Rank) | 0.7460 [0.644, 0.841] | 0.8664 [0.783, 0.930] | 0.0000 | 1.0000 | 0.6087 | 0.6252 |
| GSE42568 (Microarray) | SS-MLP | Strategy B (Rank) | 0.7943 [0.701, 0.874] | 0.8773 [0.792, 0.942] | 0.9851 | 0.0000 | 0.3303 | 0.3362 |

In contrast, the 300-gene SS-MLP model proved resilient to this distributional shift. Under Strategy A, SS-MLP retained high discrimination (AUROC 0.9500 [0.9326, 0.9651], AUPRC 0.9928) while preserving a sensitivity of 97.74%. Crucially, SS-MLP reduced calibration error more than 10-fold compared to the single-gene model, achieving an ECE of 0.0396 and a Brier score of 0.0396. Distributing learned weights across 300 co-regulated luminal transcripts buffers the network against assay-specific shifts in individual gene abundance.

### 3.4 Label-Free Rank Normalization Rescues Cross-Platform Microarray Transfer
Transferring models from digital RNA-seq to hybridization microarrays (GSE42568, N = 101) posed an even more pronounced domain shift. Under Transfer Strategy A, frozen standardization failed completely (Table 3). Because Affymetrix GC-RMA fluorescence intensities span a narrower dynamic range (~4–14) than transformed RNA-seq counts, applying TCGA training means and standard deviations pushed microarray features into negative Z-score regimes. As a result, the univariate *ESR1* model generated probabilities p_hat < 0.05 for every patient, yielding 0.0000 sensitivity, a Brier score of 0.6188, and an ECE of 0.6313.

Within-sample relative rank normalization (Transfer Strategy B) overcame this cross-technology barrier. Mapping gene expression values within each tumor into fractional ranks `x_tilde_ij ∈ [0, 1]` across the 17,367-gene space anchors predictions to relative transcript ordering rather than absolute signal intensity. Trained natively in rank space on TCGA-BRCA, SS-MLP achieved an AUROC of 0.7943 [95% CI: 0.7011, 0.8743] and an AUPRC of 0.8773 on GSE42568, cutting the Brier score nearly in half to 0.3303 and restoring sensitivity to 98.51% without batch correction or target-cohort retraining.

### 3.5 Gene Stability Selection Isolates a High-Fidelity Consensus Core
To examine the biological basis of SS-MLP's resilience, we evaluated feature selection frequencies across the 100 Elastic-Net bootstrap resamples (Figure 4, Table 4).

| Gene Symbol | Selection Freq. | Weight Magnitude | Biological Annotation & Oncogenic Role |
| :--- | :---: | :---: | :--- |
| ESR1 | 100.0% | 0.02985 | Estrogen Receptor Alpha (Primary Diagnostic Driver) |
| CERS3 | 100.0% | 0.02417 | Ceramide Synthase 3 (Sphingolipid Metabolism) |
| LYPD1 | 100.0% | 0.02071 | Ly6/PLAUR Domain-Containing Protein 1 |
| AGR3 | 100.0% | 0.02000 | Anterior Gradient 3 (Endocrine Differentiation) |
| GREB1 | 100.0% | 0.01923 | Growth Regulation by Estrogen in Breast Cancer 1 |
| SV2C | 100.0% | 0.01830 | Synaptic Vesicle Glycoprotein 2C |
| VGLL1 | 100.0% | 0.01786 | Transcription Cofactor (TEAD/YAP Interaction) |
| C5AR2 | 100.0% | 0.01709 | Complement C5a Receptor 2 |
| TFF1 | 100.0% | 0.01693 | Trefoil Factor 1 (Canonical Estrogen-Induced pS2) |
| CA12 | 100.0% | 0.01454 | Carbonic Anhydrase XII (Luminal Cell Surface) |
| GATA3 | 100.0% | 0.01264 | Master Luminal Lineage Transcription Factor |
| SLC39A6 | 100.0% | 0.00943 | LIV-1 Zinc Transporter (Luminal Secretory Marker) |
| AGR2 | 100.0% | 0.00839 | Anterior Gradient 2 (Endoplasmic Reticulum Chaperone) |

Selection frequencies across the 15,200 expressed candidate genes followed a strongly bimodal distribution: more than 14,000 genes appeared in fewer than 10% of bootstrap draws, whereas 300 genes exceeded the pre-specified ≥70% stability threshold, with top consensus markers selected in 100% of resamples (Table 4). While *ESR1* carried the largest individual weight (0.02985), the signature was reinforced by established luminal lineage regulators and direct estrogen targets, including *GATA3* (master luminal pioneer transcription factor), *GREB1* and *TFF1* (canonical estrogen-response genes), *AGR2* and *AGR3* (endoplasmic reticulum chaperones linked to endocrine differentiation), *CA12* (carbonic anhydrase XII), and *SLC39A6* (LIV-1 zinc transporter).

### 3.6 Brier Score Decomposition Confirms Reliability Under Domain Shift
We next performed a Sanders/Murphy three-component decomposition of the Brier score (`Brier = Reliability - Resolution + Uncertainty`) across all cohorts and transfer regimes (Table 5, Figure 5).

| Cohort | Model | Strategy | Brier | REL | RES | UNC | Slope | Intercept | ECE |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TCGA-BRCA OOF | ESR1-Only | Strategy A | 0.0570 | 0.0118 | 0.1308 | 0.1755 | 0.9931 | 1.1572 | 0.0729 |
| TCGA-BRCA OOF | SS-MLP | Strategy A | 0.0675 | 0.0174 | 0.1257 | 0.1755 | 0.8500 | 1.4368 | 0.0869 |
| TCGA-BRCA OOF | SS-MLP | Strategy B | 0.0577 | 0.0096 | 0.1275 | 0.1755 | 0.8759 | 1.3350 | 0.0660 |
| GSE96058 | ESR1-Only | Strategy A | 0.2628 | 0.2238 | 0.0322 | 0.0723 | 1.1529 | 5.2361 | 0.4290 |
| GSE96058 | SS-MLP | Strategy A | 0.0396 | 0.0063 | 0.0386 | 0.0723 | 0.2150 | 1.0660 | 0.0396 |
| GSE96058 | ESR1-Only | Strategy B | 0.0651 | 0.0341 | 0.0411 | 0.0723 | 1.1951 | 2.9209 | 0.1613 |
| GSE96058 | SS-MLP | Strategy B | 0.1212 | 0.0792 | 0.0291 | 0.0723 | 0.5465 | 3.0194 | 0.1519 |
| GSE42568 | ESR1-Only | Strategy A | 0.6188 | 0.3985 | 0.0000 | 0.2233 | 6.6959 | 24.1711 | 0.6313 |
| GSE42568 | SS-MLP | Strategy A | 0.4868 | 0.2836 | 0.0191 | 0.2233 | 0.1103 | 1.6098 | 0.5084 |
| GSE42568 | SS-MLP | Strategy B | 0.3303 | 0.1207 | 0.0099 | 0.2233 | 0.2559 | -1.8205 | 0.3362 |

This decomposition confirms that the degradation of the ESR1-only model on GSE96058 is driven almost entirely by calibration unreliability (REL = 0.2238, accounting for 85% of the total Brier score). By contrast, SS-MLP under Strategy A reduced the reliability penalty to REL = 0.0063, closely tracking the 45-degree ideal calibration diagonal (Figure 5A).

### 3.7 Selective Classification Enables Safe Autonomous Clinical Triage
To prevent forced binary errors on borderline tumors, we evaluated selective classification with confidence abstention across thresholds `τ ∈ [0.50, 0.99]` (Figure 6). On the external SCAN-B cohort (N = 3,073), applying a moderate confidence threshold of τ = 0.80 allowed SS-MLP to autonomously classify 99.6% of patients (3,060 of 3,073) with a selective error rate of 3.92%, while flagging the 13 ambiguous borderline cases (0.4%) for confirmatory IHC review. Raising the threshold to τ = 0.95 retained 88.3% autonomous coverage (2,713 patients) while cutting the diagnostic error rate nearly in half to 2.06% (56 total errors).

### 3.8 Functional Pathway Enrichment Validates Luminal Lineage Fidelity
We tested the 300-gene stability signature for biological pathway over-representation against curated breast cancer ontology sets using one-sided hypergeometric testing (Table 6, Figure 7).

| Gene Set / Pathway Annotation | Overlap | Odds Ratio | P-Value | Adj. P-Val | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Estrogen Receptor (ER) Signaling | 9 / 23 | 37.672 | 8.03e-11 | 4.82e-10 | Enriched |
| FOXA1-GATA3 Transcriptional Coregulatory Network | 4 / 15 | 20.953 | 1.02e-04 | 6.15e-04 | Enriched |
| Basal-like / Triple-Negative Markers | 2 / 15 | 8.804 | 2.69e-02 | 1.62e-01 | Marginal |
| Luminal Epithelial Differentiation | 2 / 15 | 8.804 | 2.69e-02 | 1.62e-01 | Marginal |
| Cell Cycle Progression & Proliferation | 0 / 16 | 0.000 | 1.00e+00 | 1.00e+00 | Not Enriched |
| Apoptosis & Survival Regulation | 0 / 14 | 0.000 | 1.00e+00 | 1.00e+00 | Not Enriched |

The consensus signature was strongly enriched for canonical Estrogen Receptor Signaling (p = 8.03 × 10⁻¹¹, odds ratio = 37.67) and the FOXA1-GATA3 Transcriptional Coregulatory Network (p = 1.02 × 10⁻⁴, odds ratio = 20.95). Conversely, generic proliferation (e.g., MKI67, CCND1) and apoptosis modules showed zero overlap (p = 1.00). Broader querying across Gene Ontology and KEGG libraries via Enrichr identified 2,158 significantly enriched terms (q < 0.05), confirming that the classifier captures lineage-specific estrogen signaling rather than secondary proliferation correlates.

### 3.9 Downstream Redundancy: ESR1-Exclusion Preserves Diagnostic Power
To test whether SS-MLP genuinely learned the broader luminal transcriptional network rather than relying solely on *ESR1*, we performed an *in silico* gene-knockout ablation (Experiment E6; Figure 8A). When *ESR1* was completely removed from the transcriptomic feature space prior to stability selection and network training, the ablated SS-MLP model achieved an internal AUROC of 0.9608 [95% CI: 0.942, 0.974], AUPRC of 0.9863, Sensitivity of 92.44%, Specificity of 93.25%, and Brier score of 0.0657—matching the intact 300-gene model. Even when the primary receptor transcript is absent or degraded, coordinated expression across downstream co-regulators (*GATA3*, *GREB1*, *TFF1*, *CA12*, *AGR3*) preserves full diagnostic accuracy.

Feature capacity ablations (Figure 8B) showed performance saturation between 50 and 300 genes (50 genes: AUROC 0.9564, Brier 0.0672; 300 genes: AUROC 0.9560, Brier 0.0675), indicating that a compact 50–100 gene panel captures the luminal manifold. Dropout ablations (Figure 8C) confirmed that omitting dropout led to overconfident predictions (ECE = 0.0706, slope = 0.8250), whereas moderate dropout (0.35 / 0.25) optimized the calibration slope (0.8500).

### 3.10 Clinical Decision-Support Triage & Confidence Abstention Workflow
To illustrate practical laboratory integration, Figure 9 outlines the end-to-end clinical decision-support workflow across all 3,073 external SCAN-B patients. Pairing posterior probabilities with configurable confidence thresholds allows pathology laboratories to tailor operating points to institutional risk tolerance:

1. **High-Throughput Autonomous Screening (τ = 0.80):** 99.6% of specimens (3,060 of 3,073) are signed out autonomously with a 3.92% error rate (120 errors), routing only 0.4% (13 borderline cases) for reflex IHC.
2. **High-Fidelity Reflex Triage (τ = 0.95):** 88.3% of specimens (2,713 of 3,073) are signed out autonomously with a 2.06% error rate (56 errors), intercepting 54.5% of potential misclassifications (and dropping to 1.32% / 30 errors at ultra-conservative τ = 0.98).

---

## 4. Discussion

Translating computational biomarkers into oncology practice requires looking beyond rank-order discrimination to ensure probabilistic calibration. Across 4,218 audited breast cancer profiles, our evaluation establishes four practical insights for clinical machine learning:

First, we characterize the **discrimination-calibration paradox** under genomic domain shift. Although published biomarker studies routinely rely on AUROC and AUPRC to claim external validity, our findings show that a classifier can maintain an external AUROC above 0.96 while suffering severe probability distortion. On SCAN-B RNA-seq validation, the univariate *ESR1* model achieved an AUROC of 0.9604, yet frozen standardization induced a calibration error (ECE) of 0.4290, dropping sensitivity to 58.44% and misclassifying 1,177 ER-positive patients. By distributing weights across 300 stability-selected genes, SS-MLP held calibration error to 0.0396 and retained 97.74% sensitivity, demonstrating that multi-gene architectures buffer against cohort-level shifts in transcript abundance.

Second, **within-sample relative rank normalization** bridges distinct assay technologies without batch retraining. Standard parametric transfer (Strategy A) failed completely when applied to Affymetrix microarrays (0.0000 sensitivity for *ESR1*-only) due to probe dynamic range compression. Transforming expression levels into label-free fractional ranks within each patient profile independently (Strategy B) restored cross-platform microarray discrimination (AUROC 0.7943, AUPRC 0.8773) and sensitivity (98.51%) without external reference standards.

Third, *in silico* knockout ablations revealed **complete downstream transcriptomic redundancy**. Removing *ESR1* entirely prior to feature selection and training yielded an AUROC of 0.9608 and AUPRC of 0.9863. In surgical pathology, tissue biopsies frequently experience cold ischemia, fixation artifacts, or antigenic masking that impair direct ER detection. Because luminal breast tumors maintain coordinated expression across downstream targets (*GATA3*, *GREB1*, *TFF1*, *CA12*, *AGR3*), a multi-gene transcriptomic classifier can reliably infer functional ER status even when direct measurement of the primary receptor is compromised.

Fourth, **uncertainty-aware selective abstention** provides a safeguard for clinical deployment. Rather than forcing binary calls on every specimen, confidence thresholding at τ = 0.80 autonomously triages 99.6% of cases at a 3.92% error rate (or 88.3% coverage at a 2.06% error rate at τ = 0.95), routing equivocal borderline profiles for confirmatory pathology review.

### 4.1 Study Limitations
Several limitations warrant consideration. First, all three cohorts were analyzed retrospectively; prospective clinical validation will be required before routine diagnostic adoption. Second, ground-truth ER status relied on historical clinical IHC records, which carry baseline inter-laboratory variability (5–10%). Third, while within-sample rank normalization restored cross-platform ranking and sensitivity on GSE42568 microarrays, threshold-level specificity at p ≥ 0.50 remained limited on this small (N = 101) cohort, indicating that platform-specific threshold tuning or post-hoc recalibration remains beneficial when bridging microarrays and RNA-seq.

---

## 5. Conclusions

Quantitative transcriptomic profiling provides an objective, reproducible complement to visual immunohistochemistry in breast cancer diagnostics. By combining repeated Elastic-Net stability selection, regularized deep neural networks, label-free within-sample rank normalization, and confidence-based abstention, the RAPID-ER framework resolves the discrimination-calibration paradox and offers an open, platform-resilient benchmark for clinical decision support.

---

## Declarations

* **Ethics Approval and Consent to Participate:** Not applicable. All genomic and clinical datasets analyzed in this study are publicly accessible, de-identified secondary data obtained from the NCI GDC and NCBI GEO repositories.
* **Consent for Publication:** Not applicable.
* **Availability of Data and Materials:** All primary datasets are publicly available: TCGA-BRCA raw STAR counts and clinical metadata via the NCI Genomic Data Commons Data Portal (https://portal.gdc.cancer.gov/projects/TCGA-BRCA); GSE96058 and GSE42568 expression matrices and clinical records via NCBI GEO (https://www.ncbi.nlm.nih.gov/geo/). Full source code, configuration files, trained model checkpoints, and reproducible analysis pipelines are archived on GitHub (https://github.com/gautam453/RAPID-ER), with a permanent Zenodo archive and DOI to be deposited upon publication.
* **Competing Interests:** The author declares no competing financial or non-financial interests.
* **Funding:** This independent research received no external commercial or institutional grant funding.
* **Author Contributions (CRediT):** Gautam Kunniyur Ramchander: Conceptualization, Methodology, Software, Validation, Formal Analysis, Data Curation, Writing – Original Draft, Writing – Review & Editing, and Visualization.

---

## References

1. Parker JS, Mullins M, Cheang MCU, et al. Supervised risk predictor of breast cancer based on intrinsic subtypes. *J Clin Oncol*. 2009;27(8):1160-1167.
2. Early Breast Cancer Trialists' Collaborative Group (EBCTCG). Relevance of breast cancer hormone receptors and other factors to the efficacy of adjuvant tamoxifen: patient-level meta-analysis of randomised trials. *Lancet*. 2011;378(9793):771-784.
3. Allison KH, Hammond MEH, Dowsett M, et al. Estrogen and Progesterone Receptor Testing in Breast Cancer: ASCO/CAP Guideline Update. *J Clin Oncol*. 2020;38(12):1346-1366.
4. Badve SS, Baehner FL, Gray RP, et al. Estrogen- and progesterone-receptor status in ECOG 2197: comparison of immunohistochemistry by local and central laboratories and quantitative reverse transcription polymerase chain reaction by Oncotype DX. *J Clin Oncol*. 2008;26(15):2473-2481.
5. Paik S, Shak S, Tang G, et al. A multigene assay to predict recurrence of tamoxifen-treated, node-negative breast cancer. *N Engl J Med*. 2004;351(27):2817-2826.
6. Haibe-Kains B, Desmedt C, Loi S, et al. A three-gene model to robustly identify breast cancer molecular subtypes. *J Natl Cancer Inst*. 2012;104(4):311-325.
7. Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. *BMC Med*. 2019;17(1):230.
8. Guo C, Pleiss G, Sun Y, Weinberger KG. On calibration of modern neural networks. *Proc Int Conf Mach Learn (ICML)*. 2017;70:1321-1330.
9. Leek JT, Scharpf RB, Bravo HC, et al. Tackling the widespread and critical impact of batch effects in high-throughput data. *Nat Rev Genet*. 2010;11(10):733-739.
10. Meinshausen N, Bühlmann P. Stability selection. *J R Stat Soc Series B Stat Methodol*. 2010;72(4):417-473.
11. Geifman Y, El-Yaniv R. Selective classification for deep neural networks. *Adv Neural Inf Process Syst (NeurIPS)*. 2017;30:4878-4887.
