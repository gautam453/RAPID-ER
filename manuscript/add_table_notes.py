import re
from pathlib import Path

notes = {
"tab:table1": r"""\textit{Table notes.} \textbf{Role in Study:} use of each cohort (training/internal CV vs.\ external validation). \textbf{Platform:} profiling technology. \textbf{Expression Scale:} native units of the deposited data (raw STAR read counts; $\log_2$-transformed FPKM; $\log_2$ GC-RMA intensities). \textbf{Raw Biospecimens:} files/profiles before auditing. \textbf{Excluded Samples:} total removed, split into technical replicates (duplicate vials/runs), non-tumour controls, and records with missing or indeterminate ER status. \textbf{Final Eligible Patients ($N$):} unique primary tumours retained for analysis; percentages are of $N$. \textbf{Clinical ER Reference:} method used to establish the ground-truth ER label. Abbreviations: ER, estrogen receptor; CV, cross-validation; FPKM, fragments per kilobase per million; GC-RMA, GC-content robust multi-array average; IHC, immunohistochemistry.""",
"tab:table2": r"""\textit{Table notes.} Out-of-fold (OOF) predictions pooled across the five patient-stratified folds of TCGA-BRCA ($N=1{,}044$). \textbf{AUROC/AUPRC:} area under the ROC / precision--recall curve (higher is better; discrimination only). \textbf{Sens./Spec.:} sensitivity (true-positive rate for ER-positive) and specificity (true-negative rate for ER-negative) at a probability threshold of 0.50. \textbf{Brier:} mean squared error of predicted probabilities (lower is better; 0 is perfect). \textbf{ECE:} expected calibration error over 10 equal-width bins (lower is better). Values in brackets are 95\% confidence intervals. Bold row: proposed SS-MLP. All models used identical patient folds.""",
"tab:table3": r"""\textit{Table notes.} Models were trained on TCGA-BRCA only and applied unchanged to each external cohort. \textbf{Strategy A (Frozen):} training-set feature means and standard deviations applied to external data. \textbf{Strategy B (Rank):} within-sample fractional rank normalization (model trained directly in rank space). \textbf{Sens./Spec.:} at probability threshold 0.50. \textbf{Brier/ECE:} lower is better. Brackets give 95\% confidence intervals. Bold rows: proposed SS-MLP. \textbf{Interpretation caution:} on GSE42568 under Strategy B, SS-MLP achieves the highest AUROC/AUPRC and a sensitivity of 0.9851 but a specificity of 0.0000, i.e.\ at the 0.50 threshold nearly all tumours are called ER-positive; rank normalization therefore restores ranking ability but not threshold-level calibration on this small ($N=101$) microarray cohort.""",
"tab:table4": r"""\textit{Table notes.} Top consensus genes from repeated Elastic-Net stability selection (100 bootstrap resamples, 80\% subsampling), ordered by absolute coefficient magnitude. \textbf{Selection Freq.:} proportion of resamples in which the gene had a non-zero coefficient (threshold for inclusion: $\ge 70\%$). \textbf{Weight Magnitude:} absolute coefficient magnitude of the gene in the consensus model (arbitrary units on the standardized scale). \textbf{Biological Annotation:} curated function of the gene product. Only 13 of the 300 selected genes are shown; the full list is provided in the supplementary tables.""",
"tab:table5": r"""\textit{Table notes.} Sanders/Murphy decomposition: $\text{Brier} = \text{REL} - \text{RES} + \text{UNC}$. \textbf{REL (Reliability):} miscalibration penalty (lower is better; 0 = perfectly calibrated). \textbf{RES (Resolution):} ability to separate outcomes beyond the base rate (higher is better). \textbf{UNC (Uncertainty):} inherent outcome variance set by class prevalence ($p(1-p)$); identical across models within a cohort. \textbf{Slope/Intercept:} logistic recalibration of outcome on logit-predicted probability (ideal: slope $=1$, intercept $=0$). \textbf{ECE:} expected calibration error (10 bins). OOF, out-of-fold. Components may not sum exactly to Brier because of binning and rounding.""",
"tab:table6": r"""\textit{Table notes.} Hypergeometric over-representation of the 300-gene consensus signature in curated gene sets. \textbf{Overlap:} number of signature genes in the gene set / gene-set size. \textbf{Odds Ratio:} enrichment effect size. \textbf{P-Value:} one-sided hypergeometric test. \textbf{Adj. P-Val:} multiple-testing-adjusted p-value across the six gene sets. \textbf{Status:} Enriched = adjusted $p < 0.05$; Marginal = nominal $p < 0.05$ but adjusted $p \ge 0.05$; Not Enriched = no overlap.""",
}

for fname in ["main.tex", "journal_main.tex"]:
    p = Path("C:/Users/gauta/.gemini/antigravity/scratch/RAPID_ER/manuscript") / fname
    t = p.read_text(encoding="utf-8")
    for label, note in notes.items():
        block = None
        for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", t, re.DOTALL):
            if "\\label{" + label + "}" in m.group(0):
                block = m.group(0)
                break
        assert block, label
        if "Table notes." in block:
            continue
        new_block = block[: -len(r"\end{table}")] + "\\par\\vspace{4pt}\n\\begin{flushleft}\n\\footnotesize\n" + note + "\n\\end{flushleft}\n\\end{table}"
        t = t.replace(block, new_block)
    p.write_text(t, encoding="utf-8")
    print("updated", fname, t.count("Table notes."))
