import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.feature_selection import rank_normalize_within_sample
from src.statistics import compute_all_metrics, bootstrap_metric_ci, compute_ece

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
INTERNAL_DIR = RESULTS_DIR / "internal"
EXTERNAL_DIR = RESULTS_DIR / "external"
TABLES_DIR = RESULTS_DIR / "tables"

EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = {
    "GSE96058": {
        "file": "gse96058_expression.parquet",
        "platform": "Illumina HiSeq (RNA-seq)",
        "assay": "RNA-seq expression"
    },
    "GSE42568": {
        "file": "gse42568_expression.parquet",
        "platform": "Affymetrix HG-U133 Plus 2.0 (GPL570)",
        "assay": "log2 GC-RMA microarray"
    }
}

def validate_external(model_name="ss_mlp", cohort_name="GSE96058", strategy="frozen_training_norm", threshold=0.5):
    print("=" * 70)
    print(f"EXTERNAL VALIDATION: Model={model_name}, Cohort={cohort_name}, Strategy={strategy}")
    print("=" * 70)

    # 1. Load frozen pipeline
    pkg_path = INTERNAL_DIR / f"frozen_pipeline_{model_name}_{strategy}.joblib"
    if not pkg_path.exists():
        # Fallback to frozen_training_norm if strategy is rank_norm and trained under rank_norm
        raise FileNotFoundError(f"Frozen pipeline package not found at {pkg_path}. Ensure internal CV / training has completed.")

    package = joblib.load(pkg_path)
    model = package["model"]
    expressed_genes = package["expressed_genes"]
    selected_genes = package["selected_genes"]
    mean_train = package["mean"]
    std_train = package["std"]

    print(f"Loaded frozen model: {model_name} with {len(selected_genes)} signature genes.")

    # 2. Load cohort data
    cohort_info = COHORTS[cohort_name]
    df_cohort = pd.read_parquet(PROCESSED_DIR / cohort_info["file"])
    
    if "patient_id" in df_cohort.columns:
        pids = df_cohort["patient_id"].values
    else:
        pids = df_cohort.index.values

    y_true = df_cohort["er_status"].values.astype(int)
    gene_cols = [c for c in df_cohort.columns if c not in ["patient_id", "er_status"]]

    # Align columns to expressed_genes
    df_expr = df_cohort[expressed_genes].copy()
    X_raw = df_expr.values.astype(np.float32)

    # 3. Apply transfer strategy
    if strategy == "frozen_training_norm":
        X_proc = (X_raw - mean_train) / std_train
    elif strategy == "rank_norm":
        X_proc = rank_normalize_within_sample(X_raw)
    else:
        raise ValueError(f"Unknown transfer strategy: {strategy}")

    # Extract selected genes
    selected_indices = [expressed_genes.index(g) for g in selected_genes]
    X_sel = X_proc[:, selected_indices]

    # 4. Predict
    probs = model.predict_proba(X_sel)[:, 1]
    preds = (probs >= threshold).astype(int)

    # 5. Evaluate
    metrics = compute_all_metrics(y_true, probs, threshold=threshold)
    metrics["Cohort"] = cohort_name
    metrics["Platform"] = cohort_info["platform"]
    metrics["Strategy"] = strategy
    metrics["Model"] = model_name
    metrics["N_samples"] = len(y_true)
    metrics["N_ER_pos"] = int(np.sum(y_true == 1))
    metrics["N_ER_neg"] = int(np.sum(y_true == 0))

    # Bootstrap 95% CIs
    ci_auroc = bootstrap_metric_ci(y_true, probs, lambda y, p: roc_auc_score(y, p))
    ci_auprc = bootstrap_metric_ci(y_true, probs, lambda y, p: average_precision_score(y, p))
    ci_brier = bootstrap_metric_ci(y_true, probs, lambda y, p: brier_score_loss(y, p))
    ci_ece = bootstrap_metric_ci(y_true, probs, lambda y, p: compute_ece(y, p))

    metrics["AUROC_CI_low"] = ci_auroc[0]
    metrics["AUROC_CI_high"] = ci_auroc[1]
    metrics["AUPRC_CI_low"] = ci_auprc[0]
    metrics["AUPRC_CI_high"] = ci_auprc[1]
    metrics["Brier_CI_low"] = ci_brier[0]
    metrics["Brier_CI_high"] = ci_brier[1]
    metrics["ECE_CI_low"] = ci_ece[0]
    metrics["ECE_CI_high"] = ci_ece[1]

    print("\nRESULTS:")
    print(f"  AUROC:       {metrics['AUROC']:.4f} [{ci_auroc[0]:.4f}, {ci_auroc[1]:.4f}]")
    print(f"  AUPRC:       {metrics['AUPRC']:.4f} [{ci_auprc[0]:.4f}, {ci_auprc[1]:.4f}]")
    print(f"  Sensitivity: {metrics['Sensitivity']:.4f}")
    print(f"  Specificity: {metrics['Specificity']:.4f}")
    print(f"  F1-Score:    {metrics['F1']:.4f}")
    print(f"  MCC:         {metrics['MCC']:.4f}")
    print(f"  Brier Score: {metrics['Brier_Score']:.4f} [{ci_brier[0]:.4f}, {ci_brier[1]:.4f}]")
    print(f"  Calib Slope: {metrics['Calibration_Slope']:.4f}")
    print(f"  Calib Int:   {metrics['Calibration_Intercept']:.4f}")
    print(f"  ECE:         {metrics['ECE']:.4f} [{ci_ece[0]:.4f}, {ci_ece[1]:.4f}]")

    # 6. Save external predictions
    df_preds = pd.DataFrame({
        "patient_id": pids,
        "true_label": y_true,
        "predicted_prob": probs,
        "predicted_class": preds
    })
    pred_path = EXTERNAL_DIR / f"predictions_{cohort_name}_{model_name}_{strategy}.csv"
    df_preds.to_csv(pred_path, index=False)
    print(f"Saved external predictions to {pred_path}")

    # 7. Update Table 3
    table3_path = TABLES_DIR / "table3_external_validation_performance.csv"
    if table3_path.exists():
        df_t3 = pd.read_csv(table3_path)
        # Drop existing row if present
        mask = (df_t3["Cohort"] == cohort_name) & (df_t3["Model"] == model_name) & (df_t3["Strategy"] == strategy)
        df_t3 = df_t3[~mask]
        df_t3 = pd.concat([df_t3, pd.DataFrame([metrics])], ignore_index=True)
    else:
        df_t3 = pd.DataFrame([metrics])
        
    df_t3.to_csv(table3_path, index=False)
    print(f"Updated Table 3 at {table3_path}")
    return metrics

def validate_all():
    models = ["esr1_logistic", "logistic_regression", "elastic_net", "svm_linear", "svm_rbf", "random_forest", "xgboost", "ss_mlp"]
    strategies = ["frozen_training_norm", "rank_norm"]
    cohorts = ["GSE96058", "GSE42568"]

    for c in cohorts:
        for s in strategies:
            for m in models:
                try:
                    validate_external(model_name=m, cohort_name=c, strategy=s)
                except Exception as e:
                    print(f"Skipping {m} on {c} ({s}): {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="esr1_logistic")
    parser.add_argument("--cohort", type=str, default="GSE96058", choices=["GSE96058", "GSE42568", "all"])
    parser.add_argument("--strategy", type=str, default="frozen_training_norm", choices=["frozen_training_norm", "rank_norm", "all"])
    args = parser.parse_args()

    cohort_list = ["GSE96058", "GSE42568"] if args.cohort == "all" else [args.cohort]
    strat_list = ["frozen_training_norm", "rank_norm"] if args.strategy == "all" else [args.strategy]

    for c in cohort_list:
        for s in strat_list:
            validate_external(model_name=args.model, cohort_name=c, strategy=s)
