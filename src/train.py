import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.feature_selection import (
    filter_expressed_genes,
    normalize_logcpm,
    rank_normalize_within_sample,
    StabilitySelector
)
from src.models import get_model, ESR1OnlyClassifier
from src.statistics import compute_all_metrics, bootstrap_metric_ci

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
META_DIR = BASE_DIR / "metadata"
RESULTS_DIR = BASE_DIR / "results"
INTERNAL_DIR = RESULTS_DIR / "internal"
TABLES_DIR = RESULTS_DIR / "tables"

INTERNAL_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

def train_cv(model_name="ss_mlp", transfer_strategy="frozen_training_norm", random_state=42, force=False):
    print("=" * 70)
    print(f"TRAINING 5-FOLD PATIENT-LEVEL CV: Model={model_name}, Strategy={transfer_strategy}")
    print("=" * 70)

    # 1. Load data
    df_tcga = pd.read_parquet(PROCESSED_DIR / "tcga_counts.parquet")
    df_folds = pd.read_csv(META_DIR / "patient_folds.csv")
    
    # Align rows by patient_id
    if "patient_id" in df_tcga.columns:
        df_tcga = df_tcga.set_index("patient_id")
    df_tcga = df_tcga.loc[df_folds["patient_id"]].copy()
    gene_cols = [c for c in df_tcga.columns if c != "er_status"]
    
    counts_all = df_tcga[gene_cols].values
    labels_all = df_tcga["er_status"].values.astype(int)
    patient_ids = df_tcga.index.values
    folds = df_folds["fold"].values
    
    n_samples = len(df_tcga)
    oof_probs = np.zeros(n_samples, dtype=float)
    oof_preds = np.zeros(n_samples, dtype=int)
    
    # Check if model already trained and cached
    oof_path = INTERNAL_DIR / f"oof_predictions_{model_name}_{transfer_strategy}.csv"
    pkg_path = INTERNAL_DIR / f"frozen_pipeline_{model_name}_{transfer_strategy}.joblib"
    if not force and oof_path.exists() and pkg_path.exists():
        print(f"Model {model_name} ({transfer_strategy}) already completed! Loading existing out-of-fold metrics...")
        df_oof = pd.read_csv(oof_path)
        overall_metrics = compute_all_metrics(df_oof["true_label"].values, df_oof["predicted_prob"].values, threshold=0.5)
        overall_metrics["Model"] = model_name
        overall_metrics["Strategy"] = transfer_strategy
        return overall_metrics
    
    fold_metrics = []
    fold_selected_genes = []
    
    t0 = time.time()
    
    for fold in range(5):
        print(f"\n--- FOLD {fold} ---")
        train_mask = (folds != fold)
        val_mask = (folds == fold)
        
        train_counts = counts_all[train_mask]
        train_y = labels_all[train_mask]
        
        val_counts = counts_all[val_mask]
        val_y = labels_all[val_mask]
        
        # 1. Low-count gene filtering (TRAIN ONLY)
        expressed_mask = filter_expressed_genes(train_counts, min_count=10, min_sample_ratio=0.20)
        expressed_genes = [g for i, g in enumerate(gene_cols) if expressed_mask[i]]
        print(f"  Expressed genes retained (train fold {fold}): {len(expressed_genes)}/{len(gene_cols)}")
        
        X_train_exp = train_counts[:, expressed_mask]
        X_val_exp = val_counts[:, expressed_mask]
        
        # 2. Normalization
        if transfer_strategy == "frozen_training_norm":
            X_train_log = normalize_logcpm(X_train_exp)
            X_val_log = normalize_logcpm(X_val_exp)
            
            # Standardization (fit train only)
            mean_train = np.mean(X_train_log, axis=0)
            std_train = np.std(X_train_log, axis=0) + 1e-8
            
            X_train_proc = (X_train_log - mean_train) / std_train
            X_val_proc = (X_val_log - mean_train) / std_train
        elif transfer_strategy == "rank_norm":
            X_train_log = normalize_logcpm(X_train_exp)
            X_val_log = normalize_logcpm(X_val_exp)
            X_train_proc = rank_normalize_within_sample(X_train_log)
            X_val_proc = rank_normalize_within_sample(X_val_log)
        else:
            raise ValueError(f"Unknown transfer strategy: {transfer_strategy}")
            
        # 3. Feature Selection
        if model_name == "esr1_logistic":
            esr1_idx = expressed_genes.index("ESR1")
            X_train_sel = X_train_proc[:, [esr1_idx]]
            X_val_sel = X_val_proc[:, [esr1_idx]]
            selected_gene_names = ["ESR1"]
        elif model_name in ["ss_mlp", "elastic_net", "logistic_regression", "svm_linear", "svm_rbf", "random_forest", "xgboost"]:
            # Run or load cached Stability Selection inside train fold
            cache_file = INTERNAL_DIR / f"cached_stability_selector_fold_{fold}_{transfer_strategy}.joblib"
            if cache_file.exists():
                print(f"  Loading cached Stability Selector for fold {fold}...")
                selector = joblib.load(cache_file)
            else:
                selector = StabilitySelector(
                    n_resamples=100,
                    subsample_ratio=0.8,
                    l1_ratio=0.5,
                    C=0.1,
                    frequency_threshold=0.70,
                    max_genes=300,
                    random_state=random_state + fold
                )
                selector.fit(X_train_proc, train_y)
                joblib.dump(selector, cache_file)
            X_train_sel = selector.transform(X_train_proc)
            X_val_sel = selector.transform(X_val_proc)
            selected_gene_names = [expressed_genes[i] for i in selector.selected_indices_]
            print(f"  Stability-selected genes: {len(selected_gene_names)} (Threshold >= 70%)")
            fold_selected_genes.append(selected_gene_names)
            
        # 4. Fit Model
        model = get_model(model_name, random_state=random_state + fold)
        if model_name == "ss_mlp":
            model.fit(X_train_sel, train_y, X_val=X_val_sel, y_val=val_y)
        else:
            model.fit(X_train_sel, train_y)
            
        # 5. Predict on out-of-fold validation data
        val_probs = model.predict_proba(X_val_sel)[:, 1]
        oof_probs[val_mask] = val_probs
        oof_preds[val_mask] = (val_probs >= 0.5).astype(int)
        
        fold_metrics_dict = compute_all_metrics(val_y, val_probs)
        fold_metrics.append(fold_metrics_dict)
        print(f"  Fold {fold} Results: AUROC={fold_metrics_dict['AUROC']:.4f}, AUPRC={fold_metrics_dict['AUPRC']:.4f}, Brier={fold_metrics_dict['Brier_Score']:.4f}, ECE={fold_metrics_dict['ECE']:.4f}")

    total_time = time.time() - t0
    print(f"\nCV finished in {total_time:.2f} seconds.")

    # Save OOF predictions
    df_oof = pd.DataFrame({
        "patient_id": patient_ids,
        "true_label": labels_all,
        "fold": folds,
        "predicted_prob": oof_probs,
        "predicted_class": oof_preds
    })
    oof_path = INTERNAL_DIR / f"oof_predictions_{model_name}_{transfer_strategy}.csv"
    df_oof.to_csv(oof_path, index=False)
    print(f"Saved OOF predictions to {oof_path}")

    # Overall OOF metrics
    overall_metrics = compute_all_metrics(labels_all, oof_probs)
    print("\n" + "=" * 60)
    print(f"OVERALL OUT-OF-FOLD METRICS ({model_name.upper()}):")
    print("=" * 60)
    for k, v in overall_metrics.items():
        if isinstance(v, float):
            print(f"  {k:22s}: {v:.4f}")
        else:
            print(f"  {k:22s}: {v}")

    # Compute bootstrap 95% CIs for AUROC, AUPRC, Brier, ECE
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from src.statistics import compute_ece
    
    ci_auroc = bootstrap_metric_ci(labels_all, oof_probs, lambda y, p: roc_auc_score(y, p))
    ci_auprc = bootstrap_metric_ci(labels_all, oof_probs, lambda y, p: average_precision_score(y, p))
    ci_brier = bootstrap_metric_ci(labels_all, oof_probs, lambda y, p: brier_score_loss(y, p))
    ci_ece = bootstrap_metric_ci(labels_all, oof_probs, lambda y, p: compute_ece(y, p))
    
    print(f"  AUROC 95% CI: [{ci_auroc[0]:.4f}, {ci_auroc[1]:.4f}]")
    print(f"  AUPRC 95% CI: [{ci_auprc[0]:.4f}, {ci_auprc[1]:.4f}]")
    print(f"  Brier 95% CI: [{ci_brier[0]:.4f}, {ci_brier[1]:.4f}]")
    print(f"  ECE 95% CI:   [{ci_ece[0]:.4f}, {ci_ece[1]:.4f}]")

    overall_metrics["AUROC_CI_low"] = ci_auroc[0]
    overall_metrics["AUROC_CI_high"] = ci_auroc[1]
    overall_metrics["AUPRC_CI_low"] = ci_auprc[0]
    overall_metrics["AUPRC_CI_high"] = ci_auprc[1]
    overall_metrics["Brier_CI_low"] = ci_brier[0]
    overall_metrics["Brier_CI_high"] = ci_brier[1]
    overall_metrics["ECE_CI_low"] = ci_ece[0]
    overall_metrics["ECE_CI_high"] = ci_ece[1]
    overall_metrics["Model"] = model_name
    overall_metrics["Strategy"] = transfer_strategy

    # Fit frozen pipeline on 100% of TCGA cohort
    print("\nFitting Frozen Pipeline on full TCGA-BRCA cohort...")
    exp_mask_full = filter_expressed_genes(counts_all, min_count=10, min_sample_ratio=0.20)
    genes_full = [g for i, g in enumerate(gene_cols) if exp_mask_full[i]]
    X_full_exp = counts_all[:, exp_mask_full]
    
    X_full_log = normalize_logcpm(X_full_exp)
    mean_full = np.mean(X_full_log, axis=0)
    std_full = np.std(X_full_log, axis=0) + 1e-8
    
    if transfer_strategy == "frozen_training_norm":
        X_full_proc = (X_full_log - mean_full) / std_full
    elif transfer_strategy == "rank_norm":
        X_full_proc = rank_normalize_within_sample(X_full_log)
    else:
        raise ValueError(f"Unknown transfer strategy: {transfer_strategy}")
    
    if model_name == "esr1_logistic":
        esr1_idx = genes_full.index("ESR1")
        X_full_sel = X_full_proc[:, [esr1_idx]]
        frozen_genes = ["ESR1"]
        selector_full = None
    else:
        cache_full = INTERNAL_DIR / f"cached_stability_selector_full_{transfer_strategy}.joblib"
        if cache_full.exists():
            print("Loading cached full dataset Stability Selector...")
            selector_full = joblib.load(cache_full)
        else:
            selector_full = StabilitySelector(
                n_resamples=100,
                subsample_ratio=0.8,
                l1_ratio=0.5,
                C=0.1,
                frequency_threshold=0.70,
                max_genes=300,
                random_state=random_state
            )
            selector_full.fit(X_full_proc, labels_all)
            joblib.dump(selector_full, cache_full)
        X_full_sel = selector_full.transform(X_full_proc)
        frozen_genes = [genes_full[i] for i in selector_full.selected_indices_]
        print(f"Frozen model selected signature size: {len(frozen_genes)} genes")
        
        # Save signature genes with selection frequencies (Table 5)
        df_sig = pd.DataFrame({
            "gene_symbol": frozen_genes,
            "selection_frequency": selector_full.selection_frequencies_[selector_full.selected_indices_],
            "coef_magnitude": selector_full.coef_magnitudes_[selector_full.selected_indices_]
        }).sort_values(by="selection_frequency", ascending=False)
        df_sig.to_csv(TABLES_DIR / f"table5_stable_signature_{model_name}.csv", index=False)
        print(f"Saved stable signature table to {TABLES_DIR / f'table5_stable_signature_{model_name}.csv'}")

    frozen_model = get_model(model_name, random_state=random_state)
    frozen_model.fit(X_full_sel, labels_all)
    
    # Compute optimal Youden threshold strictly from OOF predictions
    from sklearn.metrics import roc_curve
    fpr_oof, tpr_oof, thresh_oof = roc_curve(labels_all, oof_probs)
    youden_idx = np.argmax(tpr_oof - fpr_oof)
    opt_threshold = float(thresh_oof[youden_idx])
    print(f"Optimal OOF decision threshold (Youden J): {opt_threshold:.4f}")
    
    frozen_package = {
        "model_name": model_name,
        "transfer_strategy": transfer_strategy,
        "common_genes": gene_cols,
        "expressed_mask": exp_mask_full,
        "expressed_genes": genes_full,
        "mean": mean_full,
        "std": std_full,
        "selected_genes": frozen_genes,
        "selector": selector_full,
        "model": frozen_model,
        "threshold": 0.5,
        "optimal_threshold": opt_threshold
    }
    pkg_path = INTERNAL_DIR / f"frozen_pipeline_{model_name}_{transfer_strategy}.joblib"
    joblib.dump(frozen_package, pkg_path)
    print(f"Saved frozen reproducible pipeline package to {pkg_path}")

    return overall_metrics

def run_all_baselines():
    models = [
        "esr1_logistic",
        "logistic_regression",
        "elastic_net",
        "svm_linear",
        "svm_rbf",
        "random_forest",
        "xgboost",
        "ss_mlp"
    ]
    all_results = []
    for m in models:
        res = train_cv(model_name=m, transfer_strategy="frozen_training_norm")
        all_results.append(res)
        
    df_all = pd.DataFrame(all_results)
    table2_path = TABLES_DIR / "table2_internal_model_comparison.csv"
    df_all.to_csv(table2_path, index=False)
    print("\n" + "=" * 70)
    print("TABLE 2: INTERNAL 5-FOLD CROSS-VALIDATION MODEL COMPARISON")
    print("=" * 70)
    cols_show = ["Model", "AUROC", "AUPRC", "Sensitivity", "Specificity", "F1", "MCC", "Brier_Score", "ECE"]
    print(df_all[cols_show].to_string(index=False))
    print(f"\nSaved Table 2 to {table2_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ss_mlp", choices=["esr1_logistic", "logistic_regression", "elastic_net", "svm_linear", "svm_rbf", "random_forest", "xgboost", "ss_mlp", "all"])
    parser.add_argument("--strategy", type=str, default="frozen_training_norm")
    parser.add_argument("--force", action="store_true", help="Force re-fitting")
    args = parser.parse_args()

    if args.model == "all":
        run_all_baselines()
    else:
        train_cv(model_name=args.model, transfer_strategy=args.strategy, force=args.force)
