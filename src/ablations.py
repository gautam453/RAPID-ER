from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import joblib

from src.feature_selection import filter_expressed_genes, normalize_logcpm, StabilitySelector
from src.models import SSMLPClassifier, get_model
from src.statistics import compute_all_metrics, bootstrap_metric_ci

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
META_DIR = BASE_DIR / "metadata"
ABLATIONS_DIR = BASE_DIR / "results" / "ablations"
TABLES_DIR = BASE_DIR / "results" / "tables"

ABLATIONS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    df_tcga = pd.read_parquet(PROCESSED_DIR / "tcga_counts.parquet")
    df_folds = pd.read_csv(META_DIR / "patient_folds.csv")
    if "patient_id" in df_tcga.columns:
        df_tcga = df_tcga.set_index("patient_id")
    df_tcga = df_tcga.loc[df_folds["patient_id"]].copy()
    gene_cols = [c for c in df_tcga.columns if c != "er_status"]
    counts = df_tcga[gene_cols].values
    labels = df_tcga["er_status"].values.astype(int)
    patient_ids = df_tcga.index.values
    folds = df_folds["fold"].values
    return counts, labels, patient_ids, folds, gene_cols

def run_esr1_exclusion_ablation():
    """Experiment E6: Train SS-MLP with ESR1 strictly excluded from the gene pool."""
    print("=" * 70)
    print("EXPERIMENT E6: ESR1 EXCLUSION ABLATION")
    print("=" * 70)
    
    counts, labels, pids, folds, gene_cols = load_data()
    n_samples = len(labels)
    oof_probs = np.zeros(n_samples)
    
    for fold in range(5):
        print(f"\n--- FOLD {fold} (ESR1 Excluded) ---")
        train_mask = (folds != fold)
        val_mask = (folds == fold)
        
        train_counts = counts[train_mask]
        train_y = labels[train_mask]
        val_counts = counts[val_mask]
        val_y = labels[val_mask]
        
        expressed_mask = filter_expressed_genes(train_counts)
        expressed_genes = [g for i, g in enumerate(gene_cols) if expressed_mask[i]]
        
        # EXCLUDE ESR1
        non_esr1_indices = [i for i, g in enumerate(expressed_genes) if g != "ESR1"]
        expressed_genes_no_esr1 = [expressed_genes[i] for i in non_esr1_indices]
        
        X_train_exp = train_counts[:, expressed_mask][:, non_esr1_indices]
        X_val_exp = val_counts[:, expressed_mask][:, non_esr1_indices]
        
        X_train_log = normalize_logcpm(X_train_exp)
        X_val_log = normalize_logcpm(X_val_exp)
        
        mean_tr = np.mean(X_train_log, axis=0)
        std_tr = np.std(X_train_log, axis=0) + 1e-8
        
        X_tr_proc = (X_train_log - mean_tr) / std_tr
        X_val_proc = (X_val_log - mean_tr) / std_tr

        # Variance screening to top 2000 most variable non-ESR1 genes for fast convergence
        vars_tr = np.var(X_tr_proc, axis=0)
        top_var_idx = np.argsort(vars_tr)[::-1][:min(2000, X_tr_proc.shape[1])]
        X_tr_screened = X_tr_proc[:, top_var_idx]
        X_val_screened = X_val_proc[:, top_var_idx]
        
        selector = StabilitySelector(
            n_resamples=50,
            subsample_ratio=0.8,
            l1_ratio=0.5,
            C=0.1,
            frequency_threshold=0.70,
            max_genes=300,
            random_state=42 + fold
        )
        selector.fit(X_tr_screened, train_y)
        X_tr_sel = selector.transform(X_tr_screened)
        X_val_sel = selector.transform(X_val_screened)
        
        model = SSMLPClassifier(random_state=42 + fold)
        model.fit(X_tr_sel, train_y, X_val=X_val_sel, y_val=val_y)
        oof_probs[val_mask] = model.predict_proba(X_val_sel)[:, 1]
        
    metrics = compute_all_metrics(labels, oof_probs)
    metrics["Ablation"] = "SS-MLP (ESR1 Excluded)"
    print(f"ESR1 Excluded Results: AUROC={metrics['AUROC']:.4f}, AUPRC={metrics['AUPRC']:.4f}, Brier={metrics['Brier_Score']:.4f}")
    
    # Save predictions and summary
    df_pred = pd.DataFrame({"patient_id": pids, "true_label": labels, "predicted_prob": oof_probs})
    df_pred.to_csv(ABLATIONS_DIR / "predictions_e6_esr1_excluded.csv", index=False)
    
    df_summary = pd.DataFrame([metrics])
    df_summary.to_csv(ABLATIONS_DIR / "table_e6_esr1_ablation.csv", index=False)
    return metrics

def run_gene_count_ablation():
    """Experiment E7: Gene count ablations (top 50, 100, 200, 300 genes)."""
    print("=" * 70)
    print("EXPERIMENT E7: GENE COUNT ABLATIONS (50, 100, 200, 300)")
    print("=" * 70)
    
    counts, labels, pids, folds, gene_cols = load_data()
    gene_counts = [50, 100, 200, 300]
    results = []
    
    for k in gene_counts:
        print(f"\nEvaluating Top {k} Genes...")
        oof_probs = np.zeros(len(labels))
        for fold in range(5):
            train_mask = (folds != fold)
            val_mask = (folds == fold)
            
            X_tr_exp = counts[train_mask][:, filter_expressed_genes(counts[train_mask])]
            X_val_exp = counts[val_mask][:, filter_expressed_genes(counts[train_mask])]
            
            X_tr_log = normalize_logcpm(X_tr_exp)
            X_val_log = normalize_logcpm(X_val_exp)
            
            mean_tr = np.mean(X_tr_log, axis=0)
            std_tr = np.std(X_tr_log, axis=0) + 1e-8
            X_tr_proc = (X_tr_log - mean_tr) / std_tr
            X_val_proc = (X_val_log - mean_tr) / std_tr
            
            # Load cached fold selector
            cache_file = BASE_DIR / "results" / "internal" / f"cached_stability_selector_fold_{fold}_frozen_training_norm.joblib"
            if cache_file.exists():
                selector = joblib.load(cache_file)
                # Select top k by frequency and magnitude
                freqs = selector.selection_frequencies_
                mags = selector.coef_magnitudes_
                ranked_idx = np.lexsort((-mags, -freqs))
                selected_idx = np.sort(ranked_idx[:min(k, len(ranked_idx))])
                X_tr_sel = X_tr_proc[:, selected_idx]
                X_val_sel = X_val_proc[:, selected_idx]
            else:
                selector = StabilitySelector(
                    n_resamples=50,
                    subsample_ratio=0.8,
                    l1_ratio=0.5,
                    C=0.1,
                    frequency_threshold=0.0,
                    max_genes=k,
                    random_state=42 + fold
                )
                selector.fit(X_tr_proc, labels[train_mask])
                X_tr_sel = selector.transform(X_tr_proc)
                X_val_sel = selector.transform(X_val_proc)
            
            model = SSMLPClassifier(random_state=42 + fold)
            model.fit(X_tr_sel, labels[train_mask], X_val=X_val_sel, y_val=labels[val_mask])
            oof_probs[val_mask] = model.predict_proba(X_val_sel)[:, 1]
            
        m = compute_all_metrics(labels, oof_probs)
        m["Max_Genes"] = k
        results.append(m)
        print(f"Top {k} Genes: AUROC={m['AUROC']:.4f}, AUPRC={m['AUPRC']:.4f}, Brier={m['Brier_Score']:.4f}, ECE={m['ECE']:.4f}")
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(ABLATIONS_DIR / "table_e7_gene_count_ablation.csv", index=False)
    return df_res

def run_dropout_ablation():
    """Experiment E7: SS-MLP Dropout regularization ablation."""
    print("=" * 70)
    print("EXPERIMENT E7: DROPOUT RATE ABLATION")
    print("=" * 70)
    
    counts, labels, pids, folds, gene_cols = load_data()
    dropout_configs = [
        ("No Dropout (0.0 / 0.0)", 0.0, 0.0),
        ("Low Dropout (0.15 / 0.10)", 0.15, 0.10),
        ("Proposed Default (0.35 / 0.25)", 0.35, 0.25),
        ("High Dropout (0.50 / 0.40)", 0.50, 0.40)
    ]
    results = []
    
    for name, d1, d2 in dropout_configs:
        print(f"\nEvaluating {name}...")
        oof_probs = np.zeros(len(labels))
        for fold in range(5):
            train_mask = (folds != fold)
            val_mask = (folds == fold)
            
            X_tr_exp = counts[train_mask][:, filter_expressed_genes(counts[train_mask])]
            X_val_exp = counts[val_mask][:, filter_expressed_genes(counts[train_mask])]
            
            X_tr_log = normalize_logcpm(X_tr_exp)
            X_val_log = normalize_logcpm(X_val_exp)
            
            mean_tr = np.mean(X_tr_log, axis=0)
            std_tr = np.std(X_tr_log, axis=0) + 1e-8
            X_tr_proc = (X_tr_log - mean_tr) / std_tr
            X_val_proc = (X_val_log - mean_tr) / std_tr
            
            cache_file = BASE_DIR / "results" / "internal" / f"cached_stability_selector_fold_{fold}_frozen_training_norm.joblib"
            if cache_file.exists():
                selector = joblib.load(cache_file)
                X_tr_sel = selector.transform(X_tr_proc)
                X_val_sel = selector.transform(X_val_proc)
            else:
                selector = StabilitySelector(
                    n_resamples=50,
                    subsample_ratio=0.8,
                    l1_ratio=0.5,
                    C=0.1,
                    frequency_threshold=0.70,
                    max_genes=300,
                    random_state=42 + fold
                )
                selector.fit(X_tr_proc, labels[train_mask])
                X_tr_sel = selector.transform(X_tr_proc)
                X_val_sel = selector.transform(X_val_proc)
            
            model = SSMLPClassifier(dropout1=d1, dropout2=d2, random_state=42 + fold)
            model.fit(X_tr_sel, labels[train_mask], X_val=X_val_sel, y_val=labels[val_mask])
            oof_probs[val_mask] = model.predict_proba(X_val_sel)[:, 1]
            
        m = compute_all_metrics(labels, oof_probs)
        m["Configuration"] = name
        m["Dropout1"] = d1
        m["Dropout2"] = d2
        results.append(m)
        print(f"{name}: AUROC={m['AUROC']:.4f}, AUPRC={m['AUPRC']:.4f}, Brier={m['Brier_Score']:.4f}, ECE={m['ECE']:.4f}")
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(ABLATIONS_DIR / "table_e7_dropout_ablation.csv", index=False)
    return df_res

if __name__ == "__main__":
    run_esr1_exclusion_ablation()
    run_gene_count_ablation()
    run_dropout_ablation()
