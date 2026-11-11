from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

from src.statistics import (
    compute_ece,
    compute_calibration_slope_intercept
)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
INTERNAL_DIR = RESULTS_DIR / "internal"
EXTERNAL_DIR = RESULTS_DIR / "external"
TABLES_DIR = RESULTS_DIR / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

def compute_calibration_curve_data(y_true, y_prob, n_bins=10):
    """Compute empirical vs predicted probability across uniform bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    records = []
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(np.sum(mask))
        if count > 0:
            obs = float(np.mean(y_true[mask]))
            pred = float(np.mean(y_prob[mask]))
        else:
            obs = np.nan
            pred = (bins[b] + bins[b+1]) / 2.0
        records.append({
            "bin": b,
            "bin_lower": bins[b],
            "bin_upper": bins[b+1],
            "bin_center": (bins[b] + bins[b+1]) / 2.0,
            "mean_pred": pred,
            "observed_prob": obs,
            "sample_count": count
        })
    return pd.DataFrame(records)

def compute_brier_decomposition(y_true, y_prob, n_bins=10):
    """Decompose Brier score into Reliability, Resolution, and Uncertainty:
    BS = REL - RES + UNC
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    N = len(y_true)
    c_base = np.mean(y_true)
    
    unc = c_base * (1.0 - c_base)
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    rel = 0.0
    res = 0.0
    
    for b in range(n_bins):
        mask = bin_indices == b
        nk = np.sum(mask)
        if nk > 0:
            ok = np.mean(y_true[mask])
            pk = np.mean(y_prob[mask])
            rel += (nk / N) * ((pk - ok) ** 2)
            res += (nk / N) * ((ok - c_base) ** 2)
            
    brier_total = float(np.mean((y_prob - y_true) ** 2))
    return {
        "Brier_Score": brier_total,
        "Reliability": float(rel),
        "Resolution": float(res),
        "Uncertainty": float(unc)
    }

def analyze_uncertainty_coverage(y_true, y_prob, thresholds=None):
    """Experiment E9: Uncertainty and coverage-error analysis.
    Evaluates accuracy and error rate among accepted predictions
    where confidence max(p, 1-p) >= tau.
    """
    if thresholds is None:
        thresholds = np.linspace(0.50, 0.99, 50)
        
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    n_total = len(y_true)
    
    records = []
    for tau in thresholds:
        accepted_mask = confidence >= tau
        n_acc = int(np.sum(accepted_mask))
        n_def = n_total - n_acc
        coverage = float(n_acc / n_total)
        
        if n_acc > 0:
            y_true_acc = y_true[accepted_mask]
            y_pred_acc = (y_prob[accepted_mask] >= 0.5).astype(int)
            acc = float(accuracy_score(y_true_acc, y_pred_acc))
            error = 1.0 - acc
            
            # Sensitivity & specificity if both classes present
            if len(np.unique(y_true_acc)) > 1:
                tn, fp, fn, tp = confusion_matrix(y_true_acc, y_pred_acc, labels=[0, 1]).ravel()
                sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            else:
                sens = np.nan
                spec = np.nan
        else:
            acc = np.nan
            error = np.nan
            sens = np.nan
            spec = np.nan
            
        records.append({
            "confidence_threshold": round(float(tau), 4),
            "coverage": coverage,
            "n_accepted": n_acc,
            "n_deferred": n_def,
            "accuracy": acc,
            "error_rate": error,
            "sensitivity": sens,
            "specificity": spec
        })
    return pd.DataFrame(records)

def evaluate_all_calibration():
    """Evaluate calibration and uncertainty for all available models and cohorts."""
    print("=" * 70)
    print("EVALUATING CALIBRATION AND UNCERTAINTY (EXPERIMENTS E8 & E9)")
    print("=" * 70)
    
    calibration_records = []
    
    # 1. Internal TCGA OOF files
    internal_files = list(INTERNAL_DIR.glob("oof_predictions_*.csv"))
    for f in internal_files:
        name_parts = f.stem.replace("oof_predictions_", "").split("_")
        strategy = "frozen_training_norm" if "frozen_training_norm" in f.stem else "rank_norm"
        model_name = f.stem.replace("oof_predictions_", "").replace(f"_{strategy}", "")
        
        df = pd.read_csv(f)
        y_true = df["true_label"].values
        y_prob = df["predicted_prob"].values
        
        ece = compute_ece(y_true, y_prob)
        intercept, slope = compute_calibration_slope_intercept(y_true, y_prob)
        decomp = compute_brier_decomposition(y_true, y_prob)
        
        calibration_records.append({
            "Cohort": "TCGA-BRCA (Internal OOF)",
            "Model": model_name,
            "Strategy": strategy,
            "Brier_Score": decomp["Brier_Score"],
            "Reliability": decomp["Reliability"],
            "Resolution": decomp["Resolution"],
            "Uncertainty": decomp["Uncertainty"],
            "Calibration_Slope": slope,
            "Calibration_Intercept": intercept,
            "ECE": ece
        })
        
        # Coverage analysis
        df_cov = analyze_uncertainty_coverage(y_true, y_prob)
        cov_out = TABLES_DIR / f"table_uncertainty_coverage_TCGA_{model_name}_{strategy}.csv"
        df_cov.to_csv(cov_out, index=False)
        
    # 2. External predictions files
    external_files = list(EXTERNAL_DIR.glob("predictions_*.csv"))
    for f in external_files:
        # Format: predictions_{cohort}_{model}_{strategy}.csv
        parts = f.stem.replace("predictions_", "").split("_")
        cohort = parts[0]
        strategy = "frozen_training_norm" if "frozen_training_norm" in f.stem else "rank_norm"
        model_name = f.stem.replace(f"predictions_{cohort}_", "").replace(f"_{strategy}", "")
        
        df = pd.read_csv(f)
        y_true = df["true_label"].values
        y_prob = df["predicted_prob"].values
        
        ece = compute_ece(y_true, y_prob)
        intercept, slope = compute_calibration_slope_intercept(y_true, y_prob)
        decomp = compute_brier_decomposition(y_true, y_prob)
        
        calibration_records.append({
            "Cohort": cohort,
            "Model": model_name,
            "Strategy": strategy,
            "Brier_Score": decomp["Brier_Score"],
            "Reliability": decomp["Reliability"],
            "Resolution": decomp["Resolution"],
            "Uncertainty": decomp["Uncertainty"],
            "Calibration_Slope": slope,
            "Calibration_Intercept": intercept,
            "ECE": ece
        })
        
        # Coverage analysis
        df_cov = analyze_uncertainty_coverage(y_true, y_prob)
        cov_out = TABLES_DIR / f"table_uncertainty_coverage_{cohort}_{model_name}_{strategy}.csv"
        df_cov.to_csv(cov_out, index=False)
        
    df_cal = pd.DataFrame(calibration_records)
    cal_table_path = TABLES_DIR / "table4_calibration_and_uncertainty_summary.csv"
    df_cal.to_csv(cal_table_path, index=False)
    print(f"\nSaved Table 4 Calibration Summary to {cal_table_path}")
    print(df_cal.to_string(index=False))
    return df_cal

if __name__ == "__main__":
    evaluate_all_calibration()
