import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    brier_score_loss
)
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    n_total = len(y_true)
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_size = np.sum(mask)
            ece += (bin_size / n_total) * np.abs(bin_acc - bin_conf)
    return float(ece)

def compute_calibration_slope_intercept(y_true, y_prob):
    """Compute logistic calibration intercept (a) and slope (b).
    Ideal: intercept = 0, slope = 1.
    """
    eps = 1e-6
    clipped_probs = np.clip(y_prob, eps, 1.0 - eps)
    logits = logit(clipped_probs).reshape(-1, 1)
    
    # Fit logistic regression on logits (calibration slope and intercept)
    lr = LogisticRegression(C=1e9, solver="lbfgs")
    try:
        lr.fit(logits, y_true)
        intercept = float(lr.intercept_[0])
        slope = float(lr.coef_[0][0])
    except Exception:
        intercept = np.nan
        slope = np.nan
    return intercept, slope

def compute_all_metrics(y_true, y_prob, threshold=0.5):
    """Compute comprehensive discrimination and calibration metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    
    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    auprc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    brier = float(brier_score_loss(y_true, y_prob))
    cal_intercept, cal_slope = compute_calibration_slope_intercept(y_true, y_prob)
    ece = compute_ece(y_true, y_prob, n_bins=10)
    
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1,
        "MCC": mcc,
        "Brier_Score": brier,
        "Calibration_Intercept": cal_intercept,
        "Calibration_Slope": cal_slope,
        "ECE": ece,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    }

def bootstrap_metric_ci(y_true, y_prob, metric_fn, n_bootstraps=1000, alpha=0.05, seed=42):
    """Compute non-parametric bootstrap confidence interval for a given metric."""
    np.random.seed(seed)
    n = len(y_true)
    boot_vals = []
    
    for _ in range(n_bootstraps):
        idx = np.random.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            val = metric_fn(y_true[idx], y_prob[idx])
            if not np.isnan(val):
                boot_vals.append(val)
        except Exception:
            continue
            
    if len(boot_vals) < 100:
        return np.nan, np.nan
        
    lower = float(np.percentile(boot_vals, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    return lower, upper
