import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import Parallel, delayed

def filter_expressed_genes(X_counts, min_count=10, min_sample_ratio=0.20):
    """Filter low-expression genes using training partition counts only.
    Retains genes having count >= min_count in >= min_sample_ratio of samples.
    """
    n_samples = X_counts.shape[0]
    expressed_mask = np.sum(X_counts >= min_count, axis=0) >= (n_samples * min_sample_ratio)
    return expressed_mask

def normalize_logcpm(X_counts, scale_factor=1e6):
    """Normalize raw counts to log2(CPM + 1)."""
    lib_sizes = np.sum(X_counts, axis=1, keepdims=True)
    lib_sizes = np.maximum(lib_sizes, 1.0)
    cpm = (X_counts / lib_sizes) * scale_factor
    return np.log2(cpm + 1.0)

def rank_normalize_within_sample(X):
    """Label-free within-sample fractional rank normalization (Strategy B).
    Maps gene expression to [0, 1] relative ranks within each sample independently.
    """
    n_samples, n_genes = X.shape
    ranks = np.argsort(np.argsort(X, axis=1), axis=1).astype(np.float32)
    return ranks / float(n_genes - 1)

def _fit_single_resample(X_sub, y_sub, l1_ratio, C, seed):
    """Fit a single Elastic-Net resample."""
    import warnings
    warnings.filterwarnings("ignore")
    clf = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=l1_ratio,
        C=C,
        class_weight="balanced",
        max_iter=80,
        tol=1e-2,
        random_state=seed
    )
    try:
        clf.fit(X_sub, y_sub)
        coefs = clf.coef_[0]
        non_zero = np.abs(coefs) > 1e-5
        return non_zero, np.abs(coefs)
    except Exception:
        return np.zeros(X_sub.shape[1], dtype=bool), np.zeros(X_sub.shape[1], dtype=np.float32)

class StabilitySelector:
    """Repeated Elastic-Net feature selection on bootstrap/subsample partitions."""
    def __init__(
        self,
        n_resamples=100,
        subsample_ratio=0.8,
        l1_ratio=0.5,
        C=0.1,
        frequency_threshold=0.70,
        max_genes=300,
        n_jobs=-1,
        random_state=42
    ):
        self.n_resamples = n_resamples
        self.subsample_ratio = subsample_ratio
        self.l1_ratio = l1_ratio
        self.C = C
        self.frequency_threshold = frequency_threshold
        self.max_genes = max_genes
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.selection_frequencies_ = None
        self.selected_indices_ = None
        self.coef_magnitudes_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        sss = StratifiedShuffleSplit(
            n_splits=self.n_resamples,
            train_size=self.subsample_ratio,
            random_state=self.random_state
        )

        resample_splits = list(sss.split(X, y))
        seeds = [rng.randint(0, 1000000) for _ in range(self.n_resamples)]

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_resample)(
                X[sub_idx], y[sub_idx], self.l1_ratio, self.C, seeds[b]
            )
            for b, (sub_idx, _) in enumerate(resample_splits)
        )

        selection_counts = np.zeros(n_features, dtype=int)
        coef_sums = np.zeros(n_features, dtype=np.float32)

        for non_zero, coef_mag in results:
            selection_counts[non_zero] += 1
            coef_sums[non_zero] += coef_mag[non_zero]

        self.selection_frequencies_ = selection_counts / float(self.n_resamples)
        self.coef_magnitudes_ = coef_sums / np.maximum(selection_counts, 1)

        # Select genes meeting threshold
        qualified = np.where(self.selection_frequencies_ >= self.frequency_threshold)[0]
        
        if len(qualified) == 0:
            # Fallback: top 50 genes by frequency
            qualified = np.argsort(self.selection_frequencies_)[::-1][:min(50, n_features)]
        elif len(qualified) > self.max_genes:
            sorted_idx = qualified[np.lexsort((-self.coef_magnitudes_[qualified], -self.selection_frequencies_[qualified]))]
            qualified = sorted_idx[:self.max_genes]

        self.selected_indices_ = np.sort(qualified)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
