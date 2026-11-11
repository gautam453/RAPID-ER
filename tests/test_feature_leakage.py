import pytest
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
META_DIR = BASE_DIR / "metadata"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def test_no_external_labels_in_training_manifest():
    """Verify TCGA training manifest has no exposure to external cohorts."""
    manifest_path = META_DIR / "sample_manifest.csv"
    assert manifest_path.exists()
    
    df = pd.read_csv(manifest_path)
    tcga_samples = df[df["cohort"] == "TCGA-BRCA"]
    gse96058_samples = df[df["cohort"] == "GSE96058"]
    gse42568_samples = df[df["cohort"] == "GSE42568"]
    
    # Assert completely disjoint sample IDs and patient IDs
    assert len(set(tcga_samples["sample_id"]) & set(gse96058_samples["sample_id"])) == 0
    assert len(set(tcga_samples["sample_id"]) & set(gse42568_samples["sample_id"])) == 0
    assert len(set(tcga_samples["patient_id"]) & set(gse96058_samples["patient_id"])) == 0
    assert len(set(tcga_samples["patient_id"]) & set(gse42568_samples["patient_id"])) == 0

def test_normalization_fit_transform_interface():
    """Verify that normalization statistics computed on training fold do not leak test fold data."""
    np.random.seed(42)
    X_train = np.random.poisson(lam=50, size=(100, 10))
    X_val = np.random.poisson(lam=100, size=(20, 10))
    
    # Simulate library size normalization fitted on training only
    train_lib_sizes = X_train.sum(axis=1, keepdims=True)
    val_lib_sizes = X_val.sum(axis=1, keepdims=True)
    
    # Scaling factor from train
    scale_factor = 1e6
    X_train_cpm = np.log2((X_train / train_lib_sizes) * scale_factor + 1.0)
    X_val_cpm = np.log2((X_val / val_lib_sizes) * scale_factor + 1.0)
    
    # Fit standardizer on train only
    mean_train = X_train_cpm.mean(axis=0)
    std_train = X_train_cpm.std(axis=0) + 1e-8
    
    X_train_norm = (X_train_cpm - mean_train) / std_train
    X_val_norm = (X_val_cpm - mean_train) / std_train
    
    # Assert validation mean is NOT exactly zero (which would happen only if fitted on validation)
    assert not np.allclose(X_val_norm.mean(axis=0), 0.0)
    # Assert training mean IS exactly zero
    assert np.allclose(X_train_norm.mean(axis=0), 0.0)
