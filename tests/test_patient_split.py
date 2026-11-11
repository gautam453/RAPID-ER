import pytest
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
META_DIR = BASE_DIR / "metadata"

def test_patient_folds_exist():
    folds_path = META_DIR / "patient_folds.csv"
    assert folds_path.exists(), "patient_folds.csv does not exist"

def test_no_patient_overlap_between_folds():
    folds_path = META_DIR / "patient_folds.csv"
    if not folds_path.exists():
        pytest.skip("patient_folds.csv not yet generated")
    
    df = pd.read_csv(folds_path)
    assert "patient_id" in df.columns
    assert "fold" in df.columns
    assert "er_status" in df.columns
    
    # Assert each patient is unique (exactly one row per patient)
    assert df["patient_id"].nunique() == len(df), "Duplicate patient IDs found in fold assignments!"
    
    # Check fold sets are disjoint
    unique_folds = sorted(df["fold"].unique())
    assert len(unique_folds) == 5, f"Expected 5 folds, found {len(unique_folds)}"
    
    patient_sets = [set(df[df["fold"] == f]["patient_id"]) for f in unique_folds]
    for i in range(len(patient_sets)):
        for j in range(i + 1, len(patient_sets)):
            overlap = patient_sets[i] & patient_sets[j]
            assert len(overlap) == 0, f"Patient leakage between fold {i} and fold {j}: {overlap}"

def test_stratification_preserves_class_ratio():
    folds_path = META_DIR / "patient_folds.csv"
    if not folds_path.exists():
        pytest.skip("patient_folds.csv not yet generated")
        
    df = pd.read_csv(folds_path)
    overall_pos_ratio = (df["er_status"] == 1).mean()
    
    for fold in df["fold"].unique():
        f_sub = df[df["fold"] == fold]
        fold_pos_ratio = (f_sub["er_status"] == 1).mean()
        # Ensure within 3% tolerance of overall ratio
        assert abs(fold_pos_ratio - overall_pos_ratio) < 0.03, (
            f"Fold {fold} ER-positive ratio {fold_pos_ratio:.3f} deviates from overall {overall_pos_ratio:.3f}"
        )
