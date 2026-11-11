import json
import pandas as pd
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
TCGA_DIR = BASE_DIR / "data" / "raw" / "tcga"
META_DIR = BASE_DIR / "metadata"
META_DIR.mkdir(parents=True, exist_ok=True)

def audit_tcga():
    print("=" * 60)
    print("AUDITING TCGA-BRCA (Primary Training & Internal-Validation Cohort)")
    print("=" * 60)

    # 1. Load file-case mapping from STAR-counts query
    map_file = TCGA_DIR / "file_case_mapping.json"
    with open(map_file, "r", encoding="utf-8") as f:
        star_files = json.load(f)
    print(f"Total raw primary tumour STAR-counts files: {len(star_files)}")

    df_star = pd.DataFrame(star_files)
    # Filter for Primary Tumor specifically (sample type code '01')
    # Submitter ID pattern: TCGA-XX-XXXX-01...
    df_star["sample_code"] = df_star["sample_submitter_id"].apply(lambda s: s.split("-")[3][:2] if len(s.split("-")) >= 4 else "")
    print("Sample codes in STAR counts files:", Counter(df_star["sample_code"]))

    # 2. Load nationwidechildrens clinical table
    clin_file = TCGA_DIR / "nationwidechildrens.org_clinical_patient_brca.txt"
    # Skip lines 1 and 2 (descriptions and CDE IDs), header is line 0
    df_clin = pd.read_csv(clin_file, sep="\t", skiprows=[1, 2], low_memory=False)
    print(f"Total patients in clinical table: {len(df_clin)}")

    # Check ER status values
    raw_er_counts = df_clin["er_status_by_ihc"].value_counts(dropna=False).to_dict()
    print("Raw clinical er_status_by_ihc counts:", raw_er_counts)

    # Join STAR files with clinical data on patient barcode (case_submitter_id == bcr_patient_barcode)
    merged = df_star.merge(df_clin, left_on="case_submitter_id", right_on="bcr_patient_barcode", how="left")
    print(f"Total merged primary tumour records: {len(merged)}")

    # Audit exclusions
    records = []
    exclusions = []
    patient_seen = set()

    # Track duplicate samples per patient to keep exactly ONE primary tumour per patient
    # Sort merged so that '01A' comes before '01B' etc.
    merged = merged.sort_values(by=["case_submitter_id", "sample_submitter_id"])

    for idx, row in merged.iterrows():
        case_id = row["case_submitter_id"]
        sample_id = row["sample_submitter_id"]
        er_raw = str(row.get("er_status_by_ihc", ""))
        
        status = "eligible"
        reason = "Passed all criteria"
        norm_er = None

        if row["sample_code"] != "01":
            status = "excluded"
            reason = f"Non-primary tumour sample code ({row['sample_code']})"
        elif pd.isna(row.get("bcr_patient_barcode")):
            status = "excluded"
            reason = "No matching clinical patient record found"
        elif er_raw in ["[Not Available]", "[Not Evaluated]", "Indeterminate", "Equivocal", "nan", ""]:
            status = "excluded"
            reason = f"Missing, equivocal or indeterminate ER status ({er_raw})"
        elif er_raw.lower() == "positive":
            norm_er = "Positive"
        elif er_raw.lower() == "negative":
            norm_er = "Negative"
        else:
            status = "excluded"
            reason = f"Unrecognized ER status ({er_raw})"

        # Deduplication: keep only one primary tumour per patient
        if status == "eligible":
            if case_id in patient_seen:
                status = "excluded"
                reason = "Duplicate primary tumour sample for patient (retained primary vial A)"
            else:
                patient_seen.add(case_id)

        rec = {
            "cohort": "TCGA-BRCA",
            "file_id": row["file_id"],
            "file_name": row["file_name"],
            "patient_id": case_id,
            "sample_id": sample_id,
            "sample_code": row["sample_code"],
            "raw_er_status": er_raw,
            "er_status": norm_er,
            "pr_status": row.get("pr_status_by_ihc"),
            "her2_status": row.get("her2_status_by_ihc"),
            "age": row.get("age_at_diagnosis"),
            "stage": row.get("ajcc_pathologic_tumor_stage"),
            "histology": row.get("histological_type"),
            "vital_status": row.get("vital_status"),
            "eligibility": status,
            "exclusion_reason": reason
        }
        records.append(rec)
        if status == "excluded":
            exclusions.append({
                "cohort": "TCGA-BRCA",
                "sample_id": sample_id,
                "patient_id": case_id,
                "reason": reason
            })

    df_manifest = pd.DataFrame(records)
    df_eligible = df_manifest[df_manifest["eligibility"] == "eligible"]
    
    print("\n" + "-" * 50)
    print("TCGA-BRCA AUDIT SUMMARY")
    print("-" * 50)
    print(f"Total STAR-counts files evaluated: {len(df_manifest)}")
    print(f"Total Eligible Unique Patients: {len(df_eligible)}")
    print("ER Status Breakdown (Eligible Primary Tumours):")
    print(df_eligible["er_status"].value_counts())
    print("\nTotal Excluded Samples:", len(exclusions))
    print(Counter(e["reason"] for e in exclusions))

    # Save to metadata directory
    df_manifest.to_csv(META_DIR / "tcga_sample_audit.csv", index=False)
    
    # Check Day 7 Go/No-Go Checkpoint
    print("\n" + "=" * 50)
    print("DAY 7 GO / NO-GO CHECKPOINT EVALUATION (TCGA):")
    n_eligible = len(df_eligible)
    n_pos = (df_eligible["er_status"] == "Positive").sum()
    n_neg = (df_eligible["er_status"] == "Negative").sum()
    
    crit1 = n_eligible >= 500
    crit2 = n_pos >= 100 and n_neg >= 100
    crit3 = len(patient_seen) == n_eligible
    
    print(f"1. At least 500 eligible primary tumours: {'PASS' if crit1 else 'FAIL'} ({n_eligible})")
    print(f"2. At least 100 patients per ER class: {'PASS' if crit2 else 'FAIL'} (Positive: {n_pos}, Negative: {n_neg})")
    print(f"3. Exactly one sample per patient (no duplicate splits): {'PASS' if crit3 else 'FAIL'} ({n_eligible} unique patients)")
    
    if crit1 and crit2 and crit3:
        print("\n>>> TCGA DAY 7 GO/NO-GO CHECKPOINT: PASSED! <<<")
    else:
        print("\n>>> TCGA DAY 7 GO/NO-GO CHECKPOINT: FAILED! <<<")
    print("=" * 50)

    return df_manifest

if __name__ == "__main__":
    audit_tcga()
