import gzip
import re
from pathlib import Path
from collections import Counter
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
META_DIR = BASE_DIR / "metadata"
META_DIR.mkdir(parents=True, exist_ok=True)

def audit_gse42568():
    print("=" * 60)
    print("AUDITING GSE42568 (Cross-Platform Microarray Cohort)")
    print("=" * 60)
    matrix_file = RAW_DIR / "GSE42568" / "GSE42568_series_matrix.txt.gz"
    if not matrix_file.exists():
        print(f"Error: {matrix_file} not found!")
        return None

    gsm_ids = []
    titles = []
    tissues = []
    er_statuses = []
    age = []
    grade = []
    size = []

    with gzip.open(matrix_file, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("!Sample_title"):
                titles = [x.strip('"\n\r ') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession"):
                gsm_ids = [x.strip('"\n\r ') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                parts = [x.strip('"\n\r ') for x in line.split("\t")[1:]]
                header = parts[0].split(":")[0].strip().lower()
                vals = [x.split(":", 1)[1].strip() if ":" in x else x for x in parts]
                if header == "tissue":
                    tissues = vals
                elif header == "er_status":
                    er_statuses = vals
                elif header == "age":
                    age = vals
                elif header == "grade":
                    grade = vals
                elif header == "size":
                    size = vals

    total_samples = len(gsm_ids)
    print(f"Total raw samples in series matrix: {total_samples}")
    print("Tissue distribution:", Counter(tissues))
    print("Raw ER status distribution:", Counter(er_statuses))

    records = []
    exclusions = []
    for gsm, title, tissue, er, a, g, s in zip(gsm_ids, titles, tissues, er_statuses, age, grade, size):
        status = "eligible"
        reason = "Passed all criteria"
        norm_er = None

        if "normal" in tissue.lower():
            status = "excluded"
            reason = "Normal breast tissue (non-tumour)"
        elif er in ["NA", "unknown", ""]:
            status = "excluded"
            reason = f"Missing or unknown ER status ({er})"
        elif str(er).strip() == "1":
            norm_er = "Positive"
        elif str(er).strip() == "0":
            norm_er = "Negative"
        else:
            status = "excluded"
            reason = f"Equivocal/indeterminate ER status ({er})"

        records.append({
            "cohort": "GSE42568",
            "sample_id": gsm,
            "patient_id": gsm,
            "sample_title": title,
            "tissue": tissue,
            "raw_er_status": er,
            "er_status": norm_er,
            "age": a,
            "grade": g,
            "size": s,
            "eligibility": status,
            "exclusion_reason": reason
        })
        if status == "excluded":
            exclusions.append({
                "cohort": "GSE42568",
                "sample_id": gsm,
                "patient_id": gsm,
                "reason": reason
            })

    df = pd.DataFrame(records)
    df_eligible = df[df["eligibility"] == "eligible"]
    print(f"\nGSE42568 Eligible Tumour Samples: {len(df_eligible)}")
    print("ER Status Breakdown in Eligible Tumours:")
    print(df_eligible["er_status"].value_counts())
    print(f"Total Excluded Samples: {len(exclusions)}")
    print(Counter(e["reason"] for e in exclusions))

    return df, exclusions

def audit_gse96058():
    print("\n" + "=" * 60)
    print("AUDITING GSE96058 (Independent RNA-Seq Cohort)")
    print("=" * 60)
    matrix_files = [
        RAW_DIR / "GSE96058" / "GSE96058-GPL11154_series_matrix.txt.gz",
        RAW_DIR / "GSE96058" / "GSE96058-GPL18573_series_matrix.txt.gz"
    ]

    all_records = []
    for mf in matrix_files:
        if not mf.exists():
            print(f"Warning: {mf} does not exist!")
            continue

        gsm_ids = []
        titles = []
        chars = {}

        with gzip.open(mf, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("!Sample_title"):
                    titles = [x.strip('"\n\r ') for x in line.split("\t")[1:]]
                elif line.startswith("!Sample_geo_accession"):
                    gsm_ids = [x.strip('"\n\r ') for x in line.split("\t")[1:]]
                elif line.startswith("!Sample_characteristics_ch1"):
                    parts = [x.strip('"\n\r ') for x in line.split("\t")[1:]]
                    if ":" in parts[0]:
                        key = parts[0].split(":", 1)[0].strip().lower()
                        vals = [x.split(":", 1)[1].strip() if ":" in x else "" for x in parts]
                        chars[key] = vals

        for i, gsm in enumerate(gsm_ids):
            rec = {
                "cohort": "GSE96058",
                "sample_id": gsm,
                "sample_title": titles[i] if i < len(titles) else "",
            }
            for k, v in chars.items():
                rec[k] = v[i] if i < len(v) else ""
            all_records.append(rec)

    df_raw = pd.DataFrame(all_records)
    print(f"Total raw metadata records parsed across GSE96058 matrix files: {len(df_raw)}")

    df_raw["is_replicate"] = df_raw["sample_title"].str.lower().str.endswith("repl")
    n_replicates = df_raw["is_replicate"].sum()
    n_patients = (~df_raw["is_replicate"]).sum()
    print(f"Technical Replicates identified: {n_replicates}")
    print(f"Unique Patient Samples identified: {n_patients}")

    records = []
    exclusions = []

    for idx, row in df_raw.iterrows():
        gsm = row["sample_id"]
        title = row["sample_title"]
        er = row.get("er status", "")
        is_rep = row["is_replicate"]

        status = "eligible"
        reason = "Passed all criteria"
        norm_er = None

        if is_rep:
            status = "excluded"
            reason = "Technical replicate sample"
        elif er in ["NA", "unknown", "", "nan"] or pd.isna(er):
            status = "excluded"
            reason = f"Missing or unknown ER status ({er})"
        elif str(er).strip() == "1":
            norm_er = "Positive"
        elif str(er).strip() == "0":
            norm_er = "Negative"
        else:
            status = "excluded"
            reason = f"Equivocal/indeterminate ER status ({er})"

        records.append({
            "cohort": "GSE96058",
            "sample_id": gsm,
            "patient_id": title.replace("repl", ""),
            "sample_title": title,
            "tissue": "breast tumour",
            "raw_er_status": er,
            "er_status": norm_er,
            "age": row.get("age at diagnosis"),
            "grade": row.get("nhg"),
            "size": row.get("tumor size"),
            "pam50": row.get("pam50 subtype"),
            "her2_status": row.get("her2 status"),
            "pgr_status": row.get("pgr status"),
            "eligibility": status,
            "exclusion_reason": reason
        })

        if status == "excluded":
            exclusions.append({
                "cohort": "GSE96058",
                "sample_id": gsm,
                "patient_id": title,
                "reason": reason
            })

    df = pd.DataFrame(records)
    df_eligible = df[df["eligibility"] == "eligible"]
    print(f"\nGSE96058 Eligible Tumour Samples: {len(df_eligible)}")
    print("ER Status Breakdown in Eligible GSE96058 Tumours:")
    print(df_eligible["er_status"].value_counts())
    print(f"Total Excluded Samples in GSE96058: {len(exclusions)}")
    print(Counter(e["reason"] for e in exclusions))

    return df, exclusions

def generate_unified_manifests():
    df_42568, excl_42568 = audit_gse42568()
    df_96058, excl_96058 = audit_gse96058()

    tcga_audit_file = META_DIR / "tcga_sample_audit.csv"
    if tcga_audit_file.exists():
        df_tcga = pd.read_csv(tcga_audit_file)
        excl_tcga = [
            {
                "cohort": "TCGA-BRCA",
                "sample_id": r["sample_id"],
                "patient_id": r["patient_id"],
                "reason": r["exclusion_reason"]
            }
            for _, r in df_tcga[df_tcga["eligibility"] == "excluded"].iterrows()
        ]
    else:
        df_tcga = pd.DataFrame()
        excl_tcga = []

    dfs = []
    if not df_tcga.empty:
        dfs.append(df_tcga)
    if not df_42568.empty:
        dfs.append(df_42568)
    if not df_96058.empty:
        dfs.append(df_96058)

    combined_manifest = pd.concat(dfs, ignore_index=True)
    manifest_path = META_DIR / "sample_manifest.csv"
    combined_manifest.to_csv(manifest_path, index=False)
    print(f"\nSaved combined sample manifest ({len(combined_manifest)} total samples) to {manifest_path}")

    all_exclusions = excl_tcga + excl_42568 + excl_96058
    df_excl = pd.DataFrame(all_exclusions)
    excl_path = META_DIR / "exclusion_log.csv"
    df_excl.to_csv(excl_path, index=False)
    print(f"Saved complete exclusion log ({len(df_excl)} excluded samples) to {excl_path}")

    eligible = combined_manifest[combined_manifest["eligibility"] == "eligible"]
    clin_labels_path = META_DIR / "clinical_labels.csv"
    eligible[["cohort", "sample_id", "patient_id", "er_status", "age"]].to_csv(clin_labels_path, index=False)
    print(f"Saved clinical labels for {len(eligible)} eligible samples to {clin_labels_path}")

    print("\n" + "=" * 60)
    print("TABLE 1: DATASET CHARACTERISTICS AND ER CLASS COUNTS")
    print("=" * 60)
    summary = []
    for cohort in ["TCGA-BRCA", "GSE96058", "GSE42568"]:
        c_all = combined_manifest[combined_manifest["cohort"] == cohort]
        c_elig = c_all[c_all["eligibility"] == "eligible"]
        n_pos = (c_elig["er_status"] == "Positive").sum()
        n_neg = (c_elig["er_status"] == "Negative").sum()
        pct_pos = (n_pos / len(c_elig) * 100) if len(c_elig) > 0 else 0
        summary.append({
            "Cohort": cohort,
            "Role": "Primary Training & Internal-CV" if cohort == "TCGA-BRCA" else ("External RNA-seq" if cohort == "GSE96058" else "External Microarray"),
            "Platform": "Illumina HiSeq (STAR Counts)" if cohort == "TCGA-BRCA" else ("Illumina HiSeq (transformed)" if cohort == "GSE96058" else "Affymetrix HG-U133_Plus_2 (GPL570)"),
            "Raw Samples": len(c_all),
            "Excluded": len(c_all) - len(c_elig),
            "Eligible Patients": len(c_elig),
            "ER-Positive": n_pos,
            "ER-Negative": n_neg,
            "ER-Positive (%)": f"{pct_pos:.1f}%"
        })
    df_table1 = pd.DataFrame(summary)
    print(df_table1.to_string(index=False))

    results_dir = BASE_DIR / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    df_table1.to_csv(results_dir / "table1_dataset_characteristics.csv", index=False)
    print(f"\nSaved Table 1 to {results_dir / 'table1_dataset_characteristics.csv'}")

if __name__ == "__main__":
    generate_unified_manifests()
