import gzip
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
META_DIR = BASE_DIR / "metadata"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_common_genes():
    with open(META_DIR / "common_genes.txt", "r") as f:
        common_genes = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(common_genes)} common genes.")
    return common_genes

def process_tcga_counts(common_genes):
    print("\n" + "=" * 60)
    print("PROCESSING TCGA-BRCA COUNT MATRIX")
    print("=" * 60)
    
    # Load eligible sample audit
    df_audit = pd.read_csv(META_DIR / "tcga_sample_audit.csv")
    df_elig = df_audit[df_audit["eligibility"] == "eligible"].copy()
    print(f"Total eligible TCGA primary tumours: {len(df_elig)}")
    
    # Map file_name / file_id to file path on disk
    file_map = {}
    for p in (RAW_DIR / "tcga" / "files").glob("**/*.tsv"):
        file_map[p.name] = p
        
    print(f"Discovered {len(file_map)} files on disk.")
    
    common_gene_set = set(common_genes)
    gene_indices = {g: i for i, g in enumerate(common_genes)}
    n_samples = len(df_elig)
    n_genes = len(common_genes)
    
    counts_matrix = np.zeros((n_samples, n_genes), dtype=np.int32)
    sample_ids = []
    patient_ids = []
    er_labels = []
    
    for row_idx, (_, row) in enumerate(df_elig.iterrows()):
        fname = row["file_name"]
        sid = row["sample_id"]
        pid = row["patient_id"]
        er = row["er_status"]
        
        sample_ids.append(sid)
        patient_ids.append(pid)
        er_labels.append(1 if er == "Positive" else 0)
        
        fpath = file_map.get(fname)
        if not fpath:
            raise FileNotFoundError(f"Missing file on disk: {fname}")
            
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or line.startswith("gene_id") or line.startswith("N_"):
                    continue
                parts = line.split("\t")
                sym = parts[1]
                if sym in common_gene_set:
                    # unstranded count is at index 3
                    cnt = int(parts[3])
                    col_idx = gene_indices[sym]
                    counts_matrix[row_idx, col_idx] = cnt
                    
        if (row_idx + 1) % 200 == 0 or (row_idx + 1) == n_samples:
            print(f"  Processed {row_idx + 1}/{n_samples} TCGA samples...")
            
    df_counts = pd.DataFrame(counts_matrix, index=sample_ids, columns=common_genes)
    df_counts.insert(0, "patient_id", patient_ids)
    df_counts.insert(1, "er_status", er_labels)
    
    out_path = PROCESSED_DIR / "tcga_counts.parquet"
    df_counts.to_parquet(out_path, engine="pyarrow", compression="snappy")
    print(f"Saved TCGA counts matrix {df_counts.shape} to {out_path}")
    return df_counts

def process_gse96058_expression(common_genes):
    print("\n" + "=" * 60)
    print("PROCESSING GSE96058 EXPRESSION MATRIX")
    print("=" * 60)
    
    # Load manifest and filter for eligible GSE96058 samples
    manifest = pd.read_csv(META_DIR / "sample_manifest.csv")
    gse_elig = manifest[(manifest["cohort"] == "GSE96058") & (manifest["eligibility"] == "eligible")].copy()
    print(f"Eligible GSE96058 unique patient samples: {len(gse_elig)}")
    
    # Target sample titles (e.g. F1, F2...)
    eligible_titles = set(gse_elig["sample_title"].dropna().tolist())
    title_to_pid = dict(zip(gse_elig["sample_title"], gse_elig["patient_id"]))
    title_to_er = dict(zip(gse_elig["sample_title"], gse_elig["er_status"]))
    
    csv_file = RAW_DIR / "GSE96058" / "GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz"
    
    with gzip.open(csv_file, "rt", encoding="utf-8", errors="ignore") as f:
        header_line = f.readline().strip()
        all_cols = [x.strip('"\r\n ') for x in header_line.split(",")]
        
    sample_col_indices = []
    ordered_titles = []
    
    for idx, col in enumerate(all_cols[1:], start=1):
        if col in eligible_titles:
            sample_col_indices.append(idx)
            ordered_titles.append(col)
            
    print(f"Found {len(ordered_titles)} matching eligible columns in expression matrix.")
    
    common_gene_set = set(common_genes)
    gene_to_row = {}
    
    # Read row by row for common genes
    n_samples = len(ordered_titles)
    expr_data = defaultdict(lambda: np.zeros(n_samples, dtype=np.float32))
    
    with gzip.open(csv_file, "rt", encoding="utf-8", errors="ignore") as f:
        f.readline() # Header
        for line_idx, line in enumerate(f):
            # Extract gene symbol before first comma
            comma_idx = line.find(",")
            if comma_idx == -1:
                continue
            sym = line[:comma_idx].strip('"\r\n ')
            if sym in common_gene_set:
                parts = line.strip().split(",")
                # Extract values for eligible samples
                vals = [float(parts[i]) for i in sample_col_indices]
                expr_data[sym] = np.array(vals, dtype=np.float32)
                
            if (line_idx + 1) % 5000 == 0:
                print(f"  Scanned {line_idx + 1} gene rows... (captured {len(expr_data)}/{len(common_genes)})")
                
    print(f"Captured expression for {len(expr_data)} common genes in GSE96058.")
    
    # Assemble DataFrame
    matrix = np.zeros((n_samples, len(common_genes)), dtype=np.float32)
    for j, g in enumerate(common_genes):
        if g in expr_data:
            matrix[:, j] = expr_data[g]
            
    df_expr = pd.DataFrame(matrix, index=ordered_titles, columns=common_genes)
    df_expr.insert(0, "patient_id", [title_to_pid[t] for t in ordered_titles])
    df_expr.insert(1, "er_status", [1 if title_to_er[t] == "Positive" else 0 for t in ordered_titles])
    
    out_path = PROCESSED_DIR / "gse96058_expression.parquet"
    df_expr.to_parquet(out_path, engine="pyarrow", compression="snappy")
    print(f"Saved GSE96058 expression matrix {df_expr.shape} to {out_path}")
    return df_expr

def process_gse42568_expression(common_genes):
    print("\n" + "=" * 60)
    print("PROCESSING GSE42568 MICROARRAY MATRIX (GPL570)")
    print("=" * 60)
    
    manifest = pd.read_csv(META_DIR / "sample_manifest.csv")
    gse_elig = manifest[(manifest["cohort"] == "GSE42568") & (manifest["eligibility"] == "eligible")].copy()
    print(f"Eligible GSE42568 tumour samples: {len(gse_elig)}")
    
    eligible_gsms = set(gse_elig["sample_id"].dropna().tolist())
    gsm_to_er = dict(zip(gse_elig["sample_id"], gse_elig["er_status"]))
    
    # Load probe mapping
    df_mapping = pd.read_csv(META_DIR / "gene_mapping.csv")
    # Build probe-to-gene dictionary
    probe_to_gene = {}
    for _, row in df_mapping.iterrows():
        g = row["gene_symbol"]
        probes = str(row["gpl570_probes"]).split(";")
        for p in probes:
            if p:
                probe_to_gene[p] = g
                
    print(f"Loaded {len(probe_to_gene)} probe mappings to common genes.")
    
    # Parse series matrix table
    matrix_file = RAW_DIR / "GSE42568" / "GSE42568_series_matrix.txt.gz"
    ordered_gsms = []
    gsm_indices = []
    probe_rows = defaultdict(dict)
    
    with gzip.open(matrix_file, "rt", encoding="utf-8", errors="ignore") as f:
        in_table = False
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if not in_table:
                continue
            if line.startswith('"ID_REF"') or line.startswith("ID_REF"):
                all_cols = [x.strip('"\r\n ') for x in line.split("\t")]
                for idx, col in enumerate(all_cols[1:], start=1):
                    if col in eligible_gsms:
                        gsm_indices.append(idx)
                        ordered_gsms.append(col)
                print(f"Matched {len(ordered_gsms)} eligible GSM columns in series matrix.")
                continue
                
            parts = line.strip().split("\t")
            probe_id = parts[0].strip('"\r\n ')
            if probe_id in probe_to_gene:
                gene = probe_to_gene[probe_id]
                vals = [float(parts[i]) for i in gsm_indices]
                if gene not in probe_rows:
                    probe_rows[gene] = []
                probe_rows[gene].append(vals)
                
    print(f"Captured probes for {len(probe_rows)} common genes in GSE42568.")
    
    # Collapse multiple probes per gene using MEDIAN rule
    n_samples = len(ordered_gsms)
    matrix = np.zeros((n_samples, len(common_genes)), dtype=np.float32)
    
    for j, g in enumerate(common_genes):
        if g in probe_rows:
            # Array of shape (n_probes, n_samples)
            arr = np.array(probe_rows[g], dtype=np.float32)
            # Median across probes -> shape (n_samples,)
            matrix[:, j] = np.median(arr, axis=0)
            
    df_expr = pd.DataFrame(matrix, index=ordered_gsms, columns=common_genes)
    df_expr.insert(0, "patient_id", ordered_gsms)
    df_expr.insert(1, "er_status", [1 if gsm_to_er[gsm] == "Positive" else 0 for gsm in ordered_gsms])
    
    out_path = PROCESSED_DIR / "gse42568_expression.parquet"
    df_expr.to_parquet(out_path, engine="pyarrow", compression="snappy")
    print(f"Saved GSE42568 collapsed expression matrix {df_expr.shape} to {out_path}")
    return df_expr

def create_patient_folds(n_splits=5, random_state=42):
    print("\n" + "=" * 60)
    print("CREATING PATIENT-LEVEL 5-FOLD STRATIFIED SPLIT (TCGA-BRCA)")
    print("=" * 60)
    
    df_tcga = pd.read_parquet(PROCESSED_DIR / "tcga_counts.parquet", columns=["patient_id", "er_status"])
    print(f"Total TCGA patients for cross-validation: {len(df_tcga)}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    df_tcga["fold"] = -1
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_tcga, df_tcga["er_status"])):
        df_tcga.iloc[val_idx, df_tcga.columns.get_loc("fold")] = fold
        
    print("Patient Fold distribution:")
    for fold in range(n_splits):
        f_sub = df_tcga[df_tcga["fold"] == fold]
        n_pos = (f_sub["er_status"] == 1).sum()
        n_neg = (f_sub["er_status"] == 0).sum()
        print(f"  Fold {fold}: Total={len(f_sub)}, Positive={n_pos}, Negative={n_neg} ({n_pos/len(f_sub)*100:.1f}% Pos)")
        
    folds_path = META_DIR / "patient_folds.csv"
    df_tcga[["patient_id", "er_status", "fold"]].to_csv(folds_path, index=False)
    print(f"Saved patient-level fold assignments to {folds_path}")
    return df_tcga

if __name__ == "__main__":
    common_genes = load_common_genes()
    process_tcga_counts(common_genes)
    process_gse96058_expression(common_genes)
    process_gse42568_expression(common_genes)
    create_patient_folds(n_splits=5, random_state=42)
