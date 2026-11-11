import gzip
from pathlib import Path
from collections import defaultdict
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
META_DIR = BASE_DIR / "metadata"
META_DIR.mkdir(parents=True, exist_ok=True)

def parse_tcga_genes():
    print("Parsing TCGA-BRCA gene definitions from STAR counts...")
    files = list((RAW_DIR / "tcga" / "files").glob("**/*.tsv"))
    if not files:
        raise FileNotFoundError("No TCGA STAR counts TSV files found!")
    
    sample_file = files[0]
    df = pd.read_csv(sample_file, sep="\t", skiprows=1, comment="#")
    df = df[~df["gene_id"].str.startswith("N_")]
    df["ensembl_id"] = df["gene_id"].apply(lambda x: x.split(".")[0])
    df = df.rename(columns={"gene_name": "gene_symbol"})
    df = df[df["gene_symbol"].notna() & (df["gene_symbol"] != "")]
    print(f"Total TCGA gene records: {len(df)}")
    print(f"Unique TCGA gene symbols: {df['gene_symbol'].nunique()}")
    return df[["gene_id", "ensembl_id", "gene_symbol", "gene_type"]]

def parse_gse96058_genes():
    print("Parsing GSE96058 gene identifiers...")
    csv_file = RAW_DIR / "GSE96058" / "GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz"
    genes = []
    with gzip.open(csv_file, "rt", encoding="utf-8", errors="ignore") as f:
        f.readline() # Header
        for line in f:
            if line.strip():
                g = line.split(",", 1)[0].strip('"\n\r ')
                genes.append(g)
    genes_set = set(genes)
    print(f"Total GSE96058 gene rows: {len(genes)}")
    print(f"Unique GSE96058 gene symbols: {len(genes_set)}")
    return genes_set

def parse_gpl570_probes():
    print("Parsing GPL570 probe annotations (Affymetrix HG-U133 Plus 2.0)...")
    annot_file = RAW_DIR / "GSE42568" / "GPL570.annot.gz"
    probe_map = defaultdict(list)
    total_unambiguous = 0
    with gzip.open(annot_file, "rt", encoding="utf-8", errors="ignore") as f:
        in_table = False
        header = None
        for line in f:
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if not in_table:
                continue
            if header is None:
                header = [x.strip() for x in line.split("\t")]
                id_idx = header.index("ID")
                sym_idx = header.index("Gene symbol")
                continue
            
            parts = line.strip().split("\t")
            if len(parts) <= max(id_idx, sym_idx):
                continue
            
            probe_id = parts[id_idx].strip()
            sym = parts[sym_idx].strip()
            
            if not sym or sym == "---" or "///" in sym:
                continue
            
            probe_map[sym].append(probe_id)
            total_unambiguous += 1
            
    print(f"Total unambiguous GPL570 probe mappings: {total_unambiguous}")
    print(f"Unique GPL570 gene symbols: {len(probe_map)}")
    return probe_map

def build_common_gene_space():
    df_tcga = parse_tcga_genes()
    tcga_symbols = set(df_tcga["gene_symbol"].unique())
    gse96058_symbols = parse_gse96058_genes()
    gpl570_map = parse_gpl570_probes()
    gpl570_symbols = set(gpl570_map.keys())
    
    # Intersect across all three cohorts
    common_symbols = sorted(list(tcga_symbols & gse96058_symbols & gpl570_symbols))
    print("\n" + "=" * 60)
    print(f"COMMON GENE SPACE INTERSECTION: {len(common_symbols)} genes")
    print("=" * 60)
    
    if "ESR1" in common_symbols:
        print(">>> ESR1 is PRESENT in the common gene space! <<<")
    else:
        print(">>> WARNING: ESR1 is MISSING in common gene space! <<<")

    # Build detailed mapping table
    df_tcga_sorted = df_tcga.sort_values(by=["gene_symbol", "gene_type"], ascending=[True, False])
    df_tcga_repr = df_tcga_sorted.drop_duplicates(subset=["gene_symbol"]).set_index("gene_symbol")
    
    mapping_records = []
    for g in common_symbols:
        tcga_row = df_tcga_repr.loc[g] if g in df_tcga_repr.index else None
        ens_id = tcga_row["ensembl_id"] if tcga_row is not None else ""
        gtype = tcga_row["gene_type"] if tcga_row is not None else ""
        probes = gpl570_map.get(g, [])
        mapping_records.append({
            "gene_symbol": g,
            "ensembl_id": ens_id,
            "gene_type": gtype,
            "gpl570_probe_count": len(probes),
            "gpl570_probes": ";".join(probes)
        })
        
    df_mapping = pd.DataFrame(mapping_records)
    mapping_path = META_DIR / "gene_mapping.csv"
    df_mapping.to_csv(mapping_path, index=False)
    print(f"Saved gene mapping table ({len(df_mapping)} genes) to {mapping_path}")
    
    genes_txt = META_DIR / "common_genes.txt"
    with open(genes_txt, "w") as f:
        for g in common_symbols:
            f.write(g + "\n")
    print(f"Saved common gene symbols to {genes_txt}")
    
    return common_symbols

if __name__ == "__main__":
    build_common_gene_space()
