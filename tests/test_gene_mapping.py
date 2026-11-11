import pytest
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
META_DIR = BASE_DIR / "metadata"

def test_common_genes_exist():
    genes_file = META_DIR / "common_genes.txt"
    assert genes_file.exists(), "common_genes.txt does not exist"
    
    with open(genes_file, "r") as f:
        genes = [line.strip() for line in f if line.strip()]
    assert len(genes) >= 10000, f"Expected >= 10,000 common genes, found {len(genes)}"
    assert "ESR1" in genes, "ESR1 must be present in the common gene space"

def test_gene_mapping_structure():
    mapping_path = META_DIR / "gene_mapping.csv"
    assert mapping_path.exists(), "gene_mapping.csv does not exist"
    
    df = pd.read_csv(mapping_path)
    required_cols = ["gene_symbol", "ensembl_id", "gene_type", "gpl570_probe_count", "gpl570_probes"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col} in gene_mapping.csv"
        
    assert df["gene_symbol"].nunique() == len(df), "Duplicate gene symbols found in gene_mapping.csv"
    
    esr1_row = df[df["gene_symbol"] == "ESR1"]
    assert len(esr1_row) == 1, "ESR1 record missing or ambiguous"
    assert esr1_row.iloc[0]["gpl570_probe_count"] > 0, "ESR1 must have at least one GPL570 probe mapped"
