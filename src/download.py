import os
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
TCGA_DIR = RAW_DIR / "tcga"
TCGA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_tcga_manifest():
    print("Querying GDC API for TCGA-BRCA STAR - Counts Primary Tumor files...")
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
            {"op": "in", "content": {"field": "cases.samples.sample_type", "value": ["Primary Tumor"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}}
        ]
    }
    
    url = (
        "https://api.gdc.cancer.gov/files?filters=" 
        + urllib.parse.quote(json.dumps(filters)) 
        + "&size=3000&fields=file_id,file_name,md5sum,file_size,state,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type"
    )
    
    req = urllib.request.Request(url, headers={"User-Agent": "RAPID-ER-Research/1.0"})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        
    hits = res["data"]["hits"]
    print(f"Total matching TCGA-BRCA files found: {len(hits)}")
    
    manifest_file = TCGA_DIR / "gdc_manifest.txt"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write("id\tfilename\tmd5\tsize\tstate\n")
        for h in hits:
            f.write(f"{h['file_id']}\t{h['file_name']}\t{h['md5sum']}\t{h['file_size']}\t{h['state']}\n")
    print(f"Saved manifest to {manifest_file}")
    
    # Also save file-to-case mapping
    mapping = []
    for h in hits:
        fid = h["file_id"]
        fname = h["file_name"]
        for c in h.get("cases", []):
            case_id = c.get("submitter_id")
            for s in c.get("samples", []):
                sample_id = s.get("submitter_id")
                sample_type = s.get("sample_type")
                mapping.append({
                    "file_id": fid,
                    "file_name": fname,
                    "case_submitter_id": case_id,
                    "sample_submitter_id": sample_id,
                    "sample_type": sample_type
                })
                
    map_file = TCGA_DIR / "file_case_mapping.json"
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved file-case mapping ({len(mapping)} records) to {map_file}")
    return hits

def fetch_tcga_clinical():
    print("Fetching TCGA-BRCA clinical metadata from GDC API...")
    # Fetch clinical data for TCGA-BRCA cases
    filters = {
        "op": "in",
        "content": {"field": "project.project_id", "value": ["TCGA-BRCA"]}
    }
    
    url = (
        "https://api.gdc.cancer.gov/cases?filters="
        + urllib.parse.quote(json.dumps(filters))
        + "&size=2000&expand=demographic,diagnoses,exposures,samples,project"
    )
    
    req = urllib.request.Request(url, headers={"User-Agent": "RAPID-ER-Research/1.0"})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        
    cases = res["data"]["hits"]
    print(f"Total TCGA-BRCA cases fetched: {len(cases)}")
    
    clinical_file = TCGA_DIR / "tcga_cases_clinical.json"
    with open(clinical_file, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)
    print(f"Saved clinical json to {clinical_file}")

if __name__ == "__main__":
    fetch_tcga_manifest()
    fetch_tcga_clinical()
