"""
Quick test: scan all cached GitTables tables for repurposable features per target.
Tries thresholds 0.3, 0.4, 0.5 to see how many candidates each target finds.
"""
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer
import gittables_lake
import table_discovery

logging.basicConfig(level=logging.WARNING)

TARGETS = {
    "adult":    "income above 50k",
    "nyhouse":  "house price above 1 million dollars",
    "bank":     "term deposit subscription",
    "diabetes": "diabetes diagnosis positive",
    "credit":   "credit risk good or bad",
}

THRESHOLDS = [0.3, 0.4, 0.5]
CACHE_DIR = Path("data/gittables")

print("Loading SentenceTransformer...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"Loading all GitTables tables (no column filter)...")
# Load all tables regardless of column overlap — we want to scan everything
from pathlib import Path
import json, pandas as pd

manifest_path = CACHE_DIR / "manifest.json"
with open(manifest_path) as f:
    manifest = json.load(f)

all_tables: dict[str, pd.DataFrame] = {}
for entry in manifest["tables"]:
    fpath = CACHE_DIR / entry["path"]
    if not fpath.exists():
        continue
    try:
        df = pd.read_parquet(fpath)
        all_tables[entry["table_id"]] = df
    except Exception:
        continue

print(f"Loaded {len(all_tables)} tables\n")

print(f"{'Target':<12}  ", end="")
for t in THRESHOLDS:
    print(f"thr={t:.1f}  ", end="")
print()
print("-" * 50)

for target_name, label_name in TARGETS.items():
    print(f"{target_name:<12}  ", end="", flush=True)
    for threshold in THRESHOLDS:
        candidates = table_discovery.find_repurposable_features(
            lake=all_tables,
            target_label_name=label_name,
            model=model,
            threshold=threshold,
        )
        print(f"{len(candidates):<9}", end="", flush=True)
    print()

print("\nTop repurpose candidates for each target at threshold=0.4:")
for target_name, label_name in TARGETS.items():
    candidates = table_discovery.find_repurposable_features(
        lake=all_tables,
        target_label_name=label_name,
        model=model,
        threshold=0.4,
    )
    print(f"\n  {target_name} ({label_name}):")
    if not candidates:
        print("    (none)")
    else:
        # Show top 10 with the column name
        from scipy.spatial.distance import cdist
        import numpy as np
        target_emb = table_discovery.embed_columns([label_name], model)[0]
        scored = {}
        for tid, col in candidates.items():
            col_emb = table_discovery.embed_columns([col], model)[0]
            sim = float(1.0 - cdist(target_emb.reshape(1,-1), col_emb.reshape(1,-1), metric="cosine")[0,0])
            scored[tid] = (col, sim)
        for tid, (col, sim) in sorted(scored.items(), key=lambda x: -x[1].real)[:10]:
            print(f"    {tid:<40}  col='{col}'  sim={sim:.3f}")
