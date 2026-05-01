"""
OpenML lake downloader — downloads tabular datasets from OpenML as a data lake.

Fetches all active datasets that pass quality filters (size, numeric features).
Excludes the 8 target datasets by name/ID to avoid contamination.

Usage:
    python download_openml_lake.py [--max-tables 5000] [--cache-dir data/openml_lake]

Output: data/openml_lake/ directory with parquet files + manifest.json,
        same schema as data/gittables/ so act5/act6 pipeline works unchanged.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_CACHE    = Path("data/openml_lake")
MANIFEST_FILE    = "manifest.json"
CHECKPOINT_EVERY = 200
OPENML_API       = "https://api.openml.org/api/v1/json"

# Quality filters
MIN_ROWS    = 30
MAX_ROWS    = 100_000
MIN_COLS    = 3
MAX_COLS    = 200
MIN_NUMERIC = 3

# Target dataset IDs to exclude (contamination prevention)
EXCLUDED_IDS = {
    1590,   # adult
    37,     # diabetes
    42178,  # churn (telco)
    31,     # credit (german)
    1461,   # bank marketing
    53,     # heart
    43551,  # turnover
    # common variants
    179,    # adult (variant)
    40536,  # adult (another variant)
}

# Target dataset name substrings to exclude
EXCLUDED_NAME_SUBSTRINGS = [
    "adult", "census", "credit", "creditability", "diabetes", "pima",
    "bank", "marketing", "fraud", "telco", "churn", "heart", "turnover",
]


def _is_excluded(did: int, name: str) -> bool:
    if did in EXCLUDED_IDS:
        return True
    name_lower = name.lower()
    return any(s in name_lower for s in EXCLUDED_NAME_SUBSTRINGS)


def _list_datasets() -> list[dict]:
    """Fetch all active datasets from OpenML API with quality metadata."""
    logger.info("Fetching dataset list from OpenML…")
    all_datasets = []
    offset = 0
    limit  = 1000

    while True:
        url = f"{OPENML_API}/data/list/limit/{limit}/offset/{offset}/status/active"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json().get("data", {}).get("dataset", [])
        except Exception as e:
            logger.warning("OpenML list failed at offset=%d: %s", offset, e)
            break

        if not data:
            break

        all_datasets.extend(data)
        logger.info("  Fetched %d datasets so far…", len(all_datasets))
        if len(data) < limit:
            break
        offset += limit
        time.sleep(0.2)

    logger.info("Total datasets from OpenML: %d", len(all_datasets))
    return all_datasets


def _get_quality(qualities: list[dict]) -> dict:
    """Extract numeric quality values from the nested quality list."""
    result = {}
    for q in qualities:
        try:
            result[q["name"]] = float(q["value"])
        except (KeyError, TypeError, ValueError):
            pass
    return result


def _download_dataset(did: int) -> pd.DataFrame | None:
    """Download an OpenML dataset and return as DataFrame (numeric cols only)."""
    url = f"{OPENML_API}/data/features/{did}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        features = r.json().get("data_features", {}).get("feature", [])
    except Exception as e:
        logger.debug("Failed to get features for did=%d: %s", did, e)
        return None

    # Get the actual data via ARFF download
    data_url = f"https://api.openml.org/data/v1/download/{did}"
    # Better: use the dataset description to find the file URL
    desc_url = f"{OPENML_API}/data/{did}"
    try:
        r = requests.get(desc_url, timeout=30)
        r.raise_for_status()
        dataset_info = r.json().get("data_set_description", {})
        file_id = dataset_info.get("file_id")
        if not file_id:
            return None
    except Exception as e:
        logger.debug("Failed to get description for did=%d: %s", did, e)
        return None

    # Download as CSV via the data features endpoint
    csv_url = f"https://api.openml.org/data/v1/get_csv/{file_id}"
    try:
        r = requests.get(csv_url, timeout=60)
        r.raise_for_status()
        import io
        df = pd.read_csv(io.StringIO(r.text), nrows=MAX_ROWS)
    except Exception as e:
        logger.debug("Failed to download CSV for did=%d (file_id=%s): %s", did, file_id, e)
        return None

    if df is None or len(df) < MIN_ROWS or df.shape[1] < MIN_COLS:
        return None

    # Convert to numeric where possible, drop non-numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, df.nunique() > 1]

    n_numeric = df.select_dtypes("number").shape[1]
    if n_numeric < MIN_NUMERIC or df.shape[1] < MIN_COLS:
        return None

    return df.reset_index(drop=True)


def download(cache_dir: Path = DEFAULT_CACHE, max_tables: int = 5_000) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / MANIFEST_FILE

    # Resume from checkpoint
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        saved_ids = {e["openml_id"] for e in manifest["tables"]}
        n_saved   = len(saved_ids)
        logger.info("Resuming: %d tables already saved.", n_saved)
    else:
        manifest = {"version": 1, "source": "openml", "tables": []}
        saved_ids: set[int] = set()
        n_saved = 0

    if n_saved >= max_tables:
        logger.info("Already have %d >= %d tables. Nothing to do.", n_saved, max_tables)
        return

    all_datasets = _list_datasets()

    # Filter candidates by quality metadata
    candidates = []
    for ds in all_datasets:
        did  = ds.get("did")
        name = ds.get("name", "")
        if did is None or _is_excluded(did, name):
            continue
        if int(did) in saved_ids:
            continue

        q = _get_quality(ds.get("quality", []))
        n_inst  = q.get("NumberOfInstances", 0)
        n_feat  = q.get("NumberOfFeatures", 0)
        n_num   = q.get("NumberOfNumericFeatures", 0)
        if n_inst < MIN_ROWS or n_inst > MAX_ROWS:
            continue
        if n_feat < MIN_COLS or n_feat > MAX_COLS:
            continue
        if n_num < MIN_NUMERIC:
            continue
        candidates.append({"did": int(did), "name": name, "n_rows": n_inst, "n_cols": n_feat})

    logger.info("Candidate datasets after filtering: %d", len(candidates))

    n_failed = 0
    for entry in candidates:
        if n_saved >= max_tables:
            break

        did  = entry["did"]
        name = entry["name"]
        logger.debug("  Downloading did=%d  name=%s", did, name)

        df = _download_dataset(did)
        if df is None:
            n_failed += 1
            time.sleep(0.1)
            continue

        table_id = f"oml_{did}"
        fname    = f"{table_id}.parquet"
        try:
            df.to_parquet(cache_dir / fname, index=False)
        except Exception as e:
            logger.debug("Failed to save did=%d: %s", did, e)
            n_failed += 1
            continue

        manifest["tables"].append({
            "table_id":   table_id,
            "openml_id":  did,
            "openml_name": name,
            "n_rows":     len(df),
            "n_cols":     df.shape[1],
            "columns":    df.columns.tolist(),
            "path":       fname,
        })
        saved_ids.add(did)
        n_saved += 1

        if n_saved % CHECKPOINT_EVERY == 0:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)
            logger.info("  %d saved, %d failed", n_saved, n_failed)

        time.sleep(0.2)  # Rate limiting

    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    logger.info("OpenML lake download complete: %d tables saved, %d failed.", n_saved, n_failed)


def stats(cache_dir: Path = DEFAULT_CACHE) -> None:
    manifest_path = cache_dir / MANIFEST_FILE
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}.")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    tables = manifest["tables"]
    rows   = [t["n_rows"] for t in tables]
    cols   = [t["n_cols"] for t in tables]
    print(f"OpenML lake cache: {len(tables)} tables")
    if tables:
        print(f"  rows: mean={np.mean(rows):.0f}  median={np.median(rows):.0f}  max={max(rows)}")
        print(f"  cols: mean={np.mean(cols):.1f}  median={np.median(cols):.0f}  max={max(cols)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OpenML datasets as a data lake")
    parser.add_argument("--cache-dir",  type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--max-tables", type=int,  default=5_000)
    parser.add_argument("--stats",      action="store_true")
    args = parser.parse_args()

    if args.stats:
        stats(args.cache_dir)
    else:
        download(args.cache_dir, args.max_tables)


if __name__ == "__main__":
    main()
