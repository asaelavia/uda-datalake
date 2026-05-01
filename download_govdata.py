"""
Government open data downloader — fetches CSV datasets via the Socrata Discovery API.

Searches across all Socrata-hosted government portals (data.gov, NYC, Chicago, etc.)
using domain-relevant search terms. Filters to tables with sufficient numeric columns.

Usage:
    python download_govdata.py [--max-tables 10000] [--cache-dir data/govdata]

Output: data/govdata/ directory with parquet files + manifest.json,
        same schema as data/gittables/ so act5/act6 pipeline works unchanged.
"""

import argparse
import io
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

DEFAULT_CACHE    = Path("data/govdata")
MANIFEST_FILE    = "manifest.json"
CHECKPOINT_EVERY = 200

SOCRATA_URL = "http://api.us.socrata.com/api/catalog/v1"

# Quality filters
MIN_ROWS       = 30
MAX_ROWS       = 50_000
MIN_COLS       = 3
MIN_NUMERIC    = 2

# Search terms covering all 4 target domains + general numeric gov data
SEARCH_TERMS = [
    # adult / income / census
    "census income demographics population",
    "employment salary wages earnings",
    "household income poverty education",
    "labor force statistics occupation",
    # diabetes / health
    "diabetes health statistics mortality",
    "hospital patient clinical outcomes",
    "public health disease indicators",
    "vital statistics medical records",
    # heart disease
    "cardiovascular heart disease risk",
    "blood pressure cholesterol clinical",
    # churn / customer / telecom
    "customer satisfaction service usage",
    "utility billing consumption accounts",
    # general numeric government data
    "economic indicators quarterly annual",
    "survey statistics demographics",
    "finance budget expenditure revenue",
    "environment pollution measurement",
    "transportation traffic counts",
    "crime statistics incidents",
]


def _fetch_datasets(term: str, limit: int = 50, offset: int = 0) -> list[dict]:
    params = {
        "q":              term,
        "only":           "dataset",
        "limit":          limit,
        "offset":         offset,
    }
    try:
        r = requests.get(SOCRATA_URL, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        logger.debug("Socrata search failed for '%s' offset=%d: %s", term, offset, e)
        return []


def _download_csv(domain: str, dataset_id: str, max_rows: int = MAX_ROWS) -> pd.DataFrame | None:
    url = f"https://{domain}/resource/{dataset_id}.csv?$limit={max_rows}"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "research/1.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        logger.debug("Failed to download %s/%s: %s", domain, dataset_id, e)
        return None

    if len(df) < MIN_ROWS or df.shape[1] < MIN_COLS:
        return None

    # Convert to numeric, drop non-numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, df.nunique() > 1]

    if df.select_dtypes("number").shape[1] < MIN_NUMERIC or df.shape[1] < MIN_COLS:
        return None

    # Trim to max rows
    if len(df) > MAX_ROWS:
        df = df.iloc[:MAX_ROWS]

    return df.reset_index(drop=True)


def download(cache_dir: Path = DEFAULT_CACHE, max_tables: int = 10_000) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / MANIFEST_FILE

    # Resume from checkpoint
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        saved_ids = {e["table_id"] for e in manifest["tables"]}
        n_saved   = len(saved_ids)
        logger.info("Resuming: %d tables already saved.", n_saved)
    else:
        manifest = {"version": 1, "source": "socrata_govdata", "tables": []}
        saved_ids: set[str] = set()
        n_saved = 0

    if n_saved >= max_tables:
        logger.info("Already have %d >= %d tables. Nothing to do.", n_saved, max_tables)
        return

    n_failed  = 0
    n_skipped = 0

    for term in SEARCH_TERMS:
        if n_saved >= max_tables:
            break

        logger.info("Searching: '%s'", term)
        offset = 0

        while offset < 1000 and n_saved < max_tables:
            results = _fetch_datasets(term, limit=50, offset=offset)
            if not results:
                break

            for result in results:
                if n_saved >= max_tables:
                    break

                resource = result.get("resource", {})
                domain   = result.get("metadata", {}).get("domain", "")
                ds_id    = resource.get("id", "")

                if not domain or not ds_id:
                    n_skipped += 1
                    continue

                table_id = f"gov_{domain.replace('.', '_')}_{ds_id}"
                if table_id in saved_ids:
                    n_skipped += 1
                    continue

                # Pre-filter: only datasets with some numeric columns in metadata
                col_types = resource.get("columns_datatype", [])
                n_numeric_meta = sum(
                    1 for t in col_types
                    if isinstance(t, str) and t.lower() in {"number", "integer", "double", "float", "money"}
                )
                if col_types and n_numeric_meta < MIN_NUMERIC:
                    n_skipped += 1
                    continue

                df = _download_csv(domain, ds_id)
                if df is None:
                    n_failed += 1
                    time.sleep(0.2)
                    continue

                fname = f"{table_id}.parquet"
                try:
                    df.to_parquet(cache_dir / fname, index=False)
                except Exception as e:
                    logger.debug("Save failed for %s: %s", table_id, e)
                    n_failed += 1
                    continue

                manifest["tables"].append({
                    "table_id": table_id,
                    "domain":   domain,
                    "ds_id":    ds_id,
                    "name":     resource.get("name", ""),
                    "n_rows":   len(df),
                    "n_cols":   df.shape[1],
                    "columns":  df.columns.tolist(),
                    "path":     fname,
                })
                saved_ids.add(table_id)
                n_saved += 1

                if n_saved % CHECKPOINT_EVERY == 0:
                    with open(manifest_path, "w") as f:
                        json.dump(manifest, f)
                    logger.info(
                        "  %d saved, %d failed, %d skipped", n_saved, n_failed, n_skipped
                    )

                time.sleep(0.3)  # Rate limiting

            offset += 50
            time.sleep(0.5)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    logger.info(
        "Gov data download complete: %d tables saved, %d failed, %d skipped.",
        n_saved, n_failed, n_skipped,
    )


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
    domains = list({t["domain"] for t in tables})
    print(f"Gov data cache: {len(tables)} tables from {len(domains)} domains")
    if tables:
        print(f"  rows: mean={np.mean(rows):.0f}  median={np.median(rows):.0f}  max={max(rows)}")
        print(f"  cols: mean={np.mean(cols):.1f}  median={np.median(cols):.0f}  max={max(cols)}")
        print(f"  top domains: {sorted(domains)[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download government open data via Socrata API")
    parser.add_argument("--cache-dir",  type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--max-tables", type=int,  default=10_000)
    parser.add_argument("--stats",      action="store_true")
    args = parser.parse_args()

    if args.stats:
        stats(args.cache_dir)
    else:
        download(args.cache_dir, args.max_tables)


if __name__ == "__main__":
    main()
