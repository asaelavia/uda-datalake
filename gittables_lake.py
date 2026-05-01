"""
GitTables Lake — unlabeled tabular data from GitHub for domain adaptation.

GitTables (https://arxiv.org/abs/2106.07258) is ~1.7M CSV tables scraped from
GitHub.  Most tables have no classification label, but they provide feature
distributions that can improve domain adaptation when labeled sources are scarce.

Role in the pipeline
--------------------
Unlabeled tables are NOT used for training classifiers — they have no labels.
Instead they are used ONLY in the domain classifier (Level 3) and DANN
(Level 5) to better estimate P(target | x) / P(source | x):

    domain classifier source = labeled OpenML sources  +  unlabeled GitTables
    domain classifier target = the prediction target

Because the domain classifier only needs feature vectors, unlabeled tables add
genuine signal about what "the data lake" looks like without leaking any labels.

Usage
-----
    # Download once (streaming, resumable):
    python gittables_lake.py --download --cache-dir data/gittables --max-tables 5000

    # Check cache stats:
    python gittables_lake.py --stats
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_DATASET_ID    = "target-benchmark/gittables-corpus"
DEFAULT_CACHE    = Path("data/gittables")
MANIFEST_FILE    = "manifest.json"
MIN_ROWS         = 30
MAX_ROWS         = 10_000
MIN_COLS         = 3
MAX_COLS         = 50
MIN_NUMERIC_FRAC = 0.3   # fraction of columns that must be numeric


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _table_to_dataframe(table_rows: list[list]) -> Optional[pd.DataFrame]:
    """
    Convert a GitTables row (list-of-lists, first row = header) to a DataFrame.
    Keeps only numeric columns; returns None if the table is unusable.
    """
    if len(table_rows) < MIN_ROWS + 1:
        return None
    header = [str(c).strip() for c in table_rows[0]]
    if len(header) < MIN_COLS or len(header) > MAX_COLS:
        return None

    # Deduplicate column names (duplicate cols cause df[col] to return a DataFrame)
    seen: dict[str, int] = {}
    deduped: list[str] = []
    for col in header:
        if col in seen:
            seen[col] += 1
            deduped.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            deduped.append(col)
    header = deduped

    data = table_rows[1:]
    if len(data) > MAX_ROWS:
        step = len(data) // MAX_ROWS
        data = data[::step][:MAX_ROWS]

    try:
        df = pd.DataFrame(data, columns=header)
    except Exception:
        return None

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(axis=1, how="all")

    n_numeric = df.select_dtypes("number").shape[1]
    if n_numeric < MIN_COLS or (n_numeric / max(len(df.columns), 1)) < MIN_NUMERIC_FRAC:
        return None

    df = df.loc[:, df.nunique() > 1]   # drop zero-variance columns
    if df.shape[1] < MIN_COLS:
        return None

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Download / cache
# ---------------------------------------------------------------------------

def download_gittables(
    cache_dir: Path = DEFAULT_CACHE,
    max_tables: int = 5_000,
    hf_dataset_id: str = HF_DATASET_ID,
) -> None:
    """
    Stream GitTables from HuggingFace, filter to numeric-heavy tables, and
    cache as parquet files.  Resumable — skips already-cached tables.

    Parameters
    ----------
    cache_dir:
        Directory for parquet files + manifest.json.
    max_tables:
        Stop after caching this many usable tables.
    hf_dataset_id:
        HuggingFace dataset identifier.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "GitTables download requires the 'datasets' library: pip install datasets"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / MANIFEST_FILE

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        existing = {e["table_id"] for e in manifest["tables"]}
        logger.info("Resuming: %d tables already cached.", len(existing))
    else:
        manifest = {"version": 1, "hf_dataset": hf_dataset_id, "tables": []}
        existing: set[str] = set()

    if len(existing) >= max_tables:
        logger.info("Already have %d tables — nothing to download.", len(existing))
        return

    logger.info("Streaming %s (target: %d tables) ...", hf_dataset_id, max_tables)
    ds = load_dataset(hf_dataset_id, split="train", streaming=True)

    n_seen = n_saved = 0
    for row in ds:
        if len(manifest["tables"]) >= max_tables:
            break

        table_id = str(row.get("table_id", f"row_{n_seen}"))
        n_seen += 1

        if table_id in existing:
            continue

        table_rows = row.get("table")
        if not table_rows:
            continue

        df = _table_to_dataframe(table_rows)
        if df is None:
            continue

        fname = f"gt_{table_id.replace('/', '_')}.parquet"
        fpath = cache_dir / fname
        try:
            df.to_parquet(fpath, index=False)
        except Exception as exc:
            logger.debug("Failed to write %s: %s", fname, exc)
            continue

        manifest["tables"].append({
            "table_id": table_id,
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "columns": df.columns.tolist(),
            "path": fname,
        })
        existing.add(table_id)
        n_saved += 1

        if n_saved % 200 == 0:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)
            logger.info("  Saved %d tables (scanned %d) ...", n_saved, n_seen)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    logger.info(
        "Download complete: %d tables saved, %d scanned.", n_saved, n_seen,
    )


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def _align_to_columns(
    df: pd.DataFrame,
    target_cols: list[str],
    min_overlap: int = 2,
) -> Optional[pd.DataFrame]:
    """
    Return df restricted to columns whose names match (case-insensitively) a
    target column, renamed to target spelling.  Returns None if fewer than
    min_overlap columns survive.
    """
    target_lower = {c.lower(): c for c in target_cols}
    rename_map: dict[str, str] = {}
    for col in df.columns:
        key = col.lower()
        if key in target_lower:
            rename_map[col] = target_lower[key]

    if len(rename_map) < min_overlap:
        return None

    sub = df[list(rename_map)].rename(columns=rename_map).copy()
    for c in sub.columns:
        med = float(sub[c].median())
        sub[c] = sub[c].fillna(med if not np.isnan(med) else 0.0)
    return sub


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_gittables_features(
    target_cols: list[str],
    cache_dir: Path = DEFAULT_CACHE,
    max_tables: int = 500,
    min_overlap: int = 2,
) -> dict[str, pd.DataFrame]:
    """
    Load cached GitTables tables aligned to `target_cols`.

    Only columns whose names match (case-insensitively) a target column are
    kept.  Tables with fewer than `min_overlap` matching columns are skipped.

    Parameters
    ----------
    target_cols:
        Column names of the target table (feature columns only, no label).
    cache_dir:
        Directory containing the GitTables parquet cache.
    max_tables:
        Maximum number of tables to return.
    min_overlap:
        Minimum number of matching columns required to include a table.

    Returns
    -------
    dict[table_id -> DataFrame]  — columns are a subset of target_cols,
    all numeric, NaN replaced by column median.
    """
    manifest_path = cache_dir / MANIFEST_FILE
    if not manifest_path.exists():
        logger.warning(
            "GitTables cache not found at %s. "
            "Run: python gittables_lake.py --download",
            cache_dir,
        )
        return {}

    with open(manifest_path) as f:
        manifest = json.load(f)

    target_lower_set = {c.lower() for c in target_cols}
    result: dict[str, pd.DataFrame] = {}

    for entry in manifest["tables"]:
        if len(result) >= max_tables:
            break

        # Fast pre-filter without loading parquet
        entry_cols_lower = {c.lower() for c in entry.get("columns", [])}
        if len(entry_cols_lower & target_lower_set) < min_overlap:
            continue

        fpath = cache_dir / entry["path"]
        if not fpath.exists():
            continue
        try:
            df = pd.read_parquet(fpath)
        except Exception as exc:
            logger.debug("Failed to read %s: %s", fpath, exc)
            continue

        aligned = _align_to_columns(df, target_cols, min_overlap=min_overlap)
        if aligned is not None:
            result[f"gt_{entry['table_id']}"] = aligned

    logger.info(
        "GitTables: %d unlabeled tables with >=%d matching columns loaded.",
        len(result), min_overlap,
    )
    return result


def download_gittables_zenodo(
    cache_dir: Path = DEFAULT_CACHE,
    record_id: str = "6517052",
    max_tables: Optional[int] = None,
) -> None:
    """
    Download the full GitTables 1M corpus from Zenodo (record 6517052).

    The record contains ~1400 zip files, each holding several hundred parquet
    files (one table per file).  Tables are filtered with the same criteria as
    the HuggingFace downloader and written to the same cache format so both
    sources share a single manifest.json.

    Parameters
    ----------
    cache_dir:
        Directory for parquet files + manifest.json (shared with HF cache).
    record_id:
        Zenodo record identifier (default: 6517052 = GitTables 1M parquet).
    max_tables:
        Stop after caching this many usable tables (None = unlimited).
    """
    import io
    import zipfile
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / MANIFEST_FILE

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        existing = {e["table_id"] for e in manifest["tables"]}
        logger.info("Resuming: %d tables already cached.", len(existing))
    else:
        manifest = {"version": 1, "source": f"zenodo:{record_id}", "tables": []}
        existing: set[str] = set()

    # Fetch file list from Zenodo API
    api_url = f"https://zenodo.org/api/records/{record_id}"
    logger.info("Fetching Zenodo record metadata from %s ...", api_url)
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    files = resp.json()["files"]
    zip_files = [f for f in files if f["key"].endswith(".zip")]
    logger.info("Found %d zip files in Zenodo record %s.", len(zip_files), record_id)

    n_saved = n_seen = 0

    for zip_entry in zip_files:
        if max_tables is not None and len(manifest["tables"]) >= max_tables:
            break

        zip_name = zip_entry["key"]
        zip_url = zip_entry["links"]["self"]

        logger.info("Downloading %s ...", zip_name)
        try:
            r = requests.get(zip_url, timeout=120, stream=True)
            r.raise_for_status()
            zip_bytes = io.BytesIO(r.content)
        except Exception as exc:
            logger.warning("Failed to download %s: %s", zip_name, exc)
            continue

        try:
            zf = zipfile.ZipFile(zip_bytes)
        except Exception as exc:
            logger.warning("Failed to open zip %s: %s", zip_name, exc)
            continue

        for parquet_name in zf.namelist():
            if not parquet_name.endswith(".parquet"):
                continue
            if max_tables is not None and len(manifest["tables"]) >= max_tables:
                break

            table_id = f"z_{zip_name[:-4]}_{parquet_name.replace('/', '_').replace('.parquet', '')}"
            n_seen += 1

            if table_id in existing:
                continue

            try:
                parquet_bytes = io.BytesIO(zf.read(parquet_name))
                df = pd.read_parquet(parquet_bytes)
            except Exception as exc:
                logger.debug("Failed to read %s/%s: %s", zip_name, parquet_name, exc)
                continue

            # Apply same quality filters as HF downloader
            if len(df) < MIN_ROWS or len(df) > MAX_ROWS:
                continue
            if df.shape[1] < MIN_COLS or df.shape[1] > MAX_COLS:
                continue

            # Deduplicate column names
            seen_cols: dict[str, int] = {}
            deduped: list[str] = []
            for col in df.columns:
                key = str(col).strip()
                if key in seen_cols:
                    seen_cols[key] += 1
                    deduped.append(f"{key}_{seen_cols[key]}")
                else:
                    seen_cols[key] = 0
                    deduped.append(key)
            df.columns = deduped

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(axis=1, how="all")

            n_numeric = df.select_dtypes("number").shape[1]
            if n_numeric < MIN_COLS or (n_numeric / max(len(df.columns), 1)) < MIN_NUMERIC_FRAC:
                continue

            df = df.loc[:, df.nunique() > 1]
            if df.shape[1] < MIN_COLS:
                continue

            df = df.reset_index(drop=True)

            fname = f"gt_{table_id}.parquet"
            fpath = cache_dir / fname
            try:
                df.to_parquet(fpath, index=False)
            except Exception as exc:
                logger.debug("Failed to write %s: %s", fname, exc)
                continue

            manifest["tables"].append({
                "table_id": table_id,
                "n_rows": len(df),
                "n_cols": df.shape[1],
                "columns": df.columns.tolist(),
                "path": fname,
            })
            existing.add(table_id)
            n_saved += 1

            if n_saved % 500 == 0:
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f)
                logger.info("  Saved %d tables (scanned %d) ...", n_saved, n_seen)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    logger.info("Zenodo download complete: %d tables saved, %d scanned.", n_saved, n_seen)


def stats(cache_dir: Path = DEFAULT_CACHE) -> None:
    """Print a summary of the local GitTables cache."""
    manifest_path = cache_dir / MANIFEST_FILE
    if not manifest_path.exists():
        print("No cache found.")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    tables = manifest["tables"]
    if not tables:
        print("Cache is empty.")
        return
    rows = [e["n_rows"] for e in tables]
    cols = [e["n_cols"] for e in tables]
    total_mb = sum(
        (cache_dir / e["path"]).stat().st_size
        for e in tables
        if (cache_dir / e["path"]).exists()
    ) / 1e6
    print(f"GitTables cache: {len(tables)} tables, {total_mb:.1f} MB")
    print(f"  rows: min={min(rows)}, median={int(np.median(rows))}, max={max(rows)}")
    print(f"  cols: min={min(cols)}, median={int(np.median(cols))}, max={max(cols)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="GitTables lake utility")
    parser.add_argument("--download", action="store_true", help="Download from HuggingFace.")
    parser.add_argument("--download-zenodo", action="store_true", help="Download full 1M corpus from Zenodo.")
    parser.add_argument("--stats", action="store_true", help="Print cache statistics.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--max-tables", type=int, default=None)
    parser.add_argument("--hf-dataset", default=HF_DATASET_ID)
    parser.add_argument("--zenodo-record", default="6517052")
    args = parser.parse_args()

    if args.download:
        download_gittables(args.cache_dir, args.max_tables or 5_000, args.hf_dataset)
    if args.download_zenodo:
        download_gittables_zenodo(args.cache_dir, args.zenodo_record, args.max_tables)
    if args.stats:
        stats(args.cache_dir)
    if not args.download and not args.stats:
        parser.print_help()
