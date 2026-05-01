"""
WikiTables downloader — streams penfever/wikitables from HuggingFace.

Source: 1.65M Wikipedia tables extracted for the TURL paper, hosted on HuggingFace.
Table format: {tableHeaders: [[{text, ...}]], tableData: [[{text, ...}]], numericColumns: [...]}

Usage:
    python download_wikitables.py [--max-tables 100000] [--cache-dir data/wikitables]

Output: data/wikitables/  directory with parquet files + manifest.json,
        same schema as data/gittables/ so the act5/act6 pipeline works unchanged.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_CACHE    = Path("data/wikitables")
MANIFEST_FILE    = "manifest.json"
CHECKPOINT_EVERY = 5_000  # save manifest every N tables

# Quality filters — Wikipedia tables are small and text-heavy; be permissive here.
# The pipeline's own discovery/alignment will reject tables with insufficient overlap.
MIN_ROWS    = 5
MIN_COLS    = 2
MIN_NUMERIC = 1


def _parse_table(row: dict) -> pd.DataFrame | None:
    """
    Parse a penfever/wikitables row into a DataFrame.

    tableHeaders is a list-of-lists of cell dicts; we take the first header row.
    tableData is a list-of-lists of cell dicts; each inner list is one data row.
    """
    headers_raw = row.get("tableHeaders", [])
    data_raw    = row.get("tableData", [])

    if not headers_raw or not data_raw:
        return None

    # Take first header row
    header_row = headers_raw[0] if isinstance(headers_raw[0], list) else headers_raw
    headers = []
    seen: dict[str, int] = {}
    for cell in header_row:
        text = cell.get("text", "").strip() if isinstance(cell, dict) else str(cell).strip()
        text = text if text else f"col_{len(headers)}"
        if text in seen:
            seen[text] += 1
            text = f"{text}_{seen[text]}"
        else:
            seen[text] = 0
        headers.append(text)

    if len(headers) < MIN_COLS:
        return None

    # Parse data rows
    records = []
    for data_row in data_raw:
        if not isinstance(data_row, list):
            continue
        record = []
        for cell in data_row:
            text = cell.get("text", "").strip() if isinstance(cell, dict) else str(cell).strip()
            record.append(text)
        # Pad/trim to match header length
        if len(record) < len(headers):
            record.extend([""] * (len(headers) - len(record)))
        records.append(record[: len(headers)])

    if len(records) < MIN_ROWS:
        return None

    try:
        df = pd.DataFrame(records, columns=headers)
    except Exception:
        return None

    # Convert to numeric where possible
    for col in df.columns:
        series = df[col].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
        df[col] = pd.to_numeric(series, errors="coerce")

    df = df.dropna(axis=1, how="all")
    df = df.loc[:, df.nunique() > 1]

    n_numeric = df.select_dtypes("number").shape[1]
    if n_numeric < MIN_NUMERIC or df.shape[1] < MIN_COLS:
        return None

    return df.reset_index(drop=True)


def download(cache_dir: Path = DEFAULT_CACHE, max_tables: int = 100_000) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / MANIFEST_FILE

    # Resume from checkpoint
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        existing_ids = {e["table_id"] for e in manifest["tables"]}
        n_saved = len(existing_ids)
        logger.info("Resuming: %d tables already saved.", n_saved)
    else:
        manifest = {"version": 1, "source": "wikitables_hf", "tables": []}
        existing_ids: set[str] = set()
        n_saved = 0

    if n_saved >= max_tables:
        logger.info("Already have %d >= %d tables. Nothing to do.", n_saved, max_tables)
        return

    logger.info("Streaming penfever/wikitables from HuggingFace (%d target tables)…", max_tables)
    ds = load_dataset("penfever/wikitables", split="train", streaming=True)

    n_seen = 0
    n_skipped = 0

    for row in ds:
        if n_saved >= max_tables:
            break

        n_seen += 1
        table_id = f"wt_{row['_id']}"

        if table_id in existing_ids:
            continue

        df = _parse_table(row)
        if df is None:
            n_skipped += 1
            continue

        fname = f"{table_id}.parquet"
        try:
            df.to_parquet(cache_dir / fname, index=False)
        except Exception as e:
            logger.debug("Failed to save %s: %s", table_id, e)
            n_skipped += 1
            continue

        manifest["tables"].append({
            "table_id": table_id,
            "n_rows":   len(df),
            "n_cols":   df.shape[1],
            "columns":  df.columns.tolist(),
            "path":     fname,
            "pg_title": row.get("pgTitle", ""),
        })
        existing_ids.add(table_id)
        n_saved += 1

        if n_saved % CHECKPOINT_EVERY == 0:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)
            logger.info("  %d saved, %d skipped, %d seen total", n_saved, n_skipped, n_seen)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    logger.info(
        "WikiTables download complete: %d tables saved, %d skipped, %d seen.",
        n_saved, n_skipped, n_seen,
    )


def stats(cache_dir: Path = DEFAULT_CACHE) -> None:
    manifest_path = cache_dir / MANIFEST_FILE
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}. Run downloader first.")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    tables = manifest["tables"]
    rows   = [t["n_rows"] for t in tables]
    cols   = [t["n_cols"] for t in tables]
    print(f"WikiTables cache: {len(tables)} tables")
    if tables:
        print(f"  rows: mean={np.mean(rows):.0f}  median={np.median(rows):.0f}  max={max(rows)}")
        print(f"  cols: mean={np.mean(cols):.1f}  median={np.median(cols):.0f}  max={max(cols)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Wikipedia tables from HuggingFace")
    parser.add_argument("--cache-dir",   type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--max-tables",  type=int,  default=100_000)
    parser.add_argument("--stats",       action="store_true")
    args = parser.parse_args()

    if args.stats:
        stats(args.cache_dir)
    else:
        download(args.cache_dir, args.max_tables)


if __name__ == "__main__":
    main()
