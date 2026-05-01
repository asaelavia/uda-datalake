"""
compare_lakes.py — Run Act 5 across multiple data lakes and compare AUC.

Usage:
    python compare_lakes.py --targets adult diabetes churn heart
    python compare_lakes.py --targets adult heart --lakes gittables govdata
    python compare_lakes.py --read-only       # just print existing results

Output:
    results/lake_comparison.csv
    Summary table printed to stdout.

Lake directories:
    data/gittables/     — GitTables (control, already exists)
    data/wikitables/    — Wikipedia tables (download_wikitables.py)
    data/govdata/       — Government open data (download_govdata.py)
    data/openml_lake/   — OpenML curated datasets (download_openml_lake.py)
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Map lake name → cache directory
LAKE_DIRS: dict[str, Path] = {
    "gittables":  Path("data/gittables"),
    "wikitables": Path("data/wikitables"),
    "govdata":    Path("data/govdata"),
    "openml":     Path("data/openml_lake"),
}

MANIFEST_FILE = "manifest.json"
RESULTS_FILE  = Path("results/lake_comparison.csv")

# Methods to include in the summary table
SUMMARY_METHODS = ["baseline", "level0", "level2", "level5", "ensemble", "source_ensemble", "oracle"]


def _lake_available(lake_dir: Path) -> bool:
    return (lake_dir / MANIFEST_FILE).exists()


def _lake_size(lake_dir: Path) -> int:
    try:
        with open(lake_dir / MANIFEST_FILE) as f:
            m = json.load(f)
        return len(m["tables"])
    except Exception:
        return 0


def run_lake(lake_name: str, lake_dir: Path, target_name: str) -> dict | None:
    """Run act5 for one lake × target, return metrics dict or None on failure."""
    # Import here to avoid loading act5 at module level
    import act5_gittables_lake as act5
    try:
        metrics = act5.run_experiment(target_name, lake_dir=lake_dir)
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return None

    if metrics is None:
        return None

    row = {"lake": lake_name, "target": target_name}
    for method in SUMMARY_METHODS:
        if method in metrics.index:
            row[method] = float(metrics.loc[method, "auc"])
    return row


def read_existing_results() -> pd.DataFrame:
    """Load previously saved results from results/lake_comparison.csv."""
    if not RESULTS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(RESULTS_FILE)


def load_act4_rows() -> list[dict]:
    """
    Read pre-computed Act 4 results (OpenML labeled lake) into comparison rows.
    Act 4 uses the OpenML 787-lake with labeled sources — fundamentally different
    from source repurposing but provides an important upper-bound reference.
    """
    act4_dir = Path("results/act4")
    rows = []
    for target_dir in sorted(act4_dir.iterdir()):
        if not target_dir.is_dir():
            continue
        target = target_dir.name
        metrics_path = target_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        try:
            df = pd.read_csv(metrics_path, index_col="level")
        except Exception:
            continue
        row = {"lake": "act4_openml_labeled", "target": target}
        for method in SUMMARY_METHODS:
            if method in df.index:
                row[method] = float(df.loc[method, "auc"])
        rows.append(row)
    return rows


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No results yet.")
        return

    print("\n" + "=" * 70)
    print("LAKE COMPARISON — Best method AUC per lake × target")
    print("=" * 70)

    best_col = [c for c in SUMMARY_METHODS if c != "oracle" and c in df.columns]
    if best_col:
        df = df.copy()
        df["best_uda"] = df[best_col].max(axis=1)

    pivot_cols = ["best_uda", "oracle"] if "oracle" in df.columns else ["best_uda"]
    for col in pivot_cols:
        if col not in df.columns:
            continue
        pivot = df.pivot_table(index="lake", columns="target", values=col, aggfunc="first")
        print(f"\n{col}:")
        print(pivot.to_string(float_format="%.3f"))

    print("\n\nFull method breakdown:")
    for target in df["target"].unique():
        sub = df[df["target"] == target].set_index("lake")
        cols = [c for c in SUMMARY_METHODS if c in sub.columns]
        print(f"\n  Target: {target}")
        print(sub[cols].to_string(float_format="%.3f"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare adaptation across data lakes")
    parser.add_argument(
        "--targets", nargs="+",
        default=["adult", "diabetes", "churn", "heart"],
        help="Target datasets to evaluate",
    )
    parser.add_argument(
        "--lakes", nargs="+",
        default=list(LAKE_DIRS.keys()),
        help="Lake names to include (default: all available)",
    )
    parser.add_argument(
        "--read-only", action="store_true",
        help="Skip running experiments; just print existing results",
    )
    args = parser.parse_args()

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if args.read_only:
        df = read_existing_results()
        # Always inject act4 rows for reference (read-only — not stored in CSV)
        act4_rows = load_act4_rows()
        if act4_rows:
            df = pd.concat([df, pd.DataFrame(act4_rows)], ignore_index=True)
        print_summary(df)
        return

    # Load existing results to support resuming
    existing = read_existing_results()
    existing_keys = (
        set(zip(existing["lake"], existing["target"]))
        if not existing.empty else set()
    )
    rows = existing.to_dict("records") if not existing.empty else []

    for lake_name in args.lakes:
        if lake_name not in LAKE_DIRS:
            print(f"Unknown lake '{lake_name}'. Available: {list(LAKE_DIRS)}")
            continue

        lake_dir = LAKE_DIRS[lake_name]
        if not _lake_available(lake_dir):
            print(f"Skipping lake '{lake_name}': no manifest at {lake_dir}")
            continue

        n_tables = _lake_size(lake_dir)
        print(f"\n{'='*60}")
        print(f"Lake: {lake_name}  ({n_tables} tables at {lake_dir})")
        print(f"{'='*60}")

        for target in args.targets:
            if (lake_name, target) in existing_keys:
                print(f"  {target}: already done — skipping")
                continue

            print(f"  Running {target}…")
            row = run_lake(lake_name, lake_dir, target)
            if row is not None:
                rows.append(row)
            else:
                rows.append({"lake": lake_name, "target": target})

            # Save after each run
            df = pd.DataFrame(rows)
            df.to_csv(RESULTS_FILE, index=False)
            print(f"  Results saved to {RESULTS_FILE}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_FILE, index=False)
    print_summary(df)


if __name__ == "__main__":
    main()
