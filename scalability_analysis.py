"""
Scalability analysis — "how much lake is enough?"

Reads results/act5/{target}_s{N}/metrics.csv for each (target, N) pair and
produces:
  results/scalability/scalability_table.csv      — full per-(target,size,level) table
  results/scalability/auc_vs_lakesize.png        — best UDA AUC per target
  results/scalability/f1_vs_lakesize.png         — best UDA F1 per target
  results/scalability/accuracy_vs_lakesize.png   — best UDA accuracy per target
  results/scalability/gap_closed_vs_lakesize.png — oracle gap closed (AUC) per target
  results/scalability/sources_vs_lakesize.png    — n_sources vs lake size

Usage
-----
    python scalability_analysis.py
    python scalability_analysis.py --targets adult heart diabetes turnover bank
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_BASE  = Path("results/act5")
OUTPUT_DIR    = Path("results/scalability")
RUNTIME_FILE  = Path("results/scalability/runtimes.json")

# Threshold above which a runtime is considered a bad backfill artifact (seconds)
_RUNTIME_OUTLIER_THRESH = 50_000

# Full-lake runs that loaded from done-caches only timed the adaptation step
# (seconds). Entries below this minimum are dropped as adaptation-only timings.
_FULL_LAKE_MIN_SECS = 300  # < 5 min cannot be a full 421K repurposing scan

# Sample sizes to look for (421_179 = full lake)
SAMPLE_SIZES = [5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 421_179]

_SKIP_LEVELS = {"oracle", "baseline_a", "baseline_b", "llm_zero_shot", "baseline"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_metrics(metrics_path: Path) -> pd.DataFrame:
    return pd.read_csv(metrics_path, index_col=0)


def _best_uda(metrics_path: Path, metric: str) -> tuple[str, float]:
    """Return (best_level, best_value) for `metric`, excluding oracle/baselines."""
    df = _read_metrics(metrics_path)
    valid = df[~df.index.isin(_SKIP_LEVELS)]
    if valid.empty or metric not in valid.columns:
        return ("missing", float("nan"))
    idx = valid[metric].idxmax()
    return str(idx), float(valid.loc[idx, metric])


def _get_metric(metrics_path: Path, level: str, metric: str) -> float:
    df = _read_metrics(metrics_path)
    for row in [level, level.replace("_", "")]:
        if row in df.index and metric in df.columns:
            return float(df.loc[row, metric])
    return float("nan")


def _n_sources(result_dir: Path) -> float:
    """Return number of repurposed sources found, from discovery_scores.csv.

    transferability_fast.csv n_sources is wrong for subsampled runs — it uses
    the full-lake column index regardless of lake_sample, so it's always the
    full-lake count.  discovery_scores.csv is written from the actual labeled
    lake produced by the subsampled scan and gives the correct per-size count.
    """
    disc_p = result_dir / "discovery_scores.csv"
    if disc_p.exists():
        try:
            return float(len(pd.read_csv(disc_p)))  # one row per source
        except Exception:
            pass
    return float("nan")


def _oracle_gap_closed(best_uda: float, baseline: float, oracle: float) -> float:
    gap = oracle - baseline
    if np.isnan(best_uda) or np.isnan(baseline) or np.isnan(oracle) or gap < 1e-6:
        return float("nan")
    return (best_uda - baseline) / gap


# ---------------------------------------------------------------------------
# Main table builder
# ---------------------------------------------------------------------------

def compile_table(targets: list[str]) -> pd.DataFrame:
    rows = []

    for target in targets:
        # Sizes to check: subsampled + full lake (no suffix)
        size_dirs: list[tuple[int, Path]] = []
        for n in SAMPLE_SIZES:
            if n == 421_179:
                size_dirs.append((n, RESULTS_BASE / target))
            else:
                size_dirs.append((n, RESULTS_BASE / f"{target}_s{n}"))

        for lake_size, result_dir in size_dirs:
            metrics_p = result_dir / "metrics.csv"
            if not metrics_p.exists():
                continue

            df_m = _read_metrics(metrics_p)
            baseline_auc = _get_metric(metrics_p, "baseline", "auc")
            if np.isnan(baseline_auc):
                baseline_auc = _get_metric(metrics_p, "baseline_b", "auc")
            oracle_auc  = _get_metric(metrics_p, "oracle", "auc")

            for metric in ["auc", "f1", "accuracy"]:
                if metric not in df_m.columns:
                    continue
                best_level, best_val = _best_uda(metrics_p, metric)

                row = {
                    "target":      target,
                    "lake_size":   lake_size,
                    "metric":      metric,
                    "best_level":  best_level,
                    "best_uda":    best_val,
                    "baseline":    _get_metric(metrics_p, "baseline", metric)
                                   if np.isnan(_get_metric(metrics_p, "baseline", metric))
                                   else _get_metric(metrics_p, "baseline", metric),
                    "oracle":      _get_metric(metrics_p, "oracle", metric),
                    "n_sources":   _n_sources(result_dir),
                }
                # Oracle gap closed only meaningful for AUC
                if metric == "auc":
                    row["oracle_gap_closed"] = _oracle_gap_closed(
                        best_val, baseline_auc, oracle_auc
                    )
                else:
                    row["oracle_gap_closed"] = float("nan")

                rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["target", "metric", "lake_size"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _line_plot(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    path: Path,
    hline_col: str | None = None,
    hline_label: str | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("matplotlib not installed — skipping %s", path)
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = cm.tab10.colors
    targets = df["target"].unique()

    for i, target in enumerate(sorted(targets)):
        grp = df[df["target"] == target].sort_values("lake_size").dropna(subset=["lake_size", y_col])
        if len(grp) < 1:
            continue
        color = colors[i % len(colors)]
        ax.plot(grp["lake_size"] / 1_000, grp[y_col], marker="o",
                label=target, color=color, linewidth=1.8)

        # Horizontal reference line (e.g. oracle or baseline)
        if hline_col and hline_col in grp.columns:
            ref = grp[hline_col].dropna()
            if len(ref):
                ax.axhline(ref.iloc[-1], linestyle=":", alpha=0.5, color=color)

    ax.set_xlabel("Lake size (K tables)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def _load_runtimes(targets: list[str]) -> pd.DataFrame:
    """Load runtimes.json into a DataFrame with columns [target, lake_size, runtime_min].

    Full lake (421179) entries are excluded: those runs loaded from done-caches
    and only timed the adaptation step, not the repurposing scan. They must be
    re-run with explicit timing to get accurate numbers.

    Entries exceeding _RUNTIME_OUTLIER_THRESH seconds are dropped as bad backfill artifacts.
    """
    if not RUNTIME_FILE.exists():
        return pd.DataFrame(columns=["target", "lake_size", "runtime_min"])
    try:
        raw: dict = json.loads(RUNTIME_FILE.read_text())
    except Exception:
        return pd.DataFrame(columns=["target", "lake_size", "runtime_min"])

    rows = []
    for key, secs in raw.items():
        parts = key.rsplit("_", 1)
        if len(parts) != 2:
            continue
        t, n_str = parts
        if t not in targets:
            continue
        try:
            n = int(n_str)
        except ValueError:
            continue
        # Full lake (421179): only keep entries that are plausibly a real scan
        # (>= 5 min). Shorter entries are adaptation-only timings from runs that
        # loaded sources from done-caches rather than running the full scan.
        if n == 421_179 and secs < _FULL_LAKE_MIN_SECS:
            logger.warning("Dropping full-lake runtime %s=%.0fs (adaptation-only, no scan)", key, secs)
            continue
        if secs > _RUNTIME_OUTLIER_THRESH:
            logger.warning("Dropping outlier runtime %s=%.0fs (likely bad backfill)", key, secs)
            continue
        rows.append({"target": t, "lake_size": n, "runtime_min": secs / 60.0})

    return pd.DataFrame(rows)


def _runtime_plot(rt_df: pd.DataFrame, path: Path) -> None:
    """Line plot: runtime (minutes) vs lake size per target."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("matplotlib not installed — skipping %s", path)
        return

    if rt_df.empty:
        logger.warning("No runtime data — skipping %s", path)
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = cm.tab10.colors
    targets = sorted(rt_df["target"].unique())

    for i, target in enumerate(targets):
        grp = rt_df[rt_df["target"] == target].sort_values("lake_size")
        if grp.empty:
            continue
        color = colors[i % len(colors)]
        ax.plot(grp["lake_size"] / 1_000, grp["runtime_min"], marker="o",
                label=target, color=color, linewidth=1.8)

    ax.set_xlabel("Lake size (K tables)")
    ax.set_ylabel("Wall-clock runtime (minutes)")
    ax.set_title("Experiment runtime vs. lake size")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def _pivot_print(df: pd.DataFrame, metric: str) -> None:
    sub = df[df["metric"] == metric][["target", "lake_size", "best_uda", "baseline", "oracle"]].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(index="target", columns="lake_size", values="best_uda")
    print(f"\n=== Best UDA {metric.upper()} by target × lake size ===")
    print(pivot.round(3).to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+",
                        default=["adult", "heart", "diabetes", "turnover", "bank"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = compile_table(args.targets)
    if df.empty:
        logger.error(
            "No scalability results found. Run:\n"
            "  python run_scalability.py\n"
            "or manually:\n"
            "  python act5_gittables_lake.py --target adult --lake-sample 10000\n"
            "  python act5_gittables_lake.py --target adult --lake-sample 50000\n"
            "  ... etc."
        )
        return

    out_csv = OUTPUT_DIR / "scalability_table.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Saved: %s", out_csv)

    # Print pivot tables
    for metric in ["auc", "f1", "accuracy"]:
        _pivot_print(df, metric)

    # Per-metric plots
    for metric, ylabel, fname in [
        ("auc",      "Best UDA AUC",      "auc_vs_lakesize.png"),
        ("f1",       "Best UDA F1",       "f1_vs_lakesize.png"),
        ("accuracy", "Best UDA Accuracy", "accuracy_vs_lakesize.png"),
    ]:
        sub = df[df["metric"] == metric]
        if sub.empty:
            continue
        _line_plot(
            sub,
            y_col="best_uda",
            ylabel=ylabel,
            title=f"{ylabel} vs. lake size",
            path=OUTPUT_DIR / fname,
            hline_col="oracle",
            hline_label="oracle",
        )

    # Oracle gap closed (AUC only)
    auc_df = df[df["metric"] == "auc"].copy()
    if not auc_df.empty and "oracle_gap_closed" in auc_df.columns:
        _line_plot(
            auc_df,
            y_col="oracle_gap_closed",
            ylabel="Oracle gap closed (AUC)",
            title="Oracle gap closed vs. lake size",
            path=OUTPUT_DIR / "gap_closed_vs_lakesize.png",
        )

    # Sources found vs lake size (deduplicate — same for all metrics)
    src_df = (
        df[df["metric"] == "auc"][["target", "lake_size", "n_sources"]]
        .drop_duplicates()
    )
    if not src_df.empty:
        _line_plot(
            src_df,
            y_col="n_sources",
            ylabel="Repurposed sources found",
            title="Repurposed source count vs. lake size",
            path=OUTPUT_DIR / "sources_vs_lakesize.png",
        )

    # Runtime vs lake size
    rt_df = _load_runtimes(args.targets)
    _runtime_plot(rt_df, OUTPUT_DIR / "runtime_vs_lakesize.png")

    # Print runtime pivot table
    if not rt_df.empty:
        pivot = rt_df.pivot_table(index="target", columns="lake_size", values="runtime_min")
        print("\n=== Wall-clock runtime (minutes) by target × lake size ===")
        print(pivot.round(1).to_string())


if __name__ == "__main__":
    main()
