"""
Sensitivity analysis: sweeps top-K and repurpose-threshold over selected targets.

Reads results/act5/{target}[_k{k}][_thr{thr}]/metrics.csv for each combination
and compiles comparison tables.

Usage
-----
    # First run the experiments:
    python act5_gittables_lake.py --target adult --top-k 5
    python act5_gittables_lake.py --target adult --top-k 10
    python act5_gittables_lake.py --target adult --top-k 50
    python act5_gittables_lake.py --target adult --repurpose-threshold 0.55
    python act5_gittables_lake.py --target adult --repurpose-threshold 0.65
    # (repeat for other targets)

    # Then compile:
    python sensitivity_analysis.py
    python sensitivity_analysis.py --targets adult credit heart --ks 5 10 20 50
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_BASE    = Path("results/act5")
SENSITIVITY_DIR = Path("results/sensitivity")

DEFAULT_K   = 20
DEFAULT_THR = 0.70

_ORACLE_LEVELS   = {"oracle", "baseline_a", "baseline_b"}
_SKIP_LEVELS     = _ORACLE_LEVELS | {"baseline"}


def _results_dir(target: str, k: int, thr: float) -> Path:
    """Mirror the directory naming logic in act5_gittables_lake.run_experiment()."""
    k_suf   = f"_k{k}"   if k   != DEFAULT_K   else ""
    thr_suf = f"_thr{thr:.2f}" if thr != DEFAULT_THR else ""
    if k_suf or thr_suf:
        return RESULTS_BASE / f"{target}{k_suf}{thr_suf}"
    return RESULTS_BASE / target


def _best_uda_auc(metrics_path: Path) -> tuple[str, float]:
    """Return (best_level_name, best_auc) from a metrics.csv, excluding oracle/baselines."""
    try:
        df = pd.read_csv(metrics_path, index_col=0)
    except Exception as exc:
        logger.warning("Could not read %s: %s", metrics_path, exc)
        return ("missing", float("nan"))

    valid = df[~df.index.isin(_SKIP_LEVELS)]
    if valid.empty or "auc" not in valid.columns:
        return ("missing", float("nan"))

    best_idx = valid["auc"].idxmax()
    return (str(best_idx), float(valid.loc[best_idx, "auc"]))


def compile_k_sensitivity(
    targets: list[str],
    ks: list[int],
    thr: float = DEFAULT_THR,
) -> pd.DataFrame:
    """AUC vs K (threshold fixed)."""
    rows = []
    for target in targets:
        for k in ks:
            path = _results_dir(target, k, thr) / "metrics.csv"
            if not path.exists():
                logger.warning("Missing: %s", path)
                level, auc = "missing", float("nan")
            else:
                level, auc = _best_uda_auc(path)
            rows.append({"target": target, "k": k, "best_level": level, "auc": auc})
    return pd.DataFrame(rows)


def compile_thr_sensitivity(
    targets: list[str],
    thresholds: list[float],
    k: int = DEFAULT_K,
) -> pd.DataFrame:
    """AUC vs repurpose threshold (K fixed)."""
    rows = []
    for target in targets:
        for thr in thresholds:
            path = _results_dir(target, k, thr) / "metrics.csv"
            if not path.exists():
                logger.warning("Missing: %s", path)
                level, auc = "missing", float("nan")
            else:
                level, auc = _best_uda_auc(path)
            rows.append({"target": target, "threshold": thr, "best_level": level, "auc": auc})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile sensitivity analysis tables")
    parser.add_argument("--targets", nargs="+", default=["adult", "credit", "heart"])
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20, 50])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.55, 0.65, 0.70])
    args = parser.parse_args()

    SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)

    # K sensitivity
    k_df = compile_k_sensitivity(args.targets, args.ks)
    k_path = SENSITIVITY_DIR / "sensitivity_k.csv"
    k_df.to_csv(k_path, index=False)
    logger.info("Saved: %s", k_path)

    k_pivot = k_df.pivot_table(index="target", columns="k", values="auc", aggfunc="first")
    print("\n=== AUC vs Top-K (threshold=0.70) ===")
    print(k_pivot.round(4).to_string())

    # Threshold sensitivity
    thr_df = compile_thr_sensitivity(args.targets, args.thresholds)
    thr_path = SENSITIVITY_DIR / "sensitivity_thr.csv"
    thr_df.to_csv(thr_path, index=False)
    logger.info("Saved: %s", thr_path)

    thr_pivot = thr_df.pivot_table(index="target", columns="threshold", values="auc", aggfunc="first")
    print("\n=== AUC vs Repurpose Threshold (K=20) ===")
    print(thr_pivot.round(4).to_string())


if __name__ == "__main__":
    main()
