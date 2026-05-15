"""
Transferability Analysis — validates that the transferability score correlates
with actual AUC improvement from lake-based UDA.

Reads results/act5/{target}/transferability.csv and metrics.csv for all
available targets and produces:

  results/transferability/correlation_table.csv  — per-target numbers
  results/transferability/score_vs_transfer.png  — scatter: score vs AUC gain
  results/transferability/fast_vs_true.png       — fast score vs true score
  results/transferability/topk_overlap.csv       — top-K candidate overlap %

Usage
-----
    python transferability_analysis.py
    python transferability_analysis.py --targets adult heart credit diabetes bank churn
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_BASE  = Path("results/act5")
OUTPUT_DIR    = Path("results/transferability")
_SKIP_LEVELS  = {"oracle", "baseline_a", "baseline_b", "llm_zero_shot"}


def _best_uda_auc(metrics_path: Path) -> tuple[str, float]:
    """Return (best_level, best_auc) excluding oracle and baselines."""
    df = pd.read_csv(metrics_path, index_col=0)
    valid = df[~df.index.isin(_SKIP_LEVELS | {"baseline"})]
    if valid.empty or "auc" not in valid.columns:
        return ("missing", float("nan"))
    idx = valid["auc"].idxmax()
    return str(idx), float(valid.loc[idx, "auc"])


def _baseline_auc(metrics_path: Path) -> float:
    df = pd.read_csv(metrics_path, index_col=0)
    if "baseline" in df.index and "auc" in df.columns:
        return float(df.loc["baseline", "auc"])
    return float("nan")


def _load_score(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return df.iloc[0].to_dict() if len(df) > 0 else {}


def compile_table(targets: list[str]) -> pd.DataFrame:
    rows = []
    for t in targets:
        base = RESULTS_BASE / t
        metrics_p = base / "metrics.csv"
        true_p    = base / "transferability.csv"
        fast_p    = base / "transferability_fast.csv"

        if not metrics_p.exists():
            logger.warning("Missing metrics: %s", metrics_p)
            continue

        best_level, best_uda = _best_uda_auc(metrics_p)
        baseline = _baseline_auc(metrics_p)
        auc_improvement = best_uda - baseline if not (np.isnan(best_uda) or np.isnan(baseline)) else float("nan")

        # oracle_gap_closed = fraction of the achievable oracle gap actually closed
        df_m = pd.read_csv(metrics_p, index_col=0)
        oracle_auc = float(df_m.loc["oracle", "auc"]) if "oracle" in df_m.index and "auc" in df_m.columns else float("nan")
        oracle_gap = oracle_auc - baseline
        oracle_gap_closed = (best_uda - baseline) / oracle_gap if (not np.isnan(oracle_gap) and oracle_gap > 1e-6) else float("nan")

        true_score = _load_score(true_p)
        fast_score = _load_score(fast_p)

        row = {
            "target":            t,
            "baseline_auc":      baseline,
            "best_uda_auc":      best_uda,
            "oracle_auc":        oracle_auc,
            "best_uda_level":    best_level,
            "auc_improvement":   auc_improvement,
            "oracle_gap_closed": oracle_gap_closed,
        }

        for key, prefix, src in [("true", "true_", true_score), ("fast", "fast_", fast_score)]:
            for component in ["overall", "repurpose_yield", "discovery_quality",
                               "alignment_density", "label_shift", "feature_overlap",
                               "pas_score", "spa_score", "cslp_score", "lcc_score",
                               "pas_loose_score", "pca_pas_score", "zscore_copas_score",
                               "npas_score", "tsc_score",
                               "source_consistency", "top1_score", "n_sources"]:
                row[f"{prefix}{component}"] = src.get(component, float("nan"))

        rows.append(row)

    return pd.DataFrame(rows)


def print_correlations(df: pd.DataFrame) -> None:
    outcomes = [
        ("oracle_gap_closed", "Oracle gap closed  (best_uda-baseline)/(oracle-baseline)"),
        ("auc_improvement",   "AUC improvement    (best_uda - baseline)"),
        ("best_uda_auc",      "Best UDA AUC       (absolute)"),
    ]

    score_cols = ["true_overall", "true_spa_score", "true_pas_score", "true_zscore_copas_score",
                  "true_npas_score", "true_tsc_score",
                  "fast_overall", "fast_spa_score", "fast_pas_score",
                  "fast_cslp_score", "fast_lcc_score", "fast_pas_loose_score",
                  "fast_pca_pas_score", "fast_zscore_copas_score",
                  "true_repurpose_yield", "true_discovery_quality", "true_top1_score",
                  "true_feature_overlap", "true_alignment_density",
                  "true_label_shift", "true_source_consistency"]

    for outcome_col, outcome_label in outcomes:
        if outcome_col not in df.columns:
            continue
        y = df[outcome_col].values
        valid = ~np.isnan(y)
        if valid.sum() < 3:
            continue

        print(f"\n=== Correlation vs {outcome_label} ===")
        print(f"{'Component':<35}  {'Spearman rho':>12}  {'Pearson r':>10}  {'p-value':>10}")
        print("-" * 72)

        for col in score_cols:
            if col not in df.columns:
                continue
            x = df[col].values
            mask = valid & ~np.isnan(x)
            if mask.sum() < 3:
                continue
            rho, p_sp = spearmanr(x[mask], y[mask])
            r, p_pe   = pearsonr(x[mask], y[mask])
            print(f"{col:<35}  {rho:>12.3f}  {r:>10.3f}  {p_sp:>10.4f}")


def _scatter(x, y, labels, xlabel, ylabel, title, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot %s", path)
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=80, zorder=3)
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=9)

    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() >= 2:
        m, b = np.polyfit(x[valid], y[valid], 1)
        xs = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(xs, m * xs + b, "r--", alpha=0.6, linewidth=1.5)
        rho, _ = spearmanr(x[valid], y[valid])
        ax.set_title(f"{title}  (Spearman ρ={rho:.2f})")
    else:
        ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+",
                        default=["adult", "heart", "credit", "diabetes", "bank",
                                 "turnover", "noshow", "nyhouse", "obesity",
                                 "titanic", "stroke", "churn"])
    parser.add_argument("--exclude", nargs="+", default=[],
                        help="Targets to exclude from analysis (e.g. --exclude churn)")
    parser.add_argument("--suffix", default="",
                        help="Suffix appended to output filenames (e.g. --suffix _no_churn)")
    args = parser.parse_args()

    targets = [t for t in args.targets if t not in args.exclude]

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = compile_table(targets)
    if df.empty:
        logger.error("No data found. Run act5_gittables_lake.py for at least one target first.")
        return

    sfx = args.suffix
    out_csv = out_dir / f"correlation_table{sfx}.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Saved: %s", out_csv)

    print("\n=== Transferability Score Table ===")
    display_cols = ["target", "true_overall", "true_pas_score",
                    "fast_pas_score", "fast_pas_loose_score", "fast_pca_pas_score",
                    "oracle_gap_closed", "best_uda_level"]
    print(df[[c for c in display_cols if c in df.columns]].round(3).to_string(index=False))

    print_correlations(df)

    # Scatter: true overall vs oracle gap closed
    _scatter(
        x=df["true_overall"].values.astype(float),
        y=df["oracle_gap_closed"].values.astype(float),
        labels=df["target"].tolist(),
        xlabel="Transferability score (true)",
        ylabel="Oracle gap closed",
        title="Lake transferability → oracle gap closed",
        path=out_dir / f"score_vs_transfer{sfx}.png",
    )

    # Scatter: fast vs true overall
    if "fast_overall" in df.columns:
        _scatter(
            x=df["fast_overall"].values.astype(float),
            y=df["true_overall"].values.astype(float),
            labels=df["target"].tolist(),
            xlabel="Transferability score (fast)",
            ylabel="Transferability score (true)",
            title="Fast score vs true score",
            path=out_dir / f"fast_vs_true{sfx}.png",
        )


if __name__ == "__main__":
    main()
