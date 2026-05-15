"""
Step 4 — Evaluation

Compares all adaptation levels against the naive baseline.
Primary claim: equal-weight baseline suffers from negative transfer;
discovery-weighted levels (1 and 2) prevent it.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from domain_adaptation import AdaptationResult

logger = logging.getLogger(__name__)


def _calibrate_predictions(
    probabilities: np.ndarray,
    target_pos_rate: float,
) -> np.ndarray:
    """
    Find the threshold on positive-class probabilities such that the fraction
    of predicted positives matches `target_pos_rate`, then return binary
    predictions under that threshold.

    This corrects for label distribution shift between source and target without
    requiring retraining — it only adjusts where the decision boundary sits.
    """
    pos_proba = probabilities[:, 1]
    # Binary-search over thresholds in [0, 1]
    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = (lo + hi) / 2
        if (pos_proba >= mid).mean() > target_pos_rate:
            lo = mid
        else:
            hi = mid
    threshold = (lo + hi) / 2
    logger.debug("Calibrated threshold: %.4f (target positive rate: %.3f)", threshold, target_pos_rate)
    return (pos_proba >= threshold).astype(int)


def _compute_metrics(
    y_true: np.ndarray,
    result: AdaptationResult,
    average: str = "binary",
    target_pos_rate: Optional[float] = None,
) -> dict[str, float]:
    # Restrict evaluation to the rows actually scored (e.g. LLM sample)
    if result.eval_mask is not None:
        y_true = y_true[result.eval_mask]
        if result.probabilities is not None:
            result = AdaptationResult(
                level=result.level,
                predictions=result.predictions[result.eval_mask],
                probabilities=result.probabilities[result.eval_mask],
                model=result.model,
            )
        else:
            result = AdaptationResult(
                level=result.level,
                predictions=result.predictions[result.eval_mask],
                model=result.model,
            )

    # Use calibrated predictions when a target positive rate is supplied
    # and probabilities are available; otherwise fall back to model.predict().
    if target_pos_rate is not None and result.probabilities is not None:
        preds = _calibrate_predictions(result.probabilities, target_pos_rate)
    else:
        preds = result.predictions

    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds, average=average, zero_division=0),
    }

    if result.probabilities is not None:
        try:
            if result.probabilities.shape[1] == 2:
                scores = result.probabilities[:, 1]
                metrics["auc"] = roc_auc_score(y_true, scores)
            else:
                metrics["auc"] = roc_auc_score(
                    y_true, result.probabilities, multi_class="ovr", average="macro"
                )
        except ValueError as exc:
            logger.warning("AUC could not be computed: %s", exc)

    return metrics


def evaluate(
    results: dict[str, AdaptationResult],
    y_true: np.ndarray,
    average: str = "binary",
    discovery_scores: Optional[dict[str, float]] = None,
    target_pos_rate: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build a metrics table comparing all adaptation levels.

    Parameters
    ----------
    results:
        Output of `domain_adaptation.run_all`.
    y_true:
        Ground-truth labels for the target table.
    average:
        Averaging strategy for F1 ('binary', 'macro', 'weighted').
    discovery_scores:
        If provided, appended as a reference column in the output.
    target_pos_rate:
        If provided, calibrate each model's decision threshold so that its
        predicted positive rate matches this value before computing accuracy
        and F1.  Useful when source and target have different label
        distributions (label shift).  AUC is always threshold-free.

    Returns
    -------
    DataFrame indexed by level with columns: accuracy, f1, auc (if available).
    """
    rows = {}
    for level, result in results.items():
        rows[level] = _compute_metrics(y_true, result, average=average,
                                       target_pos_rate=target_pos_rate)

    df = pd.DataFrame(rows).T.sort_index()
    df.index.name = "level"

    # canonical display order
    order = ["baseline_a", "baseline_b", "baseline", "llm_zero_shot", "level0", "level2", "level5", "level55", "level6", "ensemble", "source_ensemble", "routed", "oracle"]
    df = df.reindex([l for l in order if l in df.index])

    logger.info("\n%s", df.to_string())
    return df


def summarise(
    metrics: pd.DataFrame,
    baseline_level: str = "baseline",
) -> pd.DataFrame:
    """
    Append a delta column showing gain/loss relative to the naive baseline.
    Positive delta = improvement; negative = negative transfer.
    """
    if baseline_level not in metrics.index:
        logger.warning("Baseline level '%s' not found in metrics.", baseline_level)
        return metrics

    delta = metrics.subtract(metrics.loc[baseline_level])
    delta.columns = [f"{c}_delta" for c in delta.columns]
    return pd.concat([metrics, delta], axis=1)


def plot_label_curve(
    curve_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot AUC vs. labeled fraction (act6 semi-supervised).

    Expects columns: fraction, method, auc.
    One line per method; error bars = std across seeds.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot_label_curve.")
        return

    methods = curve_df["method"].unique()
    agg = (
        curve_df.groupby(["fraction", "method"])["auc"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    for method in methods:
        sub = agg[agg["method"] == method].sort_values("fraction")
        ax.errorbar(
            sub["fraction"], sub["mean"], yerr=sub["std"],
            label=method, marker="o", capsize=3,
        )
    ax.set_xlabel("Labeled fraction")
    ax.set_ylabel("AUC")
    ax.set_xscale("log")
    ax.legend()
    ax.set_title("Semi-supervised: AUC vs labeled fraction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Label curve plot saved: %s", output_path)


def plot_cold_start_curve(
    curve_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot AUC vs. number of labeled samples (act7 cold-start).

    Expects columns: week, n_labeled, method, auc.
    One line per method.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot_cold_start_curve.")
        return

    methods = [c for c in curve_df.columns if c not in ("week", "n_labeled")]

    fig, ax = plt.subplots(figsize=(7, 4))
    for method in methods:
        sub = curve_df.sort_values("n_labeled")
        ax.plot(sub["n_labeled"], sub[method], label=method, marker="o")
    ax.set_xlabel("Labeled samples (cumulative)")
    ax.set_ylabel("AUC")
    ax.legend()
    ax.set_title("Cold-start: AUC vs. labeled samples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Cold-start curve plot saved: %s", output_path)
