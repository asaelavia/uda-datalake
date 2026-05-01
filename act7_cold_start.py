"""
Act 7 — Cold-Start Setting

Simulates temporal label arrival: the target starts fully unlabeled and
receives T=10 weekly batches of labels in a random order.

At each week t:
  - α = n_labeled(t) / (n_labeled(t) + n_eff_src) blends lake trust → target trust
  - Train a separate lake model (score-weighted multi-source) and a target-only model
  - Blend their probabilities: p_blend = (1−α) × p_lake + α × p_target

Comparisons per week:
  lake_only   — static Act 5 UDA model (never retrained)
  target_only — XGBoost on accumulated labeled rows only
  blended     — α-blend of lake and target models
  oracle      — trained on full target train set

Output
------
  results/act7/{target}/cold_start_curve.csv   (week, n_labeled, method, auc)
  results/act7/{target}/cold_start_curve.png

Run
---
    python act7_cold_start.py --target adult
    python act7_cold_start.py --target diabetes
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from act4_openml_lake import (
    LABEL_COL,
    BANK_DID,
    DIABETES_DID,
    CREDIT_DID,
    CHURN_DID,
    HEART_DID,
    TURNOVER_DID,
    ENCODER_MODEL,
    ORACLE_TEST_SIZE,
    RANDOM_STATE,
    TOP_K,
    DISTRIBUTION_WEIGHT,
    LABEL_WEIGHT,
    BALANCE_WEIGHT,
    WEIGHT_POWER,
    DIST_THRESHOLD,
    _load_adult_target,
    _load_nyhouse_target,
    _load_openml_target,
    _make_quantile_normalizer,
    _apply_quantile_norm,
)
from act5_gittables_lake import (
    REPURPOSE_THRESHOLD,
    _TARGETS,
    _stream_load_and_repurpose,
)

import domain_adaptation
import evaluation
import gittables_lake
import schema_alignment
import table_discovery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

T_WEEKS      = 10
RESULTS_BASE = Path("results/act7")


def _make_xgb(**kwargs) -> XGBClassifier:
    defaults = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


def _safe_auc(y_true: np.ndarray, proba: Optional[np.ndarray]) -> Optional[float]:
    if proba is None or len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, proba[:, 1]))
    except Exception:
        return None


def run_cold_start(
    target_train_norm: pd.DataFrame,
    target_test_norm: pd.DataFrame,
    y_true: np.ndarray,
    aligned: dict[str, pd.DataFrame],
    top_k_scores: dict[str, float],
) -> pd.DataFrame:
    """
    Simulate T=10 weekly label arrivals and return the cold-start curve DataFrame.
    """
    n_train    = len(target_train_norm)
    batch_size = max(1, n_train // T_WEEKS)

    # Fixed random batch order (reproducible)
    rng = np.random.default_rng(RANDOM_STATE)
    batch_order = rng.permutation(n_train)

    # Static lake model — computed once, never retrained
    logger.info("Computing static lake model (act5-style) ...")
    uda_results = domain_adaptation.run_all(
        aligned=aligned,
        discovery_scores=top_k_scores,
        target=target_test_norm,
        label_col=LABEL_COL,
        weight_power=WEIGHT_POWER,
    )
    lake_proba = uda_results["ensemble"].probabilities
    if lake_proba is None:
        lake_proba = uda_results["level2"].probabilities
    if lake_proba is None:
        lake_proba = np.full((len(target_test_norm), 2), 0.5)
    lake_auc = _safe_auc(y_true, lake_proba)

    # Oracle — computed once
    r_oracle = domain_adaptation.run_oracle(
        target_train=target_train_norm,
        target_test=target_test_norm,
        label_col=LABEL_COL,
    )
    oracle_auc = _safe_auc(y_true, r_oracle.probabilities)

    rows = []
    for t in range(1, T_WEEKS + 1):
        labeled_idx = batch_order[: t * batch_size]
        labeled_df  = target_train_norm.iloc[labeled_idx]
        X_lab = labeled_df.drop(columns=[LABEL_COL])
        y_lab = labeled_df[LABEL_COL].values.astype(int)
        n_labeled = len(labeled_idx)

        alpha = domain_adaptation.compute_blend_weight(n_labeled, top_k_scores, aligned)

        # Target-only
        if len(np.unique(y_lab)) >= 2:
            m_tgt = _make_xgb()
            X_lab_clean = X_lab.replace([float("inf"), float("-inf")], float("nan"))
            m_tgt.fit(X_lab_clean, y_lab)
            p_tgt = m_tgt.predict_proba(target_test_norm)
            target_only_auc = _safe_auc(y_true, p_tgt)
        else:
            target_only_auc = None

        # Blended
        r_blend = domain_adaptation.train_blended(
            aligned=aligned,
            discovery_scores=top_k_scores,
            X_labeled=X_lab,
            y_labeled=y_lab,
            target=target_test_norm,
            label_col=LABEL_COL,
            alpha=alpha,
            weight_power=WEIGHT_POWER,
        )
        blended_auc = _safe_auc(y_true, r_blend.probabilities)

        logger.info(
            "  Week %2d  n_labeled=%4d  α=%.3f  "
            "lake=%.3f  target_only=%s  blended=%.3f  oracle=%.3f",
            t, n_labeled, alpha,
            lake_auc or 0.0,
            f"{target_only_auc:.3f}" if target_only_auc is not None else "N/A",
            blended_auc or 0.0,
            oracle_auc or 0.0,
        )

        rows.append({
            "week":         t,
            "n_labeled":    n_labeled,
            "alpha":        round(alpha, 4),
            "lake_only":    lake_auc,
            "target_only":  target_only_auc,
            "blended":      blended_auc,
            "oracle":       oracle_auc,
        })

    return pd.DataFrame(rows)


def run_experiment(target_name: str) -> None:
    cfg = _TARGETS[target_name]
    results_dir = RESULTS_BASE / target_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load manifest
    # ------------------------------------------------------------------ #
    cache_dir     = gittables_lake.DEFAULT_CACHE
    manifest_path = cache_dir / gittables_lake.MANIFEST_FILE
    if not manifest_path.exists():
        raise RuntimeError("GitTables cache not found. Run: python gittables_lake.py --download-zenodo")
    with open(manifest_path) as f:
        manifest = json.load(f)
    logger.info("Manifest loaded: %d table entries", len(manifest["tables"]))

    # ------------------------------------------------------------------ #
    # Load target
    # ------------------------------------------------------------------ #
    if target_name == "adult":
        target_df = _load_adult_target()
    elif target_name == "nyhouse":
        target_df = _load_nyhouse_target()
    elif target_name == "bank":
        target_df = _load_openml_target(BANK_DID, "Bank Marketing", positive_values={"2", "yes"})
    elif target_name == "diabetes":
        target_df = _load_openml_target(DIABETES_DID, "Pima Diabetes",
                                        positive_values={"tested_positive", "1", "pos"})
    elif target_name == "credit":
        target_df = _load_openml_target(CREDIT_DID, "German Credit", positive_values={"good", "1"})
    elif target_name == "churn":
        target_df = _load_openml_target(CHURN_DID, "Telco Churn", positive_values={"1", "yes", "True"})
    elif target_name == "heart":
        target_df = _load_openml_target(HEART_DID, "Heart Disease", positive_values={"present", "1", "yes"})
    elif target_name == "turnover":
        target_df = _load_openml_target(TURNOVER_DID, "Employee Turnover", positive_values={"Left", "1", "yes"})
    else:
        raise ValueError(f"Unknown target: {target_name!r}")

    target_train_df, target_test_df = train_test_split(
        target_df, test_size=ORACLE_TEST_SIZE, random_state=RANDOM_STATE,
        stratify=target_df[LABEL_COL],
    )
    y_true          = target_test_df[LABEL_COL].values.astype(int)
    target_features = target_test_df.drop(columns=[LABEL_COL])
    logger.info("Target split: %d train / %d test", len(target_train_df), len(target_test_df))

    # ------------------------------------------------------------------ #
    # Load encoder
    # ------------------------------------------------------------------ #
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading encoder: %s  (device=%s)", ENCODER_MODEL, _device)
    encoder = SentenceTransformer(ENCODER_MODEL, device=_device)

    # ------------------------------------------------------------------ #
    # Source repurposing (fast-path via done-cache)
    # ------------------------------------------------------------------ #
    labeled_lake, label_names = _stream_load_and_repurpose(
        manifest_tables=manifest["tables"],
        cache_dir=cache_dir,
        label_name=cfg.label_name,
        encoder=encoder,
        threshold=REPURPOSE_THRESHOLD,
        target_features=target_features,
        target_pos_rate=float(target_df[LABEL_COL].mean()),
    )
    if not labeled_lake:
        logger.error("No labeled sources found for '%s' — cannot run act7.", target_name)
        return

    # ------------------------------------------------------------------ #
    # Table discovery + schema alignment
    # ------------------------------------------------------------------ #
    lake_features    = {k: v.drop(columns=[LABEL_COL]) for k, v in labeled_lake.items()}
    source_pos_rates = {k: float(labeled_lake[k][LABEL_COL].mean()) for k in labeled_lake}
    target_pos_rate  = float(target_df[LABEL_COL].mean())

    scores = table_discovery.discover_tables(
        lake=lake_features,
        target=target_features,
        model=encoder,
        distribution_weight=DISTRIBUTION_WEIGHT,
        target_label_name=cfg.label_name,
        label_weight=LABEL_WEIGHT,
        lake_label_names=label_names,
        source_pos_rates=source_pos_rates,
        target_pos_rate=target_pos_rate,
        balance_weight=BALANCE_WEIGHT,
    )
    top_k_scores = dict(list(scores.items())[:TOP_K])
    logger.info("Top-%d sources selected.", len(top_k_scores))

    lake_top_k = {k: labeled_lake[k] for k in top_k_scores}
    aligned = schema_alignment.align_all(
        lake=lake_top_k,
        target=target_features,
        discovery_scores=top_k_scores,
        model=encoder,
        label_col=LABEL_COL,
        dist_threshold=DIST_THRESHOLD,
    )

    qt, num_cols       = _make_quantile_normalizer(target_features)
    aligned            = {k: _apply_quantile_norm(v, qt, num_cols) for k, v in aligned.items()}
    target_test_norm   = _apply_quantile_norm(target_features, qt, num_cols)
    target_train_norm  = _apply_quantile_norm(target_train_df.drop(columns=[LABEL_COL]), qt, num_cols)
    target_train_norm[LABEL_COL] = target_train_df[LABEL_COL].values

    # ------------------------------------------------------------------ #
    # Cold-start simulation
    # ------------------------------------------------------------------ #
    logger.info("=== Cold-start simulation: T=%d weeks ===", T_WEEKS)
    curve_df = run_cold_start(
        target_train_norm=target_train_norm,
        target_test_norm=target_test_norm,
        y_true=y_true,
        aligned=aligned,
        top_k_scores=top_k_scores,
    )

    csv_path = results_dir / "cold_start_curve.csv"
    curve_df.to_csv(csv_path, index=False)
    logger.info("Cold-start curve saved: %s", csv_path)

    evaluation.plot_cold_start_curve(curve_df, results_dir / "cold_start_curve.png")
    logger.info("\n%s", curve_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Act 7: Cold-start α-blend experiment")
    parser.add_argument(
        "--target",
        choices=list(_TARGETS),
        required=True,
        help="Target dataset name",
    )
    args = parser.parse_args()
    run_experiment(args.target)


if __name__ == "__main__":
    main()
