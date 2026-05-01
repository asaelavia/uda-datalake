"""
Act 6 — Semi-Supervised Setting

The target dataset has a small labeled fraction f ∈ {1%, 5%, 10%, 25%}.
The remaining target rows are unlabeled.

Validated Lake Adaptation (VLA):
  1. Use the labeled val set to score each lake source (VTS = AUC on val).
  2. Discard sources below min_vts=0.52.
  3. Pool validated sources (discovery-score-weighted) + labeled val rows
     (boosted weight) → VLA model.
  4. One round of self-training on unlabeled target → VLA+ST model.

Comparisons (per fraction × seed):
  target_only  — XGBoost on labeled val rows only
  uda          — Act 5 level (no labeled target; best of {level2, level5, ensemble})
  vla          — VLA without self-training
  vla_st       — VLA + one self-training round
  oracle       — trained on full target train set (precomputed once, same for all seeds)

Output
------
  results/act6/{target}/label_curve.csv   (fraction, seed, method, auc)
  results/act6/{target}/label_curve.png

Run
---
    python act6_semi_supervised.py --target adult
    python act6_semi_supervised.py --target diabetes
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

from act4_openml_lake import (
    LABEL_COL,
    BANK_DID,
    DIABETES_DID,
    CREDIT_DID,
    CHURN_DID,
    HEART_DID,
    TURNOVER_DID,
    CRIME_DID,
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
    EMBED_BATCH_TABLES,
    _TARGETS,
    _stream_load_and_repurpose,
    _load_cdc_obesity_target,
)

import domain_adaptation
import evaluation
import gittables_lake
import schema_alignment
import table_discovery
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FRACTIONS  = [0.001, 0.005, 0.01, 0.05, 0.10, 0.25]
N_SEEDS    = 5
MIN_VAL_N  = 5   # minimum labeled rows per class to run VLA
RESULTS_BASE = Path("results/act6")


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


def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, proba[:, 1]))
    except Exception:
        return None


def _safe_accuracy(y_true: np.ndarray, proba: np.ndarray) -> Optional[float]:
    try:
        preds = (proba[:, 1] >= 0.5).astype(int)
        return float((preds == y_true).mean())
    except Exception:
        return None


def run_fraction(
    fraction: float,
    seed: int,
    target_train_norm: pd.DataFrame,
    target_test_norm: pd.DataFrame,
    y_true: np.ndarray,
    aligned: dict[str, pd.DataFrame],
    top_k_scores: dict[str, float],
    uda_proba: np.ndarray,
    oracle_auc: Optional[float],
    oracle_acc: Optional[float] = None,
    uda_model: Optional[object] = None,
) -> Optional[dict[str, Optional[float]]]:
    """
    Run all methods for one (fraction, seed) combination.

    Returns {method: auc} dict, or None if the fraction is infeasible
    (not enough minority-class samples for a stratified split).
    """
    n_total = len(target_train_norm)
    n_val   = max(int(n_total * fraction), 2 * MIN_VAL_N)

    # Check feasibility: need >= 2 samples in each class for stratified split
    n_pos = int(target_train_norm[LABEL_COL].sum())
    n_neg = n_total - n_pos
    if min(n_pos, n_neg) < 2:
        logger.info("Skipping fraction=%.3f: not enough minority class samples", fraction)
        return None

    # Stratified sample of labeled rows
    labeled_idx, _ = train_test_split(
        np.arange(n_total),
        train_size=n_val,
        random_state=seed,
        stratify=target_train_norm[LABEL_COL].values,
    )
    labeled_df   = target_train_norm.iloc[labeled_idx]
    unlabeled_df = target_train_norm.drop(index=target_train_norm.index[labeled_idx])

    # Further split labeled into train (80%) and calibration (20%) for isotonic calibration.
    # The calibration set is held out from VLA and target_only training.
    if len(labeled_df) >= 10 and len(np.unique(labeled_df[LABEL_COL].values)) >= 2:
        lab_train_df, lab_cal_df = train_test_split(
            labeled_df,
            test_size=0.2,
            random_state=seed,
            stratify=labeled_df[LABEL_COL].values,
        )
    else:
        lab_train_df = labeled_df
        lab_cal_df   = labeled_df.iloc[:0]  # empty

    X_val   = lab_train_df.drop(columns=[LABEL_COL])
    y_val   = lab_train_df[LABEL_COL].values.astype(int)
    X_cal   = lab_cal_df.drop(columns=[LABEL_COL])
    y_cal   = lab_cal_df[LABEL_COL].values.astype(int)
    X_unlab = unlabeled_df.drop(columns=[LABEL_COL])
    target_test = target_test_norm  # features only (no label)

    row: dict[str, Optional[float]] = {}

    # --- UDA (static, same for all seeds at same fraction) ---
    row["uda"] = _safe_auc(y_true, uda_proba)
    row["uda_acc"] = _safe_accuracy(y_true, uda_proba)

    # --- UDA calibrated (use labeled val set to fix miscalibrated pseudo-label probabilities) ---
    if uda_model is not None and len(X_cal) >= 5 and len(np.unique(y_cal)) >= 2:
        _uda_cal_r = domain_adaptation.calibrate_result(
            domain_adaptation.AdaptationResult("uda_cal", (uda_proba[:, 1] >= 0.5).astype(int), uda_proba, uda_model),
            X_cal, y_cal, target_test,
        )
        row["uda_cal"] = _safe_auc(y_true, _uda_cal_r.probabilities)
        row["uda_cal_acc"] = _safe_accuracy(y_true, _uda_cal_r.probabilities)
    else:
        row["uda_cal"] = None
        row["uda_cal_acc"] = None

    # --- Target-only ---
    if len(np.unique(y_val)) >= 2:
        m = _make_xgb()
        m.fit(X_val, y_val)
        to_proba = m.predict_proba(target_test)
        row["target_only"] = _safe_auc(y_true, to_proba)
        row["target_only_acc"] = _safe_accuracy(y_true, to_proba)
    else:
        m = None
        to_proba = None
        row["target_only"] = None
        row["target_only_acc"] = None

    # --- Target-only calibrated (isotonic, using held-out cal set) ---
    if m is not None and to_proba is not None and len(X_cal) >= 5 and len(np.unique(y_cal)) >= 2:
        _cal_result = domain_adaptation.calibrate_result(
            domain_adaptation.AdaptationResult("target_only_cal", m.predict(target_test), to_proba, m),
            X_cal, y_cal, target_test,
        )
        row["target_only_cal"] = _safe_auc(y_true, _cal_result.probabilities)
        row["target_only_cal_acc"] = _safe_accuracy(y_true, _cal_result.probabilities)
    else:
        row["target_only_cal"] = None
        row["target_only_cal_acc"] = None

    # --- VLA ---
    validated = domain_adaptation.validate_sources(
        aligned, top_k_scores, X_val, y_val, LABEL_COL,
    )
    r_vla = domain_adaptation.train_vla(
        validated, aligned, top_k_scores,
        X_val, y_val, target_test, LABEL_COL,
    )
    row["vla"] = _safe_auc(y_true, r_vla.probabilities)
    row["vla_acc"] = _safe_accuracy(y_true, r_vla.probabilities)

    # --- VLA + iterative self-training ---
    r_vla_st = domain_adaptation.train_vla_self_train(
        r_vla, X_unlab, target_test, X_val, y_val, LABEL_COL,
    )
    row["vla_st"] = _safe_auc(y_true, r_vla_st.probabilities)
    row["vla_st_acc"] = _safe_accuracy(y_true, r_vla_st.probabilities)

    # --- Routed: smart decision between VLA and target-only ---
    route_reason, r_routed = domain_adaptation.route_lake_decision(
        aligned=aligned,
        discovery_scores=top_k_scores,
        X_val=X_val,
        y_val=y_val,
        target=target_test,
        label_col=LABEL_COL,
    )
    row["routed"] = _safe_auc(y_true, r_routed.probabilities)
    row["routed_acc"] = _safe_accuracy(y_true, r_routed.probabilities)

    # --- Oracle (precomputed once, same for all seeds) ---
    row["oracle"] = oracle_auc
    row["oracle_acc"] = oracle_acc

    logger.info(
        "  f=%.3f  seed=%d  n_val=%d  validated=%d/%d  "
        "target_only=%.3f  vla=%.3f  vla_st=%.3f  routed=%.3f(%s)  oracle=%.3f",
        fraction, seed, n_val, len(validated), len(aligned),
        row["target_only"] or 0.0,
        row["vla"] or 0.0,
        row["vla_st"] or 0.0,
        row["routed"] or 0.0,
        route_reason,
        oracle_auc or 0.0,
    )
    return row


def run_experiment(
    target_name: str,
    top_k: int = TOP_K,
    lake_dir: Optional[Path] = None,
) -> None:
    cfg = _TARGETS[target_name]
    lake_dir = lake_dir or gittables_lake.DEFAULT_CACHE
    lake_tag  = lake_dir.name if lake_dir != gittables_lake.DEFAULT_CACHE else None
    base      = RESULTS_BASE / target_name if top_k == TOP_K else RESULTS_BASE / f"{target_name}_k{top_k}"
    results_dir = (base.parent / lake_tag / base.name) if lake_tag else base
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load manifest
    # ------------------------------------------------------------------ #
    cache_dir     = lake_dir
    manifest_path = cache_dir / gittables_lake.MANIFEST_FILE
    if not manifest_path.exists():
        raise RuntimeError(f"Lake cache not found at {cache_dir}. Check --lake-dir.")
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
    elif target_name == "crime":
        target_df = _load_openml_target(CRIME_DID, "Communities and Crime", positive_values={"1", "yes", "true"})
    elif target_name == "obesity":
        target_df = _load_cdc_obesity_target()
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
    target_pos_rate = float(target_df[LABEL_COL].mean())
    labeled_lake, label_names = _stream_load_and_repurpose(
        manifest_tables=manifest["tables"],
        cache_dir=cache_dir,
        label_name=cfg.label_name,
        encoder=encoder,
        threshold=REPURPOSE_THRESHOLD,
        target_features=target_features,
        target_pos_rate=target_pos_rate,
    )
    if not labeled_lake:
        logger.error("No labeled sources found for '%s' — cannot run act6.", target_name)
        return

    # ------------------------------------------------------------------ #
    # Table discovery + schema alignment (same as act5)
    # ------------------------------------------------------------------ #
    lake_features    = {k: v.drop(columns=[LABEL_COL]) for k, v in labeled_lake.items()}
    source_pos_rates = {k: float(labeled_lake[k][LABEL_COL].mean()) for k in labeled_lake}

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
    top_k_scores = dict(list(scores.items())[:top_k])
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
    # Pre-compute UDA baseline (act5-style: run_all, use best of level2/level5/ensemble)
    # ------------------------------------------------------------------ #
    logger.info("Pre-computing UDA baseline (act5-style) ...")
    uda_results = domain_adaptation.run_all(
        aligned=aligned,
        discovery_scores=top_k_scores,
        target=target_test_norm,
        label_col=LABEL_COL,
        weight_power=WEIGHT_POWER,
    )
    # Pick best UDA method by AUC on test set (matches act5 "best UDA" reporting)
    uda_proba = np.full((len(target_test_norm), 2), 0.5)
    uda_model = None  # underlying XGBClassifier if available (for calibration)
    _best_uda_auc = -1.0
    for _m in ["level2", "level5", "ensemble", "source_ensemble", "level0"]:
        _r = uda_results.get(_m)
        if _r is None or _r.probabilities is None:
            continue
        try:
            _auc = float(roc_auc_score(y_true, _r.probabilities[:, 1]))
        except Exception:
            continue
        if _auc > _best_uda_auc:
            _best_uda_auc = _auc
            uda_proba = _r.probabilities
            uda_model = _r.model
    logger.info("Best UDA method AUC on test: %.4f", _best_uda_auc)

    # ------------------------------------------------------------------ #
    # Pre-compute oracle once (full target train set, never changes)
    # ------------------------------------------------------------------ #
    logger.info("Pre-computing oracle ...")
    r_oracle = domain_adaptation.run_oracle(
        target_train=target_train_norm,
        target_test=target_test_norm,
        label_col=LABEL_COL,
    )
    oracle_auc = _safe_auc(y_true, r_oracle.probabilities)
    oracle_acc = _safe_accuracy(y_true, r_oracle.probabilities)
    logger.info("Oracle AUC: %.4f", oracle_auc or 0.0)

    # ------------------------------------------------------------------ #
    # Semi-supervised experiment: fraction × seed
    # ------------------------------------------------------------------ #
    rows = []
    for fraction in FRACTIONS:
        logger.info("=== Fraction %.3f ===", fraction)
        for seed in range(N_SEEDS):
            result = run_fraction(
                fraction=fraction,
                seed=seed,
                target_train_norm=target_train_norm.copy(),
                target_test_norm=target_test_norm,
                y_true=y_true,
                aligned=aligned,
                top_k_scores=top_k_scores,
                uda_proba=uda_proba,
                oracle_auc=oracle_auc,
                oracle_acc=oracle_acc,
                uda_model=uda_model,
            )
            if result is None:
                continue
            for method, val in result.items():
                if method.endswith("_acc"):
                    rows.append({
                        "fraction": fraction,
                        "seed":     seed,
                        "method":   method[:-4],  # strip _acc suffix
                        "metric":   "accuracy",
                        "value":    val,
                    })
                else:
                    rows.append({
                        "fraction": fraction,
                        "seed":     seed,
                        "method":   method,
                        "metric":   "auc",
                        "value":    val,
                    })

    curve_df = pd.DataFrame(rows)
    csv_path = results_dir / "label_curve.csv"
    curve_df.to_csv(csv_path, index=False)
    logger.info("Label curve saved: %s", csv_path)

    # ------------------------------------------------------------------ #
    # Plot (AUC only)
    # ------------------------------------------------------------------ #
    auc_df = (
        curve_df[curve_df["metric"] == "auc"]
        .rename(columns={"value": "auc"})
    )
    evaluation.plot_label_curve(auc_df, results_dir / "label_curve.png")

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    for metric in ["auc", "accuracy"]:
        sub = curve_df[curve_df["metric"] == metric]
        summary = (
            sub.groupby(["fraction", "method"])["value"]
            .agg(["mean", "std"])
            .round(4)
        )
        logger.info("\n=== %s ===\n%s", metric.upper(), summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Act 6: Semi-supervised VLA experiment")
    parser.add_argument(
        "--target",
        choices=list(_TARGETS),
        required=True,
        help="Target dataset name",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Number of top sources to use (default: {TOP_K})")
    parser.add_argument("--lake-dir", type=Path, default=None,
                        help="Path to lake cache directory (default: data/gittables)")
    args = parser.parse_args()
    run_experiment(args.target, top_k=args.top_k, lake_dir=args.lake_dir)


if __name__ == "__main__":
    main()
