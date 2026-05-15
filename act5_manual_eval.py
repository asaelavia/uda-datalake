"""
Manual source evaluation — skip source repurposing, use hand-picked sources.

Loads a user-supplied JSON of {table_id: proxy_col}, runs schema alignment +
domain adaptation + evaluation, and saves results to a separate directory so
normal pipeline runs are never affected.

Usage
-----
    python act5_manual_eval.py --target churn --sources-json manual_sources/churn.json

    # or inline for quick tests (comma-separated table_id:proxy_col pairs):
    python act5_manual_eval.py --target churn \\
        --sources "z_attrition_rate_tables_licensed_train:Attrition,z_lead_time_tables_licensed_hotel_booking:is_canceled"

    # write results to a custom sub-directory tag (default: "manual"):
    python act5_manual_eval.py --target churn --sources-json churn.json --tag clean

Sources JSON format
-------------------
    {
        "z_attrition_rate_tables_licensed_train": "Attrition",
        "z_lead_time_tables_licensed_hotel_booking": "is_canceled"
    }

Results saved to
----------------
    results/act5/{target}/manual_{tag}/
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
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
    BREASTCANCER_DID,
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
)
from act5_gittables_lake import (
    _TARGETS,
    _build_labeled_lake,
    _mmr_select,
    _qt_within_dataset,
    RESULTS_BASE,
    MIN_DISCOVERY_SCORE_ABS,
    MIN_DISCOVERY_SCORE_REL,
    SELF_AUC_FLOOR,
    _load_cdc_obesity_target,
    _load_noshow_target,
    _load_stroke_target,
    _load_titanic_target,
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

NEIGHBOR_ALPHA = 0.5  # schema alignment neighbor context weight


def _load_target(target_name: str) -> pd.DataFrame:
    if target_name == "adult":
        return _load_adult_target()
    if target_name == "nyhouse":
        return _load_nyhouse_target()
    if target_name == "bank":
        return _load_openml_target(BANK_DID, "Bank Marketing", positive_values={"2", "yes"})
    if target_name == "diabetes":
        return _load_openml_target(DIABETES_DID, "Pima Diabetes",
                                   positive_values={"tested_positive", "1", "pos"})
    if target_name == "credit":
        return _load_openml_target(CREDIT_DID, "German Credit", positive_values={"good", "1"})
    if target_name == "churn":
        return _load_openml_target(CHURN_DID, "Telco Churn",
                                   positive_values={"1", "yes", "True"})
    if target_name == "heart":
        return _load_openml_target(HEART_DID, "Heart Disease",
                                   positive_values={"present", "1", "yes"})
    if target_name == "turnover":
        return _load_openml_target(TURNOVER_DID, "Employee Turnover",
                                   positive_values={"Left", "1", "yes"})
    if target_name == "crime":
        return _load_openml_target(CRIME_DID, "Communities and Crime",
                                   positive_values={"1", "yes", "true"})
    if target_name == "obesity":
        return _load_cdc_obesity_target()
    if target_name == "noshow":
        return _load_noshow_target()
    if target_name == "stroke":
        return _load_stroke_target()
    if target_name == "titanic":
        return _load_titanic_target()
    if target_name == "breastcancer":
        return _load_openml_target(BREASTCANCER_DID, "Breast Cancer Wisconsin",
                                   positive_values={"malignant"})
    raise ValueError(f"Unknown target '{target_name}'. Choices: {list(_TARGETS)}")


def run_manual_eval(
    target_name: str,
    manual_sources: dict[str, str],   # {table_id: proxy_col}
    flip_sources: set[str] | None = None,  # table IDs whose label should be inverted
    tag: str = "manual",
    top_k: int = TOP_K,
    lake_dir: Path | None = None,
) -> None:
    if not manual_sources:
        raise ValueError("manual_sources is empty — nothing to evaluate.")

    cfg = _TARGETS[target_name]
    results_dir = cfg.results_dir / f"manual_{tag}"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== Manual eval: target=%s  sources=%d  tag=%s ===",
                target_name, len(manual_sources), tag)
    logger.info("Results → %s", results_dir)

    # ------------------------------------------------------------------ #
    # Load encoder + manifest
    # ------------------------------------------------------------------ #
    logger.info("Loading encoder: %s", ENCODER_MODEL)
    encoder = SentenceTransformer(ENCODER_MODEL)

    cache_dir = lake_dir or Path("data/gittables")
    manifest_path = cache_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest_tables = manifest["tables"]
    id_to_path = {e["table_id"]: cache_dir / e["path"] for e in manifest_tables}
    logger.info("Manifest: %d entries", len(manifest_tables))

    # ------------------------------------------------------------------ #
    # Load target
    # ------------------------------------------------------------------ #
    target_df = _load_target(target_name)
    logger.info("Target: %d rows, positive_rate=%.3f, cols=%s",
                len(target_df), float(target_df[LABEL_COL].mean()),
                list(target_df.columns))

    target_train_df, target_test_df = train_test_split(
        target_df, test_size=ORACLE_TEST_SIZE, random_state=RANDOM_STATE, stratify=target_df[LABEL_COL],
    )
    y_true          = target_test_df[LABEL_COL].values
    target_features = target_test_df.drop(columns=[LABEL_COL])   # test features only (matches y_true)
    logger.info("Target split: %d oracle-train / %d test", len(target_train_df), len(target_test_df))

    # ------------------------------------------------------------------ #
    # Step 0: Build labeled lake from manual sources
    # ------------------------------------------------------------------ #
    logger.info("=== Step 0: Building labeled lake from %d manual sources ===",
                len(manual_sources))

    # Validate that all table IDs exist in the manifest
    missing = [tid for tid in manual_sources if tid not in id_to_path]
    if missing:
        logger.warning("%d table IDs not found in manifest (will be skipped): %s",
                       len(missing), missing)
        manual_sources = {k: v for k, v in manual_sources.items() if k not in missing}

    if not manual_sources:
        raise RuntimeError("No valid table IDs remain after manifest check.")

    # Load direction cache (same path as main pipeline)
    _dir_cache_path = Path(__file__).parent / "data" / "direction_cache.json"
    direction_cache: dict = {}
    if _dir_cache_path.exists():
        try:
            direction_cache = json.loads(_dir_cache_path.read_text())
            logger.info("Direction cache loaded: %d targets", len(direction_cache))
        except Exception as _e:
            logger.warning("Could not load direction cache: %s", _e)

    labeled_lake, label_names, proxy_quality_scores = _build_labeled_lake(
        repurpose_result=manual_sources,
        id_to_path=id_to_path,
        target_pos_rate=float(target_df[LABEL_COL].mean()),
        direction_cache=direction_cache,
        target_label=cfg.label_name,
    )

    if not labeled_lake:
        raise RuntimeError("No usable sources after _build_labeled_lake. "
                           "Check that proxy columns exist and have valid binary splits.")

    # Apply label inversion for polarity-flipped sources
    if flip_sources:
        for tid in flip_sources:
            if tid in labeled_lake:
                labeled_lake[tid][LABEL_COL] = 1 - labeled_lake[tid][LABEL_COL]
                logger.info("[Flip] Inverted label for '%s' (proxy='%s')",
                            tid, label_names.get(tid, "?"))
            else:
                logger.warning("[Flip] '%s' not in labeled lake — skipping flip.", tid)

    logger.info("Labeled lake built: %d sources", len(labeled_lake))
    for tid, df in labeled_lake.items():
        flipped = flip_sources and tid in flip_sources
        logger.info("  %-50s  col='%s'  pos_rate=%.3f  n=%d%s",
                    tid, label_names.get(tid, "?"),
                    float(df[LABEL_COL].mean()), len(df),
                    "  [FLIPPED]" if flipped else "")

    # ------------------------------------------------------------------ #
    # Step 1: Table Discovery
    # ------------------------------------------------------------------ #
    logger.info("=== Step 1: Table Discovery (%d sources) ===", len(labeled_lake))
    lake_features = {k: v.drop(columns=[LABEL_COL]) for k, v in labeled_lake.items()}
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
        target_pos_rate=float(target_df[LABEL_COL].mean()),
        balance_weight=BALANCE_WEIGHT,
    )
    scores = {tid: s * proxy_quality_scores.get(tid, 1.0) for tid, s in scores.items()}
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    logger.info("Discovery scores:")
    for tid, s in scores.items():
        logger.info("  %-50s  %.4f  col='%s'", tid, s, label_names.get(tid, "?"))

    # Use all sources (up to top_k); skip score gate so manual selection is respected
    n_use = min(top_k, len(scores))
    top_k_scores = _mmr_select(scores, label_names, encoder, n_use, lambda_=0.7)
    logger.info("Using %d sources after MMR selection:", len(top_k_scores))
    for tid, s in top_k_scores.items():
        logger.info("  %-50s  %.4f  col='%s'", tid, s, label_names.get(tid, "?"))

    pd.Series(scores, name="similarity").to_csv(results_dir / "discovery_scores.csv")

    # ------------------------------------------------------------------ #
    # Step 2: Schema Alignment
    # ------------------------------------------------------------------ #
    logger.info("=== Step 2: Schema Alignment ===")
    lake_top_k = {k: labeled_lake[k] for k in top_k_scores}
    aligned, col_mappings = schema_alignment.align_all(
        lake=lake_top_k,
        target=target_features,
        discovery_scores=top_k_scores,
        model=encoder,
        label_col=LABEL_COL,
        dist_threshold=DIST_THRESHOLD,
        min_coverage=0.0,
        min_similarity=0.35,
        fill_unmatched="nan",
        neighbor_alpha=NEIGHBOR_ALPHA,
    )

    if not aligned:
        raise RuntimeError("Schema alignment dropped all sources. "
                           "Check that the source tables have columns that overlap with the target.")

    logger.info("Aligned %d/%d sources", len(aligned), len(top_k_scores))

    # ------------------------------------------------------------------ #
    # Self-AUC diagnostic (informational only — no gate applied)
    # ------------------------------------------------------------------ #
    from sklearn.metrics import roc_auc_score as _roc_auc
    from xgboost import XGBClassifier as _XGB

    logger.info("=== Self-AUC diagnostic (no gate — informational) ===")
    sauc_rows = []
    for tid, df in aligned.items():
        X = df.drop(columns=[LABEL_COL])
        y = df[LABEL_COL]
        if y.nunique() < 2 or len(df) < 10:
            continue
        X_imp = X.copy()
        for col in X_imp.columns:
            X_imp[col] = X_imp[col].fillna(float(X_imp[col].median()))
        try:
            clf = _XGB(n_estimators=50, max_depth=3, random_state=42,
                       eval_metric="logloss", verbosity=0)
            clf.fit(X_imp, y)
            auc = float(_roc_auc(y, clf.predict_proba(X_imp)[:, 1]))
        except Exception as e:
            logger.debug("[SelfAUC] %s failed: %s", tid, e)
            auc = float("nan")
        sauc_rows.append({
            "table_id": tid,
            "proxy_col": label_names.get(tid, "?"),
            "n_rows": len(df),
            "self_auc": round(auc, 4),
            "discovery_score": round(top_k_scores.get(tid, 0.0), 4),
        })
        logger.info("  %-40s  proxy=%-20s  n=%4d  self_auc=%.3f",
                    tid[:40], label_names.get(tid, "?")[:20], len(df), auc)

    if sauc_rows:
        pd.DataFrame(sauc_rows).to_csv(results_dir / "source_self_auc.csv", index=False)

    # ------------------------------------------------------------------ #
    # Normalization (per-source QuantileTransformer)
    # ------------------------------------------------------------------ #
    num_cols = [c for c in target_features.columns
                if pd.api.types.is_numeric_dtype(target_features[c])]
    aligned = {
        tid: _qt_within_dataset(df, [c for c in num_cols if c in df.columns and c != LABEL_COL])
        for tid, df in aligned.items()
    }
    target_norm = _qt_within_dataset(target_features, num_cols)
    ttn = _qt_within_dataset(target_train_df.drop(columns=[LABEL_COL]), num_cols)
    ttn[LABEL_COL] = target_train_df[LABEL_COL].values
    target_train_norm = ttn
    logger.info("Per-source normalization applied to %d sources", len(aligned))

    # ------------------------------------------------------------------ #
    # Load unlabeled lake features for DANN (L5)
    # ------------------------------------------------------------------ #
    unlabeled_features = gittables_lake.load_gittables_features(
        target_cols=list(target_features.columns),
        max_tables=20_000,
        cache_dir=cache_dir,
    )

    # ------------------------------------------------------------------ #
    # Step 3: Domain Adaptation
    # ------------------------------------------------------------------ #
    logger.info("=== Step 3: Domain Adaptation (%d sources) ===", len(aligned))
    results = domain_adaptation.run_all(
        aligned=aligned,
        discovery_scores=top_k_scores,
        target=target_norm,
        label_col=LABEL_COL,
        weight_power=WEIGHT_POWER,
        unlabeled_features=unlabeled_features if unlabeled_features else None,
        random_state=RANDOM_STATE,
    )
    results["baseline_a"] = domain_adaptation.run_baseline_majority(target_norm)
    results["baseline_b"] = domain_adaptation.run_baseline_random(
        list(labeled_lake.values()), target_norm, LABEL_COL,
    )
    results["oracle"] = domain_adaptation.run_oracle(target_train_norm, target_norm, LABEL_COL)

    # ------------------------------------------------------------------ #
    # Step 4: Evaluation
    # ------------------------------------------------------------------ #
    logger.info("=== Step 4: Evaluation ===")
    target_pos_rate_test = float(y_true.mean())
    metrics_raw = evaluation.evaluate(results, y_true)
    metrics_cal = evaluation.evaluate(results, y_true, target_pos_rate=target_pos_rate_test)

    summary_raw = evaluation.summarise(metrics_raw)
    summary_cal = evaluation.summarise(metrics_cal)

    summary_raw.to_csv(results_dir / "metrics_uncalibrated.csv")
    summary_cal.to_csv(results_dir / "metrics.csv")

    n_sources = len(aligned)
    print(f"\n{'=' * 65}")
    print(f"MANUAL EVAL — {target_name.upper()} ({n_sources} hand-picked sources, tag='{tag}')")
    print(f"{'=' * 65}")
    print("\n--- Calibrated threshold (matched to target positive rate) ---")
    print(summary_cal.to_string())
    print(f"\nResults saved to {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run UDA pipeline on manually specified sources (no repurposing scan)."
    )
    parser.add_argument("--target", required=True, choices=list(_TARGETS),
                        help="Target dataset name.")
    parser.add_argument("--sources-json", type=Path, default=None,
                        help="Path to JSON file with {table_id: proxy_col} mapping.")
    parser.add_argument("--sources", type=str, default=None,
                        help="Inline sources: 'table_id1:col1,table_id2:col2,...'")
    parser.add_argument("--flip-sources", type=str, default=None,
                        help="Comma-separated table IDs whose label should be inverted "
                             "(polarity fix). Use 'all' to flip every source.")
    parser.add_argument("--tag", type=str, default="manual",
                        help="Sub-directory tag for results (default: 'manual').")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Max sources to use (default: {TOP_K}).")
    parser.add_argument("--lake-dir", type=Path, default=None,
                        help="Path to gittables lake directory (default: data/gittables).")
    args = parser.parse_args()

    if args.sources_json and args.sources:
        parser.error("Provide either --sources-json or --sources, not both.")
    if not args.sources_json and not args.sources:
        parser.error("Provide --sources-json or --sources.")

    if args.sources_json:
        with open(args.sources_json) as f:
            manual_sources = json.load(f)
    else:
        manual_sources = {}
        for pair in args.sources.split(","):
            pair = pair.strip()
            if ":" not in pair:
                parser.error(f"Invalid source spec '{pair}' — expected 'table_id:proxy_col'.")
            tid, col = pair.split(":", 1)
            manual_sources[tid.strip()] = col.strip()

    flip_sources: set[str] | None = None
    if args.flip_sources:
        if args.flip_sources.strip().lower() == "all":
            flip_sources = set(manual_sources.keys())
        else:
            flip_sources = {t.strip() for t in args.flip_sources.split(",")}

    run_manual_eval(
        target_name=args.target,
        manual_sources=manual_sources,
        flip_sources=flip_sources,
        tag=args.tag,
        top_k=args.top_k,
        lake_dir=args.lake_dir,
    )


if __name__ == "__main__":
    main()
