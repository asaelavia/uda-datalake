"""
shap_analysis.py — Feature importance analysis via SHAP for UDA models.

For each target, runs Level 2 domain adaptation on the GitTables lake sources
and computes SHAP values on the aligned test data.  Shows which features drive
predictions after schema alignment from heterogeneous source tables.

Usage:
    python shap_analysis.py --target adult
    python shap_analysis.py --target adult diabetes heart --output results/shap/
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
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
    _load_cdc_obesity_target,
)
import domain_adaptation
import gittables_lake
import schema_alignment
import table_discovery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_BASE = Path("results/shap")


def _load_target(target_name: str) -> pd.DataFrame:
    if target_name == "adult":
        return _load_adult_target()
    elif target_name == "nyhouse":
        return _load_nyhouse_target()
    elif target_name == "diabetes":
        return _load_openml_target(DIABETES_DID, "Pima Indians Diabetes",
                                   positive_values={"tested_positive", "1", "pos"})
    elif target_name == "credit":
        return _load_openml_target(CREDIT_DID, "German Credit", positive_values={"good", "1"})
    elif target_name == "bank":
        return _load_openml_target(BANK_DID, "Bank Marketing", positive_values={"2", "yes"})
    elif target_name == "churn":
        return _load_openml_target(CHURN_DID, "Telco Churn", positive_values={"1", "yes", "True"})
    elif target_name == "heart":
        return _load_openml_target(HEART_DID, "Heart Disease", positive_values={"present", "1", "yes"})
    elif target_name == "turnover":
        return _load_openml_target(TURNOVER_DID, "Employee Turnover", positive_values={"Left", "1", "yes"})
    elif target_name == "crime":
        return _load_openml_target(CRIME_DID, "Communities and Crime",
                                   positive_values={"1", "yes", "true"})
    elif target_name == "obesity":
        return _load_cdc_obesity_target()
    else:
        raise ValueError(f"Unknown target: {target_name}")


def run_shap(target_name: str, lake_dir: Path = None,
             output_dir: Path = RESULTS_BASE, top_n: int = 20) -> pd.DataFrame:
    """
    Run Level 2 UDA for `target_name`, compute SHAP values on test set,
    and return a DataFrame of mean |SHAP| per feature, sorted descending.
    """
    lake_dir = lake_dir or gittables_lake.DEFAULT_CACHE
    output_dir = output_dir / target_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load target
    target_df = _load_target(target_name)
    target_train_df, target_test_df = train_test_split(
        target_df, test_size=ORACLE_TEST_SIZE, random_state=RANDOM_STATE,
        stratify=target_df[LABEL_COL],
    )
    logger.info("Target %s: train=%d test=%d", target_name, len(target_train_df), len(target_test_df))

    # Load manifest
    manifest_path = lake_dir / gittables_lake.MANIFEST_FILE
    if not manifest_path.exists():
        logger.error("No manifest at %s — run downloader first", manifest_path)
        return pd.DataFrame()
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading encoder: %s (device=%s)", ENCODER_MODEL, device)
    encoder = SentenceTransformer(ENCODER_MODEL, device=device)

    # Load repurposed lake sources
    cfg = _TARGETS[target_name]
    target_features = target_test_df.drop(columns=[LABEL_COL])
    labeled_lake, label_names = _stream_load_and_repurpose(
        manifest_tables=manifest["tables"],
        cache_dir=lake_dir,
        label_name=cfg.label_name,
        encoder=encoder,
        threshold=REPURPOSE_THRESHOLD,
        target_features=target_features,
        target_pos_rate=float(target_df[LABEL_COL].mean()),
    )
    if not labeled_lake:
        logger.error("No labeled sources found for %s", target_name)
        return pd.DataFrame()
    logger.info("Loaded %d repurposed sources", len(labeled_lake))

    # Table discovery (mirror act5: fit on target_features before norm)
    lake_features = {k: v.drop(columns=[LABEL_COL]) for k, v in labeled_lake.items()}
    scores = table_discovery.discover_tables(
        lake=lake_features,
        target=target_features,
        model=encoder,
        distribution_weight=DISTRIBUTION_WEIGHT,
        target_label_name=cfg.label_name,
    )
    top_k_scores = dict(list(scores.items())[:TOP_K])

    # Schema alignment (before normalisation, same as act5)
    lake_top_k = {k: labeled_lake[k] for k in top_k_scores}
    aligned = schema_alignment.align_all(
        lake=lake_top_k,
        target=target_features,
        discovery_scores=top_k_scores,
        model=encoder,
        label_col=LABEL_COL,
        dist_threshold=DIST_THRESHOLD,
    )
    if not aligned:
        logger.error("No aligned sources for %s", target_name)
        return pd.DataFrame()

    # Quantile normalisation fitted on target_features (same as act5)
    qt, num_cols = _make_quantile_normalizer(target_features)
    aligned = {k: _apply_quantile_norm(v, qt, num_cols) for k, v in aligned.items()}
    target_norm = _apply_quantile_norm(target_features, qt, num_cols)
    target_train_norm = _apply_quantile_norm(target_train_df.drop(columns=[LABEL_COL]), qt, num_cols)
    target_train_norm[LABEL_COL] = target_train_df[LABEL_COL].values

    # Run Level 2 to get a trained model
    logger.info("Running Level 2 domain adaptation...")
    r_l2 = domain_adaptation.run_level2(
        aligned=aligned,
        discovery_scores=top_k_scores,
        target=target_norm,
        label_col=LABEL_COL,
        weight_power=WEIGHT_POWER,
    )
    model = r_l2.model
    if model is None:
        logger.error("Level 2 returned no model")
        return pd.DataFrame()

    # SHAP analysis
    logger.info("Computing SHAP values on %d test rows...", len(target_norm))
    X_test = target_norm.values
    feature_names = target_norm.columns.tolist()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, shap_values may be shape (n, p) or list of 2 arrays
    if isinstance(shap_values, list):
        sv = shap_values[1]  # positive class
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Save
    importance_df.to_csv(output_dir / "shap_importance.csv", index=False)
    logger.info("Saved SHAP importance to %s", output_dir / "shap_importance.csv")

    # Print top N
    print(f"\n=== SHAP Feature Importance: {target_name} (Level 2, GitTables lake) ===")
    print(importance_df.head(top_n).to_string(index=False))

    return importance_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", nargs="+", default=["adult", "heart", "diabetes"])
    parser.add_argument("--output", type=Path, default=RESULTS_BASE)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    for target in args.target:
        logger.info("=== SHAP analysis: %s ===", target)
        run_shap(target_name=target, output_dir=args.output, top_n=args.top_n)


if __name__ == "__main__":
    main()
