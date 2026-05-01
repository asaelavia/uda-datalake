"""
Orchestrator — runs all four pipeline steps end-to-end.

Usage
-----
python main.py --lake data/act1/ --target data/act1/CA.csv --label-col label --output results/act1/
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

import domain_adaptation
import evaluation
import schema_alignment
import table_discovery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _load_tables(directory: Path, label_col: str) -> dict[str, pd.DataFrame]:
    """Load all CSV/Parquet files in a directory as a table dict."""
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix == ".csv":
            tables[path.stem] = pd.read_csv(path)
        elif path.suffix in {".parquet", ".pq"}:
            tables[path.stem] = pd.read_parquet(path)
    logger.info("Loaded %d tables from %s", len(tables), directory)
    return tables


def run(
    lake_dir: Path,
    target_path: Path,
    output_dir: Path,
    label_col: str,
    threshold: float = 0.0,
    distribution_weight: float = 0.5,
    confidence_threshold: float = 0.85,
    pseudo_weight: float = 0.2,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load data ---
    lake = _load_tables(lake_dir, label_col)
    target_df = (
        pd.read_csv(target_path) if target_path.suffix == ".csv"
        else pd.read_parquet(target_path)
    )

    # separate ground-truth labels from target features (for evaluation only)
    y_true = None
    target_features = target_df.copy()
    if label_col in target_df.columns:
        y_true = target_df[label_col].values
        target_features = target_df.drop(columns=[label_col])
        logger.info("Target labels found — will evaluate after prediction.")

    # load encoder once and share across steps
    logger.info("Loading encoder: %s", _DEFAULT_MODEL)
    model = SentenceTransformer(_DEFAULT_MODEL)

    # --- step 1: table discovery ---
    logger.info("=== Step 1: Table Discovery ===")
    lake_features = {k: v.drop(columns=[label_col], errors="ignore") for k, v in lake.items()}
    scores = table_discovery.discover_tables(
        lake=lake_features,
        target=target_features,
        model=model,
        threshold=threshold,
        distribution_weight=distribution_weight,
    )
    logger.info("Discovery scores: %s", scores)
    pd.Series(scores, name="similarity").to_csv(output_dir / "discovery_scores.csv")

    if not scores:
        logger.error("No tables passed the similarity threshold. Exiting.")
        sys.exit(1)

    # --- step 2: schema alignment ---
    logger.info("=== Step 2: Schema Alignment ===")
    aligned = schema_alignment.align_all(
        lake=lake,
        target=target_features,
        discovery_scores=scores,
        model=model,
        label_col=label_col,
    )

    # --- step 3: domain adaptation ---
    logger.info("=== Step 3: Domain Adaptation ===")
    results = domain_adaptation.run_all(
        aligned=aligned,
        discovery_scores=scores,
        target=target_features,
        label_col=label_col,
        confidence_threshold=confidence_threshold,
        pseudo_weight=pseudo_weight,
    )

    # save predictions
    for level, result in results.items():
        pred_path = output_dir / f"predictions_{level}.csv"
        pd.Series(result.predictions, name="prediction").to_csv(pred_path, index=False)
        logger.info("Saved predictions → %s", pred_path)

    # --- step 4: evaluation (only if labels available) ---
    metrics = None
    if y_true is not None:
        logger.info("=== Step 4: Evaluation ===")
        metrics = evaluation.evaluate(results, y_true)
        summary = evaluation.summarise(metrics)
        print("\n" + summary.to_string())
        summary.to_csv(output_dir / "metrics.csv")
        logger.info("Saved metrics → %s", output_dir / "metrics.csv")

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UDA Data Lake Pipeline")
    parser.add_argument("--lake", type=Path, required=True, help="Directory of source tables")
    parser.add_argument("--target", type=Path, required=True, help="Path to target table file")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--label-col", default="label", help="Name of the label column")
    parser.add_argument("--threshold", type=float, default=0.0, help="Min discovery similarity score")
    parser.add_argument("--distribution-weight", type=float, default=0.5, help="Blend of schema vs distribution similarity (0=schema only, 1=distribution only)")
    parser.add_argument("--confidence", type=float, default=0.85, help="Pseudo-label confidence threshold")
    parser.add_argument("--pseudo-weight", type=float, default=0.2, help="Sample weight for pseudo-labels relative to source rows")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        lake_dir=args.lake,
        target_path=args.target,
        output_dir=args.output,
        label_col=args.label_col,
        threshold=args.threshold,
        distribution_weight=args.distribution_weight,
        confidence_threshold=args.confidence,
        pseudo_weight=args.pseudo_weight,
    )
