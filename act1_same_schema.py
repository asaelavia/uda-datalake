"""
Act 1 — Same Schema, Distribution Shift Only (Folktables / ACSIncome)

Setup
-----
- Lake  : labeled ACSIncome tables for MS, WV, AR, WA, OR  (source states)
- Target: unlabeled ACSIncome table for CA (labels held out, used only for eval)

Expected results
----------------
- Baseline (equal-weight) suffers from negative transfer caused by dissimilar states
  (MS/WV/AR have very different income distributions from CA)
- Level 1 (discovery-weighted) outperforms the baseline
- Level 2 (+ pseudo-labeling + prior refinement) equals or beats Level 1

Run
---
    python act1_same_schema.py
"""

import logging
from pathlib import Path

import pandas as pd
from folktables import ACSDataSource, ACSIncome
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

import domain_adaptation
import evaluation
import schema_alignment
import table_discovery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_STATES      = ["MS", "WV", "AR", "WA", "OR"]
TARGET_STATE       = "CA"
LABEL_COL          = "label"
SURVEY_YEAR        = "2018"
SAMPLE_SIZE        = 5_000   # rows per state — keeps runtime short; set None for full
RANDOM_STATE       = 42
RESULTS_DIR        = Path("results/act1")
ENCODER_MODEL      = "all-MiniLM-L6-v2"
ORACLE_TEST_SIZE   = 0.2     # fraction of target held out as test set for all methods
DISTRIBUTION_WEIGHT = 0.5   # blend of schema vs distribution similarity in discovery
PERTURBED_STATES   = {"MS", "WV", "AR"}  # bad sources: mild schema perturbation applied

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _apply_schema_perturbation(df: pd.DataFrame, state: str, label_col: str) -> pd.DataFrame:
    """
    Apply mild schema perturbation to bad-source states so that column-name
    similarity scores spread apart from the good sources (WA/OR).

    The label column is never touched.
    Transformations mirror real-world messiness: renames, binning, drops.
    """
    df = df.copy()

    if state == "MS":
        if "WKHP" in df.columns:
            df = df.rename(columns={"WKHP": "weekly_hours"})
        if "AGEP" in df.columns:
            df["age_group"] = pd.cut(
                df["AGEP"], bins=[0, 25, 40, 60, 100], labels=False
            ).astype(float)
            df = df.drop(columns=["AGEP"])
        if "POBP" in df.columns:
            df = df.drop(columns=["POBP"])

    elif state == "WV":
        if "AGEP" in df.columns:
            df = df.rename(columns={"AGEP": "age_years"})
        if "WKHP" in df.columns:
            df["hours_bucket"] = pd.cut(
                df["WKHP"], bins=[0, 20, 35, 50, 99], labels=False
            ).astype(float)
            df = df.drop(columns=["WKHP"])
        if "RELP" in df.columns:
            df = df.drop(columns=["RELP"])

    elif state == "AR":
        if "RAC1P" in df.columns:
            df = df.rename(columns={"RAC1P": "race_code"})
        if "COW" in df.columns:
            df = df.drop(columns=["COW"])

    return df


def _load_state(
    data_source: ACSDataSource,
    state: str,
    sample_size: int | None,
) -> pd.DataFrame:
    raw = data_source.get_data(states=[state], download=True)
    features, labels, _ = ACSIncome.df_to_pandas(raw)

    df = features.copy()
    df[LABEL_COL] = labels.astype(int)   # 1 = income > $50k, 0 otherwise
    df = df.dropna().reset_index(drop=True)

    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_STATE)

    if state in PERTURBED_STATES:
        df = _apply_schema_perturbation(df, state, LABEL_COL)
        logger.info("Applied schema perturbation to %s", state)

    logger.info("Loaded %s: %d rows, %d features", state, len(df), df.shape[1] - 1)
    return df


def load_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return (lake, target_df).  target_df still contains the label column."""
    data_source = ACSDataSource(
        survey_year=SURVEY_YEAR, horizon="1-Year", survey="person"
    )

    lake: dict[str, pd.DataFrame] = {}
    for state in SOURCE_STATES:
        lake[state] = _load_state(data_source, state, SAMPLE_SIZE)

    target_df = _load_state(data_source, TARGET_STATE, SAMPLE_SIZE)
    return lake, target_df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_act1() -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- load ---
    lake, target_df = load_data()

    # Split target: oracle trains on train portion; all methods evaluated on test portion
    target_train_df, target_test_df = train_test_split(
        target_df, test_size=ORACLE_TEST_SIZE, random_state=RANDOM_STATE
    )
    y_true = target_test_df[LABEL_COL].values
    target_features = target_test_df.drop(columns=[LABEL_COL])
    logger.info(
        "Target split: %d oracle-train / %d test (%.0f%% / %.0f%%)",
        len(target_train_df), len(target_test_df),
        100 * (1 - ORACLE_TEST_SIZE), 100 * ORACLE_TEST_SIZE,
    )

    # load encoder once — shared by discovery + alignment
    logger.info("Loading encoder: %s", ENCODER_MODEL)
    model = SentenceTransformer(ENCODER_MODEL)

    # --- step 1: table discovery ---
    logger.info("=== Step 1: Table Discovery ===")
    # Strip label from lake tables before computing schema similarity
    lake_features = {k: v.drop(columns=[LABEL_COL]) for k, v in lake.items()}

    scores = table_discovery.discover_tables(
        lake=lake_features,
        target=target_features,
        model=model,
        distribution_weight=DISTRIBUTION_WEIGHT,
    )
    logger.info("Discovery scores:")
    for state, score in scores.items():
        logger.info("  %s → %.4f", state, score)

    pd.Series(scores, name="similarity").to_csv(RESULTS_DIR / "discovery_scores.csv")

    # --- step 2: schema alignment ---
    # Act 1: same schema, so alignment should produce identity mappings.
    # This is the sanity-check: alignment must not corrupt anything.
    logger.info("=== Step 2: Schema Alignment ===")
    aligned = schema_alignment.align_all(
        lake=lake,
        target=target_features,
        discovery_scores=scores,
        model=model,
        label_col=LABEL_COL,
    )

    # --- step 3: domain adaptation ---
    logger.info("=== Step 3: Domain Adaptation ===")
    source_labels = {k: lake[k][LABEL_COL] for k in scores}
    results = domain_adaptation.run_all(
        aligned=aligned,
        discovery_scores=scores,
        target=target_features,
        label_col=LABEL_COL,
        source_labels=source_labels,
    )
    results["oracle"] = domain_adaptation.run_oracle(
        target_train=target_train_df,
        target_test=target_features,
        label_col=LABEL_COL,
    )

    # --- step 4: evaluation ---
    logger.info("=== Step 4: Evaluation ===")
    metrics = evaluation.evaluate(results, y_true)
    summary = evaluation.summarise(metrics)

    print("\n" + "=" * 60)
    print("ACT 1 RESULTS — Same Schema, Distribution Shift Only")
    print("=" * 60)
    print(summary.to_string())
    print()
    _print_interpretation(summary)

    summary.to_csv(RESULTS_DIR / "metrics.csv")
    logger.info("Results saved to %s", RESULTS_DIR)
    return summary


def _print_interpretation(summary: pd.DataFrame) -> None:
    """Print a human-readable verdict on whether the experiment succeeded."""
    if "accuracy" not in summary.columns or "accuracy_delta" not in summary.columns:
        return

    baseline_acc  = summary.loc["baseline", "accuracy"]
    l1_delta      = summary.loc["level1",  "accuracy_delta"] if "level1"  in summary.index else None
    l2_delta      = summary.loc["level2",  "accuracy_delta"] if "level2"  in summary.index else None
    oracle_acc    = summary.loc["oracle",  "accuracy"]        if "oracle"  in summary.index else None
    oracle_delta  = summary.loc["oracle",  "accuracy_delta"]  if "oracle"  in summary.index else None

    print("Interpretation")
    print("-" * 40)
    print(f"  Baseline accuracy : {baseline_acc:.4f}")

    if l1_delta is not None:
        verdict = "BETTER (negative transfer prevented)" if l1_delta > 0.001 else (
                  "no change" if abs(l1_delta) <= 0.001 else "WORSE — check discovery scores")
        print(f"  Level 1 delta     : {l1_delta:+.4f}  →  {verdict}")

    if l2_delta is not None:
        verdict = "BETTER" if l2_delta > 0.001 else "no gain from pseudo-labeling"
        print(f"  Level 2 delta     : {l2_delta:+.4f}  →  {verdict}")

    if oracle_acc is not None:
        gap = oracle_acc - baseline_acc
        print(f"  Oracle accuracy   : {oracle_acc:.4f}  (gap vs baseline: {gap:+.4f})")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_act1()
