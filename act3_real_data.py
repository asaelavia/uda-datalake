"""
Act 3 — Real Open Data (UCI Adult target + Folktables lake)

Setup
-----
Target  : UCI Adult 1994  (income >$50k binary classification)
          A real external dataset — completely separate from the lake.

Lake    :
  CA, WA, OR  — Folktables ACSIncome 2018 (relevant: same income task, genuine
                  schema gap between census codes and Adult column names)
  bank        — Bank Marketing (different task, different schema — noisy source)

Schema gap
----------
Folktables columns (renamed from census codes):
  age, class_of_worker, education_level, marital_status, occupation_code,
  place_of_birth, relationship_status, hours_worked_weekly, sex, race_code

UCI Adult columns (target):
  age, workclass, fnlwgt, education, education-num, marital-status,
  occupation, relationship, race, sex, capital-gain, capital-loss,
  hours-per-week, native-country

Schema alignment must bridge e.g.:
  education_level → education,  marital_status → marital-status,
  hours_worked_weekly → hours-per-week,  class_of_worker → workclass

Value alignment
---------------
After column-name alignment, numeric value ranges still differ:
Folktables uses census integer codes; Adult uses label-encoded strings.
A post-alignment min-max normalisation (scaled to the target's range) makes
cross-dataset tree splits comparable.

Label distribution note
-----------------------
Folktables 2018 positive rate ~40% vs Adult 1994 ~24%.
This distribution shift limits direct transfer (Level 0/1) but
pseudo-labeling (Level 2) adapts to the true target distribution.

Run
---
    python act3_real_data.py
"""

import logging
from pathlib import Path

import pandas as pd
from folktables import ACSDataSource, ACSIncome
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

LABEL_COL           = "label"
LAKE_STATES         = ["CA", "WA", "OR"]
SURVEY_YEAR         = "2018"
SAMPLE_SIZE         = 5_000
RANDOM_STATE        = 42
RESULTS_DIR         = Path("results/act3")
ENCODER_MODEL       = "all-MiniLM-L6-v2"
ORACLE_TEST_SIZE    = 0.2
DISTRIBUTION_WEIGHT = 0.5

# Census code → descriptive names.  Intentionally NOT identical to Adult's
# column names so schema alignment has real bridging work to do.
_FOLKTABLES_RENAME = {
    "AGEP": "age",
    "COW":  "class_of_worker",
    "SCHL": "education_level",
    "MAR":  "marital_status",
    "OCCP": "occupation_code",
    "POBP": "place_of_birth",
    "RELP": "relationship_status",
    "WKHP": "hours_worked_weekly",
    "SEX":  "sex",
    "RAC1P":"race_code",
}

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def _sample(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    if n and len(df) > n:
        return df.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)
    return df.reset_index(drop=True)


def load_folktables_state(data_source: ACSDataSource, state: str) -> pd.DataFrame:
    raw = data_source.get_data(states=[state], download=True)
    features, labels, _ = ACSIncome.df_to_pandas(raw)
    df = features.rename(columns=_FOLKTABLES_RENAME).copy()
    df[LABEL_COL] = labels.astype(int)
    df = df.dropna().reset_index(drop=True)
    df = _sample(df, SAMPLE_SIZE)
    logger.info("Folktables %s: %d rows, positive_rate=%.3f",
                state, len(df), float(df[LABEL_COL].mean()))
    return df


def load_adult() -> pd.DataFrame:
    """UCI Adult 1994 Census Income (OpenML id=1590).  Label: income >50K → 1."""
    logger.info("Downloading UCI Adult (OpenML id=1590) …")
    data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df = data.frame.copy()

    label_col = data.target_names[0] if hasattr(data, "target_names") else "class"
    raw = df.pop(label_col).astype(str).str.strip()
    df[LABEL_COL] = (raw.str.startswith(">")).astype(int)

    df = _encode_categoricals(df)
    df = df.dropna().reset_index(drop=True)
    logger.info("UCI Adult: %d rows, positive_rate=%.3f, cols=%s",
                len(df), float(df[LABEL_COL].mean()),
                list(df.drop(columns=[LABEL_COL]).columns))
    return df


def load_bank() -> pd.DataFrame:
    """Bank Marketing (OpenML id=1461).  Noisy / low-relevance lake table."""
    logger.info("Downloading Bank Marketing (OpenML id=1461) …")
    data = fetch_openml(data_id=1461, as_frame=True, parser="auto")
    df = data.frame.copy()

    target_name = data.target_names[0] if hasattr(data, "target_names") else None
    if target_name and target_name in df.columns:
        raw = df.pop(target_name).astype(str).str.strip()
    else:
        raw = df.iloc[:, -1].astype(str).str.strip()
        df = df.iloc[:, :-1].copy()

    df[LABEL_COL] = (raw.isin(["2", "yes", "Yes"])).astype(int)
    df = _encode_categoricals(df)
    df = df.dropna().reset_index(drop=True)
    logger.info("Bank Marketing: %d rows, positive_rate=%.3f",
                len(df), float(df[LABEL_COL].mean()))
    return df


# ---------------------------------------------------------------------------
# Value alignment
# ---------------------------------------------------------------------------

def _make_norm_stats(target: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Per-column min/max from the target, for numeric columns with non-zero range."""
    return {
        col: (float(target[col].min()), float(target[col].max()))
        for col in target.select_dtypes("number").columns
        if target[col].max() > target[col].min()
    }


def _apply_norm(df: pd.DataFrame, stats: dict[str, tuple[float, float]]) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in stats.items():
        if col in df.columns:
            df[col] = (df[col] - lo) / (hi - lo)
    return df


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def load_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return (lake, target_df).  target_df contains the label column."""
    data_source = ACSDataSource(survey_year=SURVEY_YEAR, horizon="1-Year", survey="person")

    lake: dict[str, pd.DataFrame] = {}
    for state in LAKE_STATES:
        lake[state] = load_folktables_state(data_source, state)

    lake["bank"] = _sample(load_bank(), SAMPLE_SIZE)

    target_df = load_adult()
    logger.info("Lake sizes: %s", {k: len(v) for k, v in lake.items()})
    return lake, target_df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_act3() -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lake, target_df = load_data()

    target_train_df, target_test_df = train_test_split(
        target_df, test_size=ORACLE_TEST_SIZE, random_state=RANDOM_STATE,
        stratify=target_df[LABEL_COL],
    )
    y_true          = target_test_df[LABEL_COL].values
    target_features = target_test_df.drop(columns=[LABEL_COL])
    logger.info(
        "Target split: %d oracle-train / %d test (%.0f%% / %.0f%%)",
        len(target_train_df), len(target_test_df),
        100 * (1 - ORACLE_TEST_SIZE), 100 * ORACLE_TEST_SIZE,
    )

    logger.info("Loading encoder: %s", ENCODER_MODEL)
    encoder = SentenceTransformer(ENCODER_MODEL)

    # --- step 1: table discovery ---
    logger.info("=== Step 1: Table Discovery ===")
    lake_features = {k: v.drop(columns=[LABEL_COL]) for k, v in lake.items()}
    scores = table_discovery.discover_tables(
        lake=lake_features,
        target=target_features,
        model=encoder,
        distribution_weight=DISTRIBUTION_WEIGHT,
    )
    logger.info("Discovery scores:")
    for name, score in scores.items():
        logger.info("  %-12s → %.4f", name, score)

    pd.Series(scores, name="similarity").to_csv(RESULTS_DIR / "discovery_scores.csv")

    # --- step 2: schema alignment ---
    logger.info("=== Step 2: Schema Alignment ===")
    aligned = schema_alignment.align_all(
        lake=lake,
        target=target_features,
        discovery_scores=scores,
        model=encoder,
        label_col=LABEL_COL,
    )

    # --- value alignment ---
    # After schema alignment, column names match but numeric encodings differ
    # (Folktables census codes vs Adult label-encoded strings).  Normalise all
    # sources and target to [0,1] using the target's own value range so that
    # cross-dataset tree splits are meaningful.
    norm_stats = _make_norm_stats(target_features)
    aligned    = {k: _apply_norm(v, norm_stats) for k, v in aligned.items()}
    target_features   = _apply_norm(target_features,   norm_stats)
    target_train_norm = _apply_norm(
        target_train_df.drop(columns=[LABEL_COL]), norm_stats
    )
    target_train_norm[LABEL_COL] = target_train_df[LABEL_COL].values

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
        target_train=target_train_norm,
        target_test=target_features,
        label_col=LABEL_COL,
    )

    # --- step 4: evaluation ---
    logger.info("=== Step 4: Evaluation ===")
    target_pos_rate = float(y_true.mean())
    logger.info("Target positive rate (used for threshold calibration): %.3f", target_pos_rate)

    metrics_raw  = evaluation.evaluate(results, y_true)
    metrics_cal  = evaluation.evaluate(results, y_true, target_pos_rate=target_pos_rate)
    summary_raw  = evaluation.summarise(metrics_raw)
    summary_cal  = evaluation.summarise(metrics_cal)

    print("\n" + "=" * 60)
    print("ACT 3 RESULTS — Real Open Data")
    print("  Target : UCI Adult 1994  (income >$50k)")
    print("  Lake   : Folktables CA/WA/OR 2018  +  Bank Marketing")
    print("=" * 60)
    print("\n--- Default threshold (0.5) ---")
    print(summary_raw.to_string())
    print("\n--- Calibrated threshold (matched to target positive rate) ---")
    print(summary_cal.to_string())
    print()
    _print_interpretation(summary_cal)

    summary_cal.to_csv(RESULTS_DIR / "metrics.csv")
    summary_raw.to_csv(RESULTS_DIR / "metrics_uncalibrated.csv")
    logger.info("Results saved to %s", RESULTS_DIR)
    return summary


def _print_interpretation(summary: pd.DataFrame) -> None:
    if "accuracy" not in summary.columns:
        return

    baseline_acc = summary.loc["baseline", "accuracy"]
    l1_delta     = summary.loc["level1",   "accuracy_delta"] if "level1"  in summary.index else None
    l2_delta     = summary.loc["level2",   "accuracy_delta"] if "level2"  in summary.index else None
    oracle_acc   = summary.loc["oracle",   "accuracy"]        if "oracle"  in summary.index else None

    print("Interpretation")
    print("-" * 40)
    print(f"  Baseline accuracy : {baseline_acc:.4f}")
    print(f"  (Note: Folktables ~40% positive rate vs Adult ~24% — label shift expected)")

    if l1_delta is not None:
        verdict = ("BETTER — discovery weighting reduced label-shift impact"
                   if l1_delta > 0.001 else
                   "no change" if abs(l1_delta) <= 0.001 else
                   "WORSE — label distribution shift dominates")
        print(f"  Level 1 delta     : {l1_delta:+.4f}  →  {verdict}")

    if l2_delta is not None:
        verdict = "BETTER — pseudo-labeling adapts to target distribution" if l2_delta > 0.001 else "no gain from pseudo-labeling"
        print(f"  Level 2 delta     : {l2_delta:+.4f}  →  {verdict}")

    if oracle_acc is not None:
        print(f"  Oracle accuracy   : {oracle_acc:.4f}  (gap vs baseline: {oracle_acc - baseline_acc:+.4f})")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_act3()
