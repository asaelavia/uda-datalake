"""
Act 4 — OpenML as a Real Data Lake

A large, heterogeneous lake of OpenML binary classification datasets.
The pipeline must discover which ones are useful for predicting the target
automatically — no hand-picking.

Supported targets
-----------------
  adult   — UCI Adult 1994 census income >$50k  (default)
  nyhouse — NY Housing price above $1M

Key insight: the label column name encodes the task. Discovery uses label-name
similarity (weight 0.4) on top of schema + distribution similarity.

Run
---
    # Step 1 — download (run once, resumable):
    python act4_openml_lake.py --download-only

    # Step 2 — experiment (adult target, 200 lake tables):
    python act4_openml_lake.py --lake-size 200

    # Step 3 — NY Housing target:
    python act4_openml_lake.py --target nyhouse --lake-size 200
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LABEL_COL           = "label"
ADULT_DID           = 1590      # UCI Adult — excluded from lake when target=adult
BANK_DID            = 1461      # Bank Marketing — excluded when target=bank
DIABETES_DID        = 37        # Pima Indians Diabetes — excluded when target=diabetes
CREDIT_DID          = 31        # German Credit — excluded when target=credit
CHURN_DID           = 42178     # IBM Telco Customer Churn
HEART_DID           = 53        # Cleveland Heart Disease (heart-statlog)
TURNOVER_DID        = 43551     # Employee Turnover at TECHCO
CRIME_DID           = 43891     # Communities and Crime (violent crime rate > 20%)
TITANIC_DID         = 40945     # Titanic passenger survival
BREASTCANCER_DID    = 15        # Wisconsin Breast Cancer (malignant/benign) — named features
DATA_DIR            = Path("data/act4")
MANIFEST_PATH       = DATA_DIR / "manifest.json"
ENCODER_MODEL       = "all-MiniLM-L6-v2"
RANDOM_STATE        = 42
ORACLE_TEST_SIZE    = 0.2
DISTRIBUTION_WEIGHT = 0.3
LABEL_WEIGHT        = 0.4
BALANCE_WEIGHT      = 0.15      # label-balance penalty in discovery
DIST_THRESHOLD      = 3.0       # drop aligned column pairs with quantile dist > this
WEIGHT_POWER        = 1.0       # exponent on discovery scores in Level 1/2 pooling
#                                  (>1 sharpens contrast; set >1 only when the top-ranked
#                                  source has a well-aligned label; 1.0 = original linear)
DANN_EPOCHS         = 200       # training epochs for Level 5 (DANN); requires: pip install torch
TOP_K               = 20        # tables selected for adaptation (overridden by --dynamic-top-k)
MAX_DISK_GB         = 3.0
MIN_INSTANCES       = 500
MAX_INSTANCES       = 50_000    # skip huge datasets (openml downloads full file before we can cap)
MIN_FEATURES        = 4         # feature columns, not counting the label
MAX_FEATURES        = 60        # skip wide datasets that bloat CSVs
MAX_ROWS_PER_DS     = 20_000    # subsample rows within the CSV to save disk
DEFAULT_LAKE_SIZE   = 200

# Dataset names to exclude from the lake regardless of target
# (prevents the lake from containing copies/variants of the target itself)
_EXCLUDE_NAME_PATTERNS = [
    r"\badult\b",           # UCI Adult / census income variants
    r"\bcensus\b",          # census income datasets
    r"credit",              # German Credit / credit card variants (substring: catches dataset_credit-g where _ precedes c)
    r"\bcreditability\b",
    r"\bdiabetes\b",        # Pima / diabetes variants
    r"\bpima\b",
    r"\bbank\b",            # Bank Marketing variants
    r"\bmarketing\b",
    r"\bfraud\b",           # Fraud detection (unrelated to most targets)
]


@dataclass
class TargetConfig:
    name: str
    label_name: str        # descriptive task name for embedding
    results_dir: Path
    exclude_dids: set[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.exclude_dids is None:
            self.exclude_dids = set()


_TARGETS: dict[str, TargetConfig] = {
    "adult": TargetConfig(
        name="adult",
        label_name="income above 50k binary classification",
        results_dir=Path("results/act4/adult"),
        exclude_dids={ADULT_DID, 6, 179, 2119, 4136, 44722, 44723, 44724, 44725, 44726, 44727},
        # ^^ known UCI Adult / census income variants on OpenML
    ),
    "nyhouse": TargetConfig(
        name="nyhouse",
        label_name="house price above 1 million dollars real estate",
        results_dir=Path("results/act4/nyhouse"),
        exclude_dids=set(),
    ),
    "bank": TargetConfig(
        name="bank",
        label_name="bank term deposit subscription telemarketing",
        results_dir=Path("results/act4/bank"),
        exclude_dids={BANK_DID},
    ),
    "diabetes": TargetConfig(
        name="diabetes",
        label_name="diabetes positive glucose blood sugar",
        results_dir=Path("results/act4/diabetes"),
        exclude_dids={DIABETES_DID, 469, 46921},
        # ^^ 469 = pima-indians-diabetes variant; 46921 = glucose dataset (too close)
    ),
    "credit": TargetConfig(
        name="credit",
        label_name="good credit risk loan repayment",
        results_dir=Path("results/act4/credit"),
        exclude_dids={CREDIT_DID, 46, 11785, 46562, 46918},
        # ^^ known German Credit variants on OpenML
    ),
    "churn": TargetConfig(
        name="churn",
        label_name="customer churn cancelled subscription",
        results_dir=Path("results/act4/churn"),
        exclude_dids={CHURN_DID},
    ),
    "heart": TargetConfig(
        name="heart",
        label_name="heart disease cardiovascular diagnosis",
        results_dir=Path("results/act4/heart"),
        exclude_dids={HEART_DID, 1497, 1565},
        # ^^ 1497 = heart-c; 1565 = heart-h (Cleveland variants on OpenML)
    ),
    "turnover": TargetConfig(
        name="turnover",
        label_name="employee turnover attrition resigned",
        results_dir=Path("results/act4/turnover"),
        exclude_dids={TURNOVER_DID},
    ),
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def _is_human_readable(columns: list[str]) -> bool:
    """Return True if ≤50% of column names are purely numeric or single-character."""
    if not columns:
        return False
    bad = 0
    for c in columns:
        stripped = str(c).strip()
        if len(stripped) <= 1:
            bad += 1
            continue
        try:
            float(stripped)
            bad += 1
        except ValueError:
            pass
    return (bad / len(columns)) <= 0.5


def _make_quantile_normalizer(
    target: pd.DataFrame,
) -> tuple["QuantileTransformer", list[str]]:
    """
    Fit a per-column quantile transformer on the target's numeric columns.

    Quantile normalization maps the entire CDF of each source column to match
    the target distribution, not just the range.  After transformation every
    column in source and target has the same marginal distribution (uniform
    [0, 1]), which is much stronger than min-max scaling and removes the most
    common form of covariate shift.

    Returns the fitted transformer and the list of columns it covers.
    """
    from sklearn.preprocessing import QuantileTransformer

    num_cols = [
        c for c in target.select_dtypes("number").columns
        if target[c].nunique() > 1
    ]
    n_quantiles = min(1000, max(10, len(target)))
    qt = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="uniform",
        random_state=42,
    )
    qt.fit(target[num_cols])
    return qt, num_cols


def _apply_quantile_norm(
    df: pd.DataFrame,
    qt: "QuantileTransformer",
    num_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()
    cols_present = [c for c in num_cols if c in df.columns]
    if not cols_present:
        return df
    # Impute NaN with column median before transforming (QuantileTransformer
    # does not accept NaN); NaN cells are restored after transformation.
    sub = df[cols_present].copy()
    nan_mask = sub.isna()
    for col in cols_present:
        med = float(sub[col].median())
        sub[col] = sub[col].fillna(med if not np.isnan(med) else 0.0)
    sub_t = qt.transform(sub)
    sub_df = pd.DataFrame(sub_t, columns=cols_present, index=df.index)
    sub_df[nan_mask] = np.nan          # restore NaN where they were
    df[cols_present] = sub_df
    return df


def _repurpose_lake_tables(
    lake: dict[str, pd.DataFrame],
    label_names: dict[str, str],
    repurpose_map: dict[str, str],
    label_col: str = LABEL_COL,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """
    For each table_id in repurpose_map, swap the nominated feature column in
    as the new label:
      1. Fill NaN in repurpose_col with its median.
      2. Binarize at median (>= median → 1).
      3. Skip if resulting column has zero variance.
      4. Drop original label_col.
      5. Rename repurpose_col → label_col.
      6. Update label_names[table_id] to the repurposed column name.

    Returns modified *copies* of lake and label_names.
    """
    lake_out = dict(lake)
    names_out = dict(label_names)

    for table_id, repurpose_col in repurpose_map.items():
        if table_id not in lake_out or repurpose_col not in lake_out[table_id].columns:
            logger.warning("Repurpose: table '%s' or col '%s' not found, skipping.",
                           table_id, repurpose_col)
            continue

        df = lake_out[table_id].copy()
        col_median = df[repurpose_col].median()
        df[repurpose_col] = df[repurpose_col].fillna(col_median)
        df[repurpose_col] = (df[repurpose_col] >= col_median).astype(int)

        if df[repurpose_col].nunique() < 2:
            logger.warning("Repurpose: '%s' col '%s' has zero variance — skipping.",
                           table_id, repurpose_col)
            continue

        if label_col in df.columns:
            df = df.drop(columns=[label_col])
        df = df.rename(columns={repurpose_col: label_col})
        names_out[table_id] = repurpose_col
        lake_out[table_id] = df

        logger.info("Repurposed '%s': feature '%s' (median=%.4g) → new label "
                    "(pos_rate=%.3f)", table_id, repurpose_col, col_median,
                    float(df[label_col].mean()))

    return lake_out, names_out


# ---------------------------------------------------------------------------
# OpenML lake construction
# ---------------------------------------------------------------------------

def _list_candidate_datasets() -> pd.DataFrame:
    """Query OpenML and return a filtered, size-sorted list of binary classification datasets."""
    import openml  # imported here so the package is optional at module level

    logger.info("Fetching OpenML dataset list …")
    all_ds = openml.datasets.list_datasets(output_format="dataframe")
    logger.info("Total OpenML datasets: %d", len(all_ds))

    mask = (
        (all_ds["NumberOfClasses"] == 2)
        & (all_ds["NumberOfInstances"] >= MIN_INSTANCES)
        & (all_ds["NumberOfInstances"] <= MAX_INSTANCES)
        & (all_ds["NumberOfFeatures"] >= MIN_FEATURES + 1)   # +1 for label
        & (all_ds["NumberOfFeatures"] <= MAX_FEATURES + 1)   # +1 for label
    )
    if "status" in all_ds.columns:
        mask = mask & (all_ds["status"] == "active")

    candidates = all_ds[mask].copy()
    # Exclude the Adult OpenML dataset by DID
    candidates = candidates[candidates["did"] != ADULT_DID]
    # Exclude datasets whose name matches any exclusion pattern (e.g. adult variants)
    if "name" in candidates.columns:
        import re
        pattern = "|".join(_EXCLUDE_NAME_PATTERNS)
        candidates = candidates[
            ~candidates["name"].str.lower().str.contains(pattern, na=False, regex=True)
        ]
    candidates = candidates.sort_values("NumberOfInstances", ascending=False)
    logger.info(
        "Candidate datasets (binary, ≥%d rows, ≥%d features, name-filtered): %d",
        MIN_INSTANCES, MIN_FEATURES, len(candidates),
    )
    return candidates.reset_index(drop=True)


def _download_one(did: int, data_dir: Path) -> Optional[dict]:
    """
    Download one OpenML dataset, clean it, write a CSV, return a manifest entry.
    Returns None if the dataset cannot be used.
    """
    import openml

    try:
        dataset = openml.datasets.get_dataset(
            did,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=False,
        )
        target_attr = dataset.default_target_attribute
        if not target_attr:
            logger.debug("Dataset %d: no default target attribute, skipping.", did)
            return None

        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=target_attr,
        )
        if X is None or y is None or len(X) == 0:
            return None

        # Check feature column readability
        if not _is_human_readable(list(X.columns)):
            logger.debug("Dataset %d: non-human-readable columns, skipping.", did)
            return None

        # Drop rows where ALL features are NaN; cap to MAX_ROWS_PER_DS
        df = X.dropna(how="all").copy()
        if len(df) > MAX_ROWS_PER_DS:
            df = df.sample(MAX_ROWS_PER_DS, random_state=42)
        df = _encode_categoricals(df)

        # Align y index after dropna and encode to 0/1
        y_aligned = y.loc[df.index]
        df[LABEL_COL] = LabelEncoder().fit_transform(y_aligned.astype(str))

        out_path = data_dir / f"dataset_{did}.csv"
        df.to_csv(out_path, index=False)

        size_bytes = out_path.stat().st_size
        return {
            "did": int(did),
            "name": dataset.name,
            "n_instances": int(len(df)),
            "n_features": int(len(X.columns)),
            "label_name": str(target_attr),
            "path": out_path.name,
            "size_bytes": int(size_bytes),
        }
    except Exception as exc:
        logger.warning("Dataset %d: download failed — %s", did, exc)
        return None


def download_lake(
    candidates: pd.DataFrame,
    data_dir: Path,
    max_gb: float = MAX_DISK_GB,
) -> None:
    """
    Download all qualifying datasets to data_dir.

    Resumable: reads the existing manifest and skips already-cached datasets.
    Stops when cumulative disk usage reaches max_gb.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        existing_dids = {entry["did"] for entry in manifest["datasets"]}
        total_bytes = sum(entry["size_bytes"] for entry in manifest["datasets"])
        logger.info(
            "Resuming download: %d datasets already cached (%.2f GB)",
            len(existing_dids), total_bytes / 1e9,
        )
    else:
        manifest = {"version": 1, "datasets": []}
        existing_dids: set[int] = set()
        total_bytes = 0

    max_bytes = max_gb * 1e9

    for _, row in candidates.iterrows():
        did = int(row["did"])
        if did in existing_dids:
            continue
        if total_bytes >= max_bytes:
            logger.info("Disk budget (%.1f GB) reached — stopping download.", max_gb)
            break

        logger.info(
            "Downloading dataset %d (%s, %d rows) …",
            did, row.get("name", "?"), int(row["NumberOfInstances"]),
        )
        entry = _download_one(did, data_dir)
        if entry is None:
            continue

        manifest["datasets"].append(entry)
        total_bytes += entry["size_bytes"]
        existing_dids.add(did)

        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "  → %s  (%.2f GB cumulative)", entry["path"], total_bytes / 1e9,
        )
        time.sleep(0.05)

    logger.info(
        "Download complete: %d datasets, %.2f GB",
        len(manifest["datasets"]), total_bytes / 1e9,
    )


# ---------------------------------------------------------------------------
# Cache loader
# ---------------------------------------------------------------------------

def _load_lake_from_cache(
    data_dir: Path,
    manifest_path: Path,
    top_k: int,
    exclude_dids: Optional[set[int]] = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """
    Load the top_k datasets (by n_instances) from cache.

    Returns
    -------
    lake : dict[table_id → DataFrame]   (includes LABEL_COL)
    label_names : dict[table_id → original OpenML target attribute name]
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Run with --download-only first."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)

    import re
    _name_re = re.compile("|".join(_EXCLUDE_NAME_PATTERNS), re.IGNORECASE)

    entries = manifest["datasets"]
    if exclude_dids:
        entries = [e for e in entries if e["did"] not in exclude_dids]
    # Apply name patterns at load time — guards against cached manifests that
    # predate pattern additions (the candidate-fetch step also filters, but the
    # manifest is persisted and may contain stale entries).
    entries = [e for e in entries if not _name_re.search(e.get("name", ""))]

    entries = sorted(entries, key=lambda e: e["n_instances"], reverse=True)[:top_k]

    lake: dict[str, pd.DataFrame] = {}
    label_names: dict[str, str] = {}

    for entry in entries:
        path = data_dir / entry["path"]
        if not path.exists():
            logger.warning("Missing cache file %s, skipping.", path)
            continue
        try:
            df = pd.read_csv(path)
            if LABEL_COL not in df.columns:
                logger.warning("Dataset %d missing label column, skipping.", entry["did"])
                continue
            table_id = f"openml_{entry['did']}"
            lake[table_id] = df
            label_names[table_id] = entry["label_name"]
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)

    logger.info("Loaded %d lake tables (top-%d by size)", len(lake), top_k)
    return lake, label_names


# ---------------------------------------------------------------------------
# Target loaders
# ---------------------------------------------------------------------------

def _load_adult_target() -> pd.DataFrame:
    """UCI Adult 1994 Census Income (OpenML id=1590). Label: income >50K → 1."""
    logger.info("Loading UCI Adult (OpenML id=%d) …", ADULT_DID)
    data = fetch_openml(data_id=ADULT_DID, as_frame=True, parser="auto")
    df = data.frame.copy()

    label_col_raw = data.target_names[0] if hasattr(data, "target_names") else "class"
    raw = df.pop(label_col_raw).astype(str).str.strip()
    df[LABEL_COL] = (raw.str.startswith(">")).astype(int)

    df = _encode_categoricals(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(
        "UCI Adult: %d rows, positive_rate=%.3f, cols=%s",
        len(df), float(df[LABEL_COL].mean()),
        list(df.drop(columns=[LABEL_COL]).columns),
    )
    return df


def _load_nyhouse_target() -> pd.DataFrame:
    """NY Housing: price above $1M binary classification. Label already 0/1."""
    path = Path("data/NY-Housing/nyhouse.csv")
    logger.info("Loading NY Housing dataset from %s …", path)
    df = pd.read_csv(path)
    if "price_above_1M" in df.columns:
        df = df.rename(columns={"price_above_1M": LABEL_COL})
    df = _encode_categoricals(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(
        "NY Housing: %d rows, positive_rate=%.3f, cols=%s",
        len(df), float(df[LABEL_COL].mean()),
        list(df.drop(columns=[LABEL_COL]).columns),
    )
    return df


def _load_openml_target(did: int, name: str, positive_values: Optional[set[str]] = None) -> pd.DataFrame:
    """
    Generic OpenML target loader.

    Parameters
    ----------
    did:
        OpenML dataset id.
    name:
        Human-readable name for log messages.
    positive_values:
        Set of raw label strings that map to 1.  When None, the minority class
        is used as the positive class (robust to unknown label encodings).
    """
    logger.info("Loading %s (OpenML id=%d) ...", name, did)
    data = fetch_openml(data_id=did, as_frame=True, parser="auto")
    df = data.frame.copy()

    label_col_raw = data.target_names[0] if hasattr(data, "target_names") else "class"
    raw = df.pop(label_col_raw).astype(str).str.strip().str.lower()

    if positive_values is not None:
        df[LABEL_COL] = raw.isin({v.lower() for v in positive_values}).astype(int)
    else:
        # Minority class → positive (1)
        le = LabelEncoder()
        encoded = le.fit_transform(raw)
        counts = np.bincount(encoded)
        minority_idx = int(np.argmin(counts))
        df[LABEL_COL] = (encoded == minority_idx).astype(int)

    df = _encode_categoricals(df)
    # Drop columns where >30% of rows are missing before dropping NaN rows,
    # to avoid losing most of the dataset due to a few sparse columns.
    thresh = int(0.7 * len(df))
    df = df.dropna(axis=1, thresh=thresh)
    df = df.dropna().reset_index(drop=True)
    logger.info(
        "%s: %d rows, positive_rate=%.3f, cols=%s",
        name, len(df), float(df[LABEL_COL].mean()),
        list(df.drop(columns=[LABEL_COL]).columns),
    )
    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_act4(
    lake_size: int = DEFAULT_LAKE_SIZE,
    download_only: bool = False,
    target_name: str = "adult",
    dynamic_top_k: bool = False,
    top_k: int = TOP_K,
    dann_epochs: int = DANN_EPOCHS,
) -> Optional[pd.DataFrame]:
    cfg = _TARGETS[target_name]
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Ensure lake cache exists ---
    candidates = _list_candidate_datasets()
    if not MANIFEST_PATH.exists() or download_only:
        download_lake(candidates, DATA_DIR)

    if download_only:
        logger.info("Download complete. Re-run without --download-only to run the experiment.")
        return None

    # --- Load lake ---
    logger.info("Loading lake (top-%d by size) …", lake_size)
    lake, label_names = _load_lake_from_cache(
        DATA_DIR, MANIFEST_PATH, top_k=lake_size, exclude_dids=cfg.exclude_dids,
    )
    if not lake:
        raise RuntimeError(
            "No lake tables loaded — run `python act4_openml_lake.py --download-only` first."
        )

    # --- Load target ---
    if target_name == "adult":
        target_df = _load_adult_target()
    elif target_name == "nyhouse":
        target_df = _load_nyhouse_target()
    elif target_name == "bank":
        target_df = _load_openml_target(BANK_DID, "Bank Marketing", positive_values={"2", "yes"})
    elif target_name == "diabetes":
        target_df = _load_openml_target(DIABETES_DID, "Pima Diabetes", positive_values={"tested_positive", "1", "pos"})
    elif target_name == "credit":
        target_df = _load_openml_target(CREDIT_DID, "German Credit", positive_values={"good", "1"})
    elif target_name == "churn":
        target_df = _load_openml_target(CHURN_DID, "Telco Churn", positive_values={"yes", "1", "true"})
    elif target_name == "heart":
        target_df = _load_openml_target(HEART_DID, "Heart Disease", positive_values={"present", "1", "2"})
    elif target_name == "turnover":
        target_df = _load_openml_target(TURNOVER_DID, "Employee Turnover", positive_values={"left", "1", "true"})
    else:
        raise ValueError(f"Unknown target: {target_name!r}. Choose from: {list(_TARGETS)}")

    target_train_df, target_test_df = train_test_split(
        target_df,
        test_size=ORACLE_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target_df[LABEL_COL],
    )
    y_true          = target_test_df[LABEL_COL].values
    target_features = target_test_df.drop(columns=[LABEL_COL])
    logger.info(
        "Target split: %d oracle-train / %d test",
        len(target_train_df), len(target_test_df),
    )

    logger.info("Loading encoder: %s", ENCODER_MODEL)
    encoder = SentenceTransformer(ENCODER_MODEL)

    # --- Task-aware source repurposing ---
    if cfg.label_name:
        logger.info("=== Source Repurposing: scanning for task-aligned features ===")
        repurpose_map = table_discovery.find_repurposable_features(
            lake={k: v.drop(columns=[LABEL_COL]) for k, v in lake.items()},
            target_label_name=cfg.label_name,
            model=encoder,
            label_col=LABEL_COL,
            threshold=0.6,
            use_kg_expansion=True,
        )
        if repurpose_map:
            logger.info("Found %d repurposable tables:", len(repurpose_map))
            for tid, col in repurpose_map.items():
                logger.info("  %s  →  feature '%s' as new label", tid, col)
            lake, label_names = _repurpose_lake_tables(lake, label_names, repurpose_map)
        else:
            logger.info("No repurposable tables found (threshold=0.6)")

    # --- Step 1: Table Discovery (label-aware, balance-penalised) ---
    logger.info("=== Step 1: Table Discovery ===")
    lake_features = {k: v.drop(columns=[LABEL_COL]) for k, v in lake.items()}
    source_pos_rates = {k: float(lake[k][LABEL_COL].mean()) for k in lake}
    target_pos_rate = float(target_df[LABEL_COL].mean())
    logger.info("Target positive rate: %.3f", target_pos_rate)
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

    logger.info("Discovery scores (top 20):")
    for tbl, score in list(scores.items())[:20]:
        logger.info("  %-35s  score=%.4f  label='%s'", tbl, score, label_names.get(tbl, "?"))

    if dynamic_top_k:
        score_vals = list(scores.values())
        threshold = max(score_vals) * 0.5
        candidates_dyn = {k: v for k, v in scores.items() if v >= threshold}
        n_dyn = max(3, min(len(candidates_dyn), top_k))
        top_k_scores = dict(list(scores.items())[:n_dyn])
        logger.info("Dynamic TOP_K=%d (threshold=%.4f, ≥50%% of max score %.4f)",
                    n_dyn, threshold, max(score_vals))
    else:
        top_k_scores = dict(list(scores.items())[:top_k])

    logger.info("Selected top-%d tables for adaptation:", len(top_k_scores))
    for tbl, score in top_k_scores.items():
        logger.info("  %-35s  %.4f  label='%s'", tbl, score, label_names.get(tbl, "?"))

    pd.Series(scores, name="similarity").to_csv(cfg.results_dir / "discovery_scores.csv")

    # --- Step 2: Schema Alignment ---
    logger.info("=== Step 2: Schema Alignment ===")
    lake_top_k = {k: lake[k] for k in top_k_scores}
    aligned = schema_alignment.align_all(
        lake=lake_top_k,
        target=target_features,
        discovery_scores=top_k_scores,
        model=encoder,
        label_col=LABEL_COL,
        dist_threshold=DIST_THRESHOLD,
    )

    # Value normalisation — quantile transform fitted on unlabeled target features.
    # Maps each column's full CDF to Uniform[0,1], removing marginal covariate
    # shift far more aggressively than min-max scaling.
    qt, num_cols = _make_quantile_normalizer(target_features)
    aligned              = {k: _apply_quantile_norm(v, qt, num_cols) for k, v in aligned.items()}
    target_features_norm = _apply_quantile_norm(target_features, qt, num_cols)
    target_train_norm    = _apply_quantile_norm(
        target_train_df.drop(columns=[LABEL_COL]), qt, num_cols
    )
    target_train_norm[LABEL_COL] = target_train_df[LABEL_COL].values

    # --- Optional: load unlabeled GitTables features for domain adaptation ---
    unlabeled_features = gittables_lake.load_gittables_features(
        target_cols=list(target_features.columns),
        max_tables=20_000,  # uncapped — use full GitTables cache
    )

    # --- Step 3: Domain Adaptation ---
    logger.info("=== Step 3: Domain Adaptation ===")
    results = domain_adaptation.run_all(
        aligned=aligned,
        discovery_scores=top_k_scores,
        target=target_features_norm,
        label_col=LABEL_COL,
        weight_power=WEIGHT_POWER,
        n_dann_epochs=dann_epochs,
        unlabeled_features=unlabeled_features if unlabeled_features else None,
    )
    results["oracle"] = domain_adaptation.run_oracle(
        target_train=target_train_norm,
        target_test=target_features_norm,
        label_col=LABEL_COL,
    )

    # --- Step 4: Evaluation ---
    logger.info("=== Step 4: Evaluation ===")
    target_pos_rate = float(y_true.mean())
    logger.info("Target positive rate (threshold calibration): %.3f", target_pos_rate)

    metrics_raw = evaluation.evaluate(results, y_true)
    metrics_cal = evaluation.evaluate(results, y_true, target_pos_rate=target_pos_rate)
    summary_raw = evaluation.summarise(metrics_raw)
    summary_cal = evaluation.summarise(metrics_cal)

    target_desc = {
        "adult":    "UCI Adult 1994 (income >$50k)",
        "nyhouse":  "NY Housing (price >$1M)",
        "bank":     "Bank Marketing (term deposit subscription)",
        "diabetes": "Pima Indians Diabetes",
        "credit":   "German Credit (good/bad)",
    }.get(target_name, target_name)

    print("\n" + "=" * 65)
    print("ACT 4 RESULTS — OpenML Data Lake")
    print(f"  Target  : {target_desc}")
    print(f"  Lake    : {len(lake)} OpenML datasets (top-{TOP_K} selected by discovery)")
    print("=" * 65)
    print("\n--- Default threshold (0.5) ---")
    print(summary_raw.to_string())
    print("\n--- Calibrated threshold (matched to target positive rate) ---")
    print(summary_cal.to_string())
    print()

    summary_cal.to_csv(cfg.results_dir / "metrics.csv")
    summary_raw.to_csv(cfg.results_dir / "metrics_uncalibrated.csv")
    logger.info("Results saved to %s", cfg.results_dir)
    return summary_cal


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Act 4 — OpenML Data Lake experiment")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download and cache lake datasets; skip the experiment.",
    )
    parser.add_argument(
        "--lake-size",
        type=int,
        default=DEFAULT_LAKE_SIZE,
        help=f"Number of lake tables to use (default: {DEFAULT_LAKE_SIZE}).",
    )
    parser.add_argument(
        "--target",
        choices=list(_TARGETS.keys()),
        default="adult",
        help="Target dataset to predict (default: adult).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of lake tables to select for adaptation (default: {TOP_K}).",
    )
    parser.add_argument(
        "--dynamic-top-k",
        action="store_true",
        help="Select tables scoring ≥50%% of the top score instead of a fixed count.",
    )
    parser.add_argument(
        "--dann-epochs",
        type=int,
        default=DANN_EPOCHS,
        help=f"Training epochs for Level 5 DANN (default: {DANN_EPOCHS}).",
    )
    args = parser.parse_args()
    run_act4(
        lake_size=args.lake_size,
        download_only=args.download_only,
        target_name=args.target,
        dynamic_top_k=args.dynamic_top_k,
        top_k=args.top_k,
        dann_epochs=args.dann_epochs,
    )


if __name__ == "__main__":
    main()
