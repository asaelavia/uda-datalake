"""
Act 5 — GitTables as a True Unlabeled Data Lake

All labeled signal comes from SOURCE REPURPOSING: scanning the GitTables
lake for columns whose name semantically matches the target label, then
binarizing that column at its median to create a pseudo-label.

No external labeled datasets are used.  Non-repurposed tables are used
as unlabeled domain data for Level 3 and Level 5.

Run
---
    # Check how many tables are cached:
    python gittables_lake.py --stats

    # Run all targets:
    python act5_gittables_lake.py --target adult
    python act5_gittables_lake.py --target nyhouse
    python act5_gittables_lake.py --target bank
    python act5_gittables_lake.py --target diabetes
    python act5_gittables_lake.py --target credit
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# Reuse target loaders and helpers from act4
from act4_openml_lake import (
    LABEL_COL,
    BANK_DID,
    DIABETES_DID,
    CREDIT_DID,
    CHURN_DID,
    HEART_DID,
    TURNOVER_DID,
    CRIME_DID,
    TITANIC_DID,
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
    _make_quantile_normalizer,
    _apply_quantile_norm,
)

import domain_adaptation
import evaluation
import gittables_lake
import schema_alignment
import table_discovery
import transferability as _xfer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPURPOSE_THRESHOLD  = 0.70
EMBED_BATCH_TABLES   = 512   # encode columns from this many tables in one GPU call
CONTEXT_THRESHOLD    = 0.15  # min cosine sim between table column centroid and target feature centroid
SIBLING_THRESHOLD    = 0.70  # max allowed sim between matched col and its nearest sibling col

RESULTS_BASE = Path("results/act5")

@dataclass
class TargetConfig:
    label_name: str          # natural-language description of the label for repurposing
    results_dir: Path

_TARGETS: dict[str, TargetConfig] = {
    "adult":    TargetConfig("income above 50k",             RESULTS_BASE / "adult"),
    "nyhouse":  TargetConfig("house price above 1 million",  RESULTS_BASE / "nyhouse"),
    "bank":     TargetConfig("term deposit subscription",    RESULTS_BASE / "bank"),
    "diabetes": TargetConfig("diabetes diagnosis positive",  RESULTS_BASE / "diabetes"),
    "credit":   TargetConfig("credit risk good or bad",      RESULTS_BASE / "credit"),
    "churn":    TargetConfig("customer churn",               RESULTS_BASE / "churn"),
    "heart":    TargetConfig("heart disease diagnosis",      RESULTS_BASE / "heart"),
    "turnover": TargetConfig("employee turnover",            RESULTS_BASE / "turnover"),
    "crime":    TargetConfig("violent crime rate high",      RESULTS_BASE / "crime"),
    "obesity":  TargetConfig("county obesity high",          RESULTS_BASE / "obesity"),
    "noshow":   TargetConfig("medical appointment no-show",  RESULTS_BASE / "noshow"),
    "titanic":  TargetConfig("passenger survival titanic",   RESULTS_BASE / "titanic"),
    "stroke":   TargetConfig("stroke diagnosis",             RESULTS_BASE / "stroke"),
    "breastcancer": TargetConfig("breast cancer diagnosis malignant", RESULTS_BASE / "breastcancer"),
}


def _load_cdc_obesity_target() -> pd.DataFrame:
    """Load CDC PLACES county obesity target from local CSV."""
    csv_path = Path("data/cdc_obesity_county.csv")
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"label": LABEL_COL})
    return df


def _load_noshow_target() -> pd.DataFrame:
    """
    Load Brazilian medical appointment no-show dataset (Kaggle).

    Expected file: data/KaggleV2-May-2016.csv
    Label: No-show == 'Yes' → 1 (patient missed appointment)
    """
    csv_path = Path("data/KaggleV2-May-2016.csv")
    df = pd.read_csv(csv_path)

    # Drop ID and high-cardinality columns
    df = df.drop(columns=["PatientId", "AppointmentID", "Neighbourhood"], errors="ignore")

    # Feature engineer: days between scheduling and appointment
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], utc=True)
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], utc=True)
    df["days_in_advance"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days.clip(0, 365)
    df = df.drop(columns=["ScheduledDay", "AppointmentDay"])

    # Filter invalid ages
    df = df[df["Age"] >= 0].copy()

    # Encode categoricals
    df["Gender"] = df["Gender"].map({"F": 0, "M": 1}).fillna(0).astype(int)

    # Label: 'Yes' = patient did NOT show up → positive class = no-show
    df[LABEL_COL] = (df["No-show"] == "Yes").astype(int)
    df = df.drop(columns=["No-show"])

    # Lowercase column names for cleaner schema alignment
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={LABEL_COL.lower(): LABEL_COL})

    return df.reset_index(drop=True)


def _load_stroke_target() -> pd.DataFrame:
    """
    Load Kaggle healthcare stroke prediction dataset.

    Expected file: data/stroke.csv  (download from Kaggle: fedesoriano/stroke-prediction-dataset)
    Label: stroke == 1 → positive class (had a stroke)
    """
    csv_path = Path("data/stroke.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            "Stroke dataset not found at data/stroke.csv. "
            "Download from https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset"
        )
    df = pd.read_csv(csv_path)

    # Drop ID column
    df = df.drop(columns=["id"], errors="ignore")

    # Label
    df[LABEL_COL] = df["stroke"].astype(int)
    df = df.drop(columns=["stroke"])

    # Encode categoricals
    cat_maps = {
        "gender":           {"Male": 1, "Female": 0, "Other": 0},
        "ever_married":     {"Yes": 1, "No": 0},
        "work_type":        {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4},
        "Residence_type":   {"Urban": 1, "Rural": 0},
        "smoking_status":   {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": -1},
    }
    for col, mapping in cat_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # bmi has some missing values — fill with median
    if "bmi" in df.columns:
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
        df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    df = df.dropna().reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={LABEL_COL.lower(): LABEL_COL})

    import logging as _log
    _log.getLogger(__name__).info(
        "Stroke: %d rows, positive_rate=%.3f, cols=%s",
        len(df), float(df[LABEL_COL].mean()), list(df.drop(columns=[LABEL_COL]).columns),
    )
    return df


def _load_titanic_target() -> pd.DataFrame:
    """
    Load Titanic passenger survival dataset (OpenML 40945).

    Drops label-leaking and high-cardinality columns (boat, body, name,
    ticket, cabin, home.dest).  Positive class = survived (1).
    """
    from sklearn.datasets import fetch_openml
    data = fetch_openml(data_id=TITANIC_DID, as_frame=True, parser="auto")
    df = data.frame.copy()

    # Drop label-leaking column (boat encodes survival directly)
    # and high-cardinality text columns that won't align with lake tables
    _drop = ["boat", "body", "name", "ticket", "cabin", "home.dest"]
    df = df.drop(columns=[c for c in _drop if c in df.columns], errors="ignore")

    # Label: survived == '1' → positive
    label_raw = df.pop("survived").astype(str).str.strip().str.lower()
    df[LABEL_COL] = label_raw.isin({"1", "yes", "true"}).astype(int)

    # Encode categoricals
    df["sex"] = df["sex"].map({"female": 0, "male": 1}).fillna(0).astype(int)
    if "embarked" in df.columns:
        df["embarked"] = df["embarked"].map({"c": 0, "q": 1, "s": 2}).fillna(1).astype(int)
    if "pclass" in df.columns:
        df["pclass"] = pd.to_numeric(df["pclass"], errors="coerce").fillna(3).astype(int)

    # Drop columns with too many missing values, then drop remaining NaN rows
    thresh = int(0.7 * len(df))
    df = df.dropna(axis=1, thresh=thresh).dropna().reset_index(drop=True)

    import logging as _log
    _log.getLogger(__name__).info(
        "Titanic: %d rows, positive_rate=%.3f, cols=%s",
        len(df), float(df[LABEL_COL].mean()), list(df.drop(columns=[LABEL_COL]).columns),
    )
    return df


def _repurpose_gittables(
    lake: dict[str, pd.DataFrame],
    label_name: str,
    encoder: SentenceTransformer,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """
    Scan GitTables lake for columns matching label_name; binarize at median.

    Returns
    -------
    labeled_lake : dict[table_id → DataFrame with LABEL_COL]
    label_names  : dict[table_id → repurposed column name]
    """
    logger.info("=== Source Repurposing: scanning %d GitTables for '%s' ===",
                len(lake), label_name)

    repurpose_map = table_discovery.find_repurposable_features(
        lake=lake,
        target_label_name=label_name,
        model=encoder,
        threshold=REPURPOSE_THRESHOLD,
    )

    if not repurpose_map:
        logger.warning("No repurposable tables found at threshold=%.2f", REPURPOSE_THRESHOLD)
        return {}, {}

    logger.info("Found %d repurposable tables:", len(repurpose_map))

    labeled_lake: dict[str, pd.DataFrame] = {}
    label_names: dict[str, str] = {}

    for table_id, repurpose_col in repurpose_map.items():
        if table_id not in lake or repurpose_col not in lake[table_id].columns:
            continue
        df = lake[table_id].copy()
        col_vals_raw = df[repurpose_col].dropna()
        if col_vals_raw.nunique() > 100:
            logger.debug("  Skipping '%s' col='%s': too many distinct values (%d > 100)",
                         table_id, repurpose_col, int(col_vals_raw.nunique()))
            continue
        col_median = df[repurpose_col].median()
        df[repurpose_col] = df[repurpose_col].fillna(col_median)
        # Use strict > when median == min to avoid promoting the zero class to 1
        # (e.g. binary 0/1 column with 80% zeros has median=0.0 → >= 0 makes all rows 1)
        col_min = df[repurpose_col].min()
        if col_median == col_min:
            df[repurpose_col] = (df[repurpose_col] > col_median).astype(int)
        else:
            df[repurpose_col] = (df[repurpose_col] >= col_median).astype(int)

        if df[repurpose_col].nunique() < 2:
            continue  # zero variance after binarization

        df = df.rename(columns={repurpose_col: LABEL_COL})
        labeled_lake[table_id] = df
        label_names[table_id] = repurpose_col
        logger.debug("  %-45s  col='%s'  pos_rate=%.3f",
                     table_id, repurpose_col, float(df[LABEL_COL].mean()))

    logger.info("Repurposing complete: %d labeled tables (from %d candidates)",
                len(labeled_lake), len(repurpose_map))
    # Log a sample of the repurposed tables at INFO level
    for table_id in list(labeled_lake)[:10]:
        logger.info("  [sample] %-45s  col='%s'  pos_rate=%.3f",
                    table_id, label_names[table_id],
                    float(labeled_lake[table_id][LABEL_COL].mean()))
    return labeled_lake, label_names


def _apply_centroid_filter(
    candidates: dict[str, str],
    manifest_col_lookup: dict[str, list[str]],
    encoder: SentenceTransformer,
    target_centroid_norm: np.ndarray,
    threshold: float,
) -> dict[str, str]:
    """
    Filter repurpose candidates by column-space domain similarity.

    For each candidate table, embeds its manifest column names, computes the
    centroid, and drops the table if the centroid is too far from the target
    feature centroid.  Used on the done-cache fast path to clean up legacy
    caches that predate this filter.
    """
    if not candidates:
        return candidates

    table_ids = list(candidates.keys())
    col_lists  = [manifest_col_lookup.get(tid, []) for tid in table_ids]

    # Flatten all column names for a single batch encode call
    all_cols: list[str] = [c for cols in col_lists for c in cols]
    if not all_cols:
        return candidates

    all_embs = encoder.encode(
        all_cols, batch_size=512, show_progress_bar=False, convert_to_numpy=True,
    )

    filtered: dict[str, str] = {}
    offset = 0
    for tid, cols in zip(table_ids, col_lists):
        n = len(cols)
        if n == 0:
            filtered[tid] = candidates[tid]
            continue
        embs = all_embs[offset : offset + n]
        offset += n
        centroid = embs.mean(axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 1e-9:
            sim = float(np.dot(target_centroid_norm, centroid / c_norm))
            if sim >= threshold:
                filtered[tid] = candidates[tid]
            else:
                logger.debug("[CentroidFilter] Dropped '%s'  domain_sim=%.3f", tid, sim)
        else:
            filtered[tid] = candidates[tid]

    logger.info(
        "[CentroidFilter] Done-cache: %d/%d candidates passed (threshold=%.2f)",
        len(filtered), len(candidates), threshold,
    )
    return filtered


def _build_labeled_lake(
    repurpose_result: dict[str, str],
    id_to_path: dict[str, Path],
    target_pos_rate: Optional[float] = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """
    Load and binarize the tables identified by `repurpose_result`.

    Parameters
    ----------
    repurpose_result : {table_id → best-matching column name}
    id_to_path       : {table_id → parquet Path}

    Returns
    -------
    labeled_lake, label_names
    """
    labeled_lake: dict[str, pd.DataFrame] = {}
    label_names:  dict[str, str]          = {}

    for table_id, repurpose_col in repurpose_result.items():
        fpath = id_to_path.get(table_id)
        if not fpath or not fpath.exists():
            continue
        try:
            df = pd.read_parquet(fpath)
        except Exception:
            continue
        if repurpose_col not in df.columns:
            continue

        df = df.copy()
        if LABEL_COL in df.columns and repurpose_col != LABEL_COL:
            df = df.drop(columns=[LABEL_COL])
        col_vals = df[repurpose_col].astype(float)
        binarize_thresh = col_vals.median()

        df[repurpose_col] = col_vals.fillna(binarize_thresh)
        # Use strict > when median == min to avoid promoting the zero class to 1
        # (e.g. binary 0/1 column with 80% zeros has median=0.0 → >= 0 makes all rows 1)
        if binarize_thresh == col_vals.min():
            df[repurpose_col] = (df[repurpose_col] > binarize_thresh).astype(int)
        else:
            df[repurpose_col] = (df[repurpose_col] >= binarize_thresh).astype(int)

        # High-cardinality check BEFORE binarization result check:
        # columns with > 100 distinct raw values are likely IDs or free-text, not proxy labels
        # (catches EMPLOYEE_ID with thousands of values, while allowing continuous scientific
        # measurements like Insulin (113 values) or risk scores to be median-binarized)
        if col_vals.nunique() > 100:
            logger.debug("  Skipping '%s' col='%s': too many distinct values (%d > 100)",
                         table_id, repurpose_col, int(col_vals.nunique()))
            continue

        if df[repurpose_col].nunique() < 2:
            continue

        df = df.rename(columns={repurpose_col: LABEL_COL})
        labeled_lake[table_id] = df
        label_names[table_id]  = repurpose_col
        logger.debug("  %-45s  col='%s'  pos_rate=%.3f",
                     table_id, repurpose_col, float(df[LABEL_COL].mean()))

    logger.info("Labeled lake built: %d tables", len(labeled_lake))
    for table_id in list(labeled_lake)[:10]:
        logger.info("  [sample] %-45s  col='%s'  pos_rate=%.3f",
                    table_id, label_names[table_id],
                    float(labeled_lake[table_id][LABEL_COL].mean()))
    return labeled_lake, label_names


def _stream_load_and_repurpose(
    manifest_tables: list[dict],
    cache_dir: Path,
    label_name: str,
    encoder: SentenceTransformer,
    threshold: float,
    target_features: Optional[pd.DataFrame] = None,
    target_pos_rate: Optional[float] = None,
    concepts_override: Optional[list[str]] = None,
    sample_tag: Optional[int] = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """
    Streaming single-pass load + repurpose scan.

    Reads each parquet file once, immediately checks the repurposing threshold,
    and discards tables that don't qualify — so only O(passing tables) are ever
    in memory at once.  Progress is checkpointed every 5 % so the scan can be
    interrupted and resumed without re-processing earlier entries.

    Parameters
    ----------
    manifest_tables     : list of {"table_id": ..., "path": ...} dicts from manifest.json
    cache_dir           : root directory for parquet files
    label_name          : e.g. "income above 50k"
    encoder             : pre-loaded SentenceTransformer
    threshold           : minimum max-concept cosine similarity to qualify
    concepts_override   : if provided, skip LLM/KG expansion and use this list directly
                          (used for ablation: pass [label_name] to test label-only baseline)

    Returns
    -------
    labeled_lake  : dict[table_id → DataFrame with LABEL_COL]
    label_names   : dict[table_id → original repurposed column name]
    """
    # --- Concept expansion (LLM → KG → label-only fallback) ---
    if concepts_override is not None:
        concepts: list[str] = concepts_override
        logger.info("Using concepts_override (%d concepts): %s", len(concepts), concepts[:5])
    else:
        concepts = table_discovery.expand_label_via_llm(label_name)
        if len(concepts) <= 1:
            logger.info("LLM expansion returned no results; falling back to KG.")
            concepts = table_discovery.expand_label_via_kg(label_name)
    if len(concepts) > 1:
        concepts = table_discovery._deduplicate_concepts(concepts, encoder)
        logger.info("Concepts after deduplication: %d", len(concepts))

    # Embed all concepts once: shape (n_concepts, embed_dim)
    target_embs = np.vstack([
        table_discovery.embed_columns([c], encoder)[0].reshape(1, -1)
        for c in concepts
    ])

    # Sort by table_id for a stable, reproducible iteration order
    sorted_entries = sorted(manifest_tables, key=lambda e: e["table_id"])

    # --- Checkpoint / done-cache setup ---
    # Include the lake name in the cache key so different lakes get separate caches.
    lake_slug = re.sub(r"[^a-z0-9]", "_", cache_dir.name.lower())
    ckpt_key  = f"{label_name}__n{len(concepts)}"
    if sample_tag is not None:
        ckpt_key += f"__s{sample_tag}"
    ckpt_slug = re.sub(r"[^a-z0-9_]", "_", ckpt_key.lower())
    ckpt_path  = Path(__file__).parent / "data" / f"stream_ckpt_{lake_slug}__{ckpt_slug}.json"
    done_path  = Path(__file__).parent / "data" / f"repurpose_done_{lake_slug}__{ckpt_slug}.json"
    # Backward-compat: if no lake-specific cache exists but the old generic one does and
    # all its table_ids belong to this lake, rename it rather than re-scanning.
    _old_done = Path(__file__).parent / "data" / f"repurpose_done_{ckpt_slug}.json"
    _old_ckpt = Path(__file__).parent / "data" / f"stream_ckpt_{ckpt_slug}.json"
    if not done_path.exists() and _old_done.exists():
        try:
            with open(_old_done) as _f:
                _old_ids = set(json.load(_f).keys())
            _manifest_ids = {e["table_id"] for e in sorted_entries}
            if _old_ids and _old_ids.issubset(_manifest_ids):
                logger.info("Renaming legacy done-cache to lake-specific path.")
                _old_done.rename(done_path)
                if _old_ckpt.exists():
                    _old_ckpt.rename(ckpt_path)
        except Exception:
            pass
    id_to_path = {e["table_id"]: cache_dir / e["path"] for e in sorted_entries}

    # --- Compute target column centroid for domain-coherence filter ---
    target_centroid_norm: Optional[np.ndarray] = None
    if target_features is not None:
        _tcols = [c for c in target_features.columns if not str(c).startswith("Unnamed")]
        if _tcols:
            _tc_embs = encoder.encode(_tcols, show_progress_bar=False, convert_to_numpy=True)
            _tc = _tc_embs.mean(axis=0)
            _tc_n = np.linalg.norm(_tc)
            target_centroid_norm = _tc / _tc_n if _tc_n > 1e-9 else _tc
            logger.info("Domain filter centroid computed from %d target columns", len(_tcols))

    # --- Fast path: load from done-cache if the scan already completed ---
    if done_path.exists():
        try:
            with open(done_path) as f:
                cached_result: dict[str, str] = json.load(f)
            logger.info(
                "Loaded repurpose done-cache '%s': %d candidates (skipping full scan)",
                done_path.name, len(cached_result),
            )
            # Apply centroid filter to remove wrong-domain tables from legacy caches
            if target_centroid_norm is not None and cached_result:
                manifest_col_lookup = {e["table_id"]: e.get("columns", []) for e in sorted_entries}
                cached_result = _apply_centroid_filter(
                    cached_result, manifest_col_lookup, encoder,
                    target_centroid_norm, CONTEXT_THRESHOLD,
                )
            return _build_labeled_lake(cached_result, id_to_path, target_pos_rate=target_pos_rate)
        except Exception as exc:
            logger.warning("Could not load done-cache: %s — re-scanning.", exc)

    # repurpose_result maps table_id → best-matching column name
    repurpose_result: dict[str, str] = {}
    resume_from = 0

    if ckpt_path.exists():
        try:
            with open(ckpt_path) as f:
                ckpt = json.load(f)
            if (ckpt.get("label") == label_name
                    and ckpt.get("n_concepts") == len(concepts)):
                resume_from = int(ckpt.get("progress_idx", 0))
                repurpose_result = ckpt.get("result", {})
                logger.info(
                    "Resuming stream scan from %d/%d (%d candidates so far)",
                    resume_from, len(sorted_entries), len(repurpose_result),
                )
            else:
                logger.info("Stream checkpoint parameters changed — starting fresh.")
                ckpt_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Could not load stream checkpoint: %s — starting fresh.", exc)

    n_total     = len(sorted_entries)
    _log_every  = max(1, n_total // 20)  # progress update every ~5 %
    _ckpt_every = max(1, n_total // 20)  # checkpoint at same interval

    logger.info("=== Streaming load + repurpose: %d manifest entries ===", n_total)

    def _flush_batch(
        batch_ids: list[str],
        batch_col_lists: list[list[str]],
        batch_start_idx: int,
    ) -> None:
        """Encode all columns from a batch of tables in one GPU call, check thresholds."""
        if not batch_ids:
            return

        # Flatten columns from all tables in this batch
        all_cols: list[str] = []
        offsets: list[int] = []
        for cols in batch_col_lists:
            offsets.append(len(all_cols))
            all_cols.extend(cols)
        offsets.append(len(all_cols))

        # Single encode call — GPU processes all columns at once
        all_embs = encoder.encode(
            all_cols,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=512,
        )

        for i, (table_id, cols) in enumerate(zip(batch_ids, batch_col_lists)):
            col_embs = all_embs[offsets[i] : offsets[i + 1]]

            # 1. Domain-coherence filter: skip tables whose column centroid is far
            #    from the target feature centroid (e.g. chemistry tables for turnover).
            if target_centroid_norm is not None:
                table_centroid = col_embs.mean(axis=0)
                tc_n = np.linalg.norm(table_centroid)
                if tc_n > 1e-9:
                    ctx_sim = float(np.dot(target_centroid_norm, table_centroid / tc_n))
                    if ctx_sim < CONTEXT_THRESHOLD:
                        logger.debug("  Domain filter: %-45s  ctx_sim=%.3f — skipped", table_id, ctx_sim)
                        continue

            # 2. Concept similarity: find best-matching column
            sim_matrix = 1.0 - cdist(target_embs, col_embs, metric="cosine")
            best_sims  = sim_matrix.max(axis=0)
            best_idx   = int(np.argmax(best_sims))
            best_sim   = float(best_sims[best_idx])

            if best_sim >= threshold:
                # 3. Sibling filter: reject if matched col has a very similar sibling.
                #    A standalone label col (Attrition) has dissimilar siblings (Age, Income…).
                #    A metric in a family (DEF_RATING next to OFF_RATING, NET_RATING…) does not.
                if len(col_embs) > 1:
                    best_n = col_embs[best_idx] / (np.linalg.norm(col_embs[best_idx]) + 1e-9)
                    others = np.delete(col_embs, best_idx, axis=0)
                    o_norms = np.linalg.norm(others, axis=1, keepdims=True)
                    others_n = others / np.where(o_norms > 1e-9, o_norms, 1.0)
                    max_sib = float(np.max(others_n @ best_n))
                    if max_sib > SIBLING_THRESHOLD:
                        logger.debug(
                            "  Sibling filter: %-45s  col='%s'  sib_sim=%.3f — skipped",
                            table_id, cols[best_idx], max_sib,
                        )
                        continue

                repurpose_result[table_id] = cols[best_idx]
                logger.debug(
                    "  Repurpose candidate: %-45s  col='%s'  sim=%.4f",
                    table_id, cols[best_idx], best_sim,
                )

    batch_ids:      list[str]        = []
    batch_col_lists: list[list[str]] = []
    batch_start_idx = resume_from

    for _i, entry in enumerate(sorted_entries):
        if _i < resume_from:
            continue

        if _i % _log_every == 0:
            logger.info(
                "  Stream scan: %d/%d (%.0f%%) — %d labeled candidates so far",
                _i, n_total, 100 * _i / n_total, len(repurpose_result),
            )

        if _i > 0 and _i % _ckpt_every == 0:
            # Flush current batch before checkpointing so result is consistent
            _flush_batch(batch_ids, batch_col_lists, batch_start_idx)
            batch_ids.clear()
            batch_col_lists.clear()
            batch_start_idx = _i
            try:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                with open(ckpt_path, "w") as f:
                    json.dump({
                        "label": label_name,
                        "n_concepts": len(concepts),
                        "progress_idx": _i,
                        "result": repurpose_result,
                    }, f)
                logger.debug("Stream checkpoint saved at %d/%d", _i, n_total)
            except Exception as exc:
                logger.warning("Could not save stream checkpoint: %s", exc)

        fpath = id_to_path.get(entry["table_id"])
        if not fpath or not fpath.exists():
            continue
        try:
            df = pd.read_parquet(fpath)
        except Exception:
            continue

        feat_cols = list(df.columns)
        if not feat_cols:
            continue

        batch_ids.append(entry["table_id"])
        batch_col_lists.append(feat_cols)

        # Flush when batch is full
        if len(batch_ids) >= EMBED_BATCH_TABLES:
            _flush_batch(batch_ids, batch_col_lists, batch_start_idx)
            batch_ids.clear()
            batch_col_lists.clear()
            batch_start_idx = _i + 1

    # Flush the final partial batch
    _flush_batch(batch_ids, batch_col_lists, batch_start_idx)

    logger.info("Stream scan complete: %d repurposable tables found", len(repurpose_result))

    # --- Persist done-cache so act6/act7 can skip the full scan ---
    try:
        done_path.parent.mkdir(parents=True, exist_ok=True)
        with open(done_path, "w") as f:
            json.dump(repurpose_result, f)
        logger.info("Saved repurpose done-cache '%s' (%d entries)", done_path.name, len(repurpose_result))
    except Exception as exc:
        logger.warning("Could not save done-cache: %s", exc)

    # Clean up in-progress checkpoint on successful completion
    ckpt_path.unlink(missing_ok=True)

    return _build_labeled_lake(repurpose_result, id_to_path, target_pos_rate=target_pos_rate)


def run_experiment(
    target_name: str,
    top_k: int = TOP_K,
    lake_dir: Optional[Path] = None,
    no_expansion: bool = False,
    repurpose_threshold: float = REPURPOSE_THRESHOLD,
    llm_baseline: bool = False,
    n_seeds: int = 1,
    fast_only: bool = False,
    lake_sample: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    cfg = _TARGETS[target_name]
    # When a non-default lake is used, write results to a lake-specific subdirectory
    lake_dir = lake_dir or gittables_lake.DEFAULT_CACHE
    lake_tag  = lake_dir.name if lake_dir != gittables_lake.DEFAULT_CACHE else None
    _k_suffix   = f"_k{top_k}" if top_k != TOP_K else ""
    _thr_suffix = f"_thr{repurpose_threshold:.2f}" if repurpose_threshold != REPURPOSE_THRESHOLD else ""
    _s_suffix   = f"_s{lake_sample}" if lake_sample is not None else ""
    if _k_suffix or _thr_suffix or _s_suffix:
        base_dir = cfg.results_dir.parent / f"{cfg.results_dir.name}{_k_suffix}{_thr_suffix}{_s_suffix}"
    else:
        base_dir = cfg.results_dir
    results_dir = (base_dir.parent / lake_tag / base_dir.name) if lake_tag else base_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load manifest (check cache exists)
    # ------------------------------------------------------------------ #
    cache_dir     = lake_dir
    manifest_path = cache_dir / gittables_lake.MANIFEST_FILE
    if not manifest_path.exists():
        raise RuntimeError(
            f"Lake cache not found at {cache_dir}. "
            "Run the appropriate downloader or check --lake-dir."
        )
    # Retry up to 5 times — the Zenodo downloader may be writing the manifest
    # concurrently, leaving it in a temporarily corrupt state.
    manifest = None
    for _attempt in range(5):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            break
        except json.JSONDecodeError:
            import time as _time
            logger.warning("manifest.json read failed (attempt %d/5), retrying in 3s…", _attempt + 1)
            _time.sleep(3)
    if manifest is None:
        raise RuntimeError("manifest.json is corrupt after 5 retries — stop the downloader and retry.")

    n_manifest = len(manifest["tables"])
    logger.info("Manifest loaded: %d table entries", n_manifest)

    # --- Lake subsampling (scalability experiment) ---
    manifest_tables = manifest["tables"]
    if lake_sample is not None and lake_sample < n_manifest:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n_manifest, size=lake_sample, replace=False))
        manifest_tables = [manifest_tables[i] for i in idx]
        logger.info("Lake subsampled: %d/%d tables (seed=42)", lake_sample, n_manifest)

    # ------------------------------------------------------------------ #
    # Load target (with labels — used for oracle/eval only, never for adaptation)
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
    elif target_name == "noshow":
        target_df = _load_noshow_target()
    elif target_name == "titanic":
        target_df = _load_titanic_target()
    elif target_name == "stroke":
        target_df = _load_stroke_target()
    elif target_name == "breastcancer":
        target_df = _load_openml_target(BREASTCANCER_DID, "Breast Cancer Wisconsin",
                                        positive_values={"malignant"})
    else:
        raise ValueError(f"Unknown target: {target_name!r}")

    target_train_df, target_test_df = train_test_split(
        target_df, test_size=ORACLE_TEST_SIZE, random_state=RANDOM_STATE,
        stratify=target_df[LABEL_COL],
    )
    y_true          = target_test_df[LABEL_COL].values
    target_features = target_test_df.drop(columns=[LABEL_COL])
    logger.info("Target split: %d oracle-train / %d test", len(target_train_df), len(target_test_df))

    # ------------------------------------------------------------------ #
    # Load encoder (needed before streaming scan)
    # ------------------------------------------------------------------ #
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading encoder: %s  (device=%s)", ENCODER_MODEL, _device)
    encoder = SentenceTransformer(ENCODER_MODEL, device=_device)

    # ------------------------------------------------------------------ #
    # Transferability score (fast) — build index on first run, query instantly
    # ------------------------------------------------------------------ #
    _index_dir = Path("data/col_name_index")
    if not (_index_dir / "embs.npy").exists():
        logger.info("Column name index not found — building (one-time, ~minutes)...")
        _xfer.build_column_index(manifest["tables"], cache_dir, encoder, _index_dir)

    try:
        _concepts_for_fast = table_discovery.expand_label_via_llm(cfg.label_name)
        if len(_concepts_for_fast) <= 1:
            _concepts_for_fast = [cfg.label_name]
        t_fast = _xfer.compute_score_fast(
            manifest_tables=manifest_tables,
            cache_dir=cache_dir,
            encoder=encoder,
            target_features=target_features,
            target_pos_rate=float(target_df[LABEL_COL].mean()),
            concepts=_concepts_for_fast,
            threshold=repurpose_threshold,
            top_k=top_k,
            index_dir=_index_dir,
        )
        pd.DataFrame([vars(t_fast)]).to_csv(results_dir / "transferability_fast.csv", index=False)
        logger.info(
            "[Transferability fast] overall=%.3f  yield=%.3f  quality=%.3f  "
            "density=%.3f  shift=%.3f  consistency=%.3f",
            t_fast.overall, t_fast.repurpose_yield, t_fast.discovery_quality,
            t_fast.alignment_density, t_fast.label_shift, t_fast.source_consistency,
        )
    except Exception as exc:
        logger.warning("Fast transferability score failed: %s — skipping.", exc)

    if fast_only:
        logger.info("--fast-only: stopping after fast transferability score.")
        return

    # ------------------------------------------------------------------ #
    # Streaming load + source repurposing → labeled lake
    # Single pass: read each parquet, check threshold, keep only candidates.
    # Progress is checkpointed every 5 % — safe to Ctrl-C and resume.
    # ------------------------------------------------------------------ #
    if repurpose_threshold != REPURPOSE_THRESHOLD:
        logger.warning(
            "Non-default repurpose_threshold=%.2f. Existing done-caches (keyed by label+n_concepts) "
            "will be reused regardless of threshold — delete stream_ckpt_*.json files for a clean scan.",
            repurpose_threshold,
        )
    labeled_lake, label_names = _stream_load_and_repurpose(
        manifest_tables=manifest_tables,
        cache_dir=cache_dir,
        label_name=cfg.label_name,
        encoder=encoder,
        threshold=repurpose_threshold,
        target_features=target_features,
        target_pos_rate=float(target_df[LABEL_COL].mean()),
        concepts_override=[cfg.label_name] if no_expansion else None,
        sample_tag=lake_sample,
    )

    if not labeled_lake:
        logger.error("No labeled sources found — cannot run adaptation for '%s'.", target_name)
        return None

    n_lake_effective = len(manifest_tables)
    logger.info("%d unlabeled tables in lake (not repurposed)", n_lake_effective - len(labeled_lake))

    # ------------------------------------------------------------------ #
    # Step 1: Table Discovery
    # ------------------------------------------------------------------ #
    logger.info("=== Step 1: Table Discovery ===")
    lake_features   = {k: v.drop(columns=[LABEL_COL]) for k, v in labeled_lake.items()}
    source_pos_rates = {k: float(labeled_lake[k][LABEL_COL].mean()) for k in labeled_lake}
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
        logger.info("  %-50s  score=%.4f  col='%s'",
                    tbl, score, label_names.get(tbl, "?"))

    top_k_scores = dict(list(scores.items())[:top_k])
    logger.info("Selected top-%d tables for adaptation:", len(top_k_scores))
    for tbl, score in top_k_scores.items():
        logger.info("  %-50s  %.4f  col='%s'", tbl, score, label_names.get(tbl, "?"))

    pd.Series(scores, name="similarity").to_csv(results_dir / "discovery_scores.csv")

    # ------------------------------------------------------------------ #
    # Step 2: Schema Alignment
    # ------------------------------------------------------------------ #
    logger.info("=== Step 2: Schema Alignment ===")
    lake_top_k = {k: labeled_lake[k] for k in top_k_scores}
    aligned = schema_alignment.align_all(
        lake=lake_top_k,
        target=target_features,
        discovery_scores=top_k_scores,
        model=encoder,
        label_col=LABEL_COL,
        dist_threshold=DIST_THRESHOLD,
        min_coverage=0.0,   # coverage gate kept in score only, not in pipeline
    )

    # Quantile normalisation fitted on target features
    qt, num_cols       = _make_quantile_normalizer(target_features)
    aligned            = {k: _apply_quantile_norm(v, qt, num_cols) for k, v in aligned.items()}
    target_norm        = _apply_quantile_norm(target_features, qt, num_cols)
    target_train_norm  = _apply_quantile_norm(target_train_df.drop(columns=[LABEL_COL]), qt, num_cols)
    target_train_norm[LABEL_COL] = target_train_df[LABEL_COL].values

    # Load unlabeled lake features aligned to target columns for L5
    unlabeled_features = gittables_lake.load_gittables_features(
        target_cols=list(target_features.columns),
        max_tables=20_000,
        cache_dir=cache_dir,
    )

    # ------------------------------------------------------------------ #
    # Transferability score (true) — computed from pipeline variables
    # ------------------------------------------------------------------ #
    try:
        t_true = _xfer.compute_score(
            labeled_lake=labeled_lake,
            top_k_scores=top_k_scores,
            aligned=aligned,
            target_pos_rate=float(target_df[LABEL_COL].mean()),
            n_lake_tables=n_lake_effective,
            label_col=LABEL_COL,
            target_features=target_norm,
        )
        pd.DataFrame([vars(t_true)]).to_csv(results_dir / "transferability.csv", index=False)
        logger.info(
            "[Transferability true] overall=%.3f  yield=%.3f  quality=%.3f  "
            "density=%.3f  shift=%.3f  consistency=%.3f",
            t_true.overall, t_true.repurpose_yield, t_true.discovery_quality,
            t_true.alignment_density, t_true.label_shift, t_true.source_consistency,
        )
    except Exception as exc:
        logger.warning("True transferability score failed: %s — skipping.", exc)

    # ------------------------------------------------------------------ #
    # Step 3 + 4: Domain Adaptation + Evaluation (multi-seed)
    # ------------------------------------------------------------------ #
    logger.info("=== Step 3: Domain Adaptation ===")
    target_pos_rate_test = float(y_true.mean())
    logger.info("Target positive rate (threshold calibration): %.3f", target_pos_rate_test)

    if not aligned:
        logger.warning(
            "No aligned sources remain after quality gate — skipping adaptation levels. "
            "Only baselines and oracle will be computed."
        )

    seeds = list(range(n_seeds))
    seed_metrics_cal: list[pd.DataFrame] = []
    seed_metrics_raw: list[pd.DataFrame] = []

    for seed in seeds:
        if n_seeds > 1:
            logger.info("--- Seed %d/%d ---", seed + 1, n_seeds)
        if aligned:
            results = domain_adaptation.run_all(
                aligned=aligned,
                discovery_scores=top_k_scores,
                target=target_norm,
                label_col=LABEL_COL,
                weight_power=WEIGHT_POWER,
                unlabeled_features=unlabeled_features if unlabeled_features else None,
                random_state=seed,
            )
        else:
            results = {}

        # Baseline A: majority class (no-information floor)
        results["baseline_a"] = domain_adaptation.run_baseline_majority(target_norm)

        # Baseline B: random sources without repurposing, position-aligned
        raw_source_list = list(labeled_lake.values())
        results["baseline_b"] = domain_adaptation.run_baseline_random(
            raw_source_list, target_norm, LABEL_COL
        )
        results["oracle"] = domain_adaptation.run_oracle(
            target_train=target_train_norm,
            target_test=target_norm,
            label_col=LABEL_COL,
            random_state=seed,  # passed as xgb_kwarg → _make_xgb(random_state=seed)
        )

        # Optional: zero-shot LLM baseline (requires Ollama running locally)
        # Only run on the first seed — LLM output is deterministic (temperature=0)
        if llm_baseline and seed == 0:
            try:
                import llm_baseline as _llm_bl
                logger.info("=== LLM zero-shot baseline ===")
                results["llm_zero_shot"] = _llm_bl.run_zero_shot(
                    target_df=target_norm,
                    label_col=LABEL_COL,
                    label_description=cfg.label_name,
                )
            except Exception as exc:
                logger.warning("LLM baseline failed: %s — skipping.", exc)

        logger.info("=== Step 4: Evaluation ===")
        seed_metrics_raw.append(evaluation.evaluate(results, y_true))
        seed_metrics_cal.append(
            evaluation.evaluate(results, y_true, target_pos_rate=target_pos_rate_test)
        )

    # Aggregate across seeds
    def _agg_metrics(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (mean_df, std_df) across seeds; single seed → std is all zeros."""
        mean_df = sum(frames) / len(frames)
        if len(frames) > 1:
            import functools
            sq_sum = functools.reduce(lambda a, b: a + b, [f**2 for f in frames])
            std_df = ((sq_sum / len(frames) - mean_df**2).clip(lower=0) ** 0.5)
        else:
            std_df = mean_df * 0.0
        return mean_df, std_df

    metrics_raw_mean, metrics_raw_std = _agg_metrics(seed_metrics_raw)
    metrics_cal_mean, metrics_cal_std = _agg_metrics(seed_metrics_cal)

    summary_raw = evaluation.summarise(metrics_raw_mean)
    summary_cal = evaluation.summarise(metrics_cal_mean)

    target_desc = {
        "adult":    "UCI Adult 1994 (income >$50k)",
        "nyhouse":  "NY Housing (price >$1M)",
        "bank":     "Bank Marketing (term deposit subscription)",
        "diabetes": "Pima Indians Diabetes",
        "credit":   "German Credit (good/bad)",
        "churn":    "IBM Telco Customer Churn",
        "heart":    "Cleveland Heart Disease",
        "turnover": "Employee Turnover (TECHCO)",
    }.get(target_name, target_name)

    print("\n" + "=" * 65)
    print("ACT 5 RESULTS — Data Lake experiment (no external labels)")
    print(f"  Target       : {target_desc}")
    print(f"  Lake         : {n_lake_effective} tables from {cache_dir.name}"
          + (f" (subsampled from {n_manifest})" if lake_sample is not None else " (manifest entries)"))
    print(f"  Labeled src  : {len(labeled_lake)} (repurposed at threshold={REPURPOSE_THRESHOLD})")
    print(f"  Top-K used   : {len(top_k_scores)}")
    print("=" * 65)
    print("\n--- Default threshold (0.5) ---")
    print(summary_raw.to_string())
    print("\n--- Calibrated threshold (matched to target positive rate) ---")
    print(summary_cal.to_string())
    print()

    summary_cal.to_csv(results_dir / "metrics.csv")
    summary_raw.to_csv(results_dir / "metrics_uncalibrated.csv")

    if n_seeds > 1:
        # Per-seed AUC table (calibrated)
        per_seed = pd.concat(
            [df[["auc"]].rename(columns={"auc": f"auc_seed{i}"}) for i, df in enumerate(seed_metrics_cal)],
            axis=1,
        )
        per_seed.to_csv(results_dir / "metrics_seeds.csv")
        # Std table
        evaluation.summarise(metrics_cal_std).to_csv(results_dir / "metrics_std.csv")
        logger.info("Multi-seed results (%d seeds) saved to %s", n_seeds, results_dir)

    logger.info("Results saved to %s", results_dir)
    return summary_cal


def main() -> None:
    parser = argparse.ArgumentParser(description="Act 5: Data lake experiment")
    parser.add_argument("--target", choices=list(_TARGETS), default="adult")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Number of top sources to use (default: {TOP_K})")
    parser.add_argument("--lake-dir", type=Path, default=None,
                        help="Path to lake cache directory (default: data/gittables)")
    parser.add_argument("--no-expansion", action="store_true",
                        help="Ablation: skip LLM/KG concept expansion; use raw label name only")
    parser.add_argument("--repurpose-threshold", type=float, default=REPURPOSE_THRESHOLD,
                        help=f"Cosine similarity threshold for source repurposing (default: {REPURPOSE_THRESHOLD})")
    parser.add_argument("--llm-baseline", action="store_true",
                        help="Run Ollama zero-shot LLM baseline (requires Ollama at localhost:11434)")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of random seeds for adaptation (default: 1). "
                             "Use 5 for paper results with mean ± std.")
    parser.add_argument("--fast-only", action="store_true",
                        help="Compute only the fast transferability score and exit (no pipeline run).")
    parser.add_argument("--lake-sample", type=int, default=None,
                        help="Subsample the lake to N tables for scalability experiments. "
                             "Results written to results/act5/{target}_s{N}/. "
                             "Uses a fixed seed (42) for reproducibility.")
    args = parser.parse_args()
    run_experiment(args.target, top_k=args.top_k, lake_dir=args.lake_dir,
                   no_expansion=args.no_expansion,
                   repurpose_threshold=args.repurpose_threshold,
                   llm_baseline=args.llm_baseline,
                   n_seeds=args.seeds,
                   fast_only=args.fast_only,
                   lake_sample=args.lake_sample)


if __name__ == "__main__":
    main()
