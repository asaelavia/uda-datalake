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

REPURPOSE_THRESHOLD        = 0.70
EMBED_BATCH_TABLES         = 512   # encode columns from this many tables in one GPU call
CONTEXT_THRESHOLD          = 0.20  # min cosine sim between table column centroid and target feature centroid
NEIGHBOR_CONTEXT_THRESHOLD = 0.12  # min cosine sim of label-column NEIGHBORS to target feature centroid (SANTOS-style)
TABLE_CENTROID_THRESHOLD   = 0.10  # min cosine sim of full remaining-table centroid to target feature centroid
MIN_DOMAIN_FRACTION        = 0.25  # min fraction of non-proxy cols that must individually exceed MIN_COL_SIM
MIN_COL_SIM                = 0.05  # individual column cosine sim floor used by domain fraction check
SIBLING_THRESHOLD          = 0.70  # max allowed sim between matched col and its nearest sibling col
CAT_MAX_CATEGORIES         = 15    # categorical cols with more unique values are skipped
CAT_SEM_SIM_THRESHOLD      = 0.25  # min cosine sim for semantic category mapping
CAT_GAP_DELTA              = 0.15  # values within this of max sim are treated as positive class

MIN_DISCOVERY_SCORE_ABS    = 0.05  # absolute floor — only removes truly garbage sources
MIN_DISCOVERY_SCORE_REL    = 0.35  # relative floor — drop sources below this × max_score
SELF_AUC_FLOOR             = 0.60  # drop sources whose features can't predict their own label

# Source quality filters (optional, controlled by --source-filters)
PROXY_SEM_SIM_THRESHOLD = 0.40   # filter "semantic": proxy col name must match concept list
POSRATE_MIN             = 0.05   # filter "posrate": drop sources with pos_rate below this
POSRATE_MAX             = 0.95   # filter "posrate": drop sources with pos_rate above this
DISTRIB_QUANTILE_CORR   = 0.20   # filter "distrib": min mean quantile corr with target cols
SANTOS_GAP_THRESHOLD    = 0.10   # filter "santos_pct": drop sources more than this far below the median SANTOS score
SANTOS_ABS_MIN          = 0.05   # minimum cutoff floor — never sets the gate lower than this

# Values that unambiguously assert the positive class, independent of domain
_AFFIRMATIVE = frozenset({"yes", "y", "true", "t", "1", "positive", "pos", "present", "found"})

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

        repurpose_col = candidates[tid]
        try:
            col_pos = cols.index(repurpose_col)
        except ValueError:
            col_pos = -1

        if col_pos >= 0 and n > 1:
            # SANTOS-style: score using only immediate neighbors of the repurposed column
            nb_embs = []
            if col_pos > 0:
                nb_embs.append(embs[col_pos - 1])
            if col_pos < n - 1:
                nb_embs.append(embs[col_pos + 1])
            if nb_embs:
                ctx_scores = [
                    float(np.dot(target_centroid_norm, nb / (np.linalg.norm(nb) + 1e-9)))
                    for nb in nb_embs
                ]
                sim = float(np.mean(ctx_scores))
                if sim >= threshold:
                    # Dual domain gate on remaining columns (all cols except proxy):
                    # 1. Full centroid ≥ TABLE_CENTROID_THRESHOLD — catches tables where
                    #    the average is too low (code complexity tables, COVID tables).
                    # 2. Fraction ≥ MIN_DOMAIN_FRACTION of cols with sim > MIN_COL_SIM —
                    #    catches tables where most cols are irrelevant even if average is OK
                    #    (e.g. cardInfo: card_id, right, top, bottom, width, height, suit, rank).
                    # Both must pass.
                    other_embs = np.delete(embs, col_pos, axis=0)
                    if len(other_embs) > 0:
                        per_col = np.array([
                            np.dot(target_centroid_norm, e / (np.linalg.norm(e) + 1e-9))
                            for e in other_embs
                        ])
                        full_centroid = other_embs.mean(axis=0)
                        fc_norm = np.linalg.norm(full_centroid)
                        full_sim = float(np.dot(target_centroid_norm, full_centroid / fc_norm)) if fc_norm > 1e-9 else 1.0
                        frac = float((per_col > MIN_COL_SIM).mean())
                        if full_sim >= TABLE_CENTROID_THRESHOLD and frac >= MIN_DOMAIN_FRACTION:
                            filtered[tid] = repurpose_col
                        else:
                            logger.debug(
                                "[DomainGate] Dropped '%s' col='%s'  santos=%.3f  centroid=%.3f  frac=%.2f",
                                tid, repurpose_col, sim, full_sim, frac,
                            )
                    else:
                        filtered[tid] = repurpose_col
                else:
                    logger.debug("[NeighborFilter] Dropped '%s' col='%s'  neighbor_ctx=%.3f",
                                 tid, repurpose_col, sim)
                continue

        # Fallback: table centroid (column not found in manifest or single-column table)
        centroid = embs.mean(axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 1e-9:
            sim = float(np.dot(target_centroid_norm, centroid / c_norm))
            if sim >= threshold:
                filtered[tid] = repurpose_col
            else:
                logger.debug("[CentroidFilter] Dropped '%s'  domain_sim=%.3f", tid, sim)
        else:
            filtered[tid] = repurpose_col

    logger.info(
        "[CentroidFilter] Done-cache: %d/%d candidates passed (threshold=%.2f)",
        len(filtered), len(candidates), threshold,
    )
    return filtered


def _compute_santos_scores(
    source_ids: list[str],
    label_names: dict[str, str],
    manifest_col_lookup: dict[str, list[str]],
    encoder: SentenceTransformer,
    target_centroid_norm: np.ndarray,
) -> dict[str, float]:
    """
    Compute a SANTOS-style neighbor context score for each source.

    For each source, embeds the manifest columns and returns the mean cosine
    similarity of the proxy column's immediate neighbors to target_centroid_norm.
    Falls back to the full table centroid when the proxy column has no neighbors
    or is not found in the manifest.
    """
    if not source_ids:
        return {}

    all_cols_flat: list[str] = []
    col_lists: list[list[str]] = []
    for tid in source_ids:
        cols = manifest_col_lookup.get(tid, [])
        col_lists.append(cols)
        all_cols_flat.extend(cols)

    if not all_cols_flat:
        return {tid: float("nan") for tid in source_ids}

    all_embs = encoder.encode(
        all_cols_flat, batch_size=512, show_progress_bar=False, convert_to_numpy=True,
    )

    scores: dict[str, float] = {}
    offset = 0
    for tid, cols in zip(source_ids, col_lists):
        n = len(cols)
        if n == 0:
            scores[tid] = float("nan")
            offset += n
            continue

        embs = all_embs[offset: offset + n]
        offset += n

        proxy_col = label_names.get(tid, "")
        try:
            col_pos = cols.index(proxy_col) if proxy_col else -1
        except ValueError:
            col_pos = -1

        if col_pos >= 0 and n > 1:
            nb_embs = []
            if col_pos > 0:
                nb_embs.append(embs[col_pos - 1])
            if col_pos < n - 1:
                nb_embs.append(embs[col_pos + 1])
            if nb_embs:
                scores[tid] = float(np.mean([
                    np.dot(target_centroid_norm, nb / (np.linalg.norm(nb) + 1e-9))
                    for nb in nb_embs
                ]))
                continue

        # Fallback: full table centroid
        centroid = embs.mean(axis=0)
        c_norm = np.linalg.norm(centroid)
        scores[tid] = float(np.dot(target_centroid_norm, centroid / c_norm)) if c_norm > 1e-9 else float("nan")

    return scores


def _mmr_select(
    scores: dict[str, float],
    label_names: dict[str, str],
    encoder: SentenceTransformer,
    k: int,
    lambda_: float = 0.7,
) -> dict[str, float]:
    """
    Select up to k tables via Maximum Marginal Relevance.

    Relevance  = normalised discovery score.
    Redundancy = cosine similarity of proxy-column name embeddings.
    lambda_=1.0 → pure relevance (current behaviour); 0.0 → pure diversity.
    """
    if len(scores) <= k:
        return scores

    tids = list(scores)
    proxy_cols = [label_names.get(t, "") for t in tids]
    embs = encoder.encode(proxy_cols, show_progress_bar=False, normalize_embeddings=True)
    emb_map = dict(zip(tids, embs))

    max_score = max(scores.values()) or 1.0
    selected: list[str] = []
    remaining = list(tids)

    while len(selected) < k and remaining:
        if not selected:
            best = max(remaining, key=lambda x: scores[x])
        else:
            sel_mat = np.array([emb_map[s] for s in selected])  # (n_sel, dim)
            def _score(x: str, _sm: np.ndarray = sel_mat) -> float:
                rel = scores[x] / max_score
                sim = float(np.max(emb_map[x] @ _sm.T))
                return lambda_ * rel - (1.0 - lambda_) * sim
            best = max(remaining, key=_score)
        selected.append(best)
        remaining.remove(best)

    return {t: scores[t] for t in selected}


def _qt_within_dataset(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    """Fit and apply a QuantileTransformer on each column using the DataFrame's own values.

    Each column maps to uniform [0, 1] based on its own distribution, so weekly,
    monthly, and yearly income all preserve relative rank within their own scale
    rather than collapsing to a boundary when mapped to the target's quantiles.
    NaN cells are preserved.
    """
    from sklearn.preprocessing import QuantileTransformer
    df = df.copy()
    cols = [c for c in num_cols if c in df.columns and df[c].nunique() > 1]
    if not cols:
        return df
    sub = df[cols].copy()
    nan_mask = sub.isna()
    for col in cols:
        med = float(sub[col].median())
        sub[col] = sub[col].fillna(med if not np.isnan(med) else 0.0)
    n_q = min(1000, max(10, len(df)))
    qt = QuantileTransformer(n_quantiles=n_q, output_distribution="uniform", random_state=42)
    transformed = qt.fit_transform(sub)
    sub_df = pd.DataFrame(transformed, columns=cols, index=df.index)
    sub_df[nan_mask] = np.nan
    df[cols] = sub_df
    return df


def _standardize_per_dataset(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    """Z-score each numeric column using the DataFrame's own mean/std."""
    df = df.copy()
    for col in num_cols:
        if col not in df.columns:
            continue
        vals = df[col]
        nan_mask = vals.isna()
        filled = vals.fillna(float(vals.median()) if not nan_mask.all() else 0.0)
        mu = float(filled.mean())
        sigma = float(filled.std())
        if sigma > 1e-9:
            result = (filled - mu) / sigma
            result[nan_mask] = np.nan
            df[col] = result
        else:
            df[col] = 0.0
    return df


def _augment_post_alignment(
    aligned: dict[str, pd.DataFrame],
    col_mappings: dict[str, dict[str, tuple[str, float]]],
    labeled_lake: dict[str, pd.DataFrame],
    target_features: pd.DataFrame,
    encoder: SentenceTransformer,
    target_label: str = "",
    min_key_sim: float = 0.70,
    min_coverage: float = 0.45,
    max_new_cols: int = 6,
    min_sources: int = 1,
    min_domain_sim: float = 0.40,
    n_bins: int = 8,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Post-alignment join augmentation — no re-alignment required.

    For each aligned source, find high-confidence join keys (sim ≥ min_key_sim).
    Aggregate the source's orphan numeric columns (not in mapping) by quantile-
    binned join-key values.  Add the aggregated statistics to BOTH the aligned
    source DataFrame and the target using the corresponding target-side key.

    Because source and target get identical aggregated values per key-bin, domain
    shift on these columns is zero by construction — but they still add predictive
    signal (e.g. mean HR income by age-group helps predict census income).

    Returns updated (aligned, target_features); does NOT re-run schema alignment.
    """
    from collections import defaultdict

    tgt_cols = list(target_features.columns)
    # Include target label name as a reference so income-related orphan columns
    # (salary, wages, income_32) pass domain relevance even when not in feature names
    ref_names = tgt_cols + ([target_label] if target_label else [])
    tgt_embs = encoder.encode(
        ref_names, convert_to_numpy=True, normalize_embeddings=True
    )

    # candidates[col_name] = list of (binned_key_series_for_target, agg_values_for_target,
    #                                  tid, binned_key_series_for_source, src_values)
    #  simplified: collect (tgt_mapped Series, src_mapped Series, tid) per orphan col
    cand_tgt: dict[str, list[pd.Series]] = defaultdict(list)
    cand_src: dict[str, dict[str, pd.Series]] = defaultdict(dict)  # col → {tid → Series}

    for tid, mapping in col_mappings.items():
        src_df_full = labeled_lake.get(tid)
        src_df_aligned = aligned.get(tid)
        if src_df_full is None or src_df_aligned is None:
            continue

        # High-confidence join keys
        join_keys = [
            (src_col, tgt_col, sim)
            for src_col, (tgt_col, sim) in mapping.items()
            if sim >= min_key_sim
            and src_col in src_df_full.columns
            and tgt_col in target_features.columns
            and pd.api.types.is_numeric_dtype(src_df_full[src_col])
            and pd.api.types.is_numeric_dtype(target_features[tgt_col])
        ]
        if not join_keys:
            continue

        # Orphan columns in the full source (not in mapping, not LABEL_COL, numeric)
        matched_src = set(mapping.keys())
        orphan_cols = [
            c for c in src_df_full.columns
            if c not in matched_src
            and c != LABEL_COL
            and c not in tgt_cols
            and pd.api.types.is_numeric_dtype(src_df_full[c])
            and src_df_full[c].nunique() > 3
            and _is_feature_col_name(c)
        ]
        if not orphan_cols:
            continue

        # Use the best join key (highest sim)
        src_key, tgt_key, _sim = max(join_keys, key=lambda x: x[2])

        # Bin the source key into quantile buckets for coverage on continuous keys
        src_key_vals = src_df_full[src_key].dropna()
        try:
            bin_labels, bin_edges = pd.qcut(
                src_key_vals, q=n_bins, duplicates="drop", labels=False, retbins=True
            )
        except Exception:
            continue

        # bin_labels: index→bin_number (0..k-1) for non-NaN rows
        bin_map = dict(zip(src_df_full.index[src_df_full[src_key].notna()], bin_labels))

        # Aggregate orphan columns by bin number
        src_bins_col = pd.Series(bin_map, name="__bin__")
        src_for_agg = src_df_full[orphan_cols].join(src_bins_col, how="inner")
        bin_agg = src_for_agg.groupby("__bin__")[orphan_cols].mean()
        # bin_agg: index=bin_number, columns=orphan_cols

        # Assign target rows to source bins using source's quantile edges
        tgt_key_vals = target_features[tgt_key].fillna(float(src_key_vals.median()))
        tgt_bins = pd.cut(
            tgt_key_vals, bins=bin_edges, labels=False, include_lowest=True
        )

        for oc in orphan_cols:
            if oc not in bin_agg.columns:
                continue
            agg_col = bin_agg[oc]  # Series: bin_number → mean_oc

            # Map target rows via their bin number
            tgt_mapped = tgt_bins.map(agg_col)
            coverage = float(tgt_mapped.notna().mean())
            if coverage < min_coverage:
                continue

            # Map aligned source rows via their bin number
            aligned_bins = pd.Series(
                {i: bin_map.get(i, np.nan) for i in src_df_aligned.index}
            )
            src_mapped = aligned_bins.map(agg_col)

            cand_tgt[oc].append(tgt_mapped.values)
            cand_src[oc][tid] = src_mapped

    if not cand_tgt:
        logger.info("[PostJoin] no join candidates found (no orphan cols passed key/coverage filters)")
        return aligned, target_features

    # Domain-relevance filter
    cand_names = list(cand_tgt.keys())
    cand_embs = encoder.encode(cand_names, convert_to_numpy=True, normalize_embeddings=True)
    sim_to_tgt = cand_embs @ tgt_embs.T
    max_sim_arr = sim_to_tgt.max(axis=1)
    domain_ok = {
        n for n, ms in zip(cand_names, max_sim_arr) if ms >= min_domain_sim
    }
    logger.debug("[PostJoin] candidates=%d domain_ok=%d (min_domain_sim=%.2f)",
                 len(cand_names), len(domain_ok), min_domain_sim)

    # Sort by source count
    sorted_cands = sorted(
        [(n, arrs) for n, arrs in cand_tgt.items() if n in domain_ok],
        key=lambda x: -len(x[1]),
    )

    aug_aligned = {tid: df.copy() for tid, df in aligned.items()}
    aug_target = target_features.copy()
    existing = set(target_features.columns)
    n_added = 0

    for col_name, tgt_arrays in sorted_cands:
        if n_added >= max_new_cols:
            break
        if col_name in existing or len(tgt_arrays) < min_sources:
            continue

        # Average target values across contributing sources
        tgt_combined = np.nanmean(np.stack(tgt_arrays, axis=0), axis=0)
        coverage = float(np.mean(~np.isnan(tgt_combined)))
        if coverage < min_coverage:
            continue

        aug_target[col_name] = tgt_combined
        existing.add(col_name)

        # Add to ALL aligned sources: contributors get real values, others get NaN
        # (XGBoost handles NaN via its built-in missing-value routing)
        for tid2 in aug_aligned:
            if tid2 in cand_src[col_name]:
                aug_aligned[tid2][col_name] = cand_src[col_name][tid2].values
            else:
                aug_aligned[tid2][col_name] = np.nan

        n_added += 1
        ms_val = float(max_sim_arr[cand_names.index(col_name)])
        logger.info(
            "[PostJoin] '%s' — %d source(s), coverage=%.0f%%, domain_sim=%.2f",
            col_name, len(tgt_arrays), coverage * 100, ms_val,
        )

    logger.info("[PostJoin] %d new column(s) injected into aligned sources + target", n_added)
    return aug_aligned, aug_target


def _augment_target_via_aligned_joins(
    col_mappings: dict[str, dict[str, tuple[str, float]]],
    labeled_lake: dict[str, pd.DataFrame],
    target_features: pd.DataFrame,
    encoder: SentenceTransformer,
    min_key_sim: float = 0.60,
    min_coverage: float = 0.50,
    max_new_cols: int = 8,
    min_sources: int = 2,
    min_domain_sim: float = 0.35,
) -> pd.DataFrame:
    """Augment the target with aggregated statistics from aligned source columns.

    After Step 2 alignment we know which source columns map to which target
    columns (the 'join keys').  For each aligned key with sim ≥ min_key_sim,
    we group the source's *unmatched* numeric columns by that key value and
    compute per-group means.  These are then mapped onto the target using the
    corresponding target column as a lookup key.

    Unlike regression-based imputation, the joined values are real aggregated
    observations from the source domain, not predictions derived from existing
    target features.  A second alignment pass then matches these new target
    columns to their source counterparts at sim ≈ 1.0, so previously-orphaned
    source columns now survive the min_similarity threshold.
    """
    from collections import defaultdict

    tgt_cols = list(target_features.columns)
    if not tgt_cols or not col_mappings:
        return target_features

    # Pre-embed target column names for domain-relevance check
    tgt_embs = encoder.encode(
        tgt_cols, convert_to_numpy=True, normalize_embeddings=True
    )

    # candidates[(orphan_col_name)] -> list of (mapped Series from one source)
    candidates: dict[str, list[pd.Series]] = defaultdict(list)

    for tid, mapping in col_mappings.items():
        src_df = labeled_lake.get(tid)
        if src_df is None:
            continue

        # Find high-confidence join keys for this source
        join_keys = [
            (src_col, tgt_col)
            for src_col, (tgt_col, sim) in mapping.items()
            if sim >= min_key_sim
            and src_col in src_df.columns
            and tgt_col in target_features.columns
            and pd.api.types.is_numeric_dtype(src_df[src_col])
            and pd.api.types.is_numeric_dtype(target_features[tgt_col])
        ]
        if not join_keys:
            continue

        # Identify orphan columns (not in mapping, numeric, reasonable cardinality)
        matched_src = set(mapping.keys())
        orphan_cols = [
            c for c in src_df.columns
            if c not in matched_src
            and c != LABEL_COL
            and pd.api.types.is_numeric_dtype(src_df[c])
            and src_df[c].nunique() > 3
        ]
        if not orphan_cols:
            continue

        # For each join key, aggregate orphans and map to target
        for src_key, tgt_key in join_keys:
            src_key_vals = src_df[src_key].dropna()
            tgt_key_vals = target_features[tgt_key]

            # Skip if the key has too many unique values to be a useful groupby key
            n_groups = src_key_vals.nunique()
            if n_groups < 2 or n_groups > 200:
                continue

            agg_df = src_df[[src_key] + orphan_cols].dropna(subset=[src_key])
            agg = agg_df.groupby(src_key)[orphan_cols].mean()

            for oc in orphan_cols:
                if oc in target_features.columns:
                    continue
                mapped = tgt_key_vals.map(agg[oc])
                coverage = float(mapped.notna().mean())
                if coverage < min_coverage:
                    continue
                candidates[oc].append(mapped)

    if not candidates:
        logger.debug("[AlignJoin] no join candidates found")
        return target_features

    # Domain-relevance filter: new column name must be similar to some target column
    candidate_names = list(candidates.keys())
    cand_embs = encoder.encode(
        candidate_names, convert_to_numpy=True, normalize_embeddings=True
    )
    sim_to_tgt = cand_embs @ tgt_embs.T
    max_sim = sim_to_tgt.max(axis=1)
    domain_ok = {
        n for n, ms in zip(candidate_names, max_sim)
        if ms >= min_domain_sim and _is_feature_col_name(n)
    }

    # Sort by source count (more sources = more reliable aggregate)
    sorted_cands = sorted(
        [(n, sl) for n, sl in candidates.items() if n in domain_ok],
        key=lambda x: -len(x[1]),
    )

    augmented = target_features.copy()
    existing = set(target_features.columns)
    n_added = 0

    for col_name, series_list in sorted_cands:
        if n_added >= max_new_cols:
            break
        if col_name in existing or len(series_list) < min_sources:
            continue
        combined = pd.concat(series_list, axis=1).mean(axis=1)
        coverage = float(combined.notna().mean())
        if coverage < min_coverage:
            continue
        augmented[col_name] = combined.values
        existing.add(col_name)
        n_added += 1
        logger.info(
            "[AlignJoin] '%s' — %d source(s), coverage=%.0f%%, domain_sim=%.2f",
            col_name, len(series_list), coverage * 100,
            float(max_sim[candidate_names.index(col_name)]),
        )

    logger.info("[AlignJoin] %d new column(s) added to target", n_added)
    return augmented


def _augment_target_with_orphans(
    col_mappings: dict[str, dict[str, tuple[str, float]]],
    labeled_lake: dict[str, pd.DataFrame],
    target_features: pd.DataFrame,
    encoder: SentenceTransformer,
    min_sources: int = 2,
    min_r2: float = 0.10,
    max_new_cols: int = 8,
    cluster_sim_threshold: float = 0.72,
) -> pd.DataFrame:
    """Predict orphan (unmatched) source columns for the target and add them as new target columns.

    After Step 2 alignment, sources often have numeric columns below min_similarity
    ("orphans"). When multiple sources share a semantically similar orphan column AND
    it can be predicted from already-aligned features (R² ≥ min_r2), we train a Ridge
    regressor (source-only normalisation — no target fitting) and predict those values
    for the target rows.  A second align_all pass with the augmented target then matches
    those columns at sim≈1.0 instead of discarding them.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import QuantileTransformer

    tgt_cols = list(target_features.columns)
    if not tgt_cols or not col_mappings:
        return target_features

    # --- 1. Collect orphan columns (numeric, >3 unique values, not matched) ---
    orphan_records: list[tuple[str, str]] = []  # (tid, orphan_col_name)
    for tid, mapping in col_mappings.items():
        src_df = labeled_lake.get(tid)
        if src_df is None:
            continue
        matched_src = set(mapping.keys())
        for col in src_df.columns:
            if col == LABEL_COL or col in matched_src:
                continue
            s = src_df[col].dropna()
            if not pd.api.types.is_numeric_dtype(s):
                continue
            if s.nunique() < 4 or len(s) < 20:
                continue
            orphan_records.append((tid, col))

    if not orphan_records:
        logger.debug("[Augment] no orphan columns found across %d sources", len(col_mappings))
        return target_features

    # --- 2. Embed orphan names; filter those redundant with existing target columns ---
    unique_orphan_names = list({col for _, col in orphan_records})
    orphan_embs = encoder.encode(
        unique_orphan_names, convert_to_numpy=True, normalize_embeddings=True
    )
    name_to_emb: dict[str, np.ndarray] = {
        n: orphan_embs[i] for i, n in enumerate(unique_orphan_names)
    }

    tgt_embs = encoder.encode(tgt_cols, convert_to_numpy=True, normalize_embeddings=True)
    sim_to_tgt = orphan_embs @ tgt_embs.T  # (n_orphans, n_tgt_cols)
    max_sim_to_tgt = sim_to_tgt.max(axis=1)
    valid_orphan_names = {
        n for n, ms in zip(unique_orphan_names, max_sim_to_tgt) if ms < 0.70
    }
    orphan_records = [(tid, col) for tid, col in orphan_records if col in valid_orphan_names]

    if not orphan_records:
        logger.debug("[Augment] all orphan columns too similar to existing target columns")
        return target_features

    # --- 3. Greedy cluster orphan names at cluster_sim_threshold ---
    filtered_names = list({col for _, col in orphan_records})
    filtered_embs = np.array([name_to_emb[n] for n in filtered_names])
    sim_matrix = filtered_embs @ filtered_embs.T

    clusters: list[tuple[str, list[str]]] = []
    assigned = [False] * len(filtered_names)
    for i, name in enumerate(filtered_names):
        if assigned[i]:
            continue
        members = [name]
        assigned[i] = True
        for j in range(i + 1, len(filtered_names)):
            if not assigned[j] and sim_matrix[i, j] >= cluster_sim_threshold:
                members.append(filtered_names[j])
                assigned[j] = True
        clusters.append((name, members))

    # Sort: prefer clusters with more sources (approximate by counting records)
    def _cluster_source_count(canonical_members: list[str]) -> int:
        mset = set(canonical_members)
        return len({tid for tid, col in orphan_records if col in mset})

    clusters.sort(key=lambda x: _cluster_source_count(x[1]), reverse=True)

    # --- 4. For each cluster: train Ridge per source, predict on target, average ---
    augmented = target_features.copy()
    n_added = 0

    for canonical, members in clusters:
        if n_added >= max_new_cols:
            break

        member_set = set(members)
        cluster_sources = [(tid, col) for tid, col in orphan_records if col in member_set]

        if len(cluster_sources) < min_sources:
            continue

        predictions: list[np.ndarray] = []

        for tid, col in cluster_sources:
            src_df = labeled_lake.get(tid)
            if src_df is None:
                continue
            mapping = col_mappings.get(tid, {})

            # Build matched (src_col, tgt_col) pairs in a deterministic order
            matched_pairs = [
                (src, tgt)
                for src, (tgt, _s) in sorted(mapping.items())
                if tgt in tgt_cols and src in src_df.columns
            ]
            if len(matched_pairs) < 2:
                continue

            src_rename = {src: tgt for src, tgt in matched_pairs}
            src_feat_cols = [src for src, _ in matched_pairs]
            tgt_feat_cols = [tgt for _, tgt in matched_pairs]

            src_feats = src_df[src_feat_cols].rename(columns=src_rename)
            orphan_vals = src_df[col]

            combined = pd.concat(
                [src_feats, orphan_vals.rename("__orphan__")], axis=1
            ).dropna()
            if len(combined) < 20:
                continue

            avail = [c for c in tgt_feat_cols if c in combined.columns]
            if len(avail) < 2:
                continue

            X_src = combined[avail].values.astype(float)
            y_src = combined["__orphan__"].values.astype(float)

            # Per-source QuantileTransformer — NEVER fit on target
            n_q = min(1000, max(10, len(X_src)))
            qt_x = QuantileTransformer(
                n_quantiles=n_q, output_distribution="normal", random_state=42
            )
            X_src_t = qt_x.fit_transform(X_src)

            qt_y = QuantileTransformer(
                n_quantiles=n_q, output_distribution="normal", random_state=42
            )
            y_src_t = qt_y.fit_transform(y_src.reshape(-1, 1)).ravel()

            model = Ridge(alpha=1.0)
            model.fit(X_src_t, y_src_t)

            r2 = float(model.score(X_src_t, y_src_t))
            if r2 < min_r2:
                logger.debug(
                    "[Augment] tid=%s col=%s R²=%.3f < threshold %.2f — skip",
                    tid, col, r2, min_r2,
                )
                continue

            # Predict on target (apply source-fitted QT to target feature values)
            missing = [c for c in avail if c not in augmented.columns]
            if missing:
                continue
            X_tgt = augmented[avail].copy()
            for c in avail:
                X_tgt[c] = X_tgt[c].fillna(float(augmented[c].median()))
            X_tgt_t = qt_x.transform(X_tgt.values.astype(float))
            y_pred_t = model.predict(X_tgt_t)
            y_pred = qt_y.inverse_transform(y_pred_t.reshape(-1, 1)).ravel()
            predictions.append(y_pred)
            logger.debug(
                "[Augment] tid=%s col=%s R²=%.3f — prediction accepted", tid, col, r2
            )

        if len(predictions) < min_sources:
            logger.debug(
                "[Augment] cluster '%s': only %d/%d sources passed R² gate — skip",
                canonical, len(predictions), len(cluster_sources),
            )
            continue

        pred_arr = np.mean(np.stack(predictions, axis=0), axis=0)
        augmented[canonical] = pred_arr
        n_added += 1
        logger.info(
            "[Augment] new column '%s' added (cluster=%d names, %d/%d sources passed R² gate)",
            canonical, len(members), len(predictions), len(cluster_sources),
        )

    logger.info("[Augment] target augmented with %d new column(s)", n_added)
    return augmented


_ID_LIKE_RE = re.compile(r"^(id|idx|index|row_id|rowid|row_num|rownum|seq|key|pk|uuid|guid)$", re.IGNORECASE)

def _is_feature_col_name(name: str) -> bool:
    """Return True if the column name looks like a real feature, not a survey question or row ID."""
    name = name.strip()
    if len(name) > 60:
        return False
    if len(name.split()) > 5:
        return False
    if "?" in name or name.endswith("."):
        return False
    if _ID_LIKE_RE.match(name):
        return False
    return True


def _augment_target_with_joins(
    labeled_lake: dict[str, pd.DataFrame],
    target_features: pd.DataFrame,
    encoder: SentenceTransformer,
    min_jaccard: float = 0.50,
    min_coverage: float = 0.50,
    max_new_cols: int = 10,
    min_domain_sim: float = 0.30,
) -> pd.DataFrame:
    """Expand the target schema by joining lake tables on shared categorical keys.

    For each low-cardinality categorical column in the target (≤30 unique values),
    search all repurposed source tables for columns whose unique value set has high
    Jaccard overlap.  When found, aggregate the source's numeric columns by that key
    and left-join into the target.

    The added columns contain real aggregated observations (not predictions), so
    they are genuinely new features.  After this augmentation, source columns that
    previously fell below min_similarity=0.35 can now find a match in the expanded
    target schema and survive alignment.

    Quality gates applied to candidate new columns:
    - Name must look like a real feature (not a survey question sentence)
    - Name must be semantically similar to at least one existing target column
      (cosine ≥ min_domain_sim) — keeps columns relevant to the target domain
    - ≥ min_coverage of target rows must receive a non-null joined value
    """
    from collections import defaultdict

    # --- 1. Identify joinable target columns (categorical, 2–30 unique values) ---
    tgt_keys: dict[str, frozenset] = {}
    for col in target_features.columns:
        s = target_features[col].dropna()
        n_uniq = s.nunique()
        if n_uniq < 2 or n_uniq > 30:
            continue
        if pd.api.types.is_object_dtype(s):
            tgt_keys[col] = frozenset(str(v).strip().lower() for v in s.unique())
        elif pd.api.types.is_numeric_dtype(s) and n_uniq <= 20:
            tgt_keys[col] = frozenset(s.unique())

    if not tgt_keys:
        logger.debug("[JoinAugment] no joinable target columns found")
        return target_features

    # Pre-embed target column names for domain-relevance check
    tgt_col_names = list(target_features.columns)
    tgt_col_embs = encoder.encode(
        tgt_col_names, convert_to_numpy=True, normalize_embeddings=True
    )  # (n_tgt, dim)

    # --- 2. Scan labeled_lake for columns with high Jaccard overlap to any tgt key ---
    candidates: dict[tuple[str, str], list[pd.Series]] = defaultdict(list)

    for tid, src_df in labeled_lake.items():
        src = src_df.drop(columns=[LABEL_COL], errors="ignore")
        for lake_col in src.columns:
            sv = src[lake_col].dropna()
            if len(sv) < 5:
                continue
            if pd.api.types.is_object_dtype(sv):
                lake_vals = frozenset(str(v).strip().lower() for v in sv.unique()
                                     if isinstance(v, str))
            elif sv.nunique() <= 30:
                lake_vals = frozenset(sv.unique())
            else:
                continue
            if len(lake_vals) < 2:
                continue

            for tgt_col, tgt_vals in tgt_keys.items():
                inter = len(tgt_vals & lake_vals)
                union = len(tgt_vals | lake_vals)
                jaccard = inter / union if union else 0.0
                if jaccard < min_jaccard:
                    continue

                # Candidate numeric columns to bring in
                num_cols = [
                    c for c in src.columns
                    if c != lake_col
                    and pd.api.types.is_numeric_dtype(src[c])
                    and src[c].nunique() > 3
                    and c not in target_features.columns
                    and _is_feature_col_name(c)
                ]
                if not num_cols:
                    continue

                agg = src[[lake_col] + num_cols].copy()
                if pd.api.types.is_object_dtype(agg[lake_col]):
                    agg[lake_col] = agg[lake_col].astype(str).str.strip().str.lower()
                agg = agg.groupby(lake_col)[num_cols].mean()

                tgt_series = target_features[tgt_col]
                if pd.api.types.is_object_dtype(tgt_series):
                    tgt_norm = tgt_series.astype(str).str.strip().str.lower()
                else:
                    tgt_norm = tgt_series

                for num_col in agg.columns:
                    mapped = tgt_norm.map(agg[num_col])
                    coverage = float(mapped.notna().mean())
                    if coverage < min_coverage:
                        continue
                    candidates[(tgt_col, num_col)].append(mapped)

    if not candidates:
        logger.debug("[JoinAugment] no join candidates passed initial gates")
        return target_features

    # --- 3. Domain-relevance filter: embed candidate col names, check sim to target ---
    unique_new_cols = list({num_col for _, num_col in candidates})
    new_col_embs = encoder.encode(
        unique_new_cols, convert_to_numpy=True, normalize_embeddings=True
    )
    sim_to_tgt = new_col_embs @ tgt_col_embs.T  # (n_new, n_tgt)
    max_sim = sim_to_tgt.max(axis=1)
    domain_ok = {
        name for name, ms in zip(unique_new_cols, max_sim) if ms >= min_domain_sim
    }
    candidates = {
        (tc, nc): sl for (tc, nc), sl in candidates.items() if nc in domain_ok
    }

    if not candidates:
        logger.debug("[JoinAugment] all candidates failed domain-relevance gate (min_sim=%.2f)", min_domain_sim)
        return target_features

    # --- 4. Add to target — sort by source count (more reliable first) ---
    sorted_candidates = sorted(candidates.items(), key=lambda kv: -len(kv[1]))

    augmented = target_features.copy()
    existing_cols = set(target_features.columns)
    n_added = 0

    for (tgt_col, num_col), series_list in sorted_candidates:
        if n_added >= max_new_cols:
            break
        if num_col in existing_cols:
            continue
        combined = pd.concat(series_list, axis=1).mean(axis=1)
        coverage = float(combined.notna().mean())
        if coverage < min_coverage:
            continue
        augmented[num_col] = combined.values
        existing_cols.add(num_col)
        n_added += 1
        logger.info(
            "[JoinAugment] '%s' (via key '%s') — %d source(s), coverage=%.0f%%",
            num_col, tgt_col, len(series_list), coverage * 100,
        )

    logger.info("[JoinAugment] %d new column(s) added to target", n_added)
    return augmented


def _otsu_threshold(vals: np.ndarray) -> float:
    """1D Otsu threshold — maximises between-class variance (minimises within-class variance)."""
    clean = vals[np.isfinite(vals)]
    if len(clean) < 10:
        return float(np.median(clean)) if len(clean) > 0 else 0.0
    hist, bin_edges = np.histogram(clean, bins=min(256, len(np.unique(clean))))
    if len(hist) == 0:
        return float(np.median(clean))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = float(hist.sum())
    best_thresh = float(bin_centers[0])
    best_var = -1.0
    w0 = 0.0
    sum0 = 0.0
    total_sum = float((hist * bin_centers).sum())
    for i in range(len(hist)):
        w0 += hist[i] / total
        w1 = 1.0 - w0
        if w0 < 1e-6 or w1 < 1e-6:
            continue
        sum0 += hist[i] * bin_centers[i]
        mu0 = sum0 / (w0 * total)
        mu1 = (total_sum - sum0) / (w1 * total)
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_thresh = float(bin_centers[i])
    return best_thresh


def _proxy_quality_score(col_vals: pd.Series, pos_rate: float) -> float:
    """
    Data-driven quality score in [0, 1] for a numeric proxy column.

    Three independent penalties, multiplied together:

    Cardinality — low unique-value count relative to row count signals an
    ordinal/Likert-scale proxy (survey "Household Income 1-5") rather than a
    real continuous measurement.  Threshold: full score when nunique ≥ 20% of rows.

    Coefficient of variation — aggregate/geographic proxies (state median income,
    per-capita income) have small spread relative to their mean (CV ≈ 0.15).
    Individual-level proxies (salary, earnings) have high CV (≥ 0.5).

    Balance — soft penalty when positive rate is far from 0.5.  Extreme rates
    (e.g., 8% positive) indicate the Otsu threshold landed poorly; the source
    contributes little signal across the class boundary.

    Returns 1.0 (neutral) for categorical proxies — cardinality and CV are not
    meaningful for string-valued columns.
    """
    clean = col_vals.dropna()
    n = len(clean)
    if n == 0:
        return 0.0

    nunique = int(clean.nunique())
    # Binary columns (nunique≤2) are ideal proxy labels for binary classification.
    # The ordinal/Likert penalty only makes sense for 3+ discretised levels.
    if nunique <= 2:
        cardinality = 1.0
    else:
        cardinality = min(1.0, nunique / max(1.0, 0.20 * n))

    mean_val = abs(float(clean.mean()))
    cv = float(clean.std()) / (mean_val + 1e-9) if mean_val > 1e-9 else 0.0
    cv_score = min(1.0, cv / 0.5)

    imbalance = max(0.0, abs(pos_rate - 0.5) - 0.25)
    balance = max(0.1, 1.0 - 2.0 * imbalance)

    return float(cardinality * cv_score * balance)


def _correlation_alignment_score(
    src_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    min_rows: int = 15,
) -> float:
    """
    Frobenius cosine similarity between the Spearman correlation matrices of
    source (aligned, normalised) and target feature columns.

    Spearman (rank-based) is invariant to monotonic transformations such as
    QuantileTransformer, so this is valid both before and after normalisation.

    Returns a value in [-1, 1]:
      ~1.0  identical dependency structure (good transfer candidate)
      ~0.0  uncorrelated structures (neutral)
      < 0   opposite structure (anti-useful — hard-drop)
    Falls back to 0.5 (neutral) on degenerate input.
    """
    from scipy.stats import rankdata as _rankdata

    shared = [
        c for c in src_df.columns
        if c in tgt_df.columns
        and pd.api.types.is_numeric_dtype(src_df[c])
        and pd.api.types.is_numeric_dtype(tgt_df[c])
    ]
    if len(shared) < 2 or len(src_df) < min_rows or len(tgt_df) < min_rows:
        return 0.5

    def _spearman(vals: np.ndarray) -> np.ndarray:
        ranked = np.apply_along_axis(_rankdata, 0, vals)
        C = np.corrcoef(ranked.T)
        return np.nan_to_num(C, nan=0.0)

    try:
        C_src = _spearman(src_df[shared].values.astype(float))
        C_tgt = _spearman(tgt_df[shared].values.astype(float))
        num   = float(np.sum(C_src * C_tgt))
        denom = float(np.linalg.norm(C_src, "fro") * np.linalg.norm(C_tgt, "fro"))
        if denom < 1e-9:
            return 0.5
        return float(np.clip(num / denom, -1.0, 1.0))
    except Exception:
        return 0.5


def _binarize_categorical(
    col_vals: pd.Series,
    encoder,
    concept_emb: np.ndarray,
    direction: str = "POSITIVE",
) -> Optional[tuple[pd.Series, str]]:
    """
    Binarize a non-numeric proxy column.

    Strategy (in order):
    1. Semantic similarity: embed each unique category value, compute cosine sim
       to the target concept centroid. Values within GAP_DELTA of the max sim
       are the positive class. Requires at least one value to exceed SEM_SIM_THRESHOLD.
    2. Affirmative-word fallback: if no value is semantically similar enough,
       map known affirmative tokens (yes/true/1/present/positive) to 1.

    Direction flip is applied after the positive-class assignment: if
    direction=NEGATIVE the 0/1 assignment is inverted so that the semantically
    DISTANT (or non-affirmative) values become the positive class.

    Returns (binarized_int_series, method_name) or None if determination fails.
    """
    clean = col_vals.dropna()
    unique_vals = clean.unique()
    n_unique = len(unique_vals)

    if n_unique < 2 or n_unique > CAT_MAX_CATEGORIES:
        return None

    str_vals = [str(v) for v in unique_vals]
    val_embs = encoder.encode(str_vals, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(val_embs, axis=1, keepdims=True)
    val_embs_norm = val_embs / np.where(norms > 1e-9, norms, 1.0)

    sims = val_embs_norm @ concept_emb  # (n_unique,)
    max_sim = float(sims.max())

    if max_sim >= CAT_SEM_SIM_THRESHOLD:
        positive_vals = {v for v, s in zip(unique_vals, sims) if s >= max_sim - CAT_GAP_DELTA}
        method = "categorical_semantic"
    else:
        positive_vals = {v for v in unique_vals if str(v).lower().strip() in _AFFIRMATIVE}
        if not positive_vals:
            return None
        method = "categorical_affirmative"

    if direction == "NEGATIVE":
        positive_vals = set(unique_vals) - positive_vals

    if not positive_vals or positive_vals == set(unique_vals):
        return None

    result = col_vals.map(lambda x: 1 if x in positive_vals else (0 if pd.notna(x) else np.nan))
    result = result.fillna(0).astype(int)
    return result, method


def _binarize_col(
    col_vals: pd.Series,
    neighbor_df: Optional[pd.DataFrame],
) -> tuple[float, str]:
    """
    Compute a data-driven binarization threshold using Otsu's method.

    Returns (threshold, method_name).
    method_name is one of: 'otsu', 'median_fallback'.
    """
    clean = col_vals.dropna()
    if len(clean) < 10 or clean.nunique() < 2:
        return float(col_vals.median()), "median_fallback"
    return float(_otsu_threshold(clean.values)), "otsu"


def _build_labeled_lake(
    repurpose_result: dict[str, str],
    id_to_path: dict[str, Path],
    target_pos_rate: Optional[float] = None,
    direction_cache: Optional[dict] = None,
    target_label: Optional[str] = None,
    diag_path: Optional[Path] = None,
    encoder=None,
    concept_emb: Optional[np.ndarray] = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, float]]:
    """
    Load and binarize the tables identified by `repurpose_result`.

    Parameters
    ----------
    repurpose_result : {table_id → best-matching column name}
    id_to_path       : {table_id → parquet Path}
    direction_cache  : {target_label → {col_name → "POSITIVE"|"NEGATIVE"}}
    target_label     : natural-language label name for direction lookup
    diag_path        : if set, save step0_repurposed.csv here
    encoder          : SentenceTransformer — required for categorical binarization
    concept_emb      : normalized concept centroid — required for categorical binarization

    Returns
    -------
    labeled_lake   : {table_id → DataFrame with LABEL_COL}
    label_names    : {table_id → original proxy column name}
    quality_scores : {table_id → float in [0,1]} proxy data-quality score
    """
    labeled_lake:   dict[str, pd.DataFrame] = {}
    label_names:    dict[str, str]          = {}
    quality_scores: dict[str, float]        = {}
    diag_rows: list[dict] = []

    target_directions: dict[str, str] = {}
    if direction_cache and target_label:
        target_directions = direction_cache.get(target_label, {})

    thresh_counts: dict[str, int] = {"otsu": 0, "median_fallback": 0}

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

        raw_col = df[repurpose_col]
        direction = target_directions.get(repurpose_col.lower(), "POSITIVE")

        # --- Categorical branch ---
        col_vals_numeric = raw_col.astype(float, errors="ignore")
        is_numeric = pd.api.types.is_numeric_dtype(col_vals_numeric)

        if not is_numeric:
            if encoder is None or concept_emb is None:
                continue
            n_unique = raw_col.dropna().nunique()
            if n_unique > CAT_MAX_CATEGORIES:
                logger.debug("  Skipping '%s' col='%s': too many categories (%d)",
                             table_id, repurpose_col, n_unique)
                if diag_path:
                    diag_rows.append({"table_id": table_id, "proxy_col": repurpose_col,
                                      "kept": False, "reason": "high_cardinality"})
                continue
            cat_result = _binarize_categorical(raw_col, encoder, concept_emb, direction)
            if cat_result is None:
                logger.debug("  Skipping '%s' col='%s': categorical binarization failed",
                             table_id, repurpose_col)
                if diag_path:
                    diag_rows.append({"table_id": table_id, "proxy_col": repurpose_col,
                                      "kept": False, "reason": "cat_no_mapping"})
                continue
            binarized, thresh_method = cat_result
            thresh_counts[thresh_method] = thresh_counts.get(thresh_method, 0) + 1
            binarize_thresh = float("nan")
            df[repurpose_col] = binarized
            pos_rate = float(df[repurpose_col].mean())
            proxy_quality = 1.0  # cardinality/CV not meaningful for categorical
        else:
            # --- Numeric branch ---
            col_vals = col_vals_numeric

            # High-cardinality check: columns with > 100 distinct raw values are likely IDs
            if col_vals.nunique() > 100:
                logger.debug("  Skipping '%s' col='%s': too many distinct values (%d > 100)",
                             table_id, repurpose_col, int(col_vals.nunique()))
                if diag_path:
                    diag_rows.append({"table_id": table_id, "proxy_col": repurpose_col,
                                      "kept": False, "reason": "high_cardinality"})
                continue

            binarize_thresh, thresh_method = _binarize_col(col_vals, None)
            thresh_counts[thresh_method] = thresh_counts.get(thresh_method, 0) + 1

            col_filled = col_vals.fillna(binarize_thresh)
            if direction == "NEGATIVE":
                if binarize_thresh == col_vals.min():
                    binarized = (col_filled < binarize_thresh).astype(int)
                else:
                    binarized = (col_filled <= binarize_thresh).astype(int)
            else:
                if binarize_thresh == col_vals.min():
                    binarized = (col_filled > binarize_thresh).astype(int)
                else:
                    binarized = (col_filled >= binarize_thresh).astype(int)

            df[repurpose_col] = binarized
            pos_rate = float(df[repurpose_col].mean())
            proxy_quality = _proxy_quality_score(col_vals, pos_rate)

        logger.debug(
            "  %-45s  col='%s'  dir=%s  method=%s  thresh=%.4g  pos=%.3f",
            table_id, repurpose_col, direction, thresh_method, binarize_thresh, pos_rate,
        )

        if df[repurpose_col].nunique() < 2 or pos_rate < 0.03 or pos_rate > 0.90:
            reason = "no_variance" if df[repurpose_col].nunique() < 2 else "extreme_pos_rate"
            logger.debug("  Skipping '%s' col='%s': %s (pos_rate=%.3f)", table_id, repurpose_col, reason, pos_rate)
            if diag_path:
                diag_rows.append({"table_id": table_id, "proxy_col": repurpose_col,
                                  "direction": direction, "thresh_method": thresh_method,
                                  "binarize_thresh": binarize_thresh, "n_rows": len(df),
                                  "positive_rate": pos_rate, "kept": False, "reason": reason})
            continue

        df = df.rename(columns={repurpose_col: LABEL_COL})
        labeled_lake[table_id]   = df
        label_names[table_id]    = repurpose_col
        quality_scores[table_id] = proxy_quality

        if diag_path:
            diag_rows.append({"table_id": table_id, "proxy_col": repurpose_col,
                               "direction": direction, "thresh_method": thresh_method,
                               "binarize_thresh": binarize_thresh, "n_rows": len(df),
                               "positive_rate": pos_rate, "kept": True, "reason": ""})

    # Deduplicate: drop tables with identical DataFrame content (same data under different IDs)
    seen_fps: set[int] = set()
    n_dedup = 0
    for tid in list(labeled_lake):
        fp = int(pd.util.hash_pandas_object(labeled_lake[tid]).sum())
        if fp in seen_fps:
            n_dedup += 1
            logger.debug("[Dedup] Dropped duplicate '%s'", tid)
            del labeled_lake[tid]
            del label_names[tid]
            quality_scores.pop(tid, None)
            for row in diag_rows:
                if row["table_id"] == tid and row.get("kept"):
                    row["kept"] = False
                    row["reason"] = "duplicate"
        else:
            seen_fps.add(fp)
    if n_dedup:
        logger.info("[Dedup] Dropped %d duplicate table(s)", n_dedup)

    # Cap: at most 3 tables per proxy_col concept to prevent any single concept dominating
    MAX_PER_COL = 3
    col_counts: dict[str, int] = {}
    for tid in list(labeled_lake):
        key = label_names[tid].lower().strip()
        col_counts[key] = col_counts.get(key, 0) + 1
        if col_counts[key] > MAX_PER_COL:
            del labeled_lake[tid]
            del label_names[tid]
            quality_scores.pop(tid, None)
            if diag_path:
                for row in diag_rows:
                    if row["table_id"] == tid and row.get("kept"):
                        row["kept"] = False
                        row["reason"] = "concept_cap"

    logger.info(
        "Labeled lake built: %d tables  [thresh: otsu=%d  median_fallback=%d  cat_semantic=%d  cat_affirmative=%d]",
        len(labeled_lake),
        thresh_counts.get("otsu", 0), thresh_counts.get("median_fallback", 0),
        thresh_counts.get("categorical_semantic", 0), thresh_counts.get("categorical_affirmative", 0),
    )
    for table_id in list(labeled_lake)[:10]:
        logger.info("  [sample] %-45s  col='%s'  pos_rate=%.3f",
                    table_id, label_names[table_id],
                    float(labeled_lake[table_id][LABEL_COL].mean()))

    if diag_path and diag_rows:
        diag_path.mkdir(parents=True, exist_ok=True)
        # Attach quality scores to the diagnostic
        qs_map = quality_scores
        for row in diag_rows:
            row["proxy_quality"] = round(qs_map.get(row["table_id"], float("nan")), 4)
        pd.DataFrame(diag_rows).to_csv(diag_path / "step0_repurposed.csv", index=False)
        logger.info("[Diag] step0_repurposed.csv → %d rows", len(diag_rows))

    return labeled_lake, label_names, quality_scores


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
    diag_path: Optional[Path] = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, float]]:
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

    # Concept centroid (normalized) — used by categorical binarization to identify positive class
    _concept_centroid = target_embs.mean(axis=0)
    _concept_centroid_n = np.linalg.norm(_concept_centroid)
    concept_centroid_norm = _concept_centroid / _concept_centroid_n if _concept_centroid_n > 1e-9 else _concept_centroid

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

    # --- Load direction cache ---
    _direction_cache_path = Path(__file__).parent / "data" / "direction_cache.json"
    direction_cache: dict = {}
    if _direction_cache_path.exists():
        try:
            with open(_direction_cache_path) as _f:
                direction_cache = json.load(_f)
            logger.info("Direction cache loaded: %d targets", len(direction_cache))
        except Exception as _e:
            logger.warning("Could not load direction cache: %s", _e)

    # --- Fast path: load from done-cache if the scan already completed ---
    if done_path.exists():
        try:
            with open(done_path) as f:
                cached_result: dict[str, str] = json.load(f)
            logger.info(
                "Loaded repurpose done-cache '%s': %d candidates (skipping full scan)",
                done_path.name, len(cached_result),
            )
            # Apply neighbor context filter to remove wrong-domain tables from legacy caches
            if target_centroid_norm is not None and cached_result:
                manifest_col_lookup = {e["table_id"]: e.get("columns", []) for e in sorted_entries}
                cached_result = _apply_centroid_filter(
                    cached_result, manifest_col_lookup, encoder,
                    target_centroid_norm, NEIGHBOR_CONTEXT_THRESHOLD,
                )
            return _build_labeled_lake(
                cached_result, id_to_path,
                target_pos_rate=target_pos_rate,
                direction_cache=direction_cache,
                target_label=label_name,
                diag_path=diag_path,
                encoder=encoder,
                concept_emb=concept_centroid_norm,
            )
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
                # 3. SANTOS neighbor context filter: the columns adjacent to the
                #    repurposed col should look like target features.  A "churn" col
                #    in a code table (neighbors: commits, lines) will fail; one in an
                #    HR table (neighbors: tenure, salary) will pass.
                if target_centroid_norm is not None and len(col_embs) > 1:
                    nb_embs = []
                    if best_idx > 0:
                        nb_embs.append(col_embs[best_idx - 1])
                    if best_idx < len(col_embs) - 1:
                        nb_embs.append(col_embs[best_idx + 1])
                    if nb_embs:
                        nb_ctx = float(np.mean([
                            np.dot(target_centroid_norm, nb / (np.linalg.norm(nb) + 1e-9))
                            for nb in nb_embs
                        ]))
                        if nb_ctx < NEIGHBOR_CONTEXT_THRESHOLD:
                            logger.debug(
                                "  NeighborCtx filter: %-45s  col='%s'  nb_ctx=%.3f — skipped",
                                table_id, cols[best_idx], nb_ctx,
                            )
                            continue
                        # Dual domain gate: centroid + fraction on remaining columns
                        if len(col_embs) > 1:
                            other_embs = np.delete(col_embs, best_idx, axis=0)
                            per_col = np.array([
                                np.dot(target_centroid_norm, e / (np.linalg.norm(e) + 1e-9))
                                for e in other_embs
                            ])
                            full_c = other_embs.mean(axis=0)
                            fc_n = np.linalg.norm(full_c)
                            full_ctx = float(np.dot(target_centroid_norm, full_c / fc_n)) if fc_n > 1e-9 else 1.0
                            frac = float((per_col > MIN_COL_SIM).mean())
                            if full_ctx < TABLE_CENTROID_THRESHOLD or frac < MIN_DOMAIN_FRACTION:
                                logger.debug(
                                    "  DomainGate: %-45s  col='%s'  centroid=%.3f  frac=%.2f — skipped",
                                    table_id, cols[best_idx], full_ctx, frac,
                                )
                                continue

                # 4. Sibling filter: reject if matched col has a very similar sibling.
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

    return _build_labeled_lake(
        repurpose_result, id_to_path,
        target_pos_rate=target_pos_rate,
        direction_cache=direction_cache,
        target_label=label_name,
        diag_path=diag_path,
        encoder=encoder,
        concept_emb=concept_centroid_norm,
    )


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
    neighbor_alpha: float = 0.2,
    normalization: str = "per-source",
    test_inject: bool = False,
    source_filters: str = "",
    filter_ablation: bool = False,
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
    diag_path = results_dir / "diagnostics"
    labeled_lake, label_names, proxy_quality_scores = _stream_load_and_repurpose(
        manifest_tables=manifest_tables,
        cache_dir=cache_dir,
        label_name=cfg.label_name,
        encoder=encoder,
        threshold=repurpose_threshold,
        target_features=target_features,
        target_pos_rate=float(target_df[LABEL_COL].mean()),
        concepts_override=[cfg.label_name] if no_expansion else None,
        sample_tag=lake_sample,
        diag_path=diag_path,
    )

    if not labeled_lake:
        logger.error("No labeled sources found — cannot run adaptation for '%s'.", target_name)
        return None

    n_lake_effective = len(manifest_tables)
    logger.info("%d unlabeled tables in lake (not repurposed)", n_lake_effective - len(labeled_lake))

    # ------------------------------------------------------------------ #
    # Planted-source injection (--test-inject)
    # Loads a noised copy of the target from data/inject_test/ into the
    # labeled lake in memory only.  No lake files are modified.
    # ------------------------------------------------------------------ #
    _inject_table_id: Optional[str] = None
    if test_inject:
        from inject_test_table import INJECT_DIR, TABLE_ID_PREFIX, make_table_id
        _inject_table_id = make_table_id(target_name)
        _inject_parquet   = INJECT_DIR / f"{_inject_table_id}.parquet"
        if not _inject_parquet.exists():
            logger.error(
                "[TestInject] parquet not found: %s — run 'python inject_test_table.py --target %s' first.",
                _inject_parquet, target_name,
            )
        else:
            _inj_df = pd.read_parquet(_inject_parquet)
            # Proxy col is the last column; binarize it as the pipeline would
            _inj_proxy = _inj_df.columns[-1]
            _inj_feat  = _inj_df.drop(columns=[_inj_proxy])
            _inj_label = (_inj_df[_inj_proxy].rank(pct=True) >= 0.5).astype(int)
            _inj_feat[LABEL_COL] = _inj_label.values
            labeled_lake[_inject_table_id]     = _inj_feat
            label_names[_inject_table_id]      = _inj_proxy
            proxy_quality_scores[_inject_table_id] = 1.0
            logger.info(
                "[TestInject] planted '%s' (proxy='%s', %d rows) into labeled lake — "
                "lake NOT modified on disk.",
                _inject_table_id, _inj_proxy, len(_inj_df),
            )

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

    # Adjust discovery scores by proxy data quality before top-K selection.
    # Ordinal/Likert proxies (low cardinality) and aggregate proxies (low CV)
    # are demoted; individual-level continuous proxies keep their score.
    scores = {tid: s * proxy_quality_scores.get(tid, 1.0) for tid, s in scores.items()}
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    logger.info("Discovery scores after quality adjustment (top 20):")
    for tbl, score in list(scores.items())[:20]:
        q = proxy_quality_scores.get(tbl, 1.0)
        logger.info("  %-50s  score=%.4f  quality=%.3f  col='%s'",
                    tbl, score, q, label_names.get(tbl, "?"))

    top_k_scores = _mmr_select(scores, label_names, encoder, top_k, lambda_=0.7)
    logger.info("Selected top-%d tables for adaptation (MMR λ=0.7):", len(top_k_scores))
    for tbl, score in top_k_scores.items():
        logger.info("  %-50s  %.4f  col='%s'", tbl, score, label_names.get(tbl, "?"))

    pd.Series(scores, name="similarity").to_csv(results_dir / "discovery_scores.csv")

    # Test-inject: report rank and score of planted source
    if _inject_table_id:
        _sorted = sorted(scores.items(), key=lambda x: -x[1])
        _rank = next((i + 1 for i, (t, _) in enumerate(_sorted) if t == _inject_table_id), None)
        _score = scores.get(_inject_table_id, 0.0)
        _in_topk = _inject_table_id in top_k_scores
        logger.info(
            "[TestInject] RESULT: planted source rank=%s/%d  score=%.4f  in_top_k=%s",
            _rank, len(scores), _score, _in_topk,
        )
        if _in_topk:
            logger.info("[TestInject] PASS — planted source selected for adaptation.")
        else:
            logger.warning("[TestInject] FAIL — planted source NOT selected for adaptation.")

    # Step 1 diagnostic
    diag_path.mkdir(parents=True, exist_ok=True)
    step1_rows = [
        {"table_id": tid, "discovery_score": s, "rank": i + 1,
         "proxy_col": label_names.get(tid, "?")}
        for i, (tid, s) in enumerate(scores.items())
    ]
    pd.DataFrame(step1_rows).to_csv(diag_path / "step1_discovery.csv", index=False)
    logger.info("[Diag] step1_discovery.csv → %d sources ranked", len(step1_rows))

    # ------------------------------------------------------------------ #
    # Step 2a: Join-based target augmentation (DISABLED — requires full-lake
    # value-set index to avoid noise from off-domain sources in top-k pool)
    # _augment_target_with_joins() is implemented but not called until a
    # MinHash/inverted-index over all 421k tables is built.
    # ------------------------------------------------------------------ #
    lake_top_k = {k: labeled_lake[k] for k in top_k_scores}

    # ------------------------------------------------------------------ #
    # Step 2: Schema Alignment
    # ------------------------------------------------------------------ #
    logger.info("=== Step 2: Schema Alignment ===")
    _min_sim = 0.35
    _min_cols = 2
    aligned, col_mappings = schema_alignment.align_all(
        lake=lake_top_k,
        target=target_features,
        discovery_scores=top_k_scores,
        model=encoder,
        label_col=LABEL_COL,
        dist_threshold=DIST_THRESHOLD,
        min_coverage=0.0,
        min_similarity=_min_sim,
        min_cols=_min_cols,
        fill_unmatched="nan",
        neighbor_alpha=neighbor_alpha,
    )
    # Intermediate fallback: relax min_cols (1-col matches are still meaningful)
    # before giving up on the quality similarity threshold entirely.
    # Targets with few repurposed sources (e.g., heart with 4 tables) benefit from
    # this: tables whose single matched column has sim ≥ 0.35 contribute as quality
    # sources rather than being demoted to volume-only.
    if len(aligned) < min(4, len(lake_top_k)):
        _min_cols = 1
        _aligned2, _col_map2 = schema_alignment.align_all(
            lake=lake_top_k,
            target=target_features,
            discovery_scores=top_k_scores,
            model=encoder,
            label_col=LABEL_COL,
            dist_threshold=DIST_THRESHOLD,
            min_coverage=0.0,
            min_similarity=_min_sim,
            min_cols=_min_cols,
            fill_unmatched="nan",
            neighbor_alpha=neighbor_alpha,
        )
        if len(_aligned2) > len(aligned):
            logger.info(
                "min_cols fallback: %d → %d sources (min_cols=2→1, min_similarity=%.2f unchanged)",
                len(aligned), len(_aligned2), _min_sim,
            )
            aligned, col_mappings = _aligned2, _col_map2
    if len(aligned) < 2:
        # Final fallback: drop similarity threshold entirely so the pipeline can run.
        logger.warning(
            "min_similarity=%.2f dropped all but %d source(s) — retrying with 0.0",
            _min_sim, len(aligned),
        )
        _min_sim = 0.0
        _min_cols = 2
        aligned, col_mappings = schema_alignment.align_all(
            lake=lake_top_k,
            target=target_features,
            discovery_scores=top_k_scores,
            model=encoder,
            label_col=LABEL_COL,
            dist_threshold=DIST_THRESHOLD,
            min_coverage=0.0,
            min_similarity=_min_sim,
            min_cols=_min_cols,
            fill_unmatched="nan",
            neighbor_alpha=neighbor_alpha,
        )

    # ------------------------------------------------------------------ #
    # Volume sources: top-K tables that failed quality alignment.
    # Re-align with no threshold and use features-only as extra SOURCE-side
    # domain data for DANN — gives the discriminator more source volume without
    # polluting the label predictor with noisy pseudo-labels.
    # ------------------------------------------------------------------ #
    _quality_tids = set(aligned.keys())
    _volume_lake_k = {tid: df for tid, df in lake_top_k.items() if tid not in _quality_tids}
    volume_src_features: Optional[pd.DataFrame] = None
    if _volume_lake_k:
        try:
            _vol_aligned, _ = schema_alignment.align_all(
                lake=_volume_lake_k,
                target=target_features,
                discovery_scores={tid: top_k_scores.get(tid, 0.0) for tid in _volume_lake_k},
                model=encoder,
                label_col=LABEL_COL,
                dist_threshold=float("inf"),
                min_coverage=0.0,
                min_similarity=0.0,
                min_cols=1,
                fill_unmatched="nan",
                neighbor_alpha=neighbor_alpha,
            )
            _vol_frames = [
                df.drop(columns=[LABEL_COL], errors="ignore")
                for df in _vol_aligned.values()
            ]
            if _vol_frames:
                volume_src_features = pd.concat(_vol_frames, ignore_index=True)
                logger.info(
                    "[VolumeAug] %d volume sources (%d rows) → DANN SOURCE-side domain discriminator",
                    len(_vol_frames), len(volume_src_features),
                )
        except Exception as _ve:
            logger.warning("[VolumeAug] Failed to build volume sources: %s", _ve)

    # ------------------------------------------------------------------ #
    # Step 2b: Post-alignment join augmentation
    # For each aligned (src_col → tgt_col) pair with sim ≥ 0.60, aggregate
    # the source's orphan columns by binned join-key value and map to both
    # the aligned source DataFrame and the target.  No re-alignment needed —
    # columns are injected directly, preserving all existing matches.
    # ------------------------------------------------------------------ #
    if col_mappings:
        aligned, target_features = _augment_post_alignment(
            aligned=aligned,
            col_mappings=col_mappings,
            labeled_lake=labeled_lake,
            target_features=target_features,
            encoder=encoder,
            target_label=cfg.label_name,
        )
        # Keep target_train_df schema consistent — add new cols filled with the
        # test-set median so dropna() in run_oracle doesn't kill all rows.
        # Oracle ignores these during training (no variation = no split),
        # so it has no information leakage while avoiding a feature mismatch.
        new_aug_cols = [c for c in target_features.columns if c not in target_train_df.columns]
        for _c in new_aug_cols:
            _fill = float(target_features[_c].median()) if target_features[_c].notna().any() else 0.0
            target_train_df[_c] = _fill

    # Step 2 diagnostic: actual src_col → tgt_col mapping with similarity scores
    _step2_rows = []
    for tid, mapping in col_mappings.items():
        for src_col, (tgt_col, sim) in mapping.items():
            _step2_rows.append({
                "table_id": tid,
                "src_col": src_col,
                "tgt_col": tgt_col,
                "sim": round(sim, 4),
                "proxy_col": label_names.get(tid, "?"),
            })
    if _step2_rows:
        pd.DataFrame(_step2_rows).to_csv(diag_path / "step2_alignment.csv", index=False)
        logger.info("[Diag] step2_alignment.csv → %d column mappings", len(_step2_rows))

    # Discovery score gate: drop sources whose discovery score is too far below
    # the best source in this run.  Uses a relative threshold (35% of max_score)
    # with an absolute floor (0.05) so the threshold adapts per-target.
    # For adult (max≈0.65) this removes garbage tables at score<0.23.
    # For churn (max≈0.09) the floor of 0.05 applies and nearly all sources pass.
    _pre_score_gate = len(aligned)
    _max_score = max((top_k_scores.get(t, 0.0) for t in aligned), default=0.0)
    _score_threshold = max(MIN_DISCOVERY_SCORE_ABS, MIN_DISCOVERY_SCORE_REL * _max_score)
    _score_dropped = []
    for _tid in list(aligned):
        _sc = top_k_scores.get(_tid, 0.0)
        if _sc < _score_threshold:
            _score_dropped.append((_tid, _sc))
            del aligned[_tid]
    if _score_dropped:
        logger.info(
            "Score gate: dropped %d/%d sources (threshold=%.3f = max(%.2f, %.2f×%.3f))",
            len(_score_dropped), _pre_score_gate,
            _score_threshold, MIN_DISCOVERY_SCORE_ABS, MIN_DISCOVERY_SCORE_REL, _max_score,
        )
        for _tid, _sc in _score_dropped:
            logger.debug("  [ScoreGate] dropped '%s'  score=%.4f", _tid, _sc)
    else:
        logger.info("Score gate: all %d sources passed (threshold=%.3f)", _pre_score_gate, _score_threshold)

    target_norm       = target_features
    target_train_norm = target_train_df

    if normalization == "per-source":
        num_cols = [c for c in target_features.columns if pd.api.types.is_numeric_dtype(target_features[c])]
        aligned = {
            tid: _qt_within_dataset(df, [c for c in num_cols if c in df.columns and c != LABEL_COL])
            for tid, df in aligned.items()
        }
        target_norm       = _qt_within_dataset(target_features, num_cols)
        ttn               = _qt_within_dataset(target_train_df.drop(columns=[LABEL_COL]), num_cols)
        ttn[LABEL_COL]    = target_train_df[LABEL_COL].values
        target_train_norm = ttn
        logger.info("Normalization: per-source QuantileTransformer applied to %d sources + target", len(aligned))
    elif normalization == "target-fitted":
        from act4_openml_lake import _make_quantile_normalizer, _apply_quantile_norm
        qt, num_cols   = _make_quantile_normalizer(target_features)
        aligned        = {k: _apply_quantile_norm(v, qt, num_cols) for k, v in aligned.items()}
        target_norm    = _apply_quantile_norm(target_features, qt, num_cols)
        ttn            = _apply_quantile_norm(target_train_df.drop(columns=[LABEL_COL]), qt, num_cols)
        ttn[LABEL_COL] = target_train_df[LABEL_COL].values
        target_train_norm = ttn
        logger.info("Normalization: target-fitted QuantileTransformer applied")

    # ------------------------------------------------------------------ #
    # Source self-AUC diagnostic
    # For each aligned source, train XGBoost on its own (aligned) features
    # to predict its own binarized label, via cross-validation.
    # self_auc ≈ 0.5  → proxy label is noise for that source
    # self_auc < 0.5  → polarity inversion (label direction is flipped)
    # self_auc > 0.7  → coherent source: features genuinely predict the proxy
    # ------------------------------------------------------------------ #
    from sklearn.metrics import roc_auc_score as _roc_auc
    from xgboost import XGBClassifier as _XGB

    _self_auc_rows = []
    for _tid, _df in aligned.items():
        _X = _df.drop(columns=[LABEL_COL])
        _y = _df[LABEL_COL]
        if _y.nunique() < 2 or len(_df) < 10:
            continue
        _X_imp = _X.copy()
        for _col in _X_imp.columns:
            _med = float(_X_imp[_col].median())
            _X_imp[_col] = _X_imp[_col].fillna(0.0 if np.isnan(_med) else _med)
        try:
            _clf = _XGB(n_estimators=50, max_depth=3, random_state=42,
                        eval_metric="logloss", verbosity=0)
            _clf.fit(_X_imp, _y)
            _auc_cv = float(_roc_auc(_y, _clf.predict_proba(_X_imp)[:, 1]))
        except Exception as _exc:
            logger.debug("[SelfAUC] %s failed: %s", _tid, _exc)
            _auc_cv = float("nan")
        _n_aligned = int((_X.notna().any()).sum())
        _self_auc_rows.append({
            "table_id": _tid,
            "proxy_col": label_names.get(_tid, "?"),
            "n_rows": len(_df),
            "n_aligned_cols": _n_aligned,
            "self_auc": round(_auc_cv, 4),
            "discovery_score": round(top_k_scores.get(_tid, 0.0), 4),
        })
        logger.info(
            "[SelfAUC] %-40s  proxy=%-20s  n=%4d  n_cols=%2d  self_auc=%.3f  disc=%.3f",
            _tid[:40], label_names.get(_tid, "?")[:20], len(_df),
            _n_aligned, _auc_cv, top_k_scores.get(_tid, 0.0),
        )
    # Store scores for use by the selfauc filter in ablation
    _self_auc_scores: dict[str, float] = {}
    if _self_auc_rows:
        _sauc_df = pd.DataFrame(_self_auc_rows).sort_values("self_auc", ascending=False)
        _sauc_df.to_csv(diag_path / "source_self_auc.csv", index=False)
        _self_auc_scores = {r["table_id"]: r["self_auc"] for r in _self_auc_rows}
        _med = float(_sauc_df["self_auc"].median())
        logger.info(
            "[Diag] source_self_auc.csv → %d sources  median=%.3f  "
            "n_below_0.6=%d  n_above_0.7=%d",
            len(_self_auc_rows), _med,
            int((_sauc_df["self_auc"] < SELF_AUC_FLOOR).sum()),
            int((_sauc_df["self_auc"] > 0.7).sum()),
        )
        if not filter_ablation:
            # In normal mode apply the gate immediately; in ablation it becomes an optional filter
            _sauc_dropped = [r["table_id"] for r in _self_auc_rows
                             if r["self_auc"] < SELF_AUC_FLOOR and not np.isnan(r["self_auc"])]
            for _tid in _sauc_dropped:
                if _tid in aligned:
                    del aligned[_tid]
            if _sauc_dropped:
                logger.info(
                    "[SelfAUC gate] dropped %d structureless sources (self_auc < %.2f): %s",
                    len(_sauc_dropped), SELF_AUC_FLOOR,
                    ", ".join(f"{t[:30]}({next(r['self_auc'] for r in _self_auc_rows if r['table_id']==t):.2f})"
                              for t in _sauc_dropped),
                )

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
    # Source quality filters + filter ablation
    # ------------------------------------------------------------------ #
    # Pre-compute concept embeddings once for the semantic filter
    _concept_list = table_discovery.expand_label_via_llm(cfg.label_name)
    _concept_embs = encoder.encode(_concept_list, batch_size=256, normalize_embeddings=True)

    # Snapshot before any optional gates — shared across all ablation combos.
    # In normal mode this snapshot is taken post-selfauc-gate (gate already applied above).
    # In ablation mode the gate was skipped above, so this snapshot includes all score-gated sources.
    _aligned_base = dict(aligned)

    # Pre-compute SANTOS scores for santos_pct filter.
    # Uses the same neighbor-context scoring as _apply_centroid_filter but returns
    # raw scores (no filtering) so we can apply a per-target percentile cutoff.
    _tcols_filter = [c for c in target_features.columns if not str(c).startswith("Unnamed")]
    _tc_embs_filter = encoder.encode(_tcols_filter, show_progress_bar=False, convert_to_numpy=True)
    _tc_filter = _tc_embs_filter.mean(axis=0)
    _tc_filter_n = np.linalg.norm(_tc_filter)
    _filter_centroid = _tc_filter / _tc_filter_n if _tc_filter_n > 1e-9 else _tc_filter
    _filter_manifest_lookup = {e["table_id"]: e.get("columns", []) for e in manifest_tables}
    _santos_scores_aligned = _compute_santos_scores(
        list(_aligned_base.keys()), label_names, _filter_manifest_lookup,
        encoder, _filter_centroid,
    )
    logger.info(
        "[SANTOSScores] Computed for %d sources. Min=%.3f  Median=%.3f  Max=%.3f",
        len(_santos_scores_aligned),
        float(np.nanmin(list(_santos_scores_aligned.values()))) if _santos_scores_aligned else float("nan"),
        float(np.nanmedian(list(_santos_scores_aligned.values()))) if _santos_scores_aligned else float("nan"),
        float(np.nanmax(list(_santos_scores_aligned.values()))) if _santos_scores_aligned else float("nan"),
    )

    def _apply_filters(src: dict, combo: frozenset) -> dict:
        """Return a filtered copy of src for the given filter combination, with verbose logging."""
        result = dict(src)
        n_in = len(result)

        if "selfauc" in combo:
            _sa_rows = []
            for _tid in list(result):
                _score = _self_auc_scores.get(_tid, float("nan"))
                _proxy = label_names.get(_tid, "?")
                _keep = np.isnan(_score) or _score >= SELF_AUC_FLOOR
                _sa_rows.append((_tid, _proxy, _score, _keep))
                if not _keep:
                    del result[_tid]
            logger.info(
                "[SelfAUCFilter] %d/%d kept (floor=%.2f):",
                sum(r[3] for r in _sa_rows), n_in, SELF_AUC_FLOOR,
            )
            for _tid, _px, _sc, _ok in sorted(_sa_rows, key=lambda x: x[2], reverse=True):
                logger.info(
                    "  %s  proxy=%-18s  self_auc=%.3f  → %s",
                    "KEEP" if _ok else "DROP", _px[:18], _sc, _tid[:40],
                )

        if "semantic" in combo:
            _sem_rows = []
            for _tid in list(result):
                _proxy = label_names.get(_tid, "")
                if not _proxy:
                    _sem_rows.append((_tid, _proxy, float("nan"), True))
                    continue
                _pe = encoder.encode([_proxy], normalize_embeddings=True)[0]
                _ms = float((_concept_embs @ _pe).max())
                _keep = _ms >= PROXY_SEM_SIM_THRESHOLD
                _sem_rows.append((_tid, _proxy, _ms, _keep))
                if not _keep:
                    del result[_tid]
            logger.info(
                "[SemanticFilter] %d/%d kept (threshold=%.2f):",
                sum(r[3] for r in _sem_rows), n_in, PROXY_SEM_SIM_THRESHOLD,
            )
            for _tid, _px, _ms, _ok in sorted(_sem_rows, key=lambda x: x[2] if not np.isnan(x[2]) else 0, reverse=True):
                logger.info(
                    "  %s  proxy=%-18s  max_sim=%.3f  → %s",
                    "KEEP" if _ok else "DROP", _px[:18],
                    _ms if not np.isnan(_ms) else -1.0, _tid[:40],
                )

        if "posrate" in combo:
            _pr_rows = []
            for _tid in list(result):
                _pr = float(result[_tid][LABEL_COL].mean())
                _keep = POSRATE_MIN <= _pr <= POSRATE_MAX
                _pr_rows.append((_tid, label_names.get(_tid, "?"), _pr, _keep))
                if not _keep:
                    del result[_tid]
            logger.info(
                "[PosRateFilter] %d/%d kept ([%.2f, %.2f]):",
                sum(r[3] for r in _pr_rows), n_in, POSRATE_MIN, POSRATE_MAX,
            )
            for _tid, _px, _pr, _ok in sorted(_pr_rows, key=lambda x: x[2]):
                logger.info(
                    "  %s  proxy=%-18s  pos_rate=%.3f  → %s",
                    "KEEP" if _ok else "DROP", _px[:18], _pr, _tid[:40],
                )

        if "distrib" in combo:
            _dist_rows = []
            _q_pts = np.linspace(0, 1, 11)[1:-1]
            for _tid in list(result):
                _df_s = result[_tid]
                _feat_cols = [c for c in _df_s.columns if c != LABEL_COL and c in target_norm.columns]
                _corrs: list[float] = []
                for col in _feat_cols:
                    sv = _df_s[col].dropna().values
                    tv = target_norm[col].dropna().values
                    if len(sv) < 10 or len(tv) < 10:
                        continue
                    sq = np.quantile(sv, _q_pts)
                    tq = np.quantile(tv, _q_pts)
                    if np.std(sq) < 1e-9 or np.std(tq) < 1e-9:
                        continue
                    c = float(np.corrcoef(sq, tq)[0, 1])
                    if not np.isnan(c):
                        _corrs.append(c)
                _mean_corr = float(np.mean(_corrs)) if _corrs else float("nan")
                _keep = np.isnan(_mean_corr) or _mean_corr >= DISTRIB_QUANTILE_CORR
                _dist_rows.append((_tid, label_names.get(_tid, "?"), _mean_corr, _keep))
                if not _keep:
                    del result[_tid]
            logger.info(
                "[DistribFilter] %d/%d kept (threshold=%.2f):",
                sum(r[3] for r in _dist_rows), n_in, DISTRIB_QUANTILE_CORR,
            )
            for _tid, _px, _mc, _ok in sorted(_dist_rows, key=lambda x: x[2] if not np.isnan(x[2]) else 0):
                logger.info(
                    "  %s  proxy=%-18s  mean_corr=%.3f  → %s",
                    "KEEP" if _ok else "DROP", _px[:18],
                    _mc if not np.isnan(_mc) else -1.0, _tid[:40],
                )

        if "santos_pct" in combo:
            _sp_rows = []
            _sp_vals = [_santos_scores_aligned.get(tid, float("nan")) for tid in result]
            _finite = [v for v in _sp_vals if not np.isnan(v)]
            if _finite:
                _median = float(np.median(_finite))
                # Drop sources more than GAP_THRESHOLD below the median.
                # Floor at ABS_MIN so the gate never triggers on uniformly weak targets.
                _cutoff = max(_median - SANTOS_GAP_THRESHOLD, SANTOS_ABS_MIN)
            else:
                _cutoff = 0.0
                _median = float("nan")
            for _tid in list(result):
                _sc = _santos_scores_aligned.get(_tid, float("nan"))
                _keep = np.isnan(_sc) or _sc >= _cutoff
                _sp_rows.append((_tid, label_names.get(_tid, "?"), _sc, _keep))
                if not _keep:
                    del result[_tid]
            logger.info(
                "[SANTOSGapFilter] %d/%d kept (median=%.3f  gap=%.2f  cutoff=%.3f):",
                sum(r[3] for r in _sp_rows), n_in,
                _median if not np.isnan(_median) else -1.0,
                SANTOS_GAP_THRESHOLD, _cutoff,
            )
            for _tid, _px, _sc, _ok in sorted(_sp_rows, key=lambda x: x[2] if not np.isnan(x[2]) else 0):
                logger.info(
                    "  %s  proxy=%-18s  santos=%.3f  → %s",
                    "KEEP" if _ok else "DROP", _px[:18],
                    _sc if not np.isnan(_sc) else -1.0, _tid[:40],
                )

        logger.info("[FilterCombo] %d → %d sources after combo [%s]", n_in, len(result), _combo_label(combo))
        return result

    def _combo_label(c: frozenset) -> str:
        return "+".join(sorted(c)) if c else "none"

    _ALL_FILTERS = ("selfauc", "semantic", "posrate", "distrib", "santos_pct")
    if filter_ablation:
        _run_combos: list[frozenset] = [
            frozenset(_ALL_FILTERS[i] for i in range(len(_ALL_FILTERS)) if (k >> i) & 1)
            for k in range(1 << len(_ALL_FILTERS))
        ]
        _ablation_dir = results_dir / "filter_ablation"
        _ablation_dir.mkdir(parents=True, exist_ok=True)
        _ablation_auc: dict[str, pd.Series] = {}
        logger.info("=== Filter ablation: %d combos (5 filters × 2^5) ===", len(_run_combos))
    else:
        _parsed = frozenset(f.strip() for f in source_filters.split(",") if f.strip())
        _unknown = _parsed - set(_ALL_FILTERS)
        if _unknown:
            raise ValueError(f"Unknown source filters: {_unknown}. Choose from {_ALL_FILTERS}")
        _run_combos = [_parsed]

    # ------------------------------------------------------------------ #
    # Step 3 + 4: Domain Adaptation + Evaluation (multi-seed)
    # Runs once per filter combo (ablation) or once (normal mode)
    # ------------------------------------------------------------------ #
    target_pos_rate_test = float(y_true.mean())
    logger.info("Target positive rate (threshold calibration): %.3f", target_pos_rate_test)

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

    _last_summary_cal: Optional[pd.DataFrame] = None

    for _active_combo in _run_combos:
        _cname = _combo_label(_active_combo)
        if filter_ablation:
            logger.info("--- Filter combo: [%s] ---", _cname)
        _results_dir_combo = (_ablation_dir / _cname) if filter_ablation else results_dir
        _results_dir_combo.mkdir(parents=True, exist_ok=True)

        # Apply filters to base snapshot
        aligned = _apply_filters(_aligned_base, _active_combo)

        logger.info("=== Step 3: Domain Adaptation [%s] (%d sources) ===", _cname, len(aligned))
        if not aligned:
            logger.warning(
                "[%s] No aligned sources after filters — only baselines and oracle computed.", _cname
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
                    volume_src=volume_src_features,
                    random_state=seed,
                )
            else:
                results = {}

            results["baseline_a"] = domain_adaptation.run_baseline_majority(target_norm)
            raw_source_list = list(labeled_lake.values())
            results["baseline_b"] = domain_adaptation.run_baseline_random(
                raw_source_list, target_norm, LABEL_COL
            )
            results["oracle"] = domain_adaptation.run_oracle(
                target_train=target_train_norm,
                target_test=target_norm,
                label_col=LABEL_COL,
                random_state=seed,
            )

            if llm_baseline and seed == 0:
                try:
                    import llm_baseline as _llm_bl
                    logger.info("=== LLM zero-shot baseline ===")
                    results["llm_zero_shot"] = _llm_bl.run_zero_shot(
                        target_df=target_test_df,
                        label_col=LABEL_COL,
                        label_description=cfg.label_name,
                    )
                except Exception as exc:
                    logger.warning("LLM baseline failed: %s — skipping.", exc)

            logger.info("=== Step 4: Evaluation [%s] ===", _cname)
            seed_metrics_raw.append(evaluation.evaluate(results, y_true))
            seed_metrics_cal.append(
                evaluation.evaluate(results, y_true, target_pos_rate=target_pos_rate_test)
            )

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

        _n_src_after = len(aligned)
        print("\n" + "=" * 65)
        print("ACT 5 RESULTS — Data Lake experiment (no external labels)")
        print(f"  Target       : {target_desc}")
        print(f"  Lake         : {n_lake_effective} tables from {cache_dir.name}"
              + (f" (subsampled from {n_manifest})" if lake_sample is not None else " (manifest entries)"))
        print(f"  Labeled src  : {len(labeled_lake)} (repurposed at threshold={REPURPOSE_THRESHOLD})")
        print(f"  Top-K used   : {len(top_k_scores)}")
        if filter_ablation:
            print(f"  Filter combo : [{_cname}]  ({_n_src_after} sources after filters)")
        print("=" * 65)
        print("\n--- Default threshold (0.5) ---")
        print(summary_raw.to_string())
        print("\n--- Calibrated threshold (matched to target positive rate) ---")
        print(summary_cal.to_string())
        print()

        summary_cal.to_csv(_results_dir_combo / "metrics.csv")
        summary_raw.to_csv(_results_dir_combo / "metrics_uncalibrated.csv")

        if n_seeds > 1:
            per_seed = pd.concat(
                [df[["auc"]].rename(columns={"auc": f"auc_seed{i}"}) for i, df in enumerate(seed_metrics_cal)],
                axis=1,
            )
            per_seed.to_csv(_results_dir_combo / "metrics_seeds.csv")
            evaluation.summarise(metrics_cal_std).to_csv(_results_dir_combo / "metrics_std.csv")
            logger.info("Multi-seed results (%d seeds) saved to %s", n_seeds, _results_dir_combo)

        logger.info("Results saved to %s", _results_dir_combo)
        _last_summary_cal = summary_cal

        if filter_ablation:
            _ablation_auc[_cname] = summary_cal["auc"]

    # ------------------------------------------------------------------ #
    # Filter ablation summary table
    # ------------------------------------------------------------------ #
    if filter_ablation and _ablation_auc:
        _abl_df = pd.DataFrame(_ablation_auc).T
        _abl_df.index.name = "combo"
        _abl_df.to_csv(_ablation_dir / "auc_comparison.csv")

        # Also save full metrics per combo in a consolidated CSV
        _all_rows = []
        for _cn in _ablation_auc:
            _m = pd.read_csv(_ablation_dir / _cn / "metrics.csv", index_col=0)
            _m["combo"] = _cn
            _all_rows.append(_m.reset_index())
        pd.concat(_all_rows).to_csv(_ablation_dir / "all_metrics.csv", index=False)

        print("\n" + "=" * 65)
        print(f"FILTER ABLATION SUMMARY — {target_name} (calibrated AUC)")
        print("=" * 65)
        print(_abl_df.to_string(float_format=lambda x: f"{x:.4f}"))
        print()
        print(f"Saved to {_ablation_dir}")

    return _last_summary_cal


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
    parser.add_argument("--neighbor-alpha", type=float, default=0.2,
                        help="Weight given to top-k neighbor context in column embeddings "
                             "(0=disabled, default: 0.2).")
    parser.add_argument("--normalization", choices=["none", "per-source", "target-fitted"],
                        default="per-source",
                        help="Feature normalization: per-source (default, QT fitted on "
                             "each source independently), none, or target-fitted (QT fitted on target).")
    parser.add_argument("--test-inject", action="store_true",
                        help="Validation mode: load a planted noised copy of the target from "
                             "data/inject_test/ and report whether the pipeline finds it. "
                             "No lake files are modified. Run inject_test_table.py first.")
    parser.add_argument("--source-filters", type=str, default="",
                        help="Comma-separated source quality filters to apply after alignment. "
                             "Choices: selfauc (drop structureless sources, self-AUC<0.60), "
                             "semantic (proxy name vs concept list), "
                             "posrate (drop extreme positive rates), "
                             "distrib (drop distributionally misaligned sources). "
                             "Example: --source-filters selfauc,semantic")
    parser.add_argument("--filter-ablation", action="store_true",
                        help="Run all 8 combinations of the 3 source quality filters and save "
                             "per-combo metrics to results/act5/{target}/filter_ablation/.")
    args = parser.parse_args()
    run_experiment(args.target, top_k=args.top_k, lake_dir=args.lake_dir,
                   no_expansion=args.no_expansion,
                   repurpose_threshold=args.repurpose_threshold,
                   llm_baseline=args.llm_baseline,
                   n_seeds=args.seeds,
                   fast_only=args.fast_only,
                   lake_sample=args.lake_sample,
                   neighbor_alpha=args.neighbor_alpha,
                   normalization=args.normalization,
                   test_inject=args.test_inject,
                   source_filters=args.source_filters,
                   filter_ablation=args.filter_ablation)


if __name__ == "__main__":
    main()
