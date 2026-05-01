"""
Step 2 — Schema Alignment

For each source table, maps its columns to the target schema via Hungarian
matching on column-name embeddings. Unmatched columns are dropped.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _build_column_mapping(
    source_cols: list[str],
    target_cols: list[str],
    model: SentenceTransformer,
    min_similarity: float = 0.0,
    source_df: Optional[pd.DataFrame] = None,
    target_df: Optional[pd.DataFrame] = None,
    name_weight: float = 1.0,
) -> dict[str, str]:
    """
    Return {source_col → target_col} for the optimal Hungarian assignment.

    Only columns that appear in the assignment are included; the caller is
    responsible for deciding what to do with unmatched target columns.
    Matches with cosine similarity < min_similarity are dropped, leaving the
    corresponding target columns to be filled with NaN by the caller.

    When source_df and target_df are provided and name_weight < 1.0, the cost
    matrix blends name-based cosine distance with KS distance on values:
        cost = name_weight * name_cost + (1 - name_weight) * ks_cost
    This improves alignment for numeric columns whose names differ across sources.
    """
    src_emb = model.encode(source_cols, show_progress_bar=False, convert_to_numpy=True)
    tgt_emb = model.encode(target_cols, show_progress_bar=False, convert_to_numpy=True)
    name_cost = cdist(src_emb, tgt_emb, metric="cosine")

    if name_weight < 1.0 and source_df is not None and target_df is not None:
        from scipy.stats import ks_2samp
        ks_cost = np.ones_like(name_cost)
        for i, sc in enumerate(source_cols):
            sv = source_df[sc].dropna() if sc in source_df.columns else pd.Series([], dtype=float)
            if len(sv) < 10 or not pd.api.types.is_numeric_dtype(sv):
                continue
            for j, tc in enumerate(target_cols):
                tv = target_df[tc].dropna() if tc in target_df.columns else pd.Series([], dtype=float)
                if len(tv) < 10 or not pd.api.types.is_numeric_dtype(tv):
                    continue
                stat, _ = ks_2samp(sv.values, tv.values)
                ks_cost[i, j] = stat
        cost = name_weight * name_cost + (1.0 - name_weight) * ks_cost
    else:
        cost = name_cost

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping: dict[str, str] = {}
    for r, c in zip(row_ind, col_ind):
        if (1.0 - cost[r, c]) >= min_similarity:
            mapping[source_cols[r]] = target_cols[c]
    return mapping


def _quantile_distance(
    src_vals: pd.Series,
    tgt_vals: pd.Series,
    quantile_levels: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> float:
    """
    Mean L1 distance between source and target quantiles, normalised by the
    target value range.  Returns inf when either series is empty or the target
    range is zero (so the caller treats the match as incompatible).
    """
    src = src_vals.dropna().astype(float)
    tgt = tgt_vals.dropna().astype(float)
    if len(src) == 0 or len(tgt) == 0:
        return float("inf")
    tgt_range = float(tgt.max() - tgt.min())
    if tgt_range < 1e-9:
        return float("inf")
    qs = np.array(quantile_levels)
    return float(np.abs(np.quantile(src, qs) - np.quantile(tgt, qs)).mean() / tgt_range)


def align_table(
    source: pd.DataFrame,
    target: pd.DataFrame,
    model: SentenceTransformer,
    label_col: Optional[str] = None,
    dist_threshold: float = float("inf"),
) -> pd.DataFrame:
    """
    Align `source` to the schema of `target`.

    Columns in `source` are renamed to their best-matching target column.
    Unmatched source columns are dropped. Missing target columns are filled
    with NaN. The label column (if given) is preserved under its original name.

    Parameters
    ----------
    source:
        Labeled source DataFrame.
    target:
        Unlabeled target DataFrame (defines the output schema).
    model:
        Pre-loaded SentenceTransformer instance.
    label_col:
        Name of the label/target column in `source`. Excluded from alignment
        and carried through as-is.
    dist_threshold:
        Maximum normalised quantile L1 distance allowed for a column match to
        be kept.  Matches above this threshold are dropped and the corresponding
        target column is filled with NaN.  Default inf = no filtering
        (backward-compatible).

    Returns
    -------
    DataFrame with columns == target.columns (plus `label_col` if provided).
    """
    # separate label from features before alignment
    labels: Optional[pd.Series] = None
    src_features = source.copy()
    if label_col and label_col in src_features.columns:
        labels = src_features.pop(label_col)

    tgt_cols = target.columns.tolist()
    src_cols = src_features.columns.tolist()

    mapping = _build_column_mapping(src_cols, tgt_cols, model, min_similarity=0.2)

    # --- distribution compatibility filter ---
    if dist_threshold < float("inf"):
        filtered: dict[str, str] = {}
        for src_col, tgt_col in mapping.items():
            src_s = src_features[src_col]
            tgt_s = target[tgt_col]
            # Only check numeric columns; skip object/categorical
            if not (pd.api.types.is_numeric_dtype(src_s) and pd.api.types.is_numeric_dtype(tgt_s)):
                filtered[src_col] = tgt_col
                continue
            dist = _quantile_distance(src_s, tgt_s)
            if dist <= dist_threshold:
                filtered[src_col] = tgt_col
            else:
                logger.debug(
                    "Dropped match '%s'→'%s' (quantile dist=%.3f > threshold=%.3f)",
                    src_col, tgt_col, dist, dist_threshold,
                )
        n_dropped = len(mapping) - len(filtered)
        if n_dropped > 0:
            logger.info(
                "  dist filter: dropped %d / %d column matches (threshold=%.2f)",
                n_dropped, len(mapping), dist_threshold,
            )
        mapping = filtered

    logger.debug("Column mapping: %s", mapping)

    aligned = src_features[list(mapping.keys())].rename(columns=mapping)

    # add any target columns that have no match
    for col in tgt_cols:
        if col not in aligned.columns:
            aligned[col] = np.nan

    # reorder to match target schema
    aligned = aligned[tgt_cols]

    if labels is not None:
        aligned[label_col] = labels.values

    return aligned.reset_index(drop=True)


def align_all(
    lake: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    discovery_scores: dict[str, float],
    model_name: str = _DEFAULT_MODEL,
    model: Optional[SentenceTransformer] = None,
    label_col: Optional[str] = None,
    dist_threshold: float = float("inf"),
    min_cols: int = 2,
    min_coverage: float = 0.40,
) -> dict[str, pd.DataFrame]:
    """
    Align every table in `lake` (filtered to those in `discovery_scores`) to
    the target schema.

    Parameters
    ----------
    lake:
        Mapping of table_id → DataFrame.
    target:
        Unlabeled target DataFrame.
    discovery_scores:
        Output of `table_discovery.discover_tables`. Only tables present here
        are aligned.
    model_name:
        Sentence-transformers model identifier (ignored if `model` is provided).
    model:
        Pre-loaded SentenceTransformer instance.
    label_col:
        Label column name to preserve through alignment.
    dist_threshold:
        Passed to `align_table`. Column matches with normalised quantile
        distance above this value are dropped (filled with NaN instead).
    min_coverage:
        Minimum fraction of feature cells (rows × columns) that must be
        non-NaN for a source to be kept.  Sources whose aligned features are
        mostly empty (e.g., wrong-domain tables that match column names by
        accident) are silently dropped.  Default 0.40 (40 % coverage).

    Returns
    -------
    dict[table_id → aligned DataFrame], same key order as discovery_scores.
    """
    if model is None:
        logger.info("Loading sentence-transformers model: %s", model_name)
        model = SentenceTransformer(model_name)

    aligned: dict[str, pd.DataFrame] = {}
    n_dropped = 0
    for table_id in discovery_scores:
        if table_id not in lake:
            logger.warning("Table '%s' in scores but not in lake — skipping.", table_id)
            continue
        logger.info("Aligning table '%s'", table_id)
        aligned_df = align_table(
            lake[table_id], target, model, label_col=label_col, dist_threshold=dist_threshold,
        )
        feature_cols = [c for c in aligned_df.columns if c != label_col]

        # Gate 1: minimum number of matched columns
        n_matched = int(sum(aligned_df[c].notna().any() for c in feature_cols))
        if n_matched < min_cols:
            logger.info(
                "  Dropping '%s': only %d/%d columns matched (min_cols=%d)",
                table_id, n_matched, len(feature_cols), min_cols,
            )
            n_dropped += 1
            continue

        # Gate 2: minimum non-NaN cell coverage across the feature matrix.
        # Catches wrong-domain sources that happen to match column names but
        # contribute mostly empty features (e.g., code-churn vs customer-churn).
        if feature_cols and min_coverage > 0.0:
            coverage = float(aligned_df[feature_cols].notna().values.mean())
            if coverage < min_coverage:
                logger.info(
                    "  Dropping '%s': feature coverage %.2f < %.2f (min_coverage)",
                    table_id, coverage, min_coverage,
                )
                n_dropped += 1
                continue

        aligned[table_id] = aligned_df

    if n_dropped:
        logger.info("align_all: dropped %d/%d tables (min_cols=%d, min_coverage=%.2f)",
                    n_dropped, len(discovery_scores), min_cols, min_coverage)
    return aligned
