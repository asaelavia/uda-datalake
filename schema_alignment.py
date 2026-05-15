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


def _col_text(col: str, series: Optional[pd.Series]) -> str:
    """
    Build an enriched text representation of a column for embedding.

    Binary columns get bare name (0/1 or Yes/No values are uninformative).
    Multi-category strings get top-3 values appended.
    Continuous numeric gets observed range appended.
    """
    if series is None:
        return col
    s = series.dropna()
    if len(s) < 5:
        return col
    n_unique = s.nunique()
    if pd.api.types.is_numeric_dtype(s):
        if n_unique <= 2:
            return col
        mn, mx = float(s.min()), float(s.max())
        return f"{col} range {mn:.0f} to {mx:.0f}"
    else:
        if n_unique <= 2:
            return col
        top = s.value_counts().head(3).index.tolist()
        vals = ", ".join(str(v) for v in top)
        return f"{col}: {vals}"


def _neighbor_context_blend(
    embs: np.ndarray,
    k: int = 3,
    alpha: float = 0.2,
) -> np.ndarray:
    """
    Blend each embedding with the mean of its top-k most similar neighbors.

    Encodes table-level domain context without relying on column order: "up" in
    [id, up, down, left, right, type, rarity] gets pulled toward game semantics
    because its nearest neighbors are other directional/game terms.
    No extra encode calls — uses the embeddings already computed for matching.
    """
    n = len(embs)
    if n <= 1 or k <= 0 or alpha <= 0.0:
        return embs
    # L2-normalise for cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = embs / np.where(norms > 1e-9, norms, 1.0)
    sim = normed @ normed.T  # (N, N) cosine similarities
    blended = np.empty_like(embs)
    effective_k = min(k, n - 1)
    for i in range(n):
        row = sim[i].copy()
        row[i] = -2.0  # exclude self
        top_idx = np.argpartition(row, -effective_k)[-effective_k:]
        neighbor_mean = embs[top_idx].mean(axis=0)
        b = (1.0 - alpha) * embs[i] + alpha * neighbor_mean
        norm = np.linalg.norm(b)
        blended[i] = b / norm if norm > 1e-9 else b
    return blended


def _build_column_mapping(
    source_cols: list[str],
    target_cols: list[str],
    model: SentenceTransformer,
    min_similarity: float = 0.0,
    source_df: Optional[pd.DataFrame] = None,
    target_df: Optional[pd.DataFrame] = None,
    name_weight: float = 1.0,
    neighbor_k: int = 3,
    neighbor_alpha: float = 0.2,
    type_gate: bool = False,
    use_enrichment: bool = False,
) -> dict[str, tuple[str, float]]:
    """
    Return {source_col → (target_col, similarity)} for the optimal Hungarian assignment.

    Only columns that appear in the assignment are included; the caller is
    responsible for deciding what to do with unmatched target columns.
    Matches with cosine similarity < min_similarity are dropped, leaving the
    corresponding target columns to be filled with NaN by the caller.

    use_enrichment: when True, append value range/top-values to column names
    before embedding (_col_text) and blend with table-level neighbor context
    (_neighbor_context_blend). Disabled by default — enrichment helps for
    clean structured datasets (e.g. GACars) but hurts heterogeneous lake tables
    where ordinal-encoded sources have different value ranges than string targets.

    When source_df and target_df are provided and name_weight < 1.0, the cost
    matrix additionally blends name-based cosine distance with KS distance on
    values:
        cost = name_weight * name_cost + (1 - name_weight) * ks_cost
    """
    if use_enrichment:
        src_texts = [
            _col_text(c, source_df[c] if source_df is not None and c in source_df.columns else None)
            for c in source_cols
        ]
        tgt_texts = [
            _col_text(c, target_df[c] if target_df is not None and c in target_df.columns else None)
            for c in target_cols
        ]
    else:
        src_texts = source_cols
        tgt_texts = target_cols

    src_emb = model.encode(src_texts, show_progress_bar=False, convert_to_numpy=True)
    tgt_emb = model.encode(tgt_texts, show_progress_bar=False, convert_to_numpy=True)

    if use_enrichment:
        src_emb = _neighbor_context_blend(src_emb, k=neighbor_k, alpha=neighbor_alpha)
        tgt_emb = _neighbor_context_blend(tgt_emb, k=neighbor_k, alpha=neighbor_alpha)

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

    # Type-gated assignment: run two independent Hungarians (numeric vs string)
    # so that cross-type blocking cannot displace valid same-type matches.
    # Without this, blocking Engine(str)->power_kw(num) causes Engine to compete
    # with Brand for brand(str), potentially displacing the correct Brand->brand match.
    if type_gate and source_df is not None and target_df is not None:
        def _is_num(col, df):
            return col in df.columns and pd.api.types.is_numeric_dtype(df[col].dropna())

        src_num = [i for i, c in enumerate(source_cols) if _is_num(c, source_df)]
        src_str = [i for i, c in enumerate(source_cols) if not _is_num(c, source_df)]
        tgt_num = [j for j, c in enumerate(target_cols) if _is_num(c, target_df)]
        tgt_str = [j for j, c in enumerate(target_cols) if not _is_num(c, target_df)]

        mapping: dict[str, tuple[str, float]] = {}
        for si, ti in [(src_num, tgt_num), (src_str, tgt_str)]:
            if not si or not ti:
                continue
            sub = cost[np.ix_(si, ti)]
            for r, c in zip(*linear_sum_assignment(sub)):
                sim = float(1.0 - sub[r, c])
                if sim >= min_similarity:
                    mapping[source_cols[si[r]]] = (target_cols[ti[c]], sim)
        return mapping

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping: dict[str, tuple[str, float]] = {}
    for r, c in zip(row_ind, col_ind):
        sim = float(1.0 - cost[r, c])
        if sim >= min_similarity:
            mapping[source_cols[r]] = (target_cols[c], sim)
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
    min_similarity: float = 0.2,
    neighbor_k: int = 3,
    neighbor_alpha: float = 0.2,
    type_gate: bool = True,
    use_enrichment: bool = False,
    fill_unmatched: str = "nan",
) -> tuple[pd.DataFrame, dict[str, tuple[str, float]]]:
    """
    Align `source` to the schema of `target`.

    Columns in `source` are renamed to their best-matching target column.
    Unmatched source columns are dropped. Missing target columns are filled
    according to `fill_unmatched`. The label column (if given) is preserved
    under its original name.

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
        target column is filled according to fill_unmatched.  Default inf = no filtering.
    min_similarity:
        Minimum cosine similarity for a column name match to be kept.
        Pairs below this are dropped and the target column filled according to
        fill_unmatched.  Default 0.2.
    fill_unmatched:
        How to fill target columns that have no valid source match.
        "nan"         — fill with NaN (current default; XGBoost handles this
                        natively but DANN sees constant-zero columns).
        "target_mean" — fill with the target column's mean; removes the
                        NaN→0 artefact that lets domain discriminators
                        trivially separate source from target on unaligned dims.

    Returns
    -------
    (aligned_df, col_mapping) where aligned_df has columns == target.columns
    (plus label_col if provided) and col_mapping is
    {source_col: (target_col, similarity)} for every kept match.
    """
    # separate label from features before alignment
    labels: Optional[pd.Series] = None
    src_features = source.copy()
    if label_col and label_col in src_features.columns:
        labels = src_features.pop(label_col)

    tgt_cols = target.columns.tolist()
    src_cols = src_features.columns.tolist()

    mapping = _build_column_mapping(
        src_cols, tgt_cols, model,
        min_similarity=min_similarity,
        source_df=src_features,
        target_df=target,
        neighbor_k=neighbor_k,
        neighbor_alpha=neighbor_alpha,
        type_gate=type_gate,
        use_enrichment=use_enrichment,
    )

    # --- distribution compatibility filter ---
    if dist_threshold < float("inf"):
        filtered: dict[str, tuple[str, float]] = {}
        for src_col, (tgt_col, sim) in mapping.items():
            src_s = src_features[src_col]
            tgt_s = target[tgt_col]
            # Only check numeric columns; skip object/categorical
            if not (pd.api.types.is_numeric_dtype(src_s) and pd.api.types.is_numeric_dtype(tgt_s)):
                filtered[src_col] = (tgt_col, sim)
                continue
            dist = _quantile_distance(src_s, tgt_s)
            if dist <= dist_threshold:
                filtered[src_col] = (tgt_col, sim)
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

    col_rename = {src: tgt for src, (tgt, _) in mapping.items()}
    logger.debug("Column mapping: %s", col_rename)

    aligned = src_features[list(col_rename.keys())].rename(columns=col_rename)

    # add any target columns that have no match
    if fill_unmatched == "target_mean":
        tgt_means = target.mean(numeric_only=True)
    for col in tgt_cols:
        if col not in aligned.columns:
            if fill_unmatched == "target_mean" and col in tgt_means.index:
                aligned[col] = float(tgt_means[col])
            else:
                aligned[col] = np.nan

    # reorder to match target schema
    aligned = aligned[tgt_cols]

    if labels is not None:
        aligned[label_col] = labels.values

    return aligned.reset_index(drop=True), mapping


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
    min_similarity: float = 0.2,
    neighbor_k: int = 3,
    neighbor_alpha: float = 0.2,
    type_gate: bool = False,
    use_enrichment: bool = False,
    fill_unmatched: str = "nan",
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, tuple[str, float]]]]:
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
    min_similarity:
        Minimum name-embedding cosine similarity for a column match to be kept.
        Passed to align_table. Default 0.2.

    Returns
    -------
    (aligned_dfs, col_mappings) where aligned_dfs is dict[table_id → DataFrame]
    and col_mappings is dict[table_id → {src_col: (tgt_col, similarity)}].
    """
    if model is None:
        logger.info("Loading sentence-transformers model: %s", model_name)
        model = SentenceTransformer(model_name)

    aligned: dict[str, pd.DataFrame] = {}
    col_mappings: dict[str, dict[str, tuple[str, float]]] = {}
    n_dropped = 0
    for table_id in discovery_scores:
        if table_id not in lake:
            logger.warning("Table '%s' in scores but not in lake — skipping.", table_id)
            continue
        logger.info("Aligning table '%s'", table_id)
        aligned_df, mapping = align_table(
            lake[table_id], target, model,
            label_col=label_col,
            dist_threshold=dist_threshold,
            min_similarity=min_similarity,
            neighbor_k=neighbor_k,
            neighbor_alpha=neighbor_alpha,
            type_gate=type_gate,
            use_enrichment=use_enrichment,
            fill_unmatched=fill_unmatched,
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
        col_mappings[table_id] = mapping

    if n_dropped:
        logger.info("align_all: dropped %d/%d tables (min_cols=%d, min_coverage=%.2f)",
                    n_dropped, len(discovery_scores), min_cols, min_coverage)
    return aligned, col_mappings
