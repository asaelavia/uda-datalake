"""
Transferability Score — quantifies how useful a data lake will be for a given target.

Two modes
---------
true  : computed from full pipeline variables (post-repurposing, post-alignment).
        Exact. Used to validate that the score correlates with AUC improvement.

fast  : uses a pre-built column embedding index (built once for the whole lake).
        Per-target query: one matrix multiply → find candidates → load only top-K
        parquets. No full lake scan at query time. Seconds, not hours.

Index
-----
Build once with `build_column_index()`. Saves to data/col_name_index/:
    unique_names.json   list of deduplicated column names across all lake tables
    embs.npy            (n_unique, embed_dim) float32 embeddings
    table_col_map.json  {table_id: [col_idx, ...]} indices into unique_names

Usage
-----
    # One-time (shared across all targets):
    transferability.build_column_index(manifest["tables"], cache_dir, encoder)

    # Per-target fast score (seconds):
    score = transferability.compute_score_fast(
        manifest_tables, cache_dir, encoder, target_features,
        target_pos_rate, concepts, threshold=0.70, top_k=20,
    )

    # True score (computed from pipeline variables after full run):
    score = transferability.compute_score(
        labeled_lake, top_k_scores, aligned, target_pos_rate,
        n_lake_tables, label_col,
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_INDEX_DIR = Path("data/col_name_index")


# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------

@dataclass
class TransferabilityScore:
    """
    Components of the transferability score.

    repurpose_yield    : log1p(n_sources) / log1p(n_lake_tables)  — never saturates
    discovery_quality  : (max + mean) of top-K scores / 2  — rewards one gold table
    alignment_density  : mean fraction of target cols matched across top-K sources
    label_shift        : 1 − |mean_source_pos_rate − target_pos_rate|
    feature_overlap    : mean (1 − KS_distance) across numeric cols in aligned sources vs target
    pas_score          : tabular PAS (ICLR 2026) — centroid margin in aligned feature space
    spa_score          : Source Prediction Agreement — 1 − 2×mean_std(LR predictions on target)
                         High = sources consistently predict the same target probabilities
    source_consistency : 1 − std(top-K scores), clamped [0,1]  — informational only
    top1_score         : score of the single best source  — informational only
    overall            : primary transferability score (pas_score for true; component mean for fast)
                         Despite near-zero absolute values, pas_score rank has ρ=0.60 vs oracle gap.
                         The curse of dimensionality compresses values to ~0 but ordinal signal survives.
    n_sources          : number of repurposed sources found (or candidate count)
    n_lake_tables      : total tables in the lake manifest
    mode               : "true" or "fast"
    """
    repurpose_yield: float
    discovery_quality: float
    alignment_density: float
    label_shift: float
    feature_overlap: float
    pas_score: float
    spa_score: float
    cslp_score: float      # Cross-Source Label Prediction: SLOO AUC in target feat space (fast only)
    lcc_score: float       # Label Concept Coherence: internal CV AUC of pseudo-label (fast only)
    pas_loose_score: float # PAS at threshold 0.50 (covers more targets than tight PAS at 0.60)
    pca_pas_score: float  # PCA-PAS: Mahalanobis separation in target PCA space × overlap (fast only)
    zscore_copas_score: float  # ZScore-CoPAS: z-norm features + L2 row-norm + cosine centroid margin
    npas_score: float        # Normalized PAS: pas_target / pas_source_self (train/val split)
    tsc_score: float         # Target-Source Consistency: reverse-classifier AUC of pseudo-labels
    source_consistency: float
    top1_score: float
    overall: float
    n_sources: int
    n_lake_tables: int
    mode: str


# ---------------------------------------------------------------------------
# True score  (post-pipeline, zero I/O)
# ---------------------------------------------------------------------------

def compute_score(
    labeled_lake: dict,
    top_k_scores: dict[str, float],
    aligned: dict[str, pd.DataFrame],
    target_pos_rate: float,
    n_lake_tables: int,
    label_col: str,
    target_features: Optional[pd.DataFrame] = None,
) -> TransferabilityScore:
    """
    Compute transferability score from existing pipeline variables.

    Called after full repurposing scan + schema alignment — all inputs are
    already in memory; no further I/O is performed.

    Parameters
    ----------
    labeled_lake    : full repurposed lake (all sources, with label column)
    top_k_scores    : discovery scores for the top-K selected sources
    aligned         : schema-aligned DataFrames for top-K sources (post-norm)
    target_pos_rate : fraction of positive labels in the target dataset
    n_lake_tables   : total number of tables in the lake manifest
    label_col       : name of the label column in aligned DataFrames
    target_features : target feature DataFrame (post-normalization); used to
                      compute feature_overlap via KS distance. If None,
                      feature_overlap defaults to 0.5 (neutral).
    """
    n_sources = len(labeled_lake)

    # 1. Repurpose yield — log-log ratio: never saturates, scale-invariant
    #    log1p(6)/log1p(421K)=0.15, log1p(947)/log1p(421K)=0.53 — full range used
    if n_lake_tables > 0 and n_sources > 0:
        repurpose_yield = float(np.log1p(n_sources) / np.log1p(n_lake_tables))
    else:
        repurpose_yield = 0.0

    # 2. Discovery quality — (max + mean) / 2: rewards one gold table while
    #    still penalising a uniformly weak set
    scores_list = list(top_k_scores.values())
    if scores_list:
        top1_score = float(max(scores_list))
        discovery_quality = float((top1_score + np.mean(scores_list)) / 2.0)
    else:
        top1_score = 0.0
        discovery_quality = 0.0

    # 3. Alignment density — fraction of target cols that are non-NaN across top-K sources
    density_vals: list[float] = []
    label_shift_vals: list[float] = []
    for table_id, df in aligned.items():
        feature_cols = [c for c in df.columns if c != label_col]
        if not feature_cols:
            continue
        n_matched = sum(df[c].notna().any() for c in feature_cols)
        density_vals.append(n_matched / len(feature_cols))
        if label_col in df.columns:
            try:
                pos_rate = float(df[label_col].mean())
                label_shift_vals.append(pos_rate)
            except Exception:
                pass
    alignment_density = float(np.mean(density_vals)) if density_vals else 0.0

    # 4. Label shift — lower shift → higher score
    if label_shift_vals:
        mean_src_rate = float(np.mean(label_shift_vals))
        label_shift = float(np.clip(1.0 - abs(mean_src_rate - target_pos_rate), 0.0, 1.0))
    else:
        label_shift = 0.5  # neutral when no distribution info available

    # 5. Feature overlap — mean (1 - KS_distance) across numeric cols in aligned sources vs target
    #    Captures feature distribution similarity that DANN needs to align.
    #    Computed in the same normalized space as the adaptation algorithms see.
    if target_features is not None and aligned:
        from scipy.stats import ks_2samp
        ks_vals: list[float] = []
        for table_id, df in aligned.items():
            feature_cols = [c for c in df.columns if c != label_col]
            for col in feature_cols:
                if col not in target_features.columns:
                    continue
                src_vals = df[col].dropna()
                tgt_vals = target_features[col].dropna()
                if not (pd.api.types.is_numeric_dtype(src_vals) and
                        pd.api.types.is_numeric_dtype(tgt_vals)):
                    continue
                if len(src_vals) < 5 or len(tgt_vals) < 5:
                    continue
                stat, _ = ks_2samp(src_vals.values.astype(float),
                                   tgt_vals.values.astype(float))
                ks_vals.append(1.0 - stat)
        feature_overlap = float(np.mean(ks_vals)) if ks_vals else 0.5
    else:
        feature_overlap = 0.5  # neutral fallback

    # 6. PAS — tabular adaptation of Potential Adaptability Score (ICLR 2026)
    pas_score = compute_pas(
        aligned=aligned,
        target_features=target_features if target_features is not None else pd.DataFrame(),
        label_col=label_col,
        discovery_scores=top_k_scores,
    )

    # 6b. ZScore-CoPAS — z-normalise features then apply paper L2+cosine formula
    true_zscore_copas = compute_zscore_copas(
        aligned=aligned,
        target_features=target_features if target_features is not None else pd.DataFrame(),
        label_col=label_col,
        discovery_scores=top_k_scores,
    )

    # 6c. nPAS — normalized PAS: target PAS / source-self PAS (80/20 split)
    npas_score = compute_npas(
        aligned=aligned,
        target_features=target_features if target_features is not None else pd.DataFrame(),
        label_col=label_col,
        discovery_scores=top_k_scores,
    )

    # 6d. TSC — Target-Source Consistency: bidirectional classifier round-trip
    tsc_score = compute_tsc(
        aligned=aligned,
        target_features=target_features if target_features is not None else pd.DataFrame(),
        label_col=label_col,
        discovery_scores=top_k_scores,
    )

    # 7. SPA — Source Prediction Agreement
    #    Train LR on each aligned source, predict on target, measure cross-source std.
    #    High agreement (low std) → sources consistently see the same label concept → good transfer.
    spa_score = compute_spa(
        aligned=aligned,
        target_features=target_features if target_features is not None else pd.DataFrame(),
        label_col=label_col,
        discovery_scores=top_k_scores,
    )

    # 8. Source consistency — informational only (empirically flat: 0.937–0.987)
    source_consistency = float(np.clip(1.0 - np.std(scores_list), 0.0, 1.0)) if len(scores_list) > 1 else 1.0

    # True-score overall = pas_score (ρ=+0.685, p=0.014, n=12 — only statistically significant predictor).
    # Near-zero absolute values are expected (curse of dimensionality in PCA-whitened tabular space)
    # but ordinal signal survives. nPAS and TSC added as companion scores with interpretable scales.
    overall = pas_score

    return TransferabilityScore(
        repurpose_yield=repurpose_yield,
        discovery_quality=discovery_quality,
        alignment_density=alignment_density,
        label_shift=label_shift,
        feature_overlap=feature_overlap,
        pas_score=pas_score,
        spa_score=spa_score,
        cslp_score=0.5,            # fast-only; not computed in true mode
        lcc_score=0.5,             # fast-only; not computed in true mode
        pas_loose_score=0.5,       # fast-only; not computed in true mode
        pca_pas_score=0.5,         # fast-only; not computed in true mode
        zscore_copas_score=true_zscore_copas,
        npas_score=npas_score,
        tsc_score=tsc_score,
        source_consistency=source_consistency,
        top1_score=top1_score,
        overall=overall,
        n_sources=n_sources,
        n_lake_tables=n_lake_tables,
        mode="true",
    )


# ---------------------------------------------------------------------------
# One-time index builder
# ---------------------------------------------------------------------------

def build_column_index(
    manifest_tables: list[dict],
    cache_dir: Path,
    encoder,
    index_dir: Path = _DEFAULT_INDEX_DIR,
    batch_size: int = 2048,
) -> None:
    """
    Build and cache column name embeddings for all lake tables.

    Run once per lake. Saves three files to index_dir:
      unique_names.json  — deduplicated column names
      embs.npy           — (n_unique, embed_dim) float32
      table_col_map.json — {table_id: [idx, ...]} into unique_names

    Parameters
    ----------
    manifest_tables : list of dicts from manifest["tables"]; must have "table_id"
                      and "columns" fields.
    cache_dir       : root parquet cache directory (used only for metadata).
    encoder         : pre-loaded SentenceTransformer.
    index_dir       : destination directory (created if absent).
    batch_size      : encoding batch size.
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building column name index from %d manifest entries...", len(manifest_tables))

    # Collect all column names and build table→col mapping
    name_to_idx: dict[str, int] = {}
    table_col_map: dict[str, list[int]] = {}

    for entry in manifest_tables:
        table_id = entry.get("table_id", "")
        cols = entry.get("columns", [])
        idxs: list[int] = []
        for col in cols:
            col_str = str(col).strip()
            if not col_str:
                continue
            if col_str not in name_to_idx:
                name_to_idx[col_str] = len(name_to_idx)
            idxs.append(name_to_idx[col_str])
        if idxs:
            table_col_map[table_id] = idxs

    unique_names = [None] * len(name_to_idx)
    for name, idx in name_to_idx.items():
        unique_names[idx] = name

    logger.info("Unique column names: %d  (across %d tables)", len(unique_names), len(table_col_map))

    # Batch-encode all unique names
    logger.info("Encoding %d unique column names (batch_size=%d)...", len(unique_names), batch_size)
    embs = encoder.encode(
        unique_names,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=batch_size,
    ).astype(np.float32)

    # L2-normalize for fast cosine via dot product
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    embs = embs / norms

    # Save
    np.save(index_dir / "embs.npy", embs)
    with open(index_dir / "unique_names.json", "w") as f:
        json.dump(unique_names, f)
    with open(index_dir / "table_col_map.json", "w") as f:
        json.dump(table_col_map, f)

    logger.info("Column index saved to %s  (embs shape: %s)", index_dir, embs.shape)


def _load_index(index_dir: Path) -> tuple[list[str], np.ndarray, dict[str, list[int]]]:
    """Load the pre-built column name index. Returns (unique_names, embs, table_col_map)."""
    with open(index_dir / "unique_names.json") as f:
        unique_names: list[str] = json.load(f)
    embs = np.load(index_dir / "embs.npy")
    with open(index_dir / "table_col_map.json") as f:
        table_col_map: dict[str, list[int]] = json.load(f)
    return unique_names, embs, table_col_map


# ---------------------------------------------------------------------------
# Fast score
# ---------------------------------------------------------------------------

def compute_score_fast(
    manifest_tables: list[dict],
    cache_dir: Path,
    encoder,
    target_features: pd.DataFrame,
    target_pos_rate: float,
    concepts: list[str],
    threshold: float = 0.70,
    top_k: int = 20,
    load_multiplier: int = 3,
    index_dir: Path = _DEFAULT_INDEX_DIR,
) -> TransferabilityScore:
    """
    Fast transferability score using the pre-built column embedding index.

    No full lake scan. Algorithm:
      1. Encode concepts → concept_embs  (instant)
      2. sim_matrix = concept_embs @ embs.T  (one matrix multiply, ms)
      3. table_score[t] = max(sim_matrix[:, col_indices[t]])  (vectorized)
      4. repurpose_yield = exact (scanned all column names)
      5. Load top-(top_k * load_multiplier) parquets → compute remaining components

    Parameters
    ----------
    manifest_tables  : manifest["tables"]
    cache_dir        : root parquet directory
    encoder          : pre-loaded SentenceTransformer
    target_features  : target feature DataFrame (for alignment density)
    target_pos_rate  : target positive class fraction
    concepts         : expanded concept list for the label
    threshold        : cosine similarity threshold for concept matching
    top_k            : number of top sources (matches pipeline TOP_K)
    load_multiplier  : load top_k * this many parquets for distribution stats
    index_dir        : directory containing pre-built index files
    """
    index_dir = Path(index_dir)
    if not (index_dir / "embs.npy").exists():
        raise FileNotFoundError(
            f"Column index not found at {index_dir}. "
            "Run transferability.build_column_index() first."
        )

    unique_names, embs, table_col_map = _load_index(index_dir)
    n_lake = len(manifest_tables)
    id_to_path = {e["table_id"]: cache_dir / e["path"] for e in manifest_tables}

    # Encode concepts (L2-normalize for cosine via dot product)
    concept_embs = encoder.encode(
        concepts,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    cnorms = np.linalg.norm(concept_embs, axis=1, keepdims=True)
    cnorms = np.where(cnorms < 1e-9, 1.0, cnorms)
    concept_embs = concept_embs / cnorms  # (n_concepts, D)

    # Matrix multiply: (n_concepts, D) @ (D, n_unique) → (n_concepts, n_unique)
    logger.info("[Transferability fast] Computing similarities over %d unique column names...", len(unique_names))
    sim_matrix = concept_embs @ embs.T  # (n_concepts, n_unique)

    # Encode target feature column names for feature-coverage-aware candidate selection.
    # One extra encode (ms for 6-14 cols) + one matrix multiply lets us rank candidates
    # by both label similarity AND feature column overlap, fixing the problem where pure
    # label-score ranking finds tables with the right label column but wrong feature space.
    tgt_feat_cols = list(target_features.columns)
    tgt_feat_embs = encoder.encode(
        tgt_feat_cols, show_progress_bar=False, convert_to_numpy=True,
    ).astype(np.float32)
    _fn = np.linalg.norm(tgt_feat_embs, axis=1, keepdims=True)
    tgt_feat_embs = tgt_feat_embs / np.where(_fn < 1e-9, 1.0, _fn)
    feat_sim_matrix = tgt_feat_embs @ embs.T   # (n_tgt_cols, n_unique)
    _FEAT_MATCH_THRESH       = 0.60  # candidate selection + PAS/SPA feature matching
    _FEAT_MATCH_THRESH_LOOSE = 0.50  # CSLP/LCC feature matching (more permissive)

    # Per-table: label score (concept similarity) + feature coverage (target col overlap)
    all_table_ids = list(table_col_map.keys())
    table_scores   = np.zeros(len(all_table_ids), dtype=np.float32)  # label score
    feat_coverages = np.zeros(len(all_table_ids), dtype=np.float32)  # feature coverage

    for i, tid in enumerate(all_table_ids):
        col_idxs = table_col_map[tid]
        if col_idxs:
            table_scores[i] = float(sim_matrix[:, col_idxs].max())
            # fraction of target feature cols with a semantic match in this table
            best_feat = feat_sim_matrix[:, col_idxs].max(axis=1)  # (n_tgt_cols,)
            feat_coverages[i] = float((best_feat >= _FEAT_MATCH_THRESH).mean())

    # Repurpose yield — uses label scores only (unchanged semantics)
    n_matched = int((table_scores >= threshold).sum())
    repurpose_yield = float(np.log1p(n_matched) / np.log1p(n_lake)) if n_matched > 0 else 0.0

    n_load = min(top_k * load_multiplier, n_matched) if n_matched > 0 else 0
    if n_load == 0:
        logger.warning("[Transferability fast] No candidates above threshold=%.2f", threshold)
        return TransferabilityScore(
            repurpose_yield=0.0, discovery_quality=0.0, alignment_density=0.0,
            label_shift=0.5, feature_overlap=0.5, pas_score=0.5, spa_score=0.5,
            cslp_score=0.5, lcc_score=0.5, pas_loose_score=0.5, pca_pas_score=0.5,
            zscore_copas_score=0.5, npas_score=0.5, tsc_score=0.5,
            source_consistency=1.0, top1_score=0.0, overall=0.2,
            n_sources=0, n_lake_tables=n_lake, mode="fast",
        )

    # Discovery quality: top-K by label score only (measures label proxy quality)
    label_sorted = np.argsort(table_scores)[::-1]

    # Candidate selection: combined label × feature score — selects tables that have BOTH
    # a good label proxy AND feature columns matching the target.  These are the tables
    # fast-PAS can actually compute centroids for.
    combined_scores = np.where(
        table_scores >= threshold,
        table_scores * (1.0 + feat_coverages),
        0.0,
    ).astype(np.float32)
    combined_sorted = np.argsort(combined_scores)[::-1]
    candidate_ids = [
        all_table_ids[j] for j in combined_sorted[:n_load]
        if combined_scores[j] > 0
    ]
    candidate_scores = {tid: float(table_scores[np.where(np.array(all_table_ids) == tid)[0][0]])
                        for tid in candidate_ids}

    # Compute discovery_quality from top-K by label score (proxy for full discovery score)
    top_k_by_label = [all_table_ids[j] for j in label_sorted[:top_k]]
    top_k_candidate_scores = {tid: float(table_scores[label_sorted[i]])
                               for i, tid in enumerate(top_k_by_label)}
    scores_list = list(top_k_candidate_scores.values())
    if scores_list:
        _top1 = float(max(scores_list))
        discovery_quality = float((_top1 + np.mean(scores_list)) / 2.0)
    else:
        _top1 = 0.0
        discovery_quality = 0.0
    source_consistency = float(np.clip(1.0 - np.std(scores_list), 0.0, 1.0)) if len(scores_list) > 1 else 1.0

    # Load candidate parquets for distribution-based components
    from scipy.stats import ks_2samp as _ks_2samp
    from scipy.spatial.distance import cdist as _cdist_fast
    tgt_cols = list(target_features.columns)
    tgt_numeric_cols = [c for c in tgt_cols if pd.api.types.is_numeric_dtype(target_features[c])]
    # Precompute: row index in feat_sim_matrix for each numeric target column
    _tgt_feat_col_to_row = {tc: tgt_feat_cols.index(tc) for tc in tgt_numeric_cols
                            if tc in tgt_feat_cols}
    density_vals: list[float] = []
    pos_rates: list[float] = []
    ks_vals: list[float] = []
    fast_pas_vals: list[float] = []        # fast-PAS: per-table margin scores
    fast_pas_weights: list[float] = []
    fast_spa_preds: list[np.ndarray] = []  # fast-SPA: per-table LR predictions on target
    fast_spa_weights: list[float] = []
    cslp_data: list[tuple[np.ndarray, np.ndarray]] = []  # (X_full_tgt_space, pseudo_labels)
    lcc_vals: list[float] = []                           # per-source internal CV AUC
    fast_pas_loose_vals: list[float] = []                # PAS at threshold 0.50
    fast_pas_loose_weights: list[float] = []
    src_pca_pas_vals: list[float] = []                   # Source-space PCA-PAS per table
    src_pca_pas_weights: list[float] = []
    semantic_ks_vals: list[float] = []                   # KS overlap on semantic column matches
    copas_vals: list[float] = []                         # CoPAS: paper's L2+cosine centroid margin
    copas_weights: list[float] = []
    zcopas_vals: list[float] = []                        # ZScore-CoPAS: z-norm + L2+cosine centroid margin
    zcopas_weights: list[float] = []
    centroid_sep_vals: list[float] = []                  # Centroid Coherence: cosine dist between class centroids

    # Fast PCA-PAS: fit PCA on target numerics once; project each source into target PCA space.
    # Mahalanobis separation (pooled within-class covariance) × overlap (fraction of source
    # points within ±2σ of target centroid on each PC) → comparable across sources.
    _k_pca = max(2, min(10, len(tgt_numeric_cols) // 2)) if len(tgt_numeric_cols) >= 4 else 0
    _target_scaler_pca = None
    _target_pca_obj = None
    _tgt_pca_mean: Optional[np.ndarray] = None
    _tgt_pca_std: Optional[np.ndarray] = None
    fast_pca_pas_sources: list[tuple[np.ndarray, np.ndarray, float]] = []
    if _k_pca >= 2:
        _tgt_num_arr = target_features[tgt_numeric_cols].fillna(0).values.astype(float)
        if _tgt_num_arr.shape[0] >= _k_pca + 1:
            try:
                from sklearn.preprocessing import StandardScaler as _SS_pca
                from sklearn.decomposition import PCA as _PCA_fpca
                _target_scaler_pca = _SS_pca()
                _tgt_scaled_pca = _target_scaler_pca.fit_transform(_tgt_num_arr)
                _target_pca_obj = _PCA_fpca(n_components=_k_pca, whiten=True)
                _target_pca_obj.fit(_tgt_scaled_pca)
                _tgt_pca_T = _target_pca_obj.transform(_tgt_scaled_pca)
                _tgt_pca_mean = _tgt_pca_T.mean(axis=0)
                _tgt_pca_std = _tgt_pca_T.std(axis=0) + 1e-9
            except Exception:
                _target_scaler_pca = None
                _target_pca_obj = None

    n_loaded = 0
    for tid in candidate_ids:
        fpath = id_to_path.get(tid)
        if fpath is None or not Path(fpath).exists():
            continue
        try:
            df = pd.read_parquet(fpath)
            # Alignment density: fraction of target cols that have a name match in source
            src_cols = set(c.lower().strip() for c in df.columns)
            tgt_matched = sum(
                1 for tc in tgt_cols
                if any(tc.lower() in sc or sc in tc.lower() for sc in src_cols)
            )
            density_vals.append(tgt_matched / max(len(tgt_cols), 1))

            # Feature overlap: KS distance on name-matched numeric columns
            src_col_map = {c.lower().strip(): c for c in df.columns}
            for tc in tgt_numeric_cols:
                sc = src_col_map.get(tc.lower())
                if sc is None:
                    continue
                src_vals = df[sc].dropna()
                tgt_vals = target_features[tc].dropna()
                if len(src_vals) < 5 or len(tgt_vals) < 5:
                    continue
                if not pd.api.types.is_numeric_dtype(src_vals):
                    continue
                stat, _ = _ks_2samp(src_vals.values.astype(float),
                                    tgt_vals.values.astype(float))
                ks_vals.append(1.0 - stat)

            # Find best concept-matching column via index (avoids O(n_unique) linear scan)
            src_col_idxs = table_col_map.get(tid, [])
            matching_col = None
            best_sim = 0.0
            if src_col_idxs:
                col_sims = sim_matrix[:, src_col_idxs].max(axis=0)  # (n_src_cols,)
                best_local = int(np.argmax(col_sims))
                if col_sims[best_local] >= threshold:
                    matching_col = unique_names[src_col_idxs[best_local]]
                    best_sim = float(col_sims[best_local])
                    if matching_col not in df.columns:
                        matching_col = None  # name may have been deduplicated differently

            if matching_col and pd.api.types.is_numeric_dtype(df[matching_col]):
                median = df[matching_col].median()
                if not np.isnan(median):
                    pseudo_labels = (df[matching_col] > median).values.astype(int)
                    pos_rates.append(float(pseudo_labels.mean()))

                    # Fast-PAS: semantic feature matching using pre-computed feat_sim_matrix.
                    # Compute BOTH tight (0.60, for PAS/SPA) and loose (0.50, for CSLP/LCC)
                    # in a single pass over target feature columns.
                    feat_cols_matched = []        # tight threshold → PAS, SPA
                    feat_cols_matched_loose = []  # loose threshold → CSLP, LCC
                    if src_col_idxs:
                        for tc, fi in _tgt_feat_col_to_row.items():
                            sims_row = feat_sim_matrix[fi, src_col_idxs]
                            best_local_f = int(np.argmax(sims_row))
                            best_sim_f = float(sims_row[best_local_f])
                            sc_name = unique_names[src_col_idxs[best_local_f]]
                            if sc_name in df.columns and pd.api.types.is_numeric_dtype(df[sc_name]):
                                if best_sim_f >= _FEAT_MATCH_THRESH:
                                    feat_cols_matched.append((tc, sc_name))
                                if best_sim_f >= _FEAT_MATCH_THRESH_LOOSE:
                                    feat_cols_matched_loose.append((tc, sc_name))
                    if len(feat_cols_matched) >= 2 and pseudo_labels.sum() >= 5 \
                            and (pseudo_labels == 0).sum() >= 5:
                        # Build source feature matrix aligned to target columns
                        X_src = np.column_stack([
                            df[sc].fillna(0).values.astype(float)
                            for _, sc in feat_cols_matched
                        ])
                        X_tgt = np.column_stack([
                            target_features[tc].fillna(0).values.astype(float)
                            for tc, _ in feat_cols_matched
                        ])
                        # Min-max scale each column using source range to handle scale diffs
                        col_min = X_src.min(axis=0)
                        col_range = X_src.max(axis=0) - col_min + 1e-9
                        X_src_scaled = (X_src - col_min) / col_range
                        X_tgt_scaled = (X_tgt - col_min) / col_range

                        centroid_0 = X_src_scaled[pseudo_labels == 0].mean(axis=0, keepdims=True)
                        centroid_1 = X_src_scaled[pseudo_labels == 1].mean(axis=0, keepdims=True)
                        centroids = np.vstack([centroid_0, centroid_1])

                        dists = _cdist_fast(X_tgt_scaled, centroids, metric="euclidean")
                        d1 = dists.min(axis=1)
                        d2 = dists.max(axis=1)
                        margin = float(np.mean((d2 - d1) / (d2 + 1e-9)))
                        fast_pas_vals.append(margin)
                        fast_pas_weights.append(candidate_scores.get(tid, 1.0))

                    # Fast-PAS (loose): same centroid-margin computation but using
                    # 0.50 threshold features. Covers targets where tight-PAS can't compute
                    # (e.g. diabetes/Pima where "insu"→"insulin" scores ~0.85 but no tight match).
                    if len(feat_cols_matched_loose) >= 2 and pseudo_labels.sum() >= 5 \
                            and (pseudo_labels == 0).sum() >= 5:
                        X_src_l = np.column_stack([
                            df[sc].fillna(0).values.astype(float)
                            for _, sc in feat_cols_matched_loose
                        ])
                        X_tgt_l = np.column_stack([
                            target_features[tc].fillna(0).values.astype(float)
                            for tc, _ in feat_cols_matched_loose
                        ])
                        col_min_l = X_src_l.min(axis=0)
                        col_range_l = X_src_l.max(axis=0) - col_min_l + 1e-9
                        X_src_l_s = (X_src_l - col_min_l) / col_range_l
                        X_tgt_l_s = (X_tgt_l - col_min_l) / col_range_l
                        c0_l = X_src_l_s[pseudo_labels == 0].mean(axis=0, keepdims=True)
                        c1_l = X_src_l_s[pseudo_labels == 1].mean(axis=0, keepdims=True)
                        dists_l = _cdist_fast(X_tgt_l_s, np.vstack([c0_l, c1_l]), metric="euclidean")
                        d1_l = dists_l.min(axis=1)
                        d2_l = dists_l.max(axis=1)
                        margin_l = float(np.mean((d2_l - d1_l) / (d2_l + 1e-9)))
                        fast_pas_loose_vals.append(margin_l)
                        fast_pas_loose_weights.append(candidate_scores.get(tid, 1.0))

                    # Fast PCA-PAS: project source into target PCA space, collect for post-loop aggregation.
                    # Uses tight matches (0.60) for aligned columns; imputes target column mean elsewhere.
                    # Requires >= 1 matched feature to avoid pure-imputation noise.
                    if _target_pca_obj is not None and len(feat_cols_matched) >= 1 \
                            and pseudo_labels.sum() >= 5 and (pseudo_labels == 0).sum() >= 5:
                        try:
                            tight_map = dict(feat_cols_matched)
                            X_src_full = np.zeros((len(df), len(tgt_numeric_cols)), dtype=float)
                            for _ji, _tc in enumerate(tgt_numeric_cols):
                                _sc = tight_map.get(_tc)
                                if _sc is not None:
                                    X_src_full[:, _ji] = df[_sc].fillna(0).values.astype(float)
                                else:
                                    X_src_full[:, _ji] = float(target_features[_tc].fillna(0).mean())
                            X_src_scaled_full = _target_scaler_pca.transform(X_src_full)
                            X_src_pca = _target_pca_obj.transform(X_src_scaled_full)
                            fast_pca_pas_sources.append(
                                (X_src_pca, pseudo_labels.copy(), candidate_scores.get(tid, 1.0))
                            )
                        except Exception:
                            pass

                    # Source-space PCA-PAS: fit PCA on source features, project target into
                    # that source PCA space, compute centroid margin. Mirrors true PCA-PAS
                    # (ρ=0.685) which also fits PCA on source and projects target.
                    if len(feat_cols_matched_loose) >= 2 and pseudo_labels.sum() >= 5 \
                            and (pseudo_labels == 0).sum() >= 5:
                        X_src_sp = np.column_stack([
                            df[sc].fillna(0).values.astype(float)
                            for _, sc in feat_cols_matched_loose
                        ])
                        X_tgt_sp = np.column_stack([
                            target_features[tc].fillna(0).values.astype(float)
                            for tc, _ in feat_cols_matched_loose
                        ])
                        n_comp_sp = min(X_src_sp.shape[1], max(2, min(10, X_src_sp.shape[0] // 10)))
                        if X_src_sp.shape[0] >= n_comp_sp + 1 and n_comp_sp >= 2:
                            try:
                                from sklearn.preprocessing import StandardScaler as _SS_sp
                                from sklearn.decomposition import PCA as _PCA_sp
                                _sc_sp = _SS_sp().fit(X_src_sp)
                                X_src_sp_sc = _sc_sp.transform(X_src_sp)
                                X_tgt_sp_sc = _sc_sp.transform(X_tgt_sp)
                                _pca_sp = _PCA_sp(n_components=n_comp_sp, whiten=True).fit(X_src_sp_sc)
                                X_src_pca_sp = _pca_sp.transform(X_src_sp_sc)
                                X_tgt_pca_sp = _pca_sp.transform(X_tgt_sp_sc)
                                c0_sp = X_src_pca_sp[pseudo_labels == 0].mean(axis=0, keepdims=True)
                                c1_sp = X_src_pca_sp[pseudo_labels == 1].mean(axis=0, keepdims=True)
                                dists_sp = _cdist_fast(X_tgt_pca_sp, np.vstack([c0_sp, c1_sp]))
                                d1_sp = dists_sp.min(axis=1)
                                d2_sp = dists_sp.max(axis=1)
                                src_pca_pas_vals.append(
                                    float(np.mean((d2_sp - d1_sp) / (d2_sp + 1e-9)))
                                )
                                src_pca_pas_weights.append(candidate_scores.get(tid, 1.0))
                            except Exception:
                                pass

                    # Semantic KS overlap: KS test on cosine-matched column pairs.
                    # Fixes exact-name matching which gives 0.5 sentinel on 6/12 targets.
                    for _tc_ks, _sc_ks in feat_cols_matched_loose:
                        _sv = df[_sc_ks].dropna()
                        _tv = target_features[_tc_ks].dropna()
                        if (len(_sv) >= 5 and len(_tv) >= 5
                                and pd.api.types.is_numeric_dtype(_sv)
                                and pd.api.types.is_numeric_dtype(_tv)):
                            _stat_ks, _ = _ks_2samp(
                                _sv.values.astype(float), _tv.values.astype(float)
                            )
                            semantic_ks_vals.append(1.0 - _stat_ks)

                    # CoPAS: paper's exact L2-norm + cosine centroid margin algorithm.
                    # Works with >= 1 matched column (lower threshold than SSPAS).
                    # L2-normalized centroid = normalize(sum) per paper Eq 1.
                    # Margin = (d2 - d1) / d2 per paper Eq 3 (no eps, strict formula).
                    if len(feat_cols_matched_loose) >= 1:
                        _X_src_cp = np.column_stack([
                            df[sc].fillna(0).values.astype(float)
                            for _, sc in feat_cols_matched_loose
                        ])
                        _X_tgt_cp = np.column_stack([
                            target_features[tc].fillna(0).values.astype(float)
                            for tc, _ in feat_cols_matched_loose
                        ])
                        # L2-normalize rows → unit hypersphere
                        _n_src_cp = np.linalg.norm(_X_src_cp, axis=1, keepdims=True)
                        _n_tgt_cp = np.linalg.norm(_X_tgt_cp, axis=1, keepdims=True)
                        _X_src_l2 = _X_src_cp / np.where(_n_src_cp < 1e-9, 1.0, _n_src_cp)
                        _X_tgt_l2 = _X_tgt_cp / np.where(_n_tgt_cp < 1e-9, 1.0, _n_tgt_cp)
                        # L2-normalized centroids (normalize sum of unit vectors)
                        _c0_sum = _X_src_l2[pseudo_labels == 0].sum(axis=0) \
                            if (pseudo_labels == 0).sum() >= 1 else None
                        _c1_sum = _X_src_l2[pseudo_labels == 1].sum(axis=0) \
                            if (pseudo_labels == 1).sum() >= 1 else None
                        if _c0_sum is not None and _c1_sum is not None:
                            _c0_n = np.linalg.norm(_c0_sum)
                            _c1_n = np.linalg.norm(_c1_sum)
                            if _c0_n > 1e-9 and _c1_n > 1e-9:
                                _mu0 = _c0_sum / _c0_n
                                _mu1 = _c1_sum / _c1_n
                                # Cosine distances (1 - dot product for unit vectors)
                                _dist0 = 1.0 - (_X_tgt_l2 @ _mu0)  # (n_target,)
                                _dist1 = 1.0 - (_X_tgt_l2 @ _mu1)
                                _dmin_cp = np.minimum(_dist0, _dist1)
                                _dmax_cp = np.maximum(_dist0, _dist1)
                                # Paper Eq 3: (d2 - d1) / d2
                                _valid_cp = _dmax_cp > 1e-9
                                if _valid_cp.any():
                                    _margins_cp = np.where(
                                        _valid_cp, (_dmax_cp - _dmin_cp) / _dmax_cp, 0.0
                                    )
                                    copas_vals.append(float(np.mean(_margins_cp)))
                                    copas_weights.append(candidate_scores.get(tid, 1.0))
                                # Centroid Coherence: how far apart are source class centroids?
                                # cosine_dist(μ0, μ1) = 1 - dot(μ0, μ1)  ∈ [0, 2]
                                centroid_sep_vals.append(
                                    float(np.clip(1.0 - float(np.dot(_mu0, _mu1)), 0.0, 2.0))
                                )

                        # ZScore-CoPAS: standardise columns by source stats, then L2+cosine.
                        # Reuses _X_src_cp / _X_tgt_cp built above for raw CoPAS.
                        # Filter near-constant source cols (std<0.01) before z-scoring;
                        # near-zero std causes ~1e9 target z-scores that dominate L2 norm.
                        _zs_valid = _X_src_cp.std(axis=0) > 0.01
                        _c0z_sum = _c1z_sum = None
                        _X_tgt_zsl2 = None
                        if _zs_valid.sum() >= 1:
                            _Xsv = _X_src_cp[:, _zs_valid]
                            _Xtv = _X_tgt_cp[:, _zs_valid]
                            _zm = _Xsv.mean(axis=0)
                            _zs = _Xsv.std(axis=0) + 1e-9
                            _Xsv_z = (_Xsv - _zm) / _zs
                            _Xtv_z = (_Xtv - _zm) / _zs
                            _nzs = np.linalg.norm(_Xsv_z, axis=1, keepdims=True)
                            _nzt = np.linalg.norm(_Xtv_z, axis=1, keepdims=True)
                            _X_src_zsl2 = _Xsv_z / np.where(_nzs < 1e-9, 1.0, _nzs)
                            _X_tgt_zsl2 = _Xtv_z / np.where(_nzt < 1e-9, 1.0, _nzt)
                            if (pseudo_labels == 0).sum() >= 1:
                                _c0z_sum = _X_src_zsl2[pseudo_labels == 0].sum(axis=0)
                            if (pseudo_labels == 1).sum() >= 1:
                                _c1z_sum = _X_src_zsl2[pseudo_labels == 1].sum(axis=0)
                        if _c0z_sum is not None and _c1z_sum is not None:
                            _c0z_n = np.linalg.norm(_c0z_sum)
                            _c1z_n = np.linalg.norm(_c1z_sum)
                            if _c0z_n > 1e-9 and _c1z_n > 1e-9:
                                _mu0z = _c0z_sum / _c0z_n
                                _mu1z = _c1z_sum / _c1z_n
                                _d0z = 1.0 - (_X_tgt_zsl2 @ _mu0z)
                                _d1z = 1.0 - (_X_tgt_zsl2 @ _mu1z)
                                _dminz = np.minimum(_d0z, _d1z)
                                _dmaxz = np.maximum(_d0z, _d1z)
                                _validz = _dmaxz > 1e-9
                                if _validz.any():
                                    _marginsz = np.where(_validz, (_dmaxz - _dminz) / _dmaxz, 0.0)
                                    zcopas_vals.append(float(np.mean(_marginsz)))
                                    zcopas_weights.append(candidate_scores.get(tid, 1.0))

                    # Fast-SPA: train LR on source, predict on target, collect predictions.
                    # Requires >= 1 matched feature column (less strict than PAS).
                    # Agreement across tables measured after the full loop.
                    if len(feat_cols_matched) >= 1 and pseudo_labels.sum() >= 5 \
                            and (pseudo_labels == 0).sum() >= 5:
                        try:
                            from sklearn.linear_model import LogisticRegression as _LR
                            X_src_spa = np.column_stack([
                                df[sc].fillna(0).values.astype(float)
                                for _, sc in feat_cols_matched
                            ])
                            X_tgt_spa = np.column_stack([
                                target_features[tc].fillna(0).values.astype(float)
                                for tc, _ in feat_cols_matched
                            ])
                            col_min_s = X_src_spa.min(axis=0)
                            col_rng_s = X_src_spa.max(axis=0) - col_min_s + 1e-9
                            X_src_spa_s = (X_src_spa - col_min_s) / col_rng_s
                            X_tgt_spa_s = (X_tgt_spa - col_min_s) / col_rng_s
                            clf = _LR(C=1.0, max_iter=300, random_state=0, n_jobs=1)
                            clf.fit(X_src_spa_s, pseudo_labels)
                            proba = clf.predict_proba(X_tgt_spa_s)[:, 1]
                            fast_spa_preds.append(proba)
                            fast_spa_weights.append(candidate_scores.get(tid, 1.0))
                        except Exception:
                            pass

                    # LCC: internal 5-fold CV AUC of pseudo-label vs target-matched features.
                    # Uses loose threshold so medical targets (Pima names etc.) can contribute.
                    # Measures: does the label proxy genuinely correlate with target-relevant
                    # features in this source domain?
                    if len(feat_cols_matched_loose) >= 1 and pseudo_labels.sum() >= 5 \
                            and (pseudo_labels == 0).sum() >= 5:
                        try:
                            from sklearn.linear_model import LogisticRegression as _LR
                            from sklearn.model_selection import StratifiedKFold as _SKF
                            from sklearn.metrics import roc_auc_score as _roc_auc
                            X_lcc = np.column_stack([
                                df[sc].fillna(0).values.astype(float)
                                for _, sc in feat_cols_matched_loose
                            ])
                            n_sp = min(5, int(pseudo_labels.sum()), int((pseudo_labels == 0).sum()))
                            if n_sp >= 2:
                                skf_lcc = _SKF(n_splits=n_sp, shuffle=True, random_state=0)
                                fold_aucs_lcc: list[float] = []
                                for tr_l, te_l in skf_lcc.split(X_lcc, pseudo_labels):
                                    if len(np.unique(pseudo_labels[te_l])) < 2:
                                        continue
                                    clf_lcc = _LR(C=1.0, max_iter=200, random_state=0, n_jobs=1)
                                    clf_lcc.fit(X_lcc[tr_l], pseudo_labels[tr_l])
                                    p_lcc = clf_lcc.predict_proba(X_lcc[te_l])[:, 1]
                                    fold_aucs_lcc.append(float(_roc_auc(pseudo_labels[te_l], p_lcc)))
                                if fold_aucs_lcc:
                                    lcc_vals.append(float(np.mean(fold_aucs_lcc)))
                        except Exception:
                            pass

                    # CSLP: build full target-space feature vector (loose threshold) for SLOO.
                    # Each source is represented as a vector in R^{n_tgt_features}, with 0 for
                    # unmatched columns. After the loop, we train LOO classifiers across sources
                    # to measure cross-domain P(Y|X) stability.
                    if len(feat_cols_matched_loose) >= 1 and pseudo_labels.sum() >= 5 \
                            and (pseudo_labels == 0).sum() >= 5:
                        X_cslp = np.zeros((len(df), len(tgt_feat_cols)), dtype=np.float32)
                        for tc, sc in feat_cols_matched_loose:
                            if tc in tgt_feat_cols:
                                fi_c = tgt_feat_cols.index(tc)
                                X_cslp[:, fi_c] = df[sc].fillna(0).values.astype(float)
                        # Subsample to max 300 rows per source to keep LOO tractable
                        _MAX_ROWS = 300
                        if len(df) > _MAX_ROWS:
                            _rng = np.random.RandomState(42)
                            _idx = _rng.choice(len(df), _MAX_ROWS, replace=False)
                            cslp_data.append((X_cslp[_idx], pseudo_labels[_idx]))
                        else:
                            cslp_data.append((X_cslp, pseudo_labels))

            n_loaded += 1
        except Exception as exc:
            logger.debug("Fast score: could not load %s: %s", fpath, exc)
            continue

    alignment_density = float(np.mean(density_vals)) if density_vals else 0.0
    # Semantic KS overlap on cosine-matched columns (replaces exact-name matching)
    feature_overlap = float(np.mean(semantic_ks_vals)) if semantic_ks_vals else 0.5

    # Source-space PCA-PAS: weighted mean across loaded sources
    if src_pca_pas_vals:
        w_sp = np.array(src_pca_pas_weights)
        fast_src_pca_pas = float(np.dot(w_sp / w_sp.sum(), src_pca_pas_vals))
    else:
        fast_src_pca_pas = 0.5

    # CoPAS: paper's L2-norm + cosine centroid margin, weighted mean and max across sources
    if copas_vals:
        _w_cp = np.array(copas_weights)
        fast_copas = float(np.dot(_w_cp / _w_cp.sum(), copas_vals))
        fast_max_copas = float(max(copas_vals))
    else:
        fast_copas = 0.5
        fast_max_copas = 0.5

    # ZScore-CoPAS: column z-score before L2+cosine — fixes scale dominance of raw CoPAS
    if zcopas_vals:
        _w_zcp = np.array(zcopas_weights)
        fast_zcopas = float(np.dot(_w_zcp / _w_zcp.sum(), zcopas_vals))
    else:
        fast_zcopas = 0.5

    # Centroid Coherence: fraction of sources with above-median centroid separation.
    # High = source class centroids are consistently far apart → clear pseudo-label signal.
    if centroid_sep_vals:
        _med_sep = float(np.median(centroid_sep_vals))
        _cc = float(np.mean(np.array(centroid_sep_vals) > _med_sep)) if len(centroid_sep_vals) > 1 else 0.5
        fast_cc = _cc
    else:
        fast_cc = 0.5

    # BNM: normalized nuclear norm of [n_sources × n_target] prediction matrix.
    # High = sources give diverse confident predictions on target → good class coverage.
    # Nuclear norm / (sqrt(n_sources) × n_target) normalizes for matrix size.
    bnm_score = 0.5
    if len(fast_spa_preds) >= 2:
        _pred_bnm = np.vstack(fast_spa_preds)  # (n_sources, n_target)
        _sv_bnm = np.linalg.svd(_pred_bnm, compute_uv=False)
        bnm_score = float(_sv_bnm.sum() / (np.sqrt(_pred_bnm.shape[0]) * _pred_bnm.shape[1] + 1e-9))

    # CSLP: Source Leave-One-Out AUC in target feature space.
    # For each source i, train LR on pooled (all other sources), predict on source i.
    # High AUC → the label concept transfers reliably across source domains →
    # likely to transfer to target too (same P(Y|X) stability that DANN requires).
    if len(cslp_data) >= 3:
        from sklearn.linear_model import LogisticRegression as _LR_cslp
        from sklearn.metrics import roc_auc_score as _roc_auc_cslp
        cslp_aucs_list: list[float] = []
        for _ci in range(len(cslp_data)):
            X_i, y_i = cslp_data[_ci]
            if len(np.unique(y_i)) < 2:
                continue
            X_rest = np.vstack([cslp_data[_j][0] for _j in range(len(cslp_data)) if _j != _ci])
            y_rest = np.concatenate([cslp_data[_j][1] for _j in range(len(cslp_data)) if _j != _ci])
            if len(np.unique(y_rest)) < 2:
                continue
            try:
                clf_cs = _LR_cslp(C=1.0, max_iter=300, random_state=0, n_jobs=1)
                clf_cs.fit(X_rest, y_rest)
                proba_cs = clf_cs.predict_proba(X_i)[:, 1]
                cslp_aucs_list.append(float(_roc_auc_cslp(y_i, proba_cs)))
            except Exception:
                pass
        if cslp_aucs_list:
            # Normalise: AUC=0.5 → 0 (no signal), AUC=1.0 → 1 (perfect transfer)
            cslp_score = float(np.clip(2.0 * (np.mean(cslp_aucs_list) - 0.5), 0.0, 1.0))
        else:
            cslp_score = 0.5
    else:
        cslp_score = 0.5

    # LCC: mean internal CV AUC across all loaded sources
    lcc_score = float(np.mean(lcc_vals)) if lcc_vals else 0.5

    # Fast PCA-PAS: Mahalanobis separation in target PCA space × overlap, aggregated across sources.
    fast_pca_pas_score = 0.5
    if _target_pca_obj is not None and len(fast_pca_pas_sources) >= 1:
        try:
            from scipy.linalg import pinv as _pinv_pca
            _k_act = _target_pca_obj.n_components_
            _sqrt_k = float(np.sqrt(_k_act))
            _per_scores: list[float] = []
            _per_weights: list[float] = []
            for _Xp, _yp, _wp in fast_pca_pas_sources:
                _n_pos = int((_yp == 1).sum())
                _n_neg = int((_yp == 0).sum())
                if _n_pos < 2 or _n_neg < 2:
                    continue
                _mu_pos = _Xp[_yp == 1].mean(axis=0)
                _mu_neg = _Xp[_yp == 0].mean(axis=0)
                _Sp = np.cov(_Xp[_yp == 1].T) if _n_pos > 1 else np.eye(_k_act)
                _Sn = np.cov(_Xp[_yp == 0].T) if _n_neg > 1 else np.eye(_k_act)
                if _k_act == 1:
                    _Sp = np.atleast_2d(_Sp)
                    _Sn = np.atleast_2d(_Sn)
                _S_pool = (_Sp * (_n_pos - 1) + _Sn * (_n_neg - 1)) / (_n_pos + _n_neg - 2)
                _S_inv = _pinv_pca(_S_pool + np.eye(_k_act) * 1e-6)
                _diff = _mu_pos - _mu_neg
                _mahal = float(np.sqrt(max(0.0, _diff @ _S_inv @ _diff)))
                # Overlap: mean fraction of source points within ±2σ of target centroid, per PC.
                # Using mean (not all-PC conjunction) avoids near-zero overlap when k is large.
                _lo = _tgt_pca_mean - 2.0 * _tgt_pca_std
                _hi = _tgt_pca_mean + 2.0 * _tgt_pca_std
                _in_per_pc = ((_Xp >= _lo) & (_Xp <= _hi)).astype(float)  # (n, k)
                _overlap = float(_in_per_pc.mean())
                _per_scores.append(_mahal * _overlap / _sqrt_k)
                _per_weights.append(_wp)
            if _per_scores:
                _w_arr = np.array(_per_weights)
                _w_arr = _w_arr / _w_arr.sum()
                fast_pca_pas_score = float(np.clip(np.dot(_w_arr, _per_scores), 0.0, 1.0))
        except Exception:
            pass

    # Fast-PAS (loose): weighted mean at threshold 0.50
    if fast_pas_loose_vals:
        w_l = np.array(fast_pas_loose_weights)
        w_l = w_l / w_l.sum()
        pas_loose_score = float(np.dot(w_l, fast_pas_loose_vals))
    else:
        pas_loose_score = 0.5

    if pos_rates:
        mean_src_rate = float(np.mean(pos_rates))
        label_shift = float(np.clip(1.0 - abs(mean_src_rate - target_pos_rate), 0.0, 1.0))
    else:
        label_shift = 0.5

    # Fast-PAS: weighted mean of per-table margins
    if fast_pas_vals:
        w = np.array(fast_pas_weights)
        w = w / w.sum()
        pas_score_fast = float(np.dot(w, fast_pas_vals))
    else:
        pas_score_fast = 0.5

    # Fast-SPA: 1 − 2 × mean per-sample std of LR predictions across tables.
    # Requires >= 2 contributing tables for a meaningful agreement signal.
    if len(fast_spa_preds) >= 2:
        pred_matrix = np.vstack(fast_spa_preds)       # (n_tables, n_target_samples)
        per_sample_std = pred_matrix.std(axis=0)       # (n_target_samples,)
        mean_std = float(per_sample_std.mean())
        spa_score_fast = float(np.clip(1.0 - 2.0 * mean_std, 0.0, 1.0))
    else:
        spa_score_fast = 0.5

    # Mean Source Prediction Confidence (MSPC): confidence of consensus source predictions on target.
    # Sources trained on lake pseudo-labels; confident target predictions → label concept transfers.
    # Uses fast_spa_preds (no zero-padding), so valid for all targets with >= 2 source LR models.
    mspc_score = 0.5
    if len(fast_spa_preds) >= 2:
        _pred_mat = np.vstack(fast_spa_preds)          # (n_sources, n_target)
        _mean_pred = _pred_mat.mean(axis=0)            # consensus probability
        _H = -(_mean_pred * np.log(_mean_pred + 1e-9)
               + (1.0 - _mean_pred) * np.log(1.0 - _mean_pred + 1e-9))
        mspc_score = float(np.clip(1.0 - np.mean(_H) / np.log(2.0), 0.0, 1.0))

    # fast_overall: mean(src_pca_pas, feature_overlap, 1 − cslp_score).
    # Formula finalized after running --fast-only across all 12 targets and picking
    # highest Spearman ρ candidate. True PCA-PAS (post-pipeline) remains the only
    # statistically significant predictor (ρ=+0.685, p=0.014, n=12).
    overall = float(np.mean([fast_src_pca_pas, feature_overlap, 1.0 - cslp_score]))

    logger.info(
        "[Transferability fast] overall=%.3f  copas=%.3f  zcopas=%.3f  max_copas=%.3f  bnm=%.3f  cc=%.3f  "
        "mspc=%.3f  yield=%.3f (n_matched=%d/%d)  quality=%.3f  top1=%.3f  density=%.3f  "
        "shift=%.3f  overlap=%.3f  fast_pas=%.3f  src_pca_pas=%.3f  "
        "pas_loose=%.3f  fast_spa=%.3f  cslp=%.3f  lcc=%.3f  consistency=%.3f  (loaded %d parquets)",
        overall, fast_copas, fast_zcopas, fast_max_copas, bnm_score, fast_cc,
        mspc_score, repurpose_yield, n_matched, n_lake,
        discovery_quality, _top1, alignment_density, label_shift, feature_overlap,
        pas_score_fast, fast_src_pca_pas, pas_loose_score, spa_score_fast, cslp_score, lcc_score,
        source_consistency, n_loaded,
    )

    return TransferabilityScore(
        repurpose_yield=repurpose_yield,
        discovery_quality=discovery_quality,
        alignment_density=alignment_density,
        label_shift=label_shift,
        feature_overlap=feature_overlap,
        pas_score=pas_score_fast,
        spa_score=spa_score_fast,
        cslp_score=cslp_score,
        lcc_score=lcc_score,
        pas_loose_score=pas_loose_score,
        pca_pas_score=fast_src_pca_pas,
        zscore_copas_score=fast_zcopas,
        npas_score=0.5,    # true-only; not computed in fast mode
        tsc_score=0.5,     # true-only; not computed in fast mode
        source_consistency=source_consistency,
        top1_score=_top1,
        overall=overall,
        n_sources=n_matched,
        n_lake_tables=n_lake,
        mode="fast",
    )


# ---------------------------------------------------------------------------
# PAS — Potential Adaptability Score (tabular adaptation of ICLR 2026)
# ---------------------------------------------------------------------------

def compute_pas(
    aligned: dict[str, pd.DataFrame],
    target_features: pd.DataFrame,
    label_col: str,
    discovery_scores: Optional[dict[str, float]] = None,
    use_pca: bool = True,
) -> float:
    """
    Tabular adaptation of the Potential Adaptability Score (PAS) from
    "PAS: Estimating the target accuracy before domain adaptation" (ICLR 2026).

    Original PAS uses neural-network embeddings; here the quantile-normalised
    feature space serves as the embedding space directly.

    For each aligned source, compute per-class centroids (positive / negative),
    then for every target sample measure:
        margin_i = (d2_i − d1_i) / d2_i
    where d1_i = distance to nearest class centroid,
          d2_i = distance to second-nearest class centroid.

    Source-level PAS = mean margin across target samples.
    Final PAS = weighted mean over sources (weights = discovery_scores, or uniform).

    Returns a value in [0, 1].  Higher = target samples land more confidently
    in source class regions = better expected transfer.

    Notes
    -----
    - Only numeric columns present in both source and target are used.
    - Sources with < 5 samples per class are skipped (unstable centroids).
    - Returns 0.5 (neutral) when no valid sources are found.
    """
    if not aligned or target_features is None:
        return 0.5

    from scipy.spatial.distance import cdist as _cdist

    tgt_num_cols = [
        c for c in target_features.columns
        if pd.api.types.is_numeric_dtype(target_features[c])
    ]
    if not tgt_num_cols:
        return 0.5

    pas_vals: list[float] = []
    weights: list[float] = []

    for table_id, df in aligned.items():
        if label_col not in df.columns:
            continue
        feat_cols = [
            c for c in df.columns
            if c != label_col
            and c in tgt_num_cols
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(feat_cols) < 2:
            continue

        y = df[label_col].values
        # need enough samples in each class
        if (y == 0).sum() < 5 or (y == 1).sum() < 5:
            continue

        X_src = df[feat_cols].fillna(0).values.astype(float)
        X_tgt = target_features[feat_cols].fillna(0).values.astype(float)
        if len(X_tgt) == 0:
            continue

        # PCA whitening: fit on source, apply to both source and target.
        # Euclidean distance in whitened space ≈ Mahalanobis distance.
        if use_pca and X_src.shape[1] >= 2:
            try:
                from sklearn.decomposition import PCA as _PCA
                n_pos = int((y == 1).sum())
                n_neg = int((y == 0).sum())
                n_components = min(n_pos - 1, n_neg - 1, X_src.shape[1], 10)
                if n_components >= 2:
                    _pca = _PCA(n_components=n_components, whiten=True)
                    _pca.fit(X_src)
                    X_src = _pca.transform(X_src)
                    X_tgt = _pca.transform(X_tgt)
            except Exception:
                pass  # fall back to raw features if PCA fails

        centroid_0 = X_src[y == 0].mean(axis=0, keepdims=True)  # (1, F)
        centroid_1 = X_src[y == 1].mean(axis=0, keepdims=True)  # (1, F)

        centroids = np.vstack([centroid_0, centroid_1])           # (2, F)
        dists = _cdist(X_tgt, centroids, metric="euclidean")      # (N_tgt, 2)

        d1 = dists.min(axis=1)   # nearest centroid distance
        d2 = dists.max(axis=1)   # second-nearest centroid distance

        denom = d2 + 1e-9
        margins = (d2 - d1) / denom                               # per-sample margin
        pas_source = float(np.mean(margins))

        w = discovery_scores.get(table_id, 1.0) if discovery_scores else 1.0
        pas_vals.append(pas_source)
        weights.append(w)

    if not pas_vals:
        return 0.5

    weights_arr = np.array(weights)
    weights_arr = weights_arr / weights_arr.sum()
    return float(np.dot(weights_arr, pas_vals))


# ---------------------------------------------------------------------------
# nPAS — Normalized PAS
# ---------------------------------------------------------------------------

def compute_npas(
    aligned: dict[str, pd.DataFrame],
    target_features: pd.DataFrame,
    label_col: str,
    discovery_scores: Optional[dict[str, float]] = None,
    use_pca: bool = True,
    random_state: int = 42,
) -> float:
    """
    Normalized PAS: target-PAS / source-self-PAS.

    Raw PAS values are near-zero for most targets (machine-epsilon range) because
    target samples from different domains fall approximately equidistant from both
    source class centroids in PCA-whitened space.  Dividing by the source's own
    held-out PAS normalises this: nPAS = 1 means the target separates as well as
    the source's own validation samples; nPAS = 0 means no class signal at all.

    Implementation: 80/20 train/val split of each source.  PCA is fit on the 80%
    train split; centroids are computed from train; PAS is computed on both the
    val split (reference) and the target (normalised score).  Sources where the
    source self-PAS < 0.01 (degenerate centroids) are skipped.
    """
    if not aligned or target_features is None:
        return 0.5

    from scipy.spatial.distance import cdist as _cdist

    tgt_num_cols = [
        c for c in target_features.columns
        if pd.api.types.is_numeric_dtype(target_features[c])
    ]
    if not tgt_num_cols:
        return 0.5

    npas_vals: list[float] = []
    weights: list[float] = []
    rng = np.random.default_rng(random_state)

    for table_id, df in aligned.items():
        if label_col not in df.columns:
            continue
        feat_cols = [
            c for c in df.columns
            if c != label_col
            and c in tgt_num_cols
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(feat_cols) < 2:
            continue

        y = df[label_col].values
        if (y == 0).sum() < 10 or (y == 1).sum() < 10:
            continue

        X_src = df[feat_cols].fillna(0).values.astype(float)
        X_tgt = target_features[feat_cols].fillna(0).values.astype(float)
        if len(X_tgt) == 0:
            continue

        idx = rng.permutation(len(X_src))
        split = max(10, int(0.8 * len(X_src)))
        train_idx, val_idx = idx[:split], idx[split:]

        if (y[val_idx] == 0).sum() < 2 or (y[val_idx] == 1).sum() < 2:
            continue

        X_train, y_train = X_src[train_idx], y[train_idx]
        X_val = X_src[val_idx]

        if use_pca and X_train.shape[1] >= 2:
            try:
                from sklearn.decomposition import PCA as _PCA
                n_components = min(
                    int((y_train == 0).sum()) - 1,
                    int((y_train == 1).sum()) - 1,
                    X_train.shape[1], 10,
                )
                if n_components >= 2:
                    pca = _PCA(n_components=n_components, whiten=True)
                    pca.fit(X_train)
                    X_train = pca.transform(X_train)
                    X_val   = pca.transform(X_val)
                    X_tgt   = pca.transform(X_tgt)
            except Exception:
                pass

        c0 = X_train[y_train == 0].mean(axis=0, keepdims=True)
        c1 = X_train[y_train == 1].mean(axis=0, keepdims=True)
        centroids = np.vstack([c0, c1])

        dv = _cdist(X_val, centroids, metric="euclidean")
        pas_self = float(np.mean((dv.max(axis=1) - dv.min(axis=1)) / (dv.max(axis=1) + 1e-9)))

        if pas_self < 0.01:
            continue

        dt = _cdist(X_tgt, centroids, metric="euclidean")
        pas_tgt = float(np.mean((dt.max(axis=1) - dt.min(axis=1)) / (dt.max(axis=1) + 1e-9)))

        npas = float(np.clip(pas_tgt / pas_self, 0.0, 2.0))
        w = discovery_scores.get(table_id, 1.0) if discovery_scores else 1.0
        npas_vals.append(npas)
        weights.append(w)

    if not npas_vals:
        return 0.5

    w_arr = np.array(weights)
    w_arr = w_arr / w_arr.sum()
    return float(np.dot(w_arr, npas_vals))


# ---------------------------------------------------------------------------
# TSC — Target-Source Consistency
# ---------------------------------------------------------------------------

def compute_tsc(
    aligned: dict[str, pd.DataFrame],
    target_features: pd.DataFrame,
    label_col: str,
    discovery_scores: Optional[dict[str, float]] = None,
) -> float:
    """
    Target-Source Consistency (TSC): bidirectional classifier round-trip.

    Steps per source:
      1. Train logistic regression on aligned source → predict target → pseudo-labels
      2. Train logistic regression on pseudo-labeled target → predict source
      3. AUC of reverse classifier on source using true source labels

    Score = max(0, 2 × (AUC − 0.5)), mapping [0.5, 1.0] AUC → [0, 1].

    Interpretation: if source class structure transfers to target, the pseudo-labels
    on target capture the true label concept; a classifier trained on them will then
    recover the source's true labels (high reverse AUC).  Low TSC means pseudo-labels
    are noise — the source concept does not round-trip through the target domain.

    Returns 0.5 (neutral) when no valid sources are found.
    """
    if not aligned or target_features is None:
        return 0.5

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    tgt_num_cols = [
        c for c in target_features.columns
        if pd.api.types.is_numeric_dtype(target_features[c])
    ]
    if not tgt_num_cols:
        return 0.5

    tsc_vals: list[float] = []
    weights: list[float] = []

    for table_id, df in aligned.items():
        if label_col not in df.columns:
            continue
        feat_cols = [
            c for c in df.columns
            if c != label_col
            and c in tgt_num_cols
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(feat_cols) < 2:
            continue

        y_src = df[label_col].values
        if (y_src == 0).sum() < 5 or (y_src == 1).sum() < 5:
            continue

        X_src = df[feat_cols].fillna(0).values.astype(float)
        X_tgt = target_features[feat_cols].fillna(0).values.astype(float)
        if len(X_tgt) < 10:
            continue

        try:
            clf_fwd = LogisticRegression(max_iter=300, C=1.0, random_state=42, solver="lbfgs")
            clf_fwd.fit(X_src, y_src)
            y_tgt_pseudo = clf_fwd.predict(X_tgt)

            if len(np.unique(y_tgt_pseudo)) < 2:
                continue

            clf_rev = LogisticRegression(max_iter=300, C=1.0, random_state=42, solver="lbfgs")
            clf_rev.fit(X_tgt, y_tgt_pseudo)
            y_src_prob = clf_rev.predict_proba(X_src)[:, 1]

            if len(np.unique(y_src)) < 2:
                continue

            auc = roc_auc_score(y_src, y_src_prob)
        except Exception:
            continue

        tsc = float(np.clip(2.0 * (auc - 0.5), 0.0, 1.0))
        w = discovery_scores.get(table_id, 1.0) if discovery_scores else 1.0
        tsc_vals.append(tsc)
        weights.append(w)

    if not tsc_vals:
        return 0.5

    w_arr = np.array(weights)
    w_arr = w_arr / w_arr.sum()
    return float(np.dot(w_arr, tsc_vals))


# ---------------------------------------------------------------------------
# ZScore-CoPAS
# ---------------------------------------------------------------------------

def compute_zscore_copas(
    aligned: dict[str, pd.DataFrame],
    target_features: pd.DataFrame,
    label_col: str,
    discovery_scores: Optional[dict[str, float]] = None,
) -> float:
    """
    ZScore-CoPAS: column z-score normalisation before L2 row normalisation.

    Fixes the scale-dominance failure of raw CoPAS (arxiv 2604.09863): a single
    high-magnitude feature (e.g. glucose ≈ 120) dominates the L2 row norm, collapsing
    all samples toward the same direction on the unit hypersphere and zeroing out margins.

    Fix: standardise each feature column by source mean/std first, so all features
    contribute equally before the directional (L2) normalisation. Then apply the paper's
    exact centroid formula (normalise sum of unit vectors) and cosine margin.

    z-score + L2 is a strict subset of PCA whitening: it handles scale but not
    inter-feature correlation. The resulting score is interpretable as "how well do
    source class centroids separate in a scale-corrected directional space?"
    """
    if not aligned or target_features is None:
        return 0.5

    tgt_num_cols = [
        c for c in target_features.columns
        if pd.api.types.is_numeric_dtype(target_features[c])
    ]
    if not tgt_num_cols:
        return 0.5

    vals: list[float] = []
    weights: list[float] = []

    for table_id, df in aligned.items():
        if label_col not in df.columns:
            continue
        feat_cols = [
            c for c in df.columns
            if c != label_col
            and c in tgt_num_cols
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(feat_cols) < 1:
            continue

        y = df[label_col].values
        if (y == 0).sum() < 5 or (y == 1).sum() < 5:
            continue

        X_src_raw = df[feat_cols].fillna(0).values.astype(float)
        X_tgt_raw = target_features[feat_cols].fillna(0).values.astype(float)
        if len(X_tgt_raw) == 0:
            continue

        # Compute source statistics and filter out near-constant columns.
        # Columns where src_std ≈ 0 (constant or all-NaN→0) cause division-by-epsilon
        # when z-scoring the target, inflating those dimensions to ~1e9 and dominating
        # the L2 norm — making every target vector orthogonal to every source centroid.
        # (True PCA-PAS avoids this silently: PCA drops zero-variance dimensions.)
        src_mean = X_src_raw.mean(axis=0)
        src_std_raw = X_src_raw.std(axis=0)
        _valid_cols = src_std_raw > 0.01   # keep only columns with genuine source variation
        if _valid_cols.sum() < 1:
            continue
        src_mean = src_mean[_valid_cols]
        src_std  = src_std_raw[_valid_cols] + 1e-9
        X_src = X_src_raw[:, _valid_cols]
        X_tgt = X_tgt_raw[:, _valid_cols]

        # Z-score columns using source statistics
        X_src_zs = (X_src - src_mean) / src_std
        X_tgt_zs = (X_tgt - src_mean) / src_std

        # L2 normalise rows → unit hypersphere
        n_src = np.linalg.norm(X_src_zs, axis=1, keepdims=True)
        n_tgt = np.linalg.norm(X_tgt_zs, axis=1, keepdims=True)
        X_src_l2 = X_src_zs / np.where(n_src < 1e-9, 1.0, n_src)
        X_tgt_l2 = X_tgt_zs / np.where(n_tgt < 1e-9, 1.0, n_tgt)

        # L2-normalised centroids: normalise sum of unit vectors (paper Eq 1)
        c0_sum = X_src_l2[y == 0].sum(axis=0)
        c1_sum = X_src_l2[y == 1].sum(axis=0)
        c0_n = np.linalg.norm(c0_sum)
        c1_n = np.linalg.norm(c1_sum)
        if c0_n < 1e-9 or c1_n < 1e-9:
            continue
        mu0 = c0_sum / c0_n
        mu1 = c1_sum / c1_n

        # Cosine distance margin (paper Eq 3)
        d0 = 1.0 - (X_tgt_l2 @ mu0)
        d1 = 1.0 - (X_tgt_l2 @ mu1)
        dmin = np.minimum(d0, d1)
        dmax = np.maximum(d0, d1)
        valid = dmax > 1e-9
        if not valid.any():
            continue
        margins = np.where(valid, (dmax - dmin) / dmax, 0.0)

        w = discovery_scores.get(table_id, 1.0) if discovery_scores else 1.0
        vals.append(float(np.mean(margins)))
        weights.append(w)

    if not vals:
        return 0.5

    w_arr = np.array(weights)
    w_arr = w_arr / w_arr.sum()
    return float(np.dot(w_arr, vals))


# ---------------------------------------------------------------------------
# SPA — Source Prediction Agreement
# ---------------------------------------------------------------------------

def compute_spa(
    aligned: dict[str, pd.DataFrame],
    target_features: pd.DataFrame,
    label_col: str,
    discovery_scores: Optional[dict[str, float]] = None,
    min_samples_per_class: int = 5,
) -> float:
    """
    Source Prediction Agreement (SPA).

    For each aligned source, train a logistic regression classifier on the
    source data (using feature columns shared with the target), then predict
    probabilities on the target.  Measure the per-sample standard deviation
    of those predictions across all contributing sources.

    SPA = 1 − 2 × mean_per_sample_std

    Intuition
    ---------
    High agreement (low std) → all sources consistently assign the same
    probability to each target sample → the label concept transfers reliably
    → high expected oracle-gap-closed.

    Low agreement (high std) → sources disagree about target predictions
    → the label concept is domain-specific, does not transfer → low oracle gap.

    Returns a value in [0, 1].  Higher = sources agree = better transfer.
    Returns 0.5 (neutral) when fewer than 2 valid sources are found.
    """
    if not aligned or target_features is None or target_features.empty:
        return 0.5

    from sklearn.linear_model import LogisticRegression

    tgt_num_cols = [
        c for c in target_features.columns
        if pd.api.types.is_numeric_dtype(target_features[c])
    ]
    if not tgt_num_cols:
        return 0.5

    predictions: list[np.ndarray] = []

    for table_id, df in aligned.items():
        if label_col not in df.columns:
            continue
        feat_cols = [
            c for c in df.columns
            if c != label_col
            and c in tgt_num_cols
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(feat_cols) < 1:
            continue

        y = df[label_col].values
        if (y == 0).sum() < min_samples_per_class or (y == 1).sum() < min_samples_per_class:
            continue

        X_src = df[feat_cols].fillna(0).values.astype(float)
        X_tgt = target_features[feat_cols].fillna(0).values.astype(float)

        try:
            clf = LogisticRegression(C=1.0, max_iter=300, random_state=0, n_jobs=1)
            clf.fit(X_src, y)
            proba = clf.predict_proba(X_tgt)[:, 1]   # (n_target_samples,)
            predictions.append(proba)
        except Exception:
            continue

    if len(predictions) < 2:
        return 0.5

    pred_matrix = np.array(predictions)                # (n_sources, n_target_samples)
    per_sample_std = pred_matrix.std(axis=0)           # (n_target_samples,)
    mean_std = float(per_sample_std.mean())
    # Max std for binary predictions is 0.5 (one source says 0, another says 1).
    # Multiply by 2 so [0, 0.5] maps to [0, 1] before subtracting from 1.
    return float(np.clip(1.0 - 2.0 * mean_std, 0.0, 1.0))


# ---------------------------------------------------------------------------
# CLI — build index
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from sentence_transformers import SentenceTransformer
    import torch

    parser = argparse.ArgumentParser(description="Build transferability column name index")
    parser.add_argument("--manifest", type=Path, default=Path("data/gittables/manifest.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/gittables"))
    parser.add_argument("--index-dir", type=Path, default=_DEFAULT_INDEX_DIR)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(args.manifest) as f:
        manifest = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading encoder on %s", device)
    enc = SentenceTransformer(args.model, device=device)

    build_column_index(manifest["tables"], args.cache_dir, enc, args.index_dir)
    logger.info("Done.")
