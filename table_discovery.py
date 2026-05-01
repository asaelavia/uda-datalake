"""
Step 1 — Table Discovery

Ranks data lake tables by relevance to a target table using column name
embeddings (sentence-transformers) and a bipartite matching score.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

_OLLAMA_BASE_URL = "http://localhost:11434"
_OLLAMA_MODEL = "qwen3.5"

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"

_GENERIC_LABEL_NAMES: frozenset[str] = frozenset({
    "label", "class", "target", "y", "binaryclass",
    "labels", "classes", "targets",
})


def embed_columns(
    columns: list[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """Return an (n_columns, embedding_dim) array for the given column names."""
    return model.encode(columns, show_progress_bar=False, convert_to_numpy=True)


def table_similarity(
    source_cols: list[str],
    target_cols: list[str],
    model: SentenceTransformer,
) -> float:
    """
    Compute similarity between two tables as the normalised cost of the
    optimal bipartite column matching (Hungarian algorithm on cosine distances).

    Returns a score in [0, 1] — higher is more similar.
    """
    src_emb = embed_columns(source_cols, model)
    tgt_emb = embed_columns(target_cols, model)

    # cosine distance matrix: shape (n_src, n_tgt)
    cost = cdist(src_emb, tgt_emb, metric="cosine")

    # match the smaller set to the larger
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_cost = cost[row_ind, col_ind].mean()

    # convert distance → similarity
    return float(1.0 - matched_cost)


def _distribution_similarity(
    source: pd.DataFrame,
    target: pd.DataFrame,
    quantile_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> float:
    """
    1 minus the mean per-column L1 Wasserstein distance (quantile approximation).
    Only numeric columns present in BOTH DataFrames (shared names) are compared.
    Returns a score in [0, 1]; higher = more similar distributions.
    """
    shared_cols = list(
        set(source.select_dtypes("number").columns)
        & set(target.select_dtypes("number").columns)
    )
    if not shared_cols:
        return 0.0

    col_distances = []
    for col in shared_cols:
        src_vals = source[col].dropna()
        tgt_vals = target[col].dropna()
        if len(src_vals) == 0 or len(tgt_vals) == 0:
            continue
        tgt_range = float(tgt_vals.max() - tgt_vals.min())
        qs = np.array(quantile_levels)
        src_q = np.quantile(src_vals, qs)
        tgt_q = np.quantile(tgt_vals, qs)
        col_dist = np.abs(src_q - tgt_q).mean() / (tgt_range + 1e-9)
        col_distances.append(col_dist)

    if not col_distances:
        return 0.0

    mean_distance = float(np.mean(col_distances))
    return max(0.0, 1.0 - mean_distance)


def _distribution_discovery_score(
    source: pd.DataFrame,
    target: pd.DataFrame,
    label_col: Optional[str] = None,
    ks_threshold: float = 0.3,
) -> float:
    """
    Score a source table by how well its numeric columns match target distributions,
    ignoring column names.

    Uses KS statistics with Hungarian assignment to find the optimal column pairing.
    Score = fraction of matched target columns with KS distance < ks_threshold.

    Works pre-alignment (column names do NOT need to match). This is the correct
    distribution score to use in discover_tables, which runs before schema alignment.
    """
    from scipy.stats import ks_2samp

    # Target numeric profiles
    tgt_cols_list: list[str] = []
    tgt_profiles: list[np.ndarray] = []
    for col in target.columns:
        vals = target[col].dropna()
        if len(vals) >= 20 and pd.api.types.is_numeric_dtype(vals):
            tgt_cols_list.append(col)
            tgt_profiles.append(vals.values)

    if not tgt_profiles:
        return 0.0

    # Source numeric columns (excluding label)
    src_numeric: list[np.ndarray] = []
    for col in source.columns:
        if col == label_col:
            continue
        vals = source[col].dropna()
        if len(vals) >= 20 and pd.api.types.is_numeric_dtype(vals):
            src_numeric.append(vals.values)

    if len(src_numeric) < 2:
        return 0.0

    n_src = len(src_numeric)
    n_tgt = len(tgt_profiles)

    # Build KS distance matrix (n_src × n_tgt)
    ks_matrix = np.ones((n_src, n_tgt))
    for i, sv in enumerate(src_numeric):
        for j, tv in enumerate(tgt_profiles):
            stat, _ = ks_2samp(sv, tv)
            ks_matrix[i, j] = stat

    # Hungarian assignment: find optimal column pairing by KS distance
    row_ind, col_ind = linear_sum_assignment(ks_matrix)

    # Score = fraction of TARGET columns that have a good match
    good = sum(1 for r, c in zip(row_ind, col_ind) if ks_matrix[r, c] < ks_threshold)
    return good / n_tgt


def discover_tables(
    lake: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    model_name: str = _DEFAULT_MODEL,
    threshold: float = 0.0,
    model: Optional[SentenceTransformer] = None,
    distribution_weight: float = 0.0,
    label_col: Optional[str] = None,
    target_label_name: Optional[str] = None,
    label_weight: float = 0.0,
    lake_label_names: Optional[dict[str, str]] = None,
    source_pos_rates: Optional[dict[str, float]] = None,
    target_pos_rate: Optional[float] = None,
    balance_weight: float = 0.15,
) -> dict[str, float]:
    """
    Rank data lake tables by similarity to the target table.

    Parameters
    ----------
    lake:
        Mapping of table_id → DataFrame. Only column names (and optionally
        values for distribution scoring) are used.
    target:
        The unlabeled target DataFrame.
    model_name:
        Sentence-transformers model identifier (ignored if `model` is provided).
    threshold:
        Minimum similarity score; tables below this are excluded.
    model:
        Pre-loaded SentenceTransformer instance (avoids reloading across calls).
    distribution_weight:
        Blend weight for distribution similarity (0.0 = schema only, 1.0 = distribution only).
        Default 0.0 preserves backward-compatible behaviour.
    label_col:
        Name of the label column in lake tables; excluded from schema/distribution
        scoring so it is not treated as a feature.
    target_label_name:
        Descriptive name of the target task (e.g. "income above 50k").
        Used for label-name similarity when ``label_weight > 0``.
    label_weight:
        Blend weight for label-name similarity.
        ``score = (1 - label_weight) * schema_dist_score + label_weight * label_sim``
        Default 0.0 preserves backward-compatible behaviour.
    lake_label_names:
        Mapping of table_id → original label attribute name for each lake table.
        When provided with ``label_weight > 0``, these names are embedded and
        compared against ``target_label_name``.  Falls back to ``label_col``
        (same name for all tables) when this dict is absent.
    source_pos_rates:
        Mapping of table_id → positive-class fraction for each lake table.
        When provided together with ``target_pos_rate``, a balance penalty is
        applied: ``score *= 1 - balance_weight * |src_rate - target_rate|``.
    target_pos_rate:
        Estimated positive-class fraction for the target task.  Required for
        the balance penalty; ignored when ``source_pos_rates`` is None.
    balance_weight:
        Strength of the label-balance penalty (default 0.15).

    Returns
    -------
    dict[table_id → similarity_score], sorted descending, filtered by threshold.
    """
    if model is None:
        logger.info("Loading sentence-transformers model: %s", model_name)
        model = SentenceTransformer(model_name)

    target_cols = target.columns.tolist()

    # Pre-compute target label embedding once (if needed)
    target_label_emb: Optional[np.ndarray] = None
    if label_weight > 0.0 and target_label_name:
        target_label_emb = embed_columns([target_label_name], model)[0]

    scores: dict[str, float] = {}

    for table_id, df in lake.items():
        # Exclude label column from feature schema / distribution scoring
        feat_cols = [c for c in df.columns if c != label_col] if label_col else df.columns.tolist()
        df_feat = df[feat_cols] if label_col and label_col in df.columns else df

        schema_score = table_similarity(feat_cols, target_cols, model)
        if distribution_weight > 0.0:
            # Use name-based quantile similarity when column names overlap (adult-style sources);
            # fall back to KS-based distribution discovery when no names match (payroll-style sources).
            shared = set(df_feat.select_dtypes("number").columns) & set(target.select_dtypes("number").columns)
            if shared:
                dist_sim = _distribution_similarity(df_feat, target)
            else:
                dist_sim = _distribution_discovery_score(df_feat, target, label_col=label_col)
            schema_dist_score = (1 - distribution_weight) * schema_score + distribution_weight * dist_sim
        else:
            dist_sim = float("nan")
            schema_dist_score = schema_score

        # Label-name similarity term
        if target_label_emb is not None:
            src_label_name: Optional[str] = None
            if lake_label_names:
                src_label_name = lake_label_names.get(table_id)
            elif label_col:
                src_label_name = label_col

            if src_label_name:
                if src_label_name.lower() in _GENERIC_LABEL_NAMES:
                    label_sim = 0.0   # meaningless generic name — skip label signal
                else:
                    src_label_emb = embed_columns([src_label_name], model)[0]
                    label_sim = float(1.0 - cdist(
                        target_label_emb.reshape(1, -1),
                        src_label_emb.reshape(1, -1),
                        metric="cosine",
                    )[0, 0])
                    label_sim = max(0.0, label_sim)
                score = (1 - label_weight) * schema_dist_score + label_weight * label_sim
            else:
                label_sim = float("nan")
                score = schema_dist_score
        else:
            label_sim = float("nan")
            score = schema_dist_score

        # Label-balance penalty
        if source_pos_rates is not None and target_pos_rate is not None:
            src_rate = source_pos_rates.get(table_id)
            if src_rate is not None:
                penalty = balance_weight * abs(src_rate - target_pos_rate)
                score *= max(0.0, 1.0 - penalty)

        logger.debug(
            "Table '%s' schema=%.4f dist=%.4f label=%.4f final=%.4f",
            table_id, schema_score, dist_sim, label_sim, score,
        )
        if score >= threshold:
            scores[table_id] = score

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


_LLM_EXPANSION_CACHE: dict[str, list[str]] = {}
_LLM_CACHE_FILE = Path(__file__).parent / "data" / "llm_expansion_cache.json"


def _load_llm_cache() -> None:
    """Load persisted LLM expansion cache from disk into memory."""
    if _LLM_CACHE_FILE.exists():
        try:
            with open(_LLM_CACHE_FILE) as f:
                _LLM_EXPANSION_CACHE.update(json.load(f))
            logger.debug("Loaded %d cached LLM expansions", len(_LLM_EXPANSION_CACHE))
        except Exception:
            pass


def _save_llm_cache() -> None:
    """Persist in-memory LLM expansion cache to disk."""
    try:
        _LLM_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_LLM_CACHE_FILE, "w") as f:
            json.dump(_LLM_EXPANSION_CACHE, f, indent=2)
    except Exception as exc:
        logger.debug("Could not save LLM cache: %s", exc)


_load_llm_cache()  # load on import

_EXPANSION_PROMPT = """\
/no_think
You are a data scientist who has analyzed thousands of real CSV files from GitHub.

Target label: "{label_name}"

Task: generate column names that could THEMSELVES serve as the label column — \
synonyms, variants, and upstream proxy measurements that could be binarized to \
approximate this target.

Rules:
  INCLUDE — direct synonyms: for "income above 50k" → salary, wages, annual_income
  INCLUDE — upstream proxies you can threshold: for "diabetes" → glucose, hba1c, bmi
  EXCLUDE — features that merely predict the target: occupation, education, age, race
  (A table with "occupation" cannot be relabeled as income. A table with "glucose" can.)

Step 1 — 2-3 sentences: what synonyms and proxy measurements exist for this concept?
Step 2 — output ONLY a JSON array on the last line, 25-35 entries, covering naming \
variants (CamelCase, snake_case, abbreviations). Example last line format:
["col1", "col2", "col3"]\
"""


def _deduplicate_concepts(
    concepts: list[str],
    model: SentenceTransformer,
    threshold: float = 0.95,
) -> list[str]:
    """
    Remove near-duplicate concepts (cosine similarity > threshold).
    Preserves insertion order; earlier concepts take priority.
    """
    if len(concepts) <= 1:
        return concepts
    embs = embed_columns(concepts, model)
    kept: list[int] = [0]
    for i in range(1, len(concepts)):
        sims = 1.0 - cdist(embs[i : i + 1], embs[kept], metric="cosine")[0]
        if float(sims.max()) < threshold:
            kept.append(i)
    return [concepts[i] for i in kept]


def expand_label_via_llm(
    label_name: str,
    ollama_model: str = _OLLAMA_MODEL,
) -> list[str]:
    """
    Use a local Ollama LLM to generate CSV column names related to label_name,
    including indirect causal chain nodes (e.g. fattening → obesity → diabetes).

    Requires Ollama running locally (http://localhost:11434) with the model
    pulled (e.g. `ollama pull qwen2.5`). No API key or internet needed.

    Results are cached in-process to avoid repeated calls.
    Falls back to [label_name] if Ollama is not running.
    """
    if label_name in _LLM_EXPANSION_CACHE:
        logger.info("LLM expansion cache hit for '%s'", label_name)
        return _LLM_EXPANSION_CACHE[label_name]

    if not _REQUESTS_AVAILABLE:
        logger.debug("requests not installed — skipping LLM expansion")
        return [label_name]

    try:
        resp = _requests.post(
            f"{_OLLAMA_BASE_URL}/api/generate",
            json={
                "model": ollama_model,
                "prompt": _EXPANSION_PROMPT.format(label_name=label_name),
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=600,  # 9.7B on CPU can take several minutes
        )
        resp.raise_for_status()
        text = resp.json()["response"].strip()

        # Extract the JSON array from the response (after the reasoning)
        json_match = re.search(r"\[.*?\]", text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON list found in LLM response")
        columns: list[str] = json.loads(json_match.group())
        if not isinstance(columns, list) or not columns:
            raise ValueError("LLM returned empty or non-list JSON")

        # Always include the original label as the first concept
        concepts = [label_name] + [str(c) for c in columns if str(c) != label_name]
        logger.info(
            "LLM expansion for '%s': %d concepts → %s",
            label_name, len(concepts), concepts[:6],
        )
        _LLM_EXPANSION_CACHE[label_name] = concepts
        _save_llm_cache()
        return concepts

    except Exception as exc:
        logger.warning(
            "LLM expansion failed for '%s': %s — will fall back to KG.",
            label_name, exc,
        )
        return [label_name]


def expand_label_via_kg(
    label_name: str,
    n_results: int = 10,
) -> list[str]:
    """
    Query DBpedia Lookup API to expand a label name into related concepts.

    For "diabetes diagnosis positive" this returns strings like:
    "Diabetes mellitus", "Insulin resistance", "Obesity", "Blood glucose", ...

    These are then embedded alongside the original label so that column names
    like "fattening" (→ obesity → diabetes risk) can be found via max similarity.

    Degrades gracefully to [label_name] if the API is unavailable.
    """
    if not _REQUESTS_AVAILABLE:
        return [label_name]

    # Strip noise/comparison words to extract the core concept noun(s).
    # "income above 50k" → "income", "diabetes diagnosis positive" → "diabetes"
    _NOISE_WORDS = frozenset({
        "above", "below", "over", "under", "positive", "negative",
        "good", "bad", "high", "low", "binary", "or", "and", "the",
        "a", "an", "for", "with", "in", "of", "at", "to", "from",
        "into", "plus", "minus", "diagnosis", "prediction", "classification",
        "subscription", "detection",
    })
    tokens = [
        t for t in label_name.lower().split()
        if t not in _NOISE_WORDS and not t.isdigit() and not t.startswith("$")
    ]
    # Use first 2 meaningful tokens as the core DBpedia query
    query = " ".join(tokens[:2]) if tokens else label_name
    url = "https://lookup.dbpedia.org/api/search"
    params = {"query": query, "maxResults": n_results, "format": "json"}

    try:
        resp = _requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("DBpedia Lookup failed for '%s': %s — using label only.", label_name, exc)
        return [label_name]

    def _strip_html(text: str) -> str:
        return re.sub(r"<[^>]+>", "", text).strip()

    # DBpedia Lookup type URIs for geographic/administrative entities to skip
    _GEO_TYPES = frozenset({
        "http://dbpedia.org/ontology/Place",
        "http://dbpedia.org/ontology/Settlement",
        "http://dbpedia.org/ontology/PopulatedPlace",
        "http://dbpedia.org/ontology/AdministrativeRegion",
        "http://dbpedia.org/ontology/Country",
        "http://schema.org/Place",
    })

    concepts: list[str] = [label_name]  # always include original
    for doc in data.get("docs", []):
        # Skip geographic entities — they match column names like "state", "city"
        # and introduce false positives in column repurposing
        type_uris = {t for t in doc.get("type", []) if isinstance(t, str)}
        if type_uris & _GEO_TYPES:
            continue

        label = _strip_html(doc.get("label", [""])[0])
        comment = _strip_html(doc.get("comment", [""])[0])
        if label and label not in concepts:
            concepts.append(label)
        # Include a short description snippet (first sentence) for richer embedding
        if comment:
            first_sent = comment.split(".")[0].strip()
            if first_sent and first_sent not in concepts:
                concepts.append(first_sent)

    logger.info("KG expansion for '%s': %d concepts → %s",
                label_name, len(concepts), concepts[:5])
    return concepts


def find_repurposable_features(
    lake: dict[str, pd.DataFrame],
    target_label_name: str,
    model: SentenceTransformer,
    label_col: str = "label",
    threshold: float = 0.6,
    use_llm_expansion: bool = True,
    use_kg_expansion: bool = True,
    kg_n_concepts: int = 10,
) -> dict[str, str]:
    """
    For each lake table, embed all feature column names and compare to
    target_label_name. If the most-similar feature column exceeds threshold,
    return it as a repurpose candidate.

    Expansion strategy (in priority order):
    1. LLM (claude-haiku): generates CSV-style column names via causal chain
       reasoning — captures multi-hop paths like fattening → obesity → diabetes.
    2. DBpedia KG fallback: used when LLM is unavailable (no API key / offline).
    3. Label-only: plain embedding of the target label string.

    Near-duplicate concepts are removed before embedding to avoid inflating
    max-similarity scores with redundant variants.

    Parameters
    ----------
    lake : feature-only DataFrames (label column already removed by caller)
    target_label_name : e.g. "income above 50k"
    model : pre-loaded SentenceTransformer
    label_col : safety filter — excluded if still present
    threshold : minimum cosine similarity to qualify (default 0.6)
    use_llm_expansion : try LLM expansion first (default True)
    use_kg_expansion : fall back to DBpedia KG if LLM unavailable (default True)
    kg_n_concepts : max concepts to fetch from DBpedia Lookup

    Returns
    -------
    dict[table_id → feature_col_name]  (only tables with a qualifying feature)
    """
    # --- Concept expansion (LLM → KG → label-only) ---
    concepts: list[str]
    if use_llm_expansion:
        concepts = expand_label_via_llm(target_label_name)
        if len(concepts) <= 1 and use_kg_expansion:
            logger.info("LLM expansion returned no results; falling back to KG.")
            concepts = expand_label_via_kg(target_label_name, n_results=kg_n_concepts)
    elif use_kg_expansion:
        concepts = expand_label_via_kg(target_label_name, n_results=kg_n_concepts)
    else:
        concepts = [target_label_name]

    # Remove near-duplicates before embedding (cosine > 0.95 → redundant)
    if len(concepts) > 1:
        concepts = _deduplicate_concepts(concepts, model)
        logger.info("Concepts after deduplication: %d", len(concepts))

    # Embed all concepts: shape (n_concepts, embed_dim)
    target_embs = np.vstack([embed_columns([c], model)[0].reshape(1, -1) for c in concepts])

    # --- Checkpoint setup ---
    # Sort lake keys for a stable, resumable iteration order.
    # Checkpoint is keyed by label + n_concepts so a change in either invalidates it.
    ckpt_key = f"{target_label_name}__n{len(concepts)}"
    ckpt_slug = re.sub(r"[^a-z0-9_]", "_", ckpt_key.lower())
    ckpt_path = Path(__file__).parent / "data" / f"repurpose_ckpt_{ckpt_slug}.json"

    sorted_ids = sorted(lake.keys())
    result: dict[str, str] = {}
    resume_from = 0

    if ckpt_path.exists():
        try:
            with open(ckpt_path) as f:
                ckpt = json.load(f)
            # Validate the checkpoint matches current run parameters
            if ckpt.get("label") == target_label_name and ckpt.get("n_concepts") == len(concepts):
                resume_from = int(ckpt.get("progress_idx", 0))
                result = ckpt.get("result", {})
                logger.info(
                    "Resuming repurposing scan from %d/%d (found %d candidates so far)",
                    resume_from, len(sorted_ids), len(result),
                )
            else:
                logger.info("Checkpoint parameters changed — starting fresh scan.")
                ckpt_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Could not load checkpoint: %s — starting fresh.", exc)

    n_total = len(sorted_ids)
    _log_every = max(1, n_total // 20)   # ~20 progress updates
    _ckpt_every = max(1, n_total // 20)  # checkpoint at same interval

    for _i, table_id in enumerate(sorted_ids):
        if _i < resume_from:
            continue

        if _i % _log_every == 0:
            logger.info(
                "  Repurposing scan: %d/%d tables (%.0f%%) — %d candidates so far",
                _i, n_total, 100 * _i / n_total, len(result),
            )

        if _i > 0 and _i % _ckpt_every == 0:
            try:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                with open(ckpt_path, "w") as f:
                    json.dump({
                        "label": target_label_name,
                        "n_concepts": len(concepts),
                        "progress_idx": _i,
                        "result": result,
                    }, f)
                logger.debug("Checkpoint saved at %d/%d", _i, n_total)
            except Exception as exc:
                logger.warning("Could not save checkpoint: %s", exc)

        df = lake[table_id]
        feat_cols = [c for c in df.columns if c != label_col]
        if not feat_cols:
            continue
        col_embs = embed_columns(feat_cols, model)
        sim_matrix = 1.0 - cdist(target_embs, col_embs, metric="cosine")
        best_sims = sim_matrix.max(axis=0)
        best_idx = int(np.argmax(best_sims))
        best_sim = float(best_sims[best_idx])
        if best_sim >= threshold:
            result[table_id] = feat_cols[best_idx]
            logger.debug("Repurpose candidate '%s' → feature '%s' (sim=%.4f)",
                         table_id, feat_cols[best_idx], best_sim)

    # Clean up checkpoint on successful completion
    ckpt_path.unlink(missing_ok=True)
    return result
