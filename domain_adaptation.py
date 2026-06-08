"""
Step 3 — Weighted Multi-Source Domain Adaptation

Adaptation strategies, in increasing sophistication:

  Baseline A   — majority-class predictor (no-information floor)
  Baseline B   — random lake tables, position-aligned (no repurposing)
  Baseline C   — naive equal-weight combination of all repurposed sources
  Level 0      — direct transfer from the single best source
  Level 2      — iterative self-training (3–5 rounds of pseudo-labeling)
  KMM          — Kernel Mean Matching instance reweighting
  Level 5      — DANN: adversarial domain adaptation (requires PyTorch)
  Level 6      — FTTA: fully test-time adaptation (AAAI 2025, requires PyTorch)
                 CDO (label dist. correction) + LCW (k-NN weighting) + DME (lr ensemble)
  Ensemble     — confidence-weighted average of Level 2 + Level 5
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize as sp_minimize
from scipy.spatial.distance import cdist
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


@dataclass
class AdaptationResult:
    level: str
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    model: Optional[object] = field(default=None, repr=False)
    eval_mask: Optional[np.ndarray] = None  # boolean mask — only these rows used for metrics


def _make_xgb(**kwargs) -> XGBClassifier:
    defaults = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


def _split_xy(
    df: pd.DataFrame,
    label_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y


def _pool_sources(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    label_col: str,
    equal_weight: bool = False,
    weight_power: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Concatenate aligned source tables and compute per-sample weights.

    Parameters
    ----------
    weight_power:
        Scores are raised to this power before normalisation.  Values > 1
        increase contrast between high- and low-scoring sources.

    Returns
    -------
    X, y, sample_weights — all aligned in row order.
    """
    frames_X, frames_y, weights = [], [], []
    if equal_weight:
        raw = {k: 1.0 for k in aligned}
    else:
        raw = {k: discovery_scores[k] ** weight_power for k in aligned}
    total_score = sum(raw.values()) or 1.0

    for table_id, df in aligned.items():
        X, y = _split_xy(df, label_col)
        score = raw[table_id] / total_score
        frames_X.append(X)
        frames_y.append(y)
        weights.append(np.full(len(X), score))

    X_all = pd.concat(frames_X, ignore_index=True)
    y_all = pd.concat(frames_y, ignore_index=True)
    w_all = np.concatenate(weights)
    return X_all, y_all, w_all


def _confident_denoising(
    X: pd.DataFrame,
    y: pd.Series,
    w: np.ndarray,
    model: XGBClassifier,
    noise_penalty: float = 0.1,
) -> np.ndarray:
    """
    Down-weight source samples where the model disagrees with their pseudo-label.

    Implements the core idea of confident learning: for each class c, compute the
    mean model confidence P(y=c | x) across samples labelled c.  Samples whose
    confidence falls below that class-specific threshold are likely noisy labels
    and receive weight *= noise_penalty.

    Returns a copy of w with adjusted weights.
    """
    proba = model.predict_proba(X)
    y_int = y.values.astype(int)
    p_label = proba[np.arange(len(y_int)), y_int]

    thresh: dict[int, float] = {}
    for c in [0, 1]:
        mask = y_int == c
        thresh[c] = float(p_label[mask].mean()) if mask.any() else 0.5

    below = np.array([p_label[i] < thresh[y_int[i]] for i in range(len(y_int))])
    w_new = w.copy()
    w_new[below] *= noise_penalty

    for c in [0, 1]:
        if w_new[y_int == c].sum() < 1e-6:
            logger.warning(
                "_confident_denoising: class %d has near-zero total weight after denoising", c
            )
    return w_new


def _domain_classifier_features(
    X_src: pd.DataFrame,
    target: pd.DataFrame,
    min_coverage: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select columns with sufficient non-NaN coverage in both source and target,
    then impute with per-column source medians.

    Used for KMM and DANN — prevents these methods from learning NaN sparsity
    patterns (artefacts of schema alignment) rather than true feature similarity.
    """
    src_cov = X_src.notna().mean()
    tgt_cov = target.notna().mean()
    valid = [c for c in X_src.columns
             if src_cov[c] >= min_coverage and tgt_cov[c] >= min_coverage]
    if len(valid) < 2:
        valid = X_src.columns.tolist()

    X_dc = X_src[valid].copy()
    T_dc = target[valid].copy()
    for col in valid:
        med = float(X_dc[col].median())
        if np.isnan(med):
            med = 0.0
        X_dc[col] = X_dc[col].fillna(med)
        T_dc[col] = T_dc[col].fillna(med)
    return X_dc, T_dc


def _pool_unlabeled(
    unlabeled_features: dict[str, pd.DataFrame],
    domain_cols: list[str],
) -> Optional[pd.DataFrame]:
    """
    Concatenate unlabeled feature tables, keeping only `domain_cols`.
    Tables without any overlap are skipped.  Returns None if nothing survives.
    """
    frames = []
    for df in unlabeled_features.values():
        shared = [c for c in domain_cols if c in df.columns]
        if len(shared) < 2:
            continue
        sub = df[shared].copy()
        for c in domain_cols:
            if c not in sub.columns:
                sub[c] = 0.0
        frames.append(sub[domain_cols])
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def run_baseline(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    target: pd.DataFrame,
    label_col: str,
    **xgb_kwargs,
) -> AdaptationResult:
    """
    Baseline C: naive equal-weight combination of all repurposed sources.
    Demonstrates negative transfer when schemas/distributions vary.
    """
    logger.info("[Baseline C] Equal-weight multi-source training")
    X_all, y_all, w_all = _pool_sources(aligned, discovery_scores, label_col, equal_weight=True)
    model = _make_xgb(**xgb_kwargs)
    model.fit(X_all, y_all, sample_weight=w_all)
    proba = model.predict_proba(target)
    return AdaptationResult(
        level="baseline",
        predictions=model.predict(target),
        probabilities=proba,
        model=model,
    )


def run_baseline_majority(
    target: pd.DataFrame,
) -> AdaptationResult:
    """
    Baseline A: predict the majority class (all zeros) for all samples.
    AUC = 0.5 by definition (constant probability scores).
    Provides the no-information floor.
    """
    n = len(target)
    proba = np.full((n, 2), 0.5)
    return AdaptationResult(
        level="baseline_a",
        predictions=np.zeros(n, dtype=int),
        probabilities=proba,
        model=None,
    )


def run_baseline_random(
    raw_sources: list[pd.DataFrame],
    target: pd.DataFrame,
    label_col: str,
    n_tables: int = 20,
    n_runs: int = 5,
    random_state: int = 42,
    **xgb_kwargs,
) -> AdaptationResult:
    """
    Baseline B: random source tables without discovery scoring, position-aligned.

    Isolates the value of source repurposing + schema alignment.  Uses the
    repurposed labeled pool but ignores repurposing quality: randomly samples
    tables and aligns columns by position (first N numeric columns mapped to
    target columns), instead of using discovery scoring + Hungarian matching.

    Averaged over n_runs with different random seeds.
    """
    rng = np.random.default_rng(random_state)
    target_cols = list(target.columns)
    n_target_cols = len(target_cols)

    all_proba: list[np.ndarray] = []
    for run in range(n_runs):
        selected = rng.choice(
            len(raw_sources),
            size=min(n_tables, len(raw_sources)),
            replace=False,
        )
        frames_X: list[pd.DataFrame] = []
        frames_y: list[pd.Series] = []

        for idx in selected:
            df = raw_sources[int(idx)]
            if label_col not in df.columns:
                continue
            X_raw, y_raw = _split_xy(df, label_col)
            num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                continue
            aligned_cols = num_cols[:n_target_cols]
            X_pos = X_raw[aligned_cols].copy()
            X_pos.columns = target_cols[: len(aligned_cols)]
            for c in target_cols:
                if c not in X_pos.columns:
                    X_pos[c] = 0.0
            frames_X.append(X_pos[target_cols])
            frames_y.append(y_raw)

        if not frames_X:
            continue

        X_all = pd.concat(frames_X, ignore_index=True)
        # Replace inf, then clip to float32 range — XGBoost converts internally
        # to float32 and raises on values that overflow even if not IEEE inf.
        _f32 = np.finfo(np.float32)
        num_cols_ = X_all.select_dtypes(include="number").columns
        X_all[num_cols_] = (
            X_all[num_cols_]
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=_f32.min, upper=_f32.max)
        )
        y_all = pd.concat(frames_y, ignore_index=True)
        model = _make_xgb(**xgb_kwargs)
        model.fit(X_all, y_all)
        all_proba.append(model.predict_proba(target))
        logger.info(
            "[Baseline B] Run %d: trained on %d rows from %d tables",
            run + 1, len(X_all), len(frames_X),
        )

    if not all_proba:
        n = len(target)
        return AdaptationResult(
            level="baseline_b",
            predictions=np.zeros(n, dtype=int),
            probabilities=np.full((n, 2), 0.5),
            model=None,
        )

    mean_proba = np.mean(all_proba, axis=0)
    return AdaptationResult(
        level="baseline_b",
        predictions=mean_proba.argmax(axis=1),
        probabilities=mean_proba,
        model=None,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_level0(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    target: pd.DataFrame,
    label_col: str,
    **xgb_kwargs,
) -> AdaptationResult:
    """
    Direct transfer from the single highest-scoring source table.
    """
    best_id = max(aligned, key=lambda k: discovery_scores.get(k, 0.0))
    logger.info("[Level 0] Direct transfer from '%s' (score=%.4f)", best_id, discovery_scores.get(best_id, 0.0))
    X, y = _split_xy(aligned[best_id], label_col)
    model = _make_xgb(**xgb_kwargs)
    model.fit(X, y)
    proba = model.predict_proba(target)
    return AdaptationResult(
        level="level0",
        predictions=model.predict(target),
        probabilities=proba,
        model=model,
    )


def run_level2(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    target: pd.DataFrame,
    label_col: str,
    n_rounds: int = 5,
    pseudo_weight: float = 0.5,
    weight_power: float = 2.0,
    denoise: bool = True,
    **xgb_kwargs,
) -> AdaptationResult:
    """
    Iterative self-training starting from the best single source (L0).

    Each round:
      1. Re-predict on ALL target samples with the current model
      2. Select the top-pct% most confident (highest max(p, 1-p))
      3. Add those as pseudo-labeled rows; retrain on source + pseudo-labels
      4. Stop early if AUC on a pseudo-labeled holdout degrades

    Round percentile schedule (fraction of target kept as pseudo-labels):
      Round 1 → top 10%, Round 2 → top 20%, Round 3 → top 35%,
      Round 4 → top 50%, Round 5 → top 70%

    Source data is never dropped — it acts as a permanent anchor.
    """
    round_pcts = [0.10, 0.20, 0.35, 0.50, 0.70][:n_rounds]
    logger.info("[Level 2] Iterative self-training (%d rounds)", n_rounds)

    # Start from L0 (best single source)
    best_id = max(aligned, key=lambda k: discovery_scores.get(k, 0.0))
    X_best, y_best = _split_xy(aligned[best_id], label_col)
    model = _make_xgb(**xgb_kwargs)
    model.fit(X_best, y_best)
    logger.info("[Level 2] Initial model from '%s'", best_id)

    # Minimum-target gate: when n_target < 100, round-1 n_per_class falls to
    # the floor of 5 (giving only 10 pseudo-labels), validation holdout is
    # < 10 (early stopping never fires), and all 5 unchecked rounds accumulate
    # noise from a potentially mis-calibrated source model.
    if len(target) < 100:
        logger.info(
            "[Level 2] Target too small for pseudo-labeling (n=%d < 100); "
            "skipping pseudo-labeling, returning L0",
            len(target),
        )
        proba_l0 = model.predict_proba(target)
        return AdaptationResult(
            level="level2",
            predictions=model.predict(target),
            probabilities=proba_l0,
            model=model,
        )

    # Source pool — permanent anchor in every retrain
    X_src, y_src, w_src = _pool_sources(
        aligned, discovery_scores, label_col, weight_power=weight_power
    )

    # Confident denoising: down-weight source samples where model disagrees with pseudo-label
    if denoise:
        w_src_orig = w_src.copy()
        w_src = _confident_denoising(X_src, y_src, w_src, model)
        n_downweighted = int((w_src < w_src_orig).sum())
        logger.info("[Level 2] Denoised source weights; down-weighted %d/%d samples",
                    n_downweighted, len(w_src))

    best_model = model
    best_proba = model.predict_proba(target)
    best_pseudo_auc: Optional[float] = None

    for round_idx, pct in enumerate(round_pcts):
        # Re-predict on ALL target samples each round
        proba_all = model.predict_proba(target)
        confidence = proba_all.max(axis=1)
        pred_cls   = proba_all.argmax(axis=1)

        # Class-balanced selection: take top (pct/2 × n) from each predicted class
        # so both classes are always represented even when the source model is
        # miscalibrated relative to the target positive rate.
        n_per_class = max(5, int(len(target) * pct / 2))
        pseudo_mask = np.zeros(len(target), dtype=bool)
        for c in range(2):
            cls_idx = np.where(pred_cls == c)[0]
            if len(cls_idx) == 0:
                continue
            top_idx = cls_idx[np.argsort(confidence[cls_idx])[-n_per_class:]]
            pseudo_mask[top_idx] = True
        n_pseudo = int(pseudo_mask.sum())

        logger.info(
            "[Level 2] Round %d: %d pseudo-labels (top %.0f%% per class, balanced)",
            round_idx + 1, n_pseudo, pct * 100,
        )
        if n_pseudo < 10:
            logger.info("[Level 2] Too few pseudo-labels — stopping early at round %d", round_idx + 1)
            break

        # 80/20 train/holdout split on pseudo-labeled set for early stopping
        pseudo_idx = np.where(pseudo_mask)[0]
        rng = np.random.default_rng(xgb_kwargs.get("random_state", 42) + round_idx)
        val_size = max(5, len(pseudo_idx) // 5)
        val_local = rng.choice(len(pseudo_idx), size=val_size, replace=False)
        val_idx = pseudo_idx[val_local]
        train_idx = np.setdiff1d(pseudo_idx, val_idx)

        train_pseudo_mask = np.zeros(len(target), dtype=bool)
        train_pseudo_mask[train_idx] = True

        pseudo_X_train = target[train_pseudo_mask].copy()
        pseudo_y_train = pd.Series(
            model.classes_[proba_all[train_pseudo_mask].argmax(axis=1)],
            name=label_col,
        )
        pseudo_w_train = np.full(len(pseudo_X_train), pseudo_weight)

        X_aug = pd.concat([X_src, pseudo_X_train], ignore_index=True)
        y_aug = pd.concat([y_src, pseudo_y_train], ignore_index=True)
        w_aug = np.concatenate([w_src, pseudo_w_train])

        new_model = _make_xgb(**xgb_kwargs)
        new_model.fit(X_aug, y_aug, sample_weight=w_aug)

        # Check for degenerate pseudo-labels (all one class) before early stopping.
        # Do NOT update model when skipping — propagating a model trained on
        # all-class-1 pseudo-labels makes subsequent rounds increasingly degenerate.
        pseudo_classes_used = np.unique(model.classes_[proba_all[pseudo_mask].argmax(axis=1)])
        if len(pseudo_classes_used) < 2:
            logger.info(
                "[Level 2] Round %d: all pseudo-labels are class %s — skipping round",
                round_idx + 1, pseudo_classes_used[0],
            )
            continue

        # Early stopping: AUC on pseudo holdout (pseudo-labels as ground truth)
        if len(val_idx) >= 10:
            val_pseudo_bin = (
                model.classes_[proba_all[val_idx].argmax(axis=1)] == model.classes_[1]
            ).astype(int)
            new_val_scores = new_model.predict_proba(target.iloc[val_idx])[:, 1]
            try:
                round_auc = float(roc_auc_score(val_pseudo_bin, new_val_scores))
            except ValueError:
                round_auc = float("nan")
            if np.isnan(round_auc):
                logger.info("[Level 2] Round %d pseudo-holdout AUC: undefined (single class) — accepting model", round_idx + 1)
                best_model = new_model
                best_proba = new_model.predict_proba(target)
            else:
                logger.info("[Level 2] Round %d pseudo-holdout AUC: %.4f", round_idx + 1, round_auc)
                if best_pseudo_auc is not None and round_auc < best_pseudo_auc - 0.01:
                    logger.info(
                        "[Level 2] Early stop: AUC degraded (%.4f < %.4f)",
                        round_auc, best_pseudo_auc,
                    )
                    break
                if best_pseudo_auc is None or round_auc >= best_pseudo_auc:
                    best_pseudo_auc = round_auc
                    best_model = new_model
                    best_proba = new_model.predict_proba(target)
        else:
            best_model = new_model
            best_proba = new_model.predict_proba(target)

        model = new_model

    return AdaptationResult(
        level="level2",
        predictions=best_model.predict(target),
        probabilities=best_proba,
        model=best_model,
    )


def run_level2_lsc(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    target: pd.DataFrame,
    label_col: str,
    n_rounds: int = 5,
    pseudo_weight: float = 0.5,
    weight_power: float = 2.0,
    denoise: bool = True,
    lsc_clip: float = 5.0,
    **xgb_kwargs,
) -> AdaptationResult:
    """
    L2 + Label Shift Correction (AdaMatch-style).

    Extends run_level2 with per-round importance reweighting of source samples
    to correct for the mismatch between source proxy-label prevalence and the
    estimated target positive rate.  At each round:

      w_lsc(y=1) = p_t_est / p_s      (upweight source positives when target has more)
      w_lsc(y=0) = (1-p_t_est)/(1-p_s)

    p_t_est is estimated from the current model's mean predicted P(y=1) on the
    full unlabeled target.  Weights are clipped to [1/lsc_clip, lsc_clip] and
    multiplied onto the existing discovery-score weights.

    This directly addresses label shift endemic to lake-based source repurposing:
    proxy columns (e.g. "Heart Failure" at 18%) often have different prevalence
    than the true target label (e.g. heart disease at 44%).
    """
    round_pcts = [0.10, 0.20, 0.35, 0.50, 0.70][:n_rounds]
    logger.info("[Level 2 LSC] Iterative self-training + label shift correction (%d rounds)", n_rounds)

    best_id = max(aligned, key=lambda k: discovery_scores.get(k, 0.0))
    X_best, y_best = _split_xy(aligned[best_id], label_col)
    model = _make_xgb(**xgb_kwargs)
    model.fit(X_best, y_best)
    logger.info("[Level 2 LSC] Initial model from '%s'", best_id)

    # Minimum-target gate (same as run_level2)
    if len(target) < 100:
        logger.info(
            "[Level 2 LSC] Target too small for pseudo-labeling (n=%d < 100); "
            "skipping pseudo-labeling, returning L0",
            len(target),
        )
        proba_l0 = model.predict_proba(target)
        return AdaptationResult(
            level="level2_lsc",
            predictions=model.predict(target),
            probabilities=proba_l0,
            model=model,
        )

    X_src, y_src, w_src_base = _pool_sources(
        aligned, discovery_scores, label_col, weight_power=weight_power
    )

    if denoise:
        w_src_base = _confident_denoising(X_src, y_src, w_src_base, model)

    classes_ = model.classes_

    def _lsc_weights(proba_tgt: np.ndarray) -> np.ndarray:
        """Compute per-source-row importance weights from current target predictions."""
        p_t = float(np.clip(proba_tgt[:, 1].mean(), 0.05, 0.95))
        p_s = float(np.clip((y_src == classes_[1]).mean(), 0.05, 0.95))
        lsc = np.where(
            y_src == classes_[1],
            p_t / p_s,
            (1.0 - p_t) / (1.0 - p_s),
        )
        lsc = np.clip(lsc, 1.0 / lsc_clip, lsc_clip)
        w = w_src_base * lsc
        w = w / w.mean()  # normalise so total weight magnitude is preserved
        logger.info(
            "[Level 2 LSC] p_s=%.3f  p_t_est=%.3f  lsc_pos=%.2f  lsc_neg=%.2f",
            p_s, p_t, p_t / p_s, (1.0 - p_t) / (1.0 - p_s),
        )
        return w

    best_model = model
    best_proba = model.predict_proba(target)
    best_pseudo_auc: Optional[float] = None

    # Apply LSC from the very first round using the initial model
    w_src = _lsc_weights(best_proba)

    for round_idx, pct in enumerate(round_pcts):
        proba_all = model.predict_proba(target)
        confidence = proba_all.max(axis=1)
        pred_cls   = proba_all.argmax(axis=1)

        n_per_class = max(5, int(len(target) * pct / 2))
        pseudo_mask = np.zeros(len(target), dtype=bool)
        for c in range(2):
            cls_idx = np.where(pred_cls == c)[0]
            if len(cls_idx) == 0:
                continue
            top_idx = cls_idx[np.argsort(confidence[cls_idx])[-n_per_class:]]
            pseudo_mask[top_idx] = True
        n_pseudo = int(pseudo_mask.sum())

        logger.info(
            "[Level 2 LSC] Round %d: %d pseudo-labels (top %.0f%% per class)",
            round_idx + 1, n_pseudo, pct * 100,
        )
        if n_pseudo < 10:
            break

        pseudo_idx = np.where(pseudo_mask)[0]
        rng = np.random.default_rng(xgb_kwargs.get("random_state", 42) + round_idx)
        val_size = max(5, len(pseudo_idx) // 5)
        val_local = rng.choice(len(pseudo_idx), size=val_size, replace=False)
        val_idx   = pseudo_idx[val_local]
        train_idx = np.setdiff1d(pseudo_idx, val_idx)

        train_pseudo_mask = np.zeros(len(target), dtype=bool)
        train_pseudo_mask[train_idx] = True

        pseudo_X_train = target[train_pseudo_mask].copy()
        pseudo_y_train = pd.Series(
            model.classes_[proba_all[train_pseudo_mask].argmax(axis=1)],
            name=label_col,
        )
        pseudo_w_train = np.full(len(pseudo_X_train), pseudo_weight)

        X_aug = pd.concat([X_src, pseudo_X_train], ignore_index=True)
        y_aug = pd.concat([y_src, pseudo_y_train], ignore_index=True)
        w_aug = np.concatenate([w_src, pseudo_w_train])

        new_model = _make_xgb(**xgb_kwargs)
        new_model.fit(X_aug, y_aug, sample_weight=w_aug)

        pseudo_classes_used = np.unique(model.classes_[proba_all[pseudo_mask].argmax(axis=1)])
        if len(pseudo_classes_used) < 2:
            logger.info("[Level 2 LSC] Round %d: all pseudo-labels single class — skipping", round_idx + 1)
            continue

        # Update LSC weights for next round using the new model's target predictions
        new_proba_tgt = new_model.predict_proba(target)
        w_src = _lsc_weights(new_proba_tgt)

        if len(val_idx) >= 3:
            val_pseudo_bin = (
                model.classes_[proba_all[val_idx].argmax(axis=1)] == model.classes_[1]
            ).astype(int)
            new_val_scores = new_model.predict_proba(target.iloc[val_idx])[:, 1]
            try:
                round_auc = float(roc_auc_score(val_pseudo_bin, new_val_scores))
            except ValueError:
                round_auc = float("nan")
            if np.isnan(round_auc):
                best_model = new_model
                best_proba = new_proba_tgt
            else:
                logger.info("[Level 2 LSC] Round %d pseudo-holdout AUC: %.4f", round_idx + 1, round_auc)
                if round_auc >= 1.0 - 1e-6:
                    logger.info("[Level 2 LSC] Round %d: pseudo-holdout AUC=1.0 (memorisation) — stopping", round_idx + 1)
                    if best_pseudo_auc is None:
                        best_model = new_model
                        best_proba = new_proba_tgt
                    break
                if best_pseudo_auc is not None and round_auc < best_pseudo_auc - 0.01:
                    logger.info("[Level 2 LSC] Early stop: AUC degraded (%.4f < %.4f)", round_auc, best_pseudo_auc)
                    break
                if best_pseudo_auc is None or round_auc >= best_pseudo_auc:
                    best_pseudo_auc = round_auc
                    best_model = new_model
                    best_proba = new_proba_tgt
        else:
            best_model = new_model
            best_proba = new_proba_tgt

        model = new_model

    return AdaptationResult(
        level="level2_lsc",
        predictions=best_model.predict(target),
        probabilities=best_proba,
        model=best_model,
    )


def run_kmm(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    target: pd.DataFrame,
    label_col: str,
    n_src_max: int = 5000,
    B: Optional[float] = None,
    eps: Optional[float] = None,
    **xgb_kwargs,
) -> AdaptationResult:
    """
    Kernel Mean Matching (KMM) — instance reweighting via RBF kernel.

    Directly minimises the Maximum Mean Discrepancy between the weighted source
    and the target distribution in kernel space, yielding per-sample weights β
    that make the source distribution resemble the target.  Trains XGBoost
    with sample_weight = β.

    Parameters
    ----------
    n_src_max:
        Subsample source to at most this many rows for the O(n²) kernel
        computation.
    B:
        Upper bound on weights.  Default: 1000 / sqrt(n_source).
    eps:
        Constraint slack: |Σβ − n_s| ≤ n_s · ε.
        Default: (sqrt(n_s) − 1) / sqrt(n_s).
    """
    logger.info("[KMM] Kernel Mean Matching instance reweighting")

    X_src, y_src, _ = _pool_sources(aligned, discovery_scores, label_col)
    X_src_dc, target_dc = _domain_classifier_features(X_src, target)

    n_s = len(X_src_dc)
    n_t = len(target_dc)

    # Optionally subsample source for the kernel computation (O(n²) memory)
    rng = np.random.default_rng(42)
    src_idx = np.arange(n_s)
    if n_s > n_src_max:
        src_idx = rng.choice(n_s, size=n_src_max, replace=False)
        logger.info("[KMM] Subsampled source for kernel: %d → %d", n_s, n_src_max)
    n_sub = len(src_idx)

    # Normalise features before kernel computation
    scaler = StandardScaler()
    X_s_scaled = scaler.fit_transform(X_src_dc.iloc[src_idx].values)
    X_t_scaled = scaler.transform(target_dc.values)

    # Bandwidth σ = median pairwise distance (on a capped subset for speed)
    n_bw = min(500, n_sub)
    pairwise = cdist(X_s_scaled[:n_bw], X_s_scaled[:n_bw])
    sigma = float(np.median(pairwise[pairwise > 0]))
    sigma = max(sigma, 1e-9)
    gamma = 1.0 / (2.0 * sigma ** 2)

    # RBF kernel matrices
    K_ss = np.exp(-gamma * cdist(X_s_scaled, X_s_scaled, metric="sqeuclidean"))
    K_st = np.exp(-gamma * cdist(X_s_scaled, X_t_scaled, metric="sqeuclidean"))
    kappa = (float(n_sub) / float(n_t)) * K_st.sum(axis=1)

    if B is None:
        B = 1000.0 / np.sqrt(float(n_sub))
    if eps is None:
        eps = (np.sqrt(float(n_sub)) - 1.0) / np.sqrt(float(n_sub))

    def _obj(beta: np.ndarray) -> float:
        return 0.5 * float(beta @ K_ss @ beta) - float(kappa @ beta)

    def _grad(beta: np.ndarray) -> np.ndarray:
        return K_ss @ beta - kappa

    constraints = [
        {"type": "ineq", "fun": lambda b: float(n_sub) * (1.0 + eps) - b.sum()},
        {"type": "ineq", "fun": lambda b: b.sum() - float(n_sub) * (1.0 - eps)},
    ]
    bounds = [(0.0, float(B))] * n_sub

    result = sp_minimize(
        _obj,
        np.ones(n_sub),
        jac=_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-6},
    )
    beta = np.clip(result.x, 0.0, float(B))
    logger.info(
        "[KMM] beta: min=%.3f  max=%.3f  mean=%.3f  (solver: %s)",
        beta.min(), beta.max(), beta.mean(), result.message,
    )

    # Map subsampled weights back to full source (unseen rows keep weight 1)
    sample_weight = np.ones(n_s)
    sample_weight[src_idx] = beta

    model = _make_xgb(**xgb_kwargs)
    model.fit(X_src, y_src, sample_weight=sample_weight)
    proba = model.predict_proba(target)
    return AdaptationResult(
        level="kmm",
        predictions=model.predict(target),
        probabilities=proba,
        model=model,
    )


# ---------------------------------------------------------------------------
# Level 5 — DANN helpers
# ---------------------------------------------------------------------------

# Generalized Cross Entropy for noisy-label robustness (Zhang & Sabuncu 2018).
# q=0 → CE; q=1 → MAE; q=0.7 is the standard noise-robust tradeoff.
_GCE_Q: float = 0.7


def _gce_loss(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    weights: Optional["torch.Tensor"] = None,
    q: float = _GCE_Q,
) -> "torch.Tensor":
    """Generalised Cross Entropy loss.  Drop-in replacement for weighted CE.

    L_GCE(f(x), y) = (1 - f_y(x)^q) / q
    Weights are applied per-sample before averaging.
    """
    import torch
    probs = torch.softmax(logits, dim=1)
    p_y = probs.gather(1, labels.unsqueeze(1)).squeeze(1)   # prob of true class
    loss = (1.0 - p_y.clamp(min=1e-7) ** q) / q
    if weights is not None:
        loss = loss * weights
    return loss.mean()

class _GRL(object):
    """Gradient Reversal Layer implemented as a torch.autograd.Function."""

    @staticmethod
    def _get_fn():
        import torch

        class _GRLFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, lambda_):  # type: ignore[override]
                ctx.save_for_backward(torch.tensor(lambda_))
                return x.view_as(x)

            @staticmethod
            def backward(ctx, grad_output):  # type: ignore[override]
                (lambda_,) = ctx.saved_tensors
                return -lambda_.item() * grad_output, None

        return _GRLFn

    @classmethod
    def apply(cls, x, lambda_):
        return cls._get_fn().apply(x, lambda_)


class _DANNNet:
    """Thin wrapper that lazily creates the PyTorch DANN network."""

    def __new__(cls, n_features: int, hidden_dim: int = 64):  # type: ignore[misc]
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature_extractor = nn.Sequential(
                    nn.Linear(n_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 32),
                    nn.ReLU(),
                )
                self.label_predictor = nn.Linear(32, 2)
                self.domain_discriminator = nn.Linear(32, 2)

            def forward(self, x, lambda_: float = 0.0):  # type: ignore[override]
                features = self.feature_extractor(x)
                label_out = self.label_predictor(features)
                domain_out = self.domain_discriminator(_GRL.apply(features, lambda_))
                return label_out, domain_out

        return _Net()


class _DANNWrapper:
    """Wraps a trained DANN _Net to provide predict_proba / predict interfaces."""

    def __init__(self, net, classes_: np.ndarray, columns: list[str], scaler=None) -> None:
        self._net = net
        self.classes_ = classes_
        self._columns = columns
        self._scaler = scaler

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        self._net.eval()
        X_sub = X[self._columns].fillna(0).values.astype(np.float32)
        if self._scaler is not None:
            X_sub = self._scaler.transform(X_sub).astype(np.float32)
        with torch.no_grad():
            x_t = torch.tensor(X_sub)
            label_out, _ = self._net(x_t, lambda_=0.0)
            proba = torch.softmax(label_out, dim=1).numpy()
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


def _sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    reg: float = 0.1,
    n_iter: int = 200,
) -> np.ndarray:
    """
    Log-domain Sinkhorn algorithm.  Returns transport plan T of shape (len(a), len(b)).
    reg: entropy regularisation — larger = more diffuse transport (closer to uniform).
    Uses scipy logsumexp for numerical stability.
    """
    from scipy.special import logsumexp

    log_K = -M / reg
    log_u = np.zeros_like(a)
    log_v = np.zeros_like(b)
    log_a = np.log(a + 1e-300)
    log_b = np.log(b + 1e-300)
    for _ in range(n_iter):
        log_u = log_a - logsumexp(log_K + log_v[None, :], axis=1)
        log_v = log_b - logsumexp(log_K + log_u[:, None], axis=0)
    return np.exp(log_K + log_u[:, None] + log_v[None, :])


def run_level_ot(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    target: pd.DataFrame,
    label_col: str,
    reg: float = 0.1,
    n_rounds: int = 3,
    weight_power: float = 2.0,
    max_tgt_ot: int = 1000,
    **xgb_kwargs,
) -> AdaptationResult:
    """
    Class-conditional Optimal Transport reweighting.

    Computes two separate Sinkhorn transport plans — one matching source
    positives to predicted target positives, another matching source
    negatives to predicted target negatives.  This corrects covariate
    shift WITHIN each class without conflating them, avoiding the failure
    mode of pure feature-space OT where inverted-label source rows get
    upweighted because they happen to be close to target rows in feature space.

    Each round: refit model → re-predict target classes → recompute OT.

    Parameters
    ----------
    reg:
        Sinkhorn entropy regularisation (smaller = sharper transport).
    n_rounds:
        Number of model-refit iterations (3 is typically sufficient).
    max_tgt_ot:
        Maximum target rows used per class for the cost matrix.
    """
    logger.info("[Level OT] Class-conditional OT (reg=%.3f, rounds=%d)", reg, n_rounds)

    X_src, y_src, w_disc = _pool_sources(
        aligned, discovery_scores, label_col, weight_power=weight_power
    )

    # Impute NaN with source column medians, then standardise.
    X_src_arr = X_src.values.astype(float)
    X_tgt_arr = target.values.astype(float)
    col_medians = np.nanmedian(X_src_arr, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    for arr in (X_src_arr, X_tgt_arr):
        mask = np.isnan(arr)
        if mask.any():
            arr[mask] = np.take(col_medians, np.where(mask)[1])

    scaler = StandardScaler()
    X_src_scaled = scaler.fit_transform(X_src_arr)
    X_tgt_scaled = scaler.transform(X_tgt_arr)

    # Identify the positive class (class 1 if available)
    try:
        pos_class = sorted(y_src.unique())[1]
    except IndexError:
        pos_class = y_src.unique()[0]
    src_pos_mask = (y_src == pos_class).values
    src_neg_mask = ~src_pos_mask

    # Initialise with the best single source (L0) to get the first class predictions
    best_id = max(aligned, key=lambda k: discovery_scores.get(k, 0.0))
    X_init, y_init = _split_xy(aligned[best_id], label_col)
    model = _make_xgb(**xgb_kwargs)
    model.fit(X_init, y_init)

    best_model = model
    best_proba = model.predict_proba(target)

    rng = np.random.default_rng(xgb_kwargs.get("random_state", 42))

    for round_idx in range(n_rounds):
        # Class-balanced target partitioning: take equal numbers of most-confident
        # positive and negative predictions to prevent model calibration bias from
        # causing one class to dominate the OT and collapse the solution.
        proba_all = best_model.predict_proba(target)
        confidence = proba_all.max(axis=1)
        pred_cls_all = proba_all.argmax(axis=1)
        n_per_class_ot = max(50, len(target) // 4)

        pos_pred_idx = np.where(pred_cls_all == 1)[0]
        neg_pred_idx = np.where(pred_cls_all == 0)[0]

        # Balanced: top-N most confident per class
        n_pos_sel = min(n_per_class_ot, len(pos_pred_idx))
        n_neg_sel = min(n_per_class_ot, len(neg_pred_idx))
        tgt_pos_idx = pos_pred_idx[np.argsort(confidence[pos_pred_idx])[-n_pos_sel:]]
        tgt_neg_idx = neg_pred_idx[np.argsort(confidence[neg_pred_idx])[-n_neg_sel:]]

        w_ot = np.ones(len(X_src_scaled))

        for cls_mask, tgt_idx in [(src_pos_mask, tgt_pos_idx),
                                   (src_neg_mask, tgt_neg_idx)]:
            n_s = int(cls_mask.sum())
            n_t = len(tgt_idx)
            if n_s == 0 or n_t == 0:
                continue

            # Subsample target class rows if large
            if n_t > max_tgt_ot:
                sel = rng.choice(n_t, size=max_tgt_ot, replace=False)
                tgt_sel = tgt_idx[sel]
            else:
                tgt_sel = tgt_idx
            n_ref = len(tgt_sel)

            Xs_cls = X_src_scaled[cls_mask]
            Xt_cls = X_tgt_scaled[tgt_sel]

            diff = Xs_cls[:, None, :] - Xt_cls[None, :, :]
            M = (diff ** 2).sum(axis=2)
            M_med = float(np.median(M))
            if M_med > 1e-9:
                M = M / M_med

            T = _sinkhorn(np.ones(n_s) / n_s, np.ones(n_ref) / n_ref, M, reg=reg)
            w_cls = T.sum(axis=1) * n_s
            w_ot[cls_mask] = w_cls

        w_ot = np.clip(w_ot, 0.01, 10.0)
        w_ot = w_ot / w_ot.mean()

        w_final = w_disc * w_ot
        w_final = w_final / w_final.mean()

        model = _make_xgb(**xgb_kwargs)
        model.fit(X_src, y_src, sample_weight=w_final)
        best_model = model
        best_proba = model.predict_proba(target)

        logger.info(
            "[Level OT] Round %d — w_std=%.3f  tgt_pos=%d/%d  pred_pos_rate=%.3f",
            round_idx + 1, w_final.std(),
            len(tgt_pos_idx), len(target), float(best_proba[:, 1].mean()),
        )

    return AdaptationResult(
        level="level_ot",
        predictions=best_model.predict(target),
        probabilities=best_proba,
        model=best_model,
    )


def run_level_qbc(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    target: pd.DataFrame,
    label_col: str,
    n_committee: int = 5,
    n_rounds: int = 5,
    pseudo_weight: float = 0.5,
    weight_power: float = 2.0,
    min_agree_rate: float = 0.6,
    **xgb_kwargs,
) -> AdaptationResult:
    """
    L2 + Committee Filter (QBC-F).

    Extends L2 pseudo-labeling with a committee-as-filter step that rejects
    pseudo-labels where the committee disagrees with the main model.

    Addresses L2's confirmation bias: the main model selects pseudo-labels
    based on its OWN confidence — a self-reinforcing loop.  Here, K source
    models trained on DIFFERENT individual sources act as independent
    validators.  A pseudo-label is accepted only when committee_agreement >=
    min_agree_rate (default 3/5 models agree with main model's prediction).

    This does not require the committee to be well-calibrated independently
    (they all predict the same systematically biased distribution) — it
    only requires that truly easy target rows (where the label is obvious
    across diverse source perspectives) get higher agreement than hard ones.

    Selection schedule and class-balanced sampling are the same as L2.
    """
    from sklearn.cluster import MiniBatchKMeans

    round_pcts = [0.10, 0.20, 0.35, 0.50, 0.70][:n_rounds]
    logger.info("[Level QBC-F] Committee filter (K=%d, min_agree=%.0f%%, rounds=%d)",
                n_committee, min_agree_rate * 100, n_rounds)

    # Minimum-target gate (same as L2)
    if len(target) < 100:
        logger.info("[Level QBC-F] Target too small (n=%d < 100) — returning L0", len(target))
        best_id = max(aligned, key=lambda k: discovery_scores.get(k, 0.0))
        X_b, y_b = _split_xy(aligned[best_id], label_col)
        m = _make_xgb(**xgb_kwargs); m.fit(X_b, y_b)
        return AdaptationResult(level="level_qbc", predictions=m.predict(target),
                                probabilities=m.predict_proba(target), model=m)

    # Build committee from top-K distinct sources
    sorted_srcs = sorted(aligned, key=lambda k: discovery_scores.get(k, 0.0), reverse=True)
    committee: list = []
    for src_id in sorted_srcs[:n_committee]:
        X_c, y_c = _split_xy(aligned[src_id], label_col)
        if y_c.nunique() < 2:
            continue
        m = _make_xgb(**xgb_kwargs)
        m.fit(X_c, y_c)
        committee.append(m)

    if len(committee) < 2:
        logger.warning("[Level QBC-F] Committee too small — falling back to Level 2")
        return run_level2(aligned, discovery_scores, target, label_col,
                          n_rounds=n_rounds, pseudo_weight=pseudo_weight,
                          weight_power=weight_power, **xgb_kwargs)

    logger.info("[Level QBC-F] Committee size: %d", len(committee))
    min_votes = max(1, int(np.ceil(len(committee) * min_agree_rate)))

    # Source pool — permanent anchor
    X_src, y_src, w_src = _pool_sources(
        aligned, discovery_scores, label_col, weight_power=weight_power
    )

    # Apply confident denoising to source (same as L2)
    best_id = sorted_srcs[0]
    X_init, y_init = _split_xy(aligned[best_id], label_col)
    model = _make_xgb(**xgb_kwargs)
    model.fit(X_init, y_init)
    logger.info("[Level QBC-F] Initial model from '%s'", best_id)

    w_src_de = _confident_denoising(X_src, y_src, w_src, model)

    best_model = model
    best_proba = model.predict_proba(target)
    best_pseudo_auc: Optional[float] = None

    for round_idx, pct in enumerate(round_pcts):
        proba_all = model.predict_proba(target)
        confidence = proba_all.max(axis=1)
        pred_cls = proba_all.argmax(axis=1)

        # Class-balanced candidate selection (same as L2)
        n_per_class = max(5, int(len(target) * pct / 2))
        pseudo_mask = np.zeros(len(target), dtype=bool)
        for c in range(2):
            cls_idx = np.where(pred_cls == c)[0]
            if len(cls_idx) == 0:
                continue
            top_idx = cls_idx[np.argsort(confidence[cls_idx])[-n_per_class:]]
            pseudo_mask[top_idx] = True

        # Committee filter: each committee member votes on each candidate
        # Accept only rows where >= min_votes committee models agree with main prediction
        com_preds_all = np.stack([m.predict(target) for m in committee], axis=0)  # (K, n_tgt)
        # Map predictions to integer class index (0/1) for comparison
        main_pred_idx = pred_cls  # 0 or 1 index into model.classes_
        # committee predictions converted to same 0/1 index space
        com_classes_ref = model.classes_
        com_votes_agree = np.zeros(len(target), dtype=int)
        for k, com_m in enumerate(committee):
            com_pred_raw = com_preds_all[k]  # raw class labels
            # Map to 0/1 matching main model's classes_
            com_pred_idx = np.array([
                int(np.where(com_classes_ref == cp)[0][0])
                if cp in com_classes_ref else -1
                for cp in com_pred_raw
            ])
            com_votes_agree += (com_pred_idx == main_pred_idx).astype(int)

        committee_ok = com_votes_agree >= min_votes
        filtered_mask = pseudo_mask & committee_ok

        n_before = int(pseudo_mask.sum())
        n_after = int(filtered_mask.sum())

        # Re-balance after committee filter: committee biases against positives,
        # so take min(n_pos_filtered, n_neg_filtered) from each class.
        filtered_idx = np.where(filtered_mask)[0]
        filt_cls = pred_cls[filtered_idx]
        pos_filt = filtered_idx[filt_cls == 1]
        neg_filt = filtered_idx[filt_cls == 0]
        n_bal = min(len(pos_filt), len(neg_filt))
        if n_bal > 0:
            sel_pos = pos_filt[np.argsort(confidence[pos_filt])[-n_bal:]]
            sel_neg = neg_filt[np.argsort(confidence[neg_filt])[-n_bal:]]
            pseudo_idx = np.concatenate([sel_pos, sel_neg])
        else:
            pseudo_idx = filtered_idx
        n_balanced = len(pseudo_idx)

        logger.info(
            "[Level QBC-F] Round %d: %d → filter → %d → balanced → %d pseudo-labels",
            round_idx + 1, n_before, n_after, n_balanced,
        )

        if n_balanced < 10:
            logger.info("[Level QBC-F] Too few balanced pseudo-labels — stopping")
            break
        rng = np.random.default_rng(xgb_kwargs.get("random_state", 42) + round_idx)
        val_size = max(5, len(pseudo_idx) // 5)
        val_local = rng.choice(len(pseudo_idx), size=val_size, replace=False)
        val_idx = pseudo_idx[val_local]
        train_idx = np.setdiff1d(pseudo_idx, val_idx)

        train_mask = np.zeros(len(target), dtype=bool)
        train_mask[train_idx] = True

        pseudo_X = target[train_mask].copy()
        pseudo_y = pd.Series(
            model.classes_[proba_all[train_mask].argmax(axis=1)],
            name=label_col,
        )
        pseudo_w = np.full(len(pseudo_X), pseudo_weight)

        if pseudo_y.nunique() < 2:
            logger.info("[Level QBC-F] Round %d: all pseudo-labels single class — skipping",
                        round_idx + 1)
            continue

        X_aug = pd.concat([X_src, pseudo_X], ignore_index=True)
        y_aug = pd.concat([y_src, pseudo_y], ignore_index=True)
        w_aug = np.concatenate([w_src_de, pseudo_w])

        new_model = _make_xgb(**xgb_kwargs)
        new_model.fit(X_aug, y_aug, sample_weight=w_aug)

        # Early stopping on pseudo-holdout AUC
        if len(val_idx) >= 10:
            val_pseudo_bin = (
                model.classes_[proba_all[val_idx].argmax(axis=1)] == model.classes_[1]
            ).astype(int)
            new_val_scores = new_model.predict_proba(target.iloc[val_idx])[:, 1]
            try:
                round_auc = float(roc_auc_score(val_pseudo_bin, new_val_scores))
            except ValueError:
                round_auc = float("nan")
            if np.isnan(round_auc):
                best_model = new_model
                best_proba = new_model.predict_proba(target)
            else:
                logger.info("[Level QBC-F] Round %d pseudo-holdout AUC: %.4f", round_idx + 1, round_auc)
                if best_pseudo_auc is not None and round_auc < best_pseudo_auc - 0.01:
                    logger.info("[Level QBC-F] Early stop: AUC degraded")
                    break
                if best_pseudo_auc is None or round_auc >= best_pseudo_auc:
                    best_pseudo_auc = round_auc
                    best_model = new_model
                    best_proba = new_model.predict_proba(target)
        else:
            best_model = new_model
            best_proba = new_model.predict_proba(target)

        model = new_model

    return AdaptationResult(
        level="level_qbc",
        predictions=best_model.predict(target),
        probabilities=best_proba,
        model=best_model,
    )


def run_level5(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    target: pd.DataFrame,
    label_col: str,
    n_epochs: int = 200,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    weight_power: float = 1.0,
    unlabeled_features: Optional[dict[str, pd.DataFrame]] = None,
    volume_src: Optional[pd.DataFrame] = None,
    random_state: int = 42,
    use_gce: bool = True,
    **_ignored_xgb_kwargs,
) -> AdaptationResult:
    """
    DANN — Domain-Adversarial Neural Network.

    Requires PyTorch (``pip install torch``).  Architecture: 2-layer MLP feature
    extractor -> label predictor + gradient-reversed domain discriminator.
    Discovery scores are used as per-sample weights on the label loss.

    Parameters
    ----------
    unlabeled_features:
        Optional dict of feature-only DataFrames (e.g. from GitTables).
        Added to the target batches fed to the domain discriminator, broadening
        the domain classifier's view of P(source) without touching the label loss.
    volume_src:
        Optional concatenated DataFrame of weakly-aligned source features (no labels).
        Added to the SOURCE side of the domain discriminator only — not the label loss.
        Restores DANN's source volume when the quality alignment threshold is strict.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "Level 5 (DANN) requires PyTorch. Install with: pip install torch"
        ) from exc

    logger.info("[Level 5] DANN — adversarial domain adaptation (%d epochs)", n_epochs)
    torch.manual_seed(random_state)

    X_src, y_src, w_src = _pool_sources(
        aligned, discovery_scores, label_col, weight_power=weight_power
    )

    classes_ = np.array(sorted(y_src.unique()))
    label_map = {c: i for i, c in enumerate(classes_)}
    y_src_idx = y_src.map(label_map).values.astype(np.int64)

    X_src_clean, target_clean = _domain_classifier_features(X_src, target)
    X_src_np = X_src_clean.values.astype(np.float32)
    X_tgt_base_np = target_clean.values.astype(np.float32)

    # Per-domain normalization: scale source by source statistics, target by target
    # statistics. Fitting the scaler on pooled heterogeneous sources (1000+ tables)
    # produces a corrupted mean/std. Per-domain scaling puts each domain on a
    # comparable scale without mixing their distributions.
    src_scaler = StandardScaler()
    X_src_np = src_scaler.fit_transform(X_src_np)
    dann_scaler = StandardScaler()
    X_tgt_base_np = dann_scaler.fit_transform(X_tgt_base_np)  # stored for inference

    X_tgt_np = X_tgt_base_np.copy()
    if unlabeled_features:
        unl_dc = _pool_unlabeled(unlabeled_features, list(target_clean.columns))
        if unl_dc is not None:
            logger.info("[Level 5] Augmenting domain discriminator with %d unlabeled rows.", len(unl_dc))
            unl_np = dann_scaler.transform(unl_dc.values.astype(np.float32))
            X_tgt_np = np.concatenate([X_tgt_np, unl_np], axis=0)

    n_features = X_src_np.shape[1]
    logger.info("[Level 5] Training on %d matched columns (of %d total)", n_features, X_src.shape[1])

    # Volume sources: weakly-aligned tables that failed quality threshold.
    # Used for SOURCE-side domain discriminator only — not the label loss.
    X_vol_t: "Optional[torch.Tensor]" = None
    n_vol = 0
    if volume_src is not None and len(volume_src) > 0:
        vol_cols = [c for c in X_src_clean.columns if c in volume_src.columns]
        if vol_cols:
            X_vol = volume_src.reindex(columns=list(X_src_clean.columns))
            for c in X_src_clean.columns:
                med = float(np.nanmedian(X_src_clean[c].values)) if X_src_clean[c].notna().any() else 0.0
                X_vol[c] = X_vol[c].fillna(med)
            X_vol_np = src_scaler.transform(X_vol.values.astype(np.float32))
            n_vol = len(X_vol_np)
            X_vol_t = torch.tensor(X_vol_np)
            logger.info("[Level 5] Volume augmentation: %d rows → SOURCE-side domain discriminator", n_vol)

    net = _DANNNet(n_features, hidden_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss(reduction="none")

    n_src = len(X_src_np)
    n_tgt = len(X_tgt_np)
    batch_size = min(256, max(32, n_src // 4))

    X_src_t = torch.tensor(X_src_np)
    y_src_t = torch.tensor(y_src_idx)
    w_src_t = torch.tensor(w_src.astype(np.float32))
    X_tgt_t = torch.tensor(X_tgt_np)

    rng = np.random.default_rng(random_state)
    net.train()
    for epoch in range(n_epochs):
        p = epoch / max(n_epochs - 1, 1)
        lambda_ = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        src_idx = rng.integers(0, n_src, batch_size)
        tgt_idx = rng.integers(0, n_tgt, batch_size)

        x_s = X_src_t[src_idx]
        y_s = y_src_t[src_idx]
        w_s = w_src_t[src_idx]
        x_t = X_tgt_t[tgt_idx]

        label_out, domain_out_s = net(x_s, lambda_)
        _, domain_out_t = net(x_t, lambda_)

        if use_gce:
            lbl_loss = _gce_loss(label_out, y_s, weights=w_s * batch_size)
        else:
            lbl_loss = (ce_loss(label_out, y_s) * w_s * batch_size).mean()

        d_labels_s = torch.zeros(batch_size, dtype=torch.long)
        d_labels_t = torch.ones(batch_size, dtype=torch.long)
        if n_vol > 0 and X_vol_t is not None:
            vol_idx = rng.integers(0, n_vol, batch_size)
            x_vol = X_vol_t[vol_idx]
            _, domain_out_vol = net(x_vol, lambda_)
            d_labels_vol = torch.zeros(batch_size, dtype=torch.long)
            dom_loss = (
                ce_loss(domain_out_s, d_labels_s).mean()
                + ce_loss(domain_out_vol, d_labels_vol).mean()
                + ce_loss(domain_out_t, d_labels_t).mean()
            ) / 3.0
        else:
            dom_loss = (
                ce_loss(domain_out_s, d_labels_s).mean()
                + ce_loss(domain_out_t, d_labels_t).mean()
            ) / 2.0

        loss = lbl_loss + lambda_ * dom_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            logger.debug(
                "[Level 5] Epoch %d/%d  lambda=%.3f  loss=%.4f",
                epoch + 1, n_epochs, lambda_, loss.detach().item(),
            )

    wrapper = _DANNWrapper(net, classes_, list(X_src_clean.columns), scaler=dann_scaler)
    proba = wrapper.predict_proba(target)
    return AdaptationResult(
        level="level5",
        predictions=wrapper.predict(target),
        probabilities=proba,
        model=wrapper,
    )


# ---------------------------------------------------------------------------
# Level 5.5 — CDAN (Conditional DANN, Long et al. 2018)
# ---------------------------------------------------------------------------

class _CDANNet:
    """CDAN network: domain discriminator conditions on feature ⊗ classifier output."""

    def __new__(cls, n_features: int, hidden_dim: int = 64):  # type: ignore[misc]
        import torch
        import torch.nn as nn

        feat_out = 32   # fixed output dim of feature extractor
        n_classes = 2
        disc_in = feat_out * n_classes   # outer-product dimension

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature_extractor = nn.Sequential(
                    nn.Linear(n_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, feat_out),
                    nn.ReLU(),
                )
                self.label_predictor = nn.Linear(feat_out, n_classes)
                self.domain_discriminator = nn.Sequential(
                    nn.Linear(disc_in, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2),
                )

            def forward(self, x, lambda_: float = 0.0):  # type: ignore[override]
                feat = self.feature_extractor(x)
                label_out = self.label_predictor(feat)
                # Outer product: (B, feat_out) × (B, n_classes) → (B, feat_out*n_classes)
                softmax_out = torch.softmax(label_out.detach(), dim=1)
                op = torch.bmm(feat.unsqueeze(2), softmax_out.unsqueeze(1))
                op = op.view(x.size(0), -1)
                domain_out = self.domain_discriminator(_GRL.apply(op, lambda_))
                return label_out, domain_out

        return _Net()


def run_level55(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    target: pd.DataFrame,
    label_col: str,
    n_epochs: int = 200,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    weight_power: float = 1.0,
    unlabeled_features: Optional[dict[str, pd.DataFrame]] = None,
    volume_src: Optional[pd.DataFrame] = None,
    random_state: int = 42,
    use_gce: bool = True,
    **_ignored_xgb_kwargs,
) -> AdaptationResult:
    """
    CDAN — Conditional DANN (Long et al. 2018).

    Conditions the domain discriminator on the outer product of extracted
    features and classifier softmax predictions, providing richer transfer
    signal than vanilla DANN when the feature extractor already captures
    label-relevant structure.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "Level 5.5 (CDAN) requires PyTorch. Install with: pip install torch"
        ) from exc

    logger.info("[Level 5.5] CDAN — conditional adversarial adaptation (%d epochs)", n_epochs)
    torch.manual_seed(random_state)

    X_src, y_src, w_src = _pool_sources(
        aligned, discovery_scores, label_col, weight_power=weight_power
    )

    classes_ = np.array(sorted(y_src.unique()))
    label_map = {c: i for i, c in enumerate(classes_)}
    y_src_idx = y_src.map(label_map).values.astype(np.int64)

    X_src_clean, target_clean = _domain_classifier_features(X_src, target)
    X_src_np = X_src_clean.values.astype(np.float32)
    X_tgt_base_np = target_clean.values.astype(np.float32)

    cdan_src_scaler = StandardScaler()
    X_src_np = cdan_src_scaler.fit_transform(X_src_np)
    cdan_scaler = StandardScaler()
    X_tgt_base_np = cdan_scaler.fit_transform(X_tgt_base_np)

    X_tgt_np = X_tgt_base_np.copy()
    if unlabeled_features:
        unl_dc = _pool_unlabeled(unlabeled_features, list(target_clean.columns))
        if unl_dc is not None:
            logger.info("[Level 5.5] Augmenting domain discriminator with %d unlabeled rows.", len(unl_dc))
            unl_np = cdan_scaler.transform(unl_dc.values.astype(np.float32))
            X_tgt_np = np.concatenate([X_tgt_np, unl_np], axis=0)

    n_features = X_src_np.shape[1]
    logger.info("[Level 5.5] Training on %d matched columns (of %d total)", n_features, X_src.shape[1])

    # Volume sources: weakly-aligned tables that failed quality threshold.
    X_vol_t_55: "Optional[torch.Tensor]" = None
    n_vol_55 = 0
    if volume_src is not None and len(volume_src) > 0:
        vol_cols_55 = [c for c in X_src_clean.columns if c in volume_src.columns]
        if vol_cols_55:
            X_vol_55 = volume_src.reindex(columns=list(X_src_clean.columns))
            for c in X_src_clean.columns:
                med = float(np.nanmedian(X_src_clean[c].values)) if X_src_clean[c].notna().any() else 0.0
                X_vol_55[c] = X_vol_55[c].fillna(med)
            X_vol_np_55 = cdan_src_scaler.transform(X_vol_55.values.astype(np.float32))
            n_vol_55 = len(X_vol_np_55)
            X_vol_t_55 = torch.tensor(X_vol_np_55)
            logger.info("[Level 5.5] Volume augmentation: %d rows → SOURCE-side domain discriminator", n_vol_55)

    # CDAN needs a rich enough source sample to train a meaningful feature
    # extractor before conditioning the domain discriminator. Fall back to
    # vanilla DANN when there is not enough labelled source data.
    # GitTables sources are small (65-116 rows/table), so 200 requires only ~2 quality tables.
    _CDAN_MIN_SRC_ROWS = 200
    if len(X_src_np) < _CDAN_MIN_SRC_ROWS:
        logger.info(
            "[Level 5.5] Only %d source rows — CDAN fallback to DANN (min=%d)",
            len(X_src_np), _CDAN_MIN_SRC_ROWS,
        )
        result = run_level5(
            aligned, discovery_scores, target, label_col,
            n_epochs=n_epochs, hidden_dim=hidden_dim, lr=lr,
            weight_power=weight_power, unlabeled_features=unlabeled_features,
            volume_src=volume_src, random_state=random_state, use_gce=False,
        )
        return AdaptationResult(
            level="level55",
            predictions=result.predictions,
            probabilities=result.probabilities,
            model=result.model,
        )

    net = _CDANNet(n_features, hidden_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss(reduction="none")

    n_src = len(X_src_np)
    n_tgt = len(X_tgt_np)
    batch_size = min(256, max(32, n_src // 4))

    X_src_t = torch.tensor(X_src_np)
    y_src_t = torch.tensor(y_src_idx)
    w_src_t = torch.tensor(w_src.astype(np.float32))
    X_tgt_t = torch.tensor(X_tgt_np)

    rng = np.random.default_rng(random_state)
    net.train()
    for epoch in range(n_epochs):
        p = epoch / max(n_epochs - 1, 1)
        lambda_ = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        src_idx = rng.integers(0, n_src, batch_size)
        tgt_idx = rng.integers(0, n_tgt, batch_size)

        x_s = X_src_t[src_idx]
        y_s = y_src_t[src_idx]
        w_s = w_src_t[src_idx]
        x_t = X_tgt_t[tgt_idx]

        label_out, domain_out_s = net(x_s, lambda_)
        _, domain_out_t = net(x_t, lambda_)

        if use_gce:
            lbl_loss = _gce_loss(label_out, y_s, weights=w_s * batch_size)
        else:
            lbl_loss = (ce_loss(label_out, y_s) * w_s * batch_size).mean()

        d_labels_s = torch.zeros(batch_size, dtype=torch.long)
        d_labels_t = torch.ones(batch_size, dtype=torch.long)
        if n_vol_55 > 0 and X_vol_t_55 is not None:
            vol_idx_55 = rng.integers(0, n_vol_55, batch_size)
            x_vol_55 = X_vol_t_55[vol_idx_55]
            _, domain_out_vol_55 = net(x_vol_55, lambda_)
            d_labels_vol_55 = torch.zeros(batch_size, dtype=torch.long)
            dom_loss = (
                ce_loss(domain_out_s, d_labels_s).mean()
                + ce_loss(domain_out_vol_55, d_labels_vol_55).mean()
                + ce_loss(domain_out_t, d_labels_t).mean()
            ) / 3.0
        else:
            dom_loss = (
                ce_loss(domain_out_s, d_labels_s).mean()
                + ce_loss(domain_out_t, d_labels_t).mean()
            ) / 2.0

        loss = lbl_loss + lambda_ * dom_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            logger.debug(
                "[Level 5.5] Epoch %d/%d  lambda=%.3f  loss=%.4f",
                epoch + 1, n_epochs, lambda_, loss.detach().item(),
            )

    # Re-use DANNWrapper — it only calls label_predictor via forward with lambda_=0
    wrapper = _DANNWrapper(net, classes_, list(X_src_clean.columns), scaler=cdan_scaler)
    proba = wrapper.predict_proba(target)
    return AdaptationResult(
        level="level55",
        predictions=wrapper.predict(target),
        probabilities=proba,
        model=wrapper,
    )


def run_ensemble(
    result_l2: AdaptationResult,
    result_l5: AdaptationResult,
) -> AdaptationResult:
    """
    Confidence-weighted ensemble of L2 and L5.

    Each model is weighted by its negative predictive entropy on the target
    (higher = more decisive predictions = higher confidence).  Models that
    make confident predictions get higher weight.
    """
    p_l2 = result_l2.probabilities[:, 1]
    p_l5 = result_l5.probabilities[:, 1]

    def _neg_entropy(p: np.ndarray) -> float:
        p_c = np.clip(p, 1e-9, 1.0 - 1e-9)
        return float(-np.mean(p_c * np.log(p_c) + (1.0 - p_c) * np.log(1.0 - p_c)))

    conf_l2 = _neg_entropy(p_l2)
    conf_l5 = _neg_entropy(p_l5)
    total = conf_l2 + conf_l5
    w_l2 = conf_l2 / total if total > 1e-12 else 0.5
    w_l2 = max(w_l2, 0.3)   # floor: prevent bad L5 from dominating when L2 is uncertain
    w_l5 = 1.0 - w_l2

    logger.info(
        "[Ensemble] L2 conf=%.4f (w=%.3f)  L5 conf=%.4f (w=%.3f)",
        conf_l2, w_l2, conf_l5, w_l5,
    )

    p_ens = w_l2 * p_l2 + w_l5 * p_l5
    proba = np.column_stack([1.0 - p_ens, p_ens])
    return AdaptationResult(
        level="ensemble",
        predictions=(p_ens >= 0.5).astype(int),
        probabilities=proba,
        model=None,
    )


def run_oracle(
    target_train: pd.DataFrame,
    target_test: pd.DataFrame,
    label_col: str,
    **xgb_kwargs,
) -> AdaptationResult:
    """
    Upper bound: supervised training directly on labeled target data.
    Shows the ceiling that UDA methods are trying to approach.
    """
    logger.info("[Oracle] Supervised training on labeled target data (%d rows)", len(target_train))
    X_train, y_train = _split_xy(target_train.dropna(), label_col)
    model = _make_xgb(**xgb_kwargs)
    model.fit(X_train, y_train)
    proba = model.predict_proba(target_test)
    return AdaptationResult(
        level="oracle",
        predictions=model.predict(target_test),
        probabilities=proba,
        model=model,
    )


def run_source_ensemble(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    target: pd.DataFrame,
    label_col: str,
    min_sources: int = 3,
    random_state: int = 42,
) -> AdaptationResult:
    """
    Train one XGBoost per aligned source, then take the discovery-score-weighted
    median of their predicted P(y=1) across target rows.

    Weighted median is more robust than mean: a single badly-aligned source
    produces extreme probabilities that pull the mean off but only shifts the
    median if it constitutes the majority by weight.
    """
    logger.info("[SourceEnsemble] Training %d per-source models ...", len(aligned))
    source_probas: list[np.ndarray] = []
    source_weights: list[float]     = []

    for table_id, df in aligned.items():
        X_src, y_src = _split_xy(df, label_col)
        if y_src.nunique() < 2 or len(X_src) < 20:
            continue
        m = _make_xgb(n_estimators=100, max_depth=3, random_state=random_state)
        try:
            m.fit(X_src, y_src)
            proba = m.predict_proba(target)[:, 1]
        except Exception:
            continue
        source_probas.append(proba)
        source_weights.append(max(float(discovery_scores.get(table_id, 1e-3)), 1e-6))

    if len(source_probas) < min_sources:
        logger.warning("[SourceEnsemble] Only %d sources — falling back to L0", len(source_probas))
        return run_level0(aligned, discovery_scores, target, label_col)

    logger.info("[SourceEnsemble] Combining %d source models via weighted median", len(source_probas))
    probas  = np.array(source_probas)        # (n_sources, n_target)
    weights = np.array(source_weights)
    weights = weights / weights.sum()

    # Weighted median per target row
    ensemble_p = np.zeros(probas.shape[1])
    for j in range(probas.shape[1]):
        col   = probas[:, j]
        order = np.argsort(col)
        cumw  = np.cumsum(weights[order])
        idx   = int(np.searchsorted(cumw, 0.5))
        idx   = min(idx, len(order) - 1)
        ensemble_p[j] = col[order[idx]]

    proba_2d = np.column_stack([1.0 - ensemble_p, ensemble_p])
    return AdaptationResult(
        level="source_ensemble",
        predictions=(ensemble_p >= 0.5).astype(int),
        probabilities=proba_2d,
        model=None,
    )


def run_all(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    target: pd.DataFrame,
    label_col: str,
    weight_power: float = 2.0,
    n_rounds: int = 5,
    pseudo_weight: float = 0.5,
    n_dann_epochs: int = 200,
    unlabeled_features: Optional[dict[str, pd.DataFrame]] = None,
    volume_src: Optional[pd.DataFrame] = None,
    random_state: int = 42,
    **xgb_kwargs,
) -> dict[str, AdaptationResult]:
    """
    Run all adaptation levels and return results keyed by level name.

    Baseline A (majority class) and Baseline B (random sources) must be added
    separately — they require inputs not available here.

    Returns keys: baseline, level0, level2, level5, ensemble, source_ensemble
    """
    _xgb = dict(random_state=random_state, **xgb_kwargs)
    r_baseline = run_baseline(aligned, discovery_scores, target, label_col, **_xgb)
    r_l0       = run_level0(aligned, discovery_scores, target, label_col, **_xgb)
    r_l2       = run_level2(aligned, discovery_scores, target, label_col,
                            n_rounds=n_rounds, pseudo_weight=pseudo_weight,
                            weight_power=weight_power, **_xgb)
    r_l2_lsc   = run_level2_lsc(aligned, discovery_scores, target, label_col,
                                 n_rounds=n_rounds, pseudo_weight=pseudo_weight,
                                 weight_power=weight_power, **_xgb)
    r_l5       = run_level5(aligned, discovery_scores, target, label_col,
                            n_epochs=n_dann_epochs, weight_power=weight_power,
                            unlabeled_features=unlabeled_features,
                            volume_src=volume_src,
                            random_state=random_state, use_gce=False)
    r_l55      = run_level55(aligned, discovery_scores, target, label_col,
                             n_epochs=n_dann_epochs, weight_power=weight_power,
                             unlabeled_features=unlabeled_features,
                             volume_src=volume_src,
                             random_state=random_state, use_gce=False)
    r_ens      = run_ensemble(r_l2, r_l5)
    r_src_ens  = run_source_ensemble(aligned, discovery_scores, target, label_col,
                                     random_state=random_state)
    r_l6 = run_level6(aligned, discovery_scores, target, label_col,
                       random_state=random_state, l5_result=r_l5, use_gce=False)
    return {
        "baseline":        r_baseline,
        "level0":          r_l0,
        "level2":          r_l2,
        "level2_lsc":      r_l2_lsc,
        "level5":          r_l5,
        "level55":         r_l55,
        "level6":          r_l6,
        "ensemble":        r_ens,
        "source_ensemble": r_src_ens,
    }


# ---------------------------------------------------------------------------
# Level 6 — FTTA helpers
# ---------------------------------------------------------------------------

def _make_ftta_net(n_features: int, hidden_dim: int):
    """MLP with BatchNorm — BatchNorm stats are updated during TTA."""
    import torch.nn as nn

    class _FTTANet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.extractor = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(hidden_dim // 2, 2)

        def forward(self, x):
            return self.classifier(self.extractor(x))

        def get_features(self, x):
            return self.extractor(x)

    return _FTTANet()


class _FTTAWrapper:
    def __init__(self, net, classes_: np.ndarray, columns: list[str]) -> None:
        self._net = net
        self.classes_ = classes_
        self._columns = columns

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        self._net.eval()
        X_sub = X[self._columns].fillna(0)
        with torch.no_grad():
            x_t = torch.tensor(X_sub.values.astype(np.float32))
            logits = self._net(x_t)
            proba = torch.softmax(logits, dim=1).numpy()
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


# ---------------------------------------------------------------------------
# Level 6 — FTTA: Fully Test-Time Adaptation for Tabular Data (AAAI 2025)
# ---------------------------------------------------------------------------

def run_level6(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    target: pd.DataFrame,
    label_col: str,
    n_pretrain_epochs: int = 200,
    n_adapt_steps: int = 50,
    hidden_dim: int = 128,
    lr_pretrain: float = 1e-3,
    adapt_lrs: tuple = (5e-5, 2e-4, 1e-3),
    k_neighbors: int = 10,
    cdo_ent_thresh: float = 0.6,
    random_state: int = 42,
    l5_result: Optional["AdaptationResult"] = None,
    use_gce: bool = True,
    **_ignored,
) -> AdaptationResult:
    """
    FTTA — Fully Test-Time Adaptation for Tabular Data (AAAI 2025).

    Three modules applied at inference time to a pre-trained MLP:

    CDO  Confident Distribution Optimizer
         Estimates target label distribution P(Y) from high-confidence
         predictions, then applies a log-ratio correction to the model's
         logits to cancel the source→target label shift.  Critical for our
         setting where all repurposed sources are median-binarised (~50/50)
         but targets have varied class balance.

    LCW  Local Consistent Weighter
         Builds a k-NN graph over target samples in feature space.
         Weights the entropy minimisation loss by each sample's neighbourhood
         prediction consistency — regions of agreement get higher weight,
         reducing the risk of noisy gradient updates on ambiguous samples.

    DME  Dynamic Model Ensembler
         Runs TTA with ``len(adapt_lrs)`` learning rates (default 3), then
         combines the resulting models weighted inversely by their final
         prediction entropy.  Reduces sensitivity to the learning rate choice.

    Parameters
    ----------
    n_pretrain_epochs : int
        Supervised training epochs on the pooled aligned sources.
    n_adapt_steps : int
        TTA gradient steps per learning rate (DME).
    hidden_dim : int
        Width of the MLP hidden layers.
    adapt_lrs : tuple of float
        Learning rates for DME — one model per entry.
    k_neighbors : int
        k for the LCW neighbourhood graph.
    cdo_ent_thresh : float
        Fraction of max-entropy (log 2) used as the CDO confidence gate.
        Samples with H < cdo_ent_thresh × log(2) are "confident".
    """
    try:
        import torch
        import torch.nn as nn
        import copy
    except ImportError as exc:
        raise ImportError("Level 6 (FTTA) requires PyTorch.") from exc

    logger.info(
        "[Level 6] FTTA — CDO+LCW+DME  (%d pretrain epochs, %d adapt steps, %d lr variants)",
        n_pretrain_epochs, n_adapt_steps, len(adapt_lrs),
    )
    torch.manual_seed(random_state)

    # --- shared feature columns (same filter as L5) -------------------------
    X_src, y_src, w_src = _pool_sources(aligned, discovery_scores, label_col)
    if len(X_src) < 10 or len(y_src.unique()) < 2:
        logger.warning("[Level 6] Insufficient source data — returning neutral predictions")
        n = len(target)
        proba = np.full((n, 2), 0.5)
        return AdaptationResult(level="level6", predictions=np.zeros(n, int), probabilities=proba)

    X_src_clean, target_clean = _domain_classifier_features(X_src, target)
    n_features = X_src_clean.shape[1]
    logger.info("[Level 6] Training on %d features", n_features)

    classes_ = np.array(sorted(y_src.unique()))
    label_map = {c: i for i, c in enumerate(classes_)}
    y_idx = y_src.map(label_map).values.astype(np.int64)

    # CDO: training-time positive rate (used to estimate label shift at test time)
    train_pos_rate = float((y_idx == 1).mean())

    X_src_np = X_src_clean.values.astype(np.float32)
    X_tgt_np = target_clean.fillna(0).values.astype(np.float32)

    X_src_t = torch.tensor(X_src_np)
    y_src_t = torch.tensor(y_idx)
    w_src_t = torch.tensor(w_src.astype(np.float32))
    X_tgt_t = torch.tensor(X_tgt_np)

    # L5 soft labels for KL anchoring during pretraining
    # This prevents FTTA from learning a bad representation before TTA.
    # The pretrained MLP is jointly trained to (a) classify source repurposed
    # labels and (b) match L5's prediction distribution on target.
    # TTA then refines from an already-reasonable starting point.
    p5_t: Optional[torch.Tensor] = None
    if l5_result is not None:
        p5_t = torch.tensor(l5_result.probabilities.astype(np.float32))

    # --- pre-training --------------------------------------------------------
    net = _make_ftta_net(n_features, hidden_dim)
    opt_pre = torch.optim.Adam(net.parameters(), lr=lr_pretrain, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss(reduction="none")

    n_src = len(X_src_np)
    batch_size = min(256, max(32, n_src // 4))
    rng = np.random.default_rng(random_state)

    net.train()
    for epoch in range(n_pretrain_epochs):
        idx = rng.integers(0, n_src, batch_size)
        logits = net(X_src_t[idx])
        if use_gce:
            loss = _gce_loss(logits, y_src_t[idx], weights=w_src_t[idx] * batch_size)
        else:
            loss = (ce_loss(logits, y_src_t[idx]) * w_src_t[idx] * batch_size).mean()
        if p5_t is not None:
            # KL(L5 || FTTA) on target: cross-entropy of L5 soft labels against
            # FTTA log-probs.  Anchors target predictions to L5 so TTA starts
            # from a position that already agrees with the domain-adapted model.
            tgt_log_probs = torch.log_softmax(net(X_tgt_t), dim=1)
            kl = -(p5_t * tgt_log_probs).sum(dim=1).mean()
            loss = loss + kl
        opt_pre.zero_grad()
        loss.backward()
        opt_pre.step()

    # --- CDO: estimate target label distribution ----------------------------
    net.eval()
    with torch.no_grad():
        logits_init = net(X_tgt_t)
        probs_init = torch.softmax(logits_init, dim=1)
        ent_init = -(probs_init * torch.log(probs_init + 1e-9)).sum(dim=1)

    max_ent = float(np.log(2))
    conf_mask = ent_init < cdo_ent_thresh * max_ent
    if conf_mask.sum() >= max(5, int(0.05 * len(X_tgt_np))):
        est_pos_rate = float(probs_init[conf_mask, 1].mean().item())
    else:
        est_pos_rate = train_pos_rate   # fallback: no shift correction
        logger.debug("[Level 6] CDO: too few confident samples — no label shift correction")

    # log-ratio logit adjustment: shifts decision boundary for P(Y) mismatch
    eps = 1e-6
    cdo_adj = torch.tensor([
        float(np.log(max(1 - train_pos_rate, eps) / max(1 - est_pos_rate, eps))),
        float(np.log(max(train_pos_rate, eps)     / max(est_pos_rate, eps))),
    ], dtype=torch.float32)
    cdo_adj = torch.clamp(cdo_adj, -3.0, 3.0)   # prevent extreme corrections
    logger.info(
        "[Level 6] CDO: train_pos=%.3f  est_target_pos=%.3f  adj=[%.3f, %.3f]",
        train_pos_rate, est_pos_rate, cdo_adj[0].item(), cdo_adj[1].item(),
    )

    # --- LCW: k-NN consistency weights -------------------------------------
    k = min(k_neighbors, max(1, len(X_tgt_np) - 1))
    with torch.no_grad():
        feats = net.get_features(X_tgt_t)          # (N, hidden//2)
        dists = torch.cdist(feats, feats)           # (N, N)
        knn_idx = dists.topk(k + 1, largest=False).indices[:, 1:]   # (N, k)

        adj_logits = logits_init + cdo_adj
        adj_probs  = torch.softmax(adj_logits, dim=1)
        knn_probs  = adj_probs[knn_idx].mean(dim=1)    # (N, 2)
        knn_ent    = -(knn_probs * torch.log(knn_probs + 1e-9)).sum(dim=1)
        lcw_w      = torch.clamp(1.0 - knn_ent / max_ent, 0.0, 1.0)   # (N,)

    logger.debug("[Level 6] LCW: mean weight=%.3f", float(lcw_w.mean()))

    # TTA entropy gate: gate on L5's entropy rather than the pretrained model's.
    # If L5 itself is near-uniform (H > 0.65 ≈ 94% of log(2)=0.693), the domain
    # gap is too large for DANN to find any discriminative structure on target.
    # TTA entropy minimisation will then latch onto noise rather than signal.
    # When gated, effective_adapt_steps=0 → _tta_run returns pretrained model
    # unchanged; since pretraining is KL-anchored to L5, the DME+L5 blend
    # collapses to L5 predictions (safe fallback, no regression).
    _TTA_ENT_GATE = 0.65   # 94% of max entropy
    h_l5 = float(
        -(l5_result.probabilities * np.log(l5_result.probabilities + 1e-9)).sum(axis=1).mean()
    ) if l5_result is not None else 0.0
    effective_adapt_steps = 0 if h_l5 > _TTA_ENT_GATE else n_adapt_steps
    logger.info(
        "[Level 6] H(L5)=%.4f → %d TTA steps (gate=%.2f)",
        h_l5, effective_adapt_steps, _TTA_ENT_GATE,
    )

    # --- DME: TTA with multiple learning rates + early stopping ---------------
    def _tta_run(base_net, lr_tta: float) -> tuple:
        """Run TTA on a copy of base_net; return (best_net, best_entropy).

        Early stopping: if mean entropy increases for 5 consecutive steps,
        roll back to the best checkpoint seen so far.  This prevents TTA from
        diverging when the pretrained model is noisy (small / bad sources).
        When effective_adapt_steps==0 (entropy gate triggered), returns the
        pretrained model unchanged.
        """
        m = copy.deepcopy(base_net)
        opt = torch.optim.Adam(m.parameters(), lr=lr_tta)

        best_ent = float("inf")
        best_state = copy.deepcopy(m.state_dict())
        patience, patience_limit = 0, 5

        for _ in range(effective_adapt_steps):
            m.train()
            logits = m(X_tgt_t) + cdo_adj
            probs  = torch.softmax(logits, dim=1)
            ent    = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
            loss   = (lcw_w.detach() * ent).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            m.eval()
            with torch.no_grad():
                curr_ent = float(
                    -(torch.softmax(m(X_tgt_t) + cdo_adj, dim=1)
                      .mul(torch.log(torch.softmax(m(X_tgt_t) + cdo_adj, dim=1) + 1e-9))
                      .sum(dim=1).mean())
                )
            if curr_ent < best_ent:
                best_ent = curr_ent
                best_state = copy.deepcopy(m.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= patience_limit:
                    break   # roll back below

        m.load_state_dict(best_state)
        return m, best_ent

    adapted_models = []
    final_entropies = []
    for lr_tta in adapt_lrs:
        m_adapted, ent_final = _tta_run(net, lr_tta)
        adapted_models.append(m_adapted)
        final_entropies.append(ent_final)
        logger.debug("[Level 6] DME lr=%.2e → best entropy=%.4f", lr_tta, ent_final)

    # --- DME + L5 anchor blend -----------------------------------------------
    # Add L5 predictions as a fourth "model" in the inverse-entropy ensemble.
    # When FTTA is unreliable (high entropy), L5 automatically dominates.
    # When FTTA is confident (low entropy), it contributes proportionally.
    all_probas: list[np.ndarray] = []
    all_entropies: list[float] = list(final_entropies)

    for m in adapted_models:
        m.eval()
        with torch.no_grad():
            logits = m(X_tgt_t) + cdo_adj
            all_probas.append(torch.softmax(logits, dim=1).numpy())

    if l5_result is not None:
        p5 = l5_result.probabilities.astype(np.float32)
        h5 = float(-(p5 * np.log(p5 + 1e-9)).sum(axis=1).mean())
        all_probas.append(p5)
        all_entropies.append(h5)
        logger.info("[Level 6] L5 anchor H=%.4f  FTTA best H=%.4f", h5, min(final_entropies))

    ent_arr = np.array(all_entropies)
    inv_w = 1.0 / (ent_arr + 1e-9)
    dme_weights = inv_w / inv_w.sum()
    logger.info(
        "[Level 6] DME+L5 weights: %s  (FTTA lrs=%s + L5)",
        np.round(dme_weights, 3).tolist(), list(adapt_lrs),
    )

    final_proba_np = np.zeros((len(X_tgt_np), 2), dtype=np.float32)
    for proba, w in zip(all_probas, dme_weights):
        final_proba_np += w * proba

    wrapper = _FTTAWrapper(
        adapted_models[int(np.argmin(final_entropies))],
        classes_,
        list(target_clean.columns),
    )
    return AdaptationResult(
        level="level6",
        predictions=classes_[final_proba_np.argmax(axis=1)],
        probabilities=final_proba_np,
        model=wrapper,
    )


# ---------------------------------------------------------------------------
# Post-hoc calibration
# ---------------------------------------------------------------------------

def calibrate_result(
    result: "AdaptationResult",
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    X_test: pd.DataFrame,
) -> "AdaptationResult":
    """
    Post-hoc isotonic calibration using a held-out labeled set.

    Wraps result.model with sklearn's CalibratedClassifierCV (cv='prefit',
    method='isotonic') fitted on (X_cal, y_cal), then re-scores X_test.

    Only applicable when result.model is an XGBClassifier instance.
    Returns the original result unchanged if the model is None, a DANN
    wrapper, or the calibration set is too small / single-class.
    """
    if result.model is None or not isinstance(result.model, XGBClassifier):
        return result
    if len(X_cal) < 5 or len(np.unique(y_cal)) < 2:
        logger.warning(
            "calibrate_result: calibration set too small or single-class — skipping"
        )
        return result
    try:
        cal = CalibratedClassifierCV(result.model, cv="prefit", method="isotonic")
        cal.fit(X_cal, y_cal)
        new_proba = cal.predict_proba(X_test)
        return AdaptationResult(
            level=result.level,
            predictions=new_proba.argmax(axis=1),
            probabilities=new_proba,
            model=cal,
        )
    except Exception as exc:
        logger.warning("calibrate_result failed (%s) — returning uncalibrated result", exc)
        return result


# ---------------------------------------------------------------------------
# Semi-supervised helpers (used by act6_semi_supervised.py)
# ---------------------------------------------------------------------------

def validate_sources(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    label_col: str,
    min_vts: float = 0.52,
) -> dict[str, float]:
    """
    Validated Transfer Score (VTS): train each source independently, evaluate
    AUC on a small labeled validation set drawn from the target.

    Returns a combined score = vts × sqrt(discovery_score) for each passing source.
    Using discovery_score in the combined weight stabilises selection when val is
    tiny (< 20 samples) and VTS estimates are noisy — a source that scores 0.55
    on 10 val points but has discovery_score = 0.01 (clearly wrong domain) is
    down-weighted relative to one with discovery_score = 0.40.

    Robustness rules for tiny val sets:
      n_val < 10  → return {} (AUC on <10 points is noise; fall back to target-only)
      n_val < 20  → raise min_vts to 0.60 (require stronger signal)
      n_val < 30  → bootstrap validation: 20 resamples, use median AUC, reject
                    sources with bootstrap std > 0.15 (high-variance = unreliable)

    Returns
    -------
    {table_id: combined_score} for sources whose VTS >= min_vts.
    """
    n_val = len(X_val)

    if n_val < 10:
        logger.info("[VTS] n_val=%d < 10 — skipping lake validation (too few labels)", n_val)
        return {}

    effective_min_vts = max(min_vts, 0.60) if n_val < 20 else min_vts
    use_bootstrap     = n_val < 30

    if effective_min_vts > min_vts:
        logger.info("[VTS] n_val=%d < 20 — raising min_vts %.2f → %.2f", n_val, min_vts, effective_min_vts)
    if use_bootstrap:
        logger.info("[VTS] n_val=%d < 30 — using bootstrap VTS (20 resamples)", n_val)

    rng_boot = np.random.default_rng(42)

    validated: dict[str, float] = {}
    for table_id, df in aligned.items():
        X_src, y_src = _split_xy(df, label_col)
        if y_src.nunique() < 2 or len(X_src) < 5:
            continue
        try:
            m = _make_xgb()
            m.fit(X_src, y_src)
            proba_val = m.predict_proba(X_val)
            if proba_val.shape[1] < 2:
                continue

            if use_bootstrap:
                boot_aucs: list[float] = []
                for _ in range(20):
                    idx = rng_boot.choice(n_val, size=n_val, replace=True)
                    y_b = y_val[idx]
                    if len(np.unique(y_b)) < 2:
                        continue
                    try:
                        boot_aucs.append(float(roc_auc_score(y_b, proba_val[idx, 1])))
                    except Exception:
                        pass
                if not boot_aucs:
                    continue
                vts     = float(np.median(boot_aucs))
                vts_std = float(np.std(boot_aucs))
                if vts_std > 0.15:
                    logger.debug("[VTS] %-45s  vts=%.4f  std=%.4f — high variance, skipped",
                                 table_id, vts, vts_std)
                    continue
            else:
                vts = float(roc_auc_score(y_val, proba_val[:, 1]))

        except Exception:
            continue

        disc = max(float(discovery_scores.get(table_id, 1e-3)), 1e-6)
        combined = vts * float(np.sqrt(disc))
        if vts >= effective_min_vts:
            validated[table_id] = combined
            logger.debug("[VTS] %-45s  vts=%.4f  disc=%.4f  combined=%.4f  ✓",
                         table_id, vts, disc, combined)
        else:
            logger.debug("[VTS] %-45s  vts=%.4f  ✗ (below %.2f)", table_id, vts, effective_min_vts)

    logger.info("[VTS] %d/%d sources pass min_vts=%.2f (effective=%.2f)",
                len(validated), len(aligned), min_vts, effective_min_vts)
    return validated


def train_vla(
    validated: dict[str, float],
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    target: pd.DataFrame,
    label_col: str,
    val_boost: float = 3.0,
    weight_power: float = 2.0,
    **xgb_kwargs,
) -> AdaptationResult:
    """
    Validated Lake Adaptation (VLA): pool validated sources + labeled val set.

    Val rows receive weight = val_boost × (max per-source weight), so the small
    labeled set has a meaningful influence without dominating the lake signal.

    If no sources pass validation, falls back to training on val only.
    """
    frames_X: list[pd.DataFrame] = []
    frames_y: list[pd.Series]    = []
    weights:  list[np.ndarray]   = []

    if validated:
        valid_aligned = {k: aligned[k] for k in validated if k in aligned}
        # Use combined VTS×sqrt(discovery) scores from validate_sources for weighting.
        # Fall back to raw discovery score if validated score is missing.
        raw = {k: validated.get(k, discovery_scores.get(k, 1.0)) ** weight_power
               for k in valid_aligned}
        total = sum(raw.values()) or 1.0
        for table_id, df in valid_aligned.items():
            X, y = _split_xy(df, label_col)
            w = raw[table_id] / total
            frames_X.append(X)
            frames_y.append(y)
            weights.append(np.full(len(X), w))
        max_src_w = max(raw.values()) / total
    else:
        logger.warning("[VLA] No validated sources — training on labeled val only.")
        max_src_w = 1.0

    # Add labeled val rows at boosted weight
    val_w = val_boost * max_src_w
    frames_X.append(X_val)
    frames_y.append(pd.Series(y_val, name=label_col))
    weights.append(np.full(len(X_val), val_w))

    X_all = pd.concat(frames_X, ignore_index=True)
    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    y_all = pd.concat(frames_y, ignore_index=True)
    w_all = np.concatenate(weights)

    model = _make_xgb(**xgb_kwargs)
    model.fit(X_all, y_all, sample_weight=w_all)
    proba = model.predict_proba(target)
    return AdaptationResult(
        level="vla",
        predictions=model.predict(target),
        probabilities=proba,
        model=model,
    )


def train_vla_self_train(
    vla_result: AdaptationResult,
    X_unlab: pd.DataFrame,
    target: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    label_col: str,
    pseudo_weight: float = 0.5,
    n_rounds: int = 3,
    **xgb_kwargs,
) -> AdaptationResult:
    """
    Iterative self-training on top of VLA (up to n_rounds).

    Each round expands the pseudo-label budget (30 % → 40 % → 50 % …) and
    retrains on labeled val + pseudo-labeled unlabeled rows.  Early stopping
    fires when val AUC degrades.
    """
    if vla_result.probabilities is None or vla_result.model is None:
        return AdaptationResult(
            level="vla_st",
            predictions=vla_result.predictions.copy(),
            probabilities=vla_result.probabilities,
        )

    best_model  = vla_result.model
    best_proba  = vla_result.probabilities
    best_val_auc: Optional[float] = None

    for round_idx in range(n_rounds):
        pseudo_pct = min(0.30 + 0.10 * round_idx, 0.80)
        proba_unlab = best_model.predict_proba(X_unlab)
        confidence  = proba_unlab.max(axis=1)
        thr = np.percentile(confidence, 100.0 * (1.0 - pseudo_pct))
        pseudo_mask = confidence >= thr
        n_pseudo = int(pseudo_mask.sum())
        logger.info("[VLA-ST] Round %d  pseudo_pct=%.0f%%  n_pseudo=%d",
                    round_idx + 1, pseudo_pct * 100, n_pseudo)

        if n_pseudo < 10:
            logger.info("[VLA-ST] Too few pseudo-labels — stopping.")
            break

        pseudo_classes = best_model.classes_[proba_unlab[pseudo_mask].argmax(axis=1)]
        if len(np.unique(pseudo_classes)) < 2:
            logger.info("[VLA-ST] All pseudo-labels one class — stopping.")
            break

        X_pseudo = X_unlab[pseudo_mask].copy()
        y_pseudo = pd.Series(pseudo_classes, name=label_col)
        X_aug = pd.concat([X_val, X_pseudo], ignore_index=True).replace([np.inf, -np.inf], np.nan)
        y_aug = pd.concat([pd.Series(y_val, name=label_col), y_pseudo], ignore_index=True)
        w_aug = np.concatenate([np.ones(len(X_val)), np.full(n_pseudo, pseudo_weight)])

        new_model = _make_xgb(**xgb_kwargs)
        new_model.fit(X_aug, y_aug, sample_weight=w_aug)

        # Early stopping on val AUC (pseudo-labels as proxy ground truth)
        if len(y_val) >= 10 and len(np.unique(y_val)) >= 2:
            try:
                val_auc = float(roc_auc_score(y_val, new_model.predict_proba(X_val)[:, 1]))
                logger.info("[VLA-ST] Round %d val AUC: %.4f", round_idx + 1, val_auc)
                if best_val_auc is not None and val_auc < best_val_auc - 0.01:
                    logger.info("[VLA-ST] Early stop: AUC degraded (%.4f < %.4f)",
                                val_auc, best_val_auc)
                    break
                if best_val_auc is None or val_auc >= best_val_auc:
                    best_val_auc = val_auc
                    best_model   = new_model
                    best_proba   = new_model.predict_proba(target)
            except Exception:
                best_model = new_model
                best_proba = new_model.predict_proba(target)
        else:
            best_model = new_model
            best_proba = new_model.predict_proba(target)

    return AdaptationResult(
        level="vla_st",
        predictions=best_model.predict(target),
        probabilities=best_proba,
        model=best_model,
    )


def train_vla_finetune(
    validated: dict[str, float],
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    target: pd.DataFrame,
    label_col: str,
    weight_power: float = 2.0,
    n_finetune_trees: int = 50,
    **xgb_kwargs,
) -> AdaptationResult:
    """
    Transfer-learning VLA: train XGBoost on validated sources, then fine-tune
    by appending trees trained on the labeled val rows (incremental boosting).

    This preserves source knowledge while steering final predictions toward the
    target distribution with minimal labeled data.
    """
    if not validated:
        return train_vla(validated, aligned, discovery_scores, X_val, y_val, target,
                         label_col, weight_power=weight_power, **xgb_kwargs)

    # Phase 1: train on validated sources
    frames_X: list[pd.DataFrame] = []
    frames_y: list[pd.Series]    = []
    weights:  list[np.ndarray]   = []
    valid_aligned = {k: aligned[k] for k in validated if k in aligned}
    raw = {k: validated.get(k, discovery_scores.get(k, 1.0)) ** weight_power
           for k in valid_aligned}
    total = sum(raw.values()) or 1.0
    for table_id, df in valid_aligned.items():
        X, y = _split_xy(df, label_col)
        w = raw[table_id] / total
        frames_X.append(X)
        frames_y.append(y)
        weights.append(np.full(len(X), w))

    X_src = pd.concat(frames_X, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    y_src = pd.concat(frames_y, ignore_index=True)
    w_src = np.concatenate(weights)

    src_model = _make_xgb(**xgb_kwargs)
    src_model.fit(X_src, y_src, sample_weight=w_src)

    # Phase 2: fine-tune with val rows (incremental boosting)
    X_val_clean = X_val.replace([np.inf, -np.inf], np.nan)
    if len(np.unique(y_val)) < 2:
        proba = src_model.predict_proba(target)
        return AdaptationResult("vla_ft", src_model.predict(target), proba, src_model)

    ft_kwargs = {**xgb_kwargs}
    ft_kwargs["n_estimators"] = n_finetune_trees
    ft_model = _make_xgb(**ft_kwargs)
    ft_model.fit(X_val_clean, y_val, xgb_model=src_model)
    proba = ft_model.predict_proba(target)
    return AdaptationResult("vla_ft", ft_model.predict(target), proba, ft_model)


def run_label_propagation(
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_unlab: pd.DataFrame,
    target: pd.DataFrame,
    label_col: str,
    gamma: float = 0.25,
    alpha: float = 0.2,
) -> AdaptationResult:
    """
    Semi-supervised label spreading on labeled val + unlabeled target train.

    Builds a graph over all training points (labeled + unlabeled), weighted by
    RBF kernel similarity, and propagates the known labels through the graph.
    Unlike self-training, this uses the actual manifold structure of the data
    without needing a confident threshold — very effective at ultra-low fractions.
    """
    from sklearn.semi_supervised import LabelSpreading
    from sklearn.preprocessing import StandardScaler

    X_all = pd.concat([X_val, X_unlab], ignore_index=True)
    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    col_means = X_all.mean()
    X_all = X_all.fillna(col_means)

    y_lp = np.concatenate([
        y_val.astype(int),
        -1 * np.ones(len(X_unlab), dtype=int),
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all.values)

    lp = LabelSpreading(kernel="rbf", gamma=gamma, alpha=alpha, max_iter=200)
    try:
        lp.fit(X_scaled, y_lp)
    except Exception as exc:
        logger.warning("[LabelProp] LabelSpreading failed: %s — falling back to uniform.", exc)
        proba = np.full((len(target), 2), 0.5)
        return AdaptationResult("vla_lp", (proba[:, 1] >= 0.5).astype(int), proba)

    target_clean = target.replace([np.inf, -np.inf], np.nan).fillna(col_means)
    X_target_scaled = scaler.transform(target_clean.values)

    try:
        proba = lp.predict_proba(X_target_scaled)
        if proba.shape[1] == 1:
            proba = np.column_stack([1.0 - proba, proba])
    except Exception as exc:
        logger.warning("[LabelProp] predict_proba failed: %s — falling back to uniform.", exc)
        proba = np.full((len(target), 2), 0.5)

    preds = (proba[:, 1] >= 0.5).astype(int)
    logger.info("[LabelProp] Label spreading done: %d labeled, %d unlabeled",
                len(y_val), len(X_unlab))
    return AdaptationResult("vla_lp", preds, proba)


# ---------------------------------------------------------------------------
# Cold-start helpers (used by act7_cold_start.py)
# ---------------------------------------------------------------------------

def compute_blend_weight(
    n_labeled: int,
    discovery_scores: dict[str, float],
    aligned: dict[str, pd.DataFrame],
    n_eff_cap: int = 1000,
) -> float:
    """
    Compute α = n_labeled / (n_labeled + n_eff_src).

    n_eff_src = Σ_k score_k × min(|aligned_k|, n_eff_cap)

    α ≈ 0  → trust the lake;  α ≈ 1  → trust target labels.
    """
    n_eff_src = sum(
        discovery_scores.get(k, 0.0) * min(len(df), n_eff_cap)
        for k, df in aligned.items()
    )
    if n_labeled + n_eff_src == 0:
        return 0.0
    return float(n_labeled / (n_labeled + n_eff_src))


def train_blended(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    target: pd.DataFrame,
    label_col: str,
    alpha: float,
    weight_power: float = 2.0,
    **xgb_kwargs,
) -> AdaptationResult:
    """
    Blend lake probabilities and target-only probabilities by α.

    p_blend = (1 - α) × p_lake + α × p_target

    If fewer than 2 classes in y_labeled, falls back to the lake model.
    """
    # Lake model (score-weighted)
    X_src, y_src, w_src = _pool_sources(
        aligned, discovery_scores, label_col, weight_power=weight_power
    )
    X_src = X_src.replace([np.inf, -np.inf], np.nan)
    lake_model = _make_xgb(**xgb_kwargs)
    lake_model.fit(X_src, y_src, sample_weight=w_src)
    p_lake = lake_model.predict_proba(target)

    if len(np.unique(y_labeled)) < 2:
        logger.info("[Blend] Only one class in labeled set — using lake model only (α=0).")
        return AdaptationResult(
            level="blended",
            predictions=lake_model.predict(target),
            probabilities=p_lake,
            model=lake_model,
        )

    # Target-only model
    X_lab = X_labeled.replace([np.inf, -np.inf], np.nan)
    tgt_model = _make_xgb(**xgb_kwargs)
    tgt_model.fit(X_lab, y_labeled)
    p_tgt = tgt_model.predict_proba(target)

    # Soft blend
    # Align column order: both models trained on same target columns
    n_classes = max(p_lake.shape[1], p_tgt.shape[1])
    if p_lake.shape[1] < n_classes:
        p_lake = np.hstack([p_lake, np.zeros((len(target), n_classes - p_lake.shape[1]))])
    if p_tgt.shape[1] < n_classes:
        p_tgt = np.hstack([p_tgt, np.zeros((len(target), n_classes - p_tgt.shape[1]))])

    p_blend = (1.0 - alpha) * p_lake + alpha * p_tgt
    return AdaptationResult(
        level="blended",
        predictions=p_blend.argmax(axis=1),
        probabilities=p_blend,
        model=None,
    )


# ---------------------------------------------------------------------------
# Act 6: Lake as stacking features (C1)
# ---------------------------------------------------------------------------

def run_stacking(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    target: pd.DataFrame,
    label_col: str,
    max_lake_features: int = 5,
) -> AdaptationResult:
    """
    Convert lake knowledge into supplementary features for the target model.

    Each aligned source model's predict_proba on target rows becomes one feature.
    The top-max_lake_features are selected by absolute correlation with y_labeled.
    A final XGBoost is trained on (target features + selected lake opinions) using
    only the small labeled target set.

    This respects the asymmetry: target features are canonical; lake knowledge
    is supplementary — and XGBoost can prune useless lake features automatically.
    """
    logger.info("[Stacking] Building lake opinion features from %d sources ...", len(aligned))

    # Combine labeled + test rows so lake models score all target rows at once
    all_target_rows = pd.concat([X_labeled, target], ignore_index=True)
    n_labeled = len(X_labeled)

    lake_opinions: dict[str, np.ndarray] = {}
    for src_id, src_df in aligned.items():
        X_src, y_src = _split_xy(src_df, label_col)
        if y_src.nunique() < 2 or len(X_src) < 20:
            continue
        m = _make_xgb(n_estimators=100, max_depth=3)
        try:
            m.fit(X_src, y_src)
            prob = m.predict_proba(all_target_rows)[:, 1]
            lake_opinions[f"lake_{src_id}"] = prob
        except Exception:
            continue

    if not lake_opinions:
        logger.warning("[Stacking] No lake opinions — falling back to target-only")
        m = _make_xgb()
        m.fit(X_labeled, y_labeled)
        proba = m.predict_proba(target)
        return AdaptationResult("stacking", m.predict(target), proba, m)

    lake_df = pd.DataFrame(lake_opinions)
    lake_labeled = lake_df.iloc[:n_labeled]
    lake_test    = lake_df.iloc[n_labeled:]

    # Select top-k lake features by absolute correlation with labels
    corr = lake_labeled.corrwith(pd.Series(y_labeled, index=lake_labeled.index)).abs()
    top_cols = corr.nlargest(max_lake_features).index.tolist()
    logger.info("[Stacking] Selected %d lake opinion features (from %d sources)",
                len(top_cols), len(lake_opinions))

    X_aug      = pd.concat([X_labeled.reset_index(drop=True),
                            lake_labeled[top_cols].reset_index(drop=True)], axis=1)
    X_test_aug = pd.concat([target.reset_index(drop=True),
                            lake_test[top_cols].reset_index(drop=True)], axis=1)

    model = _make_xgb()
    model.fit(X_aug, y_labeled)
    proba = model.predict_proba(X_test_aug)
    return AdaptationResult("stacking", model.predict(X_test_aug), proba, model)


# ---------------------------------------------------------------------------
# Act 6: Smart routing — decide whether the lake helps for this specific target
# ---------------------------------------------------------------------------

def route_lake_decision(
    aligned: dict[str, pd.DataFrame],
    discovery_scores: dict[str, float],
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    target: pd.DataFrame,
    label_col: str,
) -> tuple[str, "AdaptationResult"]:
    """
    Decide whether the lake improves over target-only training, using the
    validation set.

    Logic:
    - Validate sources via VTS (uses existing robustness rules for tiny n_val).
    - If >= 2 sources pass with mean combined score > 0.08 → use VLA.
    - Otherwise → fall back to target-only (lake is noise for this target).

    This prevents the lake from hurting (e.g. churn scenario) while preserving
    gains where it helps (e.g. heart +8 pts).

    Returns
    -------
    (reason, result) where reason is one of:
        "lake_helps"       — VLA used
        "target_only"      — fell back to target-only training
        "insufficient_classes" — val set has only one class; VLA used as default
    """
    if len(np.unique(y_val)) < 2:
        # Can't evaluate either approach on one class; default to VLA
        validated = validate_sources(aligned, discovery_scores, X_val, y_val, label_col)
        result = train_vla(validated, aligned, discovery_scores,
                           X_val, y_val, target, label_col)
        return "insufficient_classes", result

    validated = validate_sources(aligned, discovery_scores, X_val, y_val, label_col)
    n_validated = len(validated)
    mean_score = float(np.mean(list(validated.values()))) if validated else 0.0

    use_lake = n_validated >= 2 and mean_score > 0.08

    if use_lake:
        logger.info(
            "[Router] Lake helps: %d validated sources, mean_score=%.3f → using VLA",
            n_validated, mean_score,
        )
        result = train_vla(validated, aligned, discovery_scores,
                           X_val, y_val, target, label_col)
        return "lake_helps", result
    else:
        logger.info(
            "[Router] Lake noise: %d validated sources, mean_score=%.3f → target-only",
            n_validated, mean_score,
        )
        m = _make_xgb()
        m.fit(X_val, y_val)
        proba = m.predict_proba(target)
        return "target_only", AdaptationResult(
            "routed_target_only",
            m.predict(target),
            proba,
            m,
        )
