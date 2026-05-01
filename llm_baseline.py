"""
Zero-shot LLM classification baseline using local Ollama.

Runs the target test set through a local LLM (qwen3.5 by default) with a
zero-shot prompt, producing binary predictions and confidence scores.

For large test sets (N > sample_n), a random sample is scored and the
remaining rows receive neutral probability [0.5, 0.5].

Usage
-----
    from llm_baseline import run_zero_shot
    result = run_zero_shot(target_df, label_col="label", label_description="income above 50k")
"""

import json
import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
import requests

from domain_adaptation import AdaptationResult

_OLLAMA_BASE_URL = "http://localhost:11434"
logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    'You are a binary classifier. Task: predict whether this record indicates "{label_description}".\n'
    "Data: {row_text}\n\n"
    'Respond ONLY with a JSON object with two keys:\n'
    '  "prediction": 0 or 1 (1 = yes, 0 = no)\n'
    '  "confidence": your confidence score between 0.0 and 1.0\n'
    "JSON response:"
)


def _format_row(row: pd.Series, max_cols: int = 20, max_val_len: int = 50) -> str:
    """Format a row as 'col: val' pairs, truncating long values."""
    parts = []
    for col, val in row.items():
        val_str = str(val)[:max_val_len]
        parts.append(f"{col}: {val_str}")
        if len(parts) >= max_cols:
            break
    return "; ".join(parts)


def _call_ollama(prompt: str, model: str, timeout: int = 90) -> Optional[str]:
    """
    POST to Ollama /api/chat with think=False (disables qwen3 chain-of-thought).
    Returns the response text or None on error.
    """
    try:
        resp = requests.post(
            f"{_OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as exc:
        logger.debug("Ollama call failed: %s", exc)
        return None


def _parse_response(text: str) -> tuple[int, float]:
    """
    Parse LLM response into (prediction, confidence).

    Handles:
      - Clean JSON: {"prediction": 1, "confidence": 0.8}
      - JSON in markdown code block: ```json { ... } ```
      - JSON embedded after <think>...</think> reasoning blocks
      - Fallback: regex for yes/no/1/0 keywords
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # If qwen3 wraps in <think>...</think>, try post-think text first
    think_end = text.rfind("</think>")
    candidates = []
    if think_end != -1:
        candidates.append(text[think_end + len("</think>"):].strip())
    candidates.append(text)  # always try full text as fallback

    for search_text in candidates:
        # Match JSON object (allow whitespace/newlines inside)
        json_match = re.search(r"\{[^{}]*\}", search_text, re.DOTALL)
        if json_match:
            try:
                obj = json.loads(json_match.group())
                pred = int(obj.get("prediction", 0))
                conf = float(obj.get("confidence", 0.6))
                pred = max(0, min(1, pred))
                conf = max(0.0, min(1.0, conf))
                return pred, conf
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

    # Keyword fallback on full text
    lower = text.lower()
    if any(kw in lower for kw in ("yes", '"prediction": 1', "prediction: 1", "true")):
        return 1, 0.6
    return 0, 0.6


def run_zero_shot(
    target_df: pd.DataFrame,
    label_col: str,
    label_description: str,
    ollama_model: str = "qwen3.5:latest",
    sample_n: int = 60,
) -> AdaptationResult:
    """
    Zero-shot LLM classification baseline.

    Scores each row in target_df by prompting a local Ollama model. For large
    datasets (N > sample_n) only a random sample is scored; the remaining rows
    receive neutral probability [0.5, 0.5].

    Parameters
    ----------
    target_df         : feature DataFrame for the test set (label_col stripped if present)
    label_col         : name of the label column (used only for stripping)
    label_description : human-readable description of the positive class
    ollama_model      : Ollama model tag (must be pulled locally)
    sample_n          : maximum rows to score; set to 0 for full set

    Returns
    -------
    AdaptationResult with level="llm_zero_shot"
    """
    try:
        return _run_zero_shot_impl(
            target_df, label_col, label_description, ollama_model, sample_n
        )
    except Exception as exc:
        logger.warning("run_zero_shot: unexpected failure (%s) — returning neutral result", exc)
        n = len(target_df)
        return AdaptationResult(
            level="llm_zero_shot",
            predictions=np.zeros(n, dtype=int),
            probabilities=np.full((n, 2), 0.5),
            model=None,
        )


def _run_zero_shot_impl(
    target_df: pd.DataFrame,
    label_col: str,
    label_description: str,
    ollama_model: str,
    sample_n: int,
) -> AdaptationResult:
    # Strip label column if present
    df = target_df.drop(columns=[label_col], errors="ignore")
    n = len(df)

    # Determine which rows to score
    if sample_n > 0 and n > sample_n:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n, size=sample_n, replace=False)
        logger.info(
            "[LLM zero-shot] Sampling %d/%d rows (neutral prob for the rest)", sample_n, n
        )
    else:
        sample_idx = np.arange(n)

    proba = np.full((n, 2), 0.5)
    n_parse_fail = 0

    for i, idx in enumerate(sample_idx):
        row = df.iloc[idx]
        row_text = _format_row(row)
        prompt = _PROMPT_TEMPLATE.format(
            label_description=label_description,
            row_text=row_text,
        )
        text = _call_ollama(prompt, ollama_model)
        if text is None:
            n_parse_fail += 1
            continue

        pred, conf = _parse_response(text)
        if pred == 1:
            proba[idx, 1] = conf
            proba[idx, 0] = 1.0 - conf
        else:
            proba[idx, 0] = conf
            proba[idx, 1] = 1.0 - conf

        if (i + 1) % 50 == 0:
            logger.info(
                "[LLM zero-shot] Scored %d/%d rows (parse failures so far: %d)",
                i + 1, len(sample_idx), n_parse_fail,
            )

    if n_parse_fail > 0:
        logger.warning(
            "[LLM zero-shot] %d/%d rows had parse failures (kept neutral 0.5)",
            n_parse_fail, len(sample_idx),
        )

    predictions = (proba[:, 1] >= 0.5).astype(int)
    logger.info(
        "[LLM zero-shot] Done. Predicted positive rate: %.3f",
        predictions.mean(),
    )
    return AdaptationResult(
        level="llm_zero_shot",
        predictions=predictions,
        probabilities=proba,
        model=None,
    )
