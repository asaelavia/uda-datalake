"""
Generate direction cache for source repurposing.

For each (target_label, proxy_col) pair found in existing done-caches,
ask the local Ollama LLM whether HIGHER values in the proxy column mean
MORE or LESS of the target concept.

Saves result to data/direction_cache.json:
  {"customer churn": {"active": "NEGATIVE", "churn": "POSITIVE", ...}, ...}

Usage
-----
    python generate_direction_cache.py [--targets churn heart credit ...]
    python generate_direction_cache.py --all
"""
import argparse
import json
import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL = "http://localhost:11434"
_MODEL = "qwen3.5:latest"
_CACHE_PATH = Path("data/direction_cache.json")
_DONE_CACHE_DIR = Path("data")

_TARGET_LABELS = {
    "adult":        "income above 50k",
    "heart":        "heart disease diagnosis",
    "churn":        "customer churn",
    "credit":       "credit risk good or bad",
    "diabetes":     "diabetes diagnosis positive",
    "bank":         "term deposit subscription",
    "nyhouse":      "house price above 1 million",
    "noshow":       "medical appointment no-show",
    "turnover":     "employee turnover",
    "obesity":      "county obesity high",
    "titanic":      "passenger survival titanic",
    "stroke":       "stroke diagnosis",
    "breastcancer": "breast cancer diagnosis malignant",
    "crime":        "violent crime rate high",
}

_PROMPT = """\
You are deciding whether a proxy column should be flipped when creating binary labels.

Target concept: "{target}"
Proxy column name: "{col}"

Does a HIGHER numeric value in this column mean MORE of the target concept?

Answer with exactly one word: POSITIVE (higher = more) or NEGATIVE (higher = less).

Examples:
- target="customer churn", col="churn" → POSITIVE
- target="customer churn", col="churned" → POSITIVE
- target="customer churn", col="active" → NEGATIVE  (active=1 means NOT churned)
- target="customer churn", col="is_active" → NEGATIVE
- target="customer churn", col="attrition" → POSITIVE
- target="heart disease diagnosis", col="healthy" → NEGATIVE  (healthy=1 means NO disease)
- target="heart disease diagnosis", col="heart_disease" → POSITIVE
- target="income above 50k", col="salary" → POSITIVE
- target="income above 50k", col="low_income" → NEGATIVE
- target="credit risk good", col="default" → NEGATIVE  (default=1 means BAD credit)
- target="credit risk good", col="credit_score" → POSITIVE

Now answer for:
target="{target}", col="{col}"
Answer:"""


def _call_ollama(prompt: str, timeout: int = 60) -> str | None:
    try:
        resp = requests.post(
            f"{_OLLAMA_BASE_URL}/api/chat",
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0},
                "think": False,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as exc:
        logger.debug("Ollama call failed: %s", exc)
        return None


def _parse_direction(text: str | None) -> str:
    if text is None:
        return "POSITIVE"
    t = text.upper().strip()
    if t.startswith("NEG"):
        return "NEGATIVE"
    return "POSITIVE"


def _collect_pairs(target_labels: dict[str, str]) -> dict[str, set[str]]:
    """Collect unique (target_label, proxy_col) pairs from all done-caches."""
    pairs: dict[str, set[str]] = {label: set() for label in target_labels.values()}

    for path in _DONE_CACHE_DIR.glob("repurpose_done_*.json"):
        try:
            data: dict[str, str] = json.loads(path.read_text())
        except Exception:
            continue
        # Match done-cache to a target by checking label slug in filename
        matched_label = None
        fname = path.stem.lower()
        for tname, label in target_labels.items():
            slug = label.replace(" ", "_").replace(">", "").replace("%", "").lower()
            short = tname.lower()
            if slug in fname or short in fname:
                matched_label = label
                break
        if matched_label is None:
            logger.debug("Could not match done-cache to target: %s", path.name)
            continue
        for col in data.values():
            pairs[matched_label].add(col.lower().strip())

    return pairs


def run(target_names: list[str]) -> None:
    target_labels = {k: v for k, v in _TARGET_LABELS.items() if k in target_names}
    if not target_labels:
        logger.error("No matching targets found.")
        return

    # Load existing cache
    existing: dict[str, dict[str, str]] = {}
    if _CACHE_PATH.exists():
        try:
            existing = json.loads(_CACHE_PATH.read_text())
            logger.info("Loaded existing direction cache: %d targets", len(existing))
        except Exception:
            pass

    pairs = _collect_pairs(target_labels)
    total_new = sum(
        1 for label, cols in pairs.items()
        for col in cols
        if col not in existing.get(label, {})
    )
    logger.info("New (target, col) pairs to score: %d", total_new)

    done = 0
    for label, cols in pairs.items():
        if label not in existing:
            existing[label] = {}
        for col in sorted(cols):
            if col in existing[label]:
                continue
            prompt = _PROMPT.format(target=label, col=col)
            raw = _call_ollama(prompt)
            direction = _parse_direction(raw)
            existing[label][col] = direction
            done += 1
            logger.info("[%d/%d] %-35s  col=%-25s → %s", done, total_new, label, col, direction)
            # Save after every entry (resumable)
            _CACHE_PATH.write_text(json.dumps(existing, indent=2))

    logger.info("Direction cache saved → %s  (%d targets, %d entries)",
                _CACHE_PATH, len(existing), sum(len(v) for v in existing.values()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=list(_TARGET_LABELS.keys()))
    parser.add_argument("--all", action="store_true", help="Run for all known targets")
    args = parser.parse_args()

    targets = list(_TARGET_LABELS.keys()) if args.all else args.targets
    run(targets)


if __name__ == "__main__":
    main()
