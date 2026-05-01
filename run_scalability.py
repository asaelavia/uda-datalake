"""
Scalability runner — runs act5 experiments for multiple lake sizes × targets.

For each (target, lake_size) pair that doesn't already have results, calls
run_experiment() and saves to results/act5/{target}_s{N}/.

After all runs complete, calls scalability_analysis.py to produce plots/tables.

Usage
-----
    python run_scalability.py
    python run_scalability.py --targets adult heart diabetes
    python run_scalability.py --targets adult --sizes 5000 25000 100000 421179
    python run_scalability.py --dry-run       # show what would run, don't execute
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_BASE  = Path("results/act5")
RUNTIME_FILE  = Path("results/scalability/runtimes.json")  # {"{target}_{size}": seconds}

# Lake sizes to sweep (421179 = full lake sentinel; actual full run uses no --lake-sample)
DEFAULT_SIZES = [5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 421_179]

# Targets chosen for diversity of difficulty and signal quality
DEFAULT_TARGETS = ["adult", "heart", "diabetes", "turnover", "bank"]


def results_exist(target: str, lake_size: int) -> bool:
    """Return True if metrics.csv already exists for this (target, lake_size) pair."""
    if lake_size == 421_179:
        p = RESULTS_BASE / target / "metrics.csv"
    else:
        p = RESULTS_BASE / f"{target}_s{lake_size}" / "metrics.csv"
    return p.exists()


def run_one(target: str, lake_size: int) -> bool:
    """
    Run a single experiment.  Returns True on success.

    Imports run_experiment lazily so logging is configured before the import.
    """
    from act5_gittables_lake import run_experiment

    kwargs = dict(target_name=target)
    if lake_size != 421_179:
        kwargs["lake_sample"] = lake_size

    label = f"{target} @ {'full' if lake_size == 421_179 else f'{lake_size:,}'}"
    logger.info("=" * 60)
    logger.info("Running: %s", label)
    logger.info("=" * 60)

    t0 = time.perf_counter()
    success = False
    try:
        result = run_experiment(**kwargs)
        if result is None:
            logger.warning("run_experiment returned None for %s — no sources found?", label)
        else:
            success = True
    except Exception as exc:
        logger.error("run_experiment failed for %s: %s", label, exc, exc_info=True)
    finally:
        elapsed = time.perf_counter() - t0
        _save_runtime(target, lake_size, elapsed)
        logger.info("Runtime for %s: %.1f s (%.1f min)", label, elapsed, elapsed / 60)
    return success


def _save_runtime(target: str, lake_size: int, seconds: float) -> None:
    """Append a runtime entry to runtimes.json."""
    RUNTIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    runtimes: dict = {}
    if RUNTIME_FILE.exists():
        try:
            runtimes = json.loads(RUNTIME_FILE.read_text())
        except Exception:
            pass
    key = f"{target}_{lake_size}"
    runtimes[key] = round(seconds, 1)
    RUNTIME_FILE.write_text(json.dumps(runtimes, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scalability sweep runner")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS,
                        choices=["adult", "heart", "credit", "diabetes", "bank",
                                 "turnover", "noshow", "nyhouse", "obesity",
                                 "titanic", "stroke"],
                        help="Targets to sweep (default: adult heart diabetes turnover bank)")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES,
                        help="Lake sizes to test. Use 421179 for the full lake.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results already exist.")
    parser.add_argument("--no-analysis", action="store_true",
                        help="Skip scalability_analysis.py after runs complete.")
    args = parser.parse_args()

    # Deduplicate and sort sizes
    sizes = sorted(set(args.sizes))

    # Build work list
    todo: list[tuple[str, int]] = []
    skip: list[tuple[str, int]] = []
    for target in args.targets:
        for n in sizes:
            if not args.force and results_exist(target, n):
                skip.append((target, n))
            else:
                todo.append((target, n))

    logger.info("Scalability sweep: %d targets × %d sizes = %d total",
                len(args.targets), len(sizes), len(args.targets) * len(sizes))
    logger.info("  To run : %d", len(todo))
    logger.info("  Skip   : %d (results exist)", len(skip))

    if skip:
        for t, n in skip:
            label = "full" if n == 421_179 else f"{n:,}"
            logger.info("  [skip] %s @ %s", t, label)

    if args.dry_run:
        print("\n--- DRY RUN — would execute: ---")
        for t, n in todo:
            label = "full" if n == 421_179 else f"{n:,}"
            print(f"  {t} @ {label}")
        return

    if not todo:
        logger.info("Nothing to run — all results exist. Use --force to re-run.")
    else:
        failed: list[tuple[str, int]] = []
        for i, (target, n) in enumerate(todo, 1):
            label = "full" if n == 421_179 else f"{n:,}"
            logger.info("[%d/%d] %s @ %s", i, len(todo), target, label)
            ok = run_one(target, n)
            if not ok:
                failed.append((target, n))

        if failed:
            logger.warning("Failed runs (%d):", len(failed))
            for t, n in failed:
                logger.warning("  %s @ %s", t, "full" if n == 421_179 else f"{n:,}")
        else:
            logger.info("All runs complete.")

    if not args.no_analysis:
        logger.info("Running scalability_analysis.py...")
        import scalability_analysis
        # Re-invoke main with the same targets
        sys.argv = ["scalability_analysis.py", "--targets"] + args.targets
        scalability_analysis.main()


if __name__ == "__main__":
    main()
