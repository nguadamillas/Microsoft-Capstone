"""
pipeline/run_pipeline.py
─────────────────────────
Runs the full pipeline end-to-end:
    Ingest → Bronze → Silver → Gold

Usage:
    python -m pipeline.run_pipeline            # full run
    python -m pipeline.run_pipeline --skip-ingest   # skip download (data already in raw/)
    python -m pipeline.run_pipeline --only gold     # run only the gold step
"""
import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def timed(label: str, fn):
    print(f"\n{'═'*55}")
    print(f"  STEP: {label}")
    print(f"{'═'*55}")
    t0 = time.time()
    fn()
    elapsed = time.time() - t0
    print(f"\n  ⏱  {label} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run the TED procurement pipeline.")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip download step (raw XML already present)")
    parser.add_argument("--only", choices=["ingest", "bronze", "silver", "gold"],
                        help="Run only one step")
    args = parser.parse_args()

    from pipeline.ingest import run as ingest
    from pipeline.bronze import run as bronze
    from pipeline.silver import run as silver
    from pipeline.gold   import run as gold

    steps = {
        "ingest": ingest,
        "bronze": bronze,
        "silver": silver,
        "gold":   gold,
    }

    if args.only:
        timed(args.only, steps[args.only])
    else:
        if not args.skip_ingest:
            timed("Ingest", ingest)
        timed("Bronze", bronze)
        timed("Silver", silver)
        timed("Gold",   gold)

    print(f"\n{'═'*55}")
    print(f"  ✓ Pipeline complete")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
