"""
OpenCodeInstruct Downloader & Filter

Downloads NVIDIA's OpenCodeInstruct dataset (5M Python coding samples),
applies strict quality filters, and outputs a random subsample in our
pipeline's JSONL format for mixing into training data.

Filters applied:
  - average_test_score >= 0.9 (passes 9+ of 10 generated unit tests)
  - LLM judgement requirement_conformance score >= 4
  - LLM judgement logical_correctness score >= 4
  - LLM judgement edge_case_consideration score >= 4

Usage:
    python download_opencode.py                          # Default: 2500 samples
    python download_opencode.py --samples 5000           # Custom sample count
    python download_opencode.py --output custom_path.jsonl
    python download_opencode.py --min-test-score 1.0     # Only perfect test scores
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "verified" / "opencode_instruct.jsonl"

# Dataset source
DATASET_NAME = "nvidia/OpenCodeInstruct"

# The model that generated the solutions in OpenCodeInstruct
SOURCE_MODEL = "Qwen2.5-Coder-32B-Instruct"


def parse_judgement(judgement_str: str) -> dict:
    """Parse the LLM judgement JSON string and extract scores."""
    try:
        judgement = json.loads(judgement_str)
        return {
            "requirement_conformance": judgement.get("requirement_conformance", {}).get(
                "score", 0
            ),
            "logical_correctness": judgement.get("logical_correctness", {}).get(
                "score", 0
            ),
            "edge_case_consideration": judgement.get("edge_case_consideration", {}).get(
                "score", 0
            ),
        }
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {
            "requirement_conformance": 0,
            "logical_correctness": 0,
            "edge_case_consideration": 0,
        }


def passes_filters(
    sample: dict,
    min_test_score: float = 0.9,
    min_judgement_score: int = 4,
) -> bool:
    """Check if a sample passes all quality filters."""
    # Filter 1: Unit test pass rate
    try:
        test_score = float(sample.get("average_test_score", 0))
    except (ValueError, TypeError):
        return False

    if test_score < min_test_score:
        return False

    # Filter 2: LLM judgement scores
    scores = parse_judgement(sample.get("llm_judgement", "{}"))

    if scores["requirement_conformance"] < min_judgement_score:
        return False
    if scores["logical_correctness"] < min_judgement_score:
        return False
    if scores["edge_case_consideration"] < min_judgement_score:
        return False

    # Filter 3: Must have non-empty input and output
    if not sample.get("input", "").strip():
        return False
    if not sample.get("output", "").strip():
        return False

    return True


def convert_to_pipeline_format(sample: dict, index: int) -> dict:
    """Convert an OpenCodeInstruct sample to our pipeline's JSONL format."""
    return {
        "id": f"oci_{index:05d}",
        "instruction": sample["input"].strip(),
        "response": sample["output"].strip(),
        "teacher": "nvidia-oci",
        "domain": "general_coding",
        "model": SOURCE_MODEL,
        "generation_time_s": 0,
        "token_count": 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main():
    parser = argparse.ArgumentParser(description="Download and filter OpenCodeInstruct")
    parser.add_argument(
        "--samples",
        type=int,
        default=2500,
        help="Number of samples to include in the output (default: 2500)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--min-test-score",
        type=float,
        default=0.9,
        help="Minimum average_test_score threshold (default: 0.9)",
    )
    parser.add_argument(
        "--min-judgement-score",
        type=int,
        default=4,
        help="Minimum LLM judgement score for all three criteria (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("OpenCodeInstruct Download & Filter")
    log.info(f"  Dataset:          {DATASET_NAME}")
    log.info(f"  Target samples:   {args.samples}")
    log.info(f"  Min test score:   {args.min_test_score}")
    log.info(f"  Min judgement:    {args.min_judgement_score}")
    log.info(f"  Output:           {args.output}")
    log.info(f"  Random seed:      {args.seed}")
    log.info("=" * 60)

    # Import datasets here so it fails fast with a clear message
    try:
        from datasets import load_dataset
    except ImportError:
        log.error(
            "The 'datasets' package is required. Install with: pip install datasets"
        )
        return

    log.info("Loading dataset (streaming mode to avoid downloading all 5M samples)...")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    # Stream through and collect samples that pass filters
    filtered = []
    total_scanned = 0
    start_time = time.monotonic()

    log.info("Scanning and filtering samples...")
    for sample in dataset:
        total_scanned += 1

        if passes_filters(sample, args.min_test_score, args.min_judgement_score):
            filtered.append(sample)

        # Progress logging
        if total_scanned % 100_000 == 0:
            elapsed = time.monotonic() - start_time
            rate = total_scanned / elapsed
            log.info(
                f"  Scanned {total_scanned:,} samples, "
                f"{len(filtered):,} passed filters "
                f"({len(filtered) / total_scanned * 100:.1f}%), "
                f"{rate:.0f} samples/sec"
            )

        # Early exit: if we have enough filtered samples, we can stop early
        # We need at least 10x the target to get a good random sample
        # But cap at reasonable amount to avoid scanning all 5M if not needed
        if len(filtered) >= args.samples * 10:
            log.info(
                f"  Collected {len(filtered):,} filtered samples "
                f"(10x target), stopping scan early"
            )
            break

    elapsed = time.monotonic() - start_time
    log.info(f"Scan complete in {elapsed:.1f}s")
    log.info(f"  Total scanned:  {total_scanned:,}")
    log.info(
        f"  Passed filters: {len(filtered):,} ({len(filtered) / max(total_scanned, 1) * 100:.1f}%)"
    )

    if len(filtered) < args.samples:
        log.warning(
            f"Only {len(filtered)} samples passed filters, "
            f"fewer than the requested {args.samples}. "
            f"Using all {len(filtered)} samples."
        )
        selected = filtered
    else:
        # Random subsample
        random.seed(args.seed)
        selected = random.sample(filtered, args.samples)
        log.info(f"  Randomly selected {len(selected):,} samples (seed={args.seed})")

    # Convert and write
    log.info(f"Writing {len(selected)} samples to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        for i, sample in enumerate(selected):
            record = convert_to_pipeline_format(sample, i)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("=" * 60)
    log.info("DOWNLOAD SUMMARY")
    log.info(f"  Scanned:    {total_scanned:,} samples")
    log.info(f"  Filtered:   {len(filtered):,} samples")
    log.info(f"  Selected:   {len(selected):,} samples")
    log.info(f"  Output:     {args.output}")
    log.info(f"  File size:  {args.output.stat().st_size / 1024 / 1024:.1f} MB")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
