"""
Parallel Training Data Generation Runner

Runs all 3 teachers (DeepSeek, Kimi, MiniMax) concurrently in a single process,
with interleaved log output. Each teacher processes its assigned files sequentially.

Usage:
    python run_generation.py                    # Generate only
    python run_generation.py --verify           # Generate + verify + format
    python run_generation.py --concurrency 8    # Custom concurrency per teacher
"""

import argparse
import asyncio
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
VERIFIED_DIR = PROJECT_DIR / "data" / "verified"
EXPANDED_DIR = SCRIPT_DIR.parent / "prompts" / "expanded"

# Teacher -> list of (expanded_yaml_stem, output_name)
TEACHERS = {
    "DeepSeek": [
        ("fsharp_core_expanded", "fsharp_core"),
        ("fsharp_libraries_expanded", "fsharp_libraries"),
        ("dotnet_aspnet_expanded", "dotnet_aspnet"),
        ("general_coding_expanded", "general_coding"),
    ],
    "Kimi": [
        ("svelte_typescript_expanded", "svelte_typescript"),
        ("cross_domain_expanded", "cross_domain"),
        ("long_context_expanded", "long_context"),
    ],
    "MiniMax": [
        ("docker_kubernetes_expanded", "docker_kubernetes"),
        ("agentic_swe_expanded", "agentic_swe"),
    ],
}

# Files that need F# verification
FSHARP_DOMAINS = {"fsharp_core", "fsharp_libraries", "dotnet_aspnet", "cross_domain"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def count_prompts(yaml_path: Path) -> int:
    """Count prompts in an expanded YAML without loading full file."""
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return len(data.get("prompts", []))


async def run_generate(config: Path, output: Path, concurrency: int, label: str):
    """Run generate_data.py as a subprocess with live output."""
    output.parent.mkdir(parents=True, exist_ok=True)

    existing = count_lines(output)
    total = count_prompts(config)
    remaining = total - existing

    if remaining <= 0:
        log.info(f"[{label}] Already complete ({existing}/{total} samples)")
        return

    log.info(f"[{label}] Starting: {remaining} remaining of {total} ({existing} done)")

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(SCRIPT_DIR / "generate_data.py"),
        "--config",
        str(config),
        "--output",
        str(output),
        "--concurrency",
        str(concurrency),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(SCRIPT_DIR),
    )

    async for line in proc.stdout:
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            print(f"[{label:8s}] {text}")

    await proc.wait()

    final_count = count_lines(output)
    if proc.returncode == 0:
        log.info(f"[{label}] Complete: {final_count} samples")
    else:
        log.warning(
            f"[{label}] Exited with code {proc.returncode}, {final_count} samples so far"
        )


async def run_teacher(teacher: str, files: list, concurrency: int):
    """Run all files for a single teacher sequentially."""
    log.info(f"=== {teacher} starting ({len(files)} files) ===")
    start = time.monotonic()

    for yaml_stem, output_name in files:
        config = EXPANDED_DIR / f"{yaml_stem}.yaml"
        output = RAW_DIR / f"{output_name}.jsonl"
        await run_generate(config, output, concurrency, f"{teacher[:4]}:{output_name}")

    elapsed = time.monotonic() - start
    log.info(f"=== {teacher} finished in {elapsed / 3600:.1f}h ===")


async def generate_all(concurrency: int):
    """Run all 3 teachers in parallel."""
    log.info("=" * 60)
    log.info("PARALLEL GENERATION: 3 teachers, concurrency=%d each", concurrency)
    log.info("=" * 60)

    # Count totals
    grand_total = 0
    for teacher, files in TEACHERS.items():
        teacher_total = 0
        for yaml_stem, output_name in files:
            config = EXPANDED_DIR / f"{yaml_stem}.yaml"
            n = count_prompts(config)
            teacher_total += n
        log.info(f"  {teacher}: {teacher_total} prompts across {len(files)} files")
        grand_total += teacher_total
    log.info(f"  TOTAL: {grand_total} prompts")
    log.info("=" * 60)

    start = time.monotonic()

    # Launch all 3 teachers concurrently
    await asyncio.gather(
        run_teacher("DeepSeek", TEACHERS["DeepSeek"], concurrency),
        run_teacher("Kimi", TEACHERS["Kimi"], concurrency),
        run_teacher("MiniMax", TEACHERS["MiniMax"], concurrency),
    )

    elapsed = time.monotonic() - start
    log.info("=" * 60)
    log.info(f"ALL GENERATION COMPLETE in {elapsed / 3600:.1f}h")

    # Print final counts
    total = 0
    for teacher, files in TEACHERS.items():
        for _, output_name in files:
            n = count_lines(RAW_DIR / f"{output_name}.jsonl")
            total += n
            log.info(f"  {output_name}: {n} samples")
    log.info(f"  TOTAL: {total} samples")
    log.info("=" * 60)


def run_verify():
    """Run F# verification on applicable domains."""
    log.info("=" * 60)
    log.info("F# VERIFICATION")
    log.info("=" * 60)

    VERIFIED_DIR.mkdir(parents=True, exist_ok=True)

    for teacher, files in TEACHERS.items():
        for _, output_name in files:
            raw_path = RAW_DIR / f"{output_name}.jsonl"
            verified_path = VERIFIED_DIR / f"{output_name}.jsonl"

            if not raw_path.exists():
                log.warning(f"  {output_name}: raw file missing, skipping")
                continue

            if output_name in FSHARP_DOMAINS:
                log.info(f"  {output_name}: verifying F# samples...")
                result = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "verify_fsharp.py"),
                        "--input",
                        str(raw_path),
                        "--output",
                        str(verified_path),
                    ],
                    cwd=str(SCRIPT_DIR),
                )
                if result.returncode != 0:
                    log.warning(f"  {output_name}: verification had errors")
            else:
                log.info(f"  {output_name}: copying (no F# verification needed)")
                shutil.copy2(raw_path, verified_path)


def run_format():
    """Format verified data for training."""
    log.info("=" * 60)
    log.info("FORMATTING DATASET")
    log.info("=" * 60)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "format_dataset.py"),
            "--input",
            str(VERIFIED_DIR),
            "--output",
            str(PROJECT_DIR / "data" / "formatted"),
            "--format",
            "chatml",
            "--split-by-length",
        ],
        cwd=str(SCRIPT_DIR),
    )


def print_progress():
    """Print current progress across all domains."""
    total = 0
    print(f"\n{'Domain':<30s} {'Samples':>8s}")
    print("-" * 40)
    for teacher, files in TEACHERS.items():
        for _, output_name in files:
            n = count_lines(RAW_DIR / f"{output_name}.jsonl")
            total += n
            print(f"  {output_name:<28s} {n:>8d}")
    print("-" * 40)
    print(f"  {'TOTAL':<28s} {total:>8d}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Parallel Training Data Generation")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=7,
        help="Concurrent requests per teacher (default: 7, 21 total across 3 teachers)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Also run verification and formatting after generation",
    )
    parser.add_argument(
        "--status", action="store_true", help="Just print current progress and exit"
    )
    args = parser.parse_args()

    if args.status:
        print_progress()
        return

    asyncio.run(generate_all(args.concurrency))

    if args.verify:
        run_verify()
        run_format()

    print_progress()
    log.info("Done! Run with --verify to also verify F# and format the dataset.")


if __name__ == "__main__":
    main()
