# Second Pass Generation Plan

## Goal

Run data generation a second time at a higher temperature to produce diverse alternative solutions to the same prompts, doubling the distilled dataset from ~4,569 to ~9,138 samples.

## Why

- Same prompt + different temperature = structurally different code solutions (different variable names, algorithms, patterns)
- This diversity helps the student model generalize rather than memorize specific patterns
- It's the cheapest way to double the dataset — no new prompts, no new expansion, just different sampling

## Temperature Strategy

| Teacher | Run 1 (current) | Run 2 (second pass) |
|---------|-----------------|---------------------|
| DeepSeek V3.2 | 0.7 | 0.9 |
| Kimi K2.5 | 0.6 | 0.85 |
| MiniMax M2.7 | 0.4 | 0.7 |

Rationale: Each teacher's second-pass temperature is bumped ~0.25-0.3 higher than its default. MiniMax gets a bigger bump since its default is already low (0.4) and was intentionally set low to reduce verbosity — at 0.7 it will produce more varied output while still being reasonable.

## Changes Required

### 1. `generate_data.py` — Add `--temperature` CLI override

```python
# In the argparse section (~line 444), add:
parser.add_argument(
    "--temperature",
    type=float,
    default=None,
    help="Override temperature for all prompts (ignores YAML/teacher defaults)",
)
```

Then in `main_async` (~line 410), after building prompts, override their temperature:

```python
if args.temperature is not None:
    for prompt in prompts:
        prompt.temperature = args.temperature
    log.info(f"Temperature override: {args.temperature}")
```

### 2. `run_generation.py` — Add `--suffix` and `--temperature` flags

```python
parser.add_argument(
    "--suffix", type=str, default="",
    help="Suffix for output files (e.g., '_t2' for second pass)",
)
parser.add_argument(
    "--temperature", type=float, default=None,
    help="Override temperature for all teachers",
)
```

Then in `run_generate()`, pass the temperature through to the subprocess call:

```python
# Build the command
cmd = [
    sys.executable,
    str(SCRIPT_DIR / "generate_data.py"),
    "--config", str(config),
    "--output", str(output),  # output uses suffix in filename
    "--concurrency", str(concurrency),
]
if temperature is not None:
    cmd.extend(["--temperature", str(temperature)])
```

And in `TEACHERS` file mapping, append the suffix to output names:

```python
output = RAW_DIR / f"{output_name}{suffix}.jsonl"
```

### 3. `run_generation.py` — Update verify/format steps for suffixed files

The verify step needs to find both `fsharp_core.jsonl` and `fsharp_core_t2.jsonl`:

```python
# In run_verify(), iterate both base and suffixed files
for _, output_name in files:
    for s in ["", suffix]:
        raw_path = RAW_DIR / f"{output_name}{s}.jsonl"
        verified_path = VERIFIED_DIR / f"{output_name}{s}.jsonl"
        # ... verify or copy as before
```

### 4. Create `run_generation_pass2.bat`

```batch
@echo off
cd /d "%~dp0"
python run_generation.py --suffix _t2 --temperature 0.9 --verify %*
pause
```

Note: Using a single `--temperature 0.9` is simpler than per-teacher overrides. The per-teacher strategy in the table above is ideal but adds complexity. A flat 0.9 across all teachers is a reasonable simplification — we can always adjust later.

### 5. Deduplication (optional but recommended)

After both runs complete, deduplicate near-identical responses before formatting:

- Simple approach: hash the first 200 chars of each response, skip duplicates
- Better approach: use cosine similarity of TF-IDF vectors, reject pairs > 0.95 similarity
- Safest approach: skip dedup entirely — the temperature difference should produce sufficiently different outputs, and the model benefits from seeing multiple valid solutions

**Recommendation**: Skip dedup for now. If we see quality issues during training eval, add it later.

## Output File Layout

```
data/raw/
  fsharp_core.jsonl         # Run 1 (temp 0.7)
  fsharp_core_t2.jsonl      # Run 2 (temp 0.9)
  fsharp_libraries.jsonl
  fsharp_libraries_t2.jsonl
  svelte_typescript.jsonl
  svelte_typescript_t2.jsonl
  ... (same pattern for all 9 domains)

data/verified/
  fsharp_core.jsonl         # Run 1 verified
  fsharp_core_t2.jsonl      # Run 2 verified
  svelte_typescript.jsonl    # Copied (no F# verification needed)
  svelte_typescript_t2.jsonl
  opencode_instruct.jsonl    # OpenCodeInstruct (unchanged)
  ...
```

## ID Strategy

Run 2 samples need unique IDs to avoid collisions with run 1 during resume. The expanded YAML prompt IDs look like `fsharp_core_0001_exp_003`. For run 2, we have two options:

- **Option A**: Rely on output file separation — same IDs are fine since they're in different JSONL files. Resume logic checks per-file.
- **Option B**: Append `_t2` to all IDs in the second run.

**Recommendation**: Option A is fine. The resume logic in `generate_data.py` loads existing IDs from the specific output file, so `fsharp_core_t2.jsonl` starts with an empty ID set and there's no collision.

## Final Dataset Composition

| Source | Samples | % of total |
|--------|---------|------------|
| Run 1 (temp 0.7, domain-specific) | ~4,569 | 39% |
| Run 2 (temp 0.9, domain-specific) | ~4,569 | 39% |
| OpenCodeInstruct (strict filtered) | 2,500 | 22% |
| **Total** | **~11,638** | 100% |

## Estimated Time & Cost

| Phase | Time | Notes |
|-------|------|-------|
| Run 1 (current, in progress) | ~10h | 3 teachers parallel, concurrency 7 |
| Run 2 (second pass) | ~10h | Same parallelism, same prompts |
| F# verification (both runs) | ~1-2h | Compiler checks on F# domains |
| Formatting | ~5min | CPU only |
| **Total generation wall time** | **~20-22h** | |
| Cloud training (4 stages) | ~20-30h | 2×A100 or L40S |
| Cloud training cost | ~$65-100 | At $3.2/h for 2×A100 |

## Execution Steps

```bash
# 1. Wait for current run to finish (check with --status)
python run_generation.py --status

# 2. Apply the code changes above (generate_data.py, run_generation.py)

# 3. Run second pass
run_generation_pass2.bat

# 4. After both runs complete, verify + format covers everything
# (handled automatically by --verify flag in the bat)
```

## Risks

1. **Quality at higher temperature**: Some teachers may produce worse code at temp 0.9. The F# compiler verification catches bad F# code, but non-F# domains rely on teacher quality. Mitigation: inspect samples manually before training.

2. **Near-duplicate responses**: At only 0.2 temp difference, some responses will be very similar. This wastes training compute but doesn't hurt quality. Mitigation: acceptable for now.

3. **MiniMax verbosity**: MiniMax was set to 0.4 specifically to control verbosity. At 0.7 it may produce longer, more verbose responses. Mitigation: the system prompt still says "Be concise" which should help.
