# 32GB GPU — Local Coding Model Guide

Recommended models and configuration for GPUs with 32GB VRAM (RTX 5090, etc.).

> See [README.md](README.md) for universal setup instructions (environment variables, quantization, KV cache, context size reference).

## Target Hardware

- **VRAM:** 32GB
- **Quantization:** Q4_K_M
- **KV Cache:** q8_0 (via `OLLAMA_KV_CACHE_TYPE`)
- **Inference:** Ollama

---

## Recommended Models

### Tier 1 — Best Choices

| Model | Type | Total / Active Params | Context | VRAM (Q4) | Best For |
|---|---|---|---|---|---|
| **Qwen3-Coder 30B-A3B** | MoE | 30B / 3.3B | 256K | ~18.6 GB | Agentic coding, tool calling (OpenCode, Aider, Cline) |
| **Qwen3 32B** | Dense | 32B / 32B | 32K (128K w/ YaRN) | ~20 GB | General + reasoning + coding (thinking/non-thinking modes) |
| **Qwen 2.5 Coder 32B** | Dense | 32B / 32B | 128K | ~19 GB | Day-to-day coding, code completion, code repair |
| **DeepSeek R1 Distill 32B** | Dense | 32B / 32B | 128K | ~19 GB | Reasoning, complex debugging, algorithmic tasks |
| **GPT-OSS 20B** | MoE | 21B / 3.6B | 128K | ~13 GB | Lightweight agentic workflows, tool use |
| **Devstral-Small-2 24B** | Dense | 24B / 24B | 256K | ~15 GB | Multi-file agentic editing, repo navigation |

### Tier 2 — Worth Considering

| Model | Type | Total / Active Params | Context | VRAM (Q4) | Notes |
|---|---|---|---|---|---|
| Nemotron 3 Nano 30B-A3B | Hybrid MoE | 31.6B / 3.5B | 128K | ~18 GB | Tight at 24GB, comfortable here. 68.3% LiveCodeBench v6 |
| Qwen3-30B-A3B-Thinking-2507 | MoE | 30B / 3.3B | 262K | ~18.6 GB | Deep reasoning with massive context headroom |
| Mistral Small 3.2 24B | Dense | 24B / 24B | 128K | ~15 GB | General coding, native tool calling |
| Qwen3-14B | Dense | 14.8B / 14.8B | 128K | ~9 GB | Fast model with tons of context headroom |

### Models That Do NOT Fit (for reference)

| Model | Total Params | Active Params | Context Window | VRAM (Q4) |
|---|---|---|---|---|
| GLM-4.7 Thinking | 358B MoE | ~45B | 200K | ~200+ GB |
| Kimi K2 | 1T MoE | 32B | 128K-256K | ~245 GB |
| DeepSeek V3.2 | 685B MoE | ~37B | 128K | ~380+ GB |

### Skip — Quantized 70B+ on 32GB

70B dense models at Q4 need ~35 GB. Even on 32GB, they require CPU offloading and drop to ~2-5 tok/s. The dense 32B and MoE models above deliver better performance while fitting entirely in VRAM.

---

## Ollama Pull Commands

```bash
# Tier 1
ollama pull qwen3-coder:30b
ollama pull qwen3:32b
ollama pull qwen2.5-coder:32b
ollama pull deepseek-r1:32b
ollama pull gpt-oss:20b
ollama pull devstral-small-2:24b

# Tier 2
ollama pull nemotron-3-nano:30b
ollama pull qwen3:30b-a3b-thinking-2507-q4_K_M
ollama pull mistral-small3.2:24b
ollama pull qwen3:14b
```

---

## Benchmark Summary

| Model | SWE-Bench Verified | HumanEval | Aider Polyglot | CodeForces Elo | ArenaHard |
|---|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | 69.6% | — | 61.8% | — | — |
| Qwen3 32B | — | — | — | — | — |
| Qwen2.5-Coder-32B | — | 91.0% | 73.7% (repair) | — | — |
| DeepSeek-R1-Distill-32B | — | — | — | 1691 | — |
| GPT-OSS 20B | — | Matches o3-mini | — | — | — |
| Devstral-Small-2 24B | 68% | — | — | — | — |
| Nemotron 3 Nano 30B-A3B | — | 78.1% | — | — | — |

---

## VRAM Budget (32GB)

### With q8_0 KV Cache (recommended)

| Component | VRAM |
|---|---|
| OS / display compositor | ~1 GB |
| Model weights (Q4_K_M, 32B) | ~19 GB |
| KV cache (q8_0) at 48K ctx | ~7.5 GB |
| Compute buffers / overhead | ~0.5 GB |
| **Total** | **~29 GB** |

### Max Context by Model (32GB, q8_0 KV cache)

| Model | VRAM for Weights | Remaining for KV | Approx Max Context |
|---|---|---|---|
| Qwen3-14B | ~9 GB | ~22 GB | ~128K (full) |
| GPT-OSS 20B | ~13 GB | ~18 GB | ~128K (full) |
| Devstral-Small-2 24B | ~15 GB | ~16 GB | ~128K+ |
| Mistral Small 3.2 24B | ~15 GB | ~16 GB | ~128K (full) |
| Qwen3-Coder 30B (MoE) | ~18.6 GB | ~12.4 GB | ~96-128K |
| DeepSeek R1 32B | ~19 GB | ~12 GB | ~64-96K |
| Qwen 2.5 Coder 32B | ~19 GB | ~12 GB | ~64-96K |
| Qwen3 32B | ~20 GB | ~11 GB | ~64-96K |

**32GB advantage:** Dense 32B models get 64-96K context here vs. only ~24-32K on 24GB GPUs. This is why 32B dense models belong on 32GB cards.

---

## Modelfile Examples

### DeepSeek R1 (custom context)

```
FROM deepseek-r1:32b
PARAMETER num_ctx 49152
PARAMETER num_gpu 999
```

```bash
ollama create deepseek-r1-48k -f DeepSeek-R1/Modelfile
```

Once the custom model is created, the original can be removed. The custom model references the same weight blobs — they won't be deleted.

```bash
ollama rm deepseek-r1:32b
```

### Qwen3-Coder (max context)

```
FROM qwen3-coder:30b
PARAMETER num_ctx 262144
```

```bash
ollama create qwen3-coder-256k -f Qwen3-Coder/Modelfile
```

---

## OpenCode Configuration

### Recommended model pairing

- **Primary:** `qwen3-coder:30b` — agentic coding, tool calling, file edits
- **Reasoning fallback:** `qwen3:32b` or `deepseek-r1-48k` — complex problems requiring step-by-step reasoning

### Inference settings (Qwen3-Coder recommended)

- temperature: 0.7
- top_p: 0.8
- top_k: 20
- repetition_penalty: 1.05

---

## Verify GPU Offload

After running a model, verify full GPU offload in another terminal:

```bash
ollama ps
```

Expected output should show **100% GPU** (or `0%/100% CPU/GPU`). If you see any CPU percentage, reduce `num_ctx` or switch to `q4_0` KV cache.

---

## Model Selection Guide

| Task | Use This |
|---|---|
| Day-to-day coding with OpenCode/Aider | Qwen3-Coder-30B-A3B |
| Proven code completion / repair | Qwen2.5-Coder-32B |
| Deep reasoning with traces | DeepSeek-R1-Distill-32B |
| General reasoning + coding | Qwen3 32B |
| Need speed or large context headroom | GPT-OSS 20B |
| Multi-file repo edits | Devstral-Small-2 24B |

---

## Tips

- **DeepSeek R1 temperature:** Set between 0.5-0.7 (0.6 recommended) to avoid incoherent outputs
- **DeepSeek R1 system prompt:** Works best without one — put all instructions in the user message
- **Qwen3-Coder mode:** Non-thinking only — no `<think>` blocks. Use Qwen3 32B if you need thinking mode
- **System RAM:** 64GB recommended alongside 32GB VRAM for smooth operation
