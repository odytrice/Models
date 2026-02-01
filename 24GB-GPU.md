# 24GB GPU — Local Coding Model Guide

Recommended models and configuration for GPUs with 24GB VRAM (RTX 4090, RTX 3090, RTX A5000, etc.).

> See [README.md](README.md) for universal setup instructions (environment variables, quantization, KV cache, context size reference).

## Target Hardware

- **VRAM:** 24GB
- **Quantization:** Q4_K_M (sweet spot), Q5_K_M (better quality if it fits)
- **KV Cache:** q8_0 (via `OLLAMA_KV_CACHE_TYPE`)
- **Inference:** Ollama

---

## Recommended Models

### Tier 1 — Best Choices

| Model | Type | Total / Active Params | Context | VRAM (Q4) | Best For |
|---|---|---|---|---|---|
| **GPT-OSS 20B** | MoE | 21B / 3.6B | 128K | ~13 GB | Agentic workflows, tool use, large context headroom |
| **Devstral-Small-2 24B** | Dense | 24B / 24B | 256K | ~15 GB | Multi-file agentic editing, repo navigation |
| **Mistral Small 3.2 24B** | Dense | 24B / 24B | 128K | ~15 GB | General coding, tool calling, multilingual |
| **Qwen3-14B** | Dense | 14.8B / 14.8B | 128K | ~9 GB | Reasoning + coding with massive context headroom |

### Tier 2 — Worth Considering

| Model | Type | Total / Active Params | Context | VRAM (Q4) | Notes |
|---|---|---|---|---|---|
| Ministral 3 14B | Dense | 14B / 14B | 256K | ~9 GB | Native function calling, multimodal, small footprint |
| Llama 4 Scout 17B | MoE | 17B | 10M | ~10-12 GB | Huge context, not coding-specialized |

### Skip — 30B+ Models on 24GB

30B MoE models (Qwen3-Coder-30B-A3B, Qwen3-30B-A3B-Thinking) use ~18.6 GB for weights, leaving only ~4.4 GB for KV cache. This limits usable context to ~32-48K — well below the 64K minimum for reasoning and 128K minimum for coding agents. These models belong on [32GB GPUs](32GB-GPU.md) where they get proper context headroom.

Dense 32B models (Qwen2.5-Coder-32B, DeepSeek-R1-Distill-32B) are even worse at ~19-20 GB for weights, leaving only ~3-4 GB for KV cache (~24-32K context).

### Skip — Quantized 70B+ on 24GB

70B dense models at Q4 need ~35 GB. They require CPU offloading and drop to ~2-5 tok/s.

---

## Ollama Pull Commands

```bash
# Tier 1
ollama pull gpt-oss:20b
ollama pull devstral-small-2:24b
ollama pull mistral-small3.2:24b
ollama pull qwen3:14b

# Tier 2
ollama pull ministral-3:14b
```

---

## Benchmark Summary

| Model | SWE-Bench Verified | HumanEval | Aider Polyglot | CodeForces Elo | ArenaHard |
|---|---|---|---|---|---|
| GPT-OSS 20B | — | Matches o3-mini | — | — | — |
| Devstral-Small-2 24B | 68% | — | — | — | — |
| Mistral Small 3.2 24B | — | 92.9% (Pass@5) | — | — | — |
| Qwen3-14B | — | — | — | — | 85.5 |

---

## VRAM Budget (24GB)

### With q8_0 KV Cache

| Component | VRAM |
|---|---|
| OS / display compositor | ~1 GB |
| Model weights (varies by model) | 9-15 GB |
| Available for KV cache + overhead | 8-14 GB |

### Max Context by Model (24GB, q8_0 KV cache)

| Model | VRAM for Weights | Remaining for KV | Approx Max Context |
|---|---|---|---|
| Qwen3-14B | ~9 GB | ~14 GB | ~128K (full) |
| GPT-OSS 20B | ~13 GB | ~10 GB | ~128K (full) |
| Devstral-Small-2 24B | ~15 GB | ~8 GB | ~64-96K |
| Mistral Small 3.2 24B | ~15 GB | ~8 GB | ~64-96K |

Every Tier 1 model fits its full trained context or close to it — no compromises.

### Verify with `ollama ps`

After loading a model, check VRAM usage and GPU offload:

```bash
ollama ps
```

Should show **100% GPU**. If you see any CPU percentage, reduce `num_ctx` or switch to `q4_0` KV cache. See [README.md](README.md#verifying-gpu-offload-and-vram-usage) for details.

---

## OpenCode Configuration

### Recommended model pairing

- **Primary:** `gpt-oss:20b` or `devstral-small-2:24b` — agentic coding, tool calling, file edits
- **Reasoning fallback:** `qwen3:14b` — complex problems requiring step-by-step thinking

### Known issues

- If tool calls generate JSON but never execute, increase `num_ctx` to 16K-32K+
- Q5_K_M quantization is the quality sweet spot if VRAM allows

### Inference settings

- temperature: 0.7
- top_p: 0.8
- top_k: 20
- repetition_penalty: 1.05

---

## Model Selection Guide

| Task | Use This |
|---|---|
| Day-to-day coding with OpenCode/Aider | GPT-OSS 20B or Devstral-Small-2 24B |
| Hard algorithmic / debugging problems | Qwen3-14B |
| Need speed or large context headroom | GPT-OSS 20B or Qwen3-14B |
| Multi-file repo edits | Devstral-Small-2 24B |
| General coding + tool calling | Mistral Small 3.2 24B |
| Small footprint utility tasks | Ministral 3 14B |
