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
| **Qwen3-Coder-30B-A3B** | MoE | 30B / 3.3B | 256K | ~18.6 GB | Agentic coding, tool calling (OpenCode, Aider, Cline) |
| **Qwen3-30B-A3B-Thinking-2507** | MoE | 30B / 3.3B | 262K | ~18.6 GB | Deep reasoning, complex debugging |
| **GPT-OSS 20B** | MoE | 21B / 3.6B | 128K | ~13 GB | Lightweight agentic workflows, tool use |
| **Devstral-Small-2 24B** | Dense | 24B / 24B | 256K | ~15 GB | Multi-file agentic editing, repo navigation |
| **Mistral Small 3.2 24B** | Dense | 24B / 24B | 128K | ~15 GB | General coding, tool calling, multilingual |
| **Qwen3-14B** | Dense | 14.8B / 14.8B | 128K | ~9 GB | Reasoning + coding with massive context headroom |

### Tier 2 — Worth Considering

| Model | Type | Total / Active Params | Context | VRAM (Q4) | Notes |
|---|---|---|---|---|---|
| Ministral 3 14B | Dense | 14B / 14B | 256K | ~9 GB | Native function calling, multimodal, small footprint |
| Llama 4 Scout 17B | MoE | 17B | 10M | ~10-12 GB | Huge context, not coding-specialized |

### Skip — Quantized 70B+ on 24GB

70B dense models at Q4 need ~35 GB. They require CPU offloading and drop to ~2-5 tok/s. The MoE models above deliver better coding performance while fitting entirely in VRAM.

### Skip — Dense 30-32B on 24GB

Dense 32B models (Qwen2.5-Coder-32B, DeepSeek-R1-Distill-32B) use ~19-20 GB for weights alone, leaving only ~3-4 GB for KV cache. This limits usable context to ~24-32K — too tight for agentic coding. These models belong on [32GB GPUs](32GB-GPU.md) where they get proper context headroom.

---

## Key Insight: MoE > Quantized Dense

MoE models (Qwen3-Coder 30B, GPT-OSS 20B) activate only a fraction of their parameters per token. This means:
- **Faster inference** than dense 32B models (~30-40 tok/s vs ~15-20 tok/s)
- **Less VRAM** for the same or better quality
- **Better than quantized 70B** dense models which are too slow for interactive coding

For 24GB GPUs, the sweet spot for dense models is **14-25B parameters**. MoE models with low active params (3-4B) can exceed this range because their VRAM usage is driven by total weights, not active compute.

---

## Ollama Pull Commands

```bash
# Tier 1
ollama pull qwen3-coder:30b
ollama pull qwen3:30b-a3b-thinking-2507-q4_K_M
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
| Qwen3-Coder-30B-A3B | 69.6% | — | 61.8% | — | — |
| Qwen3-30B-A3B-Thinking | 69.6% | — | — | 1974 | 91.0 |
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
| Model weights (varies by model) | 9-19 GB |
| Available for KV cache + overhead | 4-14 GB |

### Max Context by Model (24GB, q8_0 KV cache)

| Model | VRAM for Weights | Remaining for KV | Approx Max Context |
|---|---|---|---|
| Qwen3-14B | ~9 GB | ~14 GB | ~128K (full) |
| GPT-OSS 20B | ~13 GB | ~10 GB | ~128K (full) |
| Devstral-Small-2 24B | ~15 GB | ~8 GB | ~64-96K |
| Mistral Small 3.2 24B | ~15 GB | ~8 GB | ~64-96K |
| Qwen3-Coder 30B (MoE) | ~18.6 GB | ~4.4 GB | ~32-48K |

**Tip:** If you need larger context with the MoE 30B models, switch KV cache to `q4_0` to roughly double the available context at the cost of slight quality loss.

---

## OpenCode Configuration

### Recommended model pairing

- **Primary:** `qwen3-coder:30b` — agentic coding, tool calling, file edits
- **Reasoning fallback:** `qwen3:30b-a3b-thinking-2507` — complex problems

### Known issues

- If tool calls generate JSON but never execute, increase `num_ctx` to 16K-32K+
- Qwen3-Coder runs in non-thinking mode only (fast, direct responses)
- Q5_K_M quantization is the quality sweet spot if VRAM allows

### Inference settings (Qwen3-Coder recommended)

- temperature: 0.7
- top_p: 0.8
- top_k: 20
- repetition_penalty: 1.05

---

## Model Selection Guide

| Task | Use This |
|---|---|
| Day-to-day coding with OpenCode/Aider | Qwen3-Coder-30B-A3B |
| Hard algorithmic / debugging problems | Qwen3-30B-A3B-Thinking-2507 |
| Need speed or large context headroom | GPT-OSS 20B or Qwen3-14B |
| Multi-file repo edits | Devstral-Small-2 24B |
| General coding + tool calling | Mistral Small 3.2 24B |
| Small footprint utility tasks | Ministral 3 14B |
