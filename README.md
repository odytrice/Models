# Models

Customized and fine-tuned open source models for local inference on consumer GPUs.

## Current Models

| Model | Base | Params | VRAM (Q4) | Best For |
|---|---|---|---|---|
| DeepSeek R1 Distill 32B | Qwen 2.5 | 32B dense | ~20 GB | Reasoning, complex debugging, algorithmic tasks |
| Qwen 2.5 Coder 32B | Qwen 2.5 | 32B dense | ~20 GB | Day-to-day coding, code completion, code repair |
| Qwen3 32B | Qwen 3 | 32B dense | ~20 GB | General + reasoning + coding (thinking/non-thinking modes) |
| GPT-OSS 20B | OpenAI | 20B (3.6B active MoE) | ~12 GB | Agentic workflows, tool use |

## Target Hardware

- **GPU:** NVIDIA RTX 5090 (32GB GDDR7)
- **Quantization:** Q4_K_M
- **KV Cache:** q8_0 (via `OLLAMA_KV_CACHE_TYPE`)
- **Inference:** Ollama

## Setup

See [Instructions.md](Instructions.md) for the full setup guide covering:

- Ollama environment variable configuration (Windows)
- Custom Modelfile creation with optimized context windows
- VRAM budget breakdowns and KV cache quantization
- API usage examples
