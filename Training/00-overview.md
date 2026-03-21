# Multi-Teacher Distillation to Qwen3.5-27B (Dense) for 204800 Token Context

## Overview

Distill domain-specialized coding capabilities from three teacher models into Qwen3.5-27B dense (student) for local inference on 32GB VRAM with 204800 token context. The distilled model is **domain-specialized** for full-stack web development with F#, Svelte, TypeScript, .NET, Docker, and Kubernetes.

## Document Index

| File | Contents |
|------|----------|
| [00-overview.md](00-overview.md) | This file -- project overview, strategy, student model |
| [01-teacher-models.md](01-teacher-models.md) | Teacher model specs, strengths/weaknesses, comparison matrix |
| [02-domain-specialization.md](02-domain-specialization.md) | Training data composition, F# libraries, training topics |
| [03-distillation-pipeline.md](03-distillation-pipeline.md) | Data generation, F# compiler verification, doc sources, training steps |
| [04-training-config.md](04-training-config.md) | LoRA config, cloud GPU providers, cost estimates |
| [05-resolved-questions.md](05-resolved-questions.md) | All resolved and open questions |

## Multi-Teacher Strategy

Rather than relying on a single teacher, this project uses three complementary teachers -- each assigned to the domains where they are strongest.

| Teacher | % of Data | Domains | Ollama Command |
|---------|-----------|---------|----------------|
| **Kimi K2.5** | ~35% | Svelte, TypeScript, long-context, cross-domain prompts | `ollama run kimi-k2.5:cloud` |
| **MiniMax M2.7** | ~35% | Agentic coding, Docker/K8s, multi-step SWE, system-level | `ollama run minimax-m2.7:cloud` |
| **DeepSeek V3.2** | ~30% | F#, Akka.NET, .NET/ASP.NET Core | `ollama run deepseek-v3.2:cloud` |

All three teachers are accessible via Ollama cloud subscription.

---

## Student Model: Qwen3.5-27B (Dense)

- **Source**: `Qwen/Qwen3.5-27B` ([HuggingFace](https://huggingface.co/Qwen/Qwen3.5-27B))
- **Architecture**: Dense transformer (NOT MoE)
- **Parameters**: 27B
- **Native Context**: 256K tokens (trained with YaRN rope scaling)
- **Quantized variant**: `Qwen/Qwen3.5-27B-GPTQ-Int4` for inference
- **GGUF variant**: `unsloth/Qwen3.5-27B-GGUF` for llama.cpp
- **Inference VRAM**: ~16GB at 4-bit quantization, fits 32GB VRAM comfortably
- **Training VRAM**: ~32-48GB with LoRA + gradient checkpointing (16-bit)

---

## Key Libraries
- **Training**: [Unsloth](https://github.com/unslothai/unsloth) (efficient LoRA), [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- **Inference**: [llama.cpp](https://github.com/ggerganov/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang)
- **Dataset**: HuggingFace `datasets`
- **Teacher access**: All three via Ollama cloud subscription
