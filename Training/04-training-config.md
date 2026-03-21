# Training Configuration & Cost Estimates

## Precision Strategy: Train in BF16, Quantize After

**Train at full BF16 precision with LoRA, then quantize to 4-bit GGUF/GPTQ for inference.**

Do NOT train on a pre-quantized (4-bit QLoRA) model. Post-training quantization produces better results because:

1. LoRA adapters learn on full-precision weights -- no quantization noise during training
2. After training, merge LoRA into the full-precision model, then quantize once
3. You keep the full BF16 merged model as an artifact -- can quantize to any precision (Q4_K_M, Q5_K_M, Q6_K, Q8_0)
4. Research consistently shows PTQ of a well-trained model outperforms training on a pre-quantized model

If VRAM is too tight at BF16 for stage 4 (256K context), fall back to QLoRA for that stage only.

---

## Context Length: Train at 256K

**Train at 256K (262144 tokens), inference at 204800.**

There is no requirement for context sizes to be powers of 2. The "powers of 2" advice is a simplification -- Flash Attention is optimized for multiples of 128/256 (and pads internally anyway), not strict powers of 2. 204800 = 200 × 1024, which is well-aligned.

Training at 256K is recommended because:
- The model learns the full range that Qwen3.5-27B natively supports
- Inferencing at 204800 uses a subset of training range -- no degradation
- A model trained at 256K handles any length below it (204800, 128K, 64K, etc.)

**Exception**: If training locally on 32GB VRAM, train at 204800 instead to save ~25% VRAM at stage 4. On cloud (48-80GB), train at 256K.

---

## LoRA Training Configuration

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-27B",
    max_seq_length=2048,  # start small, scale up after confirming it works
    load_in_4bit=False,
    load_in_16bit=True,   # BF16 -- do NOT use 4-bit for training
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,  # alpha/r ratio of 2
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # critical for long context + low VRAM
    random_state=3407,
    max_seq_length=2048,
)
```

---

## Training Time Estimates

### Per-stage breakdown (BF16 LoRA, batch size 1, gradient accumulation 8, ~10-15K samples)

| Stage | Context Length | Samples | Epochs | Steps (approx) | Time/Step | Stage Time |
|-------|--------------|---------|--------|-----------------|-----------|------------|
| **Stage 1** | 8K-16K | ~8K | 2-3 | ~2,500-3,000 | ~3-5 sec | **2-4 hours** |
| **Stage 2** | 32K-64K | ~3K | 2 | ~750 | ~15-30 sec | **3-6 hours** |
| **Stage 3** | 128K | ~1.5K | 1-2 | ~375 | ~2-4 min | **12-24 hours** |
| **Stage 4** | 256K | ~500 | 1 | ~125 | ~5-10 min | **10-20 hours** |

Stages 3 and 4 account for ~80% of total training time. Short-context stages are fast.

### Total training time

| Scenario | Time |
|----------|------|
| **Optimistic** | ~27 hours (~1.1 days) |
| **Realistic** | ~40-50 hours (~2 days) |
| **Pessimistic** (OOM retries, restarts) | ~60-70 hours (~3 days) |

---

## Local vs Cloud Training

### Option A: Local RTX 5090 (32GB)

- **Cost**: $0 (hardware already owned)
- **Time**: ~40-50 hours (2-3 days continuous)
- **Risk**: OOM at stage 4 (256K on 32GB is very tight). May need to fall back to 204800 or QLoRA for final stage.
- **Pros**: Free, no upload/download time, full control
- **Cons**: GPU pegged at 100% for days, power/cooling, OOM debugging

### Option B: Cloud GPU (recommended)

| Approach | Config | VRAM | Est. Time | Est. Cost |
|----------|--------|------|-----------|-----------|
| **Cheapest** | RunPod community RTX 5090 | 32GB | ~40-50 hrs | **$20-35** |
| **Best balance** | RunPod L40S or RTX 6000 Ada | 48GB | ~35-45 hrs | **$30-50** |
| **Fastest** | RunPod 2x A100 80GB | 2x 80GB | ~15-20 hrs | **$55-70** |
| **Simplest** | Paperspace A100 80GB (managed) | 80GB | ~25-35 hrs | **$30-40** |

**Recommended: RunPod L40S or RTX 6000 Ada (~$30-50 total).** The 48GB VRAM gives comfortable headroom for 256K context at BF16, avoids OOM risk, and keeps costs under $50. Done in ~1.5-2 days.

For sub-24-hour turnaround: **2x A100 pod (~$55-70).**

---

## Cloud GPU Providers

Prices are per-second on RunPod (shown as approximate hourly equivalents). Community cloud prices fluctuate based on supply/demand.

### For Student Training (Qwen3.5-27B LoRA)

| Platform | GPU | VRAM | Approx. $/hr | Notes |
|----------|-----|------|---------------|-------|
| **RunPod** | RTX 5090 | 32GB | ~$0.50-0.70/hr | Same VRAM as local, same OOM risk |
| **RunPod** | L40S | 48GB | ~$0.80-1.20/hr | Comfortable headroom for 256K |
| **RunPod** | RTX 6000 Ada | 48GB | ~$0.80-1.00/hr | Professional grade, good availability |
| **RunPod** | A100 80GB SXM | 80GB | ~$1.79/hr (cluster) | Best for max context, no VRAM worries |
| **RunPod** | 2x A100 80GB | 2x 80GB | ~$3.58/hr | Fastest option, ~15-20 hrs total |
| **Paperspace** | A100 80GB | 80GB | $1.15/hr (committed) | Managed Jupyter notebooks |
| **Paperspace** | A6000 | 48GB | $1.89/hr | Simpler setup |

### Important Notes
- RTX 4090 has only **24GB VRAM** -- not enough for 27B BF16 training
- RunPod uses dynamic per-second pricing on community cloud; check console for live rates
- Paperspace committed pricing requires a contract term

---

## Cost Summary

### Data Generation (Teachers)
- All three teachers accessible via **Ollama cloud subscription** -- no additional per-token API costs
- Data generation cost is effectively **$0 beyond the existing Ollama subscription**
- Primary cost is **time** -- generating 5K-50K samples across three teachers will take days of scripted API calls

### Total Estimated Budget

| Approach | Training Cost | Ollama Sub | Total |
|----------|--------------|------------|-------|
| **Local RTX 5090** | $0 | Existing sub | **$0** |
| **Cloud budget** (L40S) | $30-50 | Existing sub | **$30-50** |
| **Cloud fast** (2x A100) | $55-70 | Existing sub | **$55-70** |

### Post-training: Quantization and Export
After training completes, merge LoRA adapters and export:
- **GGUF** (Q4_K_M) for llama.cpp / Ollama local inference
- **GPTQ-Int4** for vLLM / HuggingFace Transformers inference
- Both fit comfortably in 32GB VRAM at 204800 token context
