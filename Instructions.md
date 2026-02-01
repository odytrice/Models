# Local LLM Setup Guide — RTX 5090 (32GB VRAM)

## Models

### Models That Fit on 32GB VRAM (Q4_K_M)

| Model | Total Params | Active Params | Context Window | Max Output | VRAM (Q4) |
|---|---|---|---|---|---|
| GPT-OSS 20B | 20B | 3.6B (MoE) | 128K | 16K | ~12 GB |
| DeepSeek R1 Distill 32B | 32B | 32B (dense) | 128K | 32K | ~20 GB |
| Qwen 2.5 Coder 32B | 32B | 32B (dense) | 128K | — | ~20 GB |
| Qwen3 32B | 32B | 32B (dense) | 32K (128K w/ YaRN) | — | ~20 GB |

### Models That Do NOT Fit (for reference)

| Model | Total Params | Active Params | Context Window | VRAM (Q4) |
|---|---|---|---|---|
| GLM-4.7 Thinking | 358B MoE | ~45B | 200K | ~200+ GB |
| Kimi K2 | 1T MoE | 32B | 128K-256K | ~245 GB |
| DeepSeek V3.2 | 685B MoE | ~37B | 128K | ~380+ GB |

## Ollama Setup (Windows Desktop App)

### Step 1: Set Environment Variables

Ollama on Windows reads system/user environment variables. Set these before launching:

1. Quit Ollama from the system tray (right-click icon > Quit)
2. Open **Settings** > search **"environment variables"** > **"Edit environment variables for your account"**
3. Add two new user variables:

| Variable | Value | Purpose |
|---|---|---|
| `OLLAMA_FLASH_ATTENTION` | `1` | Enables flash attention, reduces memory overhead |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | Quantizes KV cache to ~half memory usage |

4. Click OK to save
5. Relaunch Ollama from the Start menu

**Note:** `OLLAMA_KV_CACHE_TYPE` requires `OLLAMA_FLASH_ATTENTION=1`. Without it, KV cache silently falls back to f16.

### Step 2: Pull Models

```bash
ollama pull deepseek-r1:32b
ollama pull qwen2.5-coder:32b
ollama pull qwen3:32b
```

### Step 3: Create Custom Models with Context Settings

Default Ollama context is only 2048 tokens. Create Modelfiles to set proper context.

**DeepSeek R1 (64K context):**

Create a file named `Modelfile.deepseek`:
```
FROM deepseek-r1:32b
PARAMETER num_ctx 65536
```

```bash
ollama create deepseek-r1-64k -f Modelfile.deepseek
```

**Qwen 2.5 Coder (64K context):**

Create a file named `Modelfile.qwen`:
```
FROM qwen2.5-coder:32b
PARAMETER num_ctx 65536
```

```bash
ollama create qwen2.5-coder-64k -f Modelfile.qwen
```

**Qwen3 32B (64K context):**

Create a file named `Modelfile.qwen3`:
```
FROM qwen3:32b
PARAMETER num_ctx 65536
```

```bash
ollama create qwen3-64k -f Modelfile.qwen3
```

### Step 4: Remove Original Models

Once custom models are created, originals can be removed. Custom models reference the same weight blobs — they won't be deleted.

```bash
ollama rm deepseek-r1:32b
ollama rm qwen2.5-coder:32b
ollama rm qwen3:32b
```

### Step 5: Run and Verify

```bash
ollama run deepseek-r1-64k
```

In another terminal, verify full GPU offload:

```bash
ollama ps
```

Expected output should show **100% GPU** (or `0%/100% CPU/GPU`). If you see any CPU percentage, reduce `num_ctx` or switch to `q4_0` KV cache.

## VRAM Budget Breakdown

### With q8_0 KV Cache (recommended)

| Component | VRAM |
|---|---|
| OS / display compositor | ~1 GB |
| Model weights (Q4_K_M, 32B) | ~20 GB |
| KV cache (q8_0) at 64K ctx | ~10 GB |
| Compute buffers / overhead | ~0.5 GB |
| **Total** | **~31.5 GB** |

### KV Cache Memory by Type and Context Size

| KV Cache Type | 32K ctx | 64K ctx | 128K ctx |
|---|---|---|---|
| f16 (default) | ~10 GB | ~20 GB | ~40 GB |
| q8_0 | ~5 GB | ~10 GB | ~20 GB |
| q4_0 | ~2.5 GB | ~5 GB | ~10 GB |

### Max Context by KV Cache Type (32B model on 32GB VRAM)

| KV Cache Type | Max Context (100% GPU) |
|---|---|
| f16 (default) | ~24K |
| q8_0 | ~64K |
| q4_0 | ~128K |

## API Usage

Ollama exposes a REST API on port 11434:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-r1-64k",
  "messages": [{"role": "user", "content": "Your prompt here"}]
}'
```

Python client:

```bash
pip install ollama
```

```python
from ollama import chat

response = chat(
    model='deepseek-r1-64k',
    messages=[{'role': 'user', 'content': 'Your prompt here'}],
)
print(response.message.content)
```

## Tips

- **DeepSeek R1 temperature:** Set between 0.5-0.7 (0.6 recommended) to avoid incoherent outputs
- **DeepSeek R1 system prompt:** Works best without one — put all instructions in the user message
- **System RAM:** 64GB recommended alongside 32GB VRAM for smooth operation
- **Model choice:** Use DeepSeek R1 for reasoning-heavy tasks, Qwen 2.5 Coder for day-to-day coding, Qwen3 for general-purpose with thinking/non-thinking mode switching
- **Qwen3 thinking mode:** Use `/think` and `/no_think` in prompts to toggle reasoning mode. Thinking mode is enabled by default
- **Quantization quality tradeoff:** q8_0 KV cache has negligible quality loss. q4_0 may show slight degradation at very high context sizes
