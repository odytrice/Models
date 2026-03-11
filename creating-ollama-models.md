# Creating Custom Ollama Models from Hugging Face

How to take a model from Hugging Face and create a custom Ollama model with your own configuration.

---

## Prerequisites

- [Ollama](https://ollama.com/download) installed and running
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) installed

### Install Hugging Face CLI

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

**Linux/macOS:**

```bash
pip install huggingface-hub
```

After installation, close and reopen your terminal. The `hf` command (Windows) or `huggingface-cli` command (Linux/macOS) should be available.

---

## Step 1: Find and Download a GGUF File

Ollama requires models in **GGUF format**. Most Hugging Face models are published as Safetensors — you need to find a GGUF conversion.

### Finding GGUFs

For a model like `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled`:

1. **Check if the author published GGUFs** — look for a `-GGUF` variant of the same repo (e.g. `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF`)
2. **Check community quantizers** — search Hugging Face for the model name + "GGUF". Common quantizers include `mradermacher`, `bartowski`, and `unsloth`
3. **Quantize it yourself** — use `llama.cpp`'s `convert_hf_to_gguf.py` to convert Safetensors to GGUF (advanced, not covered here)

### Choosing a Quantization

| Format | Bits/Weight | Quality | Typical Size (27B) |
|---|---|---|---|
| Q2_K | ~2.5 | Poor | ~10 GB |
| Q3_K_M | ~3.5 | Acceptable | ~13 GB |
| Q4_K_M | ~4.5 | Good (sweet spot) | ~16.5 GB |
| Q5_K_M | ~5.5 | Better | ~20 GB |
| Q8_0 | 8 | Near-lossless | ~28 GB |

**Q4_K_M** is the recommended default — best balance of quality and VRAM usage. Use Q5_K_M if you have the headroom.

### Download the GGUF

**Windows (using `hf` CLI):**

```bash
hf download <repo-id> <filename> --local-dir .
```

**Linux/macOS (using `huggingface-cli`):**

```bash
huggingface-cli download <repo-id> <filename> --local-dir .
```

**Example:**

```bash
hf download Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF Qwen3.5-27B.Q4_K_M.gguf --local-dir .
```

This downloads the 16.5 GB Q4_K_M file into the current directory.

---

## Step 2: Create a Modelfile

A `Modelfile` is Ollama's configuration format — it tells Ollama which GGUF to use and what parameters to set.

Create a file called `Modelfile` in the same directory as your GGUF:

```
FROM ./Qwen3.5-27B.Q4_K_M.gguf

PARAMETER num_ctx 65536
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER repeat_penalty 1.05
```

### Key Parameters

| Parameter | Description | Recommendation |
|---|---|---|
| `num_ctx` | Context window size in tokens | Set based on VRAM budget (see below) |
| `temperature` | Randomness (0 = deterministic, 1+ = creative) | 0.7 for coding |
| `top_p` | Nucleus sampling threshold | 0.8 |
| `top_k` | Limit token choices per step | 20 |
| `repeat_penalty` | Penalize repetition | 1.05 |

### Choosing num_ctx by VRAM

The context window is the biggest VRAM variable. After loading model weights, whatever VRAM remains goes to the KV cache (which stores context).

**Formula:** `Available KV VRAM = Total GPU VRAM - Model Weights - ~1.5 GB overhead`

| GPU VRAM | Model Weights (Q4) | KV Budget (q8_0) | Recommended num_ctx |
|---|---|---|---|
| 24 GB | ~16.5 GB (27B) | ~6 GB | 65536 (64K) |
| 32 GB | ~16.5 GB (27B) | ~14 GB | 131072 (128K) |
| 32 GB | ~21 GB (35B MoE) | ~10 GB | 131072 (128K) |

**Important:** Set `OLLAMA_KV_CACHE_TYPE=q8_0` and `OLLAMA_FLASH_ATTENTION=1` as environment variables before launching Ollama. Without KV cache quantization, these context sizes won't fit.

### Optional: SYSTEM Prompt and TEMPLATE

You can also set a system prompt or custom chat template:

```
SYSTEM """You are a helpful coding assistant."""
```

For most GGUF models, Ollama auto-detects the chat template from the GGUF metadata — you usually don't need to set TEMPLATE manually.

---

## Step 3: Create the Ollama Model

```bash
ollama create <name>:<tag> -f <path-to-Modelfile>
```

**Example:**

```bash
ollama create qwopus:27b-24gb -f Modelfile
```

Ollama copies the GGUF into its blob store and registers the model. This takes a few minutes for large files.

### Using Multiple Tags for Different Configs

You can create multiple tags from the same GGUF with different `num_ctx` values — one per GPU tier:

```bash
ollama create mymodel:24gb -f Modelfile.24gb
ollama create mymodel:32gb -f Modelfile.32gb
```

Both share the same weight blob on disk — only the configuration differs. No extra disk space is used.

---

## Step 4: Verify

```bash
ollama list
```

Should show your new model(s). Then test:

```bash
ollama run qwopus:27b-24gb
```

After loading, verify full GPU offload in another terminal:

```bash
ollama ps
```

Should show **100% GPU**. If you see any CPU percentage, reduce `num_ctx`.

---

## Step 5: Push to Ollama Registry (Optional)

If you want to share your model on [ollama.com](https://ollama.com), you can push it.

### 1. Create an account on ollama.com

Sign up at https://ollama.com/signup and note your username.

### 2. Create the model namespace on ollama.com

Go to https://ollama.com/new and create a new model with your desired name (e.g. `odytrice/qwopus`).

### 3. Copy your local model to the registry name

```bash
ollama cp qwopus:27b-24gb odytrice/qwopus:27b-24gb
ollama cp qwopus:27b-32gb odytrice/qwopus:27b-32gb
```

### 4. Set your Ollama key

Find your key at https://ollama.com/settings/keys. Copy the public key and add it to your account.

The key file is typically at:
- **Windows:** `C:\Users\<username>\.ollama\id_ed25519.pub`
- **Linux/macOS:** `~/.ollama/id_ed25519.pub`

Copy the contents and paste it at https://ollama.com/settings/keys.

### 5. Push

```bash
ollama push odytrice/qwopus:27b-24gb
ollama push odytrice/qwopus:27b-32gb
```

This uploads the model weights and configuration to Ollama's registry. Others can then pull it with:

```bash
ollama pull odytrice/qwopus:27b-24gb
```

---

## Complete Example: End-to-End

```bash
# 1. Create a working directory
mkdir qwopus && cd qwopus

# 2. Download the GGUF
hf download Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF \
  Qwen3.5-27B.Q4_K_M.gguf --local-dir .

# 3. Create the Modelfile
cat > Modelfile <<'EOF'
FROM ./Qwen3.5-27B.Q4_K_M.gguf
PARAMETER num_ctx 65536
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER repeat_penalty 1.05
EOF

# 4. Create the Ollama model
ollama create qwopus:27b -f Modelfile

# 5. Test it
ollama run qwopus:27b

# 6. (Optional) Push to registry
ollama cp qwopus:27b odytrice/qwopus:27b
ollama push odytrice/qwopus:27b
```

---

## Cleanup

After creating the Ollama model, the original GGUF file in your working directory is no longer needed — Ollama copies it into its own blob store. You can safely delete the downloaded GGUF to reclaim disk space:

```bash
# After ollama create succeeds
rm Qwen3.5-27B.Q4_K_M.gguf
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ollama create` hangs | The GGUF is being copied to Ollama's blob store. For 16+ GB files this takes several minutes. Wait. |
| `ollama ps` shows CPU % | Model + KV cache exceeds VRAM. Reduce `num_ctx` or switch to `q4_0` KV cache type. |
| Model outputs garbage | Check you downloaded the correct GGUF quant. Q2_K has noticeable quality loss. |
| `huggingface-cli: command not found` | On Windows, use `hf` instead, or call via the full path `C:\Users\<user>\.local\bin\hf.exe` |
| Chat template errors | Most GGUFs include embedded chat templates. If not, you may need to add a `TEMPLATE` block in the Modelfile. |
| GGUF not available for a model | Search for `<model-name> GGUF` on Hugging Face. Community quantizers like `mradermacher` or `bartowski` often publish GGUFs within days of a model release. |
