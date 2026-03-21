# Teacher Models: Strengths & Weaknesses

## 1. Kimi K2.5 (Moonshot AI) -- The Generalist King

- **Source**: `moonshotai/Kimi-K2.5` ([HuggingFace](https://huggingface.co/moonshotai/Kimi-K2.5))
- **Released**: January 27, 2026
- **Architecture**: MoE -- 1.1T total, 32B active per token (384 experts, 8 routed + 1 shared)
- **Context**: 256K tokens (longest of the three teachers)
- **Multimodal**: Yes (MoonViT 400M vision encoder)
- **License**: Modified MIT
- **Access**: Ollama cloud (`ollama run kimi-k2.5:cloud`) or Moonshot API ($0.60/MTok in, $3.00/MTok out)
- **Paper**: [arXiv:2602.02276](https://arxiv.org/abs/2602.02276)

**Strengths:**
- Best all-around benchmarks -- top-3 or better in nearly every category (HumanEval 99%, AIME 96.1%, MMLU-Pro 87.1%, GPQA 87.6%)
- Best instruction following (IFEval 94%) -- critical for generating clean, consistent training data
- Longest native context (256K) -- the only teacher that can generate truly long-context training samples at full quality
- Strong frontend/JS/TS -- 73% SWE-bench Multilingual, specifically highlighted for "particularly strong frontend capabilities"
- Thinking mode generates reasoning traces valuable for distillation
- Cheap API ($0.60/MTok input)

**Weaknesses:**
- Not the best at real-world SWE tasks -- 76.8% SWE-bench Verified, beaten by MiniMax M2.5 (80.2%) and M2.7
- Terminal/system tasks lag -- 50.8% Terminal-Bench 2, beaten by M2.7 (57.0%)
- No self-evolution or agent-team native behaviors
- F# capability unproven -- no F# benchmarks exist; large corpus helps but not verified
- Requires 8x A100 80GB to self-host -- API-only for most people

**Best used for**: Svelte/SvelteKit, TypeScript, long-context samples (32K-204K), cross-domain prompts, instructional content

---

## 2. MiniMax M2.7 (MiniMax) -- The Agentic Engineer

- **Source**: Cloud-only via [Ollama](https://ollama.com/library/minimax-m2.7) and [MiniMax API](https://platform.minimax.io)
- **Released**: March 18, 2026 (brand new)
- **Architecture**: MoE -- 229B total, 10B active per token
- **Context**: 200K tokens
- **License**: Cloud-only (no downloadable weights)
- **Access**: Ollama cloud (`ollama run minimax-m2.7:cloud`) or MiniMax API
- **Announcement**: [minimax.io/news/minimax-m27-en](https://www.minimax.io/news/minimax-m27-en)

**Strengths:**
- Best real-world software engineering -- SWE-Pro 56.22% (matches GPT-5.3-Codex), SWE Multilingual 76.5 (beats K2.5's 73.0)
- Best system-level understanding -- Terminal Bench 2: 57.0% (beats K2.5 by 6+ points), critical for Docker/K8s training data
- Native Agent Teams -- multi-agent collaboration with role boundaries, adversarial reasoning, and protocol adherence. The only teacher that natively understands multi-agent coordination
- 97% skill adherence across 40 complex skills (each 2K+ tokens) -- training data will be consistent and high-fidelity
- First model to deeply participate in its own evolution -- trained via recursive self-improvement, produces more sophisticated reasoning patterns
- Available on Ollama cloud -- works with existing Ollama subscription
- Extremely efficient -- only 10B active params from 229B total
- MLE Bench Lite: 66.6% medal rate in autonomous ML competitions, tying with Gemini 3.1

**Weaknesses:**
- Cloud-only -- no downloadable weights, fully dependent on MiniMax API/Ollama cloud
- No open weights -- can't inspect, modify, or understand what the model learned
- 200K context (not 256K) -- slightly shorter than K2.5
- Weaker on pure reasoning benchmarks -- AIME, GPQA, MMLU-Pro all lower than K2.5
- Brand new (released 3 days ago) -- limited community validation, potential undiscovered issues
- Pricing unclear -- cloud-only means costs depend on MiniMax/Ollama pricing
- Verbose output -- generates more tokens per response, may increase data generation costs
- Hallucinates on underspecified prompts -- needs well-structured prompts (mitigatable for distillation since prompts are controlled)

**Best used for**: Agentic coding tasks, multi-step bug fixes, Docker/K8s/infrastructure, system-level DevOps, real-world SWE patterns

---

## 3. DeepSeek V3.2 (DeepSeek) -- The Corpus Giant

- **Source**: `deepseek-ai/DeepSeek-V3.2` ([HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3.2))
- **Released**: December 1, 2025 (most battle-tested of the three)
- **Architecture**: MoE -- 685B total, 37B active per token
- **Context**: 128K tokens
- **License**: Full MIT (most permissive)
- **Access**: Ollama cloud (`ollama run deepseek-v3.2:cloud`), DeepSeek API, or self-hosted via vLLM/SGLang
- **Variants**: V3.2-Speciale (96.0% AIME, IMO gold medal)

**Strengths:**
- Largest training corpus -- 685B params trained on the most diverse code data; highest probability of F# exposure among all open models
- Full MIT license -- no restrictions whatsoever, most permissive of the three
- Strong on statically-typed languages -- best Java results in Multi-SWE-bench (22.66% MagentLess), which is the best proxy for F#/C#
- "Thinking in Tool-Use" architecture -- retains chain-of-thought across tool calls, good for generating agentic training data
- DeepSeek Sparse Attention -- 3x faster long-context processing
- Speciale variant achieves 96.0% AIME, IMO 2025 gold medal for math reasoning
- Self-hostable -- open weights on HuggingFace, can run via vLLM/SGLang
- Proven and battle-tested -- released Dec 2025, extensive community validation over 4 months

**Weaknesses:**
- Shorter context -- 128K tokens, half of K2.5's 256K. Cannot generate training samples at 200K+ context
- Weaker SWE-bench -- 67.8-73.1% Verified, significantly behind M2.7 and K2.5
- Lower Terminal-Bench -- 46.4%, worst of the three teachers on system-level tasks
- Older model -- Dec 2025 release, surpassed by both K2.5 (Jan 2026) and M2.7 (Mar 2026) on most benchmarks
- No multimodal -- text-only
- Heavy to self-host -- 685B params requires significant GPU resources
- No agent team capability -- single-agent only, no native multi-agent coordination

**Best used for**: F#, Akka.NET, .NET/ASP.NET Core, statically-typed language patterns, reasoning-heavy training data

---

## Teacher Comparison Matrix

| Dimension | Kimi K2.5 | MiniMax M2.7 | DeepSeek V3.2 |
|-----------|-----------|--------------|---------------|
| **Overall intelligence** | **Best** | Good | Good |
| **SWE (real-world coding)** | Good (76.8%) | **Best** (SWE-Pro 56.2%) | Weakest (67-73%) |
| **System/DevOps** | Good (50.8%) | **Best** (57.0%) | Weakest (46.4%) |
| **Math/reasoning** | **Best** (AIME 96.1%) | Moderate | Strong (Speciale 96.0%) |
| **Instruction following** | **Best** (IFEval 94%) | Good (97% skill adherence) | Good |
| **Long context** | **Best** (256K native) | Good (200K) | Weakest (128K) |
| **F# suitability** | Medium | Medium | **Best** (largest corpus) |
| **Frontend/Svelte/TS** | **Best** | Good | Good |
| **Docker/K8s** | Good | **Best** | Weakest |
| **Agentic tasks** | Good | **Best** | Good |
| **License** | Modified MIT | Cloud-only (no weights) | **Full MIT** |
| **Availability** | Ollama cloud + API | Ollama cloud + API | Ollama cloud + API + open weights |
| **Cost control** | Ollama subscription | Ollama subscription | Ollama subscription |
| **Maturity** | 2 months | **3 days** | 4 months (most proven) |
