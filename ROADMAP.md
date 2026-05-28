# Roadmap

## v1.0.0 — NVIDIA + ship it (current)

**Status: Ready to publish**

Everything targets NVIDIA datacenter and consumer GPUs with vLLM.

- [x] VRAM calculator (inference + training)
- [x] Multi-GPU with TP/DP split and topology warnings
- [x] vLLM command generator
- [x] 3-tier cost comparison (hyperscaler / specialized / spot)
- [x] Workload presets (chat, coding, RAG, batch, etc.)
- [x] Published benchmark data with source citations
- [x] HuggingFace fetch + config.json + safetensors import
- [x] KV cache dtype (BF16 / FP8)
- [x] MoE: shared vs routed experts
- [x] Quantization: BF16, FP8, INT4/AWQ/GPTQ, GGUF Q2–Q8
- [x] PDF report card
- [x] URL state sharing + comparison snapshots
- [x] Offline-capable (single HTML file)
- [x] GitHub project with contribution guide

**GPUs covered:** T4, RTX 4090, A100 40/80GB, L40S, H100, RTX PRO 6000, B200

**Inference engine:** vLLM only

**Priority after launch:** collect benchmark data from community contributions

---

## v1.1.0 — AMD GPUs

Add ROCm / AMD Instinct support.

**GPUs to add:**
- MI210 (64GB HBM2e, 1.6 TB/s)
- MI250X (128GB HBM2e, 3.2 TB/s)
- MI300X (192GB HBM3, 5.3 TB/s)
- MI325X (256GB HBM3e, 6.0 TB/s)
- RX 7900 XTX (24GB, consumer)

**What changes:**
- GPU dropdown gets an AMD section with correct VRAM, bandwidth, pricing
- vLLM command adds `--device rocm` flag when AMD is selected
- Cost data for AMD GPUs (CoreWeave, Azure, Lambda pricing)
- Benchmark data structure already supports it — just needs entries
- Notes section explains ROCm vs CUDA compatibility caveats

**What doesn't change:**
- VRAM math is the same — params × bytes, KV cache formula, etc.
- Training estimation works identically
- Import/export, comparison, PDF — all unchanged

**ROCm-specific caveats to document:**
- Flash Attention support varies by GPU arch (CDNA2 vs CDNA3)
- Some quantization kernels (AWQ, GPTQ) have limited ROCm support
- vLLM ROCm builds require specific Docker images or source builds

---

## v1.2.0 — Apple Silicon

Add unified memory Macs for local inference (Ollama / llama.cpp / MLX).

**Chips to add:**
- M1/M2/M3/M4 (base, Pro, Max, Ultra)
- Unified memory: 8GB, 16GB, 24GB, 32GB, 36GB, 48GB, 64GB, 96GB, 128GB, 192GB

**What changes:**
- GPU dropdown gets "Apple Silicon" section
- Memory model switches from discrete VRAM to unified memory (shared with OS)
  - Usable memory ≈ total unified memory × 0.65-0.75 (OS + other apps take the rest)
- Bandwidth values per chip (M4 Max: ~546 GB/s, M2 Ultra: ~800 GB/s)
- Inference engine switches from vLLM to Ollama / llama.cpp / MLX
  - Command generator outputs `ollama run` or `mlx_lm.server` instead of `vllm serve`
- Quantization focus shifts to GGUF (Q4_K_M, Q5_K_M, Q6_K) — native format for llama.cpp
- Training section: MLX supports LoRA fine-tuning on Apple Silicon
- Cost section: shows "hardware purchase" cost instead of cloud hourly rates
  - Mac Studio M4 Ultra 192GB: ~$8,000 one-time
  - Break-even vs cloud at X months of 24/7 usage

**What doesn't change:**
- VRAM math core — same formulas, different memory pool
- Import/export, comparison, PDF
- Benchmark data structure

**Apple-specific caveats:**
- No tensor parallel (single unified memory pool)
- Metal performance shaders vs CUDA — different perf characteristics
- Thermal throttling on laptops vs desktops
- llama.cpp GGUF is the dominant format, not safetensors

---

## v2.0.0 — UI polish + product feel

Only after v1.0-1.2 are stable and there's validated user demand.

**Design:**
- Responsive mobile layout (currently desktop-focused)
- Smooth transitions and animations on slider changes
- Dark/light mode toggle (currently auto from OS preference)
- Collapsible sections for cleaner first impression
- Progress indicator showing how full the tool is configured
- Screenshot-ready layout for og:image generation

**UX:**
- Guided mode: "What are you trying to do?" wizard that walks through choices
- Tooltips on every metric explaining what it means (hover/tap)
- "Explain this" expandable sections for each VRAM component
- History: browser localStorage for recent configurations
- Side-by-side comparison as a first-class layout (not just appended cards)

**Distribution:**
- PWA support (installable, works offline with service worker)
- Embed mode (iframe-friendly for blog posts and docs)
- API endpoint (optional — serverless function that returns VRAM calc as JSON)

**Do NOT do in v2:**
- Don't add a backend or database
- Don't require a build step or npm
- Don't break offline capability
- Don't add accounts or auth
- Keep it a tool, not a platform

---

## Not planned (but open to PRs)

- **Pipeline parallel** — TP/DP covers most deployments, PP is niche
- **Power consumption estimation** — interesting but hard to validate
- **Network bandwidth requirements** — multi-node InfiniBand sizing
- **Storage planning** — model download sizes, disk I/O for model loading
- **Kubernetes / Helm chart generation** — too deployment-specific
- **Price comparison with API providers** — different product (inference-as-a-service vs self-hosted)
