# Roadmap

## v1.0.0 — NVIDIA + ship it (shipped)

**Status: published** at [israelhen153.github.io/llm-vram-planner](https://israelhen153.github.io/llm-vram-planner/).

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
- [x] PDF report card (via `generate_report.py`; the web tool copies Markdown)
- [x] URL state sharing + comparison snapshots
- [x] Offline-capable (single HTML file)
- [x] GitHub project with contribution guide
- [x] Info tooltips on the inputs and results

**GPUs covered:** whatever is in [`data/gpus.json`](data/gpus.json). That file is the
authority — `tools/sync_data.py` generates the catalog in `index.html` and
`generate_report.py` from it. A list here would be a second copy of the catalog and
would go stale the first time a row was added, which is exactly what happened to the
list that used to sit on this line.

**Inference engine:** vLLM only

**Corrected after launch:** the multi-GPU checkbox above was ticked at launch and was
only half true. The TP/DP split was displayed correctly but the VRAM math did not use
it, so per-device figures were optimistic above 8 devices. Fixed in v1.1 — see below.

---

## v1.1.0 — AMD, and correct above 8 GPUs

Two halves. The second one is done and awaiting release.

### Correct above 8 GPUs — shipped, not yet released

Per-device VRAM divided by every device in the cluster, which is not how the emitted
command shards. Above 8 devices the split becomes TP × DP: each data-parallel replica
loads a full copy of the model and only the TP group inside a replica shards it. The
figure was correct to 8 devices, 2× optimistic at 16 and **16× at 128** — the slider's
maximum, where the tool claimed a 70B needed about 1 GiB per card.

Weights and activations now divide by TP; the KV cache keeps the full device count,
because data parallelism partitions the request stream rather than replicating the
cache. The board-count recommendation is searched rather than derived, since dividing a
cluster total by one device's capacity is circular. Mixture-of-experts is deliberately
held at the old divisor and says so on every surface. Full write-up in
[docs/MODEL.md §4](docs/MODEL.md).

### AMD / ROCm — not started

**GPUs to add:**
- MI210 (64GB HBM2e, 1.6 TB/s)
- MI250X (128GB HBM2e, 3.2 TB/s — one catalog row, two GCDs)
- MI300X (192GB HBM3, 5.3 TB/s)
- MI325X (256GB HBM3e, 6.0 TB/s)
- RX 7900 XTX (24GB, consumer)

**What changes:**
- GPU dropdown gets an AMD section with correct VRAM, bandwidth, pricing
- AMD gets its own MBU/MFU constants instead of borrowing NVIDIA's
- ROCm guidance: the `rocm/vllm` image and `HIP_VISIBLE_DEVICES`. **vLLM has no
  `--device` flag** — an earlier draft of this roadmap promised one, and it does not
  exist. Which accelerator you get is decided by the image and the environment.
- FP8 gated off CDNA2 (MI210, MI250X); it needs CDNA3
- Cost data for AMD GPUs (CoreWeave, Azure, Lambda pricing)
- Benchmark data structure already supports it — it just needs entries

**What doesn't change:**
- VRAM math is the same — params × bytes, KV cache formula, etc.
- Training estimation works identically
- Import/export, comparison, PDF — all unchanged

**ROCm-specific caveats to document:**
- Flash Attention support varies by GPU arch (CDNA2 vs CDNA3)
- Some quantization kernels (AWQ, GPTQ) have limited ROCm support
- vLLM ROCm builds require specific Docker images or source builds

---

## v1.2.0 — Repo-side pipeline + catalog import

Keeping the baked-in data fresh without giving the tool a runtime network dependency.

**What changes:**
- A scheduled job opens a PR against `data/gpus.json` when prices move. The file stays
  baked into the single HTML file; the tool stays offline.
- Catalog import, for the air-gapped "my card isn't in your list" case
- Benchmark ingestion in CI, so contributed entries are validated on arrival

**What doesn't change:**
- No backend, no database, no service, no second engine. **"Live updates" here means a
  build job that bakes fresh data into the same file** — not something the page calls
  at runtime. The offline guarantee is the product.

---

## v2.0.0 — UI polish + product feel

Only after v1.0–1.2 are stable and there's validated user demand.

**Design:**
- Responsive mobile layout (currently desktop-focused)
- Smooth transitions and animations on slider changes
- Dark/light mode toggle (currently auto from OS preference)
- Collapsible sections for cleaner first impression
- Progress indicator showing how full the tool is configured
- Screenshot-ready layout for og:image generation

**UX:**
- Guided mode: "What are you trying to do?" wizard that walks through choices
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

- **Pipeline parallel** — TP/DP covers most deployments, PP is niche. Note that the
  "Layers per device" tile currently divides layers by the device count, which *is*
  pipeline-parallel arithmetic on a command that never emits it. That tile is wrong and
  is flagged as wrong; it is a bug to fix, not a feature in progress.
- **Apple Silicon / unified memory** — was on this roadmap as v1.2, and is dropped
  rather than quietly deferred. It needs a second memory model (unified, shared with
  the OS), a second command generator (`ollama run` / `mlx_lm.server`), and a second
  engine's performance characteristics: a different tool wearing this one's interface.
  [APXML](https://apxml.com/tools/vram-calculator) already covers that ground well.
  This one stays opinionated about vLLM.
- **Power consumption estimation** — interesting but hard to validate
- **Network bandwidth requirements** — multi-node InfiniBand sizing
- **Storage planning** — model download sizes, disk I/O for model loading
- **Kubernetes / Helm chart generation** — too deployment-specific
- **Price comparison with API providers** — different product (inference-as-a-service vs self-hosted)
