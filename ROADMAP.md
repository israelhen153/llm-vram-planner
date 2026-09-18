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

The correctness work is done and awaiting release. AMD is next.

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

### A throughput ceiling that holds for FP8 — shipped, not yet released

The compute ceiling and time to first token were charged at 16-bit cost whatever
precision was selected. On cards with native FP8 that understated the ceiling by half —
real FP8 serving beat the figure labelled a ceiling — and doubled time to first token.
Both are now charged at the precision the matrix multiplies actually run in. FP8
computes in 8 bits and gets the speed-up. AWQ, GPTQ and GGUF dequantize to 16 bits
before multiplying, so they get none; their gain is in memory bandwidth, which the model
already counts. Cards that run FP8 weights without FP8 compute say so beside the figure.
The observed range, which is re-derived from the measured runs, moved with it.

### A weekly price check — shipped, not yet released

Moved up from v1.2. `tools/price_check.py` reads published prices from Azure, AWS,
Lambda, CoreWeave and Vast.ai. A weekly job runs it and opens a pull request when a
price is confirmed or has moved. It never pushes, and any move over 40% is left for a
person to judge. A confirmed price records the provider, SKU, region and date it was
read. The catalog's current prices predate the check and are corrected next.

### Documents and images that match the tool — shipped, not yet released

The README and this roadmap were corrected against the code, and tests now fail when
the counts and constants they quote drift from the source. The share images are
generated from the tool rather than drawn, and a test fails when they stop matching it.

### Unmodelled hardware says so — shipped, not yet released

A card without performance constants of its own would silently borrow NVIDIA's, and the
AMD cards would have been the first. Every catalog row now names the constants its
silicon was measured with, and the model looks them up by that name rather than by who
made the card. A row whose name has no entry gets no constants at all: its VRAM, fit,
command and cost figures are unaffected, and its throughput reads as not modelled, with
the reason, on every surface including the PDF. No shipped number moved — every row
today names the constants it was already computed with.

### AMD / ROCm — researched, not yet built

One change lands first:

1. **Prices name their source.** Each price shows the provider, SKU, region and date it
   was read, or says that its source is not recorded, and prices that have drifted move
   to current published rates. No row presents one number as three providers' price.

**GPUs to add:**
- MI210 (64GB HBM2e, 1.6 TB/s)
- MI250X (128GB HBM2e, 3.2 TB/s — one catalog row, two GCDs)
- MI300X (192GB HBM3, 5.3 TB/s)
- MI325X (256GB HBM3e, 6.0 TB/s)
- RX 7900 XTX (24GB, consumer)

**What changes:**
- GPU dropdown gets an AMD section with correct VRAM, bandwidth, pricing
- AMD never borrows NVIDIA's MBU/MFU constants. The published ROCm measurements are
  not yet enough to derive constants of its own, so AMD cards ship with full VRAM, fit,
  command and cost figures, and throughput marked as not modelled until they are
- ROCm guidance: the `rocm/vllm` image and `HIP_VISIBLE_DEVICES`. **vLLM has no
  `--device` flag** — an earlier draft of this roadmap promised one, and it does not
  exist. Which accelerator you get is decided by the image and the environment.
- FP8 gated off CDNA2 (MI210, MI250X); it needs CDNA3
- Cost data for AMD GPUs, each price with a named, dated source
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
The first piece, the weekly price check, shipped early in v1.1: it opens a PR against
`data/gpus.json`, the file stays baked into the single HTML file, and the tool stays
offline.

**What changes:**
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
