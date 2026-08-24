# 🧮 LLM VRAM Planner

**Stop guessing if your model fits. Know before you deploy.**

I was deploying vLLM on an air-gapped server and spent 3 hours fighting CUDA version mismatches, wrong PyTorch wheels, and OOM crashes — all because there was no tool that answered "does this model fit on my GPU, and what flags do I pass to vLLM?" in one place. So I built one.

> **[Use it now →](https://israelhen153.github.io/llm-vram-planner/)**  ·  Single HTML file  ·  Works offline  ·  No signup

![LLM VRAM Planner — VRAM breakdown, vllm serve command, and cost comparison](assets/vllm_planner.png)

---

## What it does

**Pick a model → pick a GPU → get the answer.** VRAM breakdown, deployment command, cost estimate.

There are several good VRAM calculators around. [APXML's](https://apxml.com/tools/vram-calculator) is the one most people use and covers ground this doesn't — pipeline parallel, multi-node, CPU/NVMe offload, Apple Silicon. [WireUnwired's](https://wireunwired.com/local-llm-vram-calculator/) also generates a run command, and spans llama.cpp and MLX as well as vLLM.

No single feature here is unique. The combination is what I wanted and couldn't find: **open source, one HTML file that runs offline, and opinionated about vLLM specifically** rather than covering every engine shallowly.

- **Generates the actual `vllm serve` command** — tensor parallel, quantization, KV cache dtype, expert parallel, max context length. Copy-paste and run.
- **3-tier cost comparison** — hyperscaler (AWS/GCP/Azure), specialized (Lambda/CoreWeave), and spot (Vast.ai) pricing side by side.
- **Training estimation** — full fine-tune, LoRA, QLoRA. See exactly what's eating your VRAM: weights, gradients, optimizer states, activations.
- **Executive mode** — one-click toggle. Shows verdict, cost range, and max users. Hand the URL to your manager.
- **Per-user and aggregate throughput, kept separate** — a single user's decode speed and total server throughput differ by 50–100× under continuous batching, so they're reported as two numbers with two rooflines, and benchmarks are only ever compared against the matching one.
- **Import any model** — paste a HuggingFace ID or drop a `config.json`. Works for models that aren't in any preset list.
- **Report export** — copy a Markdown summary straight from the tool, or run `generate_report.py` for a procurement-ready PDF.
- **Works offline** — single HTML file, no backend, no internet required after first load. The hosted copy counts page views and nothing else — see [Analytics](#analytics).

---

## How accurate is it?

It's a planning tool. It gets you to the right hardware and the right flags; it does not replace benchmarking your actual workload.

**VRAM** is close to exact — weights, KV cache and optimizer state are arithmetic. Activations, CUDA context and NCCL buffers use fixed heuristics (1% of active params, 1.5 GiB/GPU, 0.2–0.3 GiB per extra GPU), so treat the total as ±10%, and don't plan to run at 99% utilisation.

**Throughput** comes from a memory-bandwidth roofline at 70% MBU, capped by a compute roofline at 35% MFU. TTFT uses a separate 45% prefill MFU — dense prefill GEMMs run at higher utilisation than decode's starved ones:

```
tok/s = (batch × achieved_bandwidth) / (active_weight_bytes + batch × kv_bytes_per_seq)
```

Against the measured benchmarks in `benchmarks/data.json`, single-stream lands within ~4% — but read that number with its caveat: the only two single-stream entries in the dataset are llama.cpp measurements on an RTX 4090, not vLLM. Decode at batch 1 is bandwidth-bound in either engine, which is why the roofline tracks them, but nothing here validates vLLM's own single-stream behaviour. Aggregate throughput runs 1.1–2.3× optimistic — a roofline is an upper bound, and real serving loses time to scheduling and prefill interleaving. The UI shows both numbers: the ceiling, and the inverse of that optimism as an observed range (43–91% of ceiling). The range is re-derived from the measured entries in `benchmarks/data.json` by the test suite, so it tightens as contributed data comes in and cannot silently drift.

**Benchmark data** is 13 entries: 7 measured with a published source, 6 extrapolated or unverified. The estimated ones are flagged `"estimated": true` and the UI says so. Comparisons only ever run against a benchmark of the matching kind — a single-stream estimate is never scored against a batch measurement.

**Costs** are list prices from mid-2026 and drift constantly. Reserved and committed-use pricing typically runs 30–60% below on-demand.

Every formula, constant and known failure mode is written up in **[docs/MODEL.md](docs/MODEL.md)** — including why the two throughput numbers are different quantities, and where the estimates break down.

Run `./tests/run.sh` to check the math yourself.

### Known gaps

Things it does not model yet, listed because finding out the hard way is worse:

- **Pipeline parallel and multi-node.** TP and DP only. Single-node assumptions throughout.
- **CPU/NVMe offload.** If it doesn't fit in VRAM, the answer here is "it doesn't fit".
- **Non-NVIDIA hardware and non-vLLM engines.** AMD and Apple Silicon are on the roadmap; llama.cpp and MLX are not.
- **Speculative decoding and MTP.** Both change the throughput picture substantially and neither is modelled.
- **Chunked prefill scheduling.** Prefill and decode interleaving affects tail latency under mixed load.

### What it does model, that most calculators don't

- **Three KV cache regimes.** Full attention, sliding-window (Gemma 3/4, Llama 4 chunked), and MLA (DeepSeek). These differ by more than 5×, and applying the wrong one is the single biggest source of error in VRAM planning.
- **Prefix caching as a memory effect, not just a latency one.** vLLM V1 enables APC by default. With a shared system prompt it stores that prefix once rather than once per request — 64 users sharing an 8K agent preamble is 63 GiB of KV back on an H100. It cuts TTFT and never touches decode speed.
- **Agentic overhead.** System prompt and tool schemas as an explicit input, because that's where the shared prefix comes from and nobody budgets for it.

---

## Quick start

### Use online
**[israelhen153.github.io/llm-vram-planner](https://israelhen153.github.io/llm-vram-planner/)**

### Use offline
Download `index.html`. Open in any browser. Done.

### Generate a PDF report
```bash
pip install reportlab
python generate_report.py --preset gemma4-26b --gpu a100-40 --prec awq --fp8-kv
```

### Share a configuration
Every slider change updates the URL. Copy it, send it to a teammate — they see exactly what you see.

---

## Analytics

The hosted copy at [israelhen153.github.io](https://israelhen153.github.io/llm-vram-planner/) counts page views with [GoatCounter](https://www.goatcounter.com). It is there for one reason: it is how I know whether anyone actually uses this, which is what decides whether it keeps getting worked on. There is no other telemetry, no backend, and no account of any kind.

Per [GoatCounter's privacy policy](https://www.goatcounter.com/help/privacy) (8 June 2025):

- **Stored**, as per-day and per-hour aggregate counts that cannot be linked to each other: page path, referrer, browser, OS, country, language, screen width.
- **Not stored**: IP addresses, the full User-Agent, any tracker ID. Nothing is written to your browser — no cookies, no localStorage, no cache. Nothing is shared with third parties.
- **Your configuration is never sent.** This tool keeps its entire state in the URL fragment (`#gpu=h100-80&...`), and `count.js` sends `location.pathname + location.search` only. Fragments are not part of an HTTP request; the model, GPU and workload you type stay in your browser.

**Opting out**, if you'd rather not be counted: load the page once with `#toggle-goatcounter` on the end of the URL. That sets a `skipgc` flag in your browser's localStorage and stops the counting for good on that browser. It replaces whatever is in the fragment, so do it on a fresh load rather than on a configuration you want to keep.

**Self-hosting or forking?** Delete the two-line `<script data-goatcounter=...>` block at the end of `index.html`, or run `./setup.sh <username> <your-goatcounter-code>` to repoint it. Otherwise your deployment reports its traffic to my site, which is not what either of us wants. Opened straight from `file://` it is inert either way: the script has nowhere to load from, and the tool never needed it.

---

## Who it's for

**DevOps / sysadmins** — "I have 2× A100 40GB. What can I run, and what's the vLLM command?"

**ML engineers** — "Should I LoRA or QLoRA on this hardware? How much VRAM does the optimizer eat?"

**Team leads / architects** — "Executive mode. What does this cost per month? Can I show this to procurement?"

---

## Roadmap

- **v1.0** (current) — NVIDIA GPUs, vLLM
- **v1.1** — AMD ROCm (MI300X, MI250X)
- **v1.2** — Apple Silicon (M1–M4, Ollama/llama.cpp/MLX)
- **v2.0** — UI polish, guided wizard, mobile, PWA

See [ROADMAP.md](ROADMAP.md) for details.

---

## Contributing

**The most valuable contribution is benchmark data.** If you've measured vLLM throughput on real hardware, add one entry to `benchmarks/data.json` and open a PR. See [CONTRIBUTING.md](CONTRIBUTING.md) for the format — it takes 2 minutes.

Bug reports and feature requests welcome via [Issues](https://github.com/israelhen153/llm-vram-planner/issues).

---

## Project structure

```
index.html              The tool (single file, no build step)
generate_report.py      PDF report generator
benchmarks/data.json    Community benchmark data
docs/MODEL.md           Every formula, constant and limitation, explained
tests/run.sh            Full suite — node + python3, no other deps
tests/model.test.js     VRAM, throughput and TTFT math
tests/parity.test.py    Checks index.html and generate_report.py agree
CONTRIBUTING.md         How to add benchmarks
ROADMAP.md              Version plan
```

The model is implemented twice — once in `index.html` so the tool needs no backend, once in `generate_report.py` so the PDF needs no browser. `tests/parity.test.py` diffs the two engines field by field; if you change the math in one, change it in both or the suite fails.

---

MIT License · Built by an engineer who got tired of OOM crashes.
