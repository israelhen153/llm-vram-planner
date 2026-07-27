# 🧮 LLM VRAM Planner

**Stop guessing if your model fits. Know before you deploy.**

I was deploying vLLM on an air-gapped server and spent 3 hours fighting CUDA version mismatches, wrong PyTorch wheels, and OOM crashes — all because there was no tool that answered "does this model fit on my GPU, and what flags do I pass to vLLM?" in one place. So I built one.

> **[Use it now →](https://israelhen153.github.io/llm-vram-planner/)**  ·  Single HTML file  ·  Works offline  ·  No signup

![LLM VRAM Planner — VRAM breakdown, vllm serve command, and cost comparison](assets/vllm_planner.png)

---

## What it does

**Pick a model → pick a GPU → get the answer.** VRAM breakdown, deployment command, cost estimate.

There are several good VRAM calculators around — [APXML's](https://apxml.com/tools/vram-calculator) is the one most people use, and it covers ground this doesn't. What I couldn't find anywhere was a tool that goes from "does it fit" to "here is the command," so that's where this one leans:

- **Generates the actual `vllm serve` command** — tensor parallel, quantization, KV cache dtype, expert parallel, max context length. Copy-paste and run.
- **3-tier cost comparison** — hyperscaler (AWS/GCP/Azure), specialized (Lambda/CoreWeave), and spot (Vast.ai) pricing side by side.
- **Training estimation** — full fine-tune, LoRA, QLoRA. See exactly what's eating your VRAM: weights, gradients, optimizer states, activations.
- **Executive mode** — one-click toggle. Shows verdict, cost range, and max users. Hand the URL to your manager.
- **Per-user and aggregate throughput, kept separate** — a single user's decode speed and total server throughput differ by 50–100× under continuous batching. Most calculators report one number; conflating them is how you end up promising a latency you can't hit.
- **Import any model** — paste a HuggingFace ID or drop a `config.json`. Works for models that aren't in any preset list.
- **PDF report card** — export a procurement-ready document.
- **Works offline** — single HTML file, no backend, no internet required after first load.

---

## How accurate is it?

It's a planning tool. It gets you to the right hardware and the right flags; it does not replace benchmarking your actual workload.

**VRAM** is close to exact — weights, KV cache and optimizer state are arithmetic. Activations, CUDA context and NCCL buffers use fixed heuristics (1% of active params, 1.5 GiB/GPU, 0.2–0.3 GiB per extra GPU), so treat the total as ±10%, and don't plan to run at 99% utilisation.

**Throughput** comes from a memory-bandwidth roofline at 70% MBU, capped by a compute roofline at 35% MFU:

```
tok/s = (batch × achieved_bandwidth) / (active_weight_bytes + batch × kv_bytes_per_seq)
```

Against the measured benchmarks in `benchmarks/data.json`, single-stream lands within ~4%. Aggregate throughput runs 1.1–2.3× optimistic — a roofline is an upper bound, and real serving loses time to scheduling and prefill interleaving. It's labelled as a ceiling in the UI for that reason.

**Benchmark data** is 13 entries: 6 measured with a published source, 7 extrapolated or unverified. The estimated ones are flagged `"estimated": true` and the UI says so. Comparisons only ever run against a benchmark of the matching kind — a single-stream estimate is never scored against a batch measurement.

**Costs** are list prices from mid-2026 and drift constantly. Reserved and committed-use pricing typically runs 30–60% below on-demand.

Run `./tests/run.sh` to check the math yourself.

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
tests/run.sh            Full suite — node + python3, no other deps
tests/model.test.js     VRAM, throughput and TTFT math
tests/parity.test.py    Checks index.html and generate_report.py agree
CONTRIBUTING.md         How to add benchmarks
ROADMAP.md              Version plan
```

The model is implemented twice — once in `index.html` so the tool needs no backend, once in `generate_report.py` so the PDF needs no browser. `tests/parity.test.py` diffs the two engines field by field; if you change the math in one, change it in both or the suite fails.

---

MIT License · Built by an engineer who got tired of OOM crashes.
