# 🧮 LLM VRAM Planner

**Stop guessing if your model fits. Know before you deploy.**

I was deploying vLLM on an air-gapped server and spent 3 hours fighting CUDA version mismatches, wrong PyTorch wheels, and OOM crashes — all because there was no tool that answered "does this model fit on my GPU, and what flags do I pass to vLLM?" in one place. So I built one.

> **[Use it now →](https://YOURUSERNAME.github.io/llm-vram-planner/)**  ·  Single HTML file  ·  Works offline  ·  No signup

<!-- TODO: Replace with actual screenshot (1200×630px, engineer mode, configured with a model) -->
<!-- ![LLM VRAM Planner](assets/screenshot.png) -->

---

## What it does

**Pick a model → pick a GPU → get the answer.** VRAM breakdown, deployment command, cost estimate.

It also does things no other VRAM calculator does:

- **Generates the actual `vllm serve` command** — tensor parallel, quantization, KV cache dtype, expert parallel, max context length. Copy-paste and run.
- **3-tier cost comparison** — hyperscaler (AWS/GCP/Azure), specialized (Lambda/CoreWeave), and spot (Vast.ai) pricing side by side.
- **Training estimation** — full fine-tune, LoRA, QLoRA. See exactly what's eating your VRAM: weights, gradients, optimizer states, activations.
- **Executive mode** — one-click toggle. Shows verdict, cost range, and max users. Hand the URL to your manager.
- **Published benchmark data** — real measured tok/s alongside theoretical estimates, with sources.
- **Import any model** — paste a HuggingFace ID or drop a `config.json`. Works for models that aren't in any preset list.
- **PDF report card** — export a procurement-ready document.
- **Works offline** — single HTML file, no backend, no internet required after first load.

---

## Quick start

### Use online
**[YOURUSERNAME.github.io/llm-vram-planner](https://YOURUSERNAME.github.io/llm-vram-planner/)**

### Use offline
Download `index.html`. Open in any browser. Done.

### Generate a PDF report
```bash
pip install reportlab
python generate_report.py --preset gemma4-26b --gpu a100-40 --prec int4 --fp8-kv
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

Bug reports and feature requests welcome via [Issues](https://github.com/YOURUSERNAME/llm-vram-planner/issues).

---

## Project structure

```
index.html              The tool (single file, no build step)
generate_report.py      PDF report generator
benchmarks/data.json    Community benchmark data
CONTRIBUTING.md         How to add benchmarks
ROADMAP.md              Version plan
```

---

MIT License · Built by an engineer who got tired of OOM crashes.
