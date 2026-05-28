# 🧮 LLM VRAM Planner

**Plan GPU memory, throughput, and deployment for LLM inference and training.**

> **[Launch the tool →](https://israelhen153.github.io/llm-vram-planner/)**

A single-page, offline-capable tool for sysadmins and ML engineers planning LLM deployments. Generates vLLM deployment commands, compares GPU costs across providers, and exports procurement-ready PDF reports.

---

## What it does

### Inference planning
- **VRAM breakdown** — weights, KV cache, activations, overhead per GPU
- **Multi-GPU support** — tensor parallel, NVLink vs PCIe, condensed view at 5+ GPUs, TP/DP split at 8+ with topology warnings
- **Reverse calculator** — max context length, max concurrent users at 4K/8K context
- **KV cache dtype** — BF16 vs FP8 (doubles effective context on same hardware)
- **MoE-aware** — shared vs routed experts (DeepSeek-style), active parameter ratio
- **Quantization levels** — BF16, FP8, INT4/AWQ/GPTQ, GGUF Q2–Q8

### Training estimation
- **Full fine-tune / LoRA / QLoRA** — per-component VRAM breakdown
- Optimizer states (AdamW, SGD, 8-bit Adam), gradient checkpointing
- Trainable parameter count, batch size × sequence length scaling

### Deployment
- **vLLM command generator** — `--tensor-parallel-size`, `--quantization`, `--kv-cache-dtype`, `--enable-expert-parallel`, `--max-model-len`
- **TP/DP split logic** — auto-splits at >8 GPUs with warnings about multi-node topology
- **Workload presets** — chat, coding assistant, coding agent, RAG, document analysis, batch, real-time API

### Cost & throughput
- **3-tier pricing** — hyperscaler (AWS/GCP/Azure), specialized (Lambda/CoreWeave/RunPod), spot/marketplace (Vast.ai)
- **Published benchmark data** — real measured tok/s alongside theoretical estimates, with source citations
- **Theoretical throughput** — memory-bandwidth-bound decode estimate

### Import & export
- **HuggingFace fetch** — paste a model ID, auto-fills architecture from config.json
- **File upload** — drag-drop config.json or model.safetensors.index.json
- **URL state sharing** — every config serialized into the URL hash for team collaboration
- **Comparison snapshots** — save and compare multiple configs side by side
- **PDF report card** — procurement-ready document via `generate_report.py`
- **Copy report** — markdown summary to clipboard

---

## Quick start

### Use online
**[israelhen153.github.io/llm-vram-planner](https://israelhen153.github.io/llm-vram-planner/)**

### Use offline
Download `index.html` and open it in any browser. No server needed. No internet needed (icon fonts are the only CDN dependency — everything works without them, you just lose the icons).

### Generate a PDF report
```bash
pip install reportlab
python generate_report.py --preset gemma4-26b --gpu a100-40 --prec int4 --fp8-kv
```

Or interactive:
```bash
python generate_report.py
```

---

## Available model presets

| Family | Models |
|--------|--------|
| Llama | 3.1 8B, 3.1 70B, 3.3 70B, 4 Scout 109B MoE |
| Qwen | 3 8B, 3 30B MoE, 2.5 Coder 32B, 3.5 397B MoE, 3 Coder 480B MoE |
| Gemma | 4 E4B, 4 26B MoE, 4 31B Dense |
| DeepSeek | V3 671B MoE, R1 671B MoE |
| Mistral | Small 4 24B, Large 123B |

**Any model** can be imported via HuggingFace ID or config.json upload — presets are just shortcuts.

---

## Benchmark data

Published throughput measurements from third-party benchmarks. Community contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

| Model size | GPU | tok/s | Source |
|-----------|-----|-------|--------|
| 7-8B | A100 80GB | 3,200-3,363 | DatabaseMart |
| 8B | H100 80GB | 12,500 | MorphLLM |
| 14B | A100 80GB | 3,004 | DatabaseMart |
| 27B | B200 192GB | 9,500 | Google Cloud |
| 8B | RTX 4090 | 104 | Hardware Corner |

See [`benchmarks/data.json`](benchmarks/data.json) for the full dataset with conditions and source URLs.

---

## Project structure

```
llm-vram-planner/
├── index.html              # The tool (single-file, no build step)
├── generate_report.py      # PDF report generator (needs reportlab)
├── benchmarks/
│   └── data.json           # Benchmark data (community-contributed)
├── CONTRIBUTING.md          # How to contribute benchmarks
├── LICENSE                  # MIT
└── README.md
```

---

## Contributing

The most valuable contribution is **benchmark data**. If you've measured vLLM throughput on specific hardware, add it to [`benchmarks/data.json`](benchmarks/data.json). See [CONTRIBUTING.md](CONTRIBUTING.md) for the format.

Bug reports, feature requests, and PRs are welcome via [Issues](https://github.com/israelhen153/llm-vram-planner/issues).

---

## License

MIT — use it however you want.
