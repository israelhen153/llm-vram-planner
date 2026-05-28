# Contributing to LLM VRAM Planner

The most impactful contribution is **benchmark data**. Every new entry makes the tool more accurate for everyone.

## Adding benchmark data

### What we need

Real measured throughput (tokens/sec) from vLLM running on specific hardware. Not theoretical, not from marketing materials — actual numbers from your deployment or benchmark run.

### How to measure

```bash
# Standard vLLM benchmark command
python -m vllm.entrypoints.openai.api_server \
    --model YOUR_MODEL \
    --dtype auto \
    --gpu-memory-utilization 0.90

# Then benchmark with:
python -m vllm.entrypoints.openai.benchmark \
    --model YOUR_MODEL \
    --num-prompts 1000 \
    --request-rate inf
```

Or use the ShareGPT benchmark:
```bash
python benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --model YOUR_MODEL \
    --dataset-name sharegpt \
    --num-prompts 500
```

### Entry format

Add your benchmark to `benchmarks/data.json` under the `data` key:

```json
"8b-a100-80": {
    "tokS": 3200,
    "src": "Your name or org",
    "note": "Llama 3.1 8B, BF16, batch throughput, 500 ShareGPT prompts",
    "prec": "bf16",
    "date": "2026-05",
    "url": "https://link-to-your-benchmark-post-or-repo"
}
```

### Key format

The key follows the pattern: `{paramBucket}-{gpuKey}`

**Parameter buckets:**
| Bucket | Range |
|--------|-------|
| `4b` | 1-4B params |
| `7b` | 5-7B |
| `8b` | 8-10B |
| `14b` | 11-14B |
| `27b` | 15-30B |
| `32b` | 31-40B |
| `70b` | 41-80B |
| `100b` | 81-130B |
| `400b` | 131-500B |
| `671b` | 500B+ |

**GPU keys:**
| Key | GPU |
|-----|-----|
| `t4-16` | T4 16GB |
| `rtx4090-24` | RTX 4090 24GB |
| `a100-40` | A100 40GB |
| `l40s-48` | L40S 48GB |
| `a100-80` | A100 80GB |
| `h100-80` | H100 80GB |
| `b200-192` | B200 192GB |

If your GPU isn't listed, add it and document the specs in your PR description.

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `tokS` | Yes | Output tokens per second (integer) |
| `src` | Yes | Who ran the benchmark — your name, org, or publication |
| `note` | Yes | Model name, precision, batch/single-user, context length, vLLM version, num prompts |
| `prec` | Yes | Weight precision: `bf16`, `fp8`, `int4`, `q4`, `q8` |
| `date` | Yes | When the benchmark was run: `YYYY-MM` |
| `url` | Recommended | Link to your blog post, repo, or benchmark report |

### What makes a good entry

- **Reproducible conditions**: model name, precision, batch size or concurrent users, context length, vLLM version
- **Sustained throughput**: not peak burst, but sustained over 100+ prompts
- **Standard workload**: ShareGPT prompts are the community standard. Random prompts also work. Synthetic single-token prompts are not useful.
- **One GPU per entry**: multi-GPU benchmarks should note TP size in the `note` field

### What to avoid

- Marketing numbers from GPU vendors
- Numbers from other engines (SGLang, TensorRT-LLM) — this is a vLLM-focused tool
- Theoretical bandwidth calculations — we already compute those, we need measured reality
- Numbers without conditions (a bare "3000 tok/s" with no model or GPU is useless)

## Submitting

1. Fork the repo
2. Add your entry to `benchmarks/data.json`
3. Open a PR with the title: `benchmark: {model} on {GPU} — {tok/s} tok/s`
4. In the PR description, include:
   - The exact `vllm serve` command you used
   - The benchmark command
   - vLLM version (`python -c "import vllm; print(vllm.__version__)"`)
   - Driver version (`nvidia-smi`)
   - Any relevant context (PCIe vs SXM, cooling, etc.)

## Other contributions

- **Bug reports**: [open an issue](https://github.com/YOURUSERNAME/llm-vram-planner/issues) with steps to reproduce
- **Feature requests**: open an issue with the use case, not just the feature
- **Code changes**: open an issue first to discuss before submitting a PR
- **Model presets**: add to the `PR` object in `index.html` — include all architecture params

## Code style

The tool is a single HTML file. Keep it that way. No build step, no npm, no framework. CSS at the top, HTML in the middle, JS at the bottom. If you're adding a feature, it should work offline.
