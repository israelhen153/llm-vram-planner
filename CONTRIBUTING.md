# Contributing to LLM VRAM Planner

The most impactful contribution is **benchmark data**. Every new entry makes the tool more accurate for everyone.

## Adding benchmark data

### What we need

Real measured throughput (tokens/sec) from vLLM running on specific hardware. Not theoretical, not from marketing materials — actual numbers from your deployment or benchmark run.

### How to measure

Serve the model, then drive it with vLLM's own benchmark CLI:

```bash
# Terminal 1 — serve
vllm serve YOUR_MODEL \
    --dtype auto \
    --gpu-memory-utilization 0.90

# Terminal 2 — measure against the running server (aggregate / "batch")
vllm bench serve \
    --backend openai-chat \
    --model YOUR_MODEL \
    --request-rate inf
```

For offline batch throughput, with no server involved:

```bash
vllm bench throughput \
    --model YOUR_MODEL \
    --dataset-name sharegpt \
    --num-prompts 500
```

`vllm bench` replaced the standalone scripts that used to live in vLLM's
`benchmarks/` directory — that directory is now a pointer to the CLI, and there
has never been a `vllm.entrypoints.openai.benchmark` module. The full flag list
for each subcommand is in the
[CLI reference](https://docs.vllm.ai/en/latest/cli/bench/serve.html)
(`serve`, `throughput` and `latency` are the three you are likely to want).

For a single-user number — `"mode": "single"` below — send one request at a
time and time it. Any client will do; what matters is that nothing else is
running on the GPU.

### Entry format

Add your benchmark to `benchmarks/data.json` under the `data` key:

```json
"8b-a100-80": {
    "tokS": 3200,
    "mode": "batch",
    "src": "Your name or org",
    "note": "Llama 3.1 8B, BF16, batch throughput, 500 ShareGPT prompts",
    "prec": "bf16",
    "date": "2026-05",
    "url": "https://link-to-your-benchmark-post-or-repo"
}
```

### `mode` is the field to get right

`mode` says what your number *is*, and it matters more than any other field:

- **`"batch"`** — total tokens/sec across all concurrent requests. What `vllm bench` and most published benchmarks report.
- **`"single"`** — tokens/sec for one request with nothing else running. What a single user experiences.

These differ by 50–100× on the same hardware, because continuous batching amortises one weight read across the whole batch. The tool compares your entry only against the matching estimate, so a mislabelled `mode` produces a number that looks wildly wrong to everyone who selects that model and GPU.

If you ran `vllm bench throughput` or sent concurrent requests, it's `batch`. If you sent one request and timed it, it's `single`.

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

`70b` is the largest bucket the tool can select. A model above 80B is compared
against the nearest bucket below it, labelled as a different model class and
deliberately not scored — so an entry keyed `100b-…` or `671b-…` would sit in
the file unreachable. If you have measurements for a model that size, open an
issue: the bucket boundaries live in `findBenchmark()` and `BUCKET_PARAMS` in
`index.html`, and adding one is a two-line change plus a row here.

**GPU keys** are whatever `data/gpus.json` currently holds — that file is the
catalog, and this document deliberately does not keep a second copy of it to go
stale. To see the list:

```bash
python3 -c "import json; print(*json.load(open('data/gpus.json'))['data'])"
```

**If your GPU isn't in the catalog**, add it there rather than in the tool:

1. Add a row to `data/gpus.json`. Every field is documented in that file's
   `_meta.schema`; cite your source for capacity, bandwidth and dense TFLOPS in
   the PR description.
2. Run `python3 tools/sync_data.py`. `index.html` and `generate_report.py` each
   carry a generated copy of the catalog — the tool has no build step and a
   browser on `file://` cannot fetch a sibling JSON — and that script is what
   regenerates them. Never hand-edit the blocks between the `GPU_TABLE:BEGIN` /
   `GPU_TABLE:END` markers.
3. Run `./tests/run.sh`. The suite fails if the generated blocks and the JSON
   disagree, which is what catches a forgotten step 2.

A benchmark entry naming a GPU key that does not exist is dead data: the suite
asserts every key resolves to a catalog row.

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `tokS` | Yes | Output tokens per second (integer) |
| `mode` | Yes | `batch` (aggregate across concurrent requests) or `single` (one request alone). See above — this one matters most |
| `src` | Yes | Who ran the benchmark — your name, org, or publication |
| `note` | Yes | Model name, precision, batch/single-user, context length, vLLM version, num prompts |
| `prec` | Yes | Weight precision: `bf16`, `fp8`, `int4`, `q4`, `q8` |
| `date` | Yes | When the benchmark was run: `YYYY-MM` |
| `url` | Yes, unless `estimated` | Link to your blog post, repo, or benchmark report |
| `estimated` | Only if true | Set `true` if the number was extrapolated rather than measured on this exact hardware. Entries without a `url` must set this — `tests/run.sh` enforces it |

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

- **Bug reports**: [open an issue](https://github.com/israelhen153/llm-vram-planner/issues) with steps to reproduce
- **Feature requests**: open an issue with the use case, not just the feature
- **Code changes**: open an issue first to discuss before submitting a PR
- **Model presets**: add to `MODEL_PRESETS` in `index.html` **and** `PRESETS` in
  `generate_report.py` — the model is implemented twice on purpose and
  `tests/parity.test.py` compares the two, so a preset added to one engine fails
  the suite. Include every architecture parameter: layers, KV heads, head dim,
  and the attention regime with the field that gives it meaning (`swa_win` +
  `swa_local`, or `mla_dim`).

## Code style

The tool is a single HTML file. Keep it that way. No build step, no npm, no framework. CSS at the top, HTML in the middle, JS at the bottom. If you're adding a feature, it should work offline.

Two blocks in `index.html` and one in `generate_report.py` are generated from
JSON by `tools/sync_data.py` and marked with `GPU_TABLE:BEGIN` /
`BENCHMARK_DATA:BEGIN` comments. Edit the JSON and re-run the script; the tests
compare the two and will tell you if you edited the wrong one.

Run `./tests/run.sh` before opening a PR. It needs `node` and `python3` and
nothing else.
