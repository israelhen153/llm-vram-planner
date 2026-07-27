# The model

Every formula the tool uses, why it has that shape, and where it breaks down.

If you only read one section, read [Two throughput numbers](#two-throughput-numbers) — conflating them is the most common error in capacity planning, and it was this tool's own biggest bug.

---

## 1. VRAM

Four components. Only the first two are arithmetic; the other two are heuristics.

```
total = weights + kv_cache + activations + overhead
```

### Weights

```
weights_bytes = params × bytes_per_param
```

Exact. `bytes_per_param` is 2 for BF16/FP16, 1 for FP8, ~0.5 for 4-bit.

The 4-bit figure is slightly optimistic. AWQ and GPTQ at group size 128 store 4 bits per weight plus one FP16 scale and one INT4 zero-point per group: `(4 × 128 + 16 + 4) / 128 / 8 ≈ 0.52` bytes. GGUF K-quants carry more metadata still, which is why `Q4_K_M` is listed at 0.63 rather than 0.5.

**For MoE, all experts are resident.** A 671B model with 5% active parameters still needs 671B worth of weights in VRAM. Only *compute* and *bandwidth* scale with the active fraction. Getting this backwards understates a DeepSeek deployment by 20×.

### KV cache — three regimes

This is where most calculators go wrong, and where the largest errors live.

**Standard (MHA / GQA)** — every layer caches a key and a value for every token:

```
kv_bytes = 2 × layers × kv_heads × head_dim × bytes_per_value × ctx
```

The `2` is K and V. `kv_heads` is *key-value* heads, not attention heads — GQA shares KV across query heads, which is the whole point of it. Llama 3 8B has 32 attention heads but only 8 KV heads, so this term is 4× smaller than MHA would be.

**Sliding-window (SWA)** — Gemma 3/4, Llama 4's "chunked attention". Layers alternate between local and global. Local layers only ever retain `window` tokens:

```
kv_bytes = 2 × kv_heads × head_dim × bytes × (L_local × min(ctx, window) + L_global × ctx)
```

Gemma 4 26B is 25 local layers at a 1024 window plus 5 global. At 128K context, charging all 30 layers as full attention overstates KV by about 5×. Below the window the two formulas agree exactly — the curve only bends once `ctx > window`.

**MLA (Multi-head Latent Attention)** — DeepSeek V2/V3/R1 compress K and V into a single latent vector per layer:

```
kv_bytes = layers × (kv_lora_rank + qk_rope_head_dim) × bytes × ctx
```

For DeepSeek V3 that is `61 × (512 + 64) = 61 × 576`. Note there is **no factor of 2** and **no head multiplier** — there is one latent, not a K/V pair per head. This is why DeepSeek serves 128K context on hardware that could not do it under GQA.

### Activations

```
activations = active_params × 2 bytes × 0.01
```

A flat 1% heuristic, floored at 0.1 GiB. Real activation memory depends on batch composition, chunked-prefill settings and the attention kernel. This is the least principled number in the tool; it is small enough relative to weights and KV that it rarely changes a verdict.

### Overhead

1.5 GiB per GPU for the CUDA context, plus 0.2–0.3 GiB per additional GPU for NCCL buffers. Also heuristic. Real CUDA context is roughly 0.3–0.8 GiB; the rest is headroom for allocator fragmentation and the framework itself.

### Units: GiB, not GB

`nvidia-smi` reports an "80 GB" H100 as 81559 MiB — that is 79.6 GiB, or 85.5 × 10⁹ bytes. A card's marketing number behaves as GiB. Computing model memory in decimal GB and comparing it against that label mixes two units and makes every fit check ~7% pessimistic. Everything here is GiB (2³⁰ bytes).

---

## 2. Two throughput numbers

**These are different quantities and must never be compared to each other.** Published benchmarks almost always report the second; intuition almost always reaches for the first.

Decode is memory-bandwidth bound. Each step reads every active weight once, plus the KV cache of every sequence in the batch. With batch size B:

```
tok/s = (B × achieved_bandwidth) / (active_weight_bytes + B × kv_bytes_per_seq)
```

Both numbers fall out of this one formula:

- **B = 1** → single-stream decode. What one user feels. Roughly `bandwidth / weights`.
- **B = batch** → aggregate throughput. What the server delivers in total.

At B=1 the weight term dominates, so throughput is `bandwidth ÷ weight_bytes`. As B grows, that single weight read is amortised across the whole batch, so aggregate throughput climbs — until the `B × kv_bytes` term starts to dominate the denominator and it flattens out.

**This is why batch benchmarks land 50–100× above single-stream numbers.** An 8B model on an H100 does ~144 tok/s for one user and ~13,800 tok/s aggregate at a saturated batch. Neither is wrong. They answer different questions.

### Why the denominator matters

The `B × kv_bytes_per_seq` term is why long context destroys batch throughput. At 1K context an 8B model on an H100 fits hundreds of sequences and the weight read amortises beautifully. At 32K, KV traffic dominates and adding sequences stops helping. This is also why the tool caps batch by what KV cache can hold — you cannot batch sequences you cannot store.

### MBU — memory bandwidth utilisation

```
achieved_bandwidth = peak_bandwidth × MBU,   MBU = 0.70
```

Real decode kernels sustain roughly 60–80% of peak HBM bandwidth. Peak is a number from a spec sheet; nobody hits it. Omitting this factor makes single-user estimates about 2× optimistic — which is exactly what this tool did before the fix.

### The compute ceiling

Decode costs about 2 FLOPs per active parameter per token, so aggregate throughput cannot exceed:

```
ceiling = MFU × peak_dense_FLOPS / (2 × active_params),   MFU = 0.35
```

This rarely binds for decode — for an 8B on H100 it sits around 22,000 tok/s, above the measured 12,500 — but it stops the batch model growing without limit.

**TFLOPS figures here are dense, not sparse.** NVIDIA datasheets headline the 2:4-structured-sparsity number: the H100 SXM's "1,979 teraFLOPS" carries a footnote reading *"With sparsity"*. LLM inference does not use structured sparsity, so the real dense figure is half that — 989.5. Using the headline number would overstate the ceiling 2×. This trips up a lot of secondary sources, and several web results during this project's research got it wrong.

### TTFT is a different phase entirely

Prefill is **compute-bound**, not bandwidth-bound. The whole prompt is processed in one parallel pass:

```
ttft = 2 × active_params × prompt_tokens / (MFU × peak_FLOPS)
```

Deriving TTFT from decode speed — as if tokens were generated one at a time during prefill — understates it by roughly 10×. That was another real bug here.

---

## 3. Prefix caching

vLLM V1 enables automatic prefix caching by default (`--no-enable-prefix-caching` opts out). It has two effects, and most summaries only mention the first.

**Latency.** A cached prefix is not prefilled again, so warm TTFT covers only the uncached remainder. The first request still pays full price — quoting only the warm number flatters the system.

**Memory.** This is the one people miss. APC deduplicates the shared prefix across concurrent sequences: it is stored once, not once per request. Sixty-four users sharing an 8K system prompt on an 8B model recovers about 63 GiB of KV cache.

```
kv_total = shareable(prefix) + concurrency × (kv_per_seq − shareable(prefix))
```

**APC never changes decode speed.** It touches prefill only. "Prefix caching made generation faster" is the standard wrong summary — if someone says it, they have not read what it does.

**Under SWA, only global layers share.** Local layers hold a rolling window over the most recent tokens, so a system prompt at position 0 has already slid out of them. There is nothing left to share. Applying prefix dedup across all layers would overstate the saving on exactly the models where SWA matters most.

---

## 4. Parallelism

**Tensor parallel** splits weights and KV cache across N GPUs; each holds `1/N`. Activations, CUDA context and NCCL buffers are replicated on every GPU, so total footprint across the cluster is *larger* than a single-device estimate of the same model.

Interconnect matters: NVLink costs ~15% versus a single GPU, PCIe ~45%. TP is communication-heavy — every layer needs an all-reduce — so PCIe-only multi-GPU is much worse than the VRAM arithmetic suggests.

Above 8 GPUs the tool splits into TP × DP, because TP beyond a single NVLink domain crosses slower links. That split is a starting point, not an answer; the right topology depends on whether you are optimising latency or throughput.

**Not modelled:** pipeline parallel, multi-node, expert parallel placement beyond the `--enable-expert-parallel` flag.

---

## 5. Hard questions

Worth being able to answer without hedging.

**"Your throughput number is way off from my benchmark."**
First question back: single-stream or aggregate? They differ by 50–100×. If aggregate, was the GPU saturated? The tool compares against benchmarks at full batch because that is how benchmarks are run. If it is still off, the roofline runs 1.1–2.3× optimistic against real batch serving — schedulers lose time that physics does not.

**"Why 70% MBU and 35% MFU?"**
Observed ranges, not derived constants. MBU for decode sits at 60–80% across published measurements; MFU for batched decode is lower than training's 40–50% because decode GEMMs have poor arithmetic intensity. They are the two least defensible numbers in the model, which is why the README states them explicitly rather than burying them.

**"Isn't a roofline just an upper bound?"**
Yes, and it is labelled as a ceiling in the UI. Single-stream lands within ~4% of measurements because at B=1 the model is genuinely bandwidth-bound and there is little else to lose. Aggregate runs optimistic because real scheduling, prefill interleaving and preemption are not modelled.

**"Why is your KV cache smaller than every other calculator for Gemma?"**
Because Gemma 4 interleaves 25 sliding-window layers with 5 global ones. Most calculators charge all 30 as full attention. At 128K that is a 5× overstatement. The config.json is the authority: `layer_types` lists `sliding_attention` and `full_attention` per layer.

**"DeepSeek has 128 KV heads — why is your KV cache so small?"**
DeepSeek uses MLA. The `num_key_value_heads` field is there for config compatibility but the model does not cache per-head K and V; it caches one 576-dimensional latent per layer. Applying the GQA formula to it is not an approximation, it is the wrong formula.

**"Does quantising weights to INT4 shrink the KV cache?"**
No. Weight precision and KV precision are independent. `--kv-cache-dtype fp8` halves KV; `--quantization awq` does not touch it. At long context with high concurrency, KV frequently exceeds weights, so the KV dtype is the more consequential lever.

**"Can I just use the numbers straight for procurement?"**
VRAM, yes, with ~10% headroom — and do not plan to run at 99% utilisation. Throughput, no: use it to pick hardware, then benchmark the real workload before committing spend.

---

## 6. Verifying any of this

```bash
./tests/run.sh
```

51 assertions on the model math, plus a field-by-field diff between the JavaScript in `index.html` and the Python in `generate_report.py` — the model is implemented twice so that the web tool needs no backend and the PDF needs no browser, and the parity suite exists because those two copies silently drifted once already.

Architecture values in `MODEL_PRESETS` were read from each model's `config.json` on HuggingFace on 2026-07-27. If you think one is wrong, that file is the authority and a PR correcting it is welcome.
