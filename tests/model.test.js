/* Tests for the VRAM and throughput math in index.html.
 *
 * The tool is deliberately a single dependency-free HTML file, so there is no
 * module to import. This extracts computeInference() from the source and runs
 * it directly — if the function is renamed or its signature changes, these
 * tests fail loudly rather than silently passing on stale code.
 *
 * Run:  node tests/model.test.js
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');

const ROOT = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const benchmarks = JSON.parse(fs.readFileSync(path.join(ROOT, 'benchmarks', 'data.json'), 'utf8'));

/* ---- extract computeInference() from the single-file source ---- */
const start = html.indexOf('function computeInference(state) {');
assert.notStrictEqual(start, -1, 'computeInference() not found in index.html');
const end = html.indexOf('\n}\n', start);
assert.notStrictEqual(end, -1, 'could not find end of computeInference()');

// Module-level constants the function closes over. Read from source rather than
// redefined here, so a change to the real value cannot silently pass these tests.
const gibDecl = html.match(/^const GIB = .+;$/m);
assert.ok(gibDecl, 'GIB constant not found in index.html');
const perfDecl = html.match(/^const PERF = \{[\s\S]*?\n\};$/m);
assert.ok(perfDecl, 'PERF constant not found in index.html');

const computeInference = new Function(
  `${gibDecl[0]}\n${perfDecl[0]}\n${html.slice(start, end + 2)}; return computeInference;`
)();
assert.strictEqual(new Function(`${gibDecl[0]}; return GIB;`)(), 1024 ** 3, 'GIB must be 2^30');
const PERF = new Function(`${perfDecl[0]}; return PERF;`)();

/* ---- GPU specs, read from the GPU_TABLE the UI builds its options from ----
   This used to scrape the <option> markup, which no longer exists — the options
   are generated from GPU_TABLE at load time. Reading the table directly is also
   strictly safer: the old regex accepted only digits, dots and pipes, so a single
   non-numeric field silently dropped a card and surfaced much later as a
   misleading "unknown GPU" failure. */
const gpuTableDecl = html.match(/^const GPU_TABLE = \{[\s\S]*?\n\};$/m);
assert.ok(gpuTableDecl, 'GPU_TABLE constant not found in index.html');
const GPU_TABLE = new Function(`${gpuTableDecl[0]}; return GPU_TABLE;`)();

/* getGpuSpec() hands computeInference() the no-space form ("H100 80GB") while the
   table and the dropdown label carry "H100 80 GB". Mirror that derivation rather
   than hand-maintaining a second list of names — the tests address a GPU by the
   name the engine actually sees. */
const displayName = (g) => g.name.replace(/ GB$/, 'GB');
const GPUS = {};
for (const g of Object.values(GPU_TABLE)) GPUS[displayName(g)] = g;

/* Benchmark keys are `<params>b-<gpu slug>`, and the slug half is a GPU_TABLE key
   by construction. This asserts rather than skipping: an unmatched slug used to be
   silently dropped, which is exactly how dead benchmark data survives unnoticed. */
const gpuForKey = (key) => {
  const m = key.match(/^(\d+)b-(.+)$/);
  assert.ok(m, `benchmark key "${key}" is not <params>b-<gpu slug>`);
  const g = GPU_TABLE[m[2]];
  assert.ok(g, `benchmark key "${key}" names GPU "${m[2]}", which is not in GPU_TABLE`);
  return { params: Number(m[1]), slug: m[2], gpu: displayName(g), gb: g.gb };
};

const state = (o = {}) => {
  const gpu = GPUS[o.gpu || 'H100 80GB'];
  assert.ok(gpu, `unknown GPU ${o.gpu}`);
  return {
    params: 8, activePercent: 100, bytesPerParam: 2, layers: 32, kvHeads: 8,
    headDim: 128, sharedExperts: 0, contextLength: 8192, concurrency: 1,
    gpuCount: 1, hasNVLink: true, kvBytesPerValue: 2, presetKey: '', hfModelId: null,
    modelMaxCtx: 1048576,
    gpuGB: gpu.gb, gpuBandwidth: gpu.bw, gpuTFLOPS: gpu.tflops,
    gpuHyperCost: gpu.hyper, gpuSpecCost: gpu.spec, gpuSpotCost: gpu.spot,
    gpuName: o.gpu || 'H100 80GB',
    ...o,
  };
};

let pass = 0, fail = 0;
const test = (name, fn) => {
  try { fn(); console.log(`  ok   ${name}`); pass++; }
  catch (e) { console.log(`  FAIL ${name}\n       ${e.message}`); fail++; }
};
const between = (v, lo, hi, what) =>
  assert.ok(v >= lo && v <= hi, `${what}: expected ${lo}..${hi}, got ${Math.round(v)}`);

console.log('\nVRAM');
test('8B bf16 weights are 14.9 GiB (16e9 bytes), not 16', () => {
  // Reported in GiB to match nvidia-smi, so 8B * 2 bytes = 16e9 B = 14.90 GiB.
  between(computeInference(state()).weightsGB, 14.8, 15.0, 'weightsGB');
});
test('memory is reported in GiB, so GPU capacity is not understated', () => {
  // The old decimal-GB math made an 8B model look like 16/80 of an H100
  // when it is really 14.9/80 — a ~7% pessimistic bias on every fit check.
  const c = computeInference(state());
  assert.ok(c.weightsGB < 16, `expected GiB (<16), got ${c.weightsGB.toFixed(2)}`);
});
test('int4 quantisation quarters the weights vs bf16', () => {
  const bf16 = computeInference(state()).weightsGB;
  const int4 = computeInference(state({ bytesPerParam: 0.5 })).weightsGB;
  between(bf16 / int4, 3.99, 4.01, 'bf16:int4 ratio');
});
test('KV cache matches 2 * layers * kvHeads * headDim * bytes', () => {
  const c = computeInference(state());
  assert.strictEqual(c.kvBytesPerToken, 2 * 32 * 8 * 128 * 2);
});
test('FP8 KV cache halves KV footprint', () => {
  const a = computeInference(state()).kvCacheGB;
  const b = computeInference(state({ kvBytesPerValue: 1 })).kvCacheGB;
  between(a / b, 1.99, 2.01, 'bf16:fp8 KV ratio');
});
test('70B bf16 does not fit on one H100, does fit on four', () => {
  assert.strictEqual(computeInference(state({ params: 70, layers: 80, kvHeads: 8 })).fits, false);
  assert.strictEqual(computeInference(state({ params: 70, layers: 80, kvHeads: 8, gpuCount: 4 })).fits, true);
});
test('tensor parallel shards weights across GPUs', () => {
  const one = computeInference(state()).perGPU.weights;
  const four = computeInference(state({ gpuCount: 4 })).perGPU.weights;
  between(one / four, 3.99, 4.01, 'weight shard ratio');
});

console.log('\nKV cache — attention regimes');
// Gemma 4 26B A4B, from config.json: 30 layers, 8 kv heads, head_dim 256,
// sliding_window 1024, 25 sliding_attention + 5 full_attention.
const GEMMA = { params: 26, layers: 30, kvHeads: 8, headDim: 256, activePercent: 15,
                attnMode: 'swa', swaWindow: 1024, swaLocalLayers: 25 };
// DeepSeek V3: 61 layers, MLA with kv_lora_rank 512 + qk_rope_head_dim 64.
// Served at FP8 across 16 GPUs, which is roughly how it is actually deployed —
// 671B at BF16 is 1.25 TiB of weights and does not fit on 8x80GB at all.
const DEEPSEEK = { params: 671, layers: 61, activePercent: 5, sharedExperts: 1,
                   bytesPerParam: 1, gpuCount: 16, attnMode: 'mla', mlaLatentDim: 576 };

test('standard attention matches 2 * L * H * D * bytes * ctx', () => {
  const ctx = 8192;
  const c = computeInference(state({ contextLength: ctx }));
  const expected = 2 * 32 * 8 * 128 * 2 * ctx / (1024 ** 3);
  between(c.kvCacheGB, expected * 0.999, expected * 1.001, 'kvCacheGB');
});

test('SWA caps local layers at the window, global layers keep growing', () => {
  const ctx = 32768;
  const c = computeInference(state({ ...GEMMA, contextLength: ctx }));
  const perLayerToken = 2 * 8 * 256 * 2;
  const expected = perLayerToken * (25 * 1024 + 5 * ctx) / (1024 ** 3);
  between(c.kvCacheGB, expected * 0.999, expected * 1.001, 'SWA kvCacheGB');
});

test('SWA below the window behaves exactly like full attention', () => {
  // At ctx <= window nothing is capped yet, so the two must agree.
  const ctx = 512;
  const swa = computeInference(state({ ...GEMMA, contextLength: ctx })).kvCacheGB;
  const full = computeInference(state({ ...GEMMA, attnMode: 'standard', contextLength: ctx })).kvCacheGB;
  between(swa / full, 0.999, 1.001, 'SWA vs full at short context');
});

test('SWA saves multiples at long context — the whole point', () => {
  const ctx = 131072;
  const swa = computeInference(state({ ...GEMMA, contextLength: ctx })).kvCacheGB;
  const full = computeInference(state({ ...GEMMA, attnMode: 'standard', contextLength: ctx })).kvCacheGB;
  // 30 layers all growing vs 5 growing + 25 pinned at 1024.
  assert.ok(full / swa > 4.5, `expected >4.5x saving at 128K, got ${(full / swa).toFixed(2)}x`);
});

test('MLA caches one latent per layer, with no K/V pair or head multiplier', () => {
  const ctx = 8192;
  const c = computeInference(state({ ...DEEPSEEK, contextLength: ctx }));
  const expected = 61 * 576 * 2 * ctx / (1024 ** 3);
  between(c.kvCacheGB, expected * 0.999, expected * 1.001, 'MLA kvCacheGB');
});

test('MLA is far cheaper than treating DeepSeek as 128-head GQA', () => {
  const ctx = 8192;
  const mla = computeInference(state({ ...DEEPSEEK, contextLength: ctx })).kvCacheGB;
  const asGqa = computeInference(state({ ...DEEPSEEK, attnMode: 'standard',
                                         kvHeads: 128, headDim: 56, contextLength: ctx })).kvCacheGB;
  assert.ok(asGqa > mla * 20, `GQA formula should be >20x MLA, got ${(asGqa / mla).toFixed(1)}x`);
});

test('max context accounts for the SWA bend rather than dividing through', () => {
  // A linear divide on the effective per-token rate would understate reachable
  // context, because past the window only the global layers keep consuming.
  const c = computeInference(state({ ...GEMMA, contextLength: 32768 }));
  const perLayerToken = 2 * 8 * 256 * 2;
  const atMax = perLayerToken * (25 * 1024 + 5 * c.maxContextSingleUser);
  assert.ok(atMax <= c.freeForKVCache * (1024 ** 3) * 1.001,
    'max context must actually fit in the free KV budget');
  // And one more token must not fit.
  const atMaxPlus = perLayerToken * (25 * 1024 + 5 * (c.maxContextSingleUser + 1024));
  assert.ok(atMaxPlus > c.freeForKVCache * (1024 ** 3),
    'max context should be tight, not conservative');
});

test('max context is reachable under every regime', () => {
  for (const [name, over] of [['standard', {}], ['swa', GEMMA], ['mla', DEEPSEEK]]) {
    const c = computeInference(state({ ...over, contextLength: 8192 }));
    assert.ok(c.maxContextSingleUser > 0, `${name}: max context should be positive`);
    assert.ok(Number.isFinite(c.maxContextSingleUser), `${name}: max context must be finite`);
  }
});

console.log('\nPrefix caching and shared prefix');
test('a shared prefix is stored once, not once per concurrent request', () => {
  const base = { contextLength: 32768, concurrency: 64 };
  const without = computeInference(state({ ...base, sharedPrefix: 0 })).kvCacheGB;
  const with8k = computeInference(state({ ...base, sharedPrefix: 8192 })).kvCacheGB;
  // 64 users x 8K of identical preamble collapses to one copy.
  const perTok = 2 * 32 * 8 * 128 * 2;
  const expectedSaving = perTok * 8192 * 63 / (1024 ** 3);
  between(without - with8k, expectedSaving * 0.99, expectedSaving * 1.01, 'KV saved');
});

test('disabling prefix caching removes the saving entirely', () => {
  const base = { contextLength: 32768, concurrency: 64, sharedPrefix: 8192 };
  const on = computeInference(state({ ...base, prefixCaching: true }));
  const off = computeInference(state({ ...base, prefixCaching: false }));
  assert.ok(on.kvCacheGB < off.kvCacheGB, 'caching should reduce KV');
  assert.strictEqual(off.kvSavedByPrefixGB, 0);
});

test('under SWA only global layers share the prefix', () => {
  const base = { ...GEMMA, contextLength: 32768, concurrency: 32, sharedPrefix: 4096 };
  const c = computeInference(state(base));
  // 5 global layers share; the 25 local ones hold a rolling window, so the
  // prefix at position 0 has already slid out of them.
  const perLayerToken = 2 * 8 * 256 * 2;
  const expected = perLayerToken * 5 * 4096 * 31 / (1024 ** 3);
  between(c.kvSavedByPrefixGB, expected * 0.99, expected * 1.01, 'SWA prefix saving');
});

test('a prefix longer than the context is clamped, not counted twice', () => {
  const c = computeInference(state({ contextLength: 4096, concurrency: 8, sharedPrefix: 32768 }));
  assert.strictEqual(c.effectivePrefix, 4096);
  assert.ok(c.kvCacheGB > 0, 'KV must stay positive when prefix equals context');
});

test('prefix caching cuts TTFT but never decode speed', () => {
  const base = { contextLength: 32768, concurrency: 8, sharedPrefix: 8192 };
  const on = computeInference(state({ ...base, prefixCaching: true }));
  const off = computeInference(state({ ...base, prefixCaching: false }));
  assert.ok(on.ttftWarmMs < on.ttftColdMs, 'warm TTFT should beat cold');
  assert.strictEqual(off.ttftWarmMs, off.ttftColdMs, 'no caching means no warm path');
  // APC touches prefill only — this is the thing people get wrong about it.
  assert.strictEqual(on.singleStreamTokS, off.singleStreamTokS,
    'decode speed must be identical with and without prefix caching');
});

test('warm TTFT scales with the uncached remainder', () => {
  const c = computeInference(state({ contextLength: 32768, sharedPrefix: 24576 }));
  // 8K of 32K left to prefill, so warm should be about a quarter of cold.
  between(c.ttftWarmMs / c.ttftColdMs, 0.24, 0.26, 'warm:cold TTFT ratio');
});

test('more sequences fit once the prefix is shared', () => {
  const base = { contextLength: 16384, concurrency: 256, sharedPrefix: 8192 };
  const on = computeInference(state({ ...base, prefixCaching: true })).maxBatchByKV;
  const off = computeInference(state({ ...base, prefixCaching: false })).maxBatchByKV;
  assert.ok(on > off, `sharing should raise batch capacity: ${off} -> ${on}`);
});

console.log('\nThroughput — single-stream');
// Llama-3-8B bf16 on one H100 measures roughly 100-160 tok/s for one user.
test('8B bf16 on H100 lands in the observed 100-160 tok/s band', () => {
  between(computeInference(state()).singleStreamTokS, 100, 160, 'singleStreamTokS');
});
test('a 70B model is slower per user than an 8B on the same GPU', () => {
  const small = computeInference(state()).singleStreamTokS;
  const big = computeInference(state({ params: 70, layers: 80, gpuCount: 4 })).singleStreamTokS;
  assert.ok(big < small, `70B (${big}) should decode slower than 8B (${small})`);
});
test('single-stream speed is independent of requested concurrency', () => {
  const a = computeInference(state({ concurrency: 1 })).singleStreamTokS;
  const b = computeInference(state({ concurrency: 64 })).singleStreamTokS;
  assert.strictEqual(a, b);
});
test('missing NVLink costs throughput on multi-GPU', () => {
  const linked = computeInference(state({ gpuCount: 4 })).singleStreamTokS;
  const pcie = computeInference(state({ gpuCount: 4, hasNVLink: false })).singleStreamTokS;
  assert.ok(pcie < linked, 'PCIe should be slower than NVLink');
});
test('PCIe penalty worsens with GPU count', () => {
  // Decode all-reduces are small, so hop latency dominates and grows with the ring.
  const gap = (n) => computeInference(state({ gpuCount: n, hasNVLink: false })).singleStreamTokS
                   / computeInference(state({ gpuCount: n })).singleStreamTokS;
  assert.ok(gap(8) < gap(2),
    `8x PCIe/NVLink ratio (${gap(8).toFixed(3)}) should be below 2x (${gap(2).toFixed(3)})`);
});
test('NVLink penalty is flat within a domain', () => {
  const perGpu = (n) => computeInference(state({ gpuCount: n })).singleStreamTokS / n;
  const s2 = perGpu(2), s8 = perGpu(8);
  assert.ok(Math.abs(s8 - s2) <= s2 * 0.02,
    `per-GPU single-stream should be ~flat on NVLink: 2x=${Math.round(s2)}, 8x=${Math.round(s8)}`);
});

console.log('\nThroughput — aggregate');
test('aggregate exceeds single-stream once batching kicks in', () => {
  const c = computeInference(state({ concurrency: 64 }));
  assert.ok(c.aggregateTokS > c.singleStreamTokS * 5,
    `aggregate ${c.aggregateTokS} should far exceed single-stream ${c.singleStreamTokS}`);
});
test('aggregate equals single-stream at batch 1', () => {
  const c = computeInference(state({ concurrency: 1 }));
  between(c.aggregateTokS / c.singleStreamTokS, 0.98, 1.02, 'batch-1 ratio');
});
test('per-user speed degrades under load', () => {
  const c = computeInference(state({ concurrency: 64 }));
  assert.ok(c.perUserAtLoadTokS < c.singleStreamTokS,
    'each user should be slower when sharing the GPU');
});
test('KV cache caps the batch below an absurd concurrency request', () => {
  const c = computeInference(state({ concurrency: 100000 }));
  assert.strictEqual(c.batchLimitedByKV, true);
  assert.ok(c.effectiveBatch < 100000, 'effective batch must be clamped to what fits');
});
test('aggregate never exceeds the compute roofline', () => {
  const c = computeInference(state({ concurrency: 100000, contextLength: 1024 }));
  const ceiling = (PERF.nvidia.mfuDecode * GPUS['H100 80GB'].tflops * 1e12) / (2 * 8e9);
  assert.ok(c.aggregateTokS <= ceiling * 1.01,
    `aggregate ${c.aggregateTokS} exceeded compute ceiling ${Math.round(ceiling)}`);
});
test('longer context reduces the batch that fits', () => {
  const short = computeInference(state({ concurrency: 512, contextLength: 2048 })).effectiveBatch;
  const long = computeInference(state({ concurrency: 512, contextLength: 32768 })).effectiveBatch;
  assert.ok(long < short, `32K ctx batch (${long}) should be smaller than 2K ctx batch (${short})`);
});

console.log('\nTTFT');
test('TTFT grows with prompt length', () => {
  const a = computeInference(state({ contextLength: 2048 })).ttftMs;
  const b = computeInference(state({ contextLength: 32768 })).ttftMs;
  assert.ok(b > a * 8, `TTFT should scale with prompt: ${a}ms -> ${b}ms`);
});
test('8B on H100 at 8K prompt gives a plausible sub-second TTFT', () => {
  between(computeInference(state()).ttftMs, 100, 1000, 'ttftMs');
});
test('TTFT uses the prefill MFU, not decode\'s', () => {
  // Prefill is dense GEMM work; it runs at the prefill MFU, not decode's lower one.
  const expected = (2 * 8e9 * 8192) / (PERF.nvidia.mfuPrefill * GPUS['H100 80GB'].tflops * 1e12) * 1000;
  between(computeInference(state()).ttftMs, expected * 0.99, expected * 1.01,
    'ttftMs vs the prefill-MFU formula');
});

console.log('\nAgreement with published benchmarks');
// The point of the rewrite: compare each estimate against the matching mode.
// Order-of-magnitude agreement (0.25x-4x) is the bar for a planning tool.
const PREC_BYTES = { bf16: 2, fp8: 1, q4: 0.5, int4: 0.5 };

for (const [key, b] of Object.entries(benchmarks.data)) {
  if (b.estimated) continue; // only score against real measurements
  const { params, gpu } = gpuForKey(key);
  const gpuCount = key === '70b-h100-80' ? 2 : 1;
  // Benchmarks are run at short context with a full batch; mirror that.
  const s = state({
    gpu, params, gpuCount,
    bytesPerParam: PREC_BYTES[b.prec] ?? 2,
    layers: params >= 60 ? 80 : params >= 20 ? 48 : 32,
    contextLength: b.mode === 'single' ? 16384 : 1024,
    concurrency: b.mode === 'single' ? 1 : 1,
  });
  const c = computeInference(s);
  // Batch benchmarks are run saturated, so score against the saturated estimate.
  const ours = b.mode === 'single' ? c.singleStreamTokS : c.saturatedTokS;
  test(`${key} (${b.mode}) within 4x of ${b.tokS} tok/s`, () => {
    const ratio = ours / b.tokS;
    assert.ok(ratio >= 0.25 && ratio <= 4,
      `estimate ${ours} vs measured ${b.tokS} = ${ratio.toFixed(2)}x (want 0.25-4x)`);
  });
}

console.log('\nObserved-efficiency band');
test('the declared band matches the measured spread, and is no wider', () => {
  // Re-derive the band from the data so it cannot silently drift as entries land.
  const ratios = [];
  for (const [key, b] of Object.entries(benchmarks.data)) {
    if (b.estimated || b.mode !== 'batch') continue;
    const { params, gpu } = gpuForKey(key);
    const s = state({
      gpu, params, gpuCount: key === '70b-h100-80' ? 2 : 1,
      bytesPerParam: PREC_BYTES[b.prec] ?? 2,
      layers: params >= 60 ? 80 : params >= 20 ? 48 : 32,
      contextLength: 1024, concurrency: 1,
    });
    ratios.push(b.tokS / computeInference(s).saturatedTokS);
  }
  assert.ok(ratios.length >= 3, `need >=3 measured batch entries, got ${ratios.length}`);
  const lo = Math.min(...ratios), hi = Math.max(...ratios);
  const { obsLo, obsHi } = PERF.nvidia;
  // Two-sided deliberately. Asserting only that the data fits inside the band lets
  // the band be widened to fit anything, and wider is the flattering direction:
  // it makes the tool look like it predicted whatever was measured. README.md and
  // MODEL.md both promise this band tracks the evidence, so pin both edges.
  assert.ok(Math.abs(obsLo - lo) <= 0.02 && Math.abs(obsHi - hi) <= 0.02,
    `measured/ceiling ratios span ${lo.toFixed(2)}-${hi.toFixed(2)} but PERF.nvidia declares ` +
    `${obsLo}-${obsHi} — the band must track the data in both directions. Update obsLo/obsHi ` +
    `in index.html and generate_report.py to match the measurements.`);
});
test('the band scales the aggregate ceiling and nothing else', () => {
  const c = computeInference(state({ concurrency: 64 }));
  const { obsLo, obsHi } = PERF.nvidia;
  between(c.aggregateObservedLoTokS, c.aggregateTokS * (obsLo - 0.01), c.aggregateTokS * (obsLo + 0.01), 'band lo');
  between(c.aggregateObservedHiTokS, c.aggregateTokS * (obsHi - 0.01), c.aggregateTokS * (obsHi + 0.01), 'band hi');
  assert.ok(c.aggregateObservedHiTokS < c.aggregateTokS, 'the band must sit below the ceiling');
});

/* One parameter count per bucket that lands squarely inside it — used to probe
   the lookup from the data side. Mirrors BUCKET_PARAMS in index.html except for
   '8b' and '32b', where the bucket's representative value sits on a boundary. */
const BUCKET_PROBE = { '4b': 4, '7b': 7, '8b': 9, '14b': 13, '27b': 27, '32b': 35, '70b': 70 };

console.log('\nBenchmark lookup');
const fbStart = html.indexOf('function findBenchmark(');
assert.notStrictEqual(fbStart, -1, 'findBenchmark() not found');
const fbEnd = html.indexOf('\n}\n', fbStart);
const bucketDecl = html.match(/^const BUCKET_PARAMS = .+;$/m);
assert.ok(bucketDecl, 'BUCKET_PARAMS not found in index.html');
/* The inline block, not benchmarks/data.json. This test file used to substitute
   the JSON in place of the inline copy here, which meant the suite validated a
   table the browser never runs — and the two had drifted by 16 fields, including
   every `mode`, so the panel scored single-stream measurements against an
   aggregate estimate with nothing to catch it. The block is generated from the
   JSON now and tests/parity.test.py compares them; this reads what ships. */
const benchDecl = html.match(/^const BENCHMARK_DATA = \{[\s\S]*?\n\};$/m);
assert.ok(benchDecl, 'BENCHMARK_DATA constant not found in index.html');
const INLINE_BENCHMARKS = new Function(`${benchDecl[0]}; return BENCHMARK_DATA;`)();
const findBenchmark = new Function(
  `${benchDecl[0]}
   ${bucketDecl[0]}
   ${html.slice(fbStart, fbEnd + 2)}; return findBenchmark;`
)();

test('the inline benchmark table is the one in benchmarks/data.json', () => {
  assert.deepStrictEqual(INLINE_BENCHMARKS, benchmarks.data,
    'index.html and benchmarks/data.json disagree — run: python3 tools/sync_data.py');
});

test('buckets match the table documented in CONTRIBUTING.md', () => {
  // 7b covers 5-7B, 8b covers 8-10B. These were inverted.
  const a = findBenchmark(7, 'a100-80'), b = findBenchmark(8, 'a100-80');
  assert.ok(a.exact && b.exact, 'both should be exact matches');
  assert.strictEqual(a.data.tokS, benchmarks.data['7b-a100-80'].tokS);
  assert.strictEqual(b.data.tokS, benchmarks.data['8b-a100-80'].tokS);
});
test('an unmatched size degrades to the nearest entry on the SAME GPU', () => {
  const hit = findBenchmark(26, 'a100-40');
  assert.ok(hit, 'should fall back rather than return nothing');
  assert.strictEqual(hit.exact, false, 'must be flagged as inexact');
  assert.strictEqual(hit.requestedParams, 26);
  assert.ok(hit.nearestParams < 26, 'nearest available on this card is smaller');
  assert.strictEqual(hit.slower, true, 'a 26B model is slower than the smaller match');
});
test('the fallback picks the closest bucket, not just any', () => {
  // H100 has 8b/14b/27b/70b. 30B lands in the 27b bucket, which exists.
  assert.ok(findBenchmark(30, 'h100-80').exact, '30B is covered by the 27b bucket');
  // 35B lands in the 32b bucket, which has no H100 entry — a genuine gap.
  // Candidates are 27 (dist 8) and 70 (dist 35), so it must choose 27.
  const gap = findBenchmark(35, 'h100-80');
  assert.strictEqual(gap.exact, false);
  assert.strictEqual(gap.nearestParams, 27, 'must pick 27B over 70B on distance');
  assert.strictEqual(gap.slower, true, 'a 35B model is slower than a 27B measurement');
});
test('the fallback reports direction correctly in both directions', () => {
  // 4B on B200: only a 27b entry exists, so the nearest match is larger and
  // the user's model is the faster one.
  const smaller = findBenchmark(4, 'b200-192');
  assert.strictEqual(smaller.slower, false, '4B is faster than the 27B match');
});
test('fallback never crosses to a different GPU', () => {
  const hit = findBenchmark(4, 'b200-192');
  assert.ok(hit, 'B200 has a 27b entry to fall back to');
  assert.strictEqual(hit.exact, false);
  assert.strictEqual(hit.data.tokS, benchmarks.data['27b-b200-192'].tokS,
    'must stay on the B200, not borrow an A100 number');
});
test('a GPU with no data at all returns nothing', () => {
  assert.strictEqual(findBenchmark(8, 't4-16'), null);
  assert.strictEqual(findBenchmark(8, 'l40s-48'), null);
});
test('the cards the old name map could never reach now answer from the data', () => {
  /* T4, RTX 6000 Ada and RTX PRO 6000 matched no branch of the display-name
     chain findBenchmark() used to open with, so they returned null before a
     single key was examined — contributed measurements for them would not have
     rendered. They have no data today, so the answer is still "nothing", but it
     is now the data saying so: adding a key for one of them is enough to make it
     appear, which is what CONTRIBUTING.md promises. */
  for (const slug of ['t4-16', 'rtx6000ada-48', 'rtxpro-96']) {
    assert.ok(GPU_TABLE[slug], `${slug} missing from GPU_TABLE`);
    const keys = Object.keys(benchmarks.data).filter(k => k.endsWith(`-${slug}`));
    assert.strictEqual(keys.length, 0, `${slug} now has data — this test needs updating`);
    assert.strictEqual(findBenchmark(8, slug), null, `${slug} should report no benchmark`);
  }
});
test('a card only reachable by slug resolves once data exists for it', () => {
  /* The point of the change: the lookup is decided by the data, not by whether
     someone remembered to add a branch. Probe with a synthetic table so the
     assertion does not depend on what benchmarks/data.json happens to hold. */
  const withData = new Function(
    `const BENCHMARK_DATA = ${JSON.stringify({ '8b-rtxpro-96': { tokS: 4242, mode: 'batch', src: 'synthetic', note: 'test', prec: 'bf16' } })};
     ${bucketDecl[0]}
     ${html.slice(fbStart, fbEnd + 2)}; return findBenchmark;`
  )();
  const hit = withData(8, 'rtxpro-96');
  assert.ok(hit && hit.exact, 'RTX PRO 6000 should match its own entry');
  assert.strictEqual(hit.data.tokS, 4242);
});
test('sizes beyond every bucket still fall back rather than throwing', () => {
  const hit = findBenchmark(400, 'h100-80');
  assert.ok(hit && hit.exact === false, '400B has no bucket but should degrade');
  assert.strictEqual(hit.nearestParams, 70);
});
test('every benchmark key is reachable as an exact match', () => {
  // A key nothing can select exactly is dead data — the bucket table and the
  // data file have drifted apart.
  const PROBE = BUCKET_PROBE;
  for (const key of Object.keys(benchmarks.data)) {
    const { slug, gpu: name } = gpuForKey(key);
    const size = key.match(/^(\d+b)-/)[1];
    assert.ok(PROBE[size], `${key} has size bucket "${size}" with no probe value`);
    const hit = findBenchmark(PROBE[size], slug);
    assert.ok(hit && hit.exact && hit.data.tokS === benchmarks.data[key].tokS,
      `${key} is unreachable: probing ${PROBE[size]}B on ${name} did not return it exactly`);
  }
});

console.log('\nThe benchmark panel compares like with like');
/* The drift repro, kept as a test rather than a scratch script.
   8b-rtx4090-24 is a single-stream llama.cpp measurement: 104 tok/s for one
   user. The inline table carried no `mode`, the panel reads `b.mode || 'batch'`,
   so it scored that measurement against the *aggregate at full batch* estimate
   and reported 159% — a tool that says it is beating a published benchmark by
   60% when it is actually 61% short of it. The comparison the panel makes is
   reproduced here from the same inputs. */
test('an 8B on an RTX 4090 is scored single-stream at ~39%, not batch at ~159%', () => {
  const b = INLINE_BENCHMARKS['8b-rtx4090-24'];
  assert.strictEqual(b.mode, 'single', 'this entry is a single-user llama.cpp figure');
  const c = computeInference(state({ gpu: 'RTX 4090 24GB', params: 8, layers: 32 }));

  // The panel's own rule: mode picks which of our two numbers is comparable.
  const ours = b.mode === 'single' ? c.singleStreamTokS : c.saturatedTokS;
  const shown = Math.round(ours / b.tokS * 100);
  assert.strictEqual(ours, c.singleStreamTokS, 'must score against the single-stream estimate');
  assert.ok(shown >= 35 && shown <= 45, `single-stream comparison showed ${shown}%, expected ~39%`);

  // And what the missing field used to produce, so the flip is pinned in both
  // directions: if this ever matches the line above, the fix has come undone.
  const wasShown = Math.round(c.saturatedTokS / b.tokS * 100);
  assert.ok(wasShown > 140, `the pre-fix batch comparison should be ~159%, got ${wasShown}%`);
  assert.notStrictEqual(shown, wasShown, 'the two modes must not produce the same number');
});
test('every entry that says it is estimated arrives at the panel saying so', () => {
  // `estimated` is what renders "estimated, not measured" next to the figure.
  // It was absent from all 13 inline entries, so six extrapolated numbers were
  // presented exactly like measured ones.
  const flagged = Object.keys(INLINE_BENCHMARKS).filter(k => INLINE_BENCHMARKS[k].estimated);
  assert.strictEqual(flagged.length, 6, `expected 6 estimated entries, got ${flagged.length}`);
  for (const key of flagged) {
    const hit = findBenchmark(BUCKET_PROBE[key.match(/^(\d+b)-/)[1]], key.replace(/^\d+b-/, ''));
    assert.ok(hit, `${key} did not resolve`);
    assert.strictEqual(hit.data.estimated, true, `${key} lost its estimated flag on the way out`);
  }
});

console.log('\nBenchmark data integrity');
test('every entry declares a mode', () => {
  for (const [k, b] of Object.entries(benchmarks.data)) {
    assert.ok(b.mode === 'single' || b.mode === 'batch', `${k} has mode="${b.mode}"`);
  }
});
test('every unsourced entry is flagged as estimated', () => {
  for (const [k, b] of Object.entries(benchmarks.data)) {
    if (!b.url) assert.ok(b.estimated, `${k} has no source URL but is not marked estimated`);
  }
});

console.log('\nShared-URL round-trip');
// Extracted from source, not reimplemented — same reason as GIB above.
const tokenDecl = html.match(/^const precisionToken = .+;$/m);
assert.ok(tokenDecl, 'precisionToken not found in index.html');
const precisionToken = new Function(`${tokenDecl[0]}; return precisionToken;`)();

const precStart = html.indexOf('<select id="weight-precision"');
const precBlock = html.slice(precStart, html.indexOf('</select>', precStart));
const PREC_OPTS = [...precBlock.matchAll(/<option value="([\d.]+)"[^>]*data-q="([^"]*)"[^>]*>([^<]+)</g)]
  .map(m => ({ value: m[1], dataset: { q: m[2] }, label: m[3].trim() }));
// What loadURLHash() does: prefer the token, fall back to a bare byte count.
const resolve = val => PREC_OPTS.find(o => precisionToken(o) === val) || PREC_OPTS.find(o => o.value === val);

test('every precision option is parsed and has a distinct URL token', () => {
  const declared = (precBlock.match(/<option /g) || []).length;
  assert.strictEqual(PREC_OPTS.length, declared, `parsed ${PREC_OPTS.length} of ${declared} options`);
  const tokens = PREC_OPTS.map(precisionToken);
  assert.strictEqual(new Set(tokens).size, tokens.length,
    `AWQ, GPTQ and GGUF Q3_K_M are all 0.50 B/param — tokens collided: ${tokens.join(' ')}`);
});
test('every precision survives a share-and-reload round trip', () => {
  for (const want of PREC_OPTS) {
    const got = resolve(precisionToken(want));
    assert.strictEqual(got, want, `${want.label} came back as ${got ? got.label : 'nothing'}`);
  }
});
test('links shared before the token format resolve as they always did', () => {
  // Bare byte counts are ambiguous by construction; the contract is only that
  // they keep landing on the first option with that value, as the old loop did.
  for (const val of ['2', '1', '0.5', '0.63', '1.1']) {
    assert.strictEqual(resolve(val), PREC_OPTS.find(o => o.value === val), `legacy bpp=${val} moved`);
  }
});

console.log('\nNVLink is a property of the card');
/* The interconnect dropdown defaults to "NVLink / NVSwitch" and seven of the
   twelve catalogued cards have no NVLink at all, so that default silently
   applied a 0.85 multi-GPU scaling factor to consumer and PCIe boards. The gate
   is a one-line function over the catalog's `form`, extracted here from source
   for the same reason PERF and GIB are: a change to the real rule must not be
   able to pass a test that carries its own copy. */
const nvDecl = html.match(/^function supportsNVLink\(gpu\) \{.*\}$/m);
assert.ok(nvDecl, 'supportsNVLink() not found in index.html');
const supportsNVLink = new Function(`${nvDecl[0]}; return supportsNVLink;`)();

test('NVLink follows the catalog form, not the card name', () => {
  for (const [key, gpu] of Object.entries(GPU_TABLE)) {
    assert.strictEqual(supportsNVLink(gpu), gpu.form === 'sxm',
      `${key} (form=${gpu.form}) answered ${supportsNVLink(gpu)}`);
  }
});
test('the consumer and PCIe boards are refused NVLink', () => {
  // Named explicitly: this is the live bug, and a catalog edit that quietly
  // relabels one of these as sxm should have to change this list too.
  for (const key of ['t4-16', 'l4-24', 'rtx4090-24', 'rtx5090-32',
                     'rtx6000ada-48', 'l40s-48', 'rtxpro-96']) {
    assert.ok(GPU_TABLE[key], `${key} missing from GPU_TABLE`);
    assert.strictEqual(supportsNVLink(GPU_TABLE[key]), false, `${key} was granted NVLink`);
  }
});
test('the SXM boards still have it', () => {
  for (const key of ['a100-40', 'a100-80', 'h100-80', 'h200-141', 'b200-192']) {
    assert.ok(GPU_TABLE[key], `${key} missing from GPU_TABLE`);
    assert.strictEqual(supportsNVLink(GPU_TABLE[key]), true, `${key} lost NVLink`);
  }
});

console.log('\nThe catalog names its own default card');
test('exactly one row carries default:true, and DEFAULT_GPU_KEY is derived from it', () => {
  const flagged = Object.keys(GPU_TABLE).filter(k => GPU_TABLE[k].default);
  assert.strictEqual(flagged.length, 1, `rows flagged default: ${flagged.join(', ') || 'none'}`);
  const decl = html.match(/^const DEFAULT_GPU_KEY = .+;$/m);
  assert.ok(decl, 'DEFAULT_GPU_KEY not found in index.html');
  const key = new Function(`const GPU_TABLE = ${JSON.stringify(GPU_TABLE)}; ${decl[0]}; return DEFAULT_GPU_KEY;`)();
  assert.strictEqual(key, flagged[0], `DEFAULT_GPU_KEY=${key} but the catalog flags ${flagged[0]}`);
});
test('every row carries the structural fields the engines read', () => {
  for (const [key, gpu] of Object.entries(GPU_TABLE)) {
    assert.strictEqual(typeof gpu.vendor, 'string', `${key}.vendor`);
    assert.strictEqual(typeof gpu.devices, 'number', `${key}.devices`);
    assert.ok(['sxm', 'pcie', 'consumer'].includes(gpu.form), `${key}.form=${gpu.form}`);
    assert.strictEqual(typeof gpu.caps?.fp8, 'boolean', `${key}.caps.fp8`);
  }
});

console.log('\nThe analytics notice and the beacon travel together');
test('both regions are marked exactly once, so setup.sh can remove them as a pair', () => {
  for (const tag of ['ANALYTICS-BEACON', 'ANALYTICS-NOTICE']) {
    for (const end of ['BEGIN', 'END']) {
      const n = (html.match(new RegExp(`${tag}:${end}`, 'g')) || []).length;
      assert.strictEqual(n, 1, `${tag}:${end} appears ${n} times, expected 1`);
    }
  }
});
test('the page never counts views without saying so, or says so without counting', () => {
  // A fork that deletes the beacon and keeps the footer line would claim
  // telemetry it does not have; keeping the beacon without the line is the
  // undisclosed-tracking case this notice exists to end. Neither may happen.
  const beacon = /<script data-goatcounter=/.test(html);
  const notice = /counts views with/.test(html);
  assert.strictEqual(beacon, notice,
    beacon ? 'the beacon is present but the footer notice is not'
           : 'the footer notice describes a beacon that is not there');
});

console.log('\nCONTRIBUTING.md describes the tool that exists');
const contributing = fs.readFileSync(path.join(ROOT, 'CONTRIBUTING.md'), 'utf8');
test('the documented parameter buckets are the ones the lookup can select', () => {
  // A bucket documented but unreachable is an invitation to contribute dead
  // data: 100b/400b/671b were listed for a year and could never match.
  const documented = [...contributing.matchAll(/^\| `(\d+b)` \| /gm)].map(m => m[1]);
  const selectable = Object.keys(new Function(`${bucketDecl[0]}; return BUCKET_PARAMS;`)());
  assert.deepStrictEqual(documented, selectable,
    `CONTRIBUTING.md lists [${documented}] but findBenchmark can select [${selectable}]`);
});
test('CONTRIBUTING.md carries no second copy of the GPU catalog', () => {
  // The catalog moved into data/gpus.json; a table here would drift the moment
  // a row is added, and it is the fifth such copy this refactor removed.
  const copied = Object.keys(GPU_TABLE).filter(k => contributing.includes(`\`${k}\``));
  assert.strictEqual(copied.length, 0,
    `CONTRIBUTING.md hardcodes catalog keys: ${copied.join(', ')}`);
});
test('CONTRIBUTING.md names objects that exist in the source', () => {
  for (const name of ['MODEL_PRESETS', 'BUCKET_PARAMS', 'findBenchmark']) {
    if (!contributing.includes(name)) continue;
    assert.ok(html.includes(name), `CONTRIBUTING.md names ${name}, which is not in index.html`);
  }
  assert.ok(!/`PR` object/.test(contributing), 'the `PR` object has never existed');
});

console.log('\nShared links resolve to the card they named');
const legacyDecl = html.match(/^function legacyGpuKeyFromPipeString\(raw\) \{[\s\S]*?\n\}$/m);
assert.ok(legacyDecl, 'legacyGpuKeyFromPipeString() not found in index.html');
const paramDecl = html.match(/^function gpuKeyFromParam\(raw\) \{[\s\S]*?\n\}$/m);
assert.ok(paramDecl, 'gpuKeyFromParam() not found in index.html');
const gpuKeyFromParam = new Function(
  `const GPU_TABLE = ${JSON.stringify(GPU_TABLE)};
   ${legacyDecl[0]}
   ${paramDecl[0]}
   return gpuKeyFromParam;`
)();
const legacyString = (g, prices) => [g.gb, g.bw, ...(prices || [g.hyper, g.spec, g.spot]), g.tflops].join('|');

test('capacity, bandwidth and TFLOPS identify a card uniquely', () => {
  // What the legacy decoder joins on. A new row colliding on all three would
  // make old links ambiguous, and the decoder would return whichever came first.
  const seen = new Map();
  for (const [key, g] of Object.entries(GPU_TABLE)) {
    const id = `${g.gb}|${g.bw}|${g.tflops}`;
    assert.ok(!seen.has(id), `${key} and ${seen.get(id)} share gb/bw/tflops (${id})`);
    seen.set(id, key);
  }
});
test('a link shared under the old format still resolves to its own card', () => {
  for (const [key, g] of Object.entries(GPU_TABLE)) {
    assert.strictEqual(gpuKeyFromParam(legacyString(g)), key, `${key} did not round-trip`);
  }
});
test('a price revision does not invalidate links shared before it', () => {
  // The reason for the change: prices are a market snapshot this catalog
  // revises, so joining on them meant every old link broke on the next update
  // — silently, landing on the default card.
  for (const [key, g] of Object.entries(GPU_TABLE)) {
    const repriced = legacyString(g, [g.hyper + 1.11, g.spec + 0.5, g.spot * 2]);
    assert.strictEqual(gpuKeyFromParam(repriced), key, `${key} was lost when its prices changed`);
  }
});
test('an inherited property name is not mistaken for a catalog key', () => {
  // GPU_TABLE[raw] is truthy for these, so they took the already-a-key branch,
  // matched no <option>, and left the default card silently selected.
  for (const raw of ['constructor', '__proto__', 'toString', 'valueOf', 'hasOwnProperty']) {
    assert.strictEqual(gpuKeyFromParam(raw), null, `"${raw}" resolved to a card`);
  }
});
test('a card the catalog no longer has resolves to nothing, not to something else', () => {
  assert.strictEqual(gpuKeyFromParam('64|1600|2|1|0.5|181'), null, 'unknown numbers matched a row');
  assert.strictEqual(gpuKeyFromParam('not-a-key'), null);
  assert.strictEqual(gpuKeyFromParam('16|320|0.76|0.35|0.15'), null, 'a five-field string is malformed');
  assert.strictEqual(gpuKeyFromParam(''), null);
});
test('the current format resolves without going near the legacy path', () => {
  for (const key of Object.keys(GPU_TABLE)) {
    assert.strictEqual(gpuKeyFromParam(key), key);
  }
});
test('an unresolvable link is reported to the user, not absorbed', () => {
  // The decoder returning null is only half the fix; loadURLHash has to say so.
  assert.ok(/urlRestoreLost\.push\(\{ id: 'gpu-model'/.test(html),
    'loadURLHash does not record the GPU when the gpu param resolves to nothing');
  assert.ok(/id="url-restore-warning"/.test(html), 'the notice has nowhere to render');
  assert.ok(/el\.textContent = /.test(html.slice(html.indexOf('function renderRestoreNotice'))),
    'the notice must be set as text, never as HTML — every raw value came from the URL');
});
test('every parameter loadURLHash sets is checked, not assumed', () => {
  // The GPU used to be the only one that reported failure. A precision or a
  // preset that a link names and this version does not have lands on a default
  // just as silently, and the number on screen is then nobody's configuration.
  const body = html.slice(html.indexOf('function loadURLHash'), html.indexOf('function copyURL'));
  const setters = body.match(/setVal\(/g) || [];
  assert.strictEqual(setters.length, 0,
    `loadURLHash still assigns ${setters.length} parameter(s) without checking they took`);
  for (const label of ['the weight precision', 'the KV cache precision', 'the model preset',
                       'the interconnect', 'the GPU']) {
    assert.ok(body.includes(label) || html.includes(`label: '${label}'`),
      `no failure path reports ${label}`);
  }
});
test('the notice names what was lost, and stops naming it once it is fixed', () => {
  /* Runs the real renderRestoreNotice() against a stub DOM rather than
     asserting on its source, so the retraction is exercised rather than
     described. The stub is three fields and a notice element — everything the
     function touches. */
  const decl = html.slice(html.indexOf('let urlRestoreLost'), html.indexOf('function loadURLHash'));
  const el = (v) => ({ value: v, style: { display: 'none' }, textContent: '' });
  const dom = {
    'url-restore-warning': el(''),
    'gpu-model': el('a100-40'),
    'weight-precision': el('2'),
  };
  const run = new Function('document', `${decl}
    return {
      set: (l) => { urlRestoreLost = l; },
      render: () => renderRestoreNotice(),
    };`)({ getElementById: (id) => dom[id] });

  run.set([
    { id: 'gpu-model', label: 'the GPU', raw: 'h100-999', fallback: 'a100-40' },
    { id: 'weight-precision', label: 'the weight precision', raw: 'q9', fallback: '2' },
  ]);
  run.render();
  const notice = dom['url-restore-warning'];
  assert.strictEqual(notice.style.display, '', 'the notice should be visible');
  assert.ok(notice.textContent.includes('h100-999') && notice.textContent.includes('q9'),
    `both lost values should be named: ${notice.textContent}`);

  // The user picks a real card. That entry must drop; the other must remain.
  dom['gpu-model'].value = 'h100-80';
  run.render();
  assert.ok(!notice.textContent.includes('h100-999'),
    `the fixed field is still named: ${notice.textContent}`);
  assert.ok(notice.textContent.includes('q9'), 'the unfixed field should still be named');
  assert.strictEqual(notice.style.display, '', 'the notice should still be visible');

  // And once everything is dealt with, it goes away entirely.
  dom['weight-precision'].value = '1';
  run.render();
  assert.strictEqual(notice.style.display, 'none', 'the notice should have retracted');
});
test('the notice retracts itself once the user fixes the field', () => {
  // It described the link at load. After the user picks a different card it is
  // describing a configuration nobody is looking at, which is its own false
  // claim — so entries are filtered against the field's current value.
  const fn = html.slice(html.indexOf('function renderRestoreNotice'));
  assert.ok(/filter\(e => document\.getElementById\(e\.id\)\.value === e\.fallback\)/.test(fn),
    'renderRestoreNotice does not drop entries whose field has since changed');
  assert.ok(/renderRestoreNotice\(\);/.test(html.slice(html.indexOf('function recalculate'))),
    'renderRestoreNotice is never re-run, so the notice cannot retract');
});

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);
