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

/* Running the real readInputState() against a stub DOM. The predicate tests
   above pin supportsNVLink(); they do not pin the two places that call it, and
   a review demonstrated that removing BOTH — the gate in readInputState and the
   clamp in syncInterconnect — reintroduces the original bug with the whole
   suite still green. This drives the actual state builder instead. */
const domStub = (gpuKey, interconnect) => {
  const el = (v, extra = {}) => ({ value: v, style: {}, options: [], ...extra });
  const fields = {
    'gpu-model': el(gpuKey),
    'interconnect': el(interconnect),
    'param-count': el('8'), 'active-percent': el('100'), 'layer-count': el('32'),
    'kv-head-count': el('8'), 'head-dim': el('128'), 'shared-experts': el('0'),
    'context-length': el('8192'), 'concurrency': el('1'), 'gpu-count': el('2'),
    'shared-prefix': el('0'), 'prefix-caching': el('1'),
    'kv-precision': el('2'), 'preset': el(''),
    'weight-precision': el('2', { selectedOptions: [{ dataset: { q: '' } }] }),
  };
  return { getElementById: (id) => fields[id], fields };
};
const readInputStateFor = (gpuKey, interconnect) => {
  const dom = domStub(gpuKey, interconnect);
  const src = html.slice(html.indexOf('function getVal(id)'), html.indexOf('/* Vendor-keyed performance constants'));
  const fn = new Function('document', 'GPU_TABLE', `
    let currentAttn = { mode: 'standard', window: 0, localLayers: 0, mlaDim: 0 };
    let currentModelMaxCtx = 131072, importedModelId = null;
    ${src}
    return readInputState;`)(dom, GPU_TABLE);
  return fn();
};

console.log('\nA board is not always one device');
/* The catalog stores per-board figures because that is the unit you buy. A
   multi-GCD module — one OAM presenting two GCDs — is one row with devices: 2.
   Everything except cost has to be scoped to devices, and each of the three
   traps below passes the existing suite while being wrong. */
const dualGCD = { gb: 128, bw: 3276.8, tflops: 383, hyper: 6.0, spec: 2.5, spot: 1.2,
                  name: 'Dual-GCD 128 GB', vendor: 'nvidia', devices: 2, form: 'sxm' };
const asState = (card, count, extra = {}) => ({
  params: 70, activePercent: 100, bytesPerParam: 2, layers: 80, kvHeads: 8, headDim: 128,
  sharedExperts: 0, contextLength: 8192, concurrency: 16, gpuCount: count,
  hasNVLink: true, kvBytesPerValue: 2, modelMaxCtx: 1048576, vendor: card.vendor,
  gpuGB: card.gb, gpuBandwidth: card.bw, gpuTFLOPS: card.tflops, gpuDevices: card.devices,
  gpuHyperCost: card.hyper, gpuSpecCost: card.spec, gpuSpotCost: card.spot,
  gpuName: card.name, ...extra,
});
// The same silicon described as one dual-device board, or as two single-device
// boards. Every number except the per-board cost must agree.
const single = { ...dualGCD, gb: 64, bw: 3276.8 / 2, tflops: 383 / 2, devices: 1 };

test('a dual-GCD board and two single-GCD boards compute the same everything', () => {
  const a = computeInference(asState(dualGCD, 1));
  const b = computeInference(asState(single, 2));
  for (const k of ['weightsGB', 'kvCacheGB', 'totalGB', 'freeForKVCache',
                   'maxContextSingleUser', 'singleStreamTokS', 'aggregateTokS', 'ttftMs']) {
    assert.ok(Math.abs(a[k] - b[k]) < Math.max(1e-6, Math.abs(b[k]) * 1e-9),
      `${k}: dual-GCD ${a[k]} vs two singles ${b[k]}`);
  }
  assert.ok(Math.abs(a.perGPU.total - b.perGPU.total) < 1e-6, 'per-device VRAM must match');
});
test('per-device VRAM halves, and the fit check halves with it', () => {
  // The trap: dividing the shares by deviceCount while still comparing against
  // the *board* capacity says a model fits when each GCD is over its limit.
  const dual = computeInference(asState(dualGCD, 1));
  const asIfOneDevice = computeInference(asState({ ...dualGCD, devices: 1 }, 1));
  assert.ok(dual.perGPU.total < asIfOneDevice.perGPU.total,
    'two devices should each hold less than one device would');
  assert.strictEqual(dual.deviceGB, 64, 'per-device capacity is the board halved');
  assert.strictEqual(dual.deviceCount, 2, 'one dual-GCD board is two devices');
});
test('aggregate bandwidth is unchanged by how the silicon is packaged', () => {
  // deviceBandwidth * deviceCount === gpuBandwidth * gpuCount, or the roofline
  // moved when nothing physical did.
  const a = computeInference(asState(dualGCD, 2));
  const b = computeInference(asState(single, 4));
  assert.ok(Math.abs(a.aggregateTokS - b.aggregateTokS) < 1, 'aggregate throughput drifted');
});
test('cost is per board, not per device', () => {
  // The double-count: a dual-GCD module is one line item on the invoice.
  const dual = computeInference(asState(dualGCD, 1));
  assert.strictEqual(dual.hourlyHyper, dualGCD.hyper, 'one module must bill as one module');
  const four = computeInference(asState(dualGCD, 4));
  assert.strictEqual(four.hourlyHyper, dualGCD.hyper * 4, 'four modules, four line items');
});
test('free KV cache moves both operands together', () => {
  // Halving the per-device capacity while still multiplying by the board count
  // (or the reverse) silently halves or doubles free KV, and every capacity
  // number downstream inherits it with no fit check to catch it.
  // A model that actually fits, so free KV is a live number and not a clamped
  // zero that would agree by accident.
  const fits = { params: 13, bytesPerParam: 2, layers: 40 };
  const dual = computeInference(asState(dualGCD, 1, fits));
  const twoSingles = computeInference(asState(single, 2, fits));
  assert.ok(dual.freeForKVCache > 0, 'the probe config must leave room for KV');
  assert.ok(Math.abs(dual.freeForKVCache - twoSingles.freeForKVCache) < 1e-6,
    `free KV differs: ${dual.freeForKVCache} vs ${twoSingles.freeForKVCache}`);
  assert.ok(dual.freeForKVCache < dualGCD.gb,
    'free KV should be a fraction of the board, not a multiple of it');
});
test('a single dual-GCD module still asks vLLM for tensor parallel 2', () => {
  // The one that leaves half the silicon idle: sized in boards, a single module
  // emits no --tensor-parallel-size at all.
  const src = html.slice(html.indexOf('function splitParallelism('), html.indexOf('function renderStrategyBadges'));
  const parallelismFor = new Function(`${src}; return parallelismFor;`)();
  assert.deepStrictEqual(parallelismFor({ gpuCount: 1, gpuDevices: 2 }), { tp: 2, dp: 1 });
  assert.deepStrictEqual(parallelismFor({ gpuCount: 4, gpuDevices: 2 }), { tp: 8, dp: 1 });
  assert.deepStrictEqual(parallelismFor({ gpuCount: 1, gpuDevices: 1 }), { tp: 1, dp: 1 });
  assert.deepStrictEqual(parallelismFor({ gpuCount: 8, gpuDevices: 1 }), { tp: 8, dp: 1 });
});

/* ---- rendered output, driven against a stub DOM --------------------------
   Every renderer below had zero coverage in either engine, and an independent
   review demonstrated the consequence: twelve separate reversions of the
   board-vs-device fixes left the whole suite green, because nothing ever read
   what these functions actually produce. The compute core is diffed against a
   second implementation; the strings around it were diffed against nothing.  */
const renderHarness = (inputs = {}) => {
  const out = {};
  const values = {
    'train-method': 'lora', 'train-optimizer': 'adam', 'train-grad-ckpt': '1',
    'train-batch-size': '1', 'train-seq-len': '2048', 'train-lora-rank': '16',
    'train-lora-targets': '4', 'train-hidden-size': '4096', ...inputs,
  };
  const el = (id) => ({
    get innerHTML() { return out[id] || ''; }, set innerHTML(v) { out[id] = v; },
    get value() { return values[id] !== undefined ? values[id] : ''; }, set value(v) { values[id] = v; },
    style: {}, options: [], selectedOptions: [{ dataset: { q: '' } }], checked: true,
    classList: { add() {}, remove() {} },
  });
  const cache = {};
  const document = {
    getElementById: (id) => (cache[id] = cache[id] || el(id)),
    // Executive mode is on: renderExecutiveSummary() returns immediately
    // otherwise, and this suite exists to read what it writes.
    body: { classList: { add() {}, remove() {}, contains: () => true } },
    // exportSummary() reads the comparison table and copies to the clipboard.
    querySelector: () => ({ textContent: '', innerHTML: '', parentElement: null }),
    querySelectorAll: () => [],
  };
  const start = html.indexOf('function getVal(id)');
  const before = [/^const GPU_TABLE = \{[\s\S]*?\n\};$/m, /^const BENCHMARK_DATA = \{[\s\S]*?\n\};$/m,
                  /^const MODEL_PRESETS = \{[\s\S]*?\n\};$/m, /^const WORKLOAD_PROFILES = \{[\s\S]*?\n\};$/m]
    .map(re => html.match(re)[0]).filter(d => html.indexOf(d) < start).join('\n');
  const navigator = { clipboard: { writeText: () => Promise.resolve() } };
  const api = new Function('document', 'navigator', `
    let currentAttn = { mode: 'standard', window: 0, localLayers: 0, mlaDim: 0 };
    let currentModelMaxCtx = 131072, importedModelId = null, urlRestoreLost = [];
    // The real name, declared before this slice begins. It was previously
    // spelled comparisonSnapshots — a name index.html does not contain — so
    // renderComparisons() threw on sight and no test could call it.
    let savedSnapshots = [];
    ${before}
    ${html.slice(start, html.indexOf('function updateURLHash()'))}
    return { renderVerdict, renderGPUCards, renderTraining, renderNotes, renderStrategyBadges,
             renderExecutiveSummary, renderThroughput, renderCapacity, exportSummary, renderCommand,
             renderMetrics, renderComparisons, computeInference, buildVllmCommand,
             pushSnapshot: (s, c) => savedSnapshots.push({ state: s, computed: c, name: 'snap' }) };`)(document, navigator);
  return { ...api, out };
};

/* index.html's own formatters, read from source so a change to one cannot make
   these assertions quietly stop comparing what the page shows. Capacity is
   pinned absolutely elsewhere, since reading it from source would follow a
   regression in it. */
const formatGBLike = new Function(`${html.match(/^function formatGB\(gb\) .+$/m)[0]}; return formatGB;`)();
const bandwidthLabelDecl = html.match(/^function bandwidthLabel\(gbs\) .+$/m);
assert.ok(bandwidthLabelDecl, 'bandwidthLabel() not found in index.html');
const bandwidthLabel = new Function(`${bandwidthLabelDecl[0]}; return bandwidthLabel;`)();
/* Capacity has its own formatter, so an integer catalog value keeps rendering
   without a decimal. Read from source for the same reason as the above. */
const capacityLabelLike = new Function(
  `${html.match(/^function formatGB\(gb\) .+$/m)[0]}
   ${html.match(/^function capacityLabel\(gb\) .+$/m)[0]}; return capacityLabel;`)();

console.log('\nWhat the renderers actually put on the page');
test('the training estimate renders at all', () => {
  /* It did not. A scoped rename rewrote its own declaration into
     `const deviceGB = deviceGB / devicesPerBoard`, so every call threw a
     ReferenceError and the whole feature was dead — with the suite green,
     because nothing called it. */
  const h = renderHarness();
  const state = asState(GPU_TABLE['h100-80'], 2, { params: 8, layers: 32 });
  h.renderTraining(state, h.computeInference(state));
  assert.ok((h.out['training-results'] || '').length > 100,
    'renderTraining produced nothing');
});
test('a training shortfall is counted in boards, like every other verdict', () => {
  const h = renderHarness();
  const state = asState(dualGCD, 1, { params: 70, layers: 80, bytesPerParam: 2 });
  h.renderTraining(state, h.computeInference(state));
  const text = h.out['training-results'] || '';
  assert.ok(!/\+ devices/.test(text),
    `training counts in devices while the other banners count in boards: ${text.match(/Need[^<]*/)}`);
  // the figure, not only the noun
  const shown = Number((text.match(/Need (\d+)\+ boards/) || [])[1]);
  assert.ok(Number.isFinite(shown), `no boards figure in the training verdict: ${text.slice(0, 200)}`);
  // Derived from the overage the banner states plus the device capacity the
  // engine reports, so the figure is checked rather than its noun.
  const computed = h.computeInference(state);
  const over = Number((text.match(/Over by ([\d.]+) GiB per device/) || [])[1]);
  assert.ok(Number.isFinite(over), `no overage in the training verdict: ${text.slice(0, 200)}`);
  const perDevice = over + computed.deviceGB;
  const devicesNeeded = Math.ceil((perDevice * computed.deviceCount) / (computed.deviceGB * 0.9));
  const perBoard = computed.deviceCount / state.gpuCount;
  assert.strictEqual(shown, Math.ceil(devicesNeeded / perBoard),
    `training says ${shown} boards; ${devicesNeeded} devices at ${perBoard} per board is ` +
    `${Math.ceil(devicesNeeded / perBoard)}`);

  // And the figures themselves must be per device: the same silicon described
  // as one dual-GCD board or as two single-GCD boards has to land identically.
  const h2 = renderHarness();
  const single = { ...dualGCD, gb: dualGCD.gb / 2, bw: dualGCD.bw / 2, tflops: dualGCD.tflops / 2, devices: 1 };
  const twoBoards = asState(single, 2, { params: 70, layers: 80, bytesPerParam: 2 });
  h2.renderTraining(twoBoards, h2.computeInference(twoBoards));
  const pct = (t) => (t.match(/(\d+)%/) || [])[1];
  assert.ok(pct(text) !== undefined, `no utilisation figure rendered: ${text.slice(0, 200)}`);
  assert.strictEqual(pct(text), pct(h2.out['training-results'] || ''),
    'a dual-GCD board and two single-GCD boards must train identically');
});
test('a verdict that does not fit says by how much, measured on the device', () => {
  const h = renderHarness();
  const state = asState(dualGCD, 1);
  const computed = h.computeInference(state);
  assert.strictEqual(computed.fits, false, 'probe config must not fit');
  h.renderVerdict(state, computed);
  const text = h.out['verdict-output'] || '';
  assert.ok(!/Over by 0 GiB/.test(text), `the overage was clamped to zero: ${text}`);
  // The figure, not just the noun. Reading board capacity here changes the
  // number while leaving the word "boards" in place.
  const wantDevices = Math.ceil(computed.totalGB / (computed.deviceGB * 0.9));
  const wantBoards = Math.ceil(wantDevices / (computed.deviceCount / state.gpuCount));
  const shown = Number((text.match(/Need (\d+)\+ boards/) || [])[1]);
  assert.strictEqual(shown, wantBoards,
    `banner says ${shown} boards, device arithmetic says ${wantBoards}: ${text}`);
});
test('the sharding a card claims is the sharding the command performs', () => {
  // Five dual-GCD boards: ten devices, and the condensed view used to caption
  // them "sharded 5-way" beside a number divided by ten.
  const h = renderHarness();
  // Both branches: four dual-GCD boards give eight devices and a pure TP=8
  // split, five give ten and a TP=2 x DP=5 one. The review found the dp>1
  // branch of this test dead because the comment said five and the code said
  // four.
  for (const boards of [4, 5]) {
  const state = asState(dualGCD, boards, { params: 8, layers: 32, bytesPerParam: 2 });
  const computed = h.computeInference(state);
  h.renderGPUCards(state, computed);
  const text = h.out['gpu-cards'] || h.out['vram-output'] || Object.values(h.out).join(' ');
  const cmd = h.buildVllmCommand(state, computed, 'm');
  const tp = Number((cmd.match(/--tensor-parallel-size (\d+)/) || [0, 1])[1]);
  const dp = Number((cmd.match(/--data-parallel-size (\d+)/) || [0, 1])[1]);
  assert.strictEqual(tp * dp, computed.deviceCount,
    'the command must account for every device the cards describe');
  // Every sharding factor the view states, not the first one that matches: the
  // tile and the interconnect line each carry one, and they are set separately.
  const ways = [...text.matchAll(/(\d+)-way/g)].map(m => Number(m[1]));
  assert.ok(ways.length, `the condensed view should state a sharding factor: ${text.slice(0, 200)}`);
  if (dp === 1) {
    for (const w of ways) {
      assert.strictEqual(w, computed.deviceCount,
        `a card says ${w}-way while the engine shards ${computed.deviceCount}-way`);
    }
  } else {
    // A data-parallel split shards weights tp-way per replica while the panel's
    // own figures divide by every device. That gap is real and is the next
    // commit's subject; what this pins is that the view admits it.
    const caveat = text.match(/assumes sharding across all (\d+)/);
    assert.ok(caveat, `a dp>1 view must carry the caveat about its own figures: ${text.slice(0, 300)}`);
    assert.strictEqual(Number(caveat[1]), computed.deviceCount,
      'the caveat must name the count the figures actually used');
  }
  }
});
test('the executive summary reads the device it fills, not the board', () => {
  const h = renderHarness();
  const state = asState(dualGCD, 2, { params: 8, layers: 32, bytesPerParam: 2 });
  const computed = h.computeInference(state);
  h.renderExecutiveSummary(state, computed);
  const text = Object.values(h.out).join(' ');
  // Anchored on the phrase itself: the summary carries several percentages and
  // an unanchored match reads whichever comes first.
  const pct = Number((text.match(/per device \((\d+)%\)/) || [])[1]);
  const want = Math.round(computed.perGPU.total / computed.deviceGB * 100);
  assert.strictEqual(pct, want,
    `exec summary shows ${pct}% of capacity, per-device arithmetic says ${want}%`);
  // And the capacity it names, not only the percentage it derives: those are
  // two separate expressions and only one of them was wrong before.
  const named = (text.match(/\/ ([\d.]+ GiB) per device/) || [])[1];
  assert.strictEqual(named, capacityLabelLike(computed.deviceGB),
    `exec summary names ${named} as the capacity of a ${capacityLabelLike(computed.deviceGB)} device`);
});
test('one board that is two devices still explains its interconnect', () => {
  // The note describing NVLink/PCIe was gated on the board count, so a single
  // dual-GCD module took the interconnect penalty in silence.
  const h = renderHarness();
  const state = asState(dualGCD, 1, { params: 8, layers: 32 });
  h.renderNotes(state);
  const notes = Object.values(h.out).join(' ');
  assert.ok(/NVLink|PCIe/.test(notes),
    'no interconnect note for a board whose two devices must talk to each other');
});

console.log('\nThe same silicon renders the same however it is packaged');
test('two dual-GCD boards and four single-GCD boards produce identical output', () => {
  /* The strongest guard available for this class, and the one that replaces
     enumerating every rendered site: any surface that reads boards where it
     should read devices differs between these two descriptions of the same
     hardware. Prices are halved on the single-device card so even the cost
     tiles match, leaving the board count in the "N× name" header as the only
     legitimate difference. */
  const dual = { ...dualGCD, name: 'X' };
  const single = { ...dual, gb: dual.gb / 2, bw: dual.bw / 2, tflops: dual.tflops / 2,
                   hyper: dual.hyper / 2, spec: dual.spec / 2, spot: dual.spot / 2, devices: 1 };
  const cfg = { params: 70, layers: 80, bytesPerParam: 2, contextLength: 8192, concurrency: 16 };
  const renderAll = (card, boards, extra = {}) => {
    const h = renderHarness();
    const state = asState(card, boards, { ...cfg, ...extra });
    const computed = h.computeInference(state);
    h.renderVerdict(state, computed);
    h.renderGPUCards(state, computed);
    h.renderExecutiveSummary(state, computed);
    h.renderStrategyBadges(state, computed);
    h.renderTraining(state, computed);
    h.renderNotes(state);
    h.renderThroughput(state, computed);
    h.renderCapacity(state, computed);
    h.renderCommand(state, computed);
    h.renderMetrics(computed);
    // A saved snapshot renders a per-device figure against a capacity, and was
    // the one view no harness could reach.
    h.pushSnapshot(state, computed);
    h.renderComparisons();
    h.out['__export'] = h.exportSummary(state, computed);
    return h.out;
  };
  // Several pairs, not one. A gate written as `gpuCount > 1` reads the same for
  // two boards as for four, so only the one-board pair can see it; a
  // denominator written in boards is invisible at one board but not at four.
  /* Two things are legitimately board-scoped and so legitimately differ: the
     "N× card" hardware line, and any shortfall counted in boards — two dual-GCD
     modules and four single-GCD ones are the same silicon but not the same
     shopping list. Everything else must match exactly. */
  const norm = (t) => (t || '').replace(/\d+× X/g, 'N× X').replace(/Need \d+\+ boards/g, 'Need N+ boards')
    .replace(/requires \d+\+ boards/g, 'requires N+ boards')
    // "per device" after a bandwidth is a statement *about* the packaging —
    // it appears only when a board is more than one device — so of course it
    // differs between two descriptions of the same silicon.
    .replace(/ GB\/s per device /g, ' GB/s ');
  // 8 boards = 16 devices, which is the only pair here that produces a
  // data-parallel split and so the only one that renders the dp>1 captions.
  for (const boards of [1, 2, 4, 8]) {
    // Both interconnects: a board that is several devices and has no NVLink is
    // exactly what an AMD OAM row will be, and every case here was NVLink.
    for (const link of [true, false]) {
      const a = renderAll(dual, boards, { hasNVLink: link });
      const b = renderAll(single, boards * 2, { hasNVLink: link });
      for (const id of new Set([...Object.keys(a), ...Object.keys(b)])) {
        assert.strictEqual(norm(a[id]), norm(b[id]),
          `${id} differs between ${boards} dual-GCD board(s) and ${boards * 2} single-GCD ` +
          `boards holding the same silicon (${link ? 'NVLink' : 'PCIe'})`);
      }
    }
  }
});
test('an integer capacity renders without a decimal, stated absolutely', () => {
  /* Every other assertion about capacity reads the formatter out of the source,
     so dropping its integer rule changed "80 GiB" to "80.0 GiB" on all twelve
     rows with the suite green: the tests followed the regression. These are
     literals on purpose. */
  assert.strictEqual(capacityLabelLike(80), '80 GiB');
  assert.strictEqual(capacityLabelLike(16), '16 GiB');
  assert.strictEqual(capacityLabelLike(141), '141 GiB');
  assert.strictEqual(capacityLabelLike(64), '64 GiB');
  assert.strictEqual(capacityLabelLike(128 / 3), '42.7 GiB');

  /* Its sibling, which prints the bare number for the two places that carry
     their own unit. It had no test at all, and it is the same regression
     class: a snapshot card read "32.6 GiB / 16" on master and must still. */
  const capacityNumberDecl = html.match(/^function capacityNumber\(gb\) .+$/m);
  assert.ok(capacityNumberDecl, 'capacityNumber() not found in index.html');
  const capacityNumber = new Function(`${capacityNumberDecl[0]}; return capacityNumber;`)();
  assert.strictEqual(capacityNumber(16), '16');
  assert.strictEqual(capacityNumber(80), '80');
  assert.strictEqual(capacityNumber(128 / 3), '42.7');
});
test('the same bandwidth reads the same in the page and in the PDF', () => {
  // The page interpolated the raw quotient — 1092.2666666666667 GB/s — while
  // the PDF printed 1092.27 for the same board.
  assert.strictEqual(bandwidthLabel(3276.8 / 3), '1092.27');
  assert.strictEqual(bandwidthLabel(3276.8), '3276.8');
  assert.strictEqual(bandwidthLabel(320), '320');
  assert.strictEqual(bandwidthLabel(1638.4), '1638.4');
});
test('a bandwidth that does not divide cleanly is rounded before it is shown', () => {
  /* Pinning the formatter is not pinning its use: the throughput line
     interpolated the raw quotient, so a three-device board read
     "1092.2666666666667 GB/s per device" on the page while the PDF printed
     1092.27 for the same hardware. */
  const h = renderHarness();
  const thirds = { ...dualGCD, devices: 3 };
  const state = asState(thirds, 1, { params: 8, layers: 32 });
  const computed = h.computeInference(state);
  h.renderThroughput(state, computed);
  const text = h.out['throughput-output'] || '';
  assert.ok(text.includes(`${bandwidthLabel(computed.deviceBandwidth)} GB/s`),
    `the line does not show the rounded bandwidth: ${text.slice(0, 240)}`);
  assert.ok(!/\d\.\d{4,}/.test(text),
    `a raw quotient reached the page: ${(text.match(/[\d.]{8,}/) || [])[0]}`);
});
test('the metrics tiles agree with the engine they are describing', () => {
  /* Total VRAM is the same in both packagings, so the equivalence test above
     cannot see it — an absolute assertion is the only thing that can. The two
     engines also compute it by different routes (deviceGB x deviceCount here,
     board GB x board count in Python), so it is compared across them too. */
  const h = renderHarness();
  const state = asState(dualGCD, 2, { params: 8, layers: 32 });
  const computed = h.computeInference(state);
  h.renderMetrics(computed);
  const text = h.out['metrics-output'] || '';
  assert.ok(text.includes(formatGBLike(computed.totalVRAM)),
    `Total VRAM tile does not show ${formatGBLike(computed.totalVRAM)}: ${text.slice(0, 200)}`);
  assert.strictEqual(computed.totalVRAM, computed.deviceGB * computed.deviceCount,
    'total VRAM must be every device summed');
  assert.strictEqual(computed.totalVRAM, state.gpuGB * state.gpuCount,
    'and equally every board summed — the two routes must agree');
});
test('the state builder carries the catalog device count to the page', () => {
  // Deleting this one line disconnects the catalog from every derivation above
  // while leaving the whole suite green — verified by review.
  for (const [key, gpu] of Object.entries(GPU_TABLE)) {
    const state = readInputStateFor(key, '1');
    assert.strictEqual(state.gpuDevices, gpu.devices,
      `${key}: state carries gpuDevices=${state.gpuDevices}, catalog says ${gpu.devices}`);
  }
});
test('a compute-bound estimate is bound by the devices, not the boards', () => {
  // The decode ceiling was the one arithmetic site no case exercised with
  // devices > 1: every parity case is bandwidth-bound there.
  const h = renderHarness();
  const bound = { params: 30, activePercent: 10, contextLength: 256, concurrency: 256, layers: 48 };
  const single = { ...dualGCD, gb: 64, bw: dualGCD.bw / 2, tflops: dualGCD.tflops / 2, devices: 1 };
  const dual = h.computeInference(asState(dualGCD, 1, bound));
  const two = h.computeInference(asState(single, 2, bound));
  // Sized in boards the product deviceTFLOPS*deviceCount is unchanged; what
  // moves is the interconnect penalty, which is 1.0 at one board and 0.85 at
  // two devices. So the ceiling has to be compared where those disagree.
  assert.ok(dual.computeBound || two.computeBound, 'the probe must sit on the compute ceiling');
  assert.ok(Math.abs(dual.saturatedTokS - two.saturatedTokS) < 1,
    `compute ceiling differs by packaging: ${dual.saturatedTokS} vs ${two.saturatedTokS}`);
});
test('a single-device card emits exactly the flags it always did', () => {
  /* The gate that fixes the dual-GCD case must not disturb the twelve real
     rows. An earlier version keyed on `tp > 1` and silently dropped
     --tensor-parallel-size at odd counts above eight, where the split is
     TP=1 x DP=N — a command-text change on real hardware, disclosed but not
     licensed by the requirement. */
  const h = renderHarness();
  for (const count of [1, 2, 8, 9, 11, 17, 33]) {
    const state = asState(GPU_TABLE['h100-80'], count, { params: 8, layers: 32 });
    const cmd = h.buildVllmCommand(state, h.computeInference(state), 'm');
    const hasTP = /--tensor-parallel-size/.test(cmd);
    assert.strictEqual(hasTP, count > 1,
      `${count} single-device GPUs: --tensor-parallel-size ${hasTP ? 'present' : 'absent'}, ` +
      'which is not what a board count above one has always meant');
  }
});
test('a single dual-GCD board emits the flags its devices require', () => {
  // Reverting the TP and expert-parallel gates in *both* engines at once is
  // invisible to a cross-engine diff. These are absolute.
  const h = renderHarness();
  const state = asState(dualGCD, 1, { params: 8, layers: 32 });
  const computed = h.computeInference(state);
  const cmd = h.buildVllmCommand(state, computed, 'm');
  assert.match(cmd, /--tensor-parallel-size 2/,
    `one dual-GCD module needs TP=2 or half the silicon idles:\n${cmd}`);
  const moe = asState(dualGCD, 1, { params: 30, activePercent: 10, layers: 48 });
  const moeComputed = h.computeInference(moe);
  assert.ok(moeComputed.isMoE, 'the probe must be an MoE');
  assert.match(h.buildVllmCommand(moe, moeComputed, 'm'), /--enable-expert-parallel/,
    'an MoE across two devices needs expert parallelism');
});

console.log('\nA verdict reads per-device numbers against per-device capacity');
test('an over-capacity dual-GCD board reports a real overage, not zero', () => {
  /* The symptom review found: per-device shares measured against the board's
     capacity. 87.6 GiB on a 64 GiB device printed "Over by 0 GiB", because the
     overage was taken against the 128 GiB board and clamped at zero. */
  const c = computeInference(asState(dualGCD, 1));
  assert.strictEqual(c.fits, false, 'the probe config must not fit');
  const overage = c.perGPU.total - c.deviceGB;
  assert.ok(overage > 1, `overage against device capacity should be real, got ${overage}`);
  assert.ok(c.perGPU.total - dualGCD.gb < 0,
    'and measuring against the board is what produced the clamped zero');
});
test('the boards-needed figure is in boards, not devices, at every board count', () => {
  /* Checked across board counts, not only at one. A plausible typo that reads
     the per-board divisor only when gpuCount === 1 doubles the purchase
     recommendation for everyone else, and the equivalence test above cannot
     see it: a shortfall counted in boards legitimately differs between the two
     packagings, so that test normalises the figure away. */
  const src = html.slice(html.indexOf('function boardsNeeded'), html.indexOf('function renderVerdict'));
  const boardsNeeded = new Function(`${src}; return boardsNeeded;`)();
  for (const devices of [1, 2, 4]) {
    for (const boards of [1, 2, 4]) {
      const card = { ...dualGCD, gb: 128, devices };
      const state = asState(card, boards, { params: 400, layers: 126, bytesPerParam: 2 });
      const c = computeInference(state);
      const needDevices = Math.ceil(c.totalGB / (c.deviceGB * 0.9));
      assert.strictEqual(boardsNeeded(state, c), Math.ceil(needDevices / devices),
        `${boards} board(s) of ${devices} device(s): ${needDevices} devices is ` +
        `${Math.ceil(needDevices / devices)} boards, not ${boardsNeeded(state, c)}`);
    }
  }
});

console.log('\nThe NVLink gate as the state builder actually applies it');
test('a card without NVLink cannot report NVLink, whatever the control says', () => {
  for (const key of ['t4-16', 'l4-24', 'rtx4090-24', 'rtx5090-32',
                     'rtx6000ada-48', 'l40s-48', 'rtxpro-96']) {
    const state = readInputStateFor(key, '1');   // the control asking for NVLink
    assert.strictEqual(state.hasNVLink, false,
      `${key} reported hasNVLink=true with the interconnect set to NVLink`);
  }
});
test('an SXM card still honours the control in both positions', () => {
  for (const key of ['a100-40', 'a100-80', 'h100-80', 'h200-141', 'b200-192']) {
    assert.strictEqual(readInputStateFor(key, '1').hasNVLink, true, `${key} lost NVLink`);
    assert.strictEqual(readInputStateFor(key, '0').hasNVLink, false, `${key} ignored PCIe`);
  }
});
test('the state carries the vendor its constants are chosen by', () => {
  const state = readInputStateFor('h100-80', '1');
  assert.strictEqual(state.vendor, GPU_TABLE['h100-80'].vendor);
  assert.ok(state.vendor, 'vendor must not be empty — PERF[vendor] would fall back silently');
});

console.log('\nThe interconnect control follows the card');
const syncFor = (gpuKey, interconnect) => {
  const dom = domStub(gpuKey, interconnect);
  dom.fields['interconnect'].options = [{ value: '1', disabled: false, textContent: 'NVLink / NVSwitch' },
                                        { value: '0', disabled: false, textContent: 'PCIe only' }];
  const src = html.slice(html.indexOf('let interconnectForcedToPCIe'), html.indexOf('function recalculate'));
  const fn = new Function('document', 'GPU_TABLE', `${nvDecl[0]}\n${src}; return syncInterconnect;`)(dom, GPU_TABLE);
  fn();
  return dom.fields['interconnect'];
};
test('selecting a card without NVLink disables the option and falls back to PCIe', () => {
  const sel = syncFor('rtx4090-24', '1');
  assert.strictEqual(sel.value, '0', 'should have fallen back to PCIe');
  assert.strictEqual(sel.options[0].disabled, true, 'the NVLink option should be disabled');
});
test('switching back to an SXM card restores the NVLink it took away', () => {
  // Reported by review: the clamp was one-way, so a reader who touched a 4090
  // was left on PCIe on every card afterwards, quietly losing 0.85 scaling they
  // never chose to give up.
  const dom = domStub('rtx4090-24', '1');
  dom.fields['interconnect'].options = [{ value: '1', disabled: false, textContent: '' },
                                        { value: '0', disabled: false, textContent: '' }];
  const src = html.slice(html.indexOf('let interconnectForcedToPCIe'), html.indexOf('function recalculate'));
  const sync = new Function('document', 'GPU_TABLE', `${nvDecl[0]}\n${src}; return syncInterconnect;`)(dom, GPU_TABLE);
  sync();
  assert.strictEqual(dom.fields['interconnect'].value, '0', 'forced to PCIe on the 4090');
  dom.fields['gpu-model'].value = 'h100-80';
  sync();
  assert.strictEqual(dom.fields['interconnect'].value, '1',
    'switching to an SXM card should give back the NVLink the clamp removed');
  assert.strictEqual(dom.fields['interconnect'].options[0].disabled, false);
});
test('a deliberate PCIe choice on an SXM card is not overridden', () => {
  const sel = syncFor('h100-80', '0');
  assert.strictEqual(sel.value, '0', 'the reader chose PCIe; leave it alone');
  assert.strictEqual(sel.options[0].disabled, false);
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

console.log('\nThe opt-out the disclosure promises');
test('the page carries its own control, because the documented gesture cannot work here', () => {
  /* GoatCounter's documented opt-out is loading the page with
     #toggle-goatcounter, which count.js reads at evaluation time. This page
     rewrites location.hash during boot — synchronously, in recalculate() ->
     updateURLHash() — while the counter script is still loading async, so the
     gesture is destroyed before it can be seen. Verified by inspection of the
     boot order; the fix is a control that sets the same flag directly. */
  const boot = html.slice(html.indexOf('renderGpuOptions();'));
  assert.ok(/recalculate\(\);/.test(boot), 'boot must still call recalculate()');
  assert.ok(/updateURLHash\(\);/.test(html.slice(html.indexOf('function recalculate'))),
    'recalculate() no longer rewrites the hash — re-check whether the URL opt-out works now');
  assert.ok(/id="counting-toggle"/.test(html), 'the footer carries no opt-out control');
  assert.ok(/renderCountingToggle\(\);/.test(boot), 'the control is never labelled at load');
});
test('the opt-out sets the flag count.js actually checks, and survives localStorage throwing', () => {
  const src = html.slice(html.indexOf('function countingDisabled'), html.indexOf('/* textContent, never innerHTML'));
  assert.ok(/localStorage\.setItem\('skipgc', 't'\)/.test(src),
    "count.js refuses to send when localStorage.skipgc === 't'; nothing else disables it");
  assert.ok(/localStorage\.removeItem\('skipgc'\)/.test(src), 'the control must be reversible');

  // Drive it against a localStorage that throws, as private modes do.
  const el = { textContent: '' };
  const api = new Function('document', 'localStorage', `${src}
    return { toggle: toggleCounting, disabled: countingDisabled };`)(
      { getElementById: () => el },
      { getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); },
        removeItem() { throw new Error('denied'); } });
  assert.strictEqual(api.disabled(), false, 'a throwing localStorage must read as "counted"');
  api.toggle({ preventDefault() {} });   // must not throw
  assert.ok(el.textContent.length > 0, 'the control should still label itself');
});
test('nothing in the repo still points readers at the URL gesture as a working opt-out', () => {
  const readme = fs.readFileSync(path.join(ROOT, 'README.md'), 'utf8');
  // Whole lines: the qualifying words often precede the mention, so matching
  // from the marker onwards would judge a sentence by its tail.
  const lines = (readme + '\n' + html).split('\n').filter(l => l.includes('toggle-goatcounter'));
  assert.ok(lines.length, 'the gesture is not mentioned at all — is the disclosure still complete?');
  for (const line of lines) {
    // A mention is fine where it is quoting count.js's own refusal string or
    // saying the gesture is dead here; what must not survive is an instruction.
    assert.ok(/does not work|cannot work|refuses with|its own opt-out/.test(line),
      `still presented as a working opt-out: ${line.slice(0, 120)}`);
  }
});

console.log('\nThe analytics notice and the beacon travel together');
test('every analytics region is marked, and marked once, wherever it survives', () => {
  /* Not "exactly one" — a fork that ran setup.sh without a site code has
     removed all of them, and the suite this repo tells that fork to run must
     stay green. The invariant is that a region is either absent or complete. */
  for (const tag of ['ANALYTICS-BEACON', 'ANALYTICS-NOTICE']) {
    const begins = (html.match(new RegExp(`${tag}:BEGIN`, 'g')) || []).length;
    const ends = (html.match(new RegExp(`${tag}:END`, 'g')) || []).length;
    assert.strictEqual(begins, ends, `${tag} has ${begins} BEGIN and ${ends} END markers`);
    assert.ok(begins <= 1, `${tag} is marked ${begins} times`);
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
