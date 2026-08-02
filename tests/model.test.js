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

const computeInference = new Function(
  `${gibDecl[0]}\n${html.slice(start, end + 2)}; return computeInference;`
)();
assert.strictEqual(new Function(`${gibDecl[0]}; return GIB;`)(), 1024 ** 3, 'GIB must be 2^30');

/* ---- GPU specs, parsed from the same <option> values the UI uses ---- */
const GPUS = {};
for (const [, val, name] of html.matchAll(/<option value="([\d.|]+)"[^>]*data-n="([^"]+)"/g)) {
  const [gb, bw, hyper, spec, spot, tflops] = val.split('|').map(Number);
  GPUS[name] = { gb, bw, hyper, spec, spot, tflops };
}

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
  const ceiling = (0.35 * GPUS['H100 80GB'].tflops * 1e12) / (2 * 8e9);
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
  // Prefill is dense GEMM work; it runs at 45%, not decode's starved 35%.
  const expected = (2 * 8e9 * 8192) / (0.45 * GPUS['H100 80GB'].tflops * 1e12) * 1000;
  between(computeInference(state()).ttftMs, expected * 0.99, expected * 1.01,
    'ttftMs vs the 45%-MFU prefill formula');
});

console.log('\nAgreement with published benchmarks');
// The point of the rewrite: compare each estimate against the matching mode.
// Order-of-magnitude agreement (0.25x-4x) is the bar for a planning tool.
const GPU_FOR_KEY = {
  'a100-80': 'A100 80GB', 'a100-40': 'A100 40GB', 'h100-80': 'H100 80GB',
  'b200-192': 'B200 192GB', 'rtx4090-24': 'RTX 4090 24GB',
};
const PREC_BYTES = { bf16: 2, fp8: 1, q4: 0.5, int4: 0.5 };

for (const [key, b] of Object.entries(benchmarks.data)) {
  if (b.estimated) continue; // only score against real measurements
  const m = key.match(/^(\d+)b-(.+)$/);
  const params = Number(m[1]);
  const gpu = GPU_FOR_KEY[m[2]];
  if (!gpu) continue;
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
test('the hardcoded 0.43-0.91 band brackets every measured batch benchmark', () => {
  // Re-derive the band from the data so it cannot silently drift as entries land.
  const ratios = [];
  for (const [key, b] of Object.entries(benchmarks.data)) {
    if (b.estimated || b.mode !== 'batch') continue;
    const m = key.match(/^(\d+)b-(.+)$/);
    const gpu = GPU_FOR_KEY[m[2]];
    if (!gpu) continue;
    const params = Number(m[1]);
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
  assert.ok(lo >= 0.43 - 0.02 && hi <= 0.91 + 0.02,
    `measured/ceiling ratios span ${lo.toFixed(2)}-${hi.toFixed(2)}, escaping the ` +
    `hardcoded 0.43-0.91 band — update OBSERVED_EFF_* in index.html and generate_report.py`);
});
test('the band scales the aggregate ceiling and nothing else', () => {
  const c = computeInference(state({ concurrency: 64 }));
  between(c.aggregateObservedLoTokS, c.aggregateTokS * 0.42, c.aggregateTokS * 0.44, 'band lo');
  between(c.aggregateObservedHiTokS, c.aggregateTokS * 0.90, c.aggregateTokS * 0.92, 'band hi');
  assert.ok(c.aggregateObservedHiTokS < c.aggregateTokS, 'the band must sit below the ceiling');
});

console.log('\nBenchmark lookup');
const fbStart = html.indexOf('function findBenchmark(');
assert.notStrictEqual(fbStart, -1, 'findBenchmark() not found');
const fbEnd = html.indexOf('\n}\n', fbStart);
const bucketDecl = html.match(/^const BUCKET_PARAMS = .+;$/m);
assert.ok(bucketDecl, 'BUCKET_PARAMS not found in index.html');
const findBenchmark = new Function(
  `const BENCHMARK_DATA = ${JSON.stringify(benchmarks.data)};
   ${bucketDecl[0]}
   ${html.slice(fbStart, fbEnd + 2)}; return findBenchmark;`
)();

test('buckets match the table documented in CONTRIBUTING.md', () => {
  // 7b covers 5-7B, 8b covers 8-10B. These were inverted.
  const a = findBenchmark(7, 'A100 80GB', 80), b = findBenchmark(8, 'A100 80GB', 80);
  assert.ok(a.exact && b.exact, 'both should be exact matches');
  assert.strictEqual(a.data.tokS, benchmarks.data['7b-a100-80'].tokS);
  assert.strictEqual(b.data.tokS, benchmarks.data['8b-a100-80'].tokS);
});
test('an unmatched size degrades to the nearest entry on the SAME GPU', () => {
  const hit = findBenchmark(26, 'A100 40GB', 40);
  assert.ok(hit, 'should fall back rather than return nothing');
  assert.strictEqual(hit.exact, false, 'must be flagged as inexact');
  assert.strictEqual(hit.requestedParams, 26);
  assert.ok(hit.nearestParams < 26, 'nearest available on this card is smaller');
  assert.strictEqual(hit.slower, true, 'a 26B model is slower than the smaller match');
});
test('the fallback picks the closest bucket, not just any', () => {
  // H100 has 8b/14b/27b/70b. 30B lands in the 27b bucket, which exists.
  assert.ok(findBenchmark(30, 'H100 80GB', 80).exact, '30B is covered by the 27b bucket');
  // 35B lands in the 32b bucket, which has no H100 entry — a genuine gap.
  // Candidates are 27 (dist 8) and 70 (dist 35), so it must choose 27.
  const gap = findBenchmark(35, 'H100 80GB', 80);
  assert.strictEqual(gap.exact, false);
  assert.strictEqual(gap.nearestParams, 27, 'must pick 27B over 70B on distance');
  assert.strictEqual(gap.slower, true, 'a 35B model is slower than a 27B measurement');
});
test('the fallback reports direction correctly in both directions', () => {
  // 4B on B200: only a 27b entry exists, so the nearest match is larger and
  // the user's model is the faster one.
  const smaller = findBenchmark(4, 'B200 192GB', 192);
  assert.strictEqual(smaller.slower, false, '4B is faster than the 27B match');
});
test('fallback never crosses to a different GPU', () => {
  const hit = findBenchmark(4, 'B200 192GB', 192);
  assert.ok(hit, 'B200 has a 27b entry to fall back to');
  assert.strictEqual(hit.exact, false);
  assert.strictEqual(hit.data.tokS, benchmarks.data['27b-b200-192'].tokS,
    'must stay on the B200, not borrow an A100 number');
});
test('a GPU with no data at all returns nothing', () => {
  assert.strictEqual(findBenchmark(8, 'T4 16GB', 16), null);
  assert.strictEqual(findBenchmark(8, 'L40S 48GB', 48), null);
});
test('sizes beyond every bucket still fall back rather than throwing', () => {
  const hit = findBenchmark(400, 'H100 80GB', 80);
  assert.ok(hit && hit.exact === false, '400B has no bucket but should degrade');
  assert.strictEqual(hit.nearestParams, 70);
});
test('every benchmark key is reachable as an exact match', () => {
  // A key nothing can select exactly is dead data — the bucket table and the
  // data file have drifted apart.
  const GPU = { 'a100-80': ['A100 80GB', 80], 'a100-40': ['A100 40GB', 40],
                'h100-80': ['H100 80GB', 80], 'b200-192': ['B200 192GB', 192],
                'rtx4090-24': ['RTX 4090 24GB', 24] };
  const PROBE = { '4b': 4, '7b': 7, '8b': 9, '14b': 13, '27b': 27, '32b': 35, '70b': 70 };
  for (const key of Object.keys(benchmarks.data)) {
    const m = key.match(/^(\d+b)-(.+)$/);
    const [name, gb] = GPU[m[2]];
    const hit = findBenchmark(PROBE[m[1]], name, gb);
    assert.ok(hit && hit.exact && hit.data.tokS === benchmarks.data[key].tokS,
      `${key} is unreachable: probing ${PROBE[m[1]]}B on ${name} did not return it exactly`);
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

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);
