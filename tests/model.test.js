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
    concurrency: b.mode === 'single' ? 1 : 100000,
  });
  const c = computeInference(s);
  const ours = b.mode === 'single' ? c.singleStreamTokS : c.aggregateTokS;
  test(`${key} (${b.mode}) within 4x of ${b.tokS} tok/s`, () => {
    const ratio = ours / b.tokS;
    assert.ok(ratio >= 0.25 && ratio <= 4,
      `estimate ${ours} vs measured ${b.tokS} = ${ratio.toFixed(2)}x (want 0.25-4x)`);
  });
}

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

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);
