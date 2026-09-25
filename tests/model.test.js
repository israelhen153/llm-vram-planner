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

/* computeInference() derives the TP/DP split it returns through
   parallelismFor()/splitParallelism(), so those come into scope with it the
   same way GIB and PERF do — without them the function is a ReferenceError.
   The two are adjacent in the source, so one slice carries both. */
const splitStart = html.indexOf('function splitParallelism(');
assert.notStrictEqual(splitStart, -1, 'splitParallelism() not found in index.html');
const splitEnd = html.indexOf('function renderStrategyBadges');
assert.notStrictEqual(splitEnd, -1, 'could not find the end of parallelismFor()');
const splitDecl = html.slice(splitStart, splitEnd);

const computeInference = new Function(
  `${gibDecl[0]}\n${perfDecl[0]}\n${splitDecl}\n${html.slice(start, end + 2)}; return computeInference;`
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

/* A state for one card, catalog row or not. state() below is this with a catalog
   lookup in front, so a synthetic card gets a state built the same way. */
const stateFor = (gpu, o = {}) => ({
    params: 8, activePercent: 100, bytesPerParam: 2, layers: 32, kvHeads: 8,
    headDim: 128, sharedExperts: 0, contextLength: 8192, concurrency: 1,
    gpuCount: 1, hasNVLink: true, kvBytesPerValue: 2, presetKey: '', hfModelId: null,
    modelMaxCtx: 1048576,
    gpuGB: gpu.gb, gpuBandwidth: gpu.bw, gpuTFLOPS: gpu.tflops,
    gpuHyperCost: gpu.hyper, gpuSpecCost: gpu.spec, gpuSpotCost: gpu.spot,
    gpuName: displayName(gpu),
    /* Read off the row, as readInputState() does. Defaulting this to false here
       would put every test on the ungated path and hide a gate that only ever
       fires on real hardware flags. */
    gpuFp8: !!(gpu.caps && gpu.caps.fp8),
    // Cost provenance, off the row like every other GPU field above.
    priceSource: gpu.priceSource,
    priceRecord: gpu.priceRecord,
    priceNote: gpu.priceNote,
    priceLead: gpu.priceLead,
    gfx: gpu.gfx,
    /* The key computeInference() looks PERF up by, off the row for the same
       reason. Left out, every test here would be computing a card with no
       constants while its name said H100. */
    perfKey: gpu.perfKey,
    // The form names the link the devices share, off the row too.
    gpuForm: gpu.form,
    ...o,
});
const state = (o = {}) => {
  const gpu = GPUS[o.gpu || 'H100 80GB'];
  assert.ok(gpu, `unknown GPU ${o.gpu}`);
  return stateFor(gpu, o);
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
test('above 8 devices, a dense model divides its weights by TP, not by the device count', () => {
  /* A literal, worked out by hand, not an expression over the same constants
     the engine uses — an expression follows whatever the divisor becomes and
     therefore pins nothing:

       70B at bf16      = 70e9 * 2         = 1.4e11 bytes of weights
       in GiB           = 1.4e11 / 2**30   = 130.385160446167
       12 devices split TP=4 x DP=3. Each of the 3 data-parallel replicas holds
       a whole copy of the model, sharded 4 ways inside itself:
                          130.385160446167 / 4  = 32.596290111542 GiB

     This literal was 10.865430037181 — the same weights over all 12 devices —
     and the comment here said the honest figure was three times that and that
     correcting it was a later commit. This is that commit, so the pin is
     rewritten to the number the arithmetic now produces and to the reason it
     is the right one: 12 devices cannot each hold a twelfth of the model when
     the command they are given loads three copies of it.

     Twelve devices, because at eight or fewer TP equals the device count and
     the two divisors are the same number: a pin down there proves nothing.
     Twelve is also ordinary and reachable — the slider goes to 128. */
  const c = computeInference(state({ params: 70, layers: 80, gpuCount: 12 }));
  assert.strictEqual(c.tp, 4, `TP at 12 devices: ${c.tp}`);
  assert.strictEqual(c.dp, 3, `DP at 12 devices: ${c.dp}`);
  assert.strictEqual(c.deviceCount, 12, `deviceCount: ${c.deviceCount}`);
  assert.ok(Math.abs(c.perGPU.weights - 32.5963) < 1e-4,
    `per-device weights on 12 devices: expected 32.5963 GiB, got ${c.perGPU.weights}`);
  /* And the same configuration as an MoE, which is deliberately *not* divided
     by TP. Without this the MoE hold-back is defended by a comment alone, and
     the obvious tidy-up — one divisor for everything — passes every other
     assertion in this file. 70B at 5% active is still 70B of weights on disk,
     so the literal is the same 130.385160446167 over all 12 devices. */
  const moe = computeInference(state({ params: 70, layers: 80, gpuCount: 12, activePercent: 5 }));
  assert.ok(moe.isMoE, 'the probe must be an MoE');
  assert.strictEqual(moe.tp, 4, `TP at 12 devices: ${moe.tp}`);
  assert.ok(Math.abs(moe.perGPU.weights - 10.8654) < 1e-4,
    `an MoE must still divide by all 12 devices: expected 10.8654 GiB, got ${moe.perGPU.weights}`);
  /* And free KV as an absolute, on the same twelve devices, because it is the
     figure that moves with the weights divisor without being the weights
     divisor. Every device keeps 90% of 80 GiB for vLLM and the fixed residents
     take what the breakdown above says they take:

       dense: weights 32.596290111542 + activations 1.3038516044617/4 = 0.325962901115
              + overhead 1.5 + 0.3 (NVLink)         = 34.722253012657 per device
              (72 - 34.722253012657) x 12           = 447.332963848114 GiB

       MoE:   weights 10.865430037181 + activations 0.1/12 = 0.008333333333
              + 1.8                                  = 12.673763370514 per device
              (72 - 12.673763370514) x 12            = 711.914839553833 GiB

     Both engines applying the same wrong free-KV formula is invisible to the
     parity suite, so these are literals rather than a second derivation. */
  assert.ok(Math.abs(c.freeForKVCache - 447.332963848114) < 1e-6,
    `free KV on 12 devices, dense: expected 447.332963848114 GiB, got ${c.freeForKVCache}`);
  assert.ok(Math.abs(moe.freeForKVCache - 711.914839553833) < 1e-6,
    `free KV on 12 devices, MoE: expected 711.914839553833 GiB, got ${moe.freeForKVCache}`);
  /* And the replicated half of the cluster total, which is the part this
     divisor owns. Stated without the KV and overhead terms so it does not move
     when the fixture's context or concurrency is retuned:

       dense, 3 replicas of one copy sharded 4 ways
         (weights 130.385160446167 + activations 1.303851604462) x 3
                                              = 395.067036151889 GiB
       MoE, held at one copy across all 12
         (weights 130.385160446167 + activations 0.1) x 1
                                              = 130.485160446167 GiB

     The cluster total summed a single copy at every split until this commit,
     so the dense figure here is three times what it was. Both engines carried
     that formula, which the parity suite cannot see, so this is a literal. */
  const replicated = (x) => x.totalGB - x.kvCacheGB - x.totalOverhead;
  assert.ok(Math.abs(replicated(c) - 395.067036151889) < 1e-6,
    `replicated cluster footprint, dense on 12 devices: expected 395.067036151889 GiB, ` +
    `got ${replicated(c)}`);
  assert.ok(Math.abs(replicated(moe) - 130.485160446167) < 1e-6,
    `replicated cluster footprint, MoE on 12 devices: expected 130.485160446167 GiB, ` +
    `got ${replicated(moe)}`);
});
test('a dense model holds one copy per TP group, an MoE one copy per cluster, at every count', () => {
  /* The literals above pin two numbers; this pins the shape. A change that
     divided by tp at every count except the one the literals happen to use
     would satisfy them and ship the wrong divisor everywhere else — an actual
     falsification found on this branch, not a hypothetical, so the invariant
     is stated directly for both regimes at once:

         dense:  perGPU.weights * tp          === weightsGB
         MoE:    perGPU.weights * deviceCount === weightsGB

     The right-hand side moved from deviceCount to tp for dense models in this
     commit, and stayed at deviceCount for MoE. Both halves are asserted here
     because holding only one of them lets the other slide: an engine that
     divided everything by tp, or everything by deviceCount, would still pass a
     one-sided version of this test.

     Activations replicate with the model, so they take the same divisor and are
     checked alongside — that operand was never pinned and moving weights alone
     is the obvious half-fix.

     The spread matters more than its length: tp is 1, 2, 4 or 8 across these
     counts, and dp runs from 2 to 25, so no single divisor coincidence covers
     them. 1-8 are the control, where tp === deviceCount and both formulas are
     the same statement; that is exactly why a pin down there proves nothing on
     its own. */
  let dpAbove1 = 0, tpBelowCount = 0;
  for (const count of [1, 2, 4, 5, 8, 9, 10, 12, 16, 20, 24, 40, 64, 100, 128])
    for (const params of [8, 70])
      for (const activePercent of [100, 5]) {
        const c = computeInference(state({ params, layers: 80, gpuCount: count, activePercent }));
        assert.strictEqual(c.deviceCount, count, `deviceCount at ${count}`);
        const want = c.isMoE ? c.deviceCount : c.tp;
        assert.strictEqual(c.shardDivisor, want,
          `${params}B ${c.isMoE ? 'MoE' : 'dense'} on ${count} devices: shardDivisor ` +
          `${c.shardDivisor}, expected ${want}`);
        for (const [field, whole] of [['weights', c.weightsGB], ['activations', c.activationsGB]])
          assert.ok(Math.abs(c.perGPU[field] * want - whole) <= whole * 1e-12,
            `${params}B ${c.isMoE ? 'MoE' : 'dense'} on ${count} devices (TP=${c.tp} x DP=${c.dp}): ` +
            `${c.perGPU[field]} ${field} per device x ${want} = ${c.perGPU[field] * want}, ` +
            `but the whole is ${whole} GiB`);
        // KV cache does not move with the model: DP partitions the request
        // stream, so the cluster-wide cache still spreads over every device.
        assert.ok(Math.abs(c.perGPU.kvCache * c.deviceCount - c.kvCacheGB) <= c.kvCacheGB * 1e-12,
          `${params}B on ${count} devices: KV cache must divide by all ${count} devices, ` +
          `got ${c.perGPU.kvCache} x ${c.deviceCount} against ${c.kvCacheGB}`);
        /* Free KV, against the per-device breakdown this same result reports —
           not against a second copy of the formula. freeForKVCache multiplies a
           per-device headroom by the device count, so raising the weights
           divisor without following it through there leaves free KV on the old
           number and silently hands back cache that does not exist. That exact
           mutation — fixedPerGPU recomputed as weightsGB / deviceCount while
           perGPU.weights divides by tp — passed every other test in this suite
           and in the parity suite, in both engines at once, which is why the
           invariant is stated here rather than assumed from the source. */
        /* The cluster total against the per-device breakdown, which are two
           views of one deployment and must reconcile. They are allowed to
           differ through the overhead term and through nothing else:
           totalOverhead charges (deviceCount - 1) NCCL peer buffers
           cluster-wide while perGPU.overhead gives every device one, a
           pre-existing asymmetry in the overhead model that predates this
           change and is not a sharding disagreement. Everything that does
           shard — weights, activations, KV — has to cancel exactly.

           Nothing asserted this before, which is how totalGB came through the
           weights correction still summing a single copy: it read 401.79 GiB
           beside a per-device figure implying 2377.42 on the same screen, and
           boardsNeeded() divides it into a hardware recommendation. */
        const overheadResidual = c.perGPU.overhead * c.deviceCount - c.totalOverhead;
        const residual = c.perGPU.total * c.deviceCount - c.totalGB;
        assert.ok(Math.abs(residual - overheadResidual) <= Math.max(c.totalGB, 1) * 1e-12,
          `${params}B ${c.isMoE ? 'MoE' : 'dense'} on ${count} devices (TP=${c.tp} x DP=${c.dp}): ` +
          `per-device total x ${c.deviceCount} = ${c.perGPU.total * c.deviceCount} against a cluster ` +
          `total of ${c.totalGB} — a gap of ${residual}, but only ${overheadResidual} of it is the ` +
          `NCCL peer-buffer asymmetry, so the two views disagree about the sharded quantities`);
        // And that escape hatch stays the size of one peer buffer, so it cannot
        // become somewhere a real drift hides.
        assert.ok(Math.abs(overheadResidual) <= 0.3 + 1e-9,
          `the overhead views differ by ${overheadResidual} GiB, more than one NCCL peer buffer`);

        const fixed = c.perGPU.total - c.perGPU.kvCache;
        const wantFree = Math.max((c.deviceGB * 0.9 - fixed) * c.deviceCount, 0);
        assert.ok(Math.abs(c.freeForKVCache - wantFree) <= Math.max(wantFree, 1) * 1e-12,
          `${params}B ${c.isMoE ? 'MoE' : 'dense'} on ${count} devices: free KV is ` +
          `${c.freeForKVCache} GiB, but the per-device figures leave ` +
          `(${c.deviceGB} x 0.9 - ${fixed}) x ${c.deviceCount} = ${wantFree}`);
        if (c.dp > 1) dpAbove1++;
        if (c.tp < c.deviceCount) tpBelowCount++;
      }
  // The counts where TP and the device count are different numbers are the
  // only ones this can speak about; without them it is a tautology.
  assert.ok(dpAbove1 >= 32, `only ${dpAbove1} of the swept configs have dp > 1`);
  assert.ok(tpBelowCount >= 32, `only ${tpBelowCount} of the swept configs have tp < deviceCount`);
});
test('a state carrying tp or dp keys computes exactly what one without them does', () => {
  /* computeInference() derives the split and reads nothing from state.tp. That
     is deliberate and load-bearing: the same two keys reaching the Python
     engine through a JSON config once emitted --data-parallel-size 3 beside a
     "Single device" label, and crashed outright on a string. Today the only
     thing defending it in this engine is a comment, and the next person to
     reason from first principles about where an override control would plug in
     will re-add exactly that. So it is pinned: these keys are inert, including
     the shapes that used to be actively harmful. */
  const base = state({ params: 70, layers: 80, gpuCount: 12 });
  const clean = computeInference(base);
  for (const poison of [{ tp: 1 }, { dp: 1 }, { tp: 16, dp: 4 }, { tp: '4' },
                        { tp: 2.0 }, { tp: 0 }, { dp: -1 }, { tp: null }]) {
    const got = computeInference({ ...base, ...poison });
    assert.deepStrictEqual(got, clean,
      `${JSON.stringify(poison)} in state changed the result — tp/dp must be derived, not accepted`);
  }
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
test('the interconnect factor is one curve on the device count, and this commit does not move it', () => {
  /* The penalty, recovered from the number it multiplies rather than read back
     out of the source. Decode at batch 1 is achievedBandwidth over a constant,
     and achievedBandwidth is deviceBandwidth x deviceCount x MBU x penalty, so
     the ratio against a single device of the same card divides everything else
     out:

         tok/s(n) / tok/s(1) / deviceCount === penalty(n)

     These are absolute, and they exist because both engines apply this formula:
     changing it in one of them is caught by the parity suite, changing it in
     both is not, and that mutation passed the entire suite before this test.

     They pin the curve as it already was. An earlier version of this commit
     replaced it with intra(tp) x inter(dp), which is the faithful reading of
     the emitted command — at TP=1 there is genuinely no all-reduce to price —
     and it was reverted, because the penalty was covering a different error:
     decode pools every device's bandwidth for a single user's request, while
     under DP that request runs on one replica. Removing the penalty at TP=1
     doubled a figure that was already about 4x optimistic. So this test now
     guards the status quo, and the commit that scopes single-stream and TTFT
     to a replica has to move these numbers deliberately.

       1 device     1.0
       8, NVLink    0.85          flat within a domain, full bisection
       8, PCIe      0.45          0.55 - 0.05*log2(8/2)
       9, PCIe      0.441503...   0.55 - 0.05*log2(9/2)
       16, PCIe     0.40          0.55 - 0.05*log2(16/2), at the floor
       128, PCIe    0.40          floored
       NVLink is 0.85 at every count above one. */
  /* A deliberately tiny model at a short context, because tok/s is reported as
     a rounded integer and these ratios have to resolve differences of about 1%.
     At ~2.3K tok/s on one device the rounding error is under 0.03%. */
  const PROBE = { params: 0.5, layers: 4, kvHeads: 1, headDim: 64, contextLength: 512 };
  const penaltyOf = (o) => {
    const many = computeInference(state({ ...PROBE, ...o }));
    const one = computeInference(state({ ...PROBE, ...o, gpuCount: 1 }));
    return many.singleStreamTokS / one.singleStreamTokS / many.deviceCount;
  };
  const pcie9 = 0.55 - 0.05 * Math.log2(9 / 2);
  for (const [count, nvlink, want] of [
    [1, true, 1.0], [1, false, 1.0],
    [8, true, 0.85], [8, false, 0.45],
    [9, true, 0.85], [9, false, pcie9],
    [16, true, 0.85], [16, false, 0.40],
    [128, true, 0.85], [128, false, 0.40],
  ]) {
    const got = penaltyOf({ gpuCount: count, hasNVLink: nvlink });
    const c = computeInference(state({ ...PROBE, gpuCount: count, hasNVLink: nvlink }));
    assert.ok(Math.abs(got - want) <= want * 1e-3,
      `${count} devices on ${nvlink ? 'NVLink' : 'PCIe'} (TP=${c.tp} x DP=${c.dp}): ` +
      `interconnect factor ${got}, expected ${want}`);
  }
  /* And the known-wrong case, pinned as known-wrong rather than left silent: at
     nine devices the split is TP=1 x DP=9 and the emitted command runs no
     tensor-parallel all-reduce at all, yet the fabric is still priced against
     decode — NVLink and PCIe give different answers for nine independent
     replicas. That is the status quo this commit deliberately does not touch,
     and the test says so out loud so the next one cannot change it by accident
     and call it a refactor. */
  const nine = computeInference(state({ ...PROBE, gpuCount: 9 }));
  assert.strictEqual(nine.tp, 1, 'nine devices should split TP=1 x DP=9');
  assert.notStrictEqual(penaltyOf({ gpuCount: 9, hasNVLink: true }),
                        penaltyOf({ gpuCount: 9, hasNVLink: false }),
    'nine one-device replicas run no TP collective, yet the fabric is still ' +
    'priced — known, deliberate, and the next commit\'s subject, not this one\'s');
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

console.log('\nThe compute roofline knows what dtype the GEMM runs in');
/* The catalog's tflops is dense BF16, and both compute figures ran on it
   regardless of the selected precision — so FP8 was 2x understated on hardware
   that has FP8 tensor cores. These pin the multiplier, and more importantly they
   pin what it must NOT do: 4-bit is W4A16, dequantized to FP16 before the GEMM,
   so it buys memory traffic and no compute. Scaling by 1/bytesPerParam is the
   intuitive fix and would give AWQ a 4x ceiling it does not have. */
const ceilingOf = (s) => {
  const c = computeInference(s);
  // Recovered rather than returned: aggregateTokS takes a min() against the
  // bandwidth roofline, so it does not expose the ceiling on its own.
  return (PERF.nvidia.mfuDecode * s.gpuTFLOPS * 1e12 * c.computeRatio) / (2 * s.params * 1e9);
};
test('FP8 on an FP8-capable card doubles the compute ceiling and halves TTFT', () => {
  const bf16 = computeInference(state({ bytesPerParam: 2, quantMethod: '' }));
  const fp8 = computeInference(state({ bytesPerParam: 1, quantMethod: 'fp8' }));
  assert.strictEqual(fp8.computeRatio, 2.0, 'H100 has FP8 tensor cores');
  assert.strictEqual(bf16.computeRatio, 1.0, 'BF16 runs at the BF16 rate');
  between(fp8.ttftMs, bf16.ttftMs * 0.49, bf16.ttftMs * 0.51,
    `TTFT should halve under FP8: ${bf16.ttftMs} -> ${fp8.ttftMs}`);
});
test('FP8 on a card without FP8 tensor cores gets no compute multiplier', () => {
  /* A100 is the case that makes the gate worth having. vLLM's fp8_marlin runs
     there and the weights genuinely halve, so the bandwidth win is real — but
     Ampere has no FP8 tensor cores and dequantizes to FP16, so the GEMM rate is
     unchanged. Memory yes, compute no. */
  const bf16 = computeInference(state({ gpu: 'A100 80GB', bytesPerParam: 2, quantMethod: '' }));
  const fp8 = computeInference(state({ gpu: 'A100 80GB', bytesPerParam: 1, quantMethod: 'fp8' }));
  assert.strictEqual(fp8.computeRatio, 1.0, 'A100 has no FP8 tensor cores');
  assert.strictEqual(fp8.ttftMs, bf16.ttftMs, 'TTFT is compute-bound and must not move on Ampere');
  assert.ok(fp8.singleStreamTokS > bf16.singleStreamTokS,
    'but the bandwidth win from half-size weights is real and must survive');
  assert.strictEqual(fp8.fp8NoTensorCores, true, 'and the config must be flagged for the caveat');
});
test('4-bit weights buy memory traffic and no compute at all', () => {
  const bf16 = computeInference(state({ bytesPerParam: 2, quantMethod: '' }));
  for (const q of ['awq', 'gptq']) {
    const c = computeInference(state({ bytesPerParam: 0.5, quantMethod: q }));
    assert.strictEqual(c.computeRatio, 1.0, `${q} is W4A16 — the GEMM still runs at FP16`);
    assert.strictEqual(c.ttftMs, bf16.ttftMs, `${q} must not move TTFT`);
    assert.strictEqual(Math.round(ceilingOf(state({ bytesPerParam: 0.5, quantMethod: q }))),
      Math.round(ceilingOf(state({ bytesPerParam: 2, quantMethod: '' }))),
      `${q} must not move the compute ceiling`);
    assert.ok(c.singleStreamTokS > bf16.singleStreamTokS,
      `${q} must still win on bandwidth`);
  }
});
test('GGUF Q8_0 is not FP8, despite being close to one byte', () => {
  // 1.1 B/param. The bpp==1 fallback that catches a bare {"bpp": 1} config must
  // not spill onto the GGUF level sitting next to it.
  const c = computeInference(state({ bytesPerParam: 1.1, quantMethod: 'gguf' }));
  assert.strictEqual(c.computeRatio, 1.0, 'Q8_0 dequantizes like every other GGUF level');
});
test('every catalog row declares the FP8 support its silicon actually has', () => {
  /* The flag's *value* was pinned on five rows and inferred on the rest, so
     flipping l40s-48 to false shipped green while halving that card's ceiling
     and printing a warning that is not true of it. A capability flag is a
     contract about hardware, not a derivable property, so it gets literals —
     and the literals are the architecture, not the catalog, or this would be
     the catalog checked against itself.

       Turing (T4)      sm75  no FP8 tensor cores
       Ampere (A100)    sm80  no FP8 tensor cores
       Ada / Hopper /   sm89+ FP8 tensor cores at 2x the BF16 rate
       Blackwell                                                            */
  const FP8_BY_ARCH = {
    't4-16': false, 'a100-40': false, 'a100-80': false,
    'l4-24': true, 'l40s-48': true, 'rtx4090-24': true, 'rtx5090-32': true,
    'rtx6000ada-48': true, 'rtxpro-96': true,
    'h100-80': true, 'h200-141': true, 'b200-192': true,
    // AMD, per ROCm's precision-support table: FP8 matrix support on CDNA3 only.
    'rx7900xtx-24': false /* RDNA3 */, 'mi210-64': false /* CDNA2 */, 'mi250x-128': false /* CDNA2 */,
    'mi300x-192': true /* CDNA3 */, 'mi325x-256': true /* CDNA3 */,
  };
  assert.deepStrictEqual(Object.keys(FP8_BY_ARCH).sort(), Object.keys(GPU_TABLE).sort(),
    'a catalog row was added or removed without deciding its FP8 support here');
  for (const [slug, expected] of Object.entries(FP8_BY_ARCH)) {
    assert.strictEqual(GPU_TABLE[slug].caps.fp8, expected,
      `${slug}.caps.fp8 is ${GPU_TABLE[slug].caps.fp8}, but its architecture says ${expected}` +
      ` — this flag halves the compute ceiling and prints a user-facing warning`);
  }
});
test('the AMD rows carry the figures AMD publishes, and are the AMD rows', () => {
  /* Literals, typed from AMD's own documents — docs/research/amd-gpu-specs.md
     cites each. A card with no throughput constants shows its TFLOPS nowhere,
     so a figure off by 2x, the MI250X's 383 read per GCD and doubled as this
     project's own research once did, passed every other test
     (engine_r6_amd_rows C3). The vendor is pinned for the same reason: the
     first AMD row relabelled nvidia slid into the end of the NVIDIA section
     with the dropdown's text unchanged (C7). A row added under vendor amd
     without deciding its figures here fails. */
  const AMD_SPECS = {
    'rx7900xtx-24': { gb: 24, bw: 960, tflops: 123, devices: 1, form: 'consumer', fp8: false },
    'mi210-64': { gb: 64, bw: 1638.4, tflops: 181, devices: 1, form: 'pcie', fp8: false },
    'mi250x-128': { gb: 128, bw: 3276.8, tflops: 383, devices: 2, form: 'oam', fp8: false },
    'mi300x-192': { gb: 192, bw: 5325, tflops: 1307.4, devices: 1, form: 'oam', fp8: true },
    'mi325x-256': { gb: 256, bw: 6000, tflops: 1307.4, devices: 1, form: 'oam', fp8: true },
  };
  const amd = Object.keys(GPU_TABLE).filter(k => GPU_TABLE[k].vendor !== 'nvidia').sort();
  assert.deepStrictEqual(amd, Object.keys(AMD_SPECS).sort(),
    'the rows whose vendor is not nvidia are not exactly the pinned AMD rows');
  for (const [slug, want] of Object.entries(AMD_SPECS)) {
    const row = GPU_TABLE[slug];
    assert.strictEqual(row.vendor, 'amd', `${slug}.vendor`);
    const got = { gb: row.gb, bw: row.bw, tflops: row.tflops, devices: row.devices, form: row.form, fp8: row.caps.fp8 };
    assert.deepStrictEqual(got, want, `${slug} no longer carries AMD's published figures`);
  }
});
test('every catalog row names the constants its silicon was measured with', () => {
  /* A literal per row, for the reason the FP8 flags above are literals: which
     constants a card runs on is a statement about hardware, and a catalog
     checked against itself passes any value. Moving one row to a key PERF does
     not have — through data/gpus.json and the sync tool, so every other check
     agrees with itself — would take that card's throughput off the page, and
     no other test reads most of these rows' throughput. The twelve NVIDIA
     cards run on nvidia, the entry PERF has for them; the five AMD cards name
     their architecture, which PERF has no entry for, so their throughput is
     absent by design. A row added without deciding its key here fails. */
  const PERF_KEY_BY_ROW = {
    't4-16': 'nvidia', 'l4-24': 'nvidia', 'rtx4090-24': 'nvidia', 'rtx5090-32': 'nvidia',
    'a100-40': 'nvidia', 'rtx6000ada-48': 'nvidia', 'l40s-48': 'nvidia', 'a100-80': 'nvidia',
    'h100-80': 'nvidia', 'rtxpro-96': 'nvidia', 'h200-141': 'nvidia', 'b200-192': 'nvidia',
    'rx7900xtx-24': 'rdna3', 'mi210-64': 'cdna2', 'mi250x-128': 'cdna2',
    'mi300x-192': 'cdna3', 'mi325x-256': 'cdna3',
  };
  assert.deepStrictEqual(Object.keys(PERF_KEY_BY_ROW).sort(), Object.keys(GPU_TABLE).sort(),
    'a catalog row was added or removed without deciding its perfKey here');
  for (const [slug, key] of Object.entries(PERF_KEY_BY_ROW)) {
    assert.strictEqual(GPU_TABLE[slug].perfKey, key,
      `${slug}.perfKey is ${GPU_TABLE[slug].perfKey}, but its silicon is measured under ${key}`);
    // And the engine agrees about whether that key has constants.
    assert.strictEqual(computeInference(stateFor(GPU_TABLE[slug])).throughputModelled, Object.hasOwn(PERF, key),
      `${slug}: the engine disagrees about whether ${key} has constants`);
  }
});
test('the FP8 multiplier never reaches the single-stream figure', () => {
  /* Single-stream decode is bandwidth-bound at batch 1 and is the one number
     the tool matches measurements on closely. The compute ratio must not touch
     it — and applying it there, symmetrically in both engines, passed every
     other test in this suite: the band test cannot see it (the only measured
     FP8 entry is batch-mode, both single-stream entries are q4) and the FP8
     assertions above run on the A100, where the ratio is 1.0 anyway. */
  for (const gpu of ['H100 80GB', 'B200 192GB', 'L40S 48GB']) {
    const bf16 = computeInference(state({ gpu, bytesPerParam: 2, quantMethod: '' }));
    const fp8 = computeInference(state({ gpu, bytesPerParam: 1, quantMethod: 'fp8' }));
    assert.strictEqual(fp8.computeRatio, 2.0, `${gpu} should take the multiplier`);
    /* Weights halve, so single-stream roughly doubles from bandwidth alone.
       What it must not do is double *again*. The bound is derived from the
       bandwidth model, not a literal: denominator is weights + one sequence's
       KV, and only the weights term halves. */
    const ratio = fp8.singleStreamTokS / bf16.singleStreamTokS;
    assert.ok(ratio > 1 && ratio < 2.05,
      `${gpu}: single-stream moved ${ratio.toFixed(2)}x under FP8. Above ~2x means the ` +
      `compute ratio has leaked into the bandwidth path, which it must never touch.`);
  }
});
test('a bare one-byte config is treated as FP8 by both the label and the maths', () => {
  /* generate_report.py's from_json() accepts {"bpp": 1} with no quant key, and
     PREC_LABELS already renders that as "FP8". If only the label knew, the PDF
     would say FP8 over BF16 arithmetic. Both engines read bpp as well as quant. */
  const c = computeInference(state({ bytesPerParam: 1, quantMethod: '' }));
  assert.strictEqual(c.computeRatio, 2.0, 'bpp of exactly 1 is FP8 on an FP8-capable card');
});

console.log('\nAgreement with published benchmarks');
// The point of the rewrite: compare each estimate against the matching mode.
// Order-of-magnitude agreement (0.25x-4x) is the bar for a planning tool.
const PREC_BYTES = { bf16: 2, fp8: 1, q4: 0.5, int4: 0.5 };

/* Which measured entries are scored, and against what estimate. A measurement on
   hardware with no constants has no estimate to be scored against: it is marked
   as such, not scored with another key's constants and not dropped in silence.
   A function of the entries and the catalog, like bandRatios() below, so the skip
   is driven with a synthetic card too — no real entry reaches it yet. */
const scoringPlan = (entries, table) => {
  const plan = [];
  for (const [key, b] of Object.entries(entries)) {
    if (b.estimated) continue; // only score against real measurements
    const m = key.match(/^(\d+)b-(.+)$/);
    assert.ok(m && Object.hasOwn(table, m[2]), `benchmark key "${key}" names no row in the catalog`);
    const params = Number(m[1]), card = table[m[2]];
    // Benchmarks are run at short context with a full batch; mirror that.
    const c = computeInference(stateFor(card, {
      params, gpuCount: key === '70b-h100-80' ? 2 : 1,
      bytesPerParam: PREC_BYTES[b.prec] ?? 2,
      layers: params >= 60 ? 80 : params >= 20 ? 48 : 32,
      contextLength: b.mode === 'single' ? 16384 : 1024,
      concurrency: 1,
    }));
    if (!Object.hasOwn(PERF, card.perfKey)) {
      plan.push({ key, b, skipped: card.perfKey, modelled: c.throughputModelled });
      continue;
    }
    // Batch benchmarks are run saturated, so score against the saturated estimate.
    plan.push({ key, b, ours: b.mode === 'single' ? c.singleStreamTokS : c.saturatedTokS });
  }
  return plan;
};
for (const { key, b, ours, skipped, modelled } of scoringPlan(benchmarks.data, GPU_TABLE)) {
  if (skipped) {
    test(`${key} (${b.mode}) is not scored: ${skipped} has no constants`, () => {
      assert.strictEqual(modelled, false, `the engine produced an estimate for ${key} anyway`);
    });
    continue;
  }
  test(`${key} (${b.mode}) within 4x of ${b.tokS} tok/s`, () => {
    const ratio = ours / b.tokS;
    assert.ok(ratio >= 0.25 && ratio <= 4,
      `estimate ${ours} vs measured ${b.tokS} = ${ratio.toFixed(2)}x (want 0.25-4x)`);
  });
}

console.log('\nObserved-efficiency band');
/* The band belongs to one set of constants, so it is derived per perfKey: every
   measured batch entry is scored on the key of its own card. A key PERF has no
   entry for is skipped, not scored — its card has no estimate to divide by, and
   borrowing PERF.nvidia's to make one is the fallback this suite keeps out. The
   test this replaces read PERF.nvidia for every entry, whatever its card.

   Takes the entries and the catalog as arguments, so the skip can be exercised
   with a card the real catalog does not have yet. */
const bandRatios = (entries, table) => {
  const byKey = {}, skipped = [];
  for (const [key, b] of Object.entries(entries)) {
    if (b.estimated || b.mode !== 'batch') continue;
    const m = key.match(/^(\d+)b-(.+)$/);
    assert.ok(m && Object.hasOwn(table, m[2]), `benchmark key "${key}" names no row in the catalog`);
    const params = Number(m[1]), card = table[m[2]];
    if (!Object.hasOwn(PERF, card.perfKey)) { skipped.push(key); continue; }
    const s = stateFor(card, {
      params, gpuCount: key === '70b-h100-80' ? 2 : 1,
      bytesPerParam: PREC_BYTES[b.prec] ?? 2,
      layers: params >= 60 ? 80 : params >= 20 ? 48 : 32,
      contextLength: 1024, concurrency: 1,
    });
    (byKey[card.perfKey] = byKey[card.perfKey] || []).push(b.tokS / computeInference(s).saturatedTokS);
  }
  return { byKey, skipped };
};
test('each declared band matches its own measured spread, and is no wider', () => {
  // Re-derive the bands from the data so they cannot silently drift as entries land.
  const { byKey } = bandRatios(benchmarks.data, GPU_TABLE);
  /* Both directions: a band with no measurements behind it is not evidence, and
     measurements whose key declares no band would have nothing to track. */
  assert.deepStrictEqual(Object.keys(byKey).sort(), Object.keys(PERF).sort(),
    `PERF declares bands for [${Object.keys(PERF)}], measured batch entries exist for [${Object.keys(byKey)}]`);
  for (const [perfKey, ratios] of Object.entries(byKey)) {
    assert.ok(ratios.length >= 3, `${perfKey}: need >=3 measured batch entries, got ${ratios.length}`);
    const lo = Math.min(...ratios), hi = Math.max(...ratios);
    const { obsLo, obsHi } = PERF[perfKey];
    // Two-sided deliberately. Asserting only that the data fits inside the band lets
    // the band be widened to fit anything, and wider is the flattering direction:
    // it makes the tool look like it predicted whatever was measured. README.md and
    // MODEL.md both promise this band tracks the evidence, so pin both edges.
    assert.ok(Math.abs(obsLo - lo) <= 0.02 && Math.abs(obsHi - hi) <= 0.02,
      `measured/ceiling ratios on ${perfKey} span ${lo.toFixed(2)}-${hi.toFixed(2)} but ` +
      `PERF.${perfKey} declares ${obsLo}-${obsHi} — the band must track the data in both ` +
      `directions. Update obsLo/obsHi in index.html and generate_report.py to match the measurements.`);
  }
});
test('a measurement on hardware with no constants is never scored — not by the band, not per entry', () => {
  /* A synthetic card with no PERF entry, and a measured batch entry on it shaped
     exactly like a real one. The catalog has no such row yet, so the case is
     built rather than found. */
  const probe = { ...GPU_TABLE['h100-80'], name: 'Unmeasured 80 GB', vendor: 'acme', perfKey: 'no-such-key' };
  const entries = { ...benchmarks.data, '8b-probe-unmeasured': { ...benchmarks.data['8b-h100-80'] } };
  assert.ok(!benchmarks.data['8b-h100-80'].estimated && benchmarks.data['8b-h100-80'].mode === 'batch',
    'the entry the probe copies has to be one the band scores');
  const real = bandRatios(benchmarks.data, GPU_TABLE);
  const withProbe = bandRatios(entries, { ...GPU_TABLE, 'probe-unmeasured': probe });
  assert.deepStrictEqual(withProbe.skipped, ['8b-probe-unmeasured'],
    `the entry on a card without constants was not the one skipped: ${withProbe.skipped}`);
  assert.ok(!Object.hasOwn(withProbe.byKey, 'no-such-key'), 'a key with no constants was given a band');
  assert.deepStrictEqual(withProbe.byKey, real.byKey,
    'a measurement on a card without constants changed a band it does not belong to');
  /* Two-sided: the same entry on the same card with constants is scored, and
     lands on that key's band. Without this the skip could be skipping
     everything it is handed. */
  const twin = bandRatios(entries, { ...GPU_TABLE, 'probe-unmeasured': { ...probe, perfKey: 'nvidia' } });
  assert.deepStrictEqual(twin.skipped, [], 'an entry on a card with constants was skipped');
  assert.strictEqual(twin.byKey.nvidia.length, real.byKey.nvidia.length + 1,
    'an entry on a card with constants was not scored on its key');
  // And the per-entry scoring above follows the same rule, both ways.
  const planned = (tbl) => scoringPlan(entries, tbl).find(e => e.key === '8b-probe-unmeasured');
  const skipped = planned({ ...GPU_TABLE, 'probe-unmeasured': probe });
  assert.ok(skipped && skipped.skipped === 'no-such-key' && skipped.ours === undefined && skipped.modelled === false,
    `the per-entry scoring scored a card with no constants: ${JSON.stringify(skipped)}`);
  const scored = planned({ ...GPU_TABLE, 'probe-unmeasured': { ...probe, perfKey: 'nvidia' } });
  assert.ok(scored && !scored.skipped && scored.ours > 0,
    `the per-entry scoring skipped a card with constants: ${JSON.stringify(scored)}`);
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
const readInputStateFor = (gpuKey, interconnect, table = GPU_TABLE) => {
  const dom = domStub(gpuKey, interconnect);
  // Up to the PERF declaration, which is where the state builder's
  // neighbourhood ends. Anchored on the code rather than on the comment above
  // it: a reworded comment made indexOf return -1, and slice(start, -1) is
  // most of the file.
  const stop = html.indexOf('\nconst PERF = {');
  assert.ok(stop > 0, 'PERF declaration not found in index.html');
  const src = html.slice(html.indexOf('function getVal(id)'), stop);
  const fn = new Function('document', 'GPU_TABLE', `
    let currentAttn = { mode: 'standard', window: 0, localLayers: 0, mlaDim: 0 };
    let currentModelMaxCtx = 131072, importedModelId = null;
    ${src}
    return readInputState;`)(dom, table);
  return fn();
};

console.log('\nA board is not always one device');
/* The catalog stores per-board figures because that is the unit you buy. A
   multi-GCD module — one OAM presenting two GCDs — is one row with devices: 2.
   Everything except cost has to be scoped to devices, and each of the three
   traps below passes the existing suite while being wrong. */
/* perfKey is explicit, and has to be: this fixture is here to test how a board
   splits into devices, and without a key it would silently compute a card with no
   throughput constants instead — still green, testing something else. */
const dualGCD = { gb: 128, bw: 3276.8, tflops: 383, hyper: 6.0, spec: 2.5, spot: 1.2,
                  name: 'Dual-GCD 128 GB', vendor: 'nvidia', perfKey: 'nvidia', devices: 2, form: 'sxm' };
const asState = (card, count, extra = {}) => ({
  params: 70, activePercent: 100, bytesPerParam: 2, layers: 80, kvHeads: 8, headDim: 128,
  sharedExperts: 0, contextLength: 8192, concurrency: 16, gpuCount: count,
  hasNVLink: true, kvBytesPerValue: 2, modelMaxCtx: 1048576, vendor: card.vendor,
  perfKey: card.perfKey,
  gpuGB: card.gb, gpuBandwidth: card.bw, gpuTFLOPS: card.tflops, gpuDevices: card.devices,
  gpuHyperCost: card.hyper, gpuSpecCost: card.spec, gpuSpotCost: card.spot,
  // Off the row, as readInputState() does — see the same line in state() above.
  gpuFp8: !!(card.caps && card.caps.fp8),
  // Cost provenance, off the row like everything above. Left undefined for
  // every synthetic card here (dualGCD, single, the perfKey overrides) since
  // none of them carries one — readInputState() leaves it undefined too, off
  // a real row with no confirmed source, and priceSourceLabel() treats that
  // as "not recorded" rather than throwing.
  priceSource: card.priceSource,
  priceRecord: card.priceRecord,
  priceNote: card.priceNote,
  priceLead: card.priceLead,
  gfx: card.gfx,
  gpuForm: card.form,
  /* The name as the page's state carries it: getGpuSpec() drops the space
     ("MI250X 128GB"), as stateFor() above does. card.name kept the catalog's
     spelling, so no state built here ever carried the name the page does — a
     leak keyed on it passed the whole suite (engine_r6_amd_rows A5), and the
     golden recorded a spelling the page never shows. */
  gpuName: displayName(card), ...extra,
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
/* The page's script, from the first helper the renderers need to the line that
   boots the page. Everything before that boundary is a declaration, so the whole
   of it can be evaluated without anything running; everything after it is the
   three calls that start the tool.

   Every `function render…` index.html declares is taken from that slice and
   returned by the harness, rather than listed by hand. The hand-written list had
   missed four — renderGpuOptions, renderSliderLabels, renderCountingToggle and
   renderRestoreNotice — so whatever they put on the page was unread, and a
   renderer added tomorrow would have been unread too. */
const HARNESS_SLICE_START = html.indexOf('function getVal(id)');
const HARNESS_SLICE_END = html.indexOf('\nrenderGpuOptions();');
assert.ok(HARNESS_SLICE_START > 0 && HARNESS_SLICE_END > HARNESS_SLICE_START,
  'index.html no longer has the boundaries the render harness slices on');
const HARNESS_SLICE = html.slice(HARNESS_SLICE_START, HARNESS_SLICE_END);
const namesIn = (src) => [...new Set([...src.matchAll(/^function (render\w+)\(/gm)].map(m => m[1]))].sort();
const RENDER_NAMES = namesIn(HARNESS_SLICE);
assert.deepStrictEqual(RENDER_NAMES, namesIn(html),
  'index.html declares a render function outside the harness slice, so nothing renders it');

const renderHarness = (inputs = {}) => {
  const out = {};
  const values = {
    'train-method': 'lora', 'train-optimizer': 'adam', 'train-grad-ckpt': '1',
    'train-batch-size': '1', 'train-seq-len': '2048', 'train-lora-rank': '16',
    'train-lora-targets': '4', 'train-hidden-size': '4096', ...inputs,
  };
  /* Everything a renderer can put in front of a reader, not only innerHTML. A
     textContent write, a title, a data-* attribute: each is as visible as a tile
     — a title is what a reader sees on hover — and each is a place a figure can
     be printed where a harness that records innerHTML alone would never look. */
  const shown = {}, props = {};
  const el = (id) => ({
    get innerHTML() { return out[id] || ''; }, set innerHTML(v) { out[id] = v; },
    get textContent() { return shown[id] || ''; }, set textContent(v) { shown[id] = String(v); },
    get title() { return (props[id] || {}).title || ''; },
    set title(v) { (props[id] = props[id] || {}).title = String(v); },
    setAttribute(name, v) { (props[id] = props[id] || {})[name] = String(v); },
    removeAttribute(name) { delete (props[id] || {})[name]; },
    dataset: new Proxy({}, {
      get: (_, k) => (props[id] || {})[`data-${String(k)}`],
      set: (_, k, v) => { (props[id] = props[id] || {})[`data-${String(k)}`] = String(v); return true; },
    }),
    /* innerText renders the same words as textContent and is a different
       property: a harness that recorded one and not the other could be written
       around by changing which one the renderer assigns. */
    get innerText() { return (props[id] || {}).innerText || ''; },
    set innerText(v) { (props[id] = props[id] || {}).innerText = String(v); },
    /* Hiding an element withholds everything in it while leaving its innerHTML
       byte-identical, so `hidden` and `style.display` are recorded per element
       rather than into a per-call object nobody reads. "Collapse the empty
       sections" keyed on the wrong flag looks exactly like this. */
    get hidden() { return (props[id] || {}).hidden === 'true'; },
    set hidden(v) { (props[id] = props[id] || {}).hidden = String(!!v); },
    style: new Proxy({}, {
      get: (_, k) => (props[id] || {})[`style.${String(k)}`] || '',
      set: (_, k, v) => { (props[id] = props[id] || {})[`style.${String(k)}`] = String(v); return true; },
    }),
    get value() { return values[id] !== undefined ? values[id] : ''; }, set value(v) { values[id] = v; },
    checked: true,
    /* The preset dropdown, so a state that names a preset renders the way the
       page renders it: the executive view and the copied report both read the
       selected option's label. */
    options: [{ text: 'Llama 3.1 8B', textContent: 'Llama 3.1 8B', disabled: false }], selectedIndex: 0,
    selectedOptions: [{ dataset: { q: '' } }],
    classList: { add() {}, remove() {} },
  });
  const cache = {};
  const document = {
    /* The browser tab. Nothing in index.html writes it today, so this records a
       surface that is empty — and an empty recorded surface is how a figure put
       there later becomes visible. It is as much in front of a reader as a tile:
       it is the window's name. */
    get title() { return (props['(document)'] || {}).title || ''; },
    set title(v) { (props['(document)'] = props['(document)'] || {}).title = String(v); },
    getElementById: (id) => (cache[id] = cache[id] || el(id)),
    // Executive mode is on: renderExecutiveSummary() returns immediately
    // otherwise, and this suite exists to read what it writes.
    body: { classList: { add() {}, remove() {}, contains: () => true } },
    /* The copied report quotes the command the page is showing. Returning an
       empty string here left that block empty for every card, so a renderer that
       dropped the command for one of them changed nothing a test could see. */
    querySelector: (sel) => {
      /* As a browser resolves them: the panel's first <code> wherever it sits, or the
         one inside the command box. The first is what the report used to quote, and a
         banner's <code> span precedes the box, so modelling both is what lets a test
         see the difference. */
      const panel = out['command-output'] || '';
      if (sel === '#command-output code') {
        const m = panel.match(/<code>([\s\S]*?)<\/code>/);
        return m ? { textContent: m[1] } : null;
      }
      if (sel === '#command-output .code-box code') {
        const m = panel.match(/<div class="code-box">[\s\S]*?<code>([\s\S]*?)<\/code>/);
        return m ? { textContent: m[1] } : null;
      }
      return { textContent: '', innerHTML: '', parentElement: null };
    },
    querySelectorAll: () => [],
  };
  const start = HARNESS_SLICE_START;
  const before = [/^const GPU_TABLE = \{[\s\S]*?\n\};$/m, /^const BENCHMARK_DATA = \{[\s\S]*?\n\};$/m,
                  /^const MODEL_PRESETS = \{[\s\S]*?\n\};$/m, /^const WORKLOAD_PROFILES = \{[\s\S]*?\n\};$/m]
    .map(re => html.match(re)[0]).filter(d => html.indexOf(d) < start).join('\n');
  const navigator = { clipboard: { writeText: () => Promise.resolve() } };
  const api = new Function('document', 'navigator', `
    let currentAttn = { mode: 'standard', window: 0, localLayers: 0, mlaDim: 0 };
    // urlRestoreLost is no longer declared here: index.html declares it inside
    // the slice below, and declaring one name twice is a syntax error.
    let currentModelMaxCtx = 131072, importedModelId = null;
    // The real name, declared before this slice begins. It was previously
    // spelled comparisonSnapshots — a name index.html does not contain — so
    // renderComparisons() threw on sight and no test could call it.
    let savedSnapshots = [];
    ${before}
    ${HARNESS_SLICE}
    return { ${RENDER_NAMES.join(', ')},
             exportSummary, computeInference, buildVllmCommand, boardsNeeded, boardsAdvice,
             // The page's other naming path: a model imported by HuggingFace id
             // rather than chosen from the presets.
             setImportedModel: (v) => { importedModelId = v; },
             /* The real saveSnapshot(), not a push that names the snapshot for
                it. The name is built in there — from the preset dropdown, or the
                imported id, or the parameter count — and a figure put in it
                reaches the comparison view, which the harness never saw while it
                was naming every snapshot "snap". Only where the state comes from
                is stubbed, which is what this harness stubs everywhere. */
             pushSnapshot: (s, c) => {
               const realState = readInputState, realCompute = computeInference;
               readInputState = () => s; computeInference = () => c;
               try { saveSnapshot(); } finally { readInputState = realState; computeInference = realCompute; }
             } };`)(document, navigator);
  return { ...api, out, shown, props };
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
const formatTokensLike = new Function(
  `${html.match(/^function formatTokens\(t\) .+$/m)[0]}; return formatTokens;`)();
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
  /* The figure, not just the noun. This used to compare the banner against
     ceil(totalGB / (deviceGB * 0.9)) — the circular formula the banner no
     longer uses — so it would have gone on agreeing with whatever that formula
     produced. It now compares the banner against the count the search names,
     and the neighbouring test proves that count actually fits. */
  const shown = Number((text.match(/Smallest fit: (\d+) boards/) || [])[1]);
  assert.strictEqual(shown, h.boardsNeeded(state, computed).boards,
    `banner says ${shown} boards, the search says ` +
    `${h.boardsNeeded(state, computed).boards}: ${text}`);
  assert.ok(h.computeInference({ ...state, gpuCount: shown }).fits,
    `the banner names ${shown} boards, which does not fit`);
});
test('the sharding a card claims is the sharding the command performs', () => {
  // Five dual-GCD boards: ten devices, and the condensed view used to caption
  // them "sharded 5-way" beside a number divided by ten.
  const h = renderHarness();
  /* Both branches: four dual-GCD boards give eight devices and a pure TP=8
     split, five give ten and a TP=2 x DP=5 one. The review found the dp>1
     branch of this test dead once already, because the comment said five and
     the code said four.

     Dense and MoE at each, because above one domain they take different
     divisors and only one of them is the emitted --tensor-parallel-size. A
     dense-only version of this passed while the dp>1 branch had silently
     drifted off the weights tile and onto the layers tile, which carries the
     same words for a different reason. */
  for (const boards of [4, 5])
    for (const activePercent of [100, 10]) {
      const state = asState(dualGCD, boards, { params: 8, layers: 32, bytesPerParam: 2, activePercent });
      const computed = h.computeInference(state);
      h.renderGPUCards(state, computed);
      const text = h.out['gpu-cards'] || h.out['vram-output'] || Object.values(h.out).join(' ');
      const cmd = h.buildVllmCommand(state, computed, 'm');
      const tp = Number((cmd.match(/--tensor-parallel-size (\d+)/) || [0, 1])[1]);
      const dp = Number((cmd.match(/--data-parallel-size (\d+)/) || [0, 1])[1]);
      assert.strictEqual(tp * dp, computed.deviceCount,
        'the command must account for every device the cards describe');
      /* The figure against the flag, directly: what the weights number was
         divided by, read back out of the number itself, against what the
         command tells vLLM to shard by. For a dense model these must be the
         same integer at every count — that is the whole subject of this
         change, and nothing else in the suite compares the two code paths. */
      const divisor = computed.weightsGB / computed.perGPU.weights;
      assert.strictEqual(divisor, computed.shardDivisor,
        `perGPU.weights implies a divisor of ${divisor} but shardDivisor says ${computed.shardDivisor}`);
      if (!computed.isMoE) {
        assert.strictEqual(divisor, tp,
          `the weights figure was divided by ${divisor} while the command shards ${tp}-way:\n${cmd}`);
      } else {
        /* An MoE is held at the device count, and that is only defensible
           because the command asks for expert parallelism. If the flag ever
           stops being emitted, the divisor loses its justification and this
           says so rather than leaving the two to drift. */
        assert.strictEqual(divisor, computed.deviceCount,
          `an MoE must stay divided by all ${computed.deviceCount} devices, got ${divisor}`);
        assert.match(cmd, /--enable-expert-parallel/,
          `the MoE divisor assumes expert parallelism; the command does not ask for it:\n${cmd}`);
      }
      // Every sharding factor the view states, not the first one that matches:
      // the tile and the interconnect line each carry one, set separately.
      const ways = [...text.matchAll(/(\d+)-way/g)].map(m => Number(m[1]));
      if (dp === 1 || !computed.isMoE) {
        assert.ok(ways.length, `the condensed view should state a sharding factor: ${text.slice(0, 200)}`);
        for (const w of ways)
          assert.strictEqual(w, divisor,
            `a card says ${w}-way while the figure beside it was divided by ${divisor}`);
      } else {
        // Above one domain an expert-parallel MoE does not shard one way, so
        // the panel states the divisor it used and must not dress it up as a
        // sharding factor.
        assert.ok(!ways.length,
          `an expert-parallel MoE across ${computed.deviceCount} devices must not caption ` +
          `a single sharding factor: ${ways.join(', ')}`);
        const claim = text.match(/divided by all (\d+) devices/);
        assert.ok(claim, `a dp>1 MoE view must name the divisor it used: ${text.slice(0, 300)}`);
        assert.strictEqual(Number(claim[1]), divisor,
          'the claim must name the count the figure was actually divided by');
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
  h.renderNotes(state, h.computeInference(state));
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
    h.renderNotes(state, computed);
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
    .replace(/Smallest fit: \d+ boards/g, 'Smallest fit: N boards')
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
test('the badge says single device exactly when the command asks for no parallelism', () => {
  /* Two gates, written differently, that have to agree for every input: the
     badge branches on `computed.deviceCount > 1` and the command on
     `tp * dp > 1`. They are equal for every valid configuration — but that is
     a property, and until it is checked it is a claim. A report that says
     "Single device" above a --data-parallel-size flag is the exact shape of
     defect this pair has produced before.

     Scoped to configurations that fit, because a command that does not fit is
     not a command: both engines short-circuit to a "does not fit" comment
     before any flag is reached, so the property genuinely does not hold there
     and pretending otherwise would just pin the short-circuit. */
  const h = renderHarness();
  let single = 0, multi = 0, skipped = 0;
  for (const card of [GPU_TABLE['h100-80'], dualGCD])
    for (const count of [1, 2, 3, 4, 5, 8, 9, 12, 16, 17, 24, 64, 100, 128])
      for (const bpp of [2, 0.5]) {
        const state = asState(card, count, { params: 70, layers: 80, bytesPerParam: bpp });
        const computed = h.computeInference(state);
        if (!computed.fits) { skipped++; continue; }
        h.renderStrategyBadges(state, computed);
        const badge = h.out['strategy-badges'] || '';
        const cmd = h.buildVllmCommand(state, computed, 'm');
        const saysSingle = badge.includes('Single device');
        const asksForNone = !/--tensor-parallel-size|--data-parallel-size/.test(cmd);
        assert.strictEqual(saysSingle, asksForNone,
          `${count}x ${card.name} (${computed.deviceCount} devices, TP=${computed.tp} x DP=${computed.dp}) ` +
          `at ${bpp} B/param: badge ${saysSingle ? 'says' : 'does not say'} "Single device" while the ` +
          `command ${asksForNone ? 'asks for no parallelism' : 'asks for it'}:\n${cmd}`);
        saysSingle ? single++ : multi++;
      }
  // Or an iff that never saw both sides of itself would pass on one branch.
  assert.ok(single > 0 && multi > 0,
    `not discriminating: ${single} single-device and ${multi} multi-device fitting configs (${skipped} skipped)`);
});
test('every sentence about the divisor states the divisor that was used', () => {
  /* A number going wrong is caught by the pins above. A *sentence* going wrong
     is not, and this tool's whole claim on a reader is that it shows its work:
     "Per-device VRAM above assumes weights sharded across all 12 devices" was a
     statement about the arithmetic, and the commit that started dividing by tp
     turned it false while every numeric assertion in this file still passed.
     That is why this test exists and why it went red on that commit rather than
     after it. So the prose is held to the arithmetic directly: each claimed
     divisor must equal weightsGB / perGPU.weights, the divisor actually used.

     Four kinds of claim, and conflating them hides exactly what the sentences
     exist to disclose:
       divisor     — what the per-device weights figure was divided by. tp for a
                     dense model, the device count for an MoE.
       tp          — what the emitted command shards by.
       dp          — how many replicas the model is copied into.
       deviceCount — what the KV cache was divided by, which is every device
                     whatever the model does, because DP partitions requests.
     For a dense model above 8 devices, divisor and deviceCount are different
     numbers on purpose; for an MoE they are the same one, on purpose. */
  const h = renderHarness();
  const CLAIMS = [
    // what the per-device weights figure was divided by
    ['weights-sharded tile', /Weights sharded<\/span><br><b>(\d+)-way<\/b>/g, 'divisor'],
    ['interconnect line', /sharded (\d+)-way, each device holds 1\/(\d+) of model/g, 'divisor'],
    ['moe weights tile', /Weights per device<\/span>[\s\S]{0,160}?divided by all (\d+) devices/g, 'divisor'],
    ['moe banner divisor', /Per-device weights here is divided by all (\d+) devices/g, 'divisor'],
    ['moe summary divisor', /weights per device divided by all (\d+) devices/g, 'divisor'],
    // what the emitted command shards by
    ['dense banner divisor', /Weights and activations above are divided by (\d+), the sharding the command performs/g, 'tp'],
    ['dense summary divisor', /weights and activations divided by (\d+), the sharding the command performs/g, 'tp'],
    ['moe aside on the dense rule', /Dense models (?:on this page )?divide by (\d+)/g, 'tp'],
    ['moe aside on attention', /shard only (\d+) ways/g, 'tp'],
    ['parallelism tile', /Parallelism<\/span><br><b>TP=(\d+)/g, 'tp'],
    ['interconnect TP group', /all-reduce inside each (\d+)-device TP group/g, 'tp'],
    // how many copies of the model exist
    ['replica note on the tile', /inside each of (\d+) replicas/g, 'dp'],
    ['dense summary replicas', /a full copy in each of the (\d+) data-parallel groups/g, 'dp'],
    ['interconnect replica count', /TP group, (\d+) replicas across the fabric/g, 'dp'],
    // what the KV cache was divided by — every device, model sharding aside
    ['dense banner KV divisor', /KV cache is divided by all (\d+), because data parallelism/g, 'deviceCount'],
    ['dense summary KV divisor', /KV cache divided by all (\d+) devices/g, 'deviceCount'],
  ];
  const seen = new Map(CLAIMS.map(([name]) => [name, 0]));
  for (const card of [GPU_TABLE['h100-80'], dualGCD])
    for (const count of [3, 5, 6, 8, 9, 10, 12, 16, 20, 24, 40, 64, 100, 128])
      // Both regimes. A dense-only sweep would let every MoE sentence go
      // unread, and the MoE sentences are the ones describing a divisor the
      // engine deliberately did not correct.
      for (const activePercent of [100, 5]) {
        const st = asState(card, count, { params: 70, layers: 80, activePercent });
        const c = h.computeInference(st);
        h.renderGPUCards(st, c);
        const text = (h.out['gpu-cards'] || '') + '\n' + h.exportSummary(st, c);
        // Derived from the output, not from deviceCount and not from tp: if the
        // engine changes what it divides by, this moves with it, and the test
        // keeps comparing the prose against the arithmetic rather than against
        // an assumption of its own.
        const divisor = c.weightsGB / c.perGPU.weights;
        const targets = { divisor, tp: c.tp, dp: c.dp, deviceCount: c.deviceCount };
        for (const [name, re, against] of CLAIMS) {
          const want = targets[against];
          for (const m of text.matchAll(re)) {
            seen.set(name, seen.get(name) + 1);
            for (const claimed of m.slice(1).map(Number))
              assert.ok(Math.abs(claimed - want) <= Math.abs(want) * 1e-9,
                `${count}x ${card.name} (${c.deviceCount} devices, TP=${c.tp} x DP=${c.dp}, ` +
                `${c.isMoE ? 'MoE' : 'dense'}): the ${name} says ${claimed}, but ${against} ` +
                `is ${want} — "${m[0].slice(0, 120)}"`);
          }
        }
      }
  // A regex that matches nothing asserts nothing, and rewording a sentence is
  // exactly how this test would stop looking without anyone noticing.
  const silent = [...seen].filter(([, n]) => n === 0).map(([name]) => name);
  assert.ok(!silent.length, `these claims were never found in any rendered output: ${silent.join(', ')}`);
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
test('the copied summary adds up, and says "cluster" only where that means something', () => {
  /* The export is the artifact that leaves the screen it was computed on, so it
     carries the same breakdown as the metrics row and has to hold the same
     relation: the three component lines sum to the figure printed beside them.

     And it must not grow a line in the regime this commit holds unchanged. A
     "Total (cluster)" line at one copy repeats the sum of the three lines above
     it and tells a reader nothing — master's export had no such line, and
     emitting one everywhere changed a document at every device count while the
     numbers in it stayed put. That went unnoticed because nothing pinned the
     export's shape, only its numbers. So the line is required to appear exactly
     where replication exists and nowhere else. */
  const h = renderHarness();
  const grain = (v) => (v >= 100 ? 0.5 : v >= 10 ? 0.05 : 0.005);
  let replicated = 0, single = 0;
  for (const count of [1, 2, 4, 8, 9, 12, 16, 64, 128])
    for (const activePercent of [100, 5])
      for (const card of [GPU_TABLE['h100-80'], dualGCD]) {
        const st = asState(card, count, { params: 70, layers: 80, activePercent });
        const c = h.computeInference(st);
        const full = h.exportSummary(st, c);
        /* Anchored inside the breakdown section: the Model section also has a
           "- KV cache:" line carrying a dtype rather than a size, and an
           unanchored match reads that one and finds no figure at all. */
        const text = full.slice(full.indexOf('## VRAM breakdown'), full.indexOf('## Capacity'));
        assert.ok(text, `no VRAM breakdown section in the export: ${full.slice(0, 200)}`);
        const line = (label) => {
          const m = text.match(new RegExp(`- ${label}[^:]*: ([\\d.]+) GiB`));
          return m ? Number(m[1]) : null;
        };
        const w = line('Weights'), kv = line('KV cache'), ao = line('Act \\+ overhead');
        assert.ok(w !== null && kv !== null && ao !== null,
          `export is missing a breakdown line: ${text.slice(0, 300)}`);
        const total = line('Total \\(cluster\\)');
        if (c.modelCopies > 1) {
          replicated++;
          assert.ok(total !== null,
            `${c.deviceCount} devices holds ${c.modelCopies} copies but the export names no ` +
            `cluster total: ${text.slice(0, 300)}`);
          const tol = grain(w) + grain(kv) + grain(ao) + grain(total);
          assert.ok(Math.abs(w + kv + ao - total) <= tol,
            `export breakdown does not sum: ${w} + ${kv} + ${ao} = ${w + kv + ao} against ${total}`);
          assert.ok(text.includes(`- Weights (${c.modelCopies} copies):`),
            `the export must say how many copies it is counting: ${text.slice(0, 300)}`);
        } else {
          single++;
          assert.strictEqual(total, null,
            `one copy, but the export prints a cluster total that just repeats the sum ` +
            `of the lines above it: ${text.slice(0, 300)}`);
          assert.ok(text.includes('- Weights: '),
            `at one copy the weights line must be unqualified: ${text.slice(0, 300)}`);
        }
      }
  assert.ok(replicated > 0 && single > 0,
    `only reached one regime: ${replicated} replicated, ${single} single-copy`);
});
test('no help text offers pipeline parallelism as something this tool does', () => {
  /* The GPU-count tooltip said weights and KV are sharded "via tensor or
     pipeline parallelism" for as long as the tool has existed. Neither engine
     has ever emitted --pipeline-parallel-size, so a reader who believed it was
     being told the wrong thing about their own command at every device count.

     Pinned over the whole set of user-visible help strings rather than that one
     tooltip: any mention of pipeline parallelism has to be a denial. That is
     what "not modelled" means, and it is the one claim in this family a reader
     can act on by choosing a topology the tool cannot cost. */
  const tips = new Set([...html.matchAll(/data-tip="([^"]*)"/g)].map(m => m[1])
    .concat([...html.matchAll(/aria-label="([^"]*)"/g)].map(m => m[1])));
  assert.ok(tips.size > 20, `only ${tips.size} help strings found — the scan has stopped reaching them`);
  for (const t of tips) {
    if (!/pipeline[- ]parallel/i.test(t)) continue;
    assert.match(t, /\bnever\b|\bnot\b|\bno\b/i,
      `help text offers pipeline parallelism without saying the tool does not use it: "${t}"`);
  }
  /* And the same string has to be right about the class this commit went out of
     its way to protect: a data-parallel replica holds a whole copy of a dense
     model, but an MoE spreads its routed experts across every rank — that is
     the entire reason the MoE divisor was held back. A sentence that describes
     only the dense case is wrong for seven of the sixteen presets. */
  for (const t of tips) {
    if (!/data-parallel replicas/i.test(t)) continue;
    assert.match(t, /mixture-of-experts|MoE/i,
      `help text describes what a data-parallel replica holds without the MoE case, ` +
      `which is the one the divisor was held back for: "${t}"`);
  }
});
test('every view that prints a throughput figure above one domain carries the caveat', () => {
  /* The device slider runs to 128 rather than stopping at one NVLink domain,
     and the owner's ruling was that the per-field label carries the honesty the
     cap would otherwise have carried. A view that prints the number and omits
     the label defeats the ruling rather than simplifying for its audience.

     The first version of this test named three surfaces, and a fourth —
     renderComparisons, which prints "~194 / 194 tok/s" for a saved snapshot at
     any device count — passed by not being on the list. An enumerated list of
     call sites is exactly what this suite is supposed to be immune to, and the
     old comment claiming it stopped that was written above a test that could
     not. So nothing is enumerated now: every renderer the harness exposes is
     invoked, with its arguments read from its own declared parameter names in
     index.html, and every rendered element whose text contains a throughput
     figure is required to carry the label. Surface five fails automatically. */
  const h = renderHarness();
  const THROUGHPUT = /tok\/s|tokens\/sec/;
  const LABEL = /unmeasured above 2 devices/i;
  // Declared parameter names, from the source, so a renderer is called the way
  // it is written rather than the way this test guesses.
  const params = {};
  for (const m of html.matchAll(/function (render\w+)\(([^)]*)\)/g))
    params[m[1]] = m[2].split(',').map(x => x.trim().split(/[=\s]/)[0]).filter(Boolean);
  const renderers = Object.keys(h).filter(k => /^render/.test(k) && typeof h[k] === 'function');
  assert.ok(renderers.length >= 8,
    `only ${renderers.length} renderers reachable — the harness has stopped exposing them`);
  for (const name of renderers)
    assert.ok(params[name], `${name} is exposed but not declared in index.html`);

  let above = 0, within = 0, surfaces = new Set();
  for (const count of [2, 4, 8, 9, 11, 12, 16, 64, 128])
    for (const card of [GPU_TABLE['h100-80'], dualGCD])
      for (const nv of [true, false]) {
        const st = asState(card, count, { params: 8, layers: 32, hasNVLink: nv });
        const c = h.computeInference(st);
        Object.keys(h.out).forEach(k => delete h.out[k]);
        // Snapshots first: renderComparisons draws saved cards, each carrying
        // its own computed result, and renders a placeholder when there are none.
        h.pushSnapshot(st, c);
        for (const name of renderers) {
          const args = params[name].map(pn => (pn === 'computed' ? c : pn === 'state' ? st : undefined));
          try { h[name](...args); } catch (e) {
            assert.fail(`${name}(${params[name].join(', ')}) threw: ${e.message}`);
          }
        }
        for (const [id, text] of Object.entries(h.out)) {
          if (!THROUGHPUT.test(text)) continue;
          surfaces.add(id);
          const labelled = LABEL.test(text);
          if (c.dp > 1) {
            above++;
            assert.ok(labelled,
              `${id} at ${c.deviceCount} devices (TP=${c.tp} x DP=${c.dp}) prints a throughput ` +
              `figure with no unmeasured-heuristic label: ${text.slice(0, 240)}`);
          } else {
            /* Two to eight devices is deliberately unlabelled, and that is a
               known gap rather than an oversight: the same curve applies there
               and is just as unmeasured, but labelling it would move renders in
               the regime this commit proves unchanged. MODEL.md says so in as
               many words. Pinned in this direction too, so the gap cannot be
               closed by accident and then reported as unchanged. */
            within++;
            assert.ok(!labelled,
              `${id} at ${c.deviceCount} devices gained a label in the regime this commit ` +
              `holds unchanged — intended, but it has to be a deliberate change`);
          }
        }
      }
  assert.ok(above > 0 && within > 0,
    `only reached one regime: ${above} above one domain, ${within} within`);
  // The surfaces are discovered, not declared, so this only guards against the
  // discovery silently collapsing to one.
  assert.ok(surfaces.size >= 3,
    `only ${surfaces.size} surfaces print throughput at all (${[...surfaces].join(', ')}) — ` +
    `the sweep has stopped reaching them`);
});

test('every view that prints a compute figure says when FP8 has no tensor cores', () => {
  /* Same discovery approach as the sweep above, and for the same reason: an
     enumerated list of call sites is what this suite is supposed to be immune
     to. FP8 on Ampere is a real configuration — vLLM's fp8_marlin runs there and
     the weights genuinely halve — so the tool offers it. What it must not do is
     let the compute figures read as FP8 ones when the GEMM runs at FP16.

     Two-sided, like the fabric sweep: the label must be absent on a card that
     does have FP8 tensor cores, or it would be noise on the common case. */
  const COMPUTE = /tok\/s|tokens\/sec/;
  const LABEL = /no FP8 tensor cores/i;
  const params = {};
  for (const m of html.matchAll(/function (render\w+)\(([^)]*)\)/g))
    params[m[1]] = m[2].split(',').map(x => x.trim().split(/[=\s]/)[0]).filter(Boolean);

  let flagged = 0, clean = 0;
  for (const slug of ['a100-80', 'a100-40', 't4-16', 'h100-80', 'l40s-48', 'b200-192']) {
    /* A fresh harness per card: savedSnapshots accumulates, so one shared
       harness would still be rendering the A100 cards when it reaches the H100
       and the negative half of this test would fail on its own leftovers. */
    const h = renderHarness();
    const renderers = Object.keys(h).filter(k => /^render/.test(k) && typeof h[k] === 'function');
    const card = GPU_TABLE[slug];
    const st = asState(card, 1, { params: 8, layers: 32, bytesPerParam: 1, quantMethod: 'fp8' });
    const c = h.computeInference(st);
    assert.strictEqual(c.fp8NoTensorCores, !card.caps.fp8,
      `${slug}: caps.fp8 is ${card.caps.fp8} but fp8NoTensorCores is ${c.fp8NoTensorCores}`);
    Object.keys(h.out).forEach(k => delete h.out[k]);
    h.pushSnapshot(st, c);
    for (const name of renderers) {
      const args = params[name].map(pn => (pn === 'computed' ? c : pn === 'state' ? st : undefined));
      try { h[name](...args); } catch (e) {
        assert.fail(`${name}(${params[name].join(', ')}) threw: ${e.message}`);
      }
    }
    for (const [id, text] of Object.entries(h.out)) {
      if (!COMPUTE.test(text)) continue;
      const labelled = LABEL.test(text);
      if (card.caps.fp8) {
        clean++;
        assert.ok(!labelled,
          `${id} labels ${slug} as lacking FP8 tensor cores, which it has: ${text.slice(0, 200)}`);
      }
    }
    if (!card.caps.fp8) {
      /* At least one surface has to carry it, and the aggregate ceiling is the
         one that must: it is the figure the multiplier moves. Asserting "some
         surface" rather than "every surface" is deliberate — single-stream is
         bandwidth-bound and the FP8 win there is real, so a label on it would
         be wrong, and requiring every throughput-bearing element would demand
         exactly that. */
      /* Not "some surface says so" — that passed with the label deleted from
         the throughput panel, because the executive view and the comparison
         card still carried their own. The requirement is derived instead: any
         surface that prints the aggregate figure is printing a number the
         compute ratio moved, so that surface has to carry the label. */
      /* Both figures the ratio multiplies, not just the aggregate. Deriving the
         requirement from the aggregate alone left the TTFT tile's label
         deletable in silence — and TTFT is the figure with no bandwidth cap,
         so it is the one the ratio moves unconditionally. MODEL.md promises
         the UI says so beside *both*. */
      const targets = [
        ['aggregate', formatTokensLike(c.aggregateTokS)],
        ['TTFT', `${c.ttftMs} ms`],
      ];
      /* Per card, not per element. renderThroughput writes the aggregate tile
         and the TTFT tile into the SAME element, so an element-level check is
         satisfied by either one carrying the label — which is exactly how
         deleting the TTFT caveat stayed green. Split on the card boundary the
         renderers actually use and require the label inside the fragment that
         prints the figure. */
      for (const [what, needle] of targets) {
        let found = 0;
        for (const [id, text] of Object.entries(h.out)) {
          for (const frag of text.split(/<div class="(?:reverse-card|compare-card|exec-row)"/)) {
            if (!frag.includes(needle)) continue;
            found++;
            assert.ok(LABEL.test(frag),
              `${id} prints the ${what} figure for ${slug}, which has no FP8 tensor cores, ` +
              `in a card that does not say so: ${frag.slice(0, 220)}`);
          }
        }
        assert.ok(found > 0,
          `no card printed the ${what} figure (${needle}) for ${slug} — the sweep is not reaching it`);
      }
      flagged++;
    }
  }
  assert.ok(flagged >= 3 && clean > 0,
    `reached ${flagged} cards without FP8 and ${clean} labelled-clean renders — need both regimes`);
});

console.log('\nCost provenance: a named source, or "not recorded" said plainly, never a guess');
test('priceSourceLabel formats provider, SKU, region and date — a fixed expectation, not self-referential', () => {
  /* The sweep below (and its Python twin) computes its own "expected" string
     by calling this same function, which proves the renderers agree with
     priceSourceLabel() but cannot catch a bug inside priceSourceLabel()
     itself — a version that quietly dropped the date would still match its
     own output. This is the check that cannot pass that way: the expected
     string is a literal, typed by hand once, not derived from the function
     under test. */
  const priceSourceLabelDecl = html.match(/^function priceSourceLabel\(state, tier\) \{[\s\S]*?\n\}$/m);
  assert.ok(priceSourceLabelDecl, 'priceSourceLabel() not found in index.html');
  const providerNamesDecl = html.match(/^const PROVIDER_NAMES = \{[\s\S]*?\};$/m);
  assert.ok(providerNamesDecl, 'PROVIDER_NAMES not found in index.html');
  const noPriceDecl = html.match(/^const NO_PRICE = .+;$/m);
  const tierFieldDecl = html.match(/^const TIER_COST_FIELD = \{.*\};$/m);
  assert.ok(noPriceDecl && tierFieldDecl, 'NO_PRICE / TIER_COST_FIELD not found in index.html');
  const priceSourceLabel = new Function(
    `${providerNamesDecl[0]}\n${noPriceDecl[0]}\n${tierFieldDecl[0]}\n${priceSourceLabelDecl[0]}; return priceSourceLabel;`)();

  const state = { priceSource: { hyper: { provider: 'azure', sku: 'Standard_ND96isr_H100_v5',
                                           region: 'eastus', date: '2026-09-16' } } };
  assert.strictEqual(priceSourceLabel(state, 'hyper'),
    'Azure · Standard_ND96isr_H100_v5 · eastus · read 2026-09-16');
  // Every known provider id renders as a proper display name, not the raw
  // lowercase source id price_check.py's SOURCE_MAP uses internally.
  const providers = { azure: 'Azure', aws: 'AWS', lambda: 'Lambda',
                       coreweave: 'CoreWeave', vast: 'Vast.ai' };
  for (const [id, name] of Object.entries(providers)) {
    const st = { priceSource: { spot: { provider: id, sku: 'X', region: 'global', date: '2026-01-01' } } };
    assert.strictEqual(priceSourceLabel(st, 'spot'), `${name} · X · global · read 2026-01-01`,
      `provider id ${id} did not render as ${name}`);
  }
  // An unrecognised provider id falls back to itself rather than throwing or
  // silently dropping the field — a real, if unlikely, SOURCE_MAP addition.
  const unknown = { priceSource: { spec: { provider: 'newvendor', sku: 'Y', region: 'r', date: 'd' } } };
  assert.strictEqual(priceSourceLabel(unknown, 'spec'), 'newvendor · Y · r · read d');
  // Absent states, all rendering as exactly "not recorded" — no partial label,
  // no undefined leaking through.
  assert.strictEqual(priceSourceLabel({ priceSource: undefined }, 'hyper'), 'not recorded');
  assert.strictEqual(priceSourceLabel({ priceSource: {} }, 'hyper'), 'not recorded');
  assert.strictEqual(priceSourceLabel({ priceSource: { spec: { provider: 'x', sku: 'y', region: 'z', date: 'd' } } }, 'hyper'),
    'not recorded', 'a sourced spec tier must not leak into a hyper lookup');
  // A tier the catalog records as null has no price, so it has no source to
  // name either — even with a source attached, which the catalog tests refuse.
  assert.strictEqual(priceSourceLabel({ gpuHyperCost: null, priceSource: undefined }, 'hyper'),
    'no confirmed hourly price');
  assert.strictEqual(priceSourceLabel({ gpuSpotCost: null, priceSource: { spot: state.priceSource.hyper } }, 'spot'),
    'no confirmed hourly price', 'a null tier rendered the source attached to it');
  assert.strictEqual(priceSourceLabel({ gpuSpecCost: 2.5, priceSource: undefined }, 'spec'), 'not recorded');
  // A price read by hand off the provider's page: named and dated, and saying so.
  const handRec = { provider: 'RunPod', sku: 'MI300X (Secure Cloud)', region: 'global', date: '2026-09-23', price: 2.39, url: 'https://www.runpod.io/gpu-models/mi300x' };
  assert.strictEqual(priceSourceLabel({ gpuSpecCost: 2.39, priceRecord: { spec: handRec } }, 'spec'),
    'RunPod · MI300X (Secure Cloud) · global · recorded by hand 2026-09-23, not re-checked weekly');
  assert.strictEqual(priceSourceLabel({ gpuSpecCost: 2.39, priceRecord: { spot: handRec } }, 'spec'), 'not recorded',
    'a hand record on another tier leaked into this one');
  assert.strictEqual(priceSourceLabel({ priceSource: state.priceSource, priceRecord: { hyper: handRec } }, 'hyper'),
    'Azure · Standard_ND96isr_H100_v5 · eastus · read 2026-09-16', 'a hand record displaced an automated reading');
  assert.strictEqual(priceSourceLabel({ gpuSpotCost: null, priceRecord: { spot: handRec } }, 'spot'),
    'no confirmed hourly price', 'a null tier rendered the hand record attached to it');
  /* Looked up by the tier asked about. A lookup hard-wired to .spec passed
     while every real record sat on spec (cold check, round 1): every tier the
     record is on, asked about every tier, each of them priced and unsourced. */
  for (const on of ['hyper', 'spec', 'spot'])
    for (const asked of ['hyper', 'spec', 'spot'])
      assert.strictEqual(
        priceSourceLabel({ gpuHyperCost: 1, gpuSpecCost: 1, gpuSpotCost: 1, priceRecord: { [on]: handRec } }, asked),
        asked === on ? 'RunPod · MI300X (Secure Cloud) · global · recorded by hand 2026-09-23, not re-checked weekly'
                     : 'not recorded',
        `a record on ${on}, asked about ${asked}`);
});

/* The catalog's own note for a tier, in the one form both engines print it: the
   reason as written, then "Checked <date>." Only where the tier has neither an
   automated reading nor a hand record, since either of those is the answer
   then. Written out here rather than read from the page: this is the contract
   the page is held to. */
const catalogNoteText = (st, tier) => {
  const note = st.priceNote && st.priceNote[tier];
  const other = (st.priceSource && st.priceSource[tier]) || (st.priceRecord && st.priceRecord[tier]);
  return note && !other ? `${note.reason} Checked ${note.checked}.` : '';
};

test('every cost surface names a source or says "not recorded", discovered not enumerated', () => {
  /* Same discovery approach as the two sweeps above: every renderer the
     harness exposes is invoked, and any rendered element whose text carries
     a cost figure ($X.XX) is a "cost surface" for this test, found by what
     it prints rather than by a list of element ids fix/cost-provenance's
     author happened to think of. The old bug this replaces (a composite
     "AWS, GCP, Azure on-demand" sub-label naming three providers for one
     provider's number) lived on exactly the kind of surface an enumerated
     list would miss on its next addition.

     priceSourceLabel() is extracted from source, the same way formatGB() and
     bandwidthLabel() are read above, so the expected string is computed by
     the page's own code and cannot silently drift from what the renderers
     actually call. */
  const priceSourceLabelDecl = html.match(/^function priceSourceLabel\(state, tier\) \{[\s\S]*?\n\}$/m);
  assert.ok(priceSourceLabelDecl, 'priceSourceLabel() not found in index.html');
  const providerNamesDecl = html.match(/^const PROVIDER_NAMES = \{[\s\S]*?\};$/m);
  assert.ok(providerNamesDecl, 'PROVIDER_NAMES not found in index.html');
  const noPriceDecl = html.match(/^const NO_PRICE = .+;$/m);
  const tierFieldDecl = html.match(/^const TIER_COST_FIELD = \{.*\};$/m);
  assert.ok(noPriceDecl && tierFieldDecl, 'NO_PRICE / TIER_COST_FIELD not found in index.html');
  const priceSourceLabel = new Function(
    `${providerNamesDecl[0]}\n${noPriceDecl[0]}\n${tierFieldDecl[0]}\n${priceSourceLabelDecl[0]}; return priceSourceLabel;`)();
  assert.strictEqual(priceSourceLabel({ priceSource: undefined }, 'hyper'), 'not recorded');

  const params = {};
  for (const m of html.matchAll(/function (render\w+)\(([^)]*)\)/g))
    params[m[1]] = m[2].split(',').map(x => x.trim().split(/[=\s]/)[0]).filter(Boolean);
  // Renderer names are the same across every harness instance (they come from
  // the same source slice), so one throwaway instance is enough to discover them.
  const renderers = Object.keys(renderHarness()).filter(
    k => /^render/.test(k) && typeof renderHarness()[k] === 'function');

  const OLD_COMPOSITES = [/AWS,\s*GCP,\s*Azure on-demand/i, /Lambda,\s*CoreWeave,\s*RunPod/i,
                          /Vast\.ai,\s*spot instances/i,
                          // Cold-check finding 1f: appending " (AWS, GCP, Azure)" after an
                          // otherwise-correct sourced label survived, because none of the
                          // three exact phrases above matches a shorter list in a different
                          // shape. General instead of exact: two or more provider-ish names
                          // joined by a comma or slash, wherever they occur — a real sourced
                          // label never joins two provider names this way (its own two
                          // tiers, if both sourced, are separated by the rest of a sentence,
                          // not a bare comma or slash), so this cannot true-positive on a
                          // correct render, only on a reintroduced list.
                          /\b(?:AWS|GCP|Azure|Lambda|CoreWeave|RunPod|Vast\.ai)(?:\s*[,\/]\s*(?:AWS|GCP|Azure|Lambda|CoreWeave|RunPod|Vast\.ai)){1,}/];
  const NOTES_PRICE_SENTENCES = [
    'GPU prices are mid-2026 per-board/hr figures across 3 tiers — see the cost table above for ' +
    "each tier's source, or \"not recorded\" where it has no confirmed source.",
    // On a card with a null tier, the one sentence that explains the table's wording for it.
    '"No confirmed hourly price" marks a tier for which no provider\'s own page prices this card by the hour.',
    // On a card with a lead, the one sentence that says what a lead is.
    'A lead is an hourly price found but not confirmed: it is shown with why, and no figure on this page uses it.'];
  const PROVIDER_NAMES_LIST = ['Azure', 'AWS', 'Lambda', 'CoreWeave', 'Vast.ai'];
  // Cold-check finding: renderExecutiveSummary's "Monthly cost range" rounds
  // to whole dollars ("$657", never "$657.00"), so a cents-only pattern
  // never discovered that surface at all — not scoped out on purpose, the
  // sweep just never looked at it, and three cold-check sabotages that
  // targeted only the exec summary (a composite name beside its source, a
  // dropped region, a composite fallback) went uncaught because of it. The
  // decimal point is now optional.
  const COST = /\$[\d,]+(?:\.\d{2})?/;

  // h100-80 carries mixed provenance today (hyper+spec sourced, spot not) —
  // real catalog shape, not invented. rtx5090-32 has no automatable source on
  // any tier, so every one of its tiers is "not recorded". A synthetic row
  // with every tier sourced reaches the all-sourced case a real one does not
  // give us yet.
  const sourced = { provider: 'lambda', sku: 'TEST PLAN', region: 'global', date: '2026-09-22' };
  const cases = [
    ['mixed (real h100-80)', GPU_TABLE['h100-80'], 'mixed'],
    ['none recorded (real rtx5090-32)', GPU_TABLE['rtx5090-32'], 'none'],
    ['all sourced (synthetic)',
     { ...GPU_TABLE['h100-80'], priceSource: { hyper: sourced, spec: sourced, spot: sourced } }, 'all'],
    // h100-80's unsourced spot tier, recorded by hand: each surface must print
    // that label under that tier, exactly as priceSourceLabel() renders it.
    ['hand-recorded spot (synthetic)', { ...GPU_TABLE['h100-80'], priceRecord: { spot: { provider: 'RunPod', sku: 'MI300X (Secure Cloud)', region: 'global', date: '2026-09-23', price: 2.39, url: 'https://www.runpod.io/gpu-models/mi300x' } } }, 'mixed'],
  ];

  const surfacesSeen = new Set();
  let mixedSourcedHit = 0, mixedNotRecordedHit = 0;
  for (const [label, card, shape] of cases) {
    // A fresh harness per card, like the FP8 sweep above: savedSnapshots
    // accumulates on one shared harness, which would leave an earlier card's
    // comparison row still rendering when this reaches the next card.
    const h = renderHarness();
    const st = asState(card, 1, { params: 8, layers: 32 });
    const c = h.computeInference(st);
    h.pushSnapshot(st, c);
    for (const name of renderers) {
      const args = params[name].map(pn => (pn === 'computed' ? c : pn === 'state' ? st : undefined));
      try { h[name](...args); } catch (e) { assert.fail(`${label}: ${name}(...) threw: ${e.message}`); }
    }
    const texts = { ...h.out, '(copied report)': h.exportSummary(st, c) };
    /* A tier's note is the catalog's own sentence about where its figure came
       from, and it names providers and says "price" by design. It has its own
       test below: verbatim, under its own tier, and nowhere else. Here it is
       taken out, exact text only, so every rule in this sweep still holds for
       the rest of the surface. A note changed by one character is not taken
       out, and meets the rules like anything else. */
    for (const tier of ['hyper', 'spec', 'spot']) {
      const note = catalogNoteText(st, tier);
      if (note) for (const id of Object.keys(texts)) texts[id] = texts[id].split(note).join(' ');
    }
    let sawCostSurface = false;
    for (const [id, text] of Object.entries(texts)) {
      if (!COST.test(text)) continue;
      sawCostSurface = true;
      surfacesSeen.add(id);

      for (const re of OLD_COMPOSITES)
        assert.ok(!re.test(text),
          `${label}/${id}: still names the old composite provider list (${text.match(re)}): ` +
          text.slice(0, 300));
      assert.ok(!/\bundefined\b|\bnull\b|\bNaN\b/.test(text),
        `${label}/${id}: prints a raw undefined/null/NaN instead of a source or "not recorded": ` +
        text.slice(0, 300));

      if (shape === 'none') {
        assert.ok(text.includes('not recorded'),
          `${label}/${id}: no tier is sourced, but no "not recorded" appears: ${text.slice(0, 300)}`);
        for (const p of PROVIDER_NAMES_LIST)
          assert.ok(!text.includes(p),
            `${label}/${id}: names provider "${p}" with nothing recorded to back it: ${text.slice(0, 300)}`);
      } else if (shape === 'all') {
        assert.ok(!text.includes('not recorded'),
          `${label}/${id}: every tier is sourced, but "not recorded" still appears: ${text.slice(0, 300)}`);
        assert.ok(text.includes('Lambda'),
          `${label}/${id}: every tier is sourced (lambda), but no source name appears: ${text.slice(0, 300)}`);
      } else {
        // Not "if the full expected string is found, count it, otherwise say
        // nothing" — that pattern is what let a cold-check sabotage through
        // on the PDF side: strip just the date off comparison-output's hyper
        // line (bypassing priceSourceLabel() itself, so the direct unit test
        // above cannot see it either) and the loop silently skipped hyper
        // while still counting spec/spot's correct labels elsewhere in the
        // same string, or in a different surface entirely. `marker` is a
        // short, reliable signal that THIS surface is attempting to show
        // THIS tier at all — the provider name for a sourced tier, or the
        // literal "not recorded" text otherwise. A surface that never
        // mentions a tier (renderExecutiveSummary never shows spec) still
        // correctly skips it; a surface that shows the marker but not the
        // rest of the expected label now fails instead of going uncounted.
        /* Anchored on the TIER's own name, and asserted on the segment that
           follows it — not on the provider name, and not by membership.

           Round-2 cold check took the previous shape apart fourteen ways. The
           marker was the provider name, so a label deleted outright left no
           marker and `continue` skipped the tier: an empty field where the
           source must be, uncounted. And `text.includes(expected)` is true of
           any text that merely CONTAINS the label, so hyper's and spec's
           labels could be swapped between rows (Lambda's SKU printed under
           Azure's price), or a composite appended after a correct label, and
           the assertion still passed.

           A tier name is present whether or not its label rendered, so the
           skip is gone. The segment runs to the next tier's marker, so a label
           in the wrong row lands in the wrong segment. And what is left of the
           segment once the expected label is removed may carry prices and
           lowercase words, but no capital letter and no letter joined to a
           letter by a comma, slash or ampersand — which is what every invented
           provider and every re-appended composite looks like, without this
           test needing to know a single provider's name. */
        const TIER_MARK = { hyper: /Hyperscaler|Hyper:/g, spec: /Specialized|Spec:/g,
                            spot: /Spot \/ marketplace|Spot:/g };
        /* Tags out first: these surfaces are HTML, and a style attribute is
           not something the page says. Replaced by a space so nothing joins. */
        const plain = text.replace(/<[^>]*>/g, ' ');
        const bounds = [];
        for (const [tier, re] of Object.entries(TIER_MARK))
          for (const m of plain.matchAll(re)) bounds.push({ tier, at: m.index, end: m.index + m[0].length });
        bounds.sort((a, b) => a.at - b.at);
        for (let i = 0; i < bounds.length; i++) {
          const { tier, end } = bounds[i];
          const segment = plain.slice(end, i + 1 < bounds.length ? bounds[i + 1].at : undefined);
          const expected = priceSourceLabel(st, tier);
          assert.ok(segment.includes(expected),
            `${label}/${id}/${tier}: the text after this tier's own name is not its source. ` +
            `expected ${JSON.stringify(expected)}, segment ${JSON.stringify(segment.slice(0, 200))}`);
          /* Only the text that sits between this tier's name and its price:
             what precedes the label, and what follows it up to the next price
             figure. A surface's trailing disclaimer comes after every price and
             belongs to no tier, so bounding at the next `$` keeps it out of a
             tier's residue without this test having to know it is there. */
          const at = segment.indexOf(expected);
          const before = segment.slice(0, at);
          const after = segment.slice(at + expected.length);
          // ...or the next element boundary, which a stripped tag leaves as a run
          // of spaces. The exec summary puts its next row straight after the
          // label with no price between them.
          const residue = before + ' ' + after.split(/\s{2,}|\n|\$/)[0];
          assert.ok(!/[A-Z]/.test(residue),
            `${label}/${id}/${tier}: names something beside its source — ` +
            `${JSON.stringify(residue.slice(0, 160))}`);
          assert.ok(!/[A-Za-z]\s*[,/&]\s*[A-Za-z]/.test(residue),
            `${label}/${id}/${tier}: a second provider is joined onto its source — ` +
            `${JSON.stringify(residue.slice(0, 160))}`);
          if (expected === 'not recorded') mixedNotRecordedHit++; else mixedSourcedHit++;
        }
      }
    }
    assert.ok(sawCostSurface, `${label}: no surface printed a cost figure at all — the sweep found nothing`);

    /* The notes carry no cost figure, so the per-surface loop above never sees
       them — and a sentence there is as much a claim about where a price came
       from as a sub-label is. Round 2 added `notes += 'Spot prices are from
       Vast.ai. '` to every card, including one with nothing recorded, and only
       the golden noticed.

       One sentence in the notes mentions price, and what it may say is a
       contract, so it is a literal. Anything else naming a source belongs in
       the cost table, beside the price it describes. */
    const notesText = (texts['notes-output'] || '').replace(/<[^>]*>/g, ' ');
    const priceSentences = notesText.split(/(?<=\.)\s+/)
      .map(x => x.trim()).filter(x => /\bprices?\b/i.test(x));
    for (const sentence of priceSentences)
      assert.ok(NOTES_PRICE_SENTENCES.some(ok => sentence.startsWith(ok)),
        `${label}: the notes say something about price that is not the one sentence they may ` +
        `say — ${JSON.stringify(sentence.slice(0, 200))}`);
    // Cold-check finding: renderNotes()'s general disclaimer line ("GPU
    // prices are mid-2026 per-board/hr estimates...") named providers and
    // called every price an estimate, but the text has no dollar figure in
    // it (COST never matches it), so the per-surface loop above never
    // discovered it at all — not scoped out on purpose, just never reached.
    // This checks every renderer's output, cost figure or not, the same way
    // the PDF-side sweep now checks its whole story rather than only the
    // cost-bearing strings.
    const wholePage = Object.values(texts).join('\n');
    for (const re of OLD_COMPOSITES)
      assert.ok(!re.test(wholePage),
        `${label}: the composite provider list (${wholePage.match(re)}) appears somewhere on ` +
        'the page, even outside a cost figure\'s own text');
    assert.ok(!wholePage.includes('per-board/hr estimates'),
      `${label}: still calls GPU prices "estimates" somewhere on the page — some tiers are ` +
      'sourced, dated, attributed figures, not guesses');
  }
  assert.ok(mixedSourcedHit > 0 && mixedNotRecordedHit > 0,
    `the mixed real row (h100-80) did not exercise both states: sourced=${mixedSourcedHit} ` +
    `not-recorded=${mixedNotRecordedHit}`);
  assert.ok(surfacesSeen.size >= 3,
    `only ${surfacesSeen.size} surfaces print a cost figure at all (${[...surfacesSeen].join(', ')}) — ` +
    'the sweep has stopped reaching them');
});

test("every tier the catalog notes says why, under its own label, in the cost table and the copied report, and nowhere else", () => {
  /* Tier honesty: a figure with no source, and a tier with no figure, both say
     how they got there. The note sits under the tier's own label in the cost
     table, and on its own line under the tier's line in the copied report; the
     compact views (the comparison card, the executive summary) keep the short
     label, since the full reason is on the same page. Every catalog row, every
     tier. */
  const ROWS = { hyper: ['Hyperscaler', 'Hyperscaler'], spec: ['Specialized', 'Specialized'],
                 spot: ['Spot / marketplace', 'Spot'] };
  let shown = 0, silent = 0;
  for (const [key, gpu] of Object.entries(GPU_TABLE)) {
    const h = renderHarness();
    const st = asState(gpu, 1, { params: 8, layers: 32 });
    const c = h.computeInference(st);
    h.pushSnapshot(st, c);
    for (const name of Object.keys(h).filter(k => /^render/.test(k) && typeof h[k] === 'function')) {
      try { h[name](st, c); } catch (e) { /* renderers that take other arguments are covered elsewhere */ }
    }
    const report = h.exportSummary(st, c);
    const costRows = (h.out['cost-output'] || '').split('<tr').slice(1);
    for (const [tier, [rowName, lineLabel]] of Object.entries(ROWS)) {
      const note = catalogNoteText(st, tier);
      const row = costRows.find(r => r.includes(`<b>${rowName}</b>`));
      assert.ok(row, `${key}: the cost table has no ${rowName} row`);
      const lines = report.split('\n');
      const at = lines.findIndex(l => l.startsWith(`- ${lineLabel}: `));
      assert.ok(at >= 0, `${key}: the copied report has no ${lineLabel} line`);
      if (note) {
        shown++;
        assert.ok(row.includes(`</span><br><span style="font-size:11px;color:var(--text-muted)">${note}</span>`),
          `${key}/${tier}: the cost table does not carry the note under the tier's label — ${row.slice(0, 400)}`);
        assert.strictEqual(lines[at + 1], `  - ${note}`, `${key}/${tier}: the copied report's line under ${lineLabel}`);
        for (const [id, text] of Object.entries(h.out))
          if (id !== 'cost-output') assert.ok(!text.includes(note), `${key}/${tier}: the note also appears in ${id}`);
        assert.strictEqual(report.split(note).length - 1, 1, `${key}/${tier}: the copied report carries the note more than once`);
      } else {
        silent++;
        assert.ok(!row.includes('</span><br><span'), `${key}/${tier}: a tier with no note has a second line — ${row.slice(0, 300)}`);
        assert.ok(!(lines[at + 1] || '').startsWith('  - '), `${key}/${tier}: a tier with no note has a line under it in the copied report`);
      }
    }
  }
  assert.ok(shown >= 20 && silent > 0, `checked ${shown} noted and ${silent} un-noted tiers — this needs both`);
});

test('a tier with a reading or a hand record shows that, not a note that slipped in beside it', () => {
  /* The catalog tests refuse a note beside either, but the page is the last line:
     the reading or the record is the answer, and a stale note next to it would
     contradict it. Every tier in the catalog that has either, with a note slipped
     in: only a reading was tested once, and a note shown beside a hand record
     passed. */
  const note = { reason: 'A note that should not show.', checked: '2026-09-23' };
  const kinds = new Set();
  for (const [key, gpu] of Object.entries(GPU_TABLE)) for (const tier of ['hyper', 'spec', 'spot']) {
    const kind = (gpu.priceSource || {})[tier] ? 'reading' : (gpu.priceRecord || {})[tier] ? 'hand record' : null;
    if (!kind) continue;
    kinds.add(kind);
    const h = renderHarness();
    const st = asState({ ...gpu, priceNote: { ...(gpu.priceNote || {}), [tier]: note } }, 1, { params: 8, layers: 32 });
    const c = h.computeInference(st);
    h.renderCost(st, c);
    assert.ok(!(h.out['cost-output'] || '').includes(note.reason), `${key}/${tier}: the cost table shows a note beside a ${kind}`);
    assert.ok(!h.exportSummary(st, c).includes(note.reason), `${key}/${tier}: the copied report shows a note beside a ${kind}`);
  }
  assert.deepStrictEqual([...kinds].sort(), ['hand record', 'reading'], 'the catalog no longer has both kinds of answered tier');
});

/* A lead as the page prints it, written out here as the contract: in the cost table
   (with its links) and in the copied report (with its addresses). */
const leadHtml = (l) => `<br><span style="font-size:11px;color:var(--text-muted)">Lead, not used: ${l.provider} lists $${l.price.toFixed(2)}/hr (read ${l.date}). ${l.why} <a href="${l.url}" target="_blank" style="color:var(--accent-text)">${l.provider}'s page</a>${l.about ? ` · <a href="${l.about}" target="_blank" style="color:var(--accent-text)">about ${l.provider}</a>` : ''}</span>`;
const leadLine = (l) => `  - Lead, not used: ${l.provider} lists $${l.price.toFixed(2)}/hr (read ${l.date}, ${l.url}). ${l.why}` +
  (l.about ? ` About ${l.provider}: ${l.about}` : '') + '\n';
const LEAD_SENTENCE = 'A lead is an hourly price found but not confirmed: it is shown with why, and no figure on this page uses it. ';
const ROWS_WITH_LEADS = Object.entries(GPU_TABLE).filter(([, g]) => g.priceLead && Object.keys(g.priceLead).length);
/* A card with a lead on every tier, which the catalog doesn't have: its leads sit on
   spec and spot, so a surface that dropped the hyperscaler's leads passed. A card
   with no price in any tier, each lead at its own figure. */
const EVERY_TIER_LEADS = ['mi210-64 (a lead on every tier)', { ...GPU_TABLE['mi210-64'],
  priceLead: Object.fromEntries(['hyper', 'spec', 'spot'].map((tier, i) => [tier,
    [{ provider: `Lead${i}`, price: 1.51 + i / 100, url: `https://lead${i}.example/p`, date: '2026-09-23', why: `Not confirmed (${tier}).` }]])) }];
/* Board counts a lead is held out of the figures at: one board, two (a lead once
   reached the per-board cell at exactly two, and only the golden saw it), odd, four
   (and at four and above, which a sweep of one and three missed), and odd above eight. */
const LEAD_BOARDS = [1, 2, 3, 4, 9];

test('a lead changes no figure: every row with one renders identically without it, but for the lead itself', () => {
  /* The owner's rule for Runcrate, and for every lead since: shown, never used as a
     price. Stated as the property that makes it true: take a lead out of the catalog
     and nothing the tool computes moves, and no surface changes but by the lead's own
     text. A lead that reached a range, a monthly total, a comparison or a summary
     would leave a difference behind here that no text removal accounts for. */
  assert.ok(ROWS_WITH_LEADS.length >= 3, `only ${ROWS_WITH_LEADS.length} catalog rows carry a lead — this checks too little`);
  const renderAll = (card, boards) => {
    const h = renderHarness();
    const st = asState(card, boards, { params: 8, layers: 32 });
    const c = h.computeInference(st);
    h.pushSnapshot(st, c);
    for (const name of Object.keys(h).filter(k => /^render/.test(k) && typeof h[k] === 'function')) {
      try { h[name](st, c); } catch (e) { /* renderers that take other arguments are covered elsewhere */ }
    }
    return { c, out: { ...h.out, '(copied report)': h.exportSummary(st, c) } };
  };
  for (const [key, gpu] of [...ROWS_WITH_LEADS, EVERY_TIER_LEADS]) {
    const bare = { ...gpu }; delete bare.priceLead;
    for (const boards of LEAD_BOARDS) {
      const withLead = renderAll(gpu, boards), without = renderAll(bare, boards);
      assert.deepStrictEqual(withLead.c, without.c, `${key} x${boards}: a lead changed what the page computes`);
      let removed = 0;
      for (const [id, text] of Object.entries(withLead.out)) {
        let rest = text;
        for (const leads of Object.values(gpu.priceLead)) for (const l of leads) {
          const html = leadHtml(l), line = leadLine(l);
          if (rest.includes(html)) { rest = rest.split(html).join(''); removed++; }
          if (rest.includes(line)) { rest = rest.split(line).join(''); removed++; }
        }
        rest = rest.split(LEAD_SENTENCE).join('');
        assert.strictEqual(rest, without.out[id] || '',
          `${key} x${boards}/${id}: with its lead taken out of the text, the surface still differs from the catalog without it`);
        for (const leads of Object.values(gpu.priceLead)) for (const l of leads)
          assert.ok(!rest.includes(`$${l.price.toFixed(2)}`), `${key} x${boards}/${id}: the lead's figure appears outside the lead`);
      }
      const expected = Object.values(gpu.priceLead).reduce((n, ls) => n + ls.length, 0) * 2;
      assert.strictEqual(removed, expected, `${key} x${boards}: expected each lead once in the cost table and once in the report`);
    }
  }
});

test('each lead sits under its own tier, after the note, and only where the tier has no price', () => {
  const ROWS = { hyper: 'Hyperscaler', spec: 'Specialized', spot: 'Spot / marketplace' };
  const LINES = { hyper: 'Hyperscaler', spec: 'Specialized', spot: 'Spot' };
  for (const [key, gpu] of [...ROWS_WITH_LEADS, EVERY_TIER_LEADS]) {
    const h = renderHarness();
    const st = asState(gpu, 1, { params: 8, layers: 32 });
    const c = h.computeInference(st);
    h.renderCost(st, c);
    const report = h.exportSummary(st, c).split('\n');
    const costRows = (h.out['cost-output'] || '').split('<tr').slice(1);
    for (const tier of ['hyper', 'spec', 'spot']) {
      const leads = (gpu.priceLead || {})[tier] || [];
      const row = costRows.find(r => r.includes(`<b>${ROWS[tier]}</b>`));
      const at = report.findIndex(l => l.startsWith(`- ${LINES[tier]}: `));
      if (!leads.length) {
        assert.ok(!row.includes('Lead, not used'), `${key}/${tier}: a lead appears under a tier that has none`);
        continue;
      }
      assert.strictEqual(gpu[tier], null, `${key}/${tier}: a lead on a priced tier — the catalog tests should refuse this`);
      const note = catalogNoteText(st, tier);
      assert.ok(row.includes(`${note}</span>${leads.map(leadHtml).join('')}</td>`),
        `${key}/${tier}: the cost table does not carry the lead right after the note — ${row.slice(0, 500)}`);
      assert.deepStrictEqual(report.slice(at + 1, at + 2 + leads.length),
        [`  - ${note}`, ...leads.map(l => leadLine(l).trimEnd())], `${key}/${tier}: the copied report's lines under ${LINES[tier]}`);
    }
  }
  // And a lead the catalog puts on a priced tier is not shown: a price is the answer there.
  const h = renderHarness();
  const priced = { ...GPU_TABLE['h100-80'], priceLead: { hyper: [{ provider: 'X', price: 1.23, url: 'https://x.example/p', date: '2026-09-23', why: 'Test.' }] } };
  const st = asState(priced, 1, { params: 8, layers: 32 });
  const c = h.computeInference(st);
  h.renderCost(st, c);
  assert.ok(!(h.out['cost-output'] || '').includes('Lead, not used') && !h.exportSummary(st, c).includes('Lead, not used'),
    'a lead on a priced tier is shown');
});

test('the notes explain leads on every card that shows one, and on no other', () => {
  /* The sentence was once gated on the specialized tier's leads alone, and only the
     golden saw a card whose leads sit on spot lose it. Every catalog row, and a card
     with a single lead on each tier in turn. */
  const lead = { provider: 'L', price: 1.5, url: 'https://l.example/p', date: '2026-09-23', why: 'Not confirmed.' };
  const cases = [...Object.entries(GPU_TABLE), ...['hyper', 'spec', 'spot'].map(tier =>
    [`mi210-64 (a lead on ${tier} only)`, { ...GPU_TABLE['mi210-64'], priceLead: { [tier]: [lead] } }])];
  let shown = 0;
  for (const [key, gpu] of cases) {
    const h = renderHarness();
    const st = asState(gpu, 1, { params: 8, layers: 32 });
    const c = h.computeInference(st);
    h.renderNotes(st, c);
    const shows = ['hyper', 'spec', 'spot'].some(tier => gpu[tier] === null && ((gpu.priceLead || {})[tier] || []).length);
    assert.strictEqual((h.out['notes-output'] || '').includes(LEAD_SENTENCE), shows,
      `${key}: the notes ${shows ? 'lack' : 'carry'} the sentence explaining leads`);
    shown += shows;
  }
  assert.ok(shown >= 5, `only ${shown} cards showed a lead`);
});

test("the notes print the peer-buffer figure the math charges, in the card's own library's name", () => {
  /* Until 2026-09-23 the note said "NCCL buffers ~0.3 GB/peer" on every link while
     computeInference() charged 0.2 without NVLink, so every PCIe plan, and every AMD
     plan, described a charge it wasn't making, in NVIDIA's library's name on AMD's
     cards. The figure the note must print is read back from the engine's own total
     (1.5 GB of runtime context per device, and the rest per extra device), not from
     its source: every catalog card, at one, two and three boards, over each link the
     card can have. */
  let checked = 0;
  for (const [key, card] of Object.entries(GPU_TABLE)) {
    for (const nvlink of supportsNVLink(card) ? [true, false] : [false]) {
      for (const boards of [1, 2, 3]) {
        const h = renderHarness();
        const st = asState(card, boards, { params: 8, layers: 32, hasNVLink: nvlink });
        const c = h.computeInference(st);
        h.renderNotes(st, c);
        const notes = (h.out['notes-output'] || '').replace(/<[^>]*>/g, ' ');
        const lib = card.vendor === 'amd' ? 'RCCL' : 'NCCL', other = lib === 'RCCL' ? 'NCCL' : 'RCCL';
        const where = `${key} x${boards}, ${nvlink ? 'NVLink' : 'no NVLink'}`;
        assert.ok(!notes.includes(other), `${where}: the notes name ${other} on a card that uses ${lib}`);
        if (c.deviceCount > 1) {
          const charged = Math.round(((c.totalOverhead - 1.5 * c.deviceCount) / (c.deviceCount - 1)) * 10) / 10;
          assert.ok(notes.includes(`${lib} buffers ~${charged} GB/peer.`),
            `${where}: the math charges ${charged} GB per extra device, the notes say ${JSON.stringify((notes.match(/\w+ buffers ~[\d.]+ GB\/peer/) || ['nothing'])[0])}`);
          /* And the charge itself, as the contract: 0.3 GB over NVLink, 0.2 over any other
             link. Both engines charging 0.3 on PCIe or Infinity Fabric, alike, agreed with
             each other and with their notes, and only the goldens saw it. */
          const link = nvlink && supportsNVLink(card);
          assert.strictEqual(charged, link ? 0.3 : 0.2,
            `${where}: the math charges ${charged} GB per extra device ${link ? 'over NVLink' : 'off NVLink'}`);
          checked++;
        } else {
          assert.ok(!/buffers ~/.test(notes), `${where}: one device, and the notes still describe peer buffers`);
        }
      }
    }
  }
  assert.ok(checked >= 30, `only ${checked} multi-device plans were checked`);
});

test('the Cost/hr and Monthly cost range surfaces span the true cheapest and priciest tier', () => {
  /* Cold-check finding: renderComparisons' "Cost/hr" and renderExecutiveSummary's
     "Monthly cost range" both hardcoded spot as the floor and hyperscaler as
     the ceiling, which held only while every catalog price satisfied
     spot <= specialized <= hyperscaler — no longer true for l40s-48 (specialized
     above hyperscaler) or rtx4090-24 (spot above specialized). The golden alone
     does not catch a wrong-but-plausible-looking range: this asserts the
     displayed figures equal Math.min/max of the three hourly costs directly,
     computed independently of what the renderer did, for every catalog row —
     not just the two known-broken ones, so a card that develops the same
     inversion later is caught the same way. Found the hard way: this specific
     sabotage (spot-to-hyperscaler instead of true min/max) survived with the
     golden regenerated and every other test green — the golden was the only
     thing that had ever looked at this line. */
  let checkedCmp = 0, checkedExec = 0, inverted = 0;
  for (const [slug, card] of Object.entries(GPU_TABLE)) {
    // A fresh harness per card: savedSnapshots accumulates on one shared
    // harness (see the FP8 sweep above), which left an earlier card's
    // comparison row still in the grid when this reached the next card's —
    // the regex below matched the FIRST Cost/hr row in the page, not
    // necessarily this card's.
    const h = renderHarness();
    const st = asState(card, 1, { params: 8, layers: 32 });
    const c = h.computeInference(st);
    /* A row with a tier the catalog records as null prices fewer than three:
       its range spans the priced tiers, one priced tier prints alone, and none
       prints the null wording. The test below walks every null pattern on a
       synthetic card; this holds each real row to the same rule. */
    const priced = [c.hourlyHyper, c.hourlySpec, c.hourlySpot].filter(v => v !== null);
    if (priced.length < 2) {
      h.pushSnapshot(st, c);
      h.renderComparisons();
      h.renderExecutiveSummary(st, c);
      const want = priced.length ? [`$${priced[0].toFixed(2)}`, `$${Math.round(priced[0] * 730).toLocaleString()}/mo`]
                                 : ['no confirmed hourly price', 'no confirmed hourly price'];
      assert.ok((h.out['comparison-output'] || '').includes(`Cost/hr</span><span class="val">${want[0]}</span>`),
        `${slug}: renderComparisons' Cost/hr is not ${want[0]}`);
      assert.ok((h.out['exec-summary'] || '').includes(`Monthly cost range</span><span class="exec-value">${want[1]}</span>`),
        `${slug}: the Monthly cost range is not ${want[1]}`);
      checkedCmp++; checkedExec++;
      continue;
    }
    const trueMin = Math.min(...priced);
    const trueMax = Math.max(...priced);
    /* The regime that broke: a range whose floor is not spot, or whose ceiling is
       not hyperscaler, so a renderer that hardcodes spot-to-hyperscaler prints the
       wrong figures. A card with no hyperscaler price is in it too: the hardcoded
       range has no ceiling to print there. */
    if (trueMin !== c.hourlySpot || trueMax !== c.hourlyHyper) inverted++;

    h.pushSnapshot(st, c);
    h.renderComparisons();
    const cmpText = h.out['comparison-output'] || '';
    const cmpMatch = /Cost\/hr<\/span><span class="val">\$([\d.]+)–\$([\d.]+)/.exec(cmpText);
    assert.ok(cmpMatch, `${slug}: renderComparisons printed no Cost/hr range at all`);
    checkedCmp++;
    assert.strictEqual(Number(cmpMatch[1]), Number(trueMin.toFixed(2)),
      `${slug}: renderComparisons' Cost/hr floor is ${cmpMatch[1]}, expected the true cheapest ` +
      `tier ${trueMin.toFixed(2)} (hyper=${c.hourlyHyper} spec=${c.hourlySpec} spot=${c.hourlySpot})`);
    assert.strictEqual(Number(cmpMatch[2]), Number(trueMax.toFixed(2)),
      `${slug}: renderComparisons' Cost/hr ceiling is ${cmpMatch[2]}, expected the true priciest ` +
      `tier ${trueMax.toFixed(2)} (hyper=${c.hourlyHyper} spec=${c.hourlySpec} spot=${c.hourlySpot})`);

    Object.keys(h.out).forEach(k => delete h.out[k]);
    h.renderExecutiveSummary(st, c);
    const execText = h.out['exec-summary'] || '';
    const execMatch = /Monthly cost range<\/span><span class="exec-value">\$([\d,]+) – \$([\d,]+)\/mo/.exec(execText);
    assert.ok(execMatch, `${slug}: renderExecutiveSummary printed no Monthly cost range at all`);
    checkedExec++;
    const gotCheap = Number(execMatch[1].replace(/,/g, ''));
    const gotExpensive = Number(execMatch[2].replace(/,/g, ''));
    assert.strictEqual(gotCheap, Math.round(trueMin * 730),
      `${slug}: renderExecutiveSummary's monthly floor is $${gotCheap}, expected the true ` +
      `cheapest tier's $${Math.round(trueMin * 730)}`);
    assert.strictEqual(gotExpensive, Math.round(trueMax * 730),
      `${slug}: renderExecutiveSummary's monthly ceiling is $${gotExpensive}, expected the true ` +
      `priciest tier's $${Math.round(trueMax * 730)}`);
  }
  assert.strictEqual(checkedCmp, Object.keys(GPU_TABLE).length);
  assert.strictEqual(checkedExec, Object.keys(GPU_TABLE).length);
  assert.ok(inverted >= 2,
    `only ${inverted} catalog row(s) have a range other than spot-to-hyperscaler — expected at least ` +
    'l40s-48 and the cards with no hyperscaler price, so this sweep is not actually exercising the broken regime');
});



console.log('\nHardware with no measured constants');
/* The ruling: a card whose perfKey has no PERF entry gets its full VRAM
   breakdown, fit verdict, cost and vLLM command, and no throughput at all. Every
   figure that needs a constant is null, and every view says why instead of
   printing one. Nothing borrows another entry's constants — the fallback to
   PERF.nvidia is what this replaces.

   Every probe is a pair: the same configuration on a card with constants and on
   the same card without. The card is derived from dualGCD rather than being it —
   the device-splitting tests above rely on dualGCD having constants, and turning
   it into this case would leave them green while testing something else. */
const noConstants = { ...dualGCD, perfKey: 'no-such-key' };

/* The probes, each a pair: a card with constants and the same card without.

   Every row in the catalog is a single-device board, and the AMD rows to come
   mostly will be, so a grid built only from the dual-GCD fixture would never
   render the shape the tool actually ships. A renderer could print an estimate,
   or withhold a figure, on `gpuDevices === 1` — or on the KV dtype, or on an
   attention mode, or on a preset being selected — and nothing would look. So
   the grid runs real catalog rows as well as the fixture, and rotates what a
   renderer can branch on: devices per board, board count (including counts that
   split data-parallel), the model's attention regime, the weight precision, the
   KV dtype, the interconnect, the load, whether a preset is named, and whether
   the card has benchmark data on file. */
const PROBE_CARDS = [
  ['H100 80GB (sxm, FP8 cores)', GPU_TABLE['h100-80'], 'h100-80'],
  ['A100 80GB (sxm, no FP8 cores)', GPU_TABLE['a100-80'], 'a100-80'],
  ['RTX 4090 (consumer, PCIe)', GPU_TABLE['rtx4090-24'], 'rtx4090-24'],
  ['T4 16GB (PCIe, no FP8 cores)', GPU_TABLE['t4-16'], 't4-16'],
  ['B200 192GB (sxm)', GPU_TABLE['b200-192'], 'b200-192'],
  ['dual-GCD board (2 devices)', dualGCD, undefined],
  /* Two cards whose vendor is not nvidia, under names no catalog row has, so a
     renderer branching on the vendor — or on a name it has never seen — is
     probed apart from the real rows below. Their perfKey is one PERF has,
     because a probe is a pair and the constants are what the pair varies:
     fixing the key here is what isolates the vendor. */
  ['MI355X 288GB (amd, oam, not in the catalog)',
   { ...GPU_TABLE['b200-192'], name: 'MI355X 288 GB', vendor: 'amd', form: 'oam' }, undefined],
  ['Radeon PRO W7900 (amd, workstation)',
   { ...GPU_TABLE['rtx6000ada-48'], name: 'Radeon PRO W7900 48 GB', vendor: 'amd' }, undefined],
  /* And every real row whose vendor is not nvidia, derived from the catalog so
     a row added later joins the grid without editing this list: their real
     names, forms, device counts and unpriced tiers, given a key PERF has for the
     side of the pair with constants. */
  ...Object.entries(GPU_TABLE).filter(([, g]) => g.vendor !== 'nvidia')
    .map(([key, g]) => [`${g.name} (catalog, ${g.vendor}, ${g.form})`, { ...g, perfKey: 'nvidia' }, key]),
];
const PROBE_MODELS = [
  ['8B dense', { params: 8, layers: 32, kvHeads: 8, headDim: 128, activePercent: 100 }],
  ['70B dense', { params: 70, layers: 80, kvHeads: 8, headDim: 128, activePercent: 100 }],
  ['30B MoE', { params: 30, layers: 48, kvHeads: 8, headDim: 128, activePercent: 10, sharedExperts: 1 }],
  ['26B SWA', { params: 26, layers: 30, kvHeads: 8, headDim: 256, activePercent: 15,
                attnMode: 'swa', swaWindow: 1024, swaLocalLayers: 25 }],
  ['671B MLA', { params: 671, layers: 61, kvHeads: 128, headDim: 56, activePercent: 5, sharedExperts: 1,
                 attnMode: 'mla', mlaLatentDim: 576 }],
];
const PROBE_PRECISIONS = [
  ['bf16', { bytesPerParam: 2, quantMethod: '' }],
  ['fp8', { bytesPerParam: 1, quantMethod: 'fp8' }],
  ['awq', { bytesPerParam: 0.5, quantMethod: 'awq' }],
  // Both of the dropdown's other quantisations. GGUF is not cosmetic:
  // renderCommand already branches on it, so it is a branch a leak can ride.
  ['gptq', { bytesPerParam: 0.5, quantMethod: 'gptq' }],
  ['gguf Q4_K_M', { bytesPerParam: 0.63, quantMethod: 'gguf' }],
];
const PROBE_LOADS = [
  ['16 at 8K', { contextLength: 8192, concurrency: 16 }],
  ['256 at 1K', { contextLength: 1024, concurrency: 256 }],
  ['4 at 32K behind an 8K prefix', { contextLength: 32768, concurrency: 4, sharedPrefix: 8192, prefixCaching: true }],
  // A prefix configured and the caching switched off, which the grid never took:
  // every load that named a prefix also enabled caching for it.
  ['4 at 32K, 8K prefix, caching off',
   { contextLength: 32768, concurrency: 4, sharedPrefix: 8192, prefixCaching: false }],
];
/* More than one key with no PERF entry, because a leak can be gated on the key
   itself rather than on its absence. Every key a catalog row names that PERF has
   no entry for — the AMD architectures — derived rather than typed, so a leak
   gated on any real one is probed, plus a placeholder no row will ever name.
   One key would have made "the key is unknown" and "the key is this string"
   the same probe. */
const PROBE_UNKNOWN_KEYS = ['no-such-key',
  ...new Set(Object.values(GPU_TABLE).map(g => g.perfKey).filter(k => !Object.hasOwn(PERF, k)))];
assert.ok(PROBE_UNKNOWN_KEYS.length >= 2, 'no catalog row names a key PERF lacks, so only the placeholder is probed');
/* Every catalog row's name as the page's state spells it, for the probe axis
   below that needs a name no row carries. */
const CATALOG_NAMES = new Set(Object.values(GPU_TABLE).map(displayName));
const absentProbes = () => {
  const probes = [];
  let i = 0;
  for (const [cardName, card, gpuKey] of PROBE_CARDS)
    // Counts that are not powers of two, that sit on an NVLink domain boundary,
    // and that sit past one with an uneven split.
    for (const count of [1, 2, 3, 8, 12, 16]) {
      const [modelName, model] = PROBE_MODELS[i % PROBE_MODELS.length];
      const [precName, precision] = PROBE_PRECISIONS[(i >> 1) % PROBE_PRECISIONS.length];
      const [loadName, load] = PROBE_LOADS[(i >> 2) % PROBE_LOADS.length];
      const kvBytesPerValue = i % 2 ? 1 : 2;
      const hasNVLink = i % 3 !== 0;
      const presetKey = i % 4 === 1 ? 'llama31-8b' : '';
      // The page's other naming path: a model imported by HuggingFace id. The
      // two are exclusive on the page — importing clears the preset.
      const hfModelId = i % 4 === 3 ? 'org/imported-27b' : null;
      // Decorrelated from the KV dtype, so "this key" and "FP8 KV" are not one probe.
      const unknownKey = PROBE_UNKNOWN_KEYS[(i >> 3) % PROBE_UNKNOWN_KEYS.length];
      const extra = { ...model, ...precision, ...load, kvBytesPerValue, hasNVLink,
                      presetKey, hfModelId, gpuKey };
      probes.push({
        label: `${cardName} x${count}, ${modelName}, ${precName}, ${loadName}, ` +
          `KV ${kvBytesPerValue === 1 ? 'FP8' : 'BF16'}, ${hasNVLink ? 'NVLink' : 'PCIe'}` +
          `${presetKey ? ', preset named' : ''}${hfModelId ? ', model imported' : ''}` +
          `, unknown key "${unknownKey}"`,
        known: asState(card, count, extra),
        unknown: asState({ ...card, perfKey: unknownKey }, count, extra),
      });
      i++;
    }
  return probes;
};

/* Every axis the grid claims to take, counted while it runs. A grid that stops
   covering one says nothing about that shape, and every test below would still
   be green, so the claim is checked rather than left in a comment. */
test('the probe grid renders the shapes the tool ships', () => {
  const axes = {
    'a single-device board': p => (p.known.gpuDevices || 1) === 1,
    'two devices on one board': p => (p.known.gpuDevices || 1) > 1,
    'one board': p => p.known.gpuCount === 1,
    'a board count that is not a power of two': p => ![1, 2, 4, 8, 16].includes(p.known.gpuCount),
    'a full NVLink domain': p => p.known.gpuCount * (p.known.gpuDevices || 1) === 8,
    'past an NVLink domain': p => p.known.gpuCount * (p.known.gpuDevices || 1) > 8,
    'a vendor that is not nvidia': p => p.known.vendor !== 'nvidia',
    'a card name unlike the catalog\'s': p => !CATALOG_NAMES.has(p.known.gpuName),
    'an unknown key that is not the placeholder': p => p.unknown.perfKey !== 'no-such-key',
    'the placeholder unknown key': p => p.unknown.perfKey === 'no-such-key',
    'FP8 KV cache': p => p.known.kvBytesPerValue < 2,
    'BF16 KV cache': p => p.known.kvBytesPerValue === 2,
    'NVLink': p => p.known.hasNVLink,
    'PCIe': p => !p.known.hasNVLink,
    'sliding-window attention': p => p.known.attnMode === 'swa',
    'MLA': p => p.known.attnMode === 'mla',
    'a mixture of experts': p => p.known.activePercent < 100,
    'GGUF weights': p => p.known.quantMethod === 'gguf',
    'GPTQ weights': p => p.known.quantMethod === 'gptq',
    'AWQ weights': p => p.known.quantMethod === 'awq',
    'FP8 weights': p => p.known.quantMethod === 'fp8',
    'unquantised weights': p => p.known.quantMethod === '',
    'a shared prefix with caching on': p => p.known.sharedPrefix > 0 && p.known.prefixCaching,
    'a shared prefix with caching off': p => p.known.sharedPrefix > 0 && !p.known.prefixCaching,
    'a model the preset named': p => !!p.known.presetKey,
    'a model imported by id': p => !!p.known.hfModelId,
    'a model neither named nor imported': p => !p.known.presetKey && !p.known.hfModelId,
    'a card with FP8 tensor cores': p => p.known.gpuFp8,
    'a card without them': p => !p.known.gpuFp8,
  };
  const probes = absentProbes();
  const missing = Object.entries(axes).filter(([, hits]) => !probes.some(hits)).map(([name]) => name);
  assert.deepStrictEqual(missing, [], `the probe grid no longer renders: ${missing.join(', ')}`);
});

/* Every field computeInference() returns that reads no PERF constant. A literal on
   purpose: it is the ruling written down. Everything else the result carries has
   to be null without constants, so a throughput figure added later that forgets
   to suppress fails the next test — and so does a new VRAM figure, until someone
   decides which side of this list it is on. */
const WITHOUT_CONSTANTS_SURVIVE = [
  // VRAM, capacity and the fit they decide
  'isMoE', 'weightsGB', 'kvCacheGB', 'activationsGB', 'totalOverhead', 'peerBufferGB', 'totalGB', 'perGPU',
  'totalVRAM', 'deviceCount', 'deviceGB', 'deviceBandwidth', 'freeForKVCache', 'kvPerTokenGB',
  'kvBytesPerToken', 'kvSavedByPrefixGB', 'effectivePrefix', 'totalTokens', 'fits', 'comfortable',
  'maxContextSingleUser', 'maxConcurrentAt8K', 'maxConcurrentAt4K',
  // the parallelism split
  'tp', 'dp', 'shardDivisor', 'modelCopies',
  // batch sizes, which are KV arithmetic
  'maxBatchByKV', 'effectiveBatch', 'saturatedBatch', 'batchLimitedByKV',
  // cost
  'hourlyHyper', 'hourlySpec', 'hourlySpot',
  // a fact about the card and the precision rather than a constant: FP8 was asked
  // for on silicon without FP8 tensor cores whether or not anything is estimated
  'fp8NoTensorCores',
];
// The fields the ruling names, so the list above cannot absorb one of them.
const NAMED_SUPPRESSED = ['singleStreamTokS', 'aggregateTokS', 'saturatedTokS', 'perUserAtLoadTokS',
  'aggregateObservedLoTokS', 'aggregateObservedHiTokS', 'ttftMs', 'ttftColdMs', 'ttftWarmMs',
  'computeBound', 'computeRatio', 'perfMbu', 'perfMfuDecode', 'perfMfuPrefill', 'perfObsLo',
  'perfObsHi', 'perfFp8Ratio'];

test('with no constants, every figure that needs one is null, enumerated from the result', () => {
  let bound = false;
  for (const { label, known: ks, unknown: us } of absentProbes()) {
    const known = computeInference(ks);
    const unknown = computeInference(us);
    assert.strictEqual(known.throughputModelled, true, `${label}: the card with constants is not modelled`);
    assert.strictEqual(unknown.throughputModelled, false, `${label}: a perfKey with no entry is modelled`);
    assert.deepStrictEqual(Object.keys(unknown).sort(), Object.keys(known).sort(),
      `${label}: the two results do not carry the same fields`);
    const suppressed = Object.keys(known)
      .filter(k => k !== 'throughputModelled' && !WITHOUT_CONSTANTS_SURVIVE.includes(k));
    for (const k of NAMED_SUPPRESSED)
      assert.ok(suppressed.includes(k), `${k} is a figure the ruling suppresses, but it is listed as surviving`);
    for (const k of suppressed) {
      // null exactly: 0 prints as "0 tok/s", and undefined vanishes from JSON.
      assert.strictEqual(unknown[k], null,
        `${label}: ${k} is ${JSON.stringify(unknown[k])} with no constants — it has to be null`);
      assert.ok(known[k] !== null && known[k] !== undefined,
        `${label}: ${k} is ${known[k]} even with constants, so this cannot tell it was suppressed`);
    }
    bound = bound || known.computeBound === true;
  }
  assert.ok(bound, 'no probe sits on the compute ceiling, so computeBound was only ever false');
});

test('with no constants, VRAM, fit, cost, the split, the batch sizes and the command do not move', () => {
  const h = renderHarness();
  let over = 0;
  for (const { label, known: ks, unknown: us } of absentProbes()) {
    const known = computeInference(ks);
    const unknown = computeInference(us);
    for (const k of WITHOUT_CONSTANTS_SURVIVE) {
      assert.ok(k in known, `${k} is listed as surviving, but computeInference() returns no such field`);
      assert.deepStrictEqual(unknown[k], known[k],
        `${label}: ${k} moved when the constants went: ${JSON.stringify(known[k])} -> ${JSON.stringify(unknown[k])}`);
    }
    assert.strictEqual(h.buildVllmCommand(us, unknown, 'm'), h.buildVllmCommand(ks, known, 'm'),
      `${label}: the vLLM command depends on the constants`);
    if (!known.fits) {
      over++;
      assert.deepStrictEqual(h.boardsNeeded(us, unknown), h.boardsNeeded(ks, known),
        `${label}: the board count the verdict recommends depends on the constants`);
      assert.strictEqual(h.boardsAdvice(us, unknown), h.boardsAdvice(ks, known),
        `${label}: the advice depends on the constants`);
    }
  }
  assert.ok(over > 0, 'every probe fits, so the board recommendation was never compared');
});

test('a perfKey with no entry gets no constants — never NVIDIA\'s, whatever it looks like', () => {
  /* The regression this replaces: an unknown key used to take PERF.nvidia. A
     typo, a case slip, a vendor name, a prototype member and a non-string all
     have to come out absent, and none of them may throw. */
  const nvidia = computeInference(state());
  assert.strictEqual(nvidia.throughputModelled, true, 'the H100 probe must be modelled');
  const typos = ['Nvidia', 'NVIDIA', 'nvidia ', ' nvidia', 'nvidia​', 'nvida', 'amd', 'cdna3', '',
                 'constructor', 'toString', '__proto__', 'hasOwnProperty', 'valueOf',
                 undefined, null, 0, 42, true, ['nvidia'], { toString: () => 'nvidia' }];
  const { perfKey: _dropped, ...keyless } = state();
  const describe = (t) => (typeof t === 'string' ? JSON.stringify(t)
    : Array.isArray(t) ? `the array ${JSON.stringify(t)}`
    : t && typeof t === 'object' ? 'an object whose toString() is "nvidia"' : String(t));
  const cases = [...typos.map(t => [describe(t), state({ perfKey: t })]), ['(absent from the state)', keyless]];
  for (const [shown, st] of cases) {
    let c;
    assert.doesNotThrow(() => { c = computeInference(st); }, `perfKey ${shown} threw`);
    assert.strictEqual(c.throughputModelled, false, `perfKey ${shown} was treated as having constants`);
    assert.notStrictEqual(c.singleStreamTokS, nvidia.singleStreamTokS,
      `perfKey ${shown} was given NVIDIA's single-stream figure`);
    for (const k of ['singleStreamTokS', 'aggregateTokS', 'ttftMs', 'perfMbu', 'perfMfuDecode'])
      assert.strictEqual(c[k], null, `perfKey ${shown}: ${k} is ${c[k]}`);
  }
  /* And the lookup reads perfKey, not vendor — the field it used to read. A
     vendor may neither grant constants nor take them away. */
  const vendorOnly = computeInference(state({ vendor: 'nvidia', perfKey: 'no-such-key' }));
  assert.strictEqual(vendorOnly.throughputModelled, false, 'vendor "nvidia" granted constants to a key with none');
  const otherVendor = computeInference(state({ vendor: 'acme', perfKey: 'nvidia' }));
  assert.strictEqual(otherVendor.throughputModelled, true, 'vendor "acme" removed constants its perfKey has');
  assert.strictEqual(otherVendor.singleStreamTokS, nvidia.singleStreamTokS,
    'the same perfKey under another vendor computed a different figure');
  assert.strictEqual(otherVendor.perfMbu, PERF.nvidia.mbu);
});

/* Every renderer the harness exposes, called the way index.html declares it,
   plus the copied report — the same discovery the two caveat sweeps use. The
   engine's own result is rendered unless another one is handed in.

   What comes back is everything the harness recorded: what each element renders,
   what a textContent write put there, and what a title or data-* attribute was
   set to. A view is all three, because a reader sees all three. */
const surfaceParams = {};
for (const m of html.matchAll(/function (render\w+)\(([^)]*)\)/g))
  surfaceParams[m[1]] = m[2].split(',').map(x => x.trim().split(/[=\s]/)[0]).filter(Boolean);
const renderEverything = (st, given) => {
  const h = renderHarness();
  const renderers = Object.keys(h).filter(k => /^render/.test(k) && typeof h[k] === 'function');
  const c = given || h.computeInference(st);
  /* readInputState() reads the imported id off a closure variable, not the
     state, so a probe that names one has to set it: otherwise the command, the
     executive view and the snapshot name all read the preset path and the
     imported-model shape is never rendered. */
  if (st.hfModelId) h.setImportedModel(st.hfModelId);
  h.pushSnapshot(st, c);
  for (const name of renderers) {
    assert.ok(surfaceParams[name], `${name} is exposed but not declared in index.html`);
    h[name](...surfaceParams[name].map(pn => (pn === 'computed' ? c : pn === 'state' ? st : undefined)));
  }
  h.out['(copied report)'] = h.exportSummary(st, c);
  return { html: h.out, written: h.shown, props: h.props };
};

/* A throughput or TTFT figure, however its unit is spelled: "tok/s", "t/s",
   "tokens/sec", "tokens per second", "per sec", "ms", "milliseconds". Nothing
   below leans on it alone — a figure under a unit nobody listed still has to be
   text the card with constants does not show, and that is the rule that
   catches it. */
const FIGURE = /\bt(?:ok(?:en)?s?)?\s*(?:\/|per)\s*s(?:ec(?:ond)?s?)?\b|\bper\s+sec(?:ond)?s?\b|\btps\b|\d\s*ms\b|\bmilli-?seconds?\b/i;
/* A claim about speed that is not a figure. Each rests on the same heuristics
   the constants do, and each may be dropped or shortened for a card without
   them — nothing else may be. */
const SPEECH = {
  'a PCIe loss percentage': /\d+-\d+% (?:perf|decode) loss/i,
  'a bandwidth-bound estimate': /bandwidth-bound/i,
  'a compute-bound label': /compute-bound/i,
  'an FP8 compute caveat': /FP8 tensor cores/i,
  'the observed band': /observed range/i,
  'a pointer to throughput figures': /throughput figures|aggregate throughput above/i,
};
const speaks = (text) => Object.values(SPEECH).some(re => re.test(text));

/* What a reader sees of a piece of markup: tags dropped, entities decoded,
   whitespace collapsed — and the card's own name taken out, because a name is
   the one place in a reason where a digit belongs. */
const seenText = (markup, st) => String(markup ?? '')
  .replace(/<[^>]*>/g, ' ')
  .replace(/&nbsp;/g, ' ').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&')
  .split(st.gpuName).join(' ')
  .replace(/\s+/g, ' ').trim();
const numbersIn = (text) => text.match(/\d+(?:[.,]\d+)*/g) || [];
const wordsOf = (text) => text.toLowerCase().match(/[a-z0-9]+/g) || [];
// Whether every item of `few` occurs in `many`, in the same order.
const inOrderWithin = (few, many) => {
  let i = 0;
  for (const x of many) if (i < few.length && x === few[i]) i++;
  return i === few.length;
};
// A sentence with words dropped and nothing added: how a caveat is shortened.
const isShorteningOf = (short, long) => inOrderWithin(wordsOf(short), wordsOf(long));
/* A view as a reader takes it in: the copied report line by line, an element
   tile by tile. Each piece carries the sentences of its text and the values of
   the attributes that carry text — an attribute is read rather than stripped
   with its tag, because a title is a place a figure can hide. */
const piecesOf = (id, markup) => (id === '(copied report)'
  ? String(markup ?? '').split('\n')
  : String(markup ?? '').split(/(?=<div\b)|(?=<span class="badge")/));
const sentencesOf = (text) => text.split(/(?<=[.!?])\s+/).filter(Boolean);
const ATTRIBUTE = /\s(?:title|aria-label|alt|data-[\w-]+)=(?:"([^"]*)"|'([^']*)')/g;
/* Every element a render touched, however it touched it. Walking the innerHTML
   keys alone skipped the thirteen slider labels, the counting link and the
   restore notice outright — they are only ever written through textContent or a
   style — so a figure put in one of them was recorded by the harness and read by
   nothing. */
const viewIds = (v) => [...new Set([...Object.keys(v.html), ...Object.keys(v.written),
                                    ...Object.keys(v.props)])];
/* And everything it put there, as one string: markup, text written directly, and
   the values of any attributes or styles set on the element. */
const viewText = (v, id, st) => [String(v.html[id] ?? ''), String(v.written[id] ?? ''),
                                 ...Object.values(v.props[id] || {})].join(' ');
const viewPieces = (views, id, st) => {
  const pieces = piecesOf(id, views.html[id]).map(raw => ({
    raw,
    texts: [...sentencesOf(seenText(raw, st)),
            // Either quoting: the page writes double quotes today, and a single
            // quote is one keystroke away from hiding a value from this scan.
            ...[...raw.matchAll(ATTRIBUTE)].map(m => seenText(m[1] ?? m[2], st))].filter(Boolean),
  }));
  /* Writes that never touch innerHTML: a textContent assignment, a title, a
     data-* attribute set on the element itself. */
  const written = seenText(views.written[id], st);
  const set = Object.entries(views.props[id] || {}).map(([k, v]) => `${k}=${seenText(v, st)}`);
  const asideTexts = [...sentencesOf(written), ...Object.values(views.props[id] || {})
    .map(v => seenText(v, st))].filter(Boolean);
  if (asideTexts.length) pieces.push({ raw: `[written] ${written} ${set.join(' ')}`, texts: asideTexts });
  return pieces.filter(piece => piece.texts.length);
};
/* The same result with every figure that needs a constant moved — numbers
   scaled and offset, flags flipped — and everything else untouched. A piece
   that reads differently under it is a piece that shows one of those figures;
   no unit has to be recognised to find it. */
const withMovedFigures = (c) => Object.fromEntries(Object.entries(c).map(([k, v]) =>
  [k, k === 'throughputModelled' || WITHOUT_CONSTANTS_SURVIVE.includes(k) ? v
    : typeof v === 'number' ? v * 3 + 7 : typeof v === 'boolean' ? !v : v]));
const firstDifference = (a = '', b = '') => {
  let i = 0;
  while (i < a.length && a[i] === b[i]) i++;
  return `from character ${i}: "${a.slice(Math.max(0, i - 80), i + 80)}" became ` +
    `"${b.slice(Math.max(0, i - 80), i + 80)}"`;
};

/* Rendered once and shared by the tests below: the card with constants, the
   same card with every figure that needs one moved, and the card without. */
const absentViews = (() => {
  let cache;
  return () => (cache = cache || absentProbes().map(({ label, known, unknown }) => {
    const computed = computeInference(known), uc = computeInference(unknown);
    return { label, ks: known, us: unknown, kc: computed, uc,
             known: renderEverything(known, computed),
             moved: renderEverything(known, withMovedFigures(computed)),
             unknown: renderEverything(unknown) };
  }));
})();

/* The only text a card with no constants may show that the card with them does
   not: the reason, as each view words it, with the card's own name taken out.
   Reword a reason and this list moves with it — which is what makes the wording
   a contract rather than whatever the renderer happens to say that day. */
const REASON = [
  // renderThroughput's tile, where the three figures were
  "Throughput and TTFT Not modelled No measured utilisation for this hardware — has no published " +
  "memory-bandwidth or compute utilisation, and estimating its throughput or time to first token " +
  "would mean borrowing another architecture's constants, which do not transfer",
  // renderExecutiveSummary's row, where the two figures were
  'Speed and server throughput Not modelled · no measured utilisation for this hardware, and an ' +
  'estimate would borrow constants that do not transfer',
  // renderComparisons' row in a saved snapshot
  'Per-user / aggregate not modelled no measured utilisation for this hardware — borrowed constants ' +
  'do not transfer',
  // renderNotes
  'Throughput is not modelled for this hardware: no measured utilisation is published for it.',
  // renderStrategyBadges, where the PCIe loss percentage was
  'PCIe — no NVLink; the speed cost is not modelled for this hardware',
  // exportSummary, in place of its three lines
  '- Throughput and TTFT: not modelled.',
  'No measured utilisation is published for , and estimating either would mean borrowing another ' +
  "architecture's constants, which do not transfer.",
];

/* The only sentences a card with no constants may show that are not in the list
   above and not shown verbatim with them: a speed sentence with its speed clause
   taken off, exactly as the renderer writes it.

   A literal list, because "any subsequence of the words" is not a shortening —
   it is any sentence that can be spelled with those words in that order, and
   that includes sentences which keep the number and drop only the words the
   speech patterns key on. "PCIe 64-128 GB/s — 45-60% loss, growing with device
   count." is a word-subsequence of the sentence below it and says more, not
   less; so is "Throughput is a decode estimate." Each of these is still checked
   to be a shortening of a sentence that really stood in that view, so listing
   one cannot smuggle it into a view that never said the long form. */
const SHORTENED = [
  // renderThroughput's queue warning, with its pointer at the aggregate figure
  'vLLM will queue the rest.',
  // renderNotes' interconnect line, with the decode loss
  'PCIe 64-128 GB/s.',
  // renderExecutiveSummary's above-one-domain caveat, with its "ceiling to test"
  'Nothing above 2 devices is measured.',
];

test('every view that talks about throughput says why, in its place, when there is none to show', () => {
  /* Discovered, not enumerated: every element that prints a throughput or TTFT
     figure for the card with constants, or makes a claim about its speed, is a
     surface. Rendered again for the card without, each one must say throughput
     is not modelled. One that printed a figure must print none, and must give
     the reason — both halves, inside one piece, beside "not modelled" — with no
     number in that piece once the card's own name is out, so the reason stands
     where the figures stood and cannot carry one. Per piece and not per element,
     because renderThroughput writes several tiles into one element and a reason
     in one tile says nothing about the next. */
  const WHY = [/no measured utilisation/i, /do not transfer/i];
  const printed = new Set(), spoke = new Set();
  for (const { label, ks, us, known, unknown } of absentViews()) {
    for (const id of viewIds(known)) {
      const shown = seenText(viewText(known, id, ks), ks);
      const figures = FIGURE.test(shown);
      if (!figures && !speaks(shown)) continue;
      (figures ? printed : spoke).add(id);
      const reads = seenText(viewText(unknown, id, us), us);
      assert.match(reads, /not modelled/i,
        `${label}: ${id} talks about throughput for a card with constants, and does not say it is ` +
        `not modelled for one without: ${reads.slice(0, 300)}`);
      if (!figures) continue;
      const figure = reads.match(FIGURE);
      assert.ok(!figure, `${label}: ${id} prints "${figure && figure[0]}" for a card with no constants: ` +
        reads.slice(Math.max(0, (figure ? figure.index : 0) - 120), (figure ? figure.index : 0) + 40));
      const reasons = viewPieces(unknown, id, us).filter(piece => WHY.every(re => re.test(piece.raw)));
      assert.ok(reasons.length > 0, `${label}: ${id} prints no figure and does not say why: ${reads.slice(0, 300)}`);
      for (const piece of reasons) {
        assert.match(seenText(piece.raw, us), /not modelled/i,
          `${label}: ${id} gives the reason away from where the figures were: ${piece.raw.slice(0, 200)}`);
        const digits = numbersIn(piece.texts.join(' '));
        assert.deepStrictEqual(digits, [],
          `${label}: ${id} puts a number beside the reason: "${piece.texts.join(' ')}"`);
      }
    }
  }
  // Floors, not lists: a new surface is found and held to the same rule.
  assert.ok(printed.size >= 4 && printed.has('(copied report)'),
    `the sweep reached ${printed.size} surfaces with figures (${[...printed].join(', ')}) — it has stopped finding them`);
  assert.ok(spoke.has('notes-output') && spoke.has('strategy-badges'),
    `the views that only talk about speed were not reached: ${[...spoke].join(', ') || 'none'}`);
});

test('no view prints null, undefined, NaN or a throughput figure when there are no constants', () => {
  /* Every element, not only the discovered surfaces, and everything each one
     renders, writes or sets: a leaked figure is as wrong in a title as in the
     throughput panel. */
  /* Infinity as a value, not as the first word of AMD's interconnect: an OAM
     board's page names "Infinity Fabric" on purpose, and a numeric Infinity is
     never followed by it. */
  const BAD = new RegExp(String.raw`\bnull\b|\bundefined\b|\bNaN\b|\bInfinity\b(?! Fabric)|N\/A|` + FIGURE.source, 'i');
  for (const sample of ['~null ms', '~N/A ms', 'is undefined', 'NaN%', '~0 tok/s', '0 tokens/sec', 'Infinity',
                        '-Infinity GiB', 'Infinity GiB free',
                        '~0 ms', '~0 tokens per second', 'first token in ~0 milliseconds', '0 tok per sec',
                        '~147 t/s per user', '40 tokens each second'.replace('each second', 'per second')])
    assert.match(sample, BAD, `the pattern cannot see "${sample}"`);
  assert.doesNotMatch('Infinity Fabric — sharded 2-way', BAD, 'the pattern bans the name of an interconnect');
  let chars = 0, knownHits = 0;
  for (const { label, ks, us, known, unknown } of absentViews()) {
    for (const id of viewIds(unknown)) {
      const parts = [String(unknown.html[id] ?? ''), seenText(unknown.html[id], us),
                     String(unknown.written[id] ?? ''), ...Object.values(unknown.props[id] || {})];
      for (const text of parts) {
        const m = text.match(BAD);
        assert.ok(!m, `${label}: ${id} prints "${m && m[0]}" with no constants: ` +
          text.slice(Math.max(0, (m ? m.index : 0) - 100), (m ? m.index : 0) + 40));
        chars += text.length;
      }
    }
    knownHits += Object.values(known.html).filter(t => BAD.test(String(t))).length;
  }
  // And the same scan does see figures where they exist, so it is looking.
  assert.ok(knownHits > 0 && chars > 20000,
    `not discriminating: ${knownHits} known-card elements matched, ${chars} characters scanned`);
});

test('the KV-derived batch figures are shown for a card with no constants, by value', () => {
  /* How many sequences the KV cache fits is arithmetic on bytes: it needs no
     PERF constant and the ruling says it stays. The identity test cannot enforce
     that on its own — it lets a piece that carries a figure disappear, and a
     refactor that folds this tile into the aggregate tile, or into the TTFT
     tile, makes it part of a piece that does. The PDF pins its "Max batch at
     this context" row by value for exactly this reason; the page pinned nothing,
     so the same refactor would take the figure with it and leave a card with no
     constants showing no batch figure anywhere.

     Pinned by value, not by markup, so the tile can be reworded or moved. */
  let limited = 0, roomy = 0;
  for (const { label, us, uc, unknown } of absentViews()) {
    const shown = seenText(unknown.html['throughput-output'], us);
    assert.ok(shown.includes('Max batch at this context'),
      `${label}: the throughput panel of a card with no constants has no max-batch tile: ${shown}`);
    assert.ok(shown.includes(`${uc.maxBatchByKV} seq`),
      `${label}: the max-batch tile does not show ${uc.maxBatchByKV} seq: ${shown}`);
    if (uc.batchLimitedByKV) {
      limited++;
      // The queue warning is KV arithmetic too, and says the same number.
      assert.ok(shown.includes(`KV cache caps you below ${us.concurrency} requested`),
        `${label}: the tile does not say the cache caps the requested concurrency: ${shown}`);
      assert.ok(shown.includes(`KV cache only fits ${uc.maxBatchByKV} at`),
        `${label}: the queue warning is missing for a card with no constants: ${shown}`);
    } else {
      roomy++;
      assert.ok(shown.includes('KV cache has room for your concurrency'),
        `${label}: the tile does not say the cache has room: ${shown}`);
    }
  }
  assert.ok(limited > 0 && roomy > 0,
    `both branches have to be reached: ${limited} capped, ${roomy} with room`);
});

test('the benchmark panel is not drawn for a card without constants, even with measurements on file', () => {
  /* Scoring a measurement against an estimate that does not exist is the units
     error the panel was rebuilt to stop, and a bare reference point beside an
     empty tile invites the same comparison by hand. Every branch of the panel is
     reached with constants first: an exact batch match, the nearest size on the
     same card, a single-stream entry, an estimated entry, and a card with no data. */
  const h = renderHarness();
  for (const [params, gpuKey] of [[8, 'h100-80'], [14, 'b200-192'], [8, 'rtx4090-24'],
                                  [27, 'h100-80'], [8, 't4-16']]) {
    const known = asState(dualGCD, 1, { params, layers: 32, gpuKey });
    const unknown = { ...known, perfKey: 'no-such-key' };
    h.renderThroughput(known, h.computeInference(known));
    assert.match(h.out['throughput-output'], /benchmark/i,
      `control: ${params}B on ${gpuKey} shows no benchmark panel even with constants`);
    h.renderThroughput(unknown, h.computeInference(unknown));
    assert.doesNotMatch(h.out['throughput-output'], /benchmark/i,
      `${params}B on ${gpuKey}: the benchmark panel is drawn for a card with no constants`);
  }
});

test('without constants no view makes a speed claim, figure or not', () => {
  /* A figure is only half of it. A loss percentage, a "bandwidth-bound" or
     "compute-bound" label, an FP8 caveat about a ceiling, the observed band, or a
     sentence pointing at "the throughput figures" is a claim about speed as well,
     and each rests on the same heuristics the constants do. Every pattern must be
     seen on a card with constants, or it guards nothing. */
  const heard = new Set();
  for (const { label, ks, us, known, unknown } of absentViews()) {
    const withConstants = Object.values(known.html).join('\n');
    for (const [name, re] of Object.entries(SPEECH)) if (re.test(withConstants)) heard.add(name);
    for (const id of Object.keys(unknown.html)) {
      const parts = [String(unknown.html[id] ?? ''), String(unknown.written[id] ?? ''),
                     ...Object.values(unknown.props[id] || {})];
      for (const [name, re] of Object.entries(SPEECH)) {
        const m = parts.join('\n').match(re);
        assert.ok(!m, `${label}: ${id} makes ${name} with no constants: "${m && m[0]}"`);
      }
    }
  }
  assert.deepStrictEqual([...heard].sort(), Object.keys(SPEECH).sort(),
    'some of these never matched a card with constants, so they guard nothing');
});

test('without constants every view reads as it does with them, except where throughput was', () => {
  /* VRAM yes, throughput absent — and "absent" may not leak into anything else,
     in either direction.

     Six views may differ between a card with constants and the same card
     without: the throughput panel, the executive view, the comparison card, the
     notes, the badges and the copied report. Every other element the harness
     renders must be identical, in what it renders, in anything written to it
     through textContent, and in every attribute set on it. A new element is held
     to that by default.

     Inside the six, three rules:

     Nothing goes. A piece that shows a throughput figure — found by moving the
     engine's figures and seeing which pieces move, so no unit has to be
     recognised — may go, and so may the benchmark panel, and a sentence of speed
     talk may be dropped or shortened. Every other piece must survive verbatim,
     markup and attributes included, in order.

     Nothing new. Every text the card without constants shows — the sentences of
     each piece, the values of its title and data-* attributes, anything written
     through textContent — must be text the card with constants shows outside its
     figures, a shortening of one of its speed sentences, or one of the reason
     strings listed above. That is what catches a figure under a unit nobody
     listed, a number spelled as a word, and a claim with no number in it at all.

     And no new numbers: what the card without constants shows must be numbers the
     card with them shows outside its figures, in the same order. */
  const MAY_DIFFER = ['throughput-output', 'exec-summary', 'comparison-output',
                      'notes-output', 'strategy-badges', '(copied report)'];
  const differed = new Set(), held = new Set(), shortened = new Set();
  let survived = 0, checked = 0;
  for (const { label, ks, us, known, moved, unknown } of absentViews()) {
    assert.deepStrictEqual(Object.keys(unknown.html).sort(), Object.keys(known.html).sort(),
      `${label}: the two cards render different sets of elements`);
    for (const id of new Set([...Object.keys(known.html), ...Object.keys(known.written),
                              ...Object.keys(known.props), ...Object.keys(unknown.html),
                              ...Object.keys(unknown.written), ...Object.keys(unknown.props)])) {
      /* Before anything about the content: an element the reader cannot see
         holds nothing, whatever its innerHTML says. Applied to every element,
         the six allowed to differ included — withholding a whole panel is not
         one of the differences they are allowed. */
      const invisible = (v) => (v.props[id] || {}).hidden === 'true' ||
                               (v.props[id] || {})['style.display'] === 'none';
      assert.strictEqual(invisible(unknown), invisible(known),
        `${label}: ${id} is ${invisible(unknown) ? 'hidden' : 'shown'} for the card with no ` +
        `constants and ${invisible(known) ? 'hidden' : 'shown'} for the card with them`);
      if (!MAY_DIFFER.includes(id)) {
        assert.ok(unknown.html[id] === known.html[id],
          `${label}: ${id} changed when the constants went, ${firstDifference(known.html[id], unknown.html[id])}`);
        assert.strictEqual(String(unknown.written[id] ?? ''), String(known.written[id] ?? ''),
          `${label}: ${id} had different text written to it when the constants went`);
        assert.deepStrictEqual(unknown.props[id] || {}, known.props[id] || {},
          `${label}: ${id} had different attributes set on it when the constants went`);
        held.add(id);
        continue;
      }
      if (unknown.html[id] !== known.html[id]) differed.add(id);
      const pieces = viewPieces(known, id, ks), movedPieces = viewPieces(moved, id, ks);
      assert.strictEqual(pieces.length, movedPieces.length,
        `${label}: moving the throughput figures changed how ${id} is laid out`);
      const kept = [], spoken = [], outsideFigures = [];
      pieces.forEach((piece, i) => {
        /* A piece may go when it shows one of the figures — either under a unit
           listed above, or by printing a number that moved when the figures did.
           Not merely because something about it changed: a piece given a
           constants-dependent space, or comma, reads differently under moved
           figures while showing no figure at all, and that was enough to buy an
           exemption from surviving. The number is the evidence. */
        const printed = numbersIn(piece.texts.join(' ')).join(' ');
        const moves = printed !== numbersIn(movedPieces[i].texts.join(' ')).join(' ');
        if (moves || piece.texts.some(t => FIGURE.test(t)) ||
            piece.texts.some(t => /benchmark/i.test(t))) return;
        outsideFigures.push(...piece.texts);
        piece.texts.filter(speaks).forEach(t => spoken.push(t));
        if (piece.texts.some(speaks)) kept.push({ sentences: piece.texts.filter(t => !speaks(t)) });
        else kept.push({ raw: piece.raw });
      });
      const after = viewPieces(unknown, id, us);
      const reads = seenText(unknown.html[id], us);
      const rawAfter = String(unknown.html[id] ?? '');
      let atRaw = 0, atText = 0;
      for (const unit of kept) {
        if (unit.raw !== undefined) {
          const found = rawAfter.indexOf(unit.raw, atRaw);
          assert.ok(found >= 0,
            `${label}: ${id} lost, or reworded, "${seenText(unit.raw, ks).slice(0, 120)}" when the constants went`);
          atRaw = found + unit.raw.length;
          survived++;
          continue;
        }
        for (const sentence of unit.sentences) {
          const found = reads.indexOf(sentence, atText);
          assert.ok(found >= 0, `${label}: ${id} lost "${sentence}" when the constants went`);
          atText = found + sentence.length;
          survived++;
        }
      }
      for (const piece of after)
        for (const text of piece.texts) {
          checked++;
          const isShortening = SHORTENED.includes(text) &&
            spoken.some(sentence => isShorteningOf(text, sentence));
          if (isShortening) shortened.add(text);
          const accounted = outsideFigures.includes(text) || REASON.includes(text) || isShortening;
          assert.ok(accounted,
            `${label}: ${id} shows text for a card with no constants that the card with them does ` +
            `not show outside its figures, and that is neither a reason string nor a listed ` +
            `shortening: "${text}"`);
        }
      const shows = numbersIn(reads), may = numbersIn(outsideFigures.join(' '));
      assert.ok(inOrderWithin(shows, may),
        `${label}: ${id} shows numbers without constants that it only shows beside figures with them — ` +
        `[${shows.join(' ')}] is not within [${may.join(' ')}]: ${reads.slice(0, 400)}`);
    }
  }
  // Both halves have to be reached: every view allowed to differ does, and the
  // rest of the page is really being held still.
  assert.deepStrictEqual([...differed].sort(), [...MAY_DIFFER].sort(),
    `allowed to differ, and never did: ${MAY_DIFFER.filter(id => !differed.has(id)).join(', ')}`);
  for (const id of ['gpu-cards', 'metrics-output', 'verdict-output', 'capacity-output', 'cost-output',
                    'command-output', 'training-results',
                    // and the four surfaces the hand-written renderer list had missed
                    'gpu-model', 'gpu-count-display', 'counting-toggle', 'url-restore-warning'])
    assert.ok(held.has(id), `${id} was never rendered, so it was never held identical`);
  assert.deepStrictEqual([...shortened].sort(), SHORTENED.slice().sort(),
    'a listed shortening is never emitted where the sentence it shortens stood, so it excuses nothing');
  assert.ok(survived > 500 && checked > 500,
    `only ${survived} pieces were required to survive and ${checked} texts were accounted for`);
});

test('the VRAM breakdown row adds up to the total it prints beside it', () => {
  /* A reader adds a breakdown up. Before the weights divisor was corrected this
     row did add up — Weights + KV + Act+OH was exactly Total, because nothing
     replicated — and correcting the divisor broke it: at 12 devices it showed
     "Weights 130 GiB · KV 2.50 · Act+OH 22.6 · Total 419", a 263 GiB gap with
     no caption. One screen contradicting itself is the defect class this branch
     exists to close, so the relation is pinned on the rendered strings rather
     than on the arithmetic behind them, which would be a tautology.

     The tolerance is the formatter's own, not a guess: formatGB rounds to the
     integer above 100, one decimal above 10 and two below, so each printed
     figure carries at most half of its last digit and the sum carries the sum
     of those. */
  const h = renderHarness();
  const grain = (v) => (v >= 100 ? 0.5 : v >= 10 ? 0.05 : 0.005);
  let withCopies = 0, singleCopy = 0;
  for (const count of [1, 2, 4, 8, 9, 12, 16, 64, 128])
    for (const activePercent of [100, 5])
      for (const card of [GPU_TABLE['h100-80'], dualGCD]) {
        const st = asState(card, count, { params: 70, layers: 80, activePercent });
        const c = h.computeInference(st);
        h.renderMetrics(c);
        const cells = [...(h.out['metrics-output'] || '').matchAll(
          /<p class="metric-label">(.*?)<\/p><p class="metric-value">(.*?)<\/p>/g)]
          .map(m => [m[1], m[2]]);
        assert.strictEqual(cells.length, 6, `expected six metric cards, got ${cells.length}`);
        const num = (t) => Number(String(t).replace(/[^0-9.]/g, ''));
        const [wLabel, wVal] = cells[0], [, kvVal] = cells[1];
        const [, aoVal] = cells[2], [, tVal] = cells[3];
        const parts = [num(wVal), num(kvVal), num(aoVal)];
        const total = num(tVal);
        const tol = parts.reduce((a, v) => a + grain(v), grain(total));
        assert.ok(Math.abs(parts[0] + parts[1] + parts[2] - total) <= tol,
          `${count}x ${card.name} (${c.deviceCount} devices, ${c.modelCopies} copies): ` +
          `${wVal} + ${kvVal} + ${aoVal} = ${parts[0] + parts[1] + parts[2]}, but the row ` +
          `prints ${tVal} beside them`);
        // And the copy count is stated wherever it is not one, so the weights
        // figure is explained rather than merely made to add up.
        if (c.modelCopies > 1) {
          withCopies++;
          assert.strictEqual(wLabel, `Weights (${c.modelCopies} copies)`,
            `weights tile is labelled "${wLabel}" on a ${c.modelCopies}-copy cluster`);
        } else {
          singleCopy++;
          assert.strictEqual(wLabel, 'Weights', `weights tile is labelled "${wLabel}" at one copy`);
        }
      }
  assert.ok(withCopies > 0 && singleCopy > 0,
    `only reached one regime: ${withCopies} replicated, ${singleCopy} single-copy`);
});
test('the board count the tool recommends is one its own arithmetic agrees with', () => {
  /* This used to pin ceil(totalGB / (deviceGB * 0.9)) as correct — a cluster
     total divided by one device's capacity. That formula is circular: buying
     boards changes the split, so the answer describes a machine that stops
     existing the moment the reader acts on it. It does not merely round badly,
     it diverges. On a 70B at AWQ over nine RTX 4090s it said 15, and re-asking
     the tool at 15 said 25, then 42. On a 123B at bf16 over T4s it said 52
     where no board count ever fits, then 216, 460, 1903.

     So the property is stated as what a reader can act on: whatever number the
     banner prints, recomputing the whole configuration at that number must fit
     — and where nothing fits, the banner has to say so instead of printing a
     number. That is checkable without knowing the formula, which is the point;
     the old test could only ever agree with whatever the formula did.

     Board counts and multi-device boards both, because the recommendation is
     in boards while the split is sized in devices, and a version that reads the
     per-board divisor only at gpuCount === 1 doubles the purchase advice for
     everyone else. */
  const harness = renderHarness();
  const probes = [];
  for (const devices of [1, 2, 4])
    for (const boards of [1, 2, 4, 9])
      for (const [params, layers, bpp, gb] of [[400, 126, 2, 128], [70, 80, 0.5, 24],
                                               [123, 88, 2, 16], [8, 32, 2, 80]])
        probes.push(asState({ ...dualGCD, gb, devices }, boards,
                            { params, layers, bytesPerParam: bpp }));
  let recommended = 0, impossible = 0, alreadyFits = 0;
  for (const state of probes) {
    const c = harness.computeInference(state);
    if (c.fits) { alreadyFits++; continue; }
    const { boards, capped } = harness.boardsNeeded(state, c);
    const advice = harness.boardsAdvice(state, c);
    if (boards === null) {
      impossible++;
      assert.ok(!/\d+ boards/.test(advice),
        `no board count fits, but the banner still names one: ${advice}`);
      assert.match(advice, /No (?:number of these boards|board count)/,
        `the banner must say plainly that nothing fits: ${advice}`);
      // And it must be true, not merely unproven: the largest cluster the
      // search will consider does not fit either.
      assert.strictEqual(capped, false, 'a capped search must not be reported as impossible');
      continue;
    }
    recommended++;
    // The claim under test. Recompute the entire configuration at the count the
    // banner names and ask the engine, not the formula.
    const at = harness.computeInference({ ...state, gpuCount: boards });
    assert.ok(at.fits,
      `banner says "${advice}" but ${boards} boards recomputes to ` +
      `${at.perGPU.total} GiB per device against ${at.deviceGB} (TP=${at.tp} x DP=${at.dp})`);
    // Smallest, not merely sufficient: everything below it must genuinely fail,
    // which is also what stops a bisection being slipped in over a fits curve
    // that is not monotonic.
    for (let n = 1; n < boards; n++)
      assert.ok(!harness.computeInference({ ...state, gpuCount: n }).fits,
        `${boards} boards was recommended but ${n} already fits`);
    assert.ok(advice.includes(`Smallest fit: ${boards} boards`),
      `advice must name the count it found: ${advice}`);
    /* And it must not claim more than it found. "Need N+ boards" asserted that
       every larger count also fits, which is false wherever the split gets
       worse — 8B bf16 on T4s fits at 2 and fails at 9, where TP drops to 1 and
       every device needs a whole copy. Pinned against the fit vector rather
       than against the wording, so a "+" can only come back on a configuration
       where it is actually true. */
    const plus = advice.match(/(\d+)\+ boards/);
    if (plus) {
      const from = Number(plus[1]);
      for (let n = from; n <= from + 24; n++)
        assert.ok(harness.computeInference({ ...state, gpuCount: n }).fits,
          `advice says "${from}+ boards" but ${n} boards does not fit ` +
          `(TP=${harness.computeInference({ ...state, gpuCount: n }).tp})`);
    }
  }
  // Both outcomes have to be exercised or this is only testing one of them.
  assert.ok(recommended > 0 && impossible > 0,
    `probes did not reach both outcomes: ${recommended} recommended, ${impossible} ` +
    `impossible, ${alreadyFits} already fitting`);
  /* And the two surfaces that print it must print the same sentence. The exec
     view used to re-word the recommendation around its own copy of the number,
     which is how one screen can recommend different hardware from another. */
  const notFitting = probes.find(st => !harness.computeInference(st).fits);
  const c2 = harness.computeInference(notFitting);
  harness.renderVerdict(notFitting, c2);
  // The harness reports exec-mode on, which is why renderExecutiveSummary
  // writes anything at all here.
  harness.renderExecutiveSummary(notFitting, c2);
  const sentence = harness.boardsAdvice(notFitting, c2);
  for (const id of ['verdict-output', 'exec-summary'])
    assert.ok((harness.out[id] || '').includes(sentence),
      `${id} does not carry the shared recommendation "${sentence}"`);
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
test('the page hands the engine each tier\'s price exactly as the catalog records it, null included', () => {
  /* The null-tier tests build their states by hand, so nothing read what
     getGpuSpec() and readInputState() pass on. A `spot: g.spot ?? g.spec`
     there printed the specialized price under Spot on the live page with every
     test green, and `hyper: g.hyper ?? 0` printed $0.00 (cold check, round 1).
     Driven through the real state builder, for every row and every tier —
     priced or null — and the provenance each tier carries. */
  let nulls = 0, priced = 0, noted = 0;
  for (const [key, gpu] of Object.entries(GPU_TABLE)) {
    const st = readInputStateFor(key, '1');
    for (const [tier, field] of [['hyper', 'gpuHyperCost'], ['spec', 'gpuSpecCost'], ['spot', 'gpuSpotCost']]) {
      assert.strictEqual(st[field], gpu[tier], `${key}: the state's ${field} is ${st[field]}, the catalog's ${tier} is ${gpu[tier]}`);
      if (gpu[tier] === null) nulls++; else priced++;
    }
    assert.deepStrictEqual(st.priceSource, gpu.priceSource, `${key}: the state's priceSource`);
    assert.deepStrictEqual(st.priceRecord, gpu.priceRecord, `${key}: the state's priceRecord`);
    assert.deepStrictEqual(st.priceNote, gpu.priceNote, `${key}: the state's priceNote`);
    assert.deepStrictEqual(st.priceLead, gpu.priceLead, `${key}: the state's priceLead`);
    assert.strictEqual(st.gfx, gpu.gfx, `${key}: the state's gfx`);
    if (gpu.priceNote) noted++;
  }
  assert.ok(nulls > 0 && priced > 0, `the catalog gave ${nulls} null and ${priced} priced tiers — this needs both`);
  assert.ok(noted > 0, 'no catalog row carries a priceNote, so the state was never shown one');
});
test('the state carries the perfKey its constants are chosen by, for every row', () => {
  /* Deleting perfKey from readInputState() — or from getGpuSpec(), which it reads
     — leaves the page with no constants for any card. Driven through the real
     state builder, row by row, so neither link can go missing unnoticed. */
  for (const [key, gpu] of Object.entries(GPU_TABLE)) {
    const state = readInputStateFor(key, '1');
    assert.strictEqual(typeof gpu.perfKey, 'string', `${key}: the catalog row has no perfKey`);
    assert.strictEqual(state.perfKey, gpu.perfKey,
      `${key}: state carries perfKey=${state.perfKey}, the catalog says ${gpu.perfKey}`);
    // vendor still rides along — it selects nothing now, but it is the card's.
    assert.strictEqual(state.vendor, gpu.vendor, `${key}: state.vendor`);
    // form too: it names the link every interconnect surface prints.
    assert.strictEqual(state.gpuForm, gpu.form, `${key}: state.gpuForm`);
  }
  /* And a row whose two fields differ. On every real row both are 'nvidia', so
     a state builder that filled perfKey from vendor passed the loop above — a
     cold sabotage run showed exactly that. */
  const table = { ...GPU_TABLE,
    'probe-row': { ...GPU_TABLE['h100-80'], vendor: 'acme', perfKey: 'acme-arch1' } };
  const probe = readInputStateFor('probe-row', '1', table);
  assert.strictEqual(probe.perfKey, 'acme-arch1',
    `a row with perfKey acme-arch1 reached the state as ${probe.perfKey}`);
  assert.strictEqual(probe.vendor, 'acme', `a row with vendor acme reached the state as ${probe.vendor}`);
  // No catalog row is an OAM board yet, so the form's other value comes in on a probe.
  const oam = readInputStateFor('probe-oam', '1', { ...GPU_TABLE, 'probe-oam': { ...GPU_TABLE['b200-192'], form: 'oam' } });
  assert.strictEqual(oam.gpuForm, 'oam', `a row with form oam reached the state as ${oam.gpuForm}`);
  assert.strictEqual(oam.hasNVLink, false, 'an OAM board was granted NVLink');
});

console.log('\nA price tier with no confirmed price');
test('a price tier with no confirmed price stays null, and every cost surface says so', () => {
  /* A tier the catalog records as null: no provider's own page confirmed an
     hourly price for this card in this tier. null * gpuCount is 0 in
     JavaScript, so the failure this guards is a free cluster, printed as
     $0.00 — or a neighbouring tier's price standing in the empty one's row.
     Every non-empty set of null tiers, one board and three, read off what the
     page prints rather than off computeInference() alone. The card carries
     h100-80's provenance, so a source attached to a null tier would show. */
  const NO_PRICE = 'no confirmed hourly price';
  const TIERS = [['hyper', 'Hyperscaler', 'hourlyHyper', 'Hyperscaler'],
                 ['spec', 'Specialized', 'hourlySpec', 'Specialized'],
                 ['spot', 'Spot / marketplace', 'hourlySpot', 'Spot']];
  const base = GPU_TABLE['h100-80'];
  let checked = 0;
  for (let mask = 1; mask < 8; mask++)
    for (const count of [1, 3]) {
      const card = { ...base };
      const nulls = TIERS.filter((_, i) => mask & (1 << i)).map(t => t[0]);
      for (const t of nulls) card[t] = null;
      const label = `null ${nulls.join('+')}, ${count} board${count > 1 ? 's' : ''}`;
      const st = asState(card, count, { params: 8, layers: 32 });
      const c = computeInference(st);
      const { html: out, written, props } = renderEverything(st, c);
      const cost = out['cost-output'];
      const rows = cost.split('</tr>');
      const priced = [];
      for (const [tier, rowLabel, field, lineLabel] of TIERS) {
        const row = rows.find(r => r.includes(`<b>${rowLabel}</b>`));
        assert.ok(row, `${label}: the cost table has no ${rowLabel} row`);
        const line = out['(copied report)'].split('\n').find(l => l.startsWith(`- ${lineLabel}: `));
        assert.ok(line, `${label}: the copied report has no ${lineLabel} line`);
        if (nulls.includes(tier)) {
          assert.strictEqual(c[field], null, `${label}: ${field} is ${c[field]}, not null`);
          assert.ok(row.includes(NO_PRICE), `${label}: the ${rowLabel} row does not say "${NO_PRICE}"`);
          assert.ok(!row.includes('$'), `${label}: the ${rowLabel} row prints a dollar figure: ${row.slice(-200)}`);
          assert.strictEqual(line, `- ${lineLabel}: ${NO_PRICE}`, `${label}: the copied report's ${lineLabel} line`);
        } else {
          assert.strictEqual(c[field], card[tier] * count, `${label}: ${field}`);
          assert.ok(row.includes(`$${card[tier].toFixed(2)}`) && row.includes(`$${(card[tier] * count).toFixed(2)}`),
            `${label}: the ${rowLabel} row does not print its own price`);
          assert.ok(line.includes(`$${(card[tier] * count).toFixed(2)}/hr`), `${label}: ${line}`);
          priced.push(card[tier] * count);
        }
      }
      const range = priced.length === 0 ? [NO_PRICE, NO_PRICE]
        : priced.length === 1 ? [`$${priced[0].toFixed(2)}`, `$${Math.round(priced[0] * 730).toLocaleString()}/mo`]
        : [`$${Math.min(...priced).toFixed(2)}–$${Math.max(...priced).toFixed(2)}`,
           `$${Math.round(Math.min(...priced) * 730).toLocaleString()} – $${Math.round(Math.max(...priced) * 730).toLocaleString()}/mo`];
      assert.ok(out['comparison-output'].includes(`Cost/hr</span><span class="val">${range[0]}</span>`),
        `${label}: the snapshot's Cost/hr is not ${range[0]}`);
      assert.ok(out['exec-summary'].includes(`Monthly cost range</span><span class="exec-value">${range[1]}</span>`),
        `${label}: the Monthly cost range is not ${range[1]}`);
      const text = [...Object.values(out), ...Object.values(written),
                    ...Object.values(props).flatMap(p => Object.values(p))].join('\n');
      for (const bad of ['$0.00', '$0/mo', '$NaN', 'NaN', 'undefined', '$null', 'null/hr'])
        assert.ok(!text.includes(bad), `${label}: the page prints "${bad}"`);
      if (nulls.includes('hyper'))
        assert.ok(!text.includes(base.priceSource.hyper.sku), `${label}: a null hyper tier still names its source`);
      // The notes explain the wording, and the spot figure is highlighted only when there is one.
      assert.ok(out['notes-output'].includes('"No confirmed hourly price" marks a tier'),
        `${label}: the notes do not explain the wording a null tier shows`);
      const spotRow = rows.find(r => r.includes('<b>Spot / marketplace</b>'));
      assert.strictEqual(spotRow.includes('var(--success)'), !nulls.includes('spot'),
        `${label}: the spot monthly cell is highlighted ${nulls.includes('spot') ? 'over a dash' : 'nowhere'}`);
      checked++;
    }
  assert.strictEqual(checked, 7 * 2, 'not every null pattern was rendered');
  // And a card that prices every tier says nothing about a wording it never shows.
  const priced = renderEverything(asState(base, 1, { params: 8, layers: 32 })).html;
  assert.ok(!priced['notes-output'].includes('No confirmed hourly price'),
    'a card with every tier priced explains a wording it never shows');
});

console.log('\nThe interconnect control follows the card');
/* Every value the catalog's `form` may take — a contract, so a literal, and the
   one list the structural-fields check and the naming checks below both walk.
   tests/report.test.py and tests/parity.test.py carry the same four. */
const FORMS = ['sxm', 'pcie', 'consumer', 'oam'];
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
test('an OAM board offers its own fabric in the control, and a PCIe card gets "PCIe only" back', () => {
  const table = { ...GPU_TABLE, 'probe-oam': { ...GPU_TABLE['b200-192'], form: 'oam' } };
  const dom = domStub('probe-oam', '1');
  dom.fields['interconnect'].options = [{ value: '1', disabled: false, textContent: 'NVLink / NVSwitch' },
                                        { value: '0', disabled: false, textContent: 'PCIe only' }];
  const src = html.slice(html.indexOf('let interconnectForcedToPCIe'), html.indexOf('function recalculate'));
  const sync = new Function('document', 'GPU_TABLE', `${nvDecl[0]}\n${src}; return syncInterconnect;`)(dom, table);
  sync();
  const sel = dom.fields['interconnect'];
  assert.strictEqual(sel.value, '0', 'an OAM board was left on the NVLink option');
  assert.strictEqual(sel.options[0].disabled, true, 'NVLink stayed selectable on an OAM board');
  assert.strictEqual(sel.options[1].textContent, 'Infinity Fabric',
    `the option an OAM board falls back to reads ${JSON.stringify(sel.options[1].textContent)}`);
  dom.fields['gpu-model'].value = 'rtx4090-24';
  sync();
  assert.strictEqual(sel.options[1].textContent, 'PCIe only', 'a PCIe card kept the OAM board\'s label');
});
test('every surface names the link the devices actually talk over, on every form', () => {
  /* What the renderers print, not what interconnectName() returns: the NVLink
     gate was once pinned by its predicate while both places that called it went
     unread. Every form the catalog allows, NVLink asked for and not, one domain
     and past it, with constants and without. */
  const seen = new Set();
  let checked = 0;
  for (const form of FORMS)
    for (const count of [2, 16])
      for (const asked of [true, false])
        for (const perfKey of ['nvidia', 'no-such-key']) {
          const card = { ...GPU_TABLE['b200-192'], form, perfKey };
          const hasNVLink = supportsNVLink(card) && asked;
          const want = hasNVLink ? 'NVLink' : form === 'oam' ? 'Infinity Fabric' : 'PCIe';
          const { html: out, written, props } = renderEverything(asState(card, count, { hasNVLink }));
          const text = [...Object.values(out), ...Object.values(written),
                        ...Object.values(props).flatMap(p => Object.values(p))].join('\n');
          const label = `${form} x${count}, NVLink ${asked ? 'asked for' : 'not asked for'}, perfKey ${perfKey}`;
          assert.ok(out['gpu-cards'].includes(`${want} — `), `${label}: the sharding line does not name ${want}`);
          assert.ok(out['(copied report)'].includes(`(${want})`), `${label}: the copied report does not name ${want}`);
          if (form === 'oam') {
            // Neither of the other two links exists on an OAM board, under any wording.
            for (const other of ['PCIe', 'NVLink'])
              assert.ok(!text.includes(other), `${label}: an OAM board's page mentions ${other}: ` +
                JSON.stringify(text.slice(Math.max(0, text.indexOf(other) - 80), text.indexOf(other) + 40)));
          } else {
            assert.ok(!text.includes('Infinity Fabric'), `${label}: a ${form} board's page mentions Infinity Fabric`);
          }
          seen.add(want);
          checked++;
        }
  assert.deepStrictEqual([...seen].sort(), ['Infinity Fabric', 'NVLink', 'PCIe'],
    'the grid never reached one of the three links, so it checks nothing about it');
  assert.strictEqual(checked, FORMS.length * 2 * 2 * 2);
  /* And every real row, at several boards: the probes above are one-device
     boards, so the MI250X — two devices a board — was never named above one
     board, and a gate keyed on devices and count at once passed (cold check,
     round 1). Its own name, form, devices and constants, NVLink as the page
     grants it. */
  let rows = 0;
  for (const [key, row] of Object.entries(GPU_TABLE))
    for (const count of [2, 3, 16])
      for (const asked of [true, false]) {
        const hasNVLink = supportsNVLink(row) && asked;
        const want = hasNVLink ? 'NVLink' : row.form === 'oam' ? 'Infinity Fabric' : 'PCIe';
        const { html: out } = renderEverything(asState(row, count, { hasNVLink }));
        const label = `${key} x${count}, NVLink ${asked ? 'asked for' : 'not asked for'}`;
        assert.ok(out['gpu-cards'].includes(`${want} — `), `${label}: the sharding line does not name ${want}`);
        assert.ok(out['(copied report)'].includes(`(${want})`), `${label}: the copied report does not name ${want}`);
        rows++;
      }
  assert.ok(Object.values(GPU_TABLE).some(r => r.form === 'oam' && r.devices > 1),
    'no real row is a multi-device OAM board, so the case that failed is not in the grid');
  assert.strictEqual(rows, Object.keys(GPU_TABLE).length * 3 * 2);
});
test('a deliberate PCIe choice on an SXM card is not overridden', () => {
  const sel = syncFor('h100-80', '0');
  assert.strictEqual(sel.value, '0', 'the reader chose PCIe; leave it alone');
  assert.strictEqual(sel.options[0].disabled, false);
});

console.log('\nThe GPU dropdown groups by vendor');
test('the GPU dropdown groups cards by vendor once the catalog has more than one', () => {
  /* ROADMAP promises the dropdown an AMD section. One vendor keeps the flat
     list the page always had; several get a section each. Run against a probe
     catalog whose vendors are interleaved, so "group by vendor" cannot pass by
     the rows happening to arrive already grouped, and with an id VENDOR_NAMES
     does not know. */
  const decl = (re, what) => { const m = html.match(re); assert.ok(m, `${what} not found in index.html`); return m[0]; };
  const src = [decl(/^const VENDOR_NAMES = .+;$/m, 'VENDOR_NAMES'),
               decl(/^const DEFAULT_GPU_KEY = .+;$/m, 'DEFAULT_GPU_KEY'),
               decl(/^function renderGpuOptions\(\) \{[\s\S]*?\n\}$/m, 'renderGpuOptions()')].join('\n');
  const render = (table) => {
    const sel = { innerHTML: '' };
    new Function('document', 'GPU_TABLE', `${src}; renderGpuOptions();`)({ getElementById: () => sel }, table);
    return sel.innerHTML;
  };
  const keysIn = (h) => [...h.matchAll(/<option value="([^"]+)"/g)].map(m => m[1]);

  const real = render(GPU_TABLE);
  assert.deepStrictEqual(keysIn(real).sort(), Object.keys(GPU_TABLE).sort(), 'the real catalog lost or repeated a card');
  if (new Set(Object.values(GPU_TABLE).map(g => g.vendor)).size === 1) {
    assert.ok(!real.includes('<optgroup'), 'a one-vendor catalog was given section headings');
    assert.deepStrictEqual(keysIn(real), Object.keys(GPU_TABLE), 'a one-vendor catalog was reordered');
  }

  const entries = Object.entries(GPU_TABLE).filter(([, g]) => g.vendor === 'nvidia');
  const probe = Object.fromEntries([
    ...entries.slice(0, 3),
    ['probe-amd-a', { ...GPU_TABLE['h100-80'], vendor: 'amd', name: 'Probe A 80 GB', default: undefined }],
    ...entries.slice(3, 6),
    ['probe-acme', { ...GPU_TABLE['h100-80'], vendor: 'acme', name: 'Probe C 80 GB', default: undefined }],
    ['probe-amd-b', { ...GPU_TABLE['h100-80'], vendor: 'amd', name: 'Probe B 80 GB', default: undefined }],
    ...entries.slice(6),
  ]);
  const grouped = render(probe);
  const groups = [...grouped.matchAll(/<optgroup label="([^"]+)">([\s\S]*?)<\/optgroup>/g)]
    .map(m => [m[1], keysIn(m[2])]);
  assert.deepStrictEqual(groups.map(g => g[0]), ['NVIDIA', 'AMD', 'acme'],
    'sections are not one per vendor in first-listed order, under their names');
  for (const [label, keys] of groups) {
    const vendor = { NVIDIA: 'nvidia', AMD: 'amd', acme: 'acme' }[label];
    assert.deepStrictEqual(keys, Object.keys(probe).filter(k => probe[k].vendor === vendor),
      `the ${label} section does not hold exactly its cards in catalog order`);
  }
  assert.deepStrictEqual(keysIn(grouped).sort(), Object.keys(probe).sort(), 'a card was lost or listed twice');
  assert.strictEqual((grouped.match(/ selected/g) || []).length, 1, 'the default card is not selected exactly once');
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
    assert.strictEqual(typeof gpu.perfKey, 'string', `${key}.perfKey`);
    assert.strictEqual(typeof gpu.devices, 'number', `${key}.devices`);
    assert.ok(FORMS.includes(gpu.form), `${key}.form=${gpu.form}`);
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

console.log('\nThe documents describe the tool that exists');
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

/* README.md and ROADMAP.md make claims a reader is invited to act on, and every one of
   them restates something the code already knows. Restated facts drift: the ROADMAP
   listed 8 of 12 catalogued cards, and the README's file listing named 9 of 21 tracked
   files. Neither was caught by anything, because prose is not executed. These pins
   derive from the source in every case — a literal here would be a third copy. */
const readmeDoc = fs.readFileSync(path.join(ROOT, 'README.md'), 'utf8');
const roadmapDoc = fs.readFileSync(path.join(ROOT, 'ROADMAP.md'), 'utf8');

test('ROADMAP.md carries no second copy of the GPU catalog', () => {
  // Same defect as the CONTRIBUTING pin above, and the reason this one exists: the
  // AMD rows land in a later commit and would silently falsify any list written here.
  const copied = Object.keys(GPU_TABLE).filter(k =>
    roadmapDoc.includes(`\`${k}\``) || new RegExp(`\\b${k}\\b`).test(roadmapDoc));
  assert.strictEqual(copied.length, 0,
    `ROADMAP.md hardcodes catalog keys: ${copied.join(', ')} — point at data/gpus.json instead`);
});

test('the benchmark counts README quotes match the data', () => {
  const m = readmeDoc.match(/\*\*Benchmark data\*\* is (\d+) entries: (\d+) measured[^,]*, (\d+) extrapolated/);
  assert.ok(m, 'README.md no longer states the benchmark counts in the expected shape');
  const [total, measured, estimated] = m.slice(1).map(Number);
  const keys = Object.keys(benchmarks.data);
  const flagged = keys.filter(k => benchmarks.data[k].estimated).length;
  assert.strictEqual(total, keys.length, `README says ${total} entries, data has ${keys.length}`);
  assert.strictEqual(estimated, flagged, `README says ${estimated} extrapolated, data flags ${flagged}`);
  assert.strictEqual(measured, keys.length - flagged,
    `README says ${measured} measured, data has ${keys.length - flagged}`);
});

test('the README is honest about which entries are multi-GPU', () => {
  /* Added after a cold review: the first version of this paragraph said "all 13
     benchmark entries are single-GPU", and one of them is 2xH100 at TP=2. The claim
     that matters is not how many devices appear in the dataset but whether any
     *measured* entry uses more than one, because that is what would validate the
     multi-GPU model. Both halves are derived from the notes, never enumerated. */
  const devicesIn = (note = '') => Math.max(1,
    ...[...note.matchAll(/\b(\d+)\s*[x×]\s*[A-Za-z]/g)].map(m => Number(m[1])),
    ...[...note.matchAll(/TP\s*=\s*(\d+)/gi)].map(m => Number(m[1])));

  const entries = Object.entries(benchmarks.data);
  const single = entries.filter(([, b]) => devicesIn(b.note) === 1);
  const multi = entries.filter(([, b]) => devicesIn(b.note) > 1);

  const m = readmeDoc.match(/(\d+) of the (\d+) benchmark entries are\s+single-GPU/);
  assert.ok(m, 'README.md no longer states how many entries are single-GPU');
  assert.strictEqual(Number(m[1]), single.length,
    `README says ${m[1]} single-GPU entries, the notes describe ${single.length}`);
  assert.strictEqual(Number(m[2]), entries.length,
    `README says ${m[2]} entries, data has ${entries.length}`);

  // The load-bearing half: a multi-GPU entry that is not flagged estimated would make
  // "no measured entry uses more than one GPU" false, and it is the sentence's claim.
  for (const [key, b] of multi) {
    assert.ok(b.estimated,
      `${key} is a ${devicesIn(b.note)}-device entry and is not flagged estimated — ` +
      `the README claims no measured entry uses more than one GPU`);
  }

  /* And the sentence has to keep tracking that fact in both directions. Checking the
     data alone leaves the claim itself free to invert: a second cold pass rewrote this
     lead to "some measured entries use multiple GPUs", left the counts correct, and the
     assertions above stayed green. Same shape as the --device pin — the sentence is
     required to say what the data says, not merely to exist. */
  const measuredMulti = multi.filter(([, b]) => !b.estimated).map(([k]) => k);
  const claimsNone = /No measured entry uses more than one GPU/.test(readmeDoc);
  assert.strictEqual(claimsNone, measuredMulti.length === 0, measuredMulti.length === 0
    ? 'no measured entry uses more than one GPU, and the README no longer says so'
    : `${measuredMulti.join(', ')} measure on more than one GPU, and the README still claims none do`);
});

test('the constants README quotes are the constants the model uses', () => {
  /* The band is the one that moves: tests above re-derive obsLo/obsHi from the
     measured entries, so a contributed benchmark can shift it and leave the README
     quoting last year's spread. The rest are quoted in a single parenthetical and
     were correct when written — this is what keeps them that way. */
  const { mbu, mfuDecode, mfuPrefill, obsLo, obsHi } = PERF.nvidia;
  const quoted = (re, what) => {
    const m = readmeDoc.match(re);
    assert.ok(m, `README.md no longer states ${what} in the expected shape`);
    return m.slice(1).map(Number);
  };
  assert.deepStrictEqual(quoted(/roofline at (\d+)% MBU/, 'MBU'), [mbu * 100]);
  assert.deepStrictEqual(quoted(/compute roofline at (\d+)% MFU/, 'decode MFU'), [mfuDecode * 100]);
  assert.deepStrictEqual(quoted(/separate (\d+)% prefill MFU/, 'prefill MFU'), [mfuPrefill * 100]);
  assert.deepStrictEqual(quoted(/observed range \((\d+)–(\d+)% of ceiling\)/, 'the observed band'),
    [Math.round(obsLo * 100), Math.round(obsHi * 100)]);

  // The overhead trio, read out of index.html rather than restated here.
  const actPct = Number(html.match(/const activationsGB = Math\.max\(\(totalActiveParams \* 1e9 \* 2 \* ([\d.]+)\)/)[1]) * 100;
  const ctxGiB = Number(html.match(/const overheadPerGPU = ([\d.]+);/)[1]);
  const [hi, lo] = html.match(/const peerBufferGB = deviceCount > 1 \? \(hasNVLink \? ([\d.]+) : ([\d.]+)\) : 0;/).slice(1).map(Number);
  assert.deepStrictEqual(
    quoted(/\((\d+)% of active params, ([\d.]+) GiB\/GPU, ([\d.]+)–([\d.]+) GiB per extra GPU\)/, 'the overhead heuristics'),
    [actPct, ctxGiB, lo, hi]);
});

test('the project-structure listing names files that exist', () => {
  const block = readmeDoc.match(/## Project structure\n+```\n([\s\S]*?)```/);
  assert.ok(block, 'README.md no longer carries a project-structure block');
  const listed = block[1].split('\n').map(l => l.split(/\s{2,}/)[0].trim()).filter(Boolean);
  for (const entry of listed) {
    assert.ok(fs.existsSync(path.join(ROOT, entry)),
      `README.md's project structure names ${entry}, which does not exist`);
  }
  /* And the other direction, which is how it went stale: the block claims to describe
     the machinery a contributor touches, so nothing in those directories may be
     missing from it. Derived by reading the directories, never enumerated. */
  for (const dir of ['tests', 'tools', 'data', 'benchmarks']) {
    for (const f of fs.readdirSync(path.join(ROOT, dir))) {
      if (f.startsWith('.') || f === '__pycache__') continue;
      assert.ok(listed.includes(`${dir}/${f}`),
        `${dir}/${f} is not in README.md's project structure`);
    }
  }
});

test('no document promises a --device flag', () => {
  /* vLLM has no --device flag; the accelerator comes from the image and the
     environment. The ROADMAP promised one for AMD for months. It may be named only to
     deny it — the same shape as MODEL.md's pipeline-parallel pin. */
  const docs = {
    'README.md': readmeDoc, 'ROADMAP.md': roadmapDoc, 'CONTRIBUTING.md': contributing,
    'docs/MODEL.md': fs.readFileSync(path.join(ROOT, 'docs', 'MODEL.md'), 'utf8'),
  };
  /* Docker has a --device flag of its own, and vLLM's ROCm image needs two of them,
     `--device /dev/kfd` and `--device /dev/dri`, to see the GPUs. Those two are not
     vLLM's flag, so they are allowed as exactly themselves, and nothing else is. */
  const DOCKER_DEVICE = /--device \/dev\/(kfd|dri)\b/g;
  for (const [name, text] of Object.entries(docs)) {
    for (const line of text.split('\n').filter(l => l.replace(DOCKER_DEVICE, '').includes('--device'))) {
      assert.ok(/\b(no|not|never)\b/i.test(line),
        `${name} mentions --device without denying it: ${line.trim()}`);
    }
  }
  // And neither engine may emit vLLM's, which is what the denial is asserting.
  const py = fs.readFileSync(path.join(ROOT, 'generate_report.py'), 'utf8');
  for (const [name, src] of [['index.html', html], ['generate_report.py', py]]) {
    assert.ok(!/['"`]--device/.test(src.replace(DOCKER_DEVICE, '')), `${name} emits a --device flag`);
  }
});

console.log('\nThe copied report quotes the command box');
/* Every weight option the page offers, read from its own <select>, so an option
   added tomorrow is covered without anyone listing it here. */
const WEIGHT_SELECT = html.slice(html.indexOf('<select id="weight-precision"'),
                                 html.indexOf('</select>', html.indexOf('<select id="weight-precision"')));
/* Each <option>'s own attributes, read in any order. The first version matched
   value and data-q only when they sat side by side, so the AWQ option, the page's
   default, which carries `selected` between them, was silently left out. A GGUF
   banner shown for every AWQ plan then passed every check here (cold check,
   fix/gguf-plugin). Hence the count below: the options read must be all the
   options there are. */
const WEIGHT_OPTION_TAGS = [...WEIGHT_SELECT.matchAll(/<option\b([^>]*)>/g)].map(m => m[1]);
const WEIGHT_OPTIONS = WEIGHT_OPTION_TAGS.map(attrs => ({
  bytesPerParam: Number((attrs.match(/\bvalue="([\d.]+)"/) || [])[1]),
  quantMethod: (attrs.match(/\bdata-q="(\w*)"/) || [])[1],
}));
assert.ok(WEIGHT_OPTIONS.length === (WEIGHT_SELECT.match(/<option\b/g) || []).length
          && WEIGHT_OPTIONS.every(o => Number.isFinite(o.bytesPerParam) && typeof o.quantMethod === 'string'),
  `read ${WEIGHT_OPTIONS.length} weight options, not every <option> the select holds, or one without a value and data-q`);
assert.ok(WEIGHT_OPTIONS.filter(o => o.quantMethod === 'gguf').length >= 6
          && ['', 'fp8', 'awq', 'gptq'].every(q => WEIGHT_OPTIONS.some(o => o.quantMethod === q)),
  'the weight options were not found in the page, so the checks below would check nothing');
const dense8BPlan = { params: 8, layers: 32, kvHeads: 8, headDim: 128, activePercent: 100 };
test('the copied report quotes the whole command, for every weight option the page offers', () => {
  /* The report used to quote the panel's first <code>, which for a GGUF plan was a
     span in the banner above the box: the report said `vllm serve` and nothing else. */
  for (const opt of WEIGHT_OPTIONS) {
    const h = renderHarness();
    const st = asState(GPU_TABLE['h100-80'], 1, { ...dense8BPlan, ...opt });
    const c = h.computeInference(st);
    assert.ok(c.fits, `8B at ${opt.bytesPerParam} B/param should fit one H100, or this checks nothing`);
    h.renderCommand(st, c);
    const box = (h.out['command-output'] || '').match(/<div class="code-box">[\s\S]*?<code>([\s\S]*?)<\/code>/);
    assert.ok(box && box[1].startsWith('vllm serve ') && box[1].includes('--max-model-len'),
      `${opt.bytesPerParam}/${opt.quantMethod}: no command box on the panel`);
    assert.ok(h.exportSummary(st, c).includes('\n## vLLM command\n```\n' + box[1] + '\n```\n'),
      `${opt.bytesPerParam}/${opt.quantMethod}: the copied report does not quote the command box`);
  }
});

console.log('\nGGUF guidance says what vLLM needs today');
/* The sentences are a contract, so they are literals here rather than read back
   from GGUF_GUIDANCE: a test that took its expectation from the page would follow
   any edit to it, including the one that put the retired single-file claim back. */
const GGUF_LINES = [
  ["GGUF support left vLLM's core in v0.24.0 and moved to a separate plugin.",
   'https://github.com/vllm-project/vllm/releases/tag/v0.24.0'],
  ["Install `vllm-gguf-plugin` before serving a GGUF model. Point the command at a GGUF checkpoint, either a Hugging Face repo as `repo_id:quant_type` (for example `unsloth/Qwen3-0.6B-GGUF:Q4_K_M`) or a local `.gguf` file, and pass the base model's tokenizer with `--tokenizer`.",
   'https://docs.vllm.ai/en/v0.30.0/features/quantization/gguf.html'],
  ['vLLM calls its GGUF support "highly experimental and under-optimized".',
   'https://docs.vllm.ai/en/v0.30.0/features/quantization/gguf.html'],
  ['For GGUF specifically, llama.cpp or Ollama is the better-supported path; AWQ or GPTQ is the usual choice on vLLM.',
   null],
];
const stripTags = (s) => s.replace(/<[^>]+>/g, '');
const ggufSurfaces = (card, extra) => {
  const h = renderHarness();
  const st = asState(card, 1, extra);
  const c = h.computeInference(st);
  h.renderCommand(st, c);
  const panel = h.out['command-output'] || '';
  return { c, panel, text: stripTags(panel), report: h.exportSummary(st, c) };
};
test('every GGUF level names the plugin, with its sources, on the command panel and in the copied report', () => {
  for (const opt of WEIGHT_OPTIONS) {
    const { c, panel, text, report } = ggufSurfaces(GPU_TABLE['h100-80'], { ...dense8BPlan, ...opt });
    assert.ok(c.fits, `8B at ${opt.bytesPerParam} B/param should fit one H100, or this checks nothing`);
    const gguf = opt.quantMethod === 'gguf';
    const where = `${opt.bytesPerParam} B/param, --quantization ${opt.quantMethod || '(none)'}`;
    for (const [line, source] of GGUF_LINES) {
      assert.strictEqual(text.includes(line.replace(/`/g, '')), gguf,
        `${where}: the command panel ${gguf ? 'lacks' : 'shows'} "${line}"`);
      assert.strictEqual(report.includes(`- ${line}${source ? ` (source: ${source})` : ''}\n`), gguf,
        `${where}: the copied report ${gguf ? 'lacks' : 'shows'} "${line}" with its source`);
      if (source) assert.strictEqual(panel.includes(`href="${source}"`), gguf,
        `${where}: the command panel ${gguf ? 'does not link' : 'links'} ${source}`);
    }
    assert.ok(!/not a repo|single \.?gguf file/i.test(text + report),
      `${where}: the retired single-file claim is back`);
  }
});
test('a GGUF plan that does not fit prints no GGUF guidance, because it prints no command', () => {
  const { c, text, report } = ggufSurfaces(GPU_TABLE['t4-16'],
    { params: 70, layers: 80, bytesPerParam: 0.63, quantMethod: 'gguf' });
  assert.ok(!c.fits, '70B at Q4_K_M should not fit one T4, or this checks nothing');
  for (const [line] of GGUF_LINES) {
    assert.ok(!text.includes(line.replace(/`/g, '')) && !report.includes(line),
      `guidance for a command the page does not print: "${line}"`);
  }
});

console.log('\nROCm: vLLM\'s own image, and what else running it needs');
/* The contract the ROCm command rests on, written out: vLLM v0.30.0's own image, the
   flags vLLM's docs give for it, word for word, and per LLVM target whether FP8
   weights load, whether AITER is there, and whether the FP8 KV path is unverified
   (docs/research/vllm-rocm.md). The table is read from the page once, here, and held
   to these literals. */
const ROCM_TABLE = new Function(`${html.match(/^const ROCM = \{[\s\S]*?\n\};$/m)[0]}; return ROCM;`)();
const ROCM_IMAGE = 'vllm/vllm-openai-rocm:v0.30.0';
const ROCM_FLAGS = ['--group-add=video', '--cap-add=SYS_PTRACE', '--security-opt seccomp=unconfined',
                    '--device /dev/kfd', '--device /dev/dri', '-v ~/.cache/huggingface:/root/.cache/huggingface',
                    '--env "HF_TOKEN=$HF_TOKEN"', '-p 8000:8000', '--ipc=host'];
const ROCM_ARCH = { gfx90a: { fp8Weights: false, aiter: false, fp8KvUnverified: false },
                    gfx942: { fp8Weights: true, aiter: true, fp8KvUnverified: false },
                    gfx1100: { fp8Weights: false, aiter: false, fp8KvUnverified: true } };
/* Every line the planner prints under an AMD command, word for word, with its source.
   The rules below it (a sentence, a source of the right kind) held while
   engine_r10_rocm_guidance's R11 swapped HIP_VISIBLE_DEVICES for CUDA_VISIBLE_DEVICES
   in both engines alike, so parity saw nothing and only the goldens noticed. A line
   is a claim a reader acts on: changing one means changing it here too. */
const ROCM_LINES = {
  image: ["This is vLLM's own ROCm image, pinned to `v0.30.0`, the release these lines were checked against on 2026-09-23. AMD's `rocm/vllm` images are deprecated.",
    'https://github.com/vllm-project/vllm/blob/v0.30.0/docs/getting_started/installation/gpu.rocm.inc.md#L353-L394'],
  wheels: ["vLLM's ROCm wheels are built for Python 3.12 only, and on any other Python the installer silently falls back to the CUDA wheel, which fails on AMD GPUs. The image avoids that.",
    'https://github.com/vllm-project/vllm/blob/v0.30.0/docs/getting_started/installation/gpu.rocm.inc.md#L30-L32'],
  hip: ['To choose GPUs, add `--env HIP_VISIBLE_DEVICES=0,1` before the image name, with your own device IDs. Since v0.30.0, vLLM on ROCm no longer falls back to `CUDA_VISIBLE_DEVICES`.',
    'https://github.com/vllm-project/vllm/releases/tag/v0.30.0'],
  hipBoth: ['If both are set and differ, vLLM stops at startup.',
    'https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/platforms/rocm.py#L127-L137'],
  gcd: ['Each board of this card is two GPUs to ROCm, one per GCD, so N boards are 2N device IDs.',
    'https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi250.html'],
  aiterOn: ["`VLLM_ROCM_USE_AITER=1` turns on AITER, AMD's kernel library; AMD's vLLM guide says to always set it on Instinct MI300-series GPUs.",
    'https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html'],
  aiterDefault: ['vLLM leaves AITER off unless it is set.',
    'https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/envs.py#L1231-L1232'],
  aiterOff: ["AITER, AMD's kernel library, is enabled only on CDNA3 and newer, so this card runs without it.",
    'https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/_aiter_ops.py#L138-L160'],
  fp8Weights: ["vLLM v0.30.0's FP8 weight kernels need CDNA3 or newer, or RDNA4, so FP8 weights are not offered on this card.",
    'https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/kernels/linear/scaled_mm/rocm.py#L89-L90'],
  awqDocs: ["vLLM's quantization table marks AWQ and GPTQ as unsupported on AMD GPUs.",
    'https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/quantization/README.md#L69-L70'],
  awqSource: ["v0.30.0's ROCm platform accepts both; this tool has not run either.",
    'https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/platforms/rocm.py#L503-L527'],
  gguf: ["GGUF needs `vllm-gguf-plugin`, which this image doesn't include. The plugin lists ROCm among its prerequisites; this tool has not run it.",
    'https://github.com/vllm-project/vllm-gguf-plugin'],
  kvUnverified: ["On RDNA, vLLM v0.30.0's custom paged-attention kernel takes only the default KV cache type, so an FP8 cache runs on another kernel path, which this tool has not verified.",
    'https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/platforms/rocm.py#L401-L410'],
  more: ["Every line here, and what could not be verified, is in the planner's ROCm notes.",
    'https://github.com/israelhen153/llm-vram-planner/blob/HEAD/docs/research/vllm-rocm.md'],
};

test('the ROCm table says what vLLM v0.30.0 and AMD say, each line with a source of the kind it claims', () => {
  assert.strictEqual(ROCM_TABLE.image, ROCM_IMAGE);
  assert.deepStrictEqual(ROCM_TABLE.dockerFlags, ROCM_FLAGS);
  assert.deepStrictEqual(ROCM_TABLE.arch, ROCM_ARCH);
  assert.strictEqual(ROCM_TABLE.vllm, 'v0.30.0');
  assert.deepStrictEqual(ROCM_TABLE.lines, ROCM_LINES);
  const PINNED = /^https:\/\/(github\.com\/vllm-project\/vllm\/(blob|releases\/tag)\/v0\.30\.0([\/#]|$)|github\.com\/vllm-project\/vllm-gguf-plugin$|rocm\.docs\.amd\.com\/|instinct\.docs\.amd\.com\/|github\.com\/israelhen153\/llm-vram-planner\/blob\/HEAD\/docs\/research\/vllm-rocm\.md$)/;
  for (const [id, [text, source]] of Object.entries(ROCM_TABLE.lines)) {
    assert.ok(typeof text === 'string' && text.trim().endsWith('.'), `line ${id} is not a sentence`);
    assert.ok(PINNED.test(source), `line ${id}: ${source} is not a v0.30.0 vLLM source, AMD's own page, the plugin, or the planner's notes`);
  }
});

test('every AMD row names an LLVM target the ROCm table knows, and no NVIDIA row names one', () => {
  const used = new Set();
  for (const [key, g] of Object.entries(GPU_TABLE)) {
    if (g.vendor === 'amd') {
      assert.ok(Object.hasOwn(ROCM_ARCH, g.gfx), `${key}: gfx ${g.gfx} is not a target the ROCm table knows`);
      used.add(g.gfx);
    } else assert.strictEqual(g.gfx, undefined, `${key}: an NVIDIA row carries a gfx`);
  }
  assert.deepStrictEqual([...used].sort(), Object.keys(ROCM_ARCH).sort(), 'the ROCm table knows a target no row uses');
});

test("an AMD card's command is vLLM's ROCm image with the same serve arguments; an NVIDIA card's is unchanged", () => {
  /* The property: the launcher differs, the plan does not. The arguments after the
     image are exactly the ones `vllm serve` gets for the same plan, since the image's
     entrypoint is `vllm serve`. Every catalog card, every weight option the page
     offers, both KV types, one to three boards, local paths in and out of /opt, and
     a hub id: with only /opt in the list, a mount made only for /opt passed. */
  let amd = 0, nvidia = 0;
  for (const [key, card] of Object.entries(GPU_TABLE)) {
    for (const opt of WEIGHT_OPTIONS) for (const kv of [2, 1]) for (const boards of [1, 2, 3]) {
      const h = renderHarness();
      const st = asState(card, boards, { ...dense8BPlan, ...opt, kvBytesPerValue: kv });
      const c = h.computeInference(st);
      for (const model of ['/opt/models/YourModel', '/mnt/models/llama-8b', 'meta-llama/Llama-3.1-8B-Instruct']) {
        const cmd = h.buildVllmCommand(st, c, model);
        const where = `${key} x${boards}, ${opt.quantMethod || 'bf16'} ${opt.bytesPerParam}, KV ${kv}, ${model}`;
        if (!c.fits) { assert.ok(cmd.startsWith('# Does not fit'), `${where}: no fit, and a command`); continue; }
        /* FP8 weights on a target vLLM has no FP8 weight kernel for: no command, the
           reason in its place. Written out here, from the table's literals above. */
        if (card.vendor === 'amd' && !ROCM_ARCH[card.gfx].fp8Weights && (opt.quantMethod === 'fp8' || opt.bytesPerParam === 1)) {
          assert.strictEqual(cmd, `# vLLM v0.30.0 has no FP8 weight kernel for ${st.gpuName} (${card.gfx}): its FP8 matrix kernels need CDNA3 or newer, or RDNA4. Choose BF16, AWQ or GPTQ.`,
            `${where}: FP8 weights on a card vLLM can't run them on`);
          continue;
        }
        const serve = h.buildVllmCommand({ ...st, vendor: 'nvidia' }, c, model);
        const head = `vllm serve ${model} \\\n`;
        assert.ok(serve.startsWith(head), `${where}: the vllm serve form lost its head`);
        if (card.vendor !== 'amd') {
          assert.strictEqual(cmd, serve, `${where}: an NVIDIA card's command is not the vllm serve one`);
          nvidia++;
          continue;
        }
        const want = ['docker run --rm \\', ...ROCM_FLAGS.map(f => `    ${f} \\`),
                      ...(ROCM_ARCH[card.gfx].aiter ? ['    --env VLLM_ROCM_USE_AITER=1 \\'] : []),
                      ...(model.startsWith('/') ? [`    -v ${model}:${model} \\`] : []),
                      `    ${ROCM_IMAGE} \\`, `    ${model} \\`].join('\n') + '\n' + serve.slice(head.length);
        assert.strictEqual(cmd, want, `${where}: the ROCm command`);
        for (const never of ['CUDA_VISIBLE_DEVICES', 'vllm serve', 'rocm/vllm'])
          assert.ok(!cmd.includes(never), `${where}: the ROCm command says ${never}`);
        amd++;
      }
    }
  }
  assert.ok(amd >= 200 && nvidia >= 400, `checked ${amd} AMD and ${nvidia} NVIDIA commands`);
});

test('the precision control offers no FP8 where vLLM has no FP8 weight kernel, falls back to BF16, and gives FP8 back', () => {
  /* syncPrecision() itself, on a stub of the page's own <select>: the options are the
     page's, read above, and the cards are the catalog's. A reader on FP8 who picks a
     card vLLM can't run FP8 weights on gets BF16 and a disabled, relabelled FP8
     option. Back on a card that runs it, FP8 is given back. A choice the reader made
     is never turned into FP8. */
  const decl = (re) => { const m = html.match(re); assert.ok(m, `${re} not found in index.html`); return m[0]; };
  const src = [decl(/^const ROCM = \{[\s\S]*?\n\};$/m), decl(/^function fp8WeightsBlocked\(gpu\) \{[\s\S]*?\n\}$/m),
               decl(/^let precisionForcedFromFp8 = false;$/m), decl(/^function syncPrecision\(\) \{[\s\S]*?\n\}$/m)].join('\n');
  const page = () => {
    const options = WEIGHT_OPTIONS.map(o => ({ value: String(o.bytesPerParam), dataset: { q: o.quantMethod },
                                                disabled: false, textContent: '' }));
    /* A single-select, as a browser runs one: selecting an option deselects the rest.
       A plain property would let two options be selected at once, which no page can. */
    const chosen = new Set();
    for (const o of options)
      Object.defineProperty(o, 'selected', {
        get: () => chosen.has(o),
        set: (v) => { if (v) { chosen.clear(); chosen.add(o); } else chosen.delete(o); },
      });
    const els = { 'gpu-model': { value: 'h100-80' }, 'weight-precision': { options } };
    const api = new Function('document', 'GPU_TABLE', `${src}; return { syncPrecision };`)(
      { getElementById: (id) => els[id] }, GPU_TABLE);
    const fp8 = options.find(o => o.dataset.q === 'fp8');
    return {
      pick: (q) => { options.find(x => x.dataset.q === q).selected = true; },
      picked: () => options.find(o => o.selected).dataset.q,
      card: (slug) => { els['gpu-model'].value = slug; api.syncPrecision(); },
      fp8,
      options,
    };
  };
  const gated = Object.entries(GPU_TABLE).filter(([, g]) => g.vendor === 'amd' && !ROCM_ARCH[g.gfx].fp8Weights).map(([k]) => k);
  assert.deepStrictEqual(gated.sort(), ['mi210-64', 'mi250x-128', 'rx7900xtx-24']);
  for (const slug of Object.keys(GPU_TABLE)) {
    const blocked = gated.includes(slug);
    // FP8 chosen on a card that runs it, then this card, then back.
    const p = page();
    p.pick('fp8'); p.card('h100-80');
    p.card(slug);
    assert.strictEqual(p.fp8.disabled, blocked, `${slug}: FP8 ${blocked ? 'still offered' : 'withheld'}`);
    assert.strictEqual(p.fp8.textContent, blocked ? 'FP8 — no vLLM FP8 weight kernel for this card' : 'FP8 (1.0 B/param)', `${slug}: the FP8 option's label`);
    assert.strictEqual(p.picked(), blocked ? '' : 'fp8', `${slug}: the precision after the switch`);
    p.card('h100-80');
    assert.strictEqual(p.picked(), 'fp8', `${slug}: FP8 was not given back on a card that runs it`);
    assert.ok(!p.fp8.disabled, `${slug}: FP8 still disabled on a card that runs it`);
    /* Every other option the reader chose is left alone, both ways: BF16 and each
       quantized one, each GGUF level its own option. Only BF16 was checked once, and
       a gate that sent AWQ, GPTQ or GGUF to BF16 on a card that can't run FP8 passed. */
    for (const [i, o] of page().options.entries()) {
      if (o.dataset.q === 'fp8') continue;
      const q = page();
      const what = `${slug}: the reader's own ${o.dataset.q || 'bf16'} at ${o.value} B/param`;
      q.options[i].selected = true; q.card('h100-80'); q.card(slug);
      assert.ok(q.options[i].selected, `${what} was changed on the way to this card`);
      q.card('h100-80');
      assert.ok(q.options[i].selected, `${what} was changed on the way back`);
    }
    // Forced to BF16, then the reader picks AWQ there: AWQ stays when FP8 would come back.
    if (blocked) {
      const r = page();
      r.pick('fp8'); r.card('h100-80'); r.card(slug); r.pick('awq'); r.card('h100-80');
      assert.strictEqual(r.picked(), 'awq', `${slug}: a choice the reader made after the fallback was overridden`);
    }
  }
});

test('recalculate() syncs the interconnect and the precision controls before it reads the state', () => {
  /* The two gates above are tested by calling them. What makes them reach the page is
     the call in recalculate(), before readInputState(), so the state it reads, and
     every figure and command built from it, already reflects the gate. A gate with no
     call site passes its own tests and gates nothing. */
  const start = html.indexOf('function recalculate() {');
  const body = html.slice(start, html.indexOf('\n}\n', start));
  const read = body.indexOf('readInputState()');
  for (const call of ['syncInterconnect();', 'syncPrecision();']) {
    const at = body.indexOf(call);
    assert.ok(at > 0 && read > at, `recalculate() does not call ${call} before it reads the state`);
  }
});

test('the ROCm lines under the command are the ones that apply, each with its source, and only on AMD', () => {
  /* Which lines apply is re-derived here from the plan, not taken from the page:
     always the image, the wheels, choosing GPUs and the both-set failure; the GCD
     line on a board of two devices; AITER on or off by target; AWQ/GPTQ's docs and
     source when either is chosen; the plugin for GGUF; the unverified FP8 KV path
     where the target has one; and the notes. None when no command is printed. */
  const L = ROCM_TABLE.lines;
  const expected = (card, st) => {
    if (card.vendor !== 'amd') return [];
    const arch = ROCM_ARCH[card.gfx];
    return [L.image, L.wheels, L.hip, L.hipBoth, ...(card.devices > 1 ? [L.gcd] : []),
            ...(arch.aiter ? [L.aiterOn, L.aiterDefault] : [L.aiterOff]),
            ...(arch.fp8Weights ? [] : [L.fp8Weights]),
            ...(['awq', 'gptq'].includes(st.quantMethod) ? [L.awqDocs, L.awqSource] : []),
            ...(st.quantMethod === 'gguf' ? [L.gguf] : []),
            ...(st.kvBytesPerValue < 2 && arch.fp8KvUnverified ? [L.kvUnverified] : []), L.more];
  };
  const pageLine = ([text, source]) => `<div>${text.replace(/`([^`]+)`/g, '<code>$1</code>')} (<a href="${source}" target="_blank" style="color:var(--accent-text)">source</a>)</div>`;
  let shown = 0;
  for (const [key, card] of Object.entries(GPU_TABLE)) {
    for (const opt of WEIGHT_OPTIONS) for (const kv of [2, 1]) for (const boards of [1, 2]) {
      const h = renderHarness();
      const st = asState(card, boards, { ...dense8BPlan, ...opt, kvBytesPerValue: kv });
      const c = h.computeInference(st);
      h.renderCommand(st, c);
      const panel = h.out['command-output'] || '';
      const report = h.exportSummary(st, c);
      // No lines under a command that isn't printed: no fit, or FP8 refused.
      const refused = card.vendor === 'amd' && !ROCM_ARCH[card.gfx].fp8Weights && (opt.quantMethod === 'fp8' || opt.bytesPerParam === 1);
      const want = c.fits && !refused ? expected(card, st) : [];
      const where = `${key} x${boards}, ${opt.quantMethod || 'bf16'} ${opt.bytesPerParam}, KV ${kv}`;
      for (const line of Object.values(L)) {
        const on = want.includes(line);
        assert.strictEqual(panel.includes(pageLine(line)), on, `${where}: the command panel ${on ? 'lacks' : 'shows'} "${line[0].slice(0, 60)}"`);
      }
      const section = report.includes('\n## Running on ROCm\n') ? report.split('\n## Running on ROCm\n')[1].split('\n## ')[0] : null;
      if (want.length) {
        assert.strictEqual(section, want.map(([t, s]) => `- ${t} (source: ${s})`).join('\n') + '\n', `${where}: the copied report's ROCm section`);
        shown++;
      } else assert.strictEqual(section, null, `${where}: a ROCm section where none applies`);
    }
  }
  assert.ok(shown >= 50, `only ${shown} plans showed ROCm lines`);
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

/* ---- what today's cards display ------------------------------------------
 *
 * Every other test in this file is differential: it renders a card with
 * constants beside the same card without them, and holds the difference to the
 * throughput figures. That catches anything added to one of the two. It is
 * blind to anything added to both — a sentence invented and shown on every card
 * in the catalog is identical on both sides, so the comparison sees nothing.
 * X2 in the sabotage corpus is exactly that shape, and it stayed green through
 * all 474 tests by sitting inside the one element the comparison is told may
 * differ.
 *
 * A comparison needs something to compare against. This is it: what the page
 * renders today, recorded. Any change to any of it fails, whether or not the
 * change was meant — which is the whole point, and also the whole cost.
 *
 * Deliberate display change?  UPDATE_GOLDEN=1 node tests/model.test.js
 * then read the diff before committing. That diff is the only place a claim
 * added to every card at once becomes visible.
 */
console.log('\nWhat today\'s cards display');

const GOLDEN_PAGE = path.join(__dirname, 'golden', 'page.json');

/* Every catalog row at one canonical load, then the axes that change what the
   page *says* rather than what it computes, on one card. Named, so a diff names
   the case that moved. Deliberately not the absent-constants probe grid: that
   one exists to vary everything, and a golden that wide would be updated so
   often nobody would read the diff. */
const goldenCases = () => {
  const dense8B = { params: 8, layers: 32, kvHeads: 8, headDim: 128, activePercent: 100 };
  /* NVLink as the page sets it: the control asks for it, and readInputState()
     grants it only to a board that has it. asState() leaves it on for any card,
     which no single-device row ever shows — the MI250X, two devices on one board
     with no NVLink, was the first row where the golden recorded a page the tool
     cannot produce. */
  const cases = Object.keys(GPU_TABLE).sort().map(slug =>
    [`${slug} — 8B bf16, 16 at 8K`,
     asState(GPU_TABLE[slug], 1, { ...dense8B, hasNVLink: supportsNVLink(GPU_TABLE[slug]) })]);
  const h = GPU_TABLE['h100-80'], t4 = GPU_TABLE['t4-16'];
  return cases.concat([
    ['h100-80 x8 NVLink — 70B bf16', asState(h, 8, { params: 70, layers: 80 })],
    ['h100-80 x8 PCIe — 70B bf16, the fabric note',
     asState(h, 8, { params: 70, layers: 80, hasNVLink: false })],
    ['h100-80 x16 PCIe — past one NVLink domain, so the heuristic caveat',
     asState(h, 16, { params: 70, layers: 80, hasNVLink: false })],
    ['h100-80 x1 — 70B bf16 does not fit, so the verdict and the board advice',
     asState(h, 1, { params: 70, layers: 80 })],
    ['h100-80 x2 — 30B MoE, the shared-expert note',
     asState(h, 2, { ...dense8B, params: 30, layers: 48, activePercent: 10, sharedExperts: 1 })],
    ['h100-80 x2 — 26B sliding window', asState(h, 2, { ...dense8B, params: 26, layers: 30,
      headDim: 256, activePercent: 15, attnMode: 'swa', swaWindow: 1024, swaLocalLayers: 25 })],
    ['h100-80 x2 — 671B MLA', asState(h, 2, { ...dense8B, params: 671, layers: 61, kvHeads: 128,
      headDim: 56, activePercent: 5, sharedExperts: 1, attnMode: 'mla', mlaLatentDim: 576 })],
    ['h100-80 x1 — fp8 weights on silicon that has the tensor cores',
     asState(h, 1, { ...dense8B, bytesPerParam: 1, quantMethod: 'fp8' })],
    ['t4-16 x1 — fp8 weights on silicon that does not, so the caveat',
     asState(t4, 1, { ...dense8B, bytesPerParam: 1, quantMethod: 'fp8', hasNVLink: false })],
    ['h100-80 x1 — 256 at 1K, so the KV queue warning',
     asState(h, 1, { ...dense8B, contextLength: 1024, concurrency: 256 })],
    // ROCm: vLLM's own image, and the lines that apply to the plan.
    ['mi300x-192 x1 — AWQ weights, so AITER on and the AWQ lines',
     asState(GPU_TABLE['mi300x-192'], 1, { ...dense8B, bytesPerParam: 0.5, quantMethod: 'awq', hasNVLink: false })],
    ['mi210-64 x1 — FP8 weights asked for, refused: vLLM has no FP8 weight kernel for gfx90a',
     asState(GPU_TABLE['mi210-64'], 1, { ...dense8B, bytesPerParam: 1, quantMethod: 'fp8', hasNVLink: false })],
    ['mi250x-128 x2 — four GCDs on two boards',
     asState(GPU_TABLE['mi250x-128'], 2, { ...dense8B, hasNVLink: false })],
    ['rx7900xtx-24 x1 — an FP8 KV cache on RDNA3, the unverified path, 4 users so it fits',
     asState(GPU_TABLE['rx7900xtx-24'], 1, { ...dense8B, kvBytesPerValue: 1, concurrency: 4, hasNVLink: false })],
    ['h100-80 x1 — a 32K context behind an 8K cached prefix',
     asState(h, 1, { ...dense8B, contextLength: 32768, concurrency: 4,
                     sharedPrefix: 8192, prefixCaching: true })],
    ['h100-80 x1 — AWQ weights, the page\'s default precision',
     asState(h, 1, { ...dense8B, bytesPerParam: 0.5, quantMethod: 'awq' })],
    ['h100-80 x1 — GGUF weights, which renderCommand branches on',
     asState(h, 1, { ...dense8B, bytesPerParam: 0.63, quantMethod: 'gguf' })],
    ['h100-80 x1 — a model imported by id rather than named by a preset',
     asState(h, 1, { ...dense8B, hfModelId: 'org/imported-8b' })],
    /* Hardware with no measured constants. Synthetic, because no catalog row is
       like this until the AMD rows land — and that is the point: what the page
       says instead of a figure is as much "what it displays" as the figure was,
       and the with/without comparisons cannot pin it, since it is the very thing
       they are comparing. A gauge with no text drawn on this side is invisible
       to them and visible here. */
    ['(no constants) h100-80 x1 — 8B bf16, the reason in place of the figures',
     asState({ ...h, perfKey: 'no-such-key' }, 1, dense8B)],
    ['(no constants) h100-80 x8 PCIe — 70B bf16, past one domain',
     asState({ ...h, perfKey: 'no-such-key' }, 8,
             { params: 70, layers: 80, kvHeads: 8, headDim: 128, activePercent: 100,
               hasNVLink: false })],
    ['(no constants) t4-16 x1 — fp8 on silicon without the tensor cores',
     asState({ ...t4, perfKey: 'no-such-key' }, 1,
             { ...dense8B, bytesPerParam: 1, quantMethod: 'fp8', hasNVLink: false })],
  ]);
};

/* Object keys in source order would make the golden depend on the order the
   renderers happened to write their elements in, which is not something this
   file should pin. */
const stable = (v) => Array.isArray(v) ? v.map(stable)
  : v && typeof v === 'object'
    ? Object.fromEntries(Object.keys(v).sort().map(k => [k, stable(v[k])]))
    : v;

/* What a reader takes off a surface, rather than the markup that carries it.
   Tag names, nesting and class names are dropped: no reader sees them, and
   renaming a class is the commonest change there is — charging a full golden
   update for it teaches people to regenerate without reading the diff, which
   is the one way this test can fail silently.
   Attribute *values* are kept and sorted, because that is where a figure hides
   when it has no text: a bar's width, a meter's value, an aria-valuenow, a
   title, an href. Sorted, so reordering two attributes is not a change. */
const readerView = (markup) => {
  const raw = String(markup ?? '');
  /* A style attribute is a list, not a string: `width:40%;background:red` and
     `background:red;width:40%` style the same box. Sorted declaration by
     declaration so reordering them is not a change, while every value — a bar's
     width among them — is still recorded. */
  const normalise = (name, value) => name.toLowerCase() !== 'style' ? value
    : value.split(';').map(d => d.trim()).filter(Boolean).sort().join(';');
  const attrs = [...raw.matchAll(/\s([a-zA-Z][\w-]*)="([^"]*)"/g)]
    .filter(([, name]) => name.toLowerCase() !== 'class')
    .map(([, name, value]) => `${name}=${normalise(name, value)}`).sort();
  const text = raw
    .replace(/<[^>]*>/g, ' ')
    .replace(/&nbsp;/g, ' ').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&')
    .replace(/\s+/g, ' ').trim();
  return attrs.length ? { text, attrs } : text;
};

const captureGolden = () => stable(Object.fromEntries(goldenCases().map(([name, st]) => {
  const s = renderEverything(st);
  const view = (rec) => Object.fromEntries(Object.entries(rec).map(([k, v]) => [k, readerView(v)]));
  /* props holds values that were never markup — hidden, style.display, a data-*
     — so they are recorded as they were set, not run through the tag stripper. */
  return [name, { shows: view(s.html), text: view(s.written), props: s.props }];
})));

/* The golden is only as good as what it renders, and its case list is a literal.
   Every other literal list in this file that decides coverage is checked against
   what the tool actually ships — the probe grid is, the renderer list is — so
   this one is too. Derived from the cases themselves, never restated. */
test('the golden records every catalog row, and the shapes that change what the page says', () => {
  const cases = goldenCases();
  const states = cases.map(([, st]) => st);
  const named = cases.map(([name]) => name);
  const missing = Object.keys(GPU_TABLE).filter(slug => !named.some(n => n.startsWith(`${slug} `)));
  assert.deepStrictEqual(missing, [], `the golden stopped recording catalog rows: ${missing}`);
  // Every case is a page the tool can produce: NVLink only on a board that has it.
  const unreachable = cases.filter(([, st]) => st.hasNVLink && !supportsNVLink({ form: st.gpuForm }))
    .map(([name]) => name);
  assert.deepStrictEqual(unreachable, [], `the golden records NVLink on a board without it: ${unreachable}`);
  const modelled = st => computeInference(st).throughputModelled;
  const shapes = {
    'an OAM board with more than one device': sts =>
      sts.some(st => st.gpuForm === 'oam' && st.gpuCount * (st.gpuDevices || 1) > 1),
    'a price tier with no confirmed price': sts =>
      sts.some(st => [st.gpuHyperCost, st.gpuSpecCost, st.gpuSpotCost].includes(null)),
    'a price recorded by hand': sts => sts.some(st => st.priceRecord && Object.keys(st.priceRecord).length),
    'a card with constants': sts => sts.some(modelled),
    'a card with none': sts => sts.some(st => !modelled(st)),
    'one board': sts => sts.some(st => st.gpuCount === 1),
    'past one NVLink domain': sts => sts.some(st => st.gpuCount * (st.gpuDevices || 1) > 8),
    'PCIe': sts => sts.some(st => !st.hasNVLink),
    'a model that does not fit': sts => sts.some(st => !computeInference(st).fits),
    'a mixture of experts': sts => sts.some(st => st.activePercent < 100),
    'sliding-window attention': sts => sts.some(st => st.attnMode === 'swa'),
    'MLA': sts => sts.some(st => st.attnMode === 'mla'),
    'fp8 weights on silicon with the tensor cores': sts =>
      sts.some(st => st.quantMethod === 'fp8' && st.gpuFp8),
    'fp8 weights on silicon without them': sts =>
      sts.some(st => st.quantMethod === 'fp8' && !st.gpuFp8),
    'GGUF weights': sts => sts.some(st => st.quantMethod === 'gguf'),
    'AWQ weights, the default': sts => sts.some(st => st.quantMethod === 'awq'),
    'a ROCm command with AITER on': sts => sts.some(st => st.vendor === 'amd' && st.gfx === 'gfx942'),
    'FP8 weights refused on a card vLLM has no FP8 kernel for': sts =>
      sts.some(st => st.quantMethod === 'fp8' && st.vendor === 'amd' && st.gfx !== 'gfx942'),
    'a ROCm command on more than one dual-GCD board': sts =>
      sts.some(st => st.vendor === 'amd' && st.gpuDevices > 1 && st.gpuCount > 1),
    // Printed only under a command, so only a plan that fits records it.
    'an FP8 KV cache on RDNA3': sts =>
      sts.some(st => st.gfx === 'gfx1100' && st.kvBytesPerValue < 2 && computeInference(st).fits),
    'a shared prefix': sts => sts.some(st => st.sharedPrefix > 0),
    'a batch the KV cache cannot hold': sts =>
      sts.some(st => computeInference(st).batchLimitedByKV),
    'a model imported by id': sts => sts.some(st => !!st.hfModelId),
  };
  const gone = Object.entries(shapes).filter(([, hits]) => !hits(states)).map(([n]) => n);
  assert.deepStrictEqual(gone, [], `the golden no longer records: ${gone.join(', ')}`);
});

test('every card still displays exactly what the golden records', () => {
  const now = captureGolden();
  if (process.env.UPDATE_GOLDEN) {
    fs.writeFileSync(GOLDEN_PAGE, JSON.stringify(now, null, 1) + '\n');
    console.log(`       (rewrote ${path.relative(ROOT, GOLDEN_PAGE)} — read the diff)`);
    return;
  }
  assert.ok(fs.existsSync(GOLDEN_PAGE),
    `${path.relative(ROOT, GOLDEN_PAGE)} is missing — UPDATE_GOLDEN=1 writes it`);
  const golden = JSON.parse(fs.readFileSync(GOLDEN_PAGE, 'utf8'));

  /* Report what moved, not that something did: a golden whose failure says only
     "not equal" is a golden nobody updates honestly. */
  const moved = [];
  const walk = (a, b, trail) => {
    if (JSON.stringify(a) === JSON.stringify(b)) return;
    const aObj = a && typeof a === 'object' && !Array.isArray(a);
    const bObj = b && typeof b === 'object' && !Array.isArray(b);
    if (aObj && bObj) {
      for (const k of [...new Set([...Object.keys(a), ...Object.keys(b)])])
        walk(a[k], b[k], trail.concat(k));
      return;
    }
    moved.push({ at: trail.join(' / '), was: a, now: b });
  };
  walk(golden, now, []);
  if (!moved.length) return;
  const shown = moved.slice(0, 6).map(({ at, was, now: n }) =>
    `  ${at}\n    was: ${was === undefined ? '(absent)' : JSON.stringify(String(was)).slice(0, 220)}` +
    `\n    now: ${n === undefined ? '(absent)' : JSON.stringify(String(n)).slice(0, 220)}`).join('\n');
  assert.fail(`${moved.length} recorded surface(s) changed. If every one of these was ` +
    `meant, UPDATE_GOLDEN=1 node tests/model.test.js and commit the diff.\n${shown}` +
    (moved.length > 6 ? `\n  ... and ${moved.length - 6} more` : ''));
});

/* The sabotage corpus documents itself in tests/sabotage/README.md — which driver
   covers what, and which residual risks are still open. That prose is how the next
   cold check decides where to start, so a driver missing from it is a driver nobody
   runs. Derived from the directory, never enumerated, for the same reason the
   project-structure guard above is: an enumerated list is the thing that goes stale.

   This guards the mechanical half only. Whether the residual-risk section is still
   true is judgement, and no test can hold it. */
test('the sabotage README names every driver, and no driver it does not have', () => {
  const dir = path.join(ROOT, 'tests', 'sabotage');
  const doc = fs.readFileSync(path.join(dir, 'README.md'), 'utf8');
  /* A driver is named for what it attacks — engine_* or workflow_* — and every other
     script here must be a named support file. Without the second half, a driver
     misnamed out of the pattern would drop out of every check below in silence,
     which is the failure this whole corpus exists to catch in other code. */
  const SUPPORT = ['harness.py', 'anchors.py', 'chain.sh', 'suites.sh'];
  const scripts = fs.readdirSync(dir).filter(f => /\.(py|sh)$/.test(f));
  const onDisk = scripts.filter(f => /^(engine|workflow)_.*\.(py|sh)$/.test(f)).sort();
  const stray = scripts.filter(f => !onDisk.includes(f) && !SUPPORT.includes(f));
  assert.deepStrictEqual(stray, [],
    `tests/sabotage holds scripts that are neither drivers nor support files: ${stray.join(', ')}`);
  assert.ok(onDisk.length > 0, 'no sabotage drivers found — has the directory moved?');

  const named = new Set((doc.match(/`(?:engine|workflow)_[0-9a-z_]+\.(?:py|sh)`/g) || [])
    .map(s => s.replace(/`/g, '')));

  const missing = onDisk.filter(f => !named.has(f));
  assert.deepStrictEqual(missing, [],
    `tests/sabotage/README.md does not name: ${missing.join(', ')}`);

  const phantom = [...named].filter(f => !onDisk.includes(f));
  assert.deepStrictEqual(phantom, [],
    `tests/sabotage/README.md names drivers that do not exist: ${phantom.join(', ')}`);

  /* The shared judge is what makes a driver's verdict mean anything, so its absence
     must fail here rather than at 2am inside a check. */
  assert.ok(fs.existsSync(path.join(dir, 'suites.sh')),
    'tests/sabotage/suites.sh is missing — the drivers have nothing to judge with');

  /* A driver judges a sabotage by a suite going red. One already red for an unrelated
     reason makes every sabotage read as caught, silently, and a whole run is then 269
     false catches that look exactly like real ones. Every driver must refuse to judge
     against a red baseline — and this is derived from the directory rather than a list,
     because the first attempt at this fix keyed off a string and missed two drivers. */
  const unguarded = onDisk.filter(f => {
    const src = fs.readFileSync(path.join(dir, f), 'utf8');
    return !/run_driver\(|require_green_baseline|already red on the unmodified/.test(src);
  });
  assert.deepStrictEqual(unguarded, [],
    `these drivers would judge against a red baseline: ${unguarded.join(', ')}`);
  /* Every Python driver now proves its baseline by calling harness.run_driver, so the
     proof lives in one place — and one deleted line there would disarm all of them at
     once while every driver above still read as guarded. Pin it where it lives. */
  const harness = fs.readFileSync(path.join(dir, 'harness.py'), 'utf8');
  const runDriver = harness.slice(harness.indexOf('def run_driver(')).split(/\ndef /)[0];
  assert.ok(harness.includes('def run_driver(') && /require_green_baseline\(\)/.test(runDriver),
    'harness.run_driver no longer proves a green baseline, so no Python driver does');

  /* The drivers import each other. A run that leaves __pycache__ behind leaves a
     gitignored directory git cannot remove on a branch switch, which strands an
     unlisted tests/sabotage and turns the guard above red on a branch that has
     nothing to do with this work. It cost two people a confusing red suite before
     it was fixed, so the fix is pinned rather than remembered. */
  /* chain.sh executes a driver directly, so one committed without the executable
     bit errors with 126 and the whole run exits non-zero — which is how
     engine_r4_cost_provenance.py failed to run at all on the commit that added
     it. Loud, but only twenty minutes in; this says it in a second. */
  const notExecutable = onDisk.filter(f => !(fs.statSync(path.join(dir, f)).mode & 0o111));
  assert.deepStrictEqual(notExecutable, [],
    `these drivers are not executable, so chain.sh cannot run them: ${notExecutable.join(', ')}`);

  const importers = onDisk.filter(f => f.endsWith('.py'))
    .filter(f => /^import\s|^from\s/m.test(fs.readFileSync(path.join(dir, f), 'utf8')));
  const writesBytecode = importers.filter(f =>
    !/dont_write_bytecode\s*=\s*True/.test(fs.readFileSync(path.join(dir, f), 'utf8')));
  assert.deepStrictEqual(writesBytecode, [],
    `these drivers would leave __pycache__ behind: ${writesBytecode.join(', ')}`);
});

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);
