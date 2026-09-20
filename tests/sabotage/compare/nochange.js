/* No-change comparison: master's index.html vs this commit's, for every
   current catalog card. Compares computeInference() field by field on a full
   grid, and every renderer's HTML, the copied Markdown report, the vLLM
   command and the board advice on a strided sample of it. Also runs each
   version's own readInputState() through a DOM stub, card by card.

   Run: node tmp/nochange.js [stride]   (stride defaults to 13) */
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const os = require('os');
const ROOT = execFileSync('git', ['rev-parse', '--show-toplevel'],
  { cwd: __dirname, encoding: 'utf8' }).trim();
const BASE_REF = process.argv[2] || 'master';

/* Both sides come from git rather than checked-in copies: a snapshot of a 200 KB
   engine goes stale the moment either side moves, and a comparison against a stale
   "master" reports zero differences for the wrong reason. */
function atRef(ref, rel) {
  const blob = execFileSync('git', ['show', `${ref}:${rel}`],
    { cwd: ROOT, maxBuffer: 256 * 1024 * 1024 });
  const tmp = path.join(os.tmpdir(), `nochange_${ref.replace(/\W/g, '_')}_${path.basename(rel)}`);
  fs.writeFileSync(tmp, blob);
  return tmp;
}
const STRIDE = Number(process.argv[2] || 13);

function load(file) {
  const html = fs.readFileSync(file, 'utf8');
  const out = {};
  const values = { 'train-method': 'lora', 'train-optimizer': 'adam', 'train-grad-ckpt': '1',
    'train-batch-size': '1', 'train-seq-len': '2048', 'train-lora-rank': '16',
    'train-lora-targets': '4', 'train-hidden-size': '4096' };
  const el = (id) => ({
    get innerHTML() { return out[id] || ''; }, set innerHTML(v) { out[id] = v; },
    get value() { return values[id] !== undefined ? values[id] : ''; }, set value(v) { values[id] = v; },
    style: {}, options: [], selectedOptions: [{ dataset: { q: '' } }], checked: true,
    classList: { add() {}, remove() {} },
  });
  const cache = {};
  const document = {
    getElementById: (id) => (cache[id] = cache[id] || el(id)),
    body: { classList: { add() {}, remove() {}, contains: () => true } },
    querySelector: () => ({ textContent: '', innerHTML: '', parentElement: null }),
    querySelectorAll: () => [],
  };
  const start = html.indexOf('function getVal(id)');
  const before = [/^const GPU_TABLE = \{[\s\S]*?\n\};$/m, /^const BENCHMARK_DATA = \{[\s\S]*?\n\};$/m,
                  /^const MODEL_PRESETS = \{[\s\S]*?\n\};$/m, /^const WORKLOAD_PROFILES = \{[\s\S]*?\n\};$/m]
    .map(re => html.match(re)[0]).filter(d => html.indexOf(d) < start).join('\n');
  const navigator = { clipboard: { writeText: () => Promise.resolve() } };
  const src = html.slice(start, html.indexOf('function updateURLHash()'));
  const names = [...src.matchAll(/^function (\w+)\(/gm)].map(m => m[1]);
  const api = new Function('document', 'navigator', `
    let currentAttn = { mode: 'standard', window: 0, localLayers: 0, mlaDim: 0 };
    let currentModelMaxCtx = 131072, importedModelId = null, urlRestoreLost = [];
    let savedSnapshots = [];
    ${before}
    ${src}
    return { ${names.join(', ')}, GPU_TABLE, MODEL_PRESETS,
             pushSnapshot: (s, c) => savedSnapshots.push({ state: s, computed: c, name: 'snap' }),
             clearSnapshots: () => { savedSnapshots.length = 0; } };`)(document, navigator);
  const params = {};
  for (const m of src.matchAll(/function (render\w+)\(([^)]*)\)/g))
    params[m[1]] = m[2].split(',').map(x => x.trim().split(/[=\s]/)[0]).filter(Boolean);
  return { ...api, out, cache, values, html, params, renderers: names.filter(n => /^render/.test(n)) };
}

const NEW = load(path.join(ROOT, 'index.html'));       // the working tree
const OLD = load(atRef(BASE_REF, 'index.html'));

// ---- readInputState through a DOM stub, each version's own -------------------
const domStub = (gpuKey, interconnect, prec = '2', q = '') => {
  const el = (v, extra = {}) => ({ value: v, style: {}, options: [], ...extra });
  const fields = {
    'gpu-model': el(gpuKey), 'interconnect': el(interconnect),
    'param-count': el('8'), 'active-percent': el('100'), 'layer-count': el('32'),
    'kv-head-count': el('8'), 'head-dim': el('128'), 'shared-experts': el('0'),
    'context-length': el('8192'), 'concurrency': el('1'), 'gpu-count': el('2'),
    'shared-prefix': el('0'), 'prefix-caching': el('1'),
    'kv-precision': el('2'), 'preset': el(''),
    'weight-precision': el(prec, { selectedOptions: [{ dataset: { q } }] }),
  };
  return { getElementById: (id) => fields[id] };
};
const readInputStateFor = (v, gpuKey, interconnect, prec, q) => {
  const stop = v.html.indexOf('\nconst PERF = {');
  const src = v.html.slice(v.html.indexOf('function getVal(id)'), stop);
  const fn = new Function('document', 'GPU_TABLE', `
    let currentAttn = { mode: 'standard', window: 0, localLayers: 0, mlaDim: 0 };
    let currentModelMaxCtx = 131072, importedModelId = null;
    ${src}
    return readInputState;`)(domStub(gpuKey, interconnect, prec, q), v.GPU_TABLE);
  return fn();
};
let stateDiffs = 0, stateChecked = 0;
for (const key of Object.keys(NEW.GPU_TABLE)) {
  for (const ic of ['1', '0']) for (const [prec, q] of [['2', ''], ['1', 'fp8'], ['0.5', 'awq'], ['0.63', 'gguf']]) {
    const a = readInputStateFor(OLD, key, ic, prec, q), b = readInputStateFor(NEW, key, ic, prec, q);
    stateChecked++;
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    for (const k of keys) {
      if (!(k in a)) { stateDiffs++; console.log(`readInputState ${key}: new adds field ${k}`); continue; }
      if (!(k in b)) { stateDiffs++; console.log(`readInputState ${key}: new dropped field ${k}`); continue; }
      if (JSON.stringify(a[k]) !== JSON.stringify(b[k])) { stateDiffs++; console.log(`readInputState ${key}/${ic}/${prec}: ${k} ${JSON.stringify(a[k])} -> ${JSON.stringify(b[k])}`); }
    }
  }
}
console.log(`readInputState: ${stateChecked} states compared, ${stateDiffs} differences`);

// ---- the grid ----------------------------------------------------------------
const P = NEW.MODEL_PRESETS;
const MODELS = ['llama31-8b', 'llama31-70b', 'qwen3-30b', 'gemma4-26b', 'dsv3-671b', 'mistral-lg-123b'];
const COUNTS = [1, 2, 4, 8, 9, 12, 16, 64, 128];
const PRECS = [[2, ''], [1, 'fp8'], [0.5, 'awq'], [0.63, 'gguf']];
const KVS = [2, 1];
const CTX = [[4096, 1], [8192, 20], [32768, 64], [131072, 1]];
const NV = [true, false];
const PREFIX = [[0, true], [2048, true], [2048, false]];

function stateFor(key, g, count, model, [bpp, q], kv, [ctx, conc], nv, [pre, pc]) {
  const p = P[model];
  return {
    params: p.p, activePercent: p.a, bytesPerParam: bpp, quantMethod: q, layers: p.l, kvHeads: p.kv,
    headDim: p.hd, sharedExperts: p.se, contextLength: ctx, concurrency: conc, sharedPrefix: pre,
    prefixCaching: pc, gpuCount: count, gpuGB: g.gb, gpuBandwidth: g.bw, gpuTFLOPS: g.tflops,
    gpuHyperCost: g.hyper, gpuSpecCost: g.spec, gpuSpotCost: g.spot, gpuName: g.name.replace(/ GB$/, 'GB'),
    gpuDevices: g.devices, gpuKey: key, perfKey: g.perfKey, vendor: g.vendor,
    gpuFp8: !!(g.caps && g.caps.fp8), hasNVLink: g.form === 'sxm' && nv, kvBytesPerValue: kv,
    presetKey: '', hfModelId: null, attnMode: p.attn || 'standard', swaWindow: p.swaWin || 0,
    swaLocalLayers: Math.min(p.swaLocal || 0, p.l), mlaLatentDim: p.mlaDim || 0, modelMaxCtx: p.maxCtx || 131072,
  };
}
const same = (a, b) => (Number.isNaN(a) && Number.isNaN(b)) || JSON.stringify(a) === JSON.stringify(b);

let n = 0, computeDiffs = 0, renderDiffs = 0, rendered = 0, newOnly = new Set(), oldOnly = new Set();
const diffSamples = [];
const renderAll = (v, st) => {
  for (const k of Object.keys(v.out)) delete v.out[k];
  v.clearSnapshots();
  const c = v.computeInference(st);
  v.pushSnapshot(st, c);
  for (const name of v.renderers) {
    const args = (v.params[name] || []).map(pn => (pn === 'computed' ? c : pn === 'state' ? st : undefined));
    v[name](...args);
  }
  const res = { ...v.out };
  res['(copied report)'] = v.exportSummary(st, c);
  res['(command)'] = v.buildVllmCommand(st, c, 'm');
  if (!c.fits) res['(advice)'] = v.boardsAdvice(st, c);
  return res;
};
for (const [key, g] of Object.entries(NEW.GPU_TABLE))
  for (const count of COUNTS) for (const model of MODELS) for (const prec of PRECS) for (const kv of KVS)
    for (const ctx of CTX) for (const nv of NV) for (const pre of PREFIX) {
      if (!nv && g.form !== 'sxm') continue; // hasNVLink is false either way; skip the duplicate
      const st = stateFor(key, g, count, model, prec, kv, ctx, nv, pre);
      const a = OLD.computeInference(st), b = NEW.computeInference(st);
      n++;
      for (const k of Object.keys(a)) {
        if (!(k in b)) { oldOnly.add(k); continue; }
        if (!same(a[k], b[k])) { computeDiffs++; if (diffSamples.length < 10) diffSamples.push(`${key} x${count} ${model} ${prec} kv${kv} ${ctx} nv${nv} ${pre}: ${k} ${JSON.stringify(a[k])} -> ${JSON.stringify(b[k])}`); }
      }
      for (const k of Object.keys(b)) if (!(k in a)) newOnly.add(k);
      if (b.throughputModelled !== true) { computeDiffs++; diffSamples.push(`${key}: throughputModelled=${b.throughputModelled}`); }
      if (n % STRIDE === 0) {
        rendered++;
        const ra = renderAll(OLD, st), rb = renderAll(NEW, st);
        for (const id of new Set([...Object.keys(ra), ...Object.keys(rb)])) {
          if (ra[id] !== rb[id]) {
            renderDiffs++;
            if (diffSamples.length < 10) {
              const i = [...(ra[id] || '')].findIndex((ch, idx) => ch !== (rb[id] || '')[idx]);
              diffSamples.push(`${key} x${count} ${model} ${prec} kv${kv} ${ctx} nv${nv} ${pre}: ${id} differs at ${i}: old "${(ra[id] || '').slice(Math.max(0, i - 60), i + 60)}" new "${(rb[id] || '').slice(Math.max(0, i - 60), i + 60)}"`);
            }
          }
        }
      }
    }
console.log(`computeInference: ${n} configurations, ${computeDiffs} field differences`);
console.log(`  fields only in new: [${[...newOnly].join(', ')}]  only in master: [${[...oldOnly].join(', ')}]`);
console.log(`renderers + copied report + command + advice: ${rendered} configurations, ${renderDiffs} differing surfaces`);
for (const d of diffSamples) console.log('  ' + d);
