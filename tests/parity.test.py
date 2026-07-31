#!/usr/bin/env python3
"""Verify the PDF generator and the web tool compute the same numbers.

index.html and generate_report.py implement the same model twice, in two
languages. That duplication is deliberate (the web tool has no backend, the
report generator has no browser) but it silently drifted once already: the
throughput and dtype bugs lived in both copies, and fixing one would have left
the PDF — which the README calls procurement-ready — quietly wrong.

Extracts compute() from generate_report.py without importing reportlab, runs
computeInference() from index.html under node, and compares field by field.

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
wanted = {"GIB", "GPUS"}
nodes = [
    n for n in tree.body
    if (isinstance(n, ast.FunctionDef) and n.name == "compute")
    or (isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) in wanted for t in n.targets))
]
assert any(isinstance(n, ast.FunctionDef) for n in nodes), "compute() not found"
ns = {"math": __import__("math")}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "<gr>", "exec"), ns)
compute, GPUS = ns["compute"], ns["GPUS"]

# ---- run the JS model on the same inputs ----------------------------------
CASES = [
    {"name": "8B bf16, 1 user, H100",
     "params": 8, "active": 100, "bpp": 2, "layers": 32, "kv_heads": 8, "h_dim": 128,
     "ctx": 8192, "conc": 1, "n_gpu": 1, "gpu": "h100-80"},
    {"name": "8B bf16, 64 users, H100",
     "params": 8, "active": 100, "bpp": 2, "layers": 32, "kv_heads": 8, "h_dim": 128,
     "ctx": 8192, "conc": 64, "n_gpu": 1, "gpu": "h100-80"},
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
const s=html.indexOf('function computeInference(state) {'), e=html.indexOf('\n}\n',s);
const ci=new Function(`${gib}\n${html.slice(s,e+2)}; return computeInference;`)();
const G={};
for(const[,v,n]of html.matchAll(/<option value="([\d.|]+)"[^>]*data-n="([^"]+)"/g)){
  const[gb,bw,h,sp,st,tf]=v.split('|').map(Number); G[n]={gb,bw,h,sp,st,tf};
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

NAMES = {"t4-16": "T4 16GB", "l4-24": "L4 24GB", "rtx4090-24": "RTX 4090 24GB",
         "rtx5090-32": "RTX 5090 32GB", "a100-40": "A100 40GB",
         "rtx6000ada-48": "RTX 6000 Ada 48GB", "l40s-48": "L40S 48GB",
         "a100-80": "A100 80GB", "h100-80": "H100 80GB", "rtxpro-96": "RTX PRO 6000 96GB",
         "h200-141": "H200 141GB", "b200-192": "B200 192GB"}

js_in = [dict(c, gpuName=NAMES[c["gpu"]], max_ctx=c.get("max_ctx", 1048576)) for c in CASES]
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
    ("per_user_load", "perUserAtLoadTokS", 1), ("eff_batch", "effectiveBatch", 0),
    ("max_batch_kv", "maxBatchByKV", 0), ("ttft_ms", "ttftMs", 1),
    ("sat_batch", "saturatedBatch", 0), ("sat_tok", "saturatedTokS", 1),
    ("ttft_cold_ms", "ttftColdMs", 1), ("ttft_warm_ms", "ttftWarmMs", 1),
    ("kv_saved_by_prefix_gb", "kvSavedByPrefixGB", 0.01),
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
const fs=require('fs');const h=fs.readFileSync(process.argv[1],'utf8');const o={};
for(const[,v,n]of h.matchAll(/<option value="([\\d.|]+)"[^>]*data-n="([^"]+)"/g)){
  const[gb,bw,hyper,spec,spot,tflops]=v.split('|').map(Number);
  o[n]={gb,bw,hyper,spec,spot,tflops};
}
console.log(JSON.stringify(o));""", os.path.join(ROOT, "index.html")],
    capture_output=True, text=True).stdout)

py_by_name = {v["name"].replace(" GB", "GB"): v for v in GPUS.values()}
js_by_name = {k.replace(" GB", "GB"): v for k, v in js_gpus.items()}
if set(py_by_name) != set(js_by_name):
    only_py = sorted(set(py_by_name) - set(js_by_name))
    only_js = sorted(set(js_by_name) - set(py_by_name))
    print(f"  FAIL GPU tables list different cards: only-python={only_py} only-js={only_js}")
    failed += 1
else:
    drift = []
    for name in sorted(py_by_name):
        for f in ("gb", "bw", "hyper", "spec", "spot", "tflops"):
            a, b = py_by_name[name][f], js_by_name[name][f]
            if abs(a - b) > 1e-9:
                drift.append(f"{name}.{f}: py={a} js={b}")
    if drift:
        print("  FAIL GPU tables have drifted")
        for d in drift:
            print(f"       {d}")
        failed += 1
    else:
        print(f"  ok   GPU tables identical across both engines ({len(py_by_name)} cards)")
        passed += 1

print(f"\n{passed} passed, {failed} failed\n")
sys.exit(1 if failed else 0)
