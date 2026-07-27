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
    modelMaxCtx:c.max_ctx||1048576});
});
console.log(JSON.stringify(out));
"""

NAMES = {"t4-16": "T4 16GB", "rtx4090-24": "RTX 4090 24GB", "a100-40": "A100 40GB",
         "l40s-48": "L40S 48GB", "a100-80": "A100 80GB", "h100-80": "H100 80GB",
         "rtxpro-96": "RTX PRO 6000", "b200-192": "B200 192GB"}

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

print(f"\n{passed} passed, {failed} failed\n")
sys.exit(1 if failed else 0)
