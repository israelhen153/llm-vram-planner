#!/usr/bin/env python3
"""No-change comparison: master's generate_report.py vs this commit's, for every
current catalog card. compute() on a full grid through each version's own
from_cli_args(); every story string of the PDF on a strided sample; the real
PDF bytes (timestamps pinned, reportlab invariant mode) on a smaller sample;
from_json and interactive_mode builders on a few.

Both versions come from git, not from checked-in copies. Snapshots of a 70 KB
engine go stale the moment either side moves, and a comparison against a stale
"master" reports zero differences for the wrong reason.

Run: python3 tests/sabotage/compare/nochange.py [base-ref] [stride] [pdf_stride]
     base-ref defaults to master; the other side is always the working tree.

Every field is compared, with no exemptions. An earlier version excused `perfKey`
because the commit it was written for was the one adding it — which made the
comparison asymmetric, reported five false differences once both sides had the
field, and, worse, would have hidden a real perfKey change in any later commit.
A field this harness is told to ignore is a field it cannot report on, so when a
change legitimately adds one, read the diff it prints rather than silencing it.
"""
import contextlib, importlib.util, io, itertools, os, subprocess, sys, tempfile, types, unittest.mock, datetime as _dt
ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                               cwd=os.path.dirname(os.path.abspath(__file__)),
                               text=True).strip()
BASE_REF = sys.argv[1] if len(sys.argv) > 1 else "master"
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 29
PDF_STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 977


def at_ref(ref, rel):
    """The file as it stands at <ref>, written somewhere importable."""
    blob = subprocess.check_output(["git", "show", f"{ref}:{rel}"], cwd=ROOT)
    fd, tmp = tempfile.mkstemp(suffix="_" + os.path.basename(rel))
    os.write(fd, blob)
    os.close(fd)
    return tmp

import reportlab.rl_config
reportlab.rl_config.invariant = 1


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


NEW = load("gr_new", "generate_report.py")                       # the working tree
OLD = load("gr_old", os.path.relpath(at_ref(BASE_REF, "generate_report.py"), ROOT))


class FixedDT(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 17, 12, 0, 0)


for m in (NEW, OLD):
    m.datetime = FixedDT

MODELS = ["llama31-8b", "llama31-70b", "qwen3-30b", "gemma4-26b", "dsv3-671b", "mistral-lg-123b"]
COUNTS = [1, 2, 4, 8, 9, 12, 16, 64, 128]
PRECS = ["bf16", "fp8", "awq", "q4km"]
KVS = [False, True]
CTX = [(4096, 1), (8192, 20), (32768, 64), (131072, 1)]
NV = [False, True]  # no_nvlink


def args_for(gpu, count, model, prec, fp8kv, ctx, conc, no_nv):
    return types.SimpleNamespace(preset=model, gpu=gpu, prec=prec, ctx=ctx, conc=conc,
                                 ngpu=count, no_nvlink=no_nv, fp8_kv=fp8kv)


def story_strings(m, cfg):
    obj = m.ReportCard(cfg, output_path=os.devnull)
    seen = []

    def harvest(item):
        text = getattr(item, "text", None)
        if text:
            seen.append(text)
        for line in getattr(item, "lines", None) or []:
            seen.append(line if isinstance(line, str) else str(line))
        for row in getattr(item, "_cellvalues", []):
            for cell in row:
                harvest(cell) if hasattr(cell, "text") or hasattr(cell, "contents") else seen.append(str(cell))
        for child in getattr(item, "contents", []) or []:
            harvest(child)

    real_build = m.SimpleDocTemplate.build
    captured = {}
    try:
        m.SimpleDocTemplate.build = lambda self, story, **kw: captured.__setitem__("story", story)
        obj.generate()
        for item in captured.get("story", []):
            harvest(item)
    finally:
        m.SimpleDocTemplate.build = real_build
    return seen


def pdf_bytes(m, cfg):
    import tempfile
    fd, out = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m.ReportCard(cfg, output_path=out).generate()
        with open(out, "rb") as f:
            return f.read()
    finally:
        os.remove(out)


n = comp_diffs = story_n = story_diffs = pdf_n = pdf_diffs = cfg_diffs = 0
new_only, old_only, samples = set(), set(), []
for gpu in NEW.GPUS:
    for count, model, prec, fp8kv, (ctx, conc), no_nv in itertools.product(COUNTS, MODELS, PRECS, KVS, CTX, NV):
        if no_nv and NEW.GPUS[gpu]["form"] != "sxm":
            continue  # nvlink_for() downgrades either way
        a = args_for(gpu, count, model, prec, fp8kv, ctx, conc, no_nv)
        with contextlib.redirect_stdout(io.StringIO()):
            ca, cb = OLD.from_cli_args(a), NEW.from_cli_args(a)
        n += 1
        for k in set(ca) | set(cb):
            if k == "gpu":
                if ca[k] != cb[k]:
                    cfg_diffs += 1; samples.append(f"{gpu}: cfg gpu row differs")
                continue
            if ca.get(k) != cb.get(k):
                cfg_diffs += 1
                if len(samples) < 12: samples.append(f"{gpu} x{count} {model} {prec}: cfg[{k}] {ca.get(k)!r} -> {cb.get(k)!r}")
        pa, pb = OLD.compute(ca), NEW.compute(cb)
        for k in pa:
            if k not in pb:
                old_only.add(k); continue
            if pa[k] != pb[k] or type(pa[k]) is not type(pb[k]):
                comp_diffs += 1
                if len(samples) < 12: samples.append(f"{gpu} x{count} {model} {prec} kv{fp8kv} {ctx}/{conc} nonv{no_nv}: {k} {pa[k]!r} -> {pb[k]!r}")
        for k in pb:
            if k not in pa: new_only.add(k)
        if pb.get("throughput_modelled") is not True:
            comp_diffs += 1; samples.append(f"{gpu}: throughput_modelled={pb.get('throughput_modelled')!r}")
        if OLD.build_vllm_cmd(ca, pa) != NEW.build_vllm_cmd(cb, pb):
            comp_diffs += 1; samples.append(f"{gpu} x{count} {model} {prec}: command differs")
        if not pa["fits"] and OLD.boards_advice(ca, pa) != NEW.boards_advice(cb, pb):
            comp_diffs += 1; samples.append(f"{gpu} x{count} {model} {prec}: advice differs")
        if n % STRIDE == 0:
            story_n += 1
            sa, sb = story_strings(OLD, ca), story_strings(NEW, cb)
            if sa != sb:
                story_diffs += 1
                if len(samples) < 12:
                    i = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x != y), min(len(sa), len(sb)))
                    samples.append(f"{gpu} x{count} {model} {prec} kv{fp8kv} {ctx}/{conc} nonv{no_nv}: story differs at {i}: old {sa[i:i+2]!r} new {sb[i:i+2]!r} (lengths {len(sa)} vs {len(sb)})")
        if n % PDF_STRIDE == 0:
            pdf_n += 1
            if pdf_bytes(OLD, ca) != pdf_bytes(NEW, cb):
                pdf_diffs += 1
                if len(samples) < 12: samples.append(f"{gpu} x{count} {model} {prec} kv{fp8kv} {ctx}/{conc} nonv{no_nv}: PDF bytes differ")

# from_json (preset and raw branches) and interactive_mode on a few cards
import json, tempfile
jn = jd = 0
for gpu in ("a100-40", "h100-80", "rtx4090-24", "t4-16", "b200-192"):
    for spec in ({"preset": "llama31-70b", "gpu": gpu, "bpp": 1, "ctx": 16384, "conc": 8, "n_gpu": 4, "nvlink": True},
                 {"params": 8, "layers": 32, "kv_heads": 8, "gpu": gpu, "n_gpu": 2, "nvlink": False, "kv_bpp": 1}):
        fd, p = tempfile.mkstemp(suffix=".json"); os.close(fd)
        with open(p, "w") as f: json.dump(spec, f)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ca, cb = OLD.from_json(p), NEW.from_json(p)
        finally:
            os.remove(p)
        jn += 1
        if story_strings(OLD, ca) != story_strings(NEW, cb) or OLD.compute(ca) != {k: v for k, v in NEW.compute(cb).items() if k in OLD.compute(ca)}:
            jd += 1; samples.append(f"from_json {gpu} {spec}: differs")
    answers = [str(list(NEW.PRESETS).index("llama31-8b") + 1), str(list(NEW.GPUS).index(gpu) + 1), "2", "y" , "1", "n", "8192", "4"]
    outs = []
    for m in (OLD, NEW):
        ans = list(answers) if m.supports_nvlink(m.GPUS[gpu]) else [a for i, a in enumerate(answers) if i != 3]
        with unittest.mock.patch("builtins.input", side_effect=ans), contextlib.redirect_stdout(io.StringIO()):
            outs.append(m.interactive_mode())
    jn += 1
    a, b = outs
    if {k: v for k, v in a.items() if k != "gpu"} != {k: v for k, v in b.items() if k != "gpu"}:
        jd += 1; samples.append(f"interactive {gpu}: {a} vs {b}")

print(f"from_cli_args cfg: {n} configurations, {cfg_diffs} differences")
print(f"compute() + command + advice: {n} configurations, {comp_diffs} differences")
print(f"  fields only in new: {sorted(new_only)}  only in master: {sorted(old_only)}")
print(f"PDF story strings: {story_n} configurations, {story_diffs} differing")
print(f"PDF bytes (pinned clock, invariant mode): {pdf_n} configurations, {pdf_diffs} differing")
print(f"from_json / interactive_mode: {jn} checks, {jd} differing")
for s in samples:
    print("  " + s)
