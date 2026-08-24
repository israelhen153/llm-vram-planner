#!/usr/bin/env python3
"""Verify tools/sync_data.py generates blocks that are safe to inline.

The GPU catalog and the benchmark table are contributor-facing JSON, rendered
into index.html and generate_report.py as source code. Two things have to hold
that the parity suite cannot check by comparing values: the rendered text must
parse as the language it lands in, and a contributor's free text must not be
able to escape the literal it is written into. Benchmark `note` and `src` are
free text, and index.html's copy sits inside a <script> element in an HTML file.

Run:  python3 tests/sync.test.py
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import sync_data

pass_ct = fail_ct = 0


def test(name, fn):
    global pass_ct, fail_ct
    try:
        fn()
        print(f"  ok   {name}")
        pass_ct += 1
    except Exception as e:
        print(f"  FAIL {name}\n       {type(e).__name__}: {e}")
        fail_ct += 1


def js_eval(block, var):
    """Evaluate a generated JS block under node and return the object it declares."""
    proc = subprocess.run(
        ["node", "-e",
         f"const b=process.argv[1];"
         f"console.log(JSON.stringify(new Function(b + '; return {var};')()));",
         block],
        capture_output=True, text=True)
    assert not proc.returncode, f"node could not evaluate the block:\n{proc.stderr}"
    return json.loads(proc.stdout)


print("\nGenerated blocks round-trip through the language they land in")

def check_gpu_js_round_trips():
    gpus = sync_data.load_gpus()
    block = sync_data.render_gpu_js(gpus)
    body = "\n".join(l for l in block.splitlines() if not l.startswith("/*"))
    assert js_eval(body, "GPU_TABLE") == gpus, "GPU_TABLE did not round-trip"

test("data/gpus.json survives rendering to JS and back", check_gpu_js_round_trips)


def check_bench_js_round_trips():
    bench = sync_data.load_benchmarks()
    block = sync_data.render_benchmark_js(bench)
    body = "\n".join(l for l in block.splitlines() if not l.startswith("/*"))
    assert js_eval(body, "BENCHMARK_DATA") == bench, "BENCHMARK_DATA did not round-trip"

test("benchmarks/data.json survives rendering to JS and back", check_bench_js_round_trips)


def check_gpu_py_round_trips():
    gpus = sync_data.load_gpus()
    block = sync_data.render_gpu_py(gpus)
    body = "\n".join(l for l in block.splitlines() if not l.startswith("#"))
    ns = {}
    exec(body, ns)
    assert ns["GPUS"] == gpus, "GPUS did not round-trip"

test("data/gpus.json survives rendering to Python and back", check_gpu_py_round_trips)


print("\nContributor free text cannot escape the literal it is written into")

# A benchmark note is prose from a pull request. Each of these ends the string,
# the script element, or the line if it reaches the file unescaped.
HOSTILE = [
    "</script><script>alert(1)</script>",
    "it's a 'quoted' note",
    r"back\slash and \'escaped quote",
    "line one\nline two",
    "carriage\rreturn",
    "paragraph separator",
]

def check_hostile_notes_are_escaped():
    for i, note in enumerate(HOSTILE):
        rows = {"8b-h100-80": {"tokS": 1, "mode": "batch", "src": note, "note": note,
                               "prec": "bf16", "date": "2026-01", "url": ""}}
        block = sync_data.render_benchmark_js(rows)
        body = "\n".join(l for l in block.splitlines() if not l.startswith("/*"))
        # It must still be the same string once JS has parsed it...
        got = js_eval(body, "BENCHMARK_DATA")
        assert got == rows, f"case {i}: value changed in transit: {got}"
        # ...and the raw text must not contain a sequence that ends the <script>
        # element, which the HTML parser resolves before JavaScript ever runs.
        assert "</script" not in body.lower(), f"case {i}: rendered block can close its own script tag"
        # One row in, one row line out: a raw line terminator inside the literal
        # would appear here as an extra line, and as a syntax error in a browser.
        assert len(body.strip().splitlines()) == 3, (
            f"case {i}: expected a declaration, one row and a close — got:\n{body}")

test("a hostile benchmark note round-trips intact and cannot close the script tag",
     check_hostile_notes_are_escaped)


def check_missing_field_names_itself():
    rows = {"8b-h100-80": {"tokS": 1, "mode": "batch", "src": "x", "note": "y", "prec": "bf16"}}
    try:
        sync_data.render_benchmark_js(rows)
    except SystemExit as e:
        assert "8b-h100-80" in str(e) and "date" in str(e) and "url" in str(e), \
            f"the error names neither the row nor the missing fields: {e}"
        return
    raise AssertionError("a row missing required fields rendered anyway")

test("a row missing a required field names the row and the field", check_missing_field_names_itself)


print("\nEvery generated block is claimed by exactly one marker pair")

def check_every_block_is_present_once():
    for rel, tag, _, _ in sync_data.BLOCKS:
        text = open(os.path.join(ROOT, rel)).read()
        for marker in (f"{tag}:BEGIN", f"{tag}:END"):
            n = text.count(marker)
            assert n == 1, f"{rel} carries {n} copies of {marker}, expected 1"

test("each tagged block appears exactly once in the file that carries it",
     check_every_block_is_present_once)


def check_sync_is_a_noop_on_a_clean_tree():
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "sync_data.py")],
                          capture_output=True, text=True, cwd=ROOT)
    assert not proc.returncode, proc.stderr
    updated = [l for l in proc.stdout.splitlines() if l.endswith("updated")]
    assert not updated, ("re-running the sync tool rewrote a block, so a generated "
                         f"block was hand-edited or a source changed:\n{chr(10).join(updated)}")

test("running the sync tool against a clean tree changes nothing",
     check_sync_is_a_noop_on_a_clean_tree)


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
