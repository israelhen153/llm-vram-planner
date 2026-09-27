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
import re
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


print("\npriceSource is optional, nests an object, and is escaped like any other field")

# No row in data/gpus.json carries priceSource yet (see _meta.schema.priceSource) — it
# is exercised here with synthetic rows so the wiring is proven before tools/price_check.py
# ever writes one for real.
def check_price_source_round_trips_js():
    rows = {
        "8b-h100-80": {"gb": 80, "bw": 3352, "hyper": 12.3, "spec": 3.99, "spot": 2.25,
                       "tflops": 990, "name": "H100 80 GB", "vendor": "nvidia", "perfKey": "nvidia",
                       "devices": 1, "form": "sxm", "caps": {"fp8": True},
                       "priceSource": {"hyper": {"provider": "azure",
                                                  "sku": "Standard_ND96isr_H100_v5",
                                                  "region": "eastus", "date": "2026-09-16"}}},
        # A second row with no priceSource at all: the optional field must not be
        # required just because a sibling row happens to carry it.
        "8b-a100-80": {"gb": 80, "bw": 2039, "hyper": 4.5, "spec": 1.79, "spot": 0.99,
                       "tflops": 312, "name": "A100 80 GB", "vendor": "nvidia", "perfKey": "nvidia",
                       "devices": 1, "form": "sxm", "caps": {"fp8": False}},
    }
    block = sync_data.render_gpu_js(rows)
    body = "\n".join(l for l in block.splitlines() if not l.startswith("/*"))
    assert js_eval(body, "GPU_TABLE") == rows, "priceSource did not round-trip through JS"

test("a row with priceSource round-trips through JS beside a row without it",
     check_price_source_round_trips_js)


def check_price_source_round_trips_py():
    rows = {
        "8b-h100-80": {"gb": 80, "bw": 3352, "hyper": 12.3, "spec": 3.99, "spot": 2.25,
                       "tflops": 990, "name": "H100 80 GB", "vendor": "nvidia", "perfKey": "nvidia",
                       "devices": 1, "form": "sxm", "caps": {"fp8": True},
                       "priceSource": {"hyper": {"provider": "azure",
                                                  "sku": "Standard_ND96isr_H100_v5",
                                                  "region": "eastus", "date": "2026-09-16"},
                                       "spec": {"provider": "lambda", "sku": "NVIDIA H100 SXM",
                                                "region": "global", "date": "2026-09-16"}}},
    }
    block = sync_data.render_gpu_py(rows)
    body = "\n".join(l for l in block.splitlines() if not l.startswith("#"))
    ns = {}
    exec(body, ns)
    assert ns["GPUS"] == rows, "priceSource did not round-trip through Python"

test("priceSource with multiple tiers round-trips through Python",
     check_price_source_round_trips_py)


def check_a_real_price_source_reaches_both_generated_blocks():
    """The two tests above prove the wiring with synthetic rows, because when
    they were written no row in data/gpus.json carried priceSource yet
    (commit 6's own note: "0/12 populated"). fix/cost-provenance populates
    real ones, so this checks the actual committed files rather than a fresh
    render: at least one real row's SKU and priceSource.price must appear
    verbatim, scoped to that row's own line, in both committed files — not
    just in what sync_data.py *would* produce (check_sync_is_a_noop_on_a_clean_tree
    below proves those are the same thing, but this is what "the same thing"
    is actually worth) and not just anywhere in a 200KB file.

    An earlier version of this test checked str(price) against the WHOLE
    file with no row scoping and passed by accident: t4-16/hyper's price,
    0.53, is common enough to turn up elsewhere in index.html by chance, so
    the assertion was satisfied whether or not priceSource actually reached
    the block it claimed to check. Caught by sabotaging
    tools/sync_data.py's GPU_OPTIONAL to drop "priceSource" and re-running
    sync_data.py: every other test in this file went red: this one did not.
    Scoping the match to the row's own extracted line, and to the SKU (a
    long provider-specific string with nothing else in the file to collide
    with) rather than a bare number, is what makes it a real check."""
    gpus = sync_data.load_gpus()
    sourced = [(slug, tier) for slug, row in gpus.items()
               for tier in row.get("priceSource", {})]
    assert sourced, "no row in the real data/gpus.json carries priceSource — nothing to check"
    slug, tier = sourced[0]
    src = gpus[slug]["priceSource"][tier]
    assert "price" in src, f"{slug}/{tier}: real priceSource has no price field"
    assert len(src["sku"]) >= 6, f"{slug}/{tier}: sku {src['sku']!r} too short to be a reliable anchor"

    # The row's own line in each generated block — the same "one GPU per
    # line" shape tools/price_check.py's apply_to_text relies on — not the
    # file as a whole, so a match cannot land in some unrelated row or notes.
    row_patterns = {
        "index.html": re.compile(r"^\s*'" + re.escape(slug) + r"':\s*\{.*\},?\s*$", re.MULTILINE),
        "generate_report.py": re.compile(r'^\s*"' + re.escape(slug) + r'":\s*\{.*\},?\s*$', re.MULTILINE),
    }
    for label, rel in (("index.html", "index.html"), ("generate_report.py", "generate_report.py")):
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            text = f.read()
        m = row_patterns[label].search(text)
        assert m, f"{label}: {slug}'s row not found as a single line in the committed GPU_TABLE block"
        row_line = m.group(0)
        assert src["sku"] in row_line, (
            f"{label}: {slug}/{tier}'s SKU {src['sku']!r} not found on {slug}'s own generated line")
        assert re.search(r"price['\"]?\s*:\s*" + re.escape(str(src["price"])) + r"\b", row_line), (
            f"{label}: {slug}/{tier}'s priceSource.price ({src['price']}) not found as a price: "
            f"field on {slug}'s own generated line: {row_line[:300]}")

test("a real row's priceSource (SKU and price) reaches both committed generated blocks, on that row's own line",
     check_a_real_price_source_reaches_both_generated_blocks)


def check_price_source_free_text_is_escaped():
    # provider/sku/region ultimately come from a fetched page (a Lambda plan
    # name, an Azure meterName): contributor-adjacent free text, same class as
    # a benchmark note, sitting inside the same <script> element in index.html.
    hostile = "</script><script>alert(1)</script>"
    rows = {"8b-h100-80": {"gb": 80, "bw": 3352, "hyper": 12.3, "spec": 3.99, "spot": 2.25,
                           "tflops": 990, "name": "H100 80 GB", "vendor": "nvidia", "perfKey": "nvidia",
                           "devices": 1, "form": "sxm", "caps": {"fp8": True},
                           "priceSource": {"hyper": {"provider": "azure", "sku": hostile,
                                                      "region": "eastus", "date": "2026-09-16"}}}}
    block = sync_data.render_gpu_js(rows)
    body = "\n".join(l for l in block.splitlines() if not l.startswith("/*"))
    assert js_eval(body, "GPU_TABLE") == rows, "value changed in transit"
    assert "</script" not in body.lower(), "rendered block can close its own script tag"

test("a hostile priceSource sku cannot close the script tag",
     check_price_source_free_text_is_escaped)


def check_marker_text_nested_in_price_source_is_refused():
    """The marker check stringifies the whole field value before searching it
    (str(row[f])), which is what lets it see into a nested dict at all —
    caps never carries free text, so priceSource is the first field where a
    marker hiding inside a *nested* value, not the top-level field, matters."""
    poisoned = "see the GPU_TABLE:END marker in tools/sync_data.py"
    rows = {"8b-h100-80": {"gb": 80, "bw": 3352, "hyper": 12.3, "spec": 3.99, "spot": 2.25,
                           "tflops": 990, "name": "H100 80 GB", "vendor": "nvidia", "perfKey": "nvidia",
                           "devices": 1, "form": "sxm", "caps": {"fp8": True},
                           "priceSource": {"hyper": {"provider": "azure", "sku": poisoned,
                                                      "region": "eastus", "date": "2026-09-16"}}}}
    try:
        sync_data.render_gpu_js(rows)
    except SystemExit as e:
        assert "8b-h100-80" in str(e) and "priceSource" in str(e), \
            f"error names neither the row nor the field: {e}"
        return
    raise AssertionError("a marker hidden inside a nested priceSource value rendered anyway")

test("a block marker nested inside priceSource is refused, naming the row and field",
     check_marker_text_nested_in_price_source_is_refused)


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


def check_a_gpu_row_without_perf_key_is_refused():
    """data/gpus.json's schema says perfKey is required. That is what stops an
    omission reaching both engines as a card with no throughput constants that
    nobody decided it should be. Both renderers, since each is its own way into
    a generated block. The message is matched on the missing list itself: every
    refusal also lists all required fields, perfKey among them, so merely
    finding the word would pass whatever was missing."""
    row = {"gb": 80, "bw": 3352, "hyper": 12.3, "spec": 3.99, "spot": 2.25,
           "tflops": 990, "name": "H100 80 GB", "vendor": "nvidia",
           "devices": 1, "form": "sxm", "caps": {"fp8": True}}
    for render in (sync_data.render_gpu_js, sync_data.render_gpu_py):
        try:
            render({"h100-80": row})
        except SystemExit as e:
            assert "'h100-80'" in str(e) and "missing required field(s) ['perfKey']" in str(e), (
                f"{render.__name__}: the refusal does not name the row and perfKey: {e}")
            continue
        raise AssertionError(f"{render.__name__} rendered a GPU row that has no perfKey")

test("a GPU row with no perfKey is refused by both renderers, naming the row and the field",
     check_a_gpu_row_without_perf_key_is_refused)


print("\nA price tier with no confirmed price is null, and nothing else may be")
NULL_TIER_ROW = {"gb": 256, "bw": 6000, "hyper": None, "spec": 3.8, "spot": None,
                 "tflops": 1307.4, "name": "Null-tier 256 GB", "vendor": "amd", "perfKey": "cdna3",
                 "devices": 1, "form": "oam", "caps": {"fp8": True}}


def check_a_null_tier_round_trips_through_both_languages():
    """A tier with no confirmed hourly price is null in data/gpus.json and has to
    arrive in each engine as that language's null — JS null, Python None — not
    as 0, not as a string, and not dropped from the row."""
    rows = {"probe-null": NULL_TIER_ROW}
    assert js_eval(sync_data.render_gpu_js(rows), "GPU_TABLE") == rows, "the JS block did not round-trip"
    block = sync_data.render_gpu_py(rows)
    ns = {}
    exec("\n".join(l for l in block.splitlines() if not l.startswith("#")), ns)
    assert ns["GPUS"] == rows, "the Python block did not round-trip"
    assert '"hyper":None' in block and "hyper:null" in sync_data.render_gpu_js(rows), (
        "a null tier did not render as each language's null literal")


test("a null price tier renders as null and None, and round-trips", check_a_null_tier_round_trips_through_both_languages)


def check_null_outside_a_price_tier_is_refused():
    """Only hyper/spec/spot may be null. Anywhere else a null is a value nobody
    decided to empty — tflops, the name, perfKey, a nested priceSource field —
    and both renderers refuse it, naming the row and the field."""
    cases = [dict(NULL_TIER_ROW, tflops=None), dict(NULL_TIER_ROW, perfKey=None),
             dict(NULL_TIER_ROW, spec=3.8, priceSource={"spec": {"provider": "azure", "sku": None,
                                                                   "region": "eastus", "date": "2026-09-23",
                                                                   "price": 3.8}})]
    for row in cases:
        for render in (sync_data.render_gpu_js, sync_data.render_gpu_py):
            try:
                render({"probe-null": row})
            except SystemExit as e:
                assert "'probe-null'" in str(e) or "None" in str(e), f"{render.__name__}: {e}"
                continue
            raise AssertionError(f"{render.__name__} rendered a null outside a price tier: {row}")


test("a null anywhere but a price tier is refused by both renderers", check_null_outside_a_price_tier_is_refused)


def check_a_price_record_round_trips_through_both_languages():
    """priceRecord is optional like priceSource: written only on the rows that
    carry it, and intact — URL included — in both generated blocks."""
    rec = {"provider": "RunPod", "sku": "MI300X (Secure Cloud)", "region": "global",
           "date": "2026-09-23", "price": 2.39, "url": "https://www.runpod.io/gpu-models/mi300x"}
    rows = {"probe-rec": dict(NULL_TIER_ROW, spec=2.39, priceRecord={"spec": rec}),
            "probe-plain": dict(NULL_TIER_ROW)}
    assert js_eval(sync_data.render_gpu_js(rows), "GPU_TABLE") == rows, "the JS block did not round-trip"
    block = sync_data.render_gpu_py(rows)
    ns = {}
    exec("\n".join(l for l in block.splitlines() if not l.startswith("#")), ns)
    assert ns["GPUS"] == rows, "the Python block did not round-trip"
    assert "priceRecord" not in sync_data.render_gpu_js({"probe-plain": NULL_TIER_ROW}), (
        "a row without priceRecord was given one")


test("priceRecord is optional and round-trips through both languages, URL included",
     check_a_price_record_round_trips_through_both_languages)


def check_a_price_note_round_trips_through_both_languages():
    """priceNote is optional like the other two provenance kinds: written only on
    the rows that carry it, and intact in both generated blocks. The reasons
    are prose, so the probe carries what prose carries: an apostrophe, double
    quotes, an em dash and a colon, each of which a string renderer can
    mangle in one language and not the other."""
    note = {"reason": "Lambda's and CoreWeave's \"A6000\" is the older card — not this one: see the docs.",
            "checked": "2026-09-16"}
    rows = {"probe-note": dict(NULL_TIER_ROW, priceNote={"hyper": note}),
            "probe-plain": dict(NULL_TIER_ROW)}
    assert js_eval(sync_data.render_gpu_js(rows), "GPU_TABLE") == rows, "the JS block did not round-trip"
    block = sync_data.render_gpu_py(rows)
    ns = {}
    exec("\n".join(l for l in block.splitlines() if not l.startswith("#")), ns)
    assert ns["GPUS"] == rows, "the Python block did not round-trip"
    assert "priceNote" not in sync_data.render_gpu_js({"probe-plain": NULL_TIER_ROW}), (
        "a row without priceNote was given one")


test("priceNote is optional and round-trips through both languages, prose intact",
     check_a_price_note_round_trips_through_both_languages)


def check_a_price_lead_round_trips_through_both_languages():
    """priceLead is optional too, and is the only field that holds a list of
    objects: written only where a row carries it, intact in both generated blocks."""
    lead = {"provider": "Runcrate", "price": 0.82, "url": "https://www.runcrate.ai/pricing/gpu/mi210",
            "date": "2026-09-23", "why": "Runcrate's own pricing page lists no AMD GPU.",
            "about": "https://github.com/x/y/blob/HEAD/docs/research/runcrate-due-diligence.md"}
    rows = {"probe-lead": dict(NULL_TIER_ROW, priceLead={"hyper": [lead, dict(lead, provider="Other", price=1.5)]}),
            "probe-plain": dict(NULL_TIER_ROW)}
    assert js_eval(sync_data.render_gpu_js(rows), "GPU_TABLE") == rows, "the JS block did not round-trip"
    block = sync_data.render_gpu_py(rows)
    ns = {}
    exec("\n".join(l for l in block.splitlines() if not l.startswith("#")), ns)
    assert ns["GPUS"] == rows, "the Python block did not round-trip"
    assert "priceLead" not in sync_data.render_gpu_js({"probe-plain": NULL_TIER_ROW}), "a row without priceLead was given one"


test("priceLead is optional and round-trips through both languages, a list of leads intact",
     check_a_price_lead_round_trips_through_both_languages)


def check_marker_text_in_a_value_is_refused():
    """A note discussing this tool by name is ordinary contributor prose.

    Before the marker had to sit on a comment line, such a note ended the
    block as far as the replacement regex was concerned: the next sync spliced
    a new block into the middle of the old one and left an orphaned `};`, so
    the whole inline script stopped parsing and nothing on the page ran.
    """
    poisoned = "see the BENCHMARK_DATA:END marker in tools/sync_data.py"
    rows = {"8b-h100-80": {"tokS": 1, "mode": "batch", "src": "x", "note": poisoned,
                           "prec": "bf16", "date": "2026-01", "url": ""}}
    try:
        sync_data.render_benchmark_js(rows)
    except SystemExit as e:
        assert "8b-h100-80" in str(e) and "note" in str(e), f"error names neither row nor field: {e}"
        return
    raise AssertionError("a value carrying a block marker was rendered anyway")

test("a value containing a block marker is refused, naming the row and field",
     check_marker_text_in_a_value_is_refused)


def check_a_data_line_cannot_end_the_block():
    """The regex itself, independently of the refusal above: an indented data
    line carrying the end tag must not terminate the match, or a file that
    somehow acquires one is unrecoverable by re-running the tool."""
    body = (
        "/* GPU_TABLE:BEGIN — generated */\n"
        "const GPU_TABLE = {\n"
        "  'x': { note: 'GPU_TABLE:END is just text here' },\n"
        "};\n"
        "/* GPU_TABLE:END */\n"
        "after = 1\n"
    )
    m = sync_data.BLOCK_RES["GPU_TABLE"].search(body)
    assert m, "the real marker pair no longer matches"
    assert m.group(0).rstrip().endswith("/* GPU_TABLE:END */"), (
        "the match stopped at a data line instead of the closing comment:\n" + m.group(0))
    assert "after = 1" not in m.group(0), "the match ran past the end marker"

test("a data line carrying the end tag does not end the block",
     check_a_data_line_cannot_end_the_block)


def check_control_characters_cannot_reach_the_page():
    """A raw NUL never reaches JavaScript: the HTML tokenizer rewrites it to
    U+FFFD first, so the note a contributor wrote is not the note the page
    shows. Node's evaluator does not tokenise HTML and cannot see this, so the
    check is on the rendered text itself."""
    for raw in ("null\x00byte", "bell\x07here", "vertical\x0btab"):
        rows = {"8b-h100-80": {"tokS": 1, "mode": "batch", "src": "x", "note": raw,
                               "prec": "bf16", "date": "2026-01", "url": ""}}
        block = sync_data.render_benchmark_js(rows)
        for ch in block:
            assert ch >= " " or ch in "\n", (
                f"a raw control character (U+{ord(ch):04X}) reached the generated block")
        body = "\n".join(l for l in block.splitlines() if not l.startswith("/*"))
        assert js_eval(body, "BENCHMARK_DATA") == rows, f"{raw!r} did not round-trip"

test("control characters are escaped, not passed through as raw bytes",
     check_control_characters_cannot_reach_the_page)


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
