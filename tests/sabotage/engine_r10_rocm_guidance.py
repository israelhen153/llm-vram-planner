#!/usr/bin/env python3
"""Round 10: feat/rocm-guidance. Sabotages against the branch that:
- gave every price tier a note saying how its figure was reached;
- showed unconfirmed prices as leads used in nothing;
- made the overhead text say the figure the math charges, in the vendor's words;
- ran AMD cards in vLLM's own ROCm image with cited guidance under the command;
- refused FP8 weights where vLLM has no FP8 weight kernel.

Families: F, the FP8 gate; R, the ROCm command and its lines; N, the notes;
L, the leads; O, the overhead's words; P, the prices unit 7 moved; T, the two
engines' ROCm tables. Every catalog probe is derived from data/gpus.json: the
gated rows are the AMD rows whose target is not gfx942, and the noted and led
tiers are the ones the catalog gives. The ones marked "both" edit the two
engines alike, so parity cannot be what catches them.

Usage: python3 tests/sabotage/engine_r10_rocm_guidance.py [name-substring ...]
       python3 tests/sabotage/engine_r10_rocm_guidance.py --from R1
"""
import json, os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver, ROOT
from anchors import (INDEX_HTML, REPORT_PY, PRICE_PY, GPUS_JSON, JS_EXEC_COSTS,
                     JS_R10_FP8_LINE, JS_R10_BLOCKED, JS_R10_BUILD_REFUSE, JS_R10_ARCH, JS_R10_AITER,
                     JS_R10_IMAGE, JS_R10_GCD, JS_R10_PANEL_ROCM, JS_R10_EXPORT_ROCM, JS_R10_PEER,
                     JS_R10_PEER_TEXT, JS_R10_SYNC_FORCE, JS_R10_SYNC_GIVE, JS_R10_SYNC_CALL,
                     JS_R10_NOTE_TEXT, JS_R10_NOTE_LINE, JS_R10_LEAD_LINE, JS_R10_LEAD_GATE,
                     JS_R10_HIP_SUB, JS_R10_COST_SPOT, JS_R10_HOURLY_SPEC, JS_R10_GFX90A,
                     PY_R10_FP8_LINE, PY_R10_BLOCKED, PY_R10_REFUSE_RAISE, PY_R10_EXIT2,
                     PY_R10_MENU_FILTER, PY_R10_MENU_QUANT, PY_R10_JSON_FP8, PY_R10_ARCH,
                     PY_R10_LIBRARY, PY_R10_PEER, PY_R10_WHY, PY_R10_LEADS, PY_R10_PDF_ROCM,
                     PY_R10_NOTE_TEXT, PY_R10_HIP_SUB, PY_R10_IMAGE, PY_R10_HOURLY_SPEC,
                     PY_R10_CONTEXT_AMD, PY_R10_GFX90A, PY_R10_GFX942, PRICE_R10_NOTE_DROP)

# The catalog, as text and as data, so every probe comes from it.
TEXT = open(os.path.join(ROOT, GPUS_JSON)).read()
ROWS = json.loads(TEXT)["data"]
LINE = {slug: next(l.strip().rstrip(",") for l in TEXT.splitlines() if l.lstrip().startswith(f'"{slug}"'))
        for slug in ROWS}
GATED = [s for s, r in ROWS.items() if r["vendor"] == "amd" and r.get("gfx") != "gfx942"]
NOTED = [(s, t) for s, r in ROWS.items() for t in (r.get("priceNote") or {})]
LED = [(s, t) for s, r in ROWS.items() for t in (r.get("priceLead") or {})]
NULLED_NVIDIA = [s for s, r in ROWS.items() if r["vendor"] == "nvidia" and r["hyper"] is None]
assert GATED and NOTED and LED and NULLED_NVIDIA, "the catalog no longer has what these sabotages attack"


def row_edit(slug, old, new):
    """Replace old with new inside one catalog row's line, asserting it is there once."""
    line = LINE[slug]
    assert line.count(old) == 1, f"{slug}: {old!r} occurs {line.count(old)} times in its row"
    return (GPUS_JSON, line, line.replace(old, new), 1)


S = {}

# ---- F: the FP8 gate ----
S["F1 js: fp8WeightsBlocked() lets every card load FP8 weights"] = [(INDEX_HTML, JS_R10_BLOCKED, "  return '';\n", 1)]
S["F2 js: the gate keyed to gfx90a alone, so RDNA3 escapes"] = [
    (INDEX_HTML, JS_R10_BLOCKED, "  if (!gpu || gpu.vendor !== 'amd' || gpu.gfx !== 'gfx90a') return '';\n", 1)]
S["F3 js: the command builder's refusal removed"] = [(INDEX_HTML, JS_R10_BUILD_REFUSE, "  if (false)\n", 1)]
S["F4 js: syncPrecision() disables FP8 but never falls back to BF16"] = [
    (INDEX_HTML, JS_R10_SYNC_FORCE, "  if (false) {\n", 1)]
S["F5 js: syncPrecision() gives FP8 back over a choice the reader made after the fallback"] = [
    (INDEX_HTML, JS_R10_SYNC_GIVE, "    fp8.selected = true;\n", 1)]
S["F6 js: recalculate() no longer calls syncPrecision()"] = [
    (INDEX_HTML, JS_R10_SYNC_CALL, "  syncInterconnect();\n", 1)]
S["F7 js: the FP8 line dropped from the ROCm guidance"] = [(INDEX_HTML, JS_R10_FP8_LINE, "", 1)]
S["F8 py: fp8_weights_blocked() lets every card load FP8 weights"] = [(REPORT_PY, PY_R10_BLOCKED, "    if True:\n", 1)]
S["F9 py: the refusal raises nothing"] = [(REPORT_PY, PY_R10_REFUSE_RAISE, "            pass\n", 1)]
S["F10 py: the CLI exits 1, not 2"] = [
    (REPORT_PY, PY_R10_EXIT2, PY_R10_EXIT2.replace("sys.exit(2)", "sys.exit(1)"), 1)]
S["F11 py: the interactive menu offers FP8 on a gated card"] = [(REPORT_PY, PY_R10_MENU_FILTER, "        pass\n", 1)]
S["F12 py: the FP8 line dropped from the PDF's ROCm lines"] = [(REPORT_PY, PY_R10_FP8_LINE, "", 1)]
for slug in GATED:
    S[f"F13 data: {slug}'s gfx set to gfx942, so FP8 weights and AITER open up"] = [
        row_edit(slug, f'"gfx": "{ROWS[slug]["gfx"]}"', '"gfx": "gfx942"')]
S[f"F14 data: {GATED[0]} loses its gfx"] = [row_edit(GATED[0], f', "gfx": "{ROWS[GATED[0]]["gfx"]}"', "")]
S["F15 both: gfx90a's fp8Weights flipped to true in both ROCm tables"] = [
    (INDEX_HTML, JS_R10_GFX90A, JS_R10_GFX90A.replace('"fp8Weights": false', '"fp8Weights": true'), 1),
    (REPORT_PY, PY_R10_GFX90A, PY_R10_GFX90A.replace("'fp8Weights': False", "'fp8Weights': True"), 1)]

# ---- R: the ROCm command and its lines ----
S["R1 js: the docker form given to NVIDIA cards too"] = [
    (INDEX_HTML, JS_R10_ARCH, "  const arch = (ROCM.arch[state.gfx] || {});\n", 1)]
S["R2 js: AITER on for every AMD card"] = [
    (INDEX_HTML, JS_R10_AITER, "      + '    --env VLLM_ROCM_USE_AITER=1 \\\\\\n'\n", 1)]
S["R3 js: AITER never on"] = [(INDEX_HTML, JS_R10_AITER, "      + ''\n", 1)]
S["R4 both: the image tag unpinned to :latest in both tables"] = [
    (INDEX_HTML, JS_R10_IMAGE, JS_R10_IMAGE.replace(":v0.30.0", ":latest"), 1),
    (REPORT_PY, PY_R10_IMAGE, PY_R10_IMAGE.replace(":v0.30.0", ":latest"), 1)]
S["R5 both: AMD's deprecated rocm/vllm image restored in both tables"] = [
    (INDEX_HTML, JS_R10_IMAGE, JS_R10_IMAGE.replace("vllm/vllm-openai-rocm:v0.30.0", "rocm/vllm:latest"), 1),
    (REPORT_PY, PY_R10_IMAGE, PY_R10_IMAGE.replace("vllm/vllm-openai-rocm:v0.30.0", "rocm/vllm:latest"), 1)]
S["R6 js: the two-GCD line on every AMD card"] = [(INDEX_HTML, JS_R10_GCD, "  out.push(L.gcd);\n", 1)]
S["R7 js: the command panel drops the ROCm lines"] = [(INDEX_HTML, JS_R10_PANEL_ROCM, "  const rocm = [];\n", 1)]
S["R8 js: the copied report drops the ROCm section"] = [
    (INDEX_HTML, JS_R10_EXPORT_ROCM, "  const rocmLines = [];\n", 1)]
S["R9 py: the PDF drops the ROCm lines"] = [(REPORT_PY, PY_R10_PDF_ROCM, "        rocm = []\n", 1)]
S["R10 py: the PDF gives NVIDIA cards the docker form"] = [
    (REPORT_PY, PY_R10_ARCH, '    arch = ROCM["arch"].get(gpu.get("gfx"), {})\n', 1)]
S["R11 both: the guidance says CUDA_VISIBLE_DEVICES where it says HIP_VISIBLE_DEVICES"] = [
    (INDEX_HTML, JS_R10_HIP_SUB, "CUDA_VISIBLE_DEVICES=0,1", 1),
    (REPORT_PY, PY_R10_HIP_SUB, "CUDA_VISIBLE_DEVICES=0,1", 1)]

# ---- N: the notes ----
S["N1 js: the cost table drops the spot tier's note"] = [
    (INDEX_HTML, JS_R10_COST_SPOT, "${priceLeadLines(state, 'spot')}</td>", 1)]
S["N2 js: the copied report drops the notes"] = [(INDEX_HTML, JS_R10_NOTE_LINE, "    + ''\n", 1)]
S["N3 py: the PDF drops its Why line"] = [(REPORT_PY, PY_R10_WHY, "            pass\n", 1)]
S["N4 js: the note loses the date it was checked"] = [
    (INDEX_HTML, JS_R10_NOTE_TEXT, "  return note ? `${note.reason}` : '';\n", 1)]
S["N5 py: the PDF's note loses the date it was checked"] = [
    (REPORT_PY, PY_R10_NOTE_TEXT, "    return f\"{note['reason']}\" if note else \"\"\n", 1)]
slug, tier = NOTED[0]
reason = ROWS[slug]["priceNote"][tier]["reason"]
S[f"N6 data: a dollar figure put in {slug}/{tier}'s note"] = [
    row_edit(slug, json.dumps(reason), json.dumps(reason[:-1] + ", about $0.99/hr."))]
S["N7 price: --apply keeps a note beside the reading it records"] = [(PRICE_PY, PRICE_R10_NOTE_DROP, "        pass\n", 1)]
slug = next(s for s, r in ROWS.items() if r["vendor"] == "nvidia" and "hyper" in (r.get("priceSource") or {}))
S[f"N8 data: {slug}'s automated hyperscaler tier also given a note"] = [
    row_edit(slug, '"priceNote": { ', '"priceNote": { "hyper": { "reason": "Stale.", "checked": "2026-09-23" }, ')]

# ---- L: the leads ----
S["L1 js: the monthly range counts a lead's price"] = [
    (INDEX_HTML, JS_EXEC_COSTS, "  const execCosts = [computed.hourlyHyper, computed.hourlySpec, computed.hourlySpot, "
     "...Object.values(state.priceLead || {}).flat().map(l => l.price * state.gpuCount)].filter(v => v != null);\n", 1)]
S["L2 js: a null specialized tier priced from its lead"] = [
    (INDEX_HTML, JS_R10_HOURLY_SPEC,
     "  const hourlySpec = tierCost(gpuSpecCost ?? (((state.priceLead || {}).spec || [])[0] || {}).price);\n", 1)]
S["L3 js: a lead shown on a tier that has a price"] = [(INDEX_HTML, JS_R10_LEAD_GATE, "  if (false) return [];\n", 1)]
S["L4 js: the copied report drops the leads"] = [(INDEX_HTML, JS_R10_LEAD_LINE, "    + '';\n", 1)]
S["L5 py: the PDF drops the leads"] = [(REPORT_PY, PY_R10_LEADS, "                pass\n", 1)]
S["L6 py: a null specialized tier priced from its lead"] = [
    (REPORT_PY, PY_R10_HOURLY_SPEC,
     "    hourly_spec = ((((gpu.get(\"priceLead\") or {}).get(\"spec\") or [{}])[0].get(\"price\") or 0) * n_gpu "
     "if gpu[\"spec\"] is None else gpu[\"spec\"] * n_gpu)\n", 1)]
slug, tier = LED[0]
priced = next(t for t in ("hyper", "spec", "spot") if ROWS["mi300x-192"][t] is not None)
S[f"L7 data: a lead put on mi300x-192's priced {priced} tier"] = [
    row_edit("mi300x-192", ', "priceSource": {',
             ', "priceLead": { "' + priced + '": [ { "provider": "X", "price": 1.0, "url": "https://x.example/p", '
             '"date": "2026-09-23", "why": "Test." } ] }, "priceSource": {')]

# ---- O: the overhead's words ----
S["O1 js: the notes say 0.3 GB/peer on every link again"] = [
    (INDEX_HTML, JS_R10_PEER_TEXT, JS_R10_PEER_TEXT.replace("${computed.peerBufferGB}", "0.3"), 1)]
S["O2 js: the notes say NCCL on AMD"] = [
    (INDEX_HTML, JS_R10_PEER_TEXT, JS_R10_PEER_TEXT.replace("${state.vendor === 'amd' ? 'RCCL' : 'NCCL'}", "NCCL"), 1)]
S["O3 py: the PDF says NCCL on AMD"] = [(REPORT_PY, PY_R10_LIBRARY, '            library = "NCCL"\n', 1)]
S["O4 py: the PDF calls it CUDA context on AMD"] = [(REPORT_PY, PY_R10_CONTEXT_AMD, "        if False:\n", 1)]
S["O5 js: the page charges 0.3 per peer on every link"] = [
    (INDEX_HTML, JS_R10_PEER, "  const peerBufferGB = deviceCount > 1 ? 0.3 : 0;\n", 1)]
S["O6 py: the PDF charges 0.3 per peer on every link, and says so"] = [
    (REPORT_PY, PY_R10_PEER, "    peer_buffer_gb = 0.3 if device_count > 1 else 0\n", 1)]

# ---- P: the prices unit 7 moved ----
for slug in NULLED_NVIDIA:
    S[f"P1 data: {slug}'s hyperscaler tier re-priced, with its note left saying nobody rents it"] = [
        row_edit(slug, '"hyper": null', '"hyper": 0.75')]
S["P2 data: rtxpro-96's hyperscaler reading dropped, leaving a price with no provenance"] = [
    row_edit("rtxpro-96", ', "hyper": { "provider": "aws", "sku": "g7e.2xlarge", "region": "US East (N. Virginia)", '
                          '"date": "2026-09-23", "price": 3.36 }', "")]

# ---- C: the PDF command names its quantization ----
S["C1 py: the interactive menu's choice carries no quantization"] = [
    (REPORT_PY, PY_R10_MENU_QUANT, '    bpp, quant = prec_opts[prec_choice][0], ""\n', 1)]
S["C2 py: a one-byte JSON config is left without fp8"] = [
    (REPORT_PY, PY_R10_JSON_FP8, PY_R10_JSON_FP8.replace('if cfg.get("bpp") == 1 and not cfg.get("quant"):', "if False:"), 1)]

# ---- T: the two engines' tables ----
S["T1 py: the Python table alone turns AITER off for gfx942"] = [
    (REPORT_PY, PY_R10_GFX942, PY_R10_GFX942.replace("'aiter': True", "'aiter': False"), 1)]

run_driver(S)
