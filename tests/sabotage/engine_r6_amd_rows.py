#!/usr/bin/env python3
"""Round 6: the AMD rows, attacked along the axes they introduce.

Round 3 recorded that every probe fixed vendor: 'nvidia' and a single unknown
key, so a leak gated on vendor === 'amd' or on a CDNA perfKey would have
passed. feat/amd-gpus puts five AMD rows in the catalog and, with them, new
axes a leak or a regression can ride:
- three architectures with no PERF entry;
- an OAM form and its fabric;
- a board of two devices;
- price tiers with no price;
- prices recorded by hand.

Every catalog probe below is derived from data/gpus.json: the AMD rows, their
keys, names, forms and device counts, their null tiers, their hand records and
their spot markers. A row added later is attacked the same way without editing
this file. The engine attacks anchor on tests/sabotage/anchors.py, so
tests/corpus.test.py fails in the pull request that moves their text.
"""
import json, os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver, ROOT
from anchors import (INDEX_HTML, REPORT_PY, PRICE_PY, GPUS_JSON, JS_LOOKUP, PY_LOOKUP,
                     JS_TIER_COST, JS_HOURLY_HYPER, JS_NULL_LABEL, JS_CMP_COSTS, JS_EXEC_COSTS, JS_USD_DASH,
                     JS_INTERCONNECT_NAME, JS_SHARDING_ONE, JS_SYNC_OAM_LABEL, JS_VENDORS, JS_STATE_FORM,
                     PY_HOURLY_HYPER, PY_NULL_LABEL, PY_INTERCONNECT_NAME, PY_INTERCONNECT_ROW, PY_MENU_SPAN,
                     PRICE_SPOT_SKU, PRICE_NULL_GUARD, PRICE_NULL_REFUSAL)
# The round-3 leak shapes, reused so a leak here is the same leak it was there.
from engine_r3_fixed_assumptions import leak_into_throughput_tile, leak_into_pdf_explanation

with open(os.path.join(ROOT, GPUS_JSON), encoding="utf-8") as f:
    TEXT = f.read()
CATALOG = json.loads(TEXT)["data"]
AMD = {slug: row for slug, row in CATALOG.items() if row["vendor"] == "amd"}
assert len(AMD) >= 5, f"only {len(AMD)} AMD rows in data/gpus.json; this driver attacks the rows feat/amd-gpus added"
TIERS = ("hyper", "spec", "spot")
KEYS = sorted({r["perfKey"] for r in AMD.values()})
MULTI = sorted({r["devices"] for r in AMD.values() if r["devices"] > 1})
FORMS = sorted({r["form"] for r in AMD.values()})
LINE = {slug: next(l for l in TEXT.split("\n") if l.startswith(f"    {json.dumps(slug)}: ")) for slug in AMD}
# The derived lists must be non-empty, or the probes built from them attack nothing.
assert KEYS and MULTI and "oam" in FORMS, (KEYS, MULTI, FORMS)
assert any(r[t] is None for r in AMD.values() for t in TIERS), "no AMD row has a null tier to attack"
assert any(r.get("priceRecord") for r in AMD.values()), "no AMD row has a hand record to attack"


def row_edit(slug, old, new):
    """One edit inside one catalog row's line, refusing an `old` the line does not hold once."""
    assert LINE[slug].count(old) == 1, f"{slug}: {old!r} occurs {LINE[slug].count(old)} times in its row"
    return (GPUS_JSON, LINE[slug], LINE[slug].replace(old, new), 1)


S = {}

# ---- throughput leaks gated on the axes the AMD rows introduce --------------------
S["A1 js: a borrowed estimate printed when state.vendor === 'amd'"] = leak_into_throughput_tile(
    "state.vendor === 'amd'")
S["A1 py: a borrowed estimate printed when the vendor is amd"] = leak_into_pdf_explanation(
    'cfg.get("vendor") == "amd"')
for k in KEYS:
    S[f"A2 js: a borrowed estimate only for the real perfKey {k!r}"] = leak_into_throughput_tile(
        f"state.perfKey === '{k}'")
    S[f"A2 py: a borrowed estimate only for the real perfKey {k!r}"] = leak_into_pdf_explanation(
        f'cfg.get("perfKey") == "{k}"')
for d in MULTI:
    S[f"A3 js: a borrowed estimate only on a board of {d} devices"] = leak_into_throughput_tile(
        f"state.gpuDevices === {d}")
    S[f"A3 py: a borrowed estimate only on a board of {d} devices"] = leak_into_pdf_explanation(
        f'cfg["gpu"].get("devices") == {d}')
for form in FORMS:
    S[f"A4 js: a borrowed estimate only on form {form!r}"] = leak_into_throughput_tile(
        f"state.gpuForm === '{form}'")
    S[f"A4 py: a borrowed estimate only on form {form!r}"] = leak_into_pdf_explanation(
        f'cfg["gpu"].get("form") == "{form}"')
for slug, r in AMD.items():
    # The page's state carries "MI250X 128GB" (getGpuSpec() drops the space);
    # the catalog, the PDF and a hand-built test state carry "MI250X 128 GB".
    # Both spellings, because a leak keyed on either one is a real leak.
    shown = r["name"][:-3] + "GB" if r["name"].endswith(" GB") else r["name"]
    S[f"A5 js: a borrowed estimate only for {shown!r}, the name the page's state carries"] = \
        leak_into_throughput_tile(f"state.gpuName === '{shown}'")
    S[f"A5 js: a borrowed estimate only for {r['name']!r}, the catalog's spelling"] = \
        leak_into_throughput_tile(f"state.gpuName === '{r['name']}'")
    S[f"A5 py: a borrowed estimate only for {r['name']!r}"] = leak_into_pdf_explanation(
        f'gpu["name"] == "{r["name"]}"')
S["A6 js: a borrowed estimate only on a card whose hyperscaler tier is null"] = leak_into_throughput_tile(
    "state.gpuHyperCost === null")
S["A6 py: a borrowed estimate only on a card whose hyperscaler tier is null"] = leak_into_pdf_explanation(
    'gpu.get("hyper") is None')

# ---- AMD borrowing NVIDIA's constants ----------------------------------------------
S["B1 js: a card of vendor amd with no PERF entry borrows nvidia's constants"] = [
    (INDEX_HTML, JS_LOOKUP,
     JS_LOOKUP.replace("? PERF[state.perfKey] : null;", "? PERF[state.perfKey] : (state.vendor === 'amd' ? PERF.nvidia : null);"), 1)]
S["B1 py: a card of vendor amd with no PERF entry borrows nvidia's constants"] = [
    (REPORT_PY, PY_LOOKUP,
     PY_LOOKUP + '    if P is None and cfg.get("vendor") == "amd":\n        P = PERF["nvidia"]\n', 1)]

# ---- the catalog, row by row ----------------------------------------------------------
for slug, r in AMD.items():
    fp8 = r["caps"]["fp8"]
    S[f"C1 data: {slug} caps.fp8 {fp8} -> {not fp8}"] = [
        row_edit(slug, f'"fp8": {json.dumps(fp8)}', f'"fp8": {json.dumps(not fp8)}')]
    S[f"C2 data: {slug} perfKey {r['perfKey']!r} -> 'nvidia', borrowing through the catalog"] = [
        row_edit(slug, f'"perfKey": "{r["perfKey"]}"', '"perfKey": "nvidia"')]
    # The shape of the MI250X error the research carried: a TFLOPS figure off by
    # 2x, on a card whose throughput is not modelled and so shows it nowhere.
    S[f"C3 data: {slug} tflops doubled, {r['tflops']} -> {round(r['tflops'] * 2, 1)}"] = [
        row_edit(slug, f'"tflops": {json.dumps(r["tflops"])}', f'"tflops": {json.dumps(round(r["tflops"] * 2, 1))}')]
    for t in TIERS:
        if r[t] is None:
            S[f"C4 data: {slug} {t} null -> 1.23, a price no page confirmed"] = [
                row_edit(slug, f'"{t}": null', f'"{t}": 1.23')]
    if r["devices"] > 1:
        S[f"C5 data: {slug} devices {r['devices']} -> 1, the board read as one device"] = [
            row_edit(slug, f'"devices": {r["devices"]}', '"devices": 1')]
    if r["form"] == "oam":
        S[f"C6 data: {slug} form oam -> sxm, claiming NVLink"] = [row_edit(slug, '"form": "oam"', '"form": "sxm"')]
        S[f"C6 data: {slug} form oam -> pcie, its fabric called PCIe"] = [row_edit(slug, '"form": "oam"', '"form": "pcie"')]
    S[f"C7 data: {slug} vendor amd -> nvidia"] = [row_edit(slug, '"vendor": "amd"', '"vendor": "nvidia"')]
    for t, rec in (r.get("priceRecord") or {}).items():
        to = next(x for x in TIERS if x != t)
        S[f"C8 data: {slug} priceRecord moved from {t} to {to}"] = [
            row_edit(slug, f'"priceRecord": {{ "{t}": ', f'"priceRecord": {{ "{to}": ')]
        S[f"C9 data: {slug}/{t} priceRecord url over plain http"] = [
            row_edit(slug, f'"url": "{rec["url"]}"', f'"url": "{rec["url"].replace("https://", "http://")}"')]
        S[f"C10 data: {slug}/{t} the catalog price no longer what the hand record read"] = [
            row_edit(slug, f'"{t}": {json.dumps(r[t])}', f'"{t}": {json.dumps(round(r[t] + 0.1, 2))}')]
        S[f"C11 data: {slug}/{t} priceRecord dated tomorrow-and-then-some"] = [
            row_edit(slug, f'"date": "{rec["date"]}", "price": {json.dumps(rec["price"])}, "url"',
                     f'"date": "2099-01-01", "price": {json.dumps(rec["price"])}, "url"')]
    for t, src in (r.get("priceSource") or {}).items():
        if " (" in src["sku"]:
            bare = src["sku"].split(" (")[0]
            S[f"C12 data: {slug}/{t} priceSource sku loses its marker, {src['sku']!r} -> {bare!r}"] = [
                row_edit(slug, f'"sku": "{src["sku"]}"', f'"sku": "{bare}"')]

# ---- the engines' null price tier ------------------------------------------------------
S["N1 js: a null tier costs $0 (null * gpuCount)"] = [
    (INDEX_HTML, JS_TIER_COST, "  const tierCost = (perBoard) => (perBoard || 0) * gpuCount;\n", 1)]
S["N2 js: a null hyperscaler tier borrows the specialized price"] = [
    (INDEX_HTML, JS_HOURLY_HYPER, "  const hourlyHyper = tierCost(gpuHyperCost ?? gpuSpecCost);\n", 1)]
S["N2 py: a null hyperscaler tier borrows the specialized price, or $0"] = [
    (REPORT_PY, PY_HOURLY_HYPER,
     '    hourly_hyper = (gpu["hyper"] if gpu["hyper"] is not None else gpu["spec"] or 0) * n_gpu\n', 1)]
S["N3 js: a null tier's label says 'not recorded'"] = [(INDEX_HTML, JS_NULL_LABEL, "", 1)]
S["N3 py: a null tier's label says 'not recorded'"] = [(REPORT_PY, PY_NULL_LABEL, "", 1)]
S["N4 js: the snapshot's Cost/hr range counts a null tier as $0"] = [
    (INDEX_HTML, JS_CMP_COSTS, JS_CMP_COSTS.replace(".filter(v => v != null)", ".map(v => v || 0)"), 1)]
S["N5 js: the executive Monthly cost range counts a null tier as $0"] = [
    (INDEX_HTML, JS_EXEC_COSTS, JS_EXEC_COSTS.replace(".filter(v => v != null)", ".map(v => v || 0)"), 1)]
S["N6 js: the cost table prints $0.00 where a null tier has no figure"] = [
    (INDEX_HTML, JS_USD_DASH, "  if (v == null) return '$0.00';\n", 1)]
S["N7 py: the CLI menu prints the old spot-hyperscaler span for every card ($None)"] = [
    (REPORT_PY, PY_MENU_SPAN, "        if True:\n", 1)]

# ---- the interconnect's name, and the vendor sections -----------------------------------
S["F1 js: interconnectName() forgets OAM, so an Infinity Fabric board reads PCIe"] = [
    (INDEX_HTML, JS_INTERCONNECT_NAME, "  return state.hasNVLink ? 'NVLink' : 'PCIe';\n", 1)]
S["F1 py: interconnect_name() forgets OAM"] = [(REPORT_PY, PY_INTERCONNECT_NAME, '    return "PCIe"\n', 1)]
S["F2 js: the one-group sharding line reads hasNVLink again, bypassing the name"] = [
    (INDEX_HTML, JS_SHARDING_ONE,
     JS_SHARDING_ONE.replace("fabric !== 'PCIe' ? fabric + ' — sharded '", "state.hasNVLink ? 'NVLink — sharded '"), 1)]
S["F3 py: the PDF's Interconnect row reads nvlink again, bypassing the name"] = [
    (REPORT_PY, PY_INTERCONNECT_ROW, '            ["Interconnect", "NVLink" if cfg.get("nvlink") else "PCIe"],\n', 1)]
S["F4 js: the control's fallback option still says 'PCIe only' on an OAM board"] = [
    (INDEX_HTML, JS_SYNC_OAM_LABEL, "", 1)]
S["F5 js: readInputState() drops the form, so the real page names no fabric"] = [
    (INDEX_HTML, JS_STATE_FORM, "", 1)]
S["G1 js: the dropdown keeps only the first vendor, so no sections are drawn"] = [
    (INDEX_HTML, JS_VENDORS, "  const vendors = [...new Set(rows.map(([, gpu]) => gpu.vendor))].slice(0, 1);\n", 1)]

# ---- price_check's rules for the new states ------------------------------------------------
S["P1 price: a spot reading's SKU loses (Spot)"] = [(PRICE_PY, PRICE_SPOT_SKU, "    recorded_sku = sku\n", 1)]
S["P2 price: run() fetches a tier the catalog records as null"] = [
    (PRICE_PY, PRICE_NULL_GUARD, "            if False:\n", 1)]
S["P3 price: main() lets SOURCE_MAP automate a null tier"] = [
    (PRICE_PY, PRICE_NULL_REFUSAL, PRICE_NULL_REFUSAL.replace("    if on_null:\n", "    if False:\n"), 1)]

if __name__ == "__main__":
    run_driver(S)
