#!/usr/bin/env python3
"""Round 7: what the first cold check of feat/amd-gpus found surviving.

A checker with no prior context reintroduced the unit's bugs one at a time
against 244b236. Of 56 sabotages, 45 were caught and 11 survived, in four gaps
the builder's round (engine_r6_amd_rows) had not reached:

1. **The page's catalog-to-state path.** Every null-tier test built its state
   by hand. So getGpuSpec() or readInputState() filling a null tier (from a
   neighbour, or with 0) printed the wrong price on the live page with every
   test green.
2. **The hand record's lookup.** Hard-wired to the spec tier, it passed in both
   engines while every real record sat on spec.
3. **The fabric above one board.** A gate keyed on a multi-device board at
   more than one board renamed the MI250X's fabric, and no test named that
   fabric above one board.
4. **price_check's writer.** It carried no fixture with a hand record, so
   dropping the record, or its URL, on --apply passed.

The checker also noted a record whose URL was a site's front page passing.
Each is kept here so the next change starts where this check ended.
"""
import json, os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver, ROOT
from anchors import (INDEX_HTML, REPORT_PY, PRICE_PY, GPUS_JSON, JS_GETSPEC_PRICES, JS_STATE_SPOT,
                     JS_REC_LOOKUP, JS_FABRIC, PY_REC_LOOKUP, PY_INTERCONNECT_NAME, PRICE_DUMP_ROW,
                     PRICE_DUMP_VALUE, PRICE_APPLY_MOVED)

with open(os.path.join(ROOT, GPUS_JSON), encoding="utf-8") as f:
    TEXT = f.read()
CATALOG = json.loads(TEXT)["data"]
# A hand record's URL, derived: the first real one.
REC_URL = next(rec["url"] for row in CATALOG.values() for rec in (row.get("priceRecord") or {}).values())
FRONT = REC_URL.split("/", 3)[0] + "//" + REC_URL.split("/", 3)[2] + "/"
assert REC_URL != FRONT and TEXT.count(json.dumps(REC_URL)) == 1, REC_URL

S = {
    # ---- 1. the page's catalog-to-state path ----
    "N10 js: getGpuSpec fills a null spot tier with the specialized price": [
        (INDEX_HTML, JS_GETSPEC_PRICES, JS_GETSPEC_PRICES.replace("spot: g.spot,", "spot: g.spot ?? g.spec,"), 1)],
    "N13 js: getGpuSpec turns a null hyperscaler tier into $0": [
        (INDEX_HTML, JS_GETSPEC_PRICES, JS_GETSPEC_PRICES.replace("hyper: g.hyper,", "hyper: g.hyper ?? 0,"), 1)],
    "N14 js: readInputState fills a null spot tier with the specialized price": [
        (INDEX_HTML, JS_STATE_SPOT, "    gpuSpotCost: gpu.spot ?? gpu.spec,\n", 1)],
    # ---- 2. the hand record's lookup ----
    "R1 js: the hand record is read off the spec tier whatever tier is asked": [
        (INDEX_HTML, JS_REC_LOOKUP, JS_REC_LOOKUP.replace("state.priceRecord[tier]", "state.priceRecord.spec"), 1)],
    "R2 py: the hand record is read off the spec tier whatever tier is asked": [
        (REPORT_PY, PY_REC_LOOKUP, PY_REC_LOOKUP.replace(".get(tier)", '.get("spec")'), 1)],
    # ---- 3. the fabric above one board ----
    "D1 js: a multi-device board above one board names NVLink or PCIe, not its fabric": [
        (INDEX_HTML, JS_FABRIC,
         "  const fabric = (state.gpuDevices || 1) > 1 && state.gpuCount > 1 ? (state.hasNVLink ? 'NVLink' : 'PCIe')"
         " : interconnectName(state);\n", 1)],
    "D2 py: a multi-device board above one board is called PCIe": [
        (REPORT_PY, PY_INTERCONNECT_NAME,
         '    if (cfg["gpu"].get("devices", 1) or 1) > 1 and cfg.get("n_gpu", 1) > 1:\n'
         '        return "PCIe"\n' + PY_INTERCONNECT_NAME, 1)],
    # ---- 4. price_check's writer ----
    "W1 price: --apply writes every row without its hand record": [
        (PRICE_PY, PRICE_DUMP_ROW, PRICE_DUMP_ROW.replace("for k, v in row.items())", "for k, v in row.items() if k != \"priceRecord\")"), 1)],
    "W2 price: --apply drops a row's hand record when it writes that row": [
        (PRICE_PY, PRICE_APPLY_MOVED, '        row.pop("priceRecord", None)\n' + PRICE_APPLY_MOVED, 1)],
    "W3 price: --apply writes every nested object without its url": [
        (PRICE_PY, PRICE_DUMP_VALUE, PRICE_DUMP_VALUE.replace("for k, x in v.items())", "for k, x in v.items() if k != \"url\")"), 1)],
    # ---- the checker's informational finding ----
    "C5 data: a hand record's url is the provider's front page": [
        (GPUS_JSON, json.dumps(REC_URL), json.dumps(FRONT), 1)],
}

if __name__ == "__main__":
    run_driver(S)
