#!/usr/bin/env python3
"""Round 4: fix/cost-provenance. Sixteen sabotages against the branch that
made every cost figure name its own source or say "not recorded" — drawn
from three places: sparkling-crunching-whistle.md Part 2's own sabotage
list (S1-S5 below), the ones the builder ran by hand while shipping the
branch (W1-W5), and the ones a cold check found the builder's first draft
missed (C1-C6) — including C4, the cost range no longer bounded by spot
and hyperscaler by name once a real price move stopped keeping
spot <= specialized <= hyperscaler true for every card.

Usage: python3 tests/sabotage/engine_r4_cost_provenance.py [name-substring ...]
       python3 tests/sabotage/engine_r4_cost_provenance.py --from C1
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import (INDEX_HTML, REPORT_PY, SYNC_PY, PRICE_PY, GPUS_JSON,
                     JS_COST_HYPER_SUBLABEL, JS_CMP_COST_SOURCE, JS_CMP_RANGE,
                     JS_PRICE_LABEL_RET, PY_PRICE_LABEL_RET, PY_TIER_HYPERSCALER,
                     PY_NOTES_COMPOSITE_OPEN, SYNC_OPTIONAL, PRICE_CROSS_CHECK_THRESHOLD,
                     PRICE_APPLY_LOOP, PRICE_FIELD_LINE, GPUS_ADA_ROW,
                     GPUS_H100_HYPER_PREFIX, GPUS_H100_SRC_BLOCK)

S = {}

# ---- S: the plan's own sabotage list (sparkling-crunching-whistle.md Part 2) ----
S["S1 js: renderCost's hyper sub-label is the old three-provider composite"] = [
    (INDEX_HTML, JS_COST_HYPER_SUBLABEL,
     JS_COST_HYPER_SUBLABEL.replace("${priceSourceLabel(state, 'hyper')}", "AWS, GCP, Azure on-demand"), 1)]
S["S2 sync: priceSource dropped from GPU_OPTIONAL, so neither generated block carries it"] = [
    (SYNC_PY, SYNC_OPTIONAL, "GPU_OPTIONAL = (\"default\",)\n", 1)]
S["S3 data: rtx6000ada-48/spot gets a priceSource for a tier SOURCE_MAP marks manual"] = [
    (GPUS_JSON, GPUS_ADA_ROW,
     GPUS_ADA_ROW[:-2] + (", \"priceSource\": { \"spot\": { \"provider\": \"vast\", "
                          "\"sku\": \"RTX 6000 Ada (median of 12 verified offers)\", "
                          "\"region\": \"global\", \"date\": \"2026-09-22\", \"price\": 0.6 } } }"), 1)]
S["S4 js: priceSourceLabel names the provider but drops the read date"] = [
    (INDEX_HTML, JS_PRICE_LABEL_RET,
     "  return `${provider} · ${src.sku} · ${src.region}`;\n", 1)]
S["S5 data: h100-80/hyper moved (12.3 -> 12.5) with priceSource.price left at 12.29"] = [
    (GPUS_JSON, GPUS_H100_HYPER_PREFIX, GPUS_H100_HYPER_PREFIX.replace("12.3,", "12.5,"), 1)]

# ---- W: sabotages the builder ran by hand before the first cold check ----
S["W1 py: price_source_label names the provider but drops the read date"] = [
    (REPORT_PY, PY_PRICE_LABEL_RET,
     "    return f\"{provider} · {src['sku']} · {src['region']}\"\n", 1)]
S["W2 price_check: apply_to_text reverts to rewriting only changed_slugs (the pre-fix bug)"] = [
    (PRICE_PY, PRICE_APPLY_LOOP,
     "    new_text = raw_text\n    for slug in changed_slugs:\n        row = gpus_data[slug]\n", 1)]
S["W3 price_check: CROSS_CHECK_FLAG_THRESHOLD widened past the b200-192 disagreement (0.20 -> 0.90)"] = [
    (PRICE_PY, PRICE_CROSS_CHECK_THRESHOLD, "CROSS_CHECK_FLAG_THRESHOLD = 0.90\n", 1)]
S["W4 price_check: priceSource.price dropped from what apply_outcomes writes"] = [
    (PRICE_PY, PRICE_FIELD_LINE, "", 1)]
S["W5 py: the PDF cost table's Hyperscaler cell regains its provider parenthetical"] = [
    (REPORT_PY, PY_TIER_HYPERSCALER, "            [\"Hyperscaler (AWS/GCP/Azure)\",\n", 1)]

# ---- C: gaps the first cold check found in the builder's initial draft ----
S["C1 js: a composite is appended after an otherwise-correct sourced label (comparison card)"] = [
    (INDEX_HTML, JS_CMP_COST_SOURCE,
     JS_CMP_COST_SOURCE.replace("Hyper: ${priceSourceLabel(s, 'hyper')}",
                                "Hyper: ${priceSourceLabel(s, 'hyper')}"
                                "${s.priceSource && s.priceSource.hyper ? ' (AWS, GCP, Azure)' : ''}"), 1)]
S["C2 data: a sourced tier's date is blank (empty string, not missing)"] = [
    (GPUS_JSON, GPUS_H100_SRC_BLOCK, GPUS_H100_SRC_BLOCK.replace("\"date\": \"2026-09-22\"", "\"date\": \"\""), 1)]
S["C3 py: the general 'GPU prices are... estimates' note regains a composite and the word estimates"] = [
    (REPORT_PY, PY_NOTES_COMPOSITE_OPEN,
     PY_NOTES_COMPOSITE_OPEN.replace("per-board/hr figures", "per-board/hr estimates")
                             .replace("hyperscaler, \"\n", "hyperscaler (AWS/GCP/Azure), \"\n"), 1)]
S["C4 js: renderComparisons' Cost/hr falls back to spot-hyperscaler instead of true min/max"] = [
    (INDEX_HTML, JS_CMP_RANGE,
     "<span class=\"val\">$${c.hourlySpot.toFixed(2)}–$${c.hourlyHyper.toFixed(2)}</span>", 1)]
S["C5 js: renderCost's spot sub-label falls back to the old composite when spot is unsourced"] = [
    (INDEX_HTML, JS_CMP_COST_SOURCE,
     JS_CMP_COST_SOURCE.replace(
         "Spot: ${priceSourceLabel(s, 'spot')}",
         "Spot: ${s.priceSource && s.priceSource.spot ? priceSourceLabel(s, 'spot') : 'Vast.ai, spot instances'}"),
     1)]
S["C6 py: the PDF's per-tier Source line drops just the region (bypasses price_source_label itself)"] = [
    (REPORT_PY,
     "            f\"Source — Hyperscaler: {price_source_label(gpu, 'hyper')}. \"\n",
     "            f\"Source — Hyperscaler: "
     "{(lambda s: s if s == 'not recorded' else ' · '.join(p for i, p in enumerate(s.split(' · ')) if i != 2))"
     "(price_source_label(gpu, 'hyper'))}. \"\n", 1)]

if __name__ == "__main__":
    run_driver(S)
