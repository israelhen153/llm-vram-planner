#!/usr/bin/env python3
"""Round 1, and the bulk of the corpus: throughput figures leaking into each of
the four JS surfaces and the PDF when a card has no measured constants.

Usage: python3 tests/sabotage/engine_r1_throughput_leaks.py [name-substring ...]
       python3 tests/sabotage/engine_r1_throughput_leaks.py --from <name-prefix>
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import (INDEX_HTML, JS_BADGE, JS_CAP_MAXCTX, JS_CMP_GATE, JS_CMP_UNKNOWN,
                     JS_COST, JS_DECL, JS_EXEC_DP, JS_EXEC_GATE, JS_EXEC_UNKNOWN, JS_FIELDS,
                     JS_GETSPEC, JS_LOOKUP, JS_MD_COST, JS_MD_GATE, JS_MD_PERDEV,
                     JS_MD_UNKNOWN, JS_METRICS_W, JS_NOTES_GATE, JS_NOTES_PCIE,
                     JS_NOTES_UNKNOWN, JS_PERF, JS_QUEUE, JS_RET_BATCH, JS_RET_COST,
                     JS_RET_FITS, JS_RET_TP, JS_RET_VRAM, JS_STATE_PK, JS_TP_GATE,
                     JS_TP_RETURN, JS_TP_TILE, JS_UNMODELLED_NOTE, JS_UNSUPPRESS, PY_B_CLI,
                     PY_B_INTERACTIVE, PY_B_JSON, PY_B_RAW, PY_COST, PY_COST_HEAD, PY_DECL,
                     PY_DP_NOTE, PY_EXPLAIN, PY_FIELDS, PY_FP8_NOTE, PY_MAXCTX_ROW,
                     PY_NOTES_PCIE, PY_NOTMOD_ROW, PY_PERF, PY_RET_BATCH, PY_RET_FITS,
                     PY_RET_TP, PY_RET_W, PY_TP_GATE, PY_UNSUPPRESS, REPORT_PY, SYNC_FIELDS,
                     SYNC_PY)

S = {}  # name -> edits

# ---- A. fallback to NVIDIA's constants ----
S["A1 js: fallback PERF.nvidia for any key without an entry"] = [
    (INDEX_HTML, "    ? PERF[state.perfKey] : null;\n", "    ? PERF[state.perfKey] : PERF.nvidia;\n", 1)]
S["A2 js: fallback only when perfKey is not a string"] = [
    (INDEX_HTML, JS_LOOKUP, "  const P = typeof state.perfKey !== 'string' ? PERF.nvidia : Object.hasOwn(PERF, state.perfKey)\n    ? PERF[state.perfKey] : null;\n", 1)]
S["A3 js: fallback only when perfKey is absent (undefined)"] = [
    (INDEX_HTML, JS_LOOKUP, "  const P = state.perfKey === undefined ? PERF.nvidia : (typeof state.perfKey === 'string' && Object.hasOwn(PERF, state.perfKey)\n    ? PERF[state.perfKey] : null);\n", 1)]
S["A4 js: look PERF up by vendor instead of perfKey"] = [
    (INDEX_HTML, JS_LOOKUP, "  const P = typeof state.vendor === 'string' && Object.hasOwn(PERF, state.vendor)\n    ? PERF[state.vendor] : null;\n", 1)]
S["A5 js: vendor as a second-chance lookup"] = [
    (INDEX_HTML, JS_LOOKUP, "  const P = typeof state.perfKey === 'string' && Object.hasOwn(PERF, state.perfKey)\n    ? PERF[state.perfKey] : (typeof state.vendor === 'string' && Object.hasOwn(PERF, state.vendor) ? PERF[state.vendor] : null);\n", 1)]
S["A6 js: `in` instead of Object.hasOwn (prototype chain)"] = [
    (INDEX_HTML, "typeof state.perfKey === 'string' && Object.hasOwn(PERF, state.perfKey)\n    ? PERF[state.perfKey] : null;", "typeof state.perfKey === 'string' && state.perfKey in PERF\n    ? PERF[state.perfKey] : null;", 1)]
S["A7 py: fallback PERF['nvidia'] for any key without an entry"] = [
    (REPORT_PY, "    P = PERF.get(perf_key) if isinstance(perf_key, str) else None\n", "    P = PERF.get(perf_key, PERF[\"nvidia\"]) if isinstance(perf_key, str) else PERF[\"nvidia\"]\n", 1)]
S["A8 py: fallback only when perfKey is not a string"] = [
    (REPORT_PY, "    P = PERF.get(perf_key) if isinstance(perf_key, str) else None\n", "    P = PERF.get(perf_key) if isinstance(perf_key, str) else PERF[\"nvidia\"]\n", 1)]
S["A9 py: fallback only when perfKey is absent (None)"] = [
    (REPORT_PY, "    P = PERF.get(perf_key) if isinstance(perf_key, str) else None\n", "    P = PERF[\"nvidia\"] if perf_key is None else (PERF.get(perf_key) if isinstance(perf_key, str) else None)\n", 1)]
S["A10 py: look PERF up by vendor instead of perfKey"] = [
    (REPORT_PY, "    perf_key = cfg.get(\"perfKey\")\n", "    perf_key = cfg.get(\"vendor\")\n", 1)]
S["A11 py: vendor as a second-chance lookup"] = [
    (REPORT_PY, "    P = PERF.get(perf_key) if isinstance(perf_key, str) else None\n", "    P = (PERF.get(perf_key) if isinstance(perf_key, str) else None) or PERF.get(str(cfg.get(\"vendor\")))\n", 1)]
S["A12 py: drop the isinstance guard (bare PERF.get)"] = [
    (REPORT_PY, "    P = PERF.get(perf_key) if isinstance(perf_key, str) else None\n", "    P = PERF.get(perf_key)\n", 1)]
S["A13 both: fallback PERF.nvidia in both engines"] = S["A1 js: fallback PERF.nvidia for any key without an entry"] + S["A7 py: fallback PERF['nvidia'] for any key without an entry"]
S["A14 both: non-string fallback in both engines"] = S["A2 js: fallback only when perfKey is not a string"] + S["A8 py: fallback only when perfKey is not a string"]
S["A15 both: absent-key fallback in both engines"] = S["A3 js: fallback only when perfKey is absent (undefined)"] + S["A9 py: fallback only when perfKey is absent (None)"]
S["A16 both: vendor lookup in both engines"] = S["A4 js: look PERF up by vendor instead of perfKey"] + S["A10 py: look PERF up by vendor instead of perfKey"]
S["A17 both: vendor second chance in both engines"] = S["A5 js: vendor as a second-chance lookup"] + S["A11 py: vendor as a second-chance lookup"]

# ---- B. suppress in one engine, not the other ----
S["B1 js: never modelled (P = null for every card)"] = [
    (INDEX_HTML, JS_LOOKUP, "  const P = null;\n", 1)]
S["B2 py: never modelled (P = None for every card)"] = [
    (REPORT_PY, "    P = PERF.get(perf_key) if isinstance(perf_key, str) else None\n", "    P = None\n", 1)]
S["B3 js: always modelled with nvidia (unknown card modelled in JS only)"] = S["A1 js: fallback PERF.nvidia for any key without an entry"]
S["B4 py: always modelled with nvidia (unknown card modelled in Python only)"] = S["A7 py: fallback PERF['nvidia'] for any key without an entry"]
S["B5 js: throughputModelled reported true while figures stay null"] = [
    (INDEX_HTML, "  const throughputModelled = P !== null;\n", "  const throughputModelled = true;\n", 1),
    (INDEX_HTML, "  if (throughputModelled) {\n    const achievedBandwidth", "  if (P !== null) {\n    const achievedBandwidth", 1),
    (INDEX_HTML, "    perfMbu: throughputModelled ? P.mbu : null,\n    perfMfuDecode: throughputModelled ? P.mfuDecode : null,\n    perfMfuPrefill: throughputModelled ? P.mfuPrefill : null,\n    perfObsLo: throughputModelled ? P.obsLo : null,\n    perfObsHi: throughputModelled ? P.obsHi : null,\n    perfFp8Ratio: throughputModelled ? P.fp8ComputeRatio : null,\n",
     "    perfMbu: P ? P.mbu : null,\n    perfMfuDecode: P ? P.mfuDecode : null,\n    perfMfuPrefill: P ? P.mfuPrefill : null,\n    perfObsLo: P ? P.obsLo : null,\n    perfObsHi: P ? P.obsHi : null,\n    perfFp8Ratio: P ? P.fp8ComputeRatio : null,\n", 1)]
S["B6 py: throughput_modelled reported True while figures stay None"] = [
    (REPORT_PY, "    throughput_modelled = P is not None\n", "    throughput_modelled = True\n", 1),
    (REPORT_PY, "    if throughput_modelled:\n        achieved_bw", "    if P is not None:\n        achieved_bw", 1),
    (REPORT_PY, "        \"perf_mbu\": P[\"mbu\"] if throughput_modelled else None,\n        \"perf_mfu_decode\": P[\"mfuDecode\"] if throughput_modelled else None,\n        \"perf_mfu_prefill\": P[\"mfuPrefill\"] if throughput_modelled else None,\n        \"perf_obs_lo\": P[\"obsLo\"] if throughput_modelled else None,\n        \"perf_obs_hi\": P[\"obsHi\"] if throughput_modelled else None,\n        \"perf_fp8_ratio\": P[\"fp8ComputeRatio\"] if throughput_modelled else None,\n",
     "        \"perf_mbu\": P[\"mbu\"] if P else None,\n        \"perf_mfu_decode\": P[\"mfuDecode\"] if P else None,\n        \"perf_mfu_prefill\": P[\"mfuPrefill\"] if P else None,\n        \"perf_obs_lo\": P[\"obsLo\"] if P else None,\n        \"perf_obs_hi\": P[\"obsHi\"] if P else None,\n        \"perf_fp8_ratio\": P[\"fp8ComputeRatio\"] if P else None,\n", 1)]
S["B7 js: throughputModelled returned as undefined/0 rather than false"] = [
    (INDEX_HTML, "  const throughputModelled = P !== null;\n", "  const throughputModelled = P !== null ? true : 0;\n", 1)]
S["B8 py: throughput_modelled returned as 0 rather than False"] = [
    (REPORT_PY, "    throughput_modelled = P is not None\n", "    throughput_modelled = True if P is not None else 0\n", 1)]

# ---- C. 0 instead of null/None, per field ----
for f in JS_FIELDS:
    S[f"C js: {f} = 0 instead of null"] = [(INDEX_HTML, JS_DECL, JS_DECL.replace(f"{f} = null", f"{f} = {'false' if f == 'computeBound' else '0'}"), 1)]
for f, expr in JS_PERF.items():
    S[f"C js: {f} = 0 instead of null"] = [(INDEX_HTML, f"    {f}: throughputModelled ? {expr} : null,\n", f"    {f}: throughputModelled ? {expr} : 0,\n", 1)]
for f in PY_FIELDS:
    S[f"C py: {f} = 0 instead of None"] = [(REPORT_PY, PY_DECL, PY_DECL + f"    {f} = 0\n", 1)]
for f, k in PY_PERF.items():
    S[f"C py: {f} = 0 instead of None"] = [(REPORT_PY, f"        \"{f}\": P[\"{k}\"] if throughput_modelled else None,\n", f"        \"{f}\": P[\"{k}\"] if throughput_modelled else 0,\n", 1)]
S["C both: singleStreamTokS/single_tok = 0 in both engines"] = S["C js: singleStreamTokS = 0 instead of null"] + S["C py: single_tok = 0 instead of None"]
S["C both: ttftMs/ttft_ms = 0 in both engines"] = S["C js: ttftMs = 0 instead of null"] + S["C py: ttft_ms = 0 instead of None"]
S["C both: perfMbu/perf_mbu = 0 in both engines"] = S["C js: perfMbu = 0 instead of null"] + S["C py: perf_mbu = 0 instead of None"]
S["C js: computeBound = false instead of null (only)"] = S["C js: computeBound = 0 instead of null"]

# ---- D. leave one field unsuppressed (computed with nvidia's constants) ----
for f in JS_FIELDS:
    S[f"D js: {f} left unsuppressed"] = [(INDEX_HTML, JS_COST, JS_UNSUPPRESS.replace("FIELD", f) + JS_COST, 1)]
for f, expr in JS_PERF.items():
    S[f"D js: {f} left unsuppressed"] = [(INDEX_HTML, f"    {f}: throughputModelled ? {expr} : null,\n", f"    {f}: throughputModelled ? {expr} : PERF.nvidia.{expr[2:]},\n", 1)]
for f in PY_FIELDS:
    S[f"D py: {f} left unsuppressed"] = [(REPORT_PY, PY_COST, PY_UNSUPPRESS.replace("FIELD", f) + PY_COST, 1)]
for f, k in PY_PERF.items():
    S[f"D py: {f} left unsuppressed"] = [(REPORT_PY, f"        \"{f}\": P[\"{k}\"] if throughput_modelled else None,\n", f"        \"{f}\": P[\"{k}\"] if throughput_modelled else PERF[\"nvidia\"][\"{k}\"],\n", 1)]
S["D both: computeRatio/compute_ratio unsuppressed in both"] = S["D js: computeRatio left unsuppressed"] + S["D py: compute_ratio left unsuppressed"]
S["D both: ttftMs/ttft_ms unsuppressed in both"] = S["D js: ttftMs left unsuppressed"] + S["D py: ttft_ms left unsuppressed"]
S["D both: perfObsLo/perf_obs_lo unsuppressed in both"] = S["D js: perfObsLo left unsuppressed"] + S["D py: perf_obs_lo left unsuppressed"]
S["D both: aggregateObservedHiTokS/agg_obs_hi unsuppressed in both"] = S["D js: aggregateObservedHiTokS left unsuppressed"] + S["D py: agg_obs_hi left unsuppressed"]

# ---- E. delete the explanation from one surface ----
S["E1 js: throughput panel — no reason at all"] = [(INDEX_HTML, JS_UNMODELLED_NOTE, "  const unmodelledNote = '';\n", 1)]
S["E2 js: throughput panel — only 'No measured utilisation' half"] = [(INDEX_HTML, JS_UNMODELLED_NOTE, "  const unmodelledNote = computed.throughputModelled ? '' : `<span style=\"color:var(--warning-text);font-weight:500\">No measured utilisation for this hardware</span>`;\n", 1)]
S["E3 js: throughput panel — 'Not modelled' label removed, reason kept"] = [(INDEX_HTML, JS_TP_TILE, JS_TP_TILE.replace("Not modelled", "—"), 1)]
S["E4 js: throughput panel — reason moved into the notes box"] = [
    (INDEX_HTML, JS_UNMODELLED_NOTE, "  const unmodelledNote = '';\n", 1),
    (INDEX_HTML, JS_NOTES_UNKNOWN, "    : 'Throughput is not modelled for this hardware: no measured utilisation is published for it, and borrowed constants do not transfer. ';\n", 1)]
S["E5 js: throughput panel — tile removed entirely for the unknown card"] = [(INDEX_HTML, JS_TP_RETURN, "    setHTML('throughput-output', `<div class=\"reverse-grid\">${maxBatchCard}</div>${queueWarning}`);\n    return;\n  }\n", 1)]
S["E6 js: comparison card — bare 'not modelled'"] = [(INDEX_HTML, JS_CMP_UNKNOWN, "      : `<div class=\"row\"><span class=\"label\">Per-user / aggregate</span><span class=\"val\">not modelled</span></div>`;\n", 1)]
S["E7 js: comparison card — row removed entirely"] = [(INDEX_HTML, JS_CMP_UNKNOWN, "      : '';\n", 1)]
S["E8 js: comparison card — reason without 'not modelled'"] = [(INDEX_HTML, JS_CMP_UNKNOWN, JS_CMP_UNKNOWN.replace("not modelled<br>", ""), 1)]
S["E9 js: exec summary — bare 'Not modelled'"] = [(INDEX_HTML, JS_EXEC_UNKNOWN, "    html += `<div class=\"exec-row\"><span class=\"exec-label\">Speed and server throughput</span><span class=\"exec-value\">Not modelled</span></div>`;\n", 1)]
S["E10 js: exec summary — row removed entirely"] = [(INDEX_HTML, JS_EXEC_UNKNOWN, "", 1)]
S["E11 js: exec summary — reason without 'Not modelled'"] = [(INDEX_HTML, JS_EXEC_UNKNOWN, JS_EXEC_UNKNOWN.replace("Not modelled ", ""), 1)]
S["E12 js: markdown export — bare 'not modelled'"] = [(INDEX_HTML, JS_MD_UNKNOWN, "    : `- Throughput and TTFT: not modelled.\\n`;\n", 1)]
S["E13 js: markdown export — line removed entirely"] = [(INDEX_HTML, JS_MD_UNKNOWN, "    : '';\n", 1)]
S["E14 js: markdown export — reason without 'not modelled'"] = [(INDEX_HTML, JS_MD_UNKNOWN, JS_MD_UNKNOWN.replace("not modelled. ", ""), 1)]
S["E15 js: notes box — unmodelled sentence removed"] = [(INDEX_HTML, JS_NOTES_UNKNOWN, "    : '';\n", 1)]
S["E16 js: badge — unmodelled wording dropped (always the 45-60% loss)"] = [(INDEX_HTML, JS_BADGE, "PCIe — 45-60% perf loss, worse with more devices", 1)]
S["E17 py: PDF — explanation paragraph removed"] = [(REPORT_PY, PY_EXPLAIN, "", 1)]
S["E18 py: PDF — only the 'No measured utilisation' half"] = [(REPORT_PY, PY_EXPLAIN, "            story.append(Paragraph(f\"No measured utilisation is published for {gpu['name']}.\", self.styles[\"Small\"]))\n", 1)]
S["E19 py: PDF — 'Not modelled' cell replaced by a dash"] = [(REPORT_PY, PY_NOTMOD_ROW, "                [\"Throughput and TTFT\", \"\\u2014\"],\n", 1)]
S["E20 py: PDF — reason moved into Notes and assumptions"] = [
    (REPORT_PY, PY_EXPLAIN, "", 1),
    (REPORT_PY, "        notes.append(\"VRAM estimates include ~1.5 GB CUDA context overhead per device.\")\n",
     "        notes.append(\"VRAM estimates include ~1.5 GB CUDA context overhead per device.\")\n        if not c[\"throughput_modelled\"]:\n            notes.append(f\"No measured utilisation is published for {gpu['name']}; borrowed constants do not transfer.\")\n", 1)]
S["E21 py: PDF — throughput section omitted entirely for the unknown card"] = [
    (REPORT_PY, "            story.append(Paragraph(\"Throughput\", self.styles[\"SectionHead\"]))\n            story.append(self._make_kv_table([\n                [\"Throughput and TTFT\", \"Not modelled\"],\n                max_batch_row,\n            ]))\n" + PY_EXPLAIN, "            pass\n", 1)]

# ---- F. suppress a constants-independent figure as well ----
S["F1 js: weightsGB null without constants"] = [(INDEX_HTML, JS_RET_VRAM, "    isMoE, weightsGB: throughputModelled ? weightsGB : null, kvCacheGB, activationsGB, totalOverhead, totalGB,\n", 1)]
S["F2 js: maxBatchByKV null without constants"] = [(INDEX_HTML, JS_RET_BATCH, "    maxBatchByKV: throughputModelled ? maxBatchByKV : null, batchLimitedByKV, computeBound, ttftMs, ttftColdMs, ttftWarmMs,\n", 1)]
S["F3 js: tp/dp null without constants"] = [(INDEX_HTML, JS_RET_TP, "    tp: throughputModelled ? tp : null, dp: throughputModelled ? dp : null,\n    freeForKVCache,", 1)]
S["F4 js: hourlyHyper null without constants"] = [(INDEX_HTML, JS_RET_COST, "    hourlyHyper: throughputModelled ? hourlyHyper : null, hourlySpec, hourlySpot,\n", 1)]
S["F5 js: fits null without constants"] = [(INDEX_HTML, JS_RET_FITS, "    fits: throughputModelled ? perGPU.total <= deviceGB : null,\n", 1)]
S["F6 js: capacity panel hides max context without constants (renderer only)"] = [(INDEX_HTML, JS_CAP_MAXCTX, "<p class=\"value\">${computed.throughputModelled ? formatContext(computed.maxContextSingleUser) + ' tokens' : 'not modelled'}</p>", 1)]
S["F7 js: metrics panel hides weights without constants (renderer only)"] = [(INDEX_HTML, JS_METRICS_W, "<p class=\"metric-label\">${weightsLabel}</p><p class=\"metric-value\">${computed.throughputModelled ? formatGB(clusterWeights) : 'not modelled'}</p>", 1)]
S["F8 js: markdown export drops the per-device fit line without constants"] = [(INDEX_HTML, JS_MD_PERDEV, "- Per-device: ${computed.throughputModelled ? formatGB(computed.perGPU.total) + ' / ' + capacityLabel(computed.deviceGB) + ' (' + (computed.fits ? 'FITS' : 'DOES NOT FIT') + ')' : 'not modelled'}\\n`;", 1)]
S["F9 js: markdown export drops the cost section without constants"] = [(INDEX_HTML, JS_MD_COST, "  if (computed.throughputModelled) report += `\\n## Cost (${state.gpuCount}× ${state.gpuName})\\n", 1)]
S["F10 py: weights_gb None without constants"] = [(REPORT_PY, PY_RET_W, "        \"weights_gb\": weights_gb if throughput_modelled else None, \"kv_gb\": kv_gb, \"act_gb\": act_gb,\n", 1)]
S["F11 py: max_batch_kv None without constants"] = [(REPORT_PY, PY_RET_BATCH, "        \"eff_batch\": eff_batch, \"max_batch_kv\": max_batch_kv if throughput_modelled else None, \"batch_limited\": batch_limited,\n", 1)]
S["F12 py: tp/dp None without constants"] = [(REPORT_PY, PY_RET_TP, "        \"tp\": tp if throughput_modelled else None, \"dp\": dp if throughput_modelled else None,\n", 1)]
S["F13 py: fits None without constants"] = [(REPORT_PY, PY_RET_FITS, "        \"fits\": fits if throughput_modelled else None, \"comfortable\": comfortable,\n", 1)]
S["F14 py: PDF max-context row hidden without constants (renderer only)"] = [(REPORT_PY, PY_MAXCTX_ROW, "            [\"Max context (1 user, 90% util)\", fmt_k(c[\"max_ctx_1\"]) + \" tokens\" if c[\"throughput_modelled\"] else \"not modelled\"],\n", 1)]
S["F15 py: PDF cost section skipped without constants (renderer only)"] = [(REPORT_PY, PY_COST_HEAD + "        cost_data = [", "        if c[\"throughput_modelled\"]:\n          story.append(Paragraph(\"Cost estimate\", self.styles[\"SectionHead\"]))\n        cost_data = [", 1),
    (REPORT_PY, "        story.append(cost_table)\n", "        if c[\"throughput_modelled\"]: story.append(cost_table)\n", 1)]
S["F16 both: weights null/None in both engines"] = S["F1 js: weightsGB null without constants"] + S["F10 py: weights_gb None without constants"]
S["F17 both: max batch null/None in both engines"] = S["F2 js: maxBatchByKV null without constants"] + S["F11 py: max_batch_kv None without constants"]

# ---- G. plumbing that drops perfKey ----
S["G1 js: getGpuSpec drops perfKey"] = [(INDEX_HTML, JS_GETSPEC, "           vendor: g.vendor, devices: g.devices, form: g.form, caps: g.caps };\n", 1)]
S["G2 js: readInputState drops perfKey"] = [(INDEX_HTML, JS_STATE_PK, "", 1)]
S["G3 js: readInputState fills perfKey from vendor"] = [(INDEX_HTML, JS_STATE_PK, "    perfKey: gpu.vendor,\n", 1)]
S["G4 js: getGpuSpec fills perfKey from vendor"] = [(INDEX_HTML, JS_GETSPEC, "           vendor: g.vendor, perfKey: g.vendor, devices: g.devices, form: g.form, caps: g.caps };\n", 1)]
S["G5 js: getGpuSpec hardcodes perfKey 'nvidia'"] = [(INDEX_HTML, JS_GETSPEC, "           vendor: g.vendor, perfKey: 'nvidia', devices: g.devices, form: g.form, caps: g.caps };\n", 1)]
S["G6 py: interactive_mode drops perfKey"] = [(REPORT_PY, PY_B_INTERACTIVE, "        \"kv_bpp\": kv_bpp,", 1)]
S["G7 py: from_json preset branch drops perfKey"] = [(REPORT_PY, PY_B_JSON, "            \"kv_bpp\": raw.get(\"kv_bpp\", 2),", 1)]
S["G8 py: from_json raw branch drops perfKey"] = [(REPORT_PY, PY_B_RAW, "", 1)]
S["G9 py: from_cli_args drops perfKey"] = [(REPORT_PY, PY_B_CLI, "            \"kv_bpp\": 1 if args.fp8_kv else 2,", 1)]
S["G10 py: from_cli_args fills perfKey from vendor"] = [(REPORT_PY, PY_B_CLI, "            \"perfKey\": gpu[\"vendor\"],\n            \"kv_bpp\": 1 if args.fp8_kv else 2,", 1)]
S["G11 py: from_json lets the JSON override perfKey"] = [(REPORT_PY, PY_B_JSON, "            \"perfKey\": raw.get(\"perfKey\", gpu[\"perfKey\"]),\n            \"kv_bpp\": raw.get(\"kv_bpp\", 2),", 1)]
S["G12 py: from_json raw branch lets the JSON override perfKey"] = [(REPORT_PY, PY_B_RAW, "    cfg.setdefault(\"perfKey\", gpu[\"perfKey\"])\n", 1)]
S["G13 py: from_cli_args hardcodes perfKey 'nvidia'"] = [(REPORT_PY, PY_B_CLI, "            \"perfKey\": \"nvidia\",\n            \"kv_bpp\": 1 if args.fp8_kv else 2,", 1)]
S["G14 sync: perfKey no longer required by tools/sync_data.py"] = [(SYNC_PY, SYNC_FIELDS, "              \"vendor\", \"devices\", \"form\", \"caps\")\n", 1)]
S["G15 sync: a missing perfKey defaults to 'nvidia' in the generated blocks"] = [
    (SYNC_PY, "    missing = [f for f in fields if f not in row]\n", "    row = dict(row); row.setdefault(\"perfKey\", \"nvidia\")\n    missing = [f for f in fields if f not in row]\n", 1)]
S["G16 both: readInputState and from_cli_args both fill perfKey from vendor"] = S["G3 js: readInputState fills perfKey from vendor"] + S["G10 py: from_cli_args fills perfKey from vendor"]

# ---- H. surfaces that still print figures or speak of speed ----
S["H1 js: throughput panel prints figures for the unknown card (gate removed)"] = [(INDEX_HTML, JS_TP_GATE, "  if (false) {\n    /* And no benchmark panel.", 1)]
S["H2 js: comparison card prints figures for the unknown card"] = [(INDEX_HTML, JS_CMP_GATE, JS_CMP_GATE.replace("c.throughputModelled", "true"), 1)]
S["H3 js: exec summary prints figures for the unknown card"] = [(INDEX_HTML, JS_EXEC_GATE, JS_EXEC_GATE.replace("computed.throughputModelled", "true"), 1)]
S["H4 js: markdown export prints figures for the unknown card"] = [(INDEX_HTML, JS_MD_GATE, JS_MD_GATE.replace("computed.throughputModelled", "true"), 1)]
S["H5 js: notes always say 'memory-bandwidth-bound decode estimate'"] = [(INDEX_HTML, JS_NOTES_GATE, JS_NOTES_GATE.replace("computed.throughputModelled", "true"), 1)]
S["H6 js: notes always quote the 45-60% PCIe decode loss"] = [(INDEX_HTML, JS_NOTES_PCIE, "'PCIe 64-128 GB/s — 45-60% decode loss, growing with device count. '", 1)]
S["H7 js: exec dp>1 note always points at 'the throughput figures'"] = [(INDEX_HTML, JS_EXEC_DP, ", so treat the throughput figures as a ceiling to test, not a quote", 1)]
S["H8 js: queue warning always points at 'aggregate throughput above'"] = [(INDEX_HTML, JS_QUEUE, " — aggregate throughput above reflects the batch that actually runs", 1)]
S["H9 py: PDF prints the modelled table for the unknown card (gate removed)"] = [(REPORT_PY, PY_TP_GATE, PY_TP_GATE.replace("if c[\"throughput_modelled\"]:", "if True:"), 1)]
S["H10 py: PDF notes always quote the 30-50% PCIe decode loss"] = [(REPORT_PY, PY_NOTES_PCIE, "'PCIe (64-128 GB/s) loses 30-50% decode throughput vs NVLink.'", 1)]
S["H11 py: PDF FP8 note printed without constants"] = [(REPORT_PY, PY_FP8_NOTE, "        if c.get(\"fp8_no_tensor_cores\"):\n", 1)]
S["H12 py: PDF dp>1 note always mentions 'the throughput and TTFT figures'"] = [(REPORT_PY, PY_DP_NOTE, "                         + \", so the throughput and TTFT figures inherit that uncertainty.\")\n", 1)]
S["H13 py: PDF keeps the 'Basis' row for the unknown card"] = [(REPORT_PY, "                [\"Throughput and TTFT\", \"Not modelled\"],\n                max_batch_row,\n", "                [\"Throughput and TTFT\", \"Not modelled\"],\n                max_batch_row,\n                [\"Basis\", f\"Memory-bandwidth bound — {c['device_bw']:g} GB/s x {c['device_count']} device(s)\"],\n", 1)]
S["H14 js: unknown card's tile says 'Bandwidth-bound' in words"] = [(INDEX_HTML, JS_TP_TILE, JS_TP_TILE.replace("<p class=\"value\">Not modelled</p>", "<p class=\"value\">Not modelled</p><p class=\"sub\">Bandwidth-bound decode would apply here</p>"), 1)]
S["H15 js: unknown card's tile keeps 'observed range' wording"] = [(INDEX_HTML, JS_TP_TILE, JS_TP_TILE.replace("<p class=\"value\">Not modelled</p>", "<p class=\"value\">Not modelled</p><p class=\"sub\">observed range unavailable</p>"), 1)]

# ---- I. structural / result-shape ----
S["I1 js: singleStreamTokS omitted from the result (undefined) without constants"] = [(INDEX_HTML, "    singleStreamTokS, aggregateTokS, perUserAtLoadTokS, effectiveBatch,\n", "    ...(throughputModelled ? { singleStreamTokS } : {}), aggregateTokS, perUserAtLoadTokS, effectiveBatch,\n", 1)]
S["I2 py: single_tok omitted from the result without constants"] = [(REPORT_PY, "        \"single_tok\": single_tok, \"agg_tok\": agg_tok, \"per_user_load\": per_user_load,\n", "        **({\"single_tok\": single_tok} if throughput_modelled else {}), \"agg_tok\": agg_tok, \"per_user_load\": per_user_load,\n", 1)]
S["I3 js: unknown key throws instead of nulling"] = [(INDEX_HTML, "  const throughputModelled = P !== null;\n", "  const throughputModelled = P !== null;\n  if (!throughputModelled) throw new Error('no constants');\n", 1)]
S["I4 py: unknown key raises instead of None"] = [(REPORT_PY, "    throughput_modelled = P is not None\n", "    throughput_modelled = P is not None\n    if not throughput_modelled:\n        raise KeyError(perf_key)\n", 1)]

if __name__ == "__main__":
    run_driver(S)
