#!/usr/bin/env python3
"""Sabotage driver. Each sabotage is a list of exact-string edits to committed
files; the driver applies them (refusing if a target string is not found the
expected number of times), runs the judging suites, prints one line per
sabotage, and restores every touched file with `git checkout --`.

Usage: python3 tests/sabotage/sab.py [name-substring ...]   (no args = all)
"""
import os, re, subprocess, sys

# No __pycache__. The drivers import each other, so a run used to leave one behind;
# it is gitignored, which means git cannot remove the directory on a branch switch,
# which leaves an unlisted tests/sabotage behind and turns the project-structure
# guard red on a branch that has nothing to do with this. It has cost two people a
# confusing red suite, so it stops being created.
sys.dont_write_bytecode = True
# The repo root, however deep this file is filed. Asking git rather than counting
# "..", because counting is what breaks silently when this directory moves: the
# sabotages would then be applied to whatever tree the wrong path happens to name.
ROOT = subprocess.check_output(
    ['git', 'rev-parse', '--show-toplevel'],
    cwd=os.path.dirname(os.path.abspath(__file__)), text=True).strip()
os.chdir(ROOT)

SUITES = [("model", ["node", "tests/model.test.js"]),
          ("parity", ["python3", "tests/parity.test.py"]),
          ("report", ["python3", "tests/report.test.py"]),
          ("sync", ["python3", "tests/sync.test.py"]),
          ("price", ["python3", "tests/price_check.test.py"]),
          ("workflow", ["python3", "tests/workflow.test.py"])]


def run_suites():
    res = {}
    for name, cmd in SUITES:
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = p.stdout + p.stderr
        fails = [l.strip() for l in out.splitlines() if l.lstrip().startswith("FAIL")]
        errs = [l.strip() for l in out.splitlines() if re.search(r"Error|Traceback|node runner failed", l)]
        tally = re.findall(r"(\d+) passed, (\d+) failed", out)
        res[name] = (p.returncode, fails, errs, tally[-1] if tally else None)
    return res


def require_green_baseline():
    """A sabotage is judged by a suite going red. If a suite is ALREADY red for an
    unrelated reason — a flaky test, a missing dependency, an unrelated regression in
    the same commit — then every sabotage in the run reads as caught, with no warning
    and no way to tell it from a genuine catch. The run is worthless and looks perfect.

    So prove the tree judges green before judging anything against it."""
    red = [k for k, v in run_suites().items() if v[0] != 0]
    if red:
        sys.exit(f"refusing to judge: {', '.join(red)} already red on the unmodified "
                 f"tree, so every sabotage would read as caught")


def apply(edits):
    """edits: list of (file, old, new, count). Returns the files touched."""
    touched = []
    for f, old, new, count in edits:
        path = os.path.join(ROOT, f)
        src = open(path).read()
        n = src.count(old)
        if n != count:
            raise RuntimeError(f"{f}: expected {count} occurrence(s) of {old[:70]!r}, found {n}")
        open(path, "w").write(src.replace(old, new))
        touched.append(f)
    return touched


def restore(files):
    # Every file a suite can rewrite, not only the ones edited: tests/sync.test.py
    # runs the real sync against the real engines, so a sabotaged tools/sync_data.py
    # regenerates both GPU_TABLE blocks under the test.
    subprocess.run(["git", "checkout", "--", "index.html", "generate_report.py",
                    "tools/sync_data.py", "data/gpus.json", *sorted(set(files))], check=True)
    st = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    assert st.strip() == "", f"tree not clean after restore:\n{st}"


# ---------------------------------------------------------------------------
# index.html anchors
JS_LOOKUP = ("  const P = typeof state.perfKey === 'string' && Object.hasOwn(PERF, state.perfKey)\n"
             "    ? PERF[state.perfKey] : null;\n")
JS_DECL = ("  let computeRatio = null, singleStreamTokS = null, aggregateTokS = null,\n"
           "    perUserAtLoadTokS = null, computeBound = null, saturatedTokS = null,\n"
           "    aggregateObservedLoTokS = null, aggregateObservedHiTokS = null,\n"
           "    ttftColdMs = null, ttftWarmMs = null, ttftMs = null;\n")
JS_COST = "  // Cost\n  // Boards, not devices: the catalog prices a board and a dual-GCD module is\n"
JS_FIELDS = ["computeRatio", "singleStreamTokS", "aggregateTokS", "perUserAtLoadTokS", "computeBound",
             "saturatedTokS", "aggregateObservedLoTokS", "aggregateObservedHiTokS", "ttftColdMs",
             "ttftWarmMs", "ttftMs"]
JS_PERF = {"perfMbu": "P.mbu", "perfMfuDecode": "P.mfuDecode", "perfMfuPrefill": "P.mfuPrefill",
           "perfObsLo": "P.obsLo", "perfObsHi": "P.obsHi", "perfFp8Ratio": "P.fp8ComputeRatio"}
JS_UNSUPPRESS = """  if (!throughputModelled) {
    const Q = PERF.nvidia;
    const ab = deviceBandwidth * 1e9 * deviceCount * Q.mbu * interconnectPenalty;
    const dAt = (b) => { const d = activeWeightBytes + b * kvBytesPerSeq; return d > 0 ? (b * ab) / d : 0; };
    const cr = (isFp8 && gpuFp8) ? Q.fp8ComputeRatio : 1.0;
    const cc = totalActiveParams > 0 ? (Q.mfuDecode * deviceTFLOPS * 1e12 * deviceCount * interconnectPenalty * cr) / (2 * totalActiveParams * 1e9) : 0;
    const apf = Q.mfuPrefill * deviceTFLOPS * 1e12 * deviceCount * interconnectPenalty * cr;
    const tf = (t) => apf > 0 ? Math.round((2 * totalActiveParams * 1e9 * Math.max(t, 0) / apf) * 1000) : 0;
    const agg = Math.round(Math.min(dAt(effectiveBatch), cc));
    const V = { computeRatio: cr, singleStreamTokS: Math.round(dAt(1)), aggregateTokS: agg,
      perUserAtLoadTokS: Math.round(agg / effectiveBatch), computeBound: dAt(effectiveBatch) > cc,
      saturatedTokS: Math.round(Math.min(dAt(saturatedBatch), cc)),
      aggregateObservedLoTokS: Math.round(agg * Q.obsLo), aggregateObservedHiTokS: Math.round(agg * Q.obsHi),
      ttftColdMs: tf(contextLength), ttftWarmMs: prefixCaching ? tf(contextLength - effectivePrefix) : tf(contextLength),
      ttftMs: prefixCaching ? tf(contextLength - effectivePrefix) : tf(contextLength) };
    FIELD = V.FIELD;
  }
"""
JS_UNMODELLED_NOTE = ("  const unmodelledNote = computed.throughputModelled ? ''\n"
                      "    : `<span style=\"color:var(--warning-text);font-weight:500\">No measured utilisation for this hardware</span>` +\n"
                      "      `<span style=\"color:var(--warning-text)\"> — ${state.gpuName} has no published memory-bandwidth or compute utilisation, and estimating its throughput or time to first token would mean borrowing another architecture's constants, which do not transfer</span>`;\n")
JS_CMP_UNKNOWN = "      : `<div class=\"row\"><span class=\"label\">Per-user / aggregate</span><span class=\"val\">not modelled<br><span style=\"color:var(--warning-text);font-size:10px;font-weight:500\">no measured utilisation for this hardware — borrowed constants do not transfer</span></span></div>`;\n"
JS_EXEC_UNKNOWN = "    html += `<div class=\"exec-row\"><span class=\"exec-label\">Speed and server throughput</span><span class=\"exec-value\">Not modelled <span style=\"color:var(--warning-text);font-weight:500\">· no measured utilisation for this hardware, and an estimate would borrow constants that do not transfer</span></span></div>`;\n"
JS_MD_UNKNOWN = "    : `- Throughput and TTFT: not modelled. No measured utilisation is published for ${state.gpuName}, and estimating either would mean borrowing another architecture's constants, which do not transfer.\\n`;\n"
JS_NOTES_UNKNOWN = "    : 'Throughput is not modelled for this hardware: no measured utilisation is published for it. ';\n"
JS_BADGE = "${computed.throughputModelled ? 'PCIe — 45-60% perf loss, worse with more devices' : 'PCIe — no NVLink; the speed cost is not modelled for this hardware'}"
JS_GETSPEC = "           vendor: g.vendor, perfKey: g.perfKey, devices: g.devices, form: g.form, caps: g.caps };\n"
JS_STATE_PK = "    perfKey: gpu.perfKey,\n"
JS_TP_GATE = "  if (!computed.throughputModelled) {\n    /* And no benchmark panel."
JS_TP_TILE = "<p class=\"label\">Throughput and TTFT</p><p class=\"value\">Not modelled</p><p class=\"sub\">${unmodelledNote}</p>"
JS_TP_RETURN = "    setHTML('throughput-output', `<div class=\"reverse-grid\"><div class=\"reverse-card\"><p class=\"label\">Throughput and TTFT</p><p class=\"value\">Not modelled</p><p class=\"sub\">${unmodelledNote}</p></div>${maxBatchCard}</div>${queueWarning}`);\n    return;\n  }\n"
JS_NOTES_GATE = "  notes += computed.throughputModelled\n    ? 'Throughput is a memory-bandwidth-bound decode estimate. '\n"
JS_NOTES_PCIE = "computed.throughputModelled ? 'PCIe 64-128 GB/s — 45-60% decode loss, growing with device count. ' : 'PCIe 64-128 GB/s. '"
JS_EXEC_DP = "${computed.throughputModelled ? ', so treat the throughput figures as a ceiling to test, not a quote' : ''}"
JS_QUEUE = "${computed.throughputModelled ? ' — aggregate throughput above reflects the batch that actually runs' : ''}"
JS_CMP_GATE = "    html += c.throughputModelled\n      ? `<div class=\"row\"><span class=\"label\">Per-user / aggregate</span>"
JS_EXEC_GATE = "  if (computed.throughputModelled) {\n    html += `<div class=\"exec-row\"><span class=\"exec-label\">Speed per user</span>"
JS_MD_GATE = "  report += computed.throughputModelled\n    ? `- Single-stream decode:"
JS_RET_VRAM = "    isMoE, weightsGB, kvCacheGB, activationsGB, totalOverhead, totalGB,\n"
JS_RET_BATCH = "    maxBatchByKV, batchLimitedByKV, computeBound, ttftMs, ttftColdMs, ttftWarmMs,\n"
JS_RET_TP = "    tp, dp,\n    freeForKVCache,"
JS_RET_COST = "    hourlyHyper, hourlySpec, hourlySpot,\n"
JS_RET_FITS = "    fits: perGPU.total <= deviceGB,\n"
JS_CAP_MAXCTX = "<p class=\"value\">${formatContext(computed.maxContextSingleUser)} tokens</p>"
JS_METRICS_W = "<p class=\"metric-label\">${weightsLabel}</p><p class=\"metric-value\">${formatGB(clusterWeights)}</p>"
JS_MD_PERDEV = "- Per-device: ${formatGB(computed.perGPU.total)} / ${capacityLabel(computed.deviceGB)} (${computed.fits ? 'FITS' : 'DOES NOT FIT'})\\n`;"
JS_MD_COST = "  report += `\\n## Cost (${state.gpuCount}× ${state.gpuName})\\n"

# generate_report.py anchors
PY_LOOKUP = ("    perf_key = cfg.get(\"perfKey\")\n"
             "    P = PERF.get(perf_key) if isinstance(perf_key, str) else None\n")
PY_DECL = ("    compute_ratio = single_tok = agg_tok = sat_tok = per_user_load = None\n"
           "    agg_obs_lo = agg_obs_hi = ttft_cold_ms = ttft_warm_ms = ttft_ms = None\n")
PY_COST = "    # Boards, not devices: a dual-GCD module is one line item on the invoice.\n"
PY_FIELDS = ["compute_ratio", "single_tok", "agg_tok", "sat_tok", "per_user_load", "agg_obs_lo",
             "agg_obs_hi", "ttft_cold_ms", "ttft_warm_ms", "ttft_ms"]
PY_PERF = {"perf_mbu": "mbu", "perf_mfu_decode": "mfuDecode", "perf_mfu_prefill": "mfuPrefill",
           "perf_obs_lo": "obsLo", "perf_obs_hi": "obsHi", "perf_fp8_ratio": "fp8ComputeRatio"}
PY_UNSUPPRESS = """    if not throughput_modelled:
        Q = PERF["nvidia"]
        _ab = device_bw * 1e9 * device_count * Q["mbu"] * nv_penalty
        def _d_at(b):
            d = active_weight_bytes + b * kv_bytes_per_seq
            return (b * _ab) / d if d > 0 else 0
        _cr = Q["fp8ComputeRatio"] if (is_fp8 and gpu_fp8) else 1.0
        _cc = ((Q["mfuDecode"] * device_tflops * 1e12 * device_count * nv_penalty * _cr) / (2 * total_active_p * 1e9)) if total_active_p > 0 else 0
        _apf = Q["mfuPrefill"] * device_tflops * 1e12 * device_count * nv_penalty * _cr
        def _tf(t):
            return round((2 * total_active_p * 1e9 * max(t, 0) / _apf) * 1000) if _apf > 0 else 0
        _agg = round(min(_d_at(eff_batch), _cc))
        _V = {"compute_ratio": _cr, "single_tok": round(_d_at(1)), "agg_tok": _agg,
              "sat_tok": round(min(_d_at(sat_batch), _cc)), "per_user_load": round(_agg / eff_batch) if eff_batch else 0,
              "agg_obs_lo": round(_agg * Q["obsLo"]), "agg_obs_hi": round(_agg * Q["obsHi"]),
              "ttft_cold_ms": _tf(ctx), "ttft_warm_ms": _tf(ctx - eff_prefix) if prefix_caching else _tf(ctx),
              "ttft_ms": _tf(ctx - eff_prefix) if prefix_caching else _tf(ctx)}
        FIELD = _V["FIELD"]
"""
PY_EXPLAIN = ("            story.append(Paragraph(\n"
              "                f\"No measured utilisation is published for {gpu['name']}: there are no memory-bandwidth \"\n"
              "                \"or compute utilisation figures for this hardware. Estimating its throughput or time to \"\n"
              "                \"first token would mean borrowing another architecture's constants, which do not \"\n"
              "                \"transfer, so this report gives neither. The VRAM, fit, cost and command figures do not \"\n"
              "                \"depend on them.\",\n"
              "                self.styles[\"Small\"]\n"
              "            ))\n")
PY_NOTMOD_ROW = "                [\"Throughput and TTFT\", \"Not modelled\"],\n"
PY_TP_GATE = "        if c[\"throughput_modelled\"]:\n            story.append(Paragraph(\"Throughput estimate\", self.styles[\"SectionHead\"]))\n"
PY_NOTES_PCIE = "'PCIe (64-128 GB/s) loses 30-50% decode throughput vs NVLink.' if c['throughput_modelled'] else 'PCIe provides 64-128 GB/s.'"
PY_FP8_NOTE = "        if c.get(\"fp8_no_tensor_cores\") and c[\"throughput_modelled\"]:\n"
PY_DP_NOTE = ("                         + (\", so the throughput and TTFT figures inherit that uncertainty.\"\n"
              "                            if c[\"throughput_modelled\"] else \".\"))\n")
PY_B_INTERACTIVE = "        \"perfKey\": gpu[\"perfKey\"],\n        \"kv_bpp\": kv_bpp,"
PY_B_JSON = "            \"perfKey\": gpu[\"perfKey\"],\n            \"kv_bpp\": raw.get(\"kv_bpp\", 2),"
PY_B_RAW = "    cfg[\"perfKey\"] = gpu[\"perfKey\"]\n"
PY_B_CLI = "            \"perfKey\": gpu[\"perfKey\"],\n            \"kv_bpp\": 1 if args.fp8_kv else 2,"
PY_MAXCTX_ROW = "            [\"Max context (1 user, 90% util)\", fmt_k(c[\"max_ctx_1\"]) + \" tokens\"],\n"
PY_RET_W = "        \"weights_gb\": weights_gb, \"kv_gb\": kv_gb, \"act_gb\": act_gb,\n"
PY_RET_BATCH = "        \"eff_batch\": eff_batch, \"max_batch_kv\": max_batch_kv, \"batch_limited\": batch_limited,\n"
PY_RET_TP = "        \"tp\": tp, \"dp\": dp,\n"
PY_RET_FITS = "        \"fits\": fits, \"comfortable\": comfortable,\n"
PY_COST_HEAD = "        story.append(Paragraph(\"Cost estimate\", self.styles[\"SectionHead\"]))\n"
SYNC_FIELDS = "              \"vendor\", \"perfKey\", \"devices\", \"form\", \"caps\")\n"

IDX, GR, SY = "index.html", "generate_report.py", "tools/sync_data.py"
S = {}  # name -> edits

# ---- A. fallback to NVIDIA's constants ----
S["A1 js: fallback PERF.nvidia for any key without an entry"] = [
    (IDX, "    ? PERF[state.perfKey] : null;\n", "    ? PERF[state.perfKey] : PERF.nvidia;\n", 1)]
S["A2 js: fallback only when perfKey is not a string"] = [
    (IDX, JS_LOOKUP, "  const P = typeof state.perfKey !== 'string' ? PERF.nvidia : Object.hasOwn(PERF, state.perfKey)\n    ? PERF[state.perfKey] : null;\n", 1)]
S["A3 js: fallback only when perfKey is absent (undefined)"] = [
    (IDX, JS_LOOKUP, "  const P = state.perfKey === undefined ? PERF.nvidia : (typeof state.perfKey === 'string' && Object.hasOwn(PERF, state.perfKey)\n    ? PERF[state.perfKey] : null);\n", 1)]
S["A4 js: look PERF up by vendor instead of perfKey"] = [
    (IDX, JS_LOOKUP, "  const P = typeof state.vendor === 'string' && Object.hasOwn(PERF, state.vendor)\n    ? PERF[state.vendor] : null;\n", 1)]
S["A5 js: vendor as a second-chance lookup"] = [
    (IDX, JS_LOOKUP, "  const P = typeof state.perfKey === 'string' && Object.hasOwn(PERF, state.perfKey)\n    ? PERF[state.perfKey] : (typeof state.vendor === 'string' && Object.hasOwn(PERF, state.vendor) ? PERF[state.vendor] : null);\n", 1)]
S["A6 js: `in` instead of Object.hasOwn (prototype chain)"] = [
    (IDX, "typeof state.perfKey === 'string' && Object.hasOwn(PERF, state.perfKey)\n    ? PERF[state.perfKey] : null;", "typeof state.perfKey === 'string' && state.perfKey in PERF\n    ? PERF[state.perfKey] : null;", 1)]
S["A7 py: fallback PERF['nvidia'] for any key without an entry"] = [
    (GR, "    P = PERF.get(perf_key) if isinstance(perf_key, str) else None\n", "    P = PERF.get(perf_key, PERF[\"nvidia\"]) if isinstance(perf_key, str) else PERF[\"nvidia\"]\n", 1)]
S["A8 py: fallback only when perfKey is not a string"] = [
    (GR, "    P = PERF.get(perf_key) if isinstance(perf_key, str) else None\n", "    P = PERF.get(perf_key) if isinstance(perf_key, str) else PERF[\"nvidia\"]\n", 1)]
S["A9 py: fallback only when perfKey is absent (None)"] = [
    (GR, "    P = PERF.get(perf_key) if isinstance(perf_key, str) else None\n", "    P = PERF[\"nvidia\"] if perf_key is None else (PERF.get(perf_key) if isinstance(perf_key, str) else None)\n", 1)]
S["A10 py: look PERF up by vendor instead of perfKey"] = [
    (GR, "    perf_key = cfg.get(\"perfKey\")\n", "    perf_key = cfg.get(\"vendor\")\n", 1)]
S["A11 py: vendor as a second-chance lookup"] = [
    (GR, "    P = PERF.get(perf_key) if isinstance(perf_key, str) else None\n", "    P = (PERF.get(perf_key) if isinstance(perf_key, str) else None) or PERF.get(str(cfg.get(\"vendor\")))\n", 1)]
S["A12 py: drop the isinstance guard (bare PERF.get)"] = [
    (GR, "    P = PERF.get(perf_key) if isinstance(perf_key, str) else None\n", "    P = PERF.get(perf_key)\n", 1)]
S["A13 both: fallback PERF.nvidia in both engines"] = S["A1 js: fallback PERF.nvidia for any key without an entry"] + S["A7 py: fallback PERF['nvidia'] for any key without an entry"]
S["A14 both: non-string fallback in both engines"] = S["A2 js: fallback only when perfKey is not a string"] + S["A8 py: fallback only when perfKey is not a string"]
S["A15 both: absent-key fallback in both engines"] = S["A3 js: fallback only when perfKey is absent (undefined)"] + S["A9 py: fallback only when perfKey is absent (None)"]
S["A16 both: vendor lookup in both engines"] = S["A4 js: look PERF up by vendor instead of perfKey"] + S["A10 py: look PERF up by vendor instead of perfKey"]
S["A17 both: vendor second chance in both engines"] = S["A5 js: vendor as a second-chance lookup"] + S["A11 py: vendor as a second-chance lookup"]

# ---- B. suppress in one engine, not the other ----
S["B1 js: never modelled (P = null for every card)"] = [
    (IDX, JS_LOOKUP, "  const P = null;\n", 1)]
S["B2 py: never modelled (P = None for every card)"] = [
    (GR, "    P = PERF.get(perf_key) if isinstance(perf_key, str) else None\n", "    P = None\n", 1)]
S["B3 js: always modelled with nvidia (unknown card modelled in JS only)"] = S["A1 js: fallback PERF.nvidia for any key without an entry"]
S["B4 py: always modelled with nvidia (unknown card modelled in Python only)"] = S["A7 py: fallback PERF['nvidia'] for any key without an entry"]
S["B5 js: throughputModelled reported true while figures stay null"] = [
    (IDX, "  const throughputModelled = P !== null;\n", "  const throughputModelled = true;\n", 1),
    (IDX, "  if (throughputModelled) {\n    const achievedBandwidth", "  if (P !== null) {\n    const achievedBandwidth", 1),
    (IDX, "    perfMbu: throughputModelled ? P.mbu : null,\n    perfMfuDecode: throughputModelled ? P.mfuDecode : null,\n    perfMfuPrefill: throughputModelled ? P.mfuPrefill : null,\n    perfObsLo: throughputModelled ? P.obsLo : null,\n    perfObsHi: throughputModelled ? P.obsHi : null,\n    perfFp8Ratio: throughputModelled ? P.fp8ComputeRatio : null,\n",
     "    perfMbu: P ? P.mbu : null,\n    perfMfuDecode: P ? P.mfuDecode : null,\n    perfMfuPrefill: P ? P.mfuPrefill : null,\n    perfObsLo: P ? P.obsLo : null,\n    perfObsHi: P ? P.obsHi : null,\n    perfFp8Ratio: P ? P.fp8ComputeRatio : null,\n", 1)]
S["B6 py: throughput_modelled reported True while figures stay None"] = [
    (GR, "    throughput_modelled = P is not None\n", "    throughput_modelled = True\n", 1),
    (GR, "    if throughput_modelled:\n        achieved_bw", "    if P is not None:\n        achieved_bw", 1),
    (GR, "        \"perf_mbu\": P[\"mbu\"] if throughput_modelled else None,\n        \"perf_mfu_decode\": P[\"mfuDecode\"] if throughput_modelled else None,\n        \"perf_mfu_prefill\": P[\"mfuPrefill\"] if throughput_modelled else None,\n        \"perf_obs_lo\": P[\"obsLo\"] if throughput_modelled else None,\n        \"perf_obs_hi\": P[\"obsHi\"] if throughput_modelled else None,\n        \"perf_fp8_ratio\": P[\"fp8ComputeRatio\"] if throughput_modelled else None,\n",
     "        \"perf_mbu\": P[\"mbu\"] if P else None,\n        \"perf_mfu_decode\": P[\"mfuDecode\"] if P else None,\n        \"perf_mfu_prefill\": P[\"mfuPrefill\"] if P else None,\n        \"perf_obs_lo\": P[\"obsLo\"] if P else None,\n        \"perf_obs_hi\": P[\"obsHi\"] if P else None,\n        \"perf_fp8_ratio\": P[\"fp8ComputeRatio\"] if P else None,\n", 1)]
S["B7 js: throughputModelled returned as undefined/0 rather than false"] = [
    (IDX, "  const throughputModelled = P !== null;\n", "  const throughputModelled = P !== null ? true : 0;\n", 1)]
S["B8 py: throughput_modelled returned as 0 rather than False"] = [
    (GR, "    throughput_modelled = P is not None\n", "    throughput_modelled = True if P is not None else 0\n", 1)]

# ---- C. 0 instead of null/None, per field ----
for f in JS_FIELDS:
    S[f"C js: {f} = 0 instead of null"] = [(IDX, JS_DECL, JS_DECL.replace(f"{f} = null", f"{f} = {'false' if f == 'computeBound' else '0'}"), 1)]
for f, expr in JS_PERF.items():
    S[f"C js: {f} = 0 instead of null"] = [(IDX, f"    {f}: throughputModelled ? {expr} : null,\n", f"    {f}: throughputModelled ? {expr} : 0,\n", 1)]
for f in PY_FIELDS:
    S[f"C py: {f} = 0 instead of None"] = [(GR, PY_DECL, PY_DECL + f"    {f} = 0\n", 1)]
for f, k in PY_PERF.items():
    S[f"C py: {f} = 0 instead of None"] = [(GR, f"        \"{f}\": P[\"{k}\"] if throughput_modelled else None,\n", f"        \"{f}\": P[\"{k}\"] if throughput_modelled else 0,\n", 1)]
S["C both: singleStreamTokS/single_tok = 0 in both engines"] = S["C js: singleStreamTokS = 0 instead of null"] + S["C py: single_tok = 0 instead of None"]
S["C both: ttftMs/ttft_ms = 0 in both engines"] = S["C js: ttftMs = 0 instead of null"] + S["C py: ttft_ms = 0 instead of None"]
S["C both: perfMbu/perf_mbu = 0 in both engines"] = S["C js: perfMbu = 0 instead of null"] + S["C py: perf_mbu = 0 instead of None"]
S["C js: computeBound = false instead of null (only)"] = S["C js: computeBound = 0 instead of null"]

# ---- D. leave one field unsuppressed (computed with nvidia's constants) ----
for f in JS_FIELDS:
    S[f"D js: {f} left unsuppressed"] = [(IDX, JS_COST, JS_UNSUPPRESS.replace("FIELD", f) + JS_COST, 1)]
for f, expr in JS_PERF.items():
    S[f"D js: {f} left unsuppressed"] = [(IDX, f"    {f}: throughputModelled ? {expr} : null,\n", f"    {f}: throughputModelled ? {expr} : PERF.nvidia.{expr[2:]},\n", 1)]
for f in PY_FIELDS:
    S[f"D py: {f} left unsuppressed"] = [(GR, PY_COST, PY_UNSUPPRESS.replace("FIELD", f) + PY_COST, 1)]
for f, k in PY_PERF.items():
    S[f"D py: {f} left unsuppressed"] = [(GR, f"        \"{f}\": P[\"{k}\"] if throughput_modelled else None,\n", f"        \"{f}\": P[\"{k}\"] if throughput_modelled else PERF[\"nvidia\"][\"{k}\"],\n", 1)]
S["D both: computeRatio/compute_ratio unsuppressed in both"] = S["D js: computeRatio left unsuppressed"] + S["D py: compute_ratio left unsuppressed"]
S["D both: ttftMs/ttft_ms unsuppressed in both"] = S["D js: ttftMs left unsuppressed"] + S["D py: ttft_ms left unsuppressed"]
S["D both: perfObsLo/perf_obs_lo unsuppressed in both"] = S["D js: perfObsLo left unsuppressed"] + S["D py: perf_obs_lo left unsuppressed"]
S["D both: aggregateObservedHiTokS/agg_obs_hi unsuppressed in both"] = S["D js: aggregateObservedHiTokS left unsuppressed"] + S["D py: agg_obs_hi left unsuppressed"]

# ---- E. delete the explanation from one surface ----
S["E1 js: throughput panel — no reason at all"] = [(IDX, JS_UNMODELLED_NOTE, "  const unmodelledNote = '';\n", 1)]
S["E2 js: throughput panel — only 'No measured utilisation' half"] = [(IDX, JS_UNMODELLED_NOTE, "  const unmodelledNote = computed.throughputModelled ? '' : `<span style=\"color:var(--warning-text);font-weight:500\">No measured utilisation for this hardware</span>`;\n", 1)]
S["E3 js: throughput panel — 'Not modelled' label removed, reason kept"] = [(IDX, JS_TP_TILE, JS_TP_TILE.replace("Not modelled", "—"), 1)]
S["E4 js: throughput panel — reason moved into the notes box"] = [
    (IDX, JS_UNMODELLED_NOTE, "  const unmodelledNote = '';\n", 1),
    (IDX, JS_NOTES_UNKNOWN, "    : 'Throughput is not modelled for this hardware: no measured utilisation is published for it, and borrowed constants do not transfer. ';\n", 1)]
S["E5 js: throughput panel — tile removed entirely for the unknown card"] = [(IDX, JS_TP_RETURN, "    setHTML('throughput-output', `<div class=\"reverse-grid\">${maxBatchCard}</div>${queueWarning}`);\n    return;\n  }\n", 1)]
S["E6 js: comparison card — bare 'not modelled'"] = [(IDX, JS_CMP_UNKNOWN, "      : `<div class=\"row\"><span class=\"label\">Per-user / aggregate</span><span class=\"val\">not modelled</span></div>`;\n", 1)]
S["E7 js: comparison card — row removed entirely"] = [(IDX, JS_CMP_UNKNOWN, "      : '';\n", 1)]
S["E8 js: comparison card — reason without 'not modelled'"] = [(IDX, JS_CMP_UNKNOWN, JS_CMP_UNKNOWN.replace("not modelled<br>", ""), 1)]
S["E9 js: exec summary — bare 'Not modelled'"] = [(IDX, JS_EXEC_UNKNOWN, "    html += `<div class=\"exec-row\"><span class=\"exec-label\">Speed and server throughput</span><span class=\"exec-value\">Not modelled</span></div>`;\n", 1)]
S["E10 js: exec summary — row removed entirely"] = [(IDX, JS_EXEC_UNKNOWN, "", 1)]
S["E11 js: exec summary — reason without 'Not modelled'"] = [(IDX, JS_EXEC_UNKNOWN, JS_EXEC_UNKNOWN.replace("Not modelled ", ""), 1)]
S["E12 js: markdown export — bare 'not modelled'"] = [(IDX, JS_MD_UNKNOWN, "    : `- Throughput and TTFT: not modelled.\\n`;\n", 1)]
S["E13 js: markdown export — line removed entirely"] = [(IDX, JS_MD_UNKNOWN, "    : '';\n", 1)]
S["E14 js: markdown export — reason without 'not modelled'"] = [(IDX, JS_MD_UNKNOWN, JS_MD_UNKNOWN.replace("not modelled. ", ""), 1)]
S["E15 js: notes box — unmodelled sentence removed"] = [(IDX, JS_NOTES_UNKNOWN, "    : '';\n", 1)]
S["E16 js: badge — unmodelled wording dropped (always the 45-60% loss)"] = [(IDX, JS_BADGE, "PCIe — 45-60% perf loss, worse with more devices", 1)]
S["E17 py: PDF — explanation paragraph removed"] = [(GR, PY_EXPLAIN, "", 1)]
S["E18 py: PDF — only the 'No measured utilisation' half"] = [(GR, PY_EXPLAIN, "            story.append(Paragraph(f\"No measured utilisation is published for {gpu['name']}.\", self.styles[\"Small\"]))\n", 1)]
S["E19 py: PDF — 'Not modelled' cell replaced by a dash"] = [(GR, PY_NOTMOD_ROW, "                [\"Throughput and TTFT\", \"\\u2014\"],\n", 1)]
S["E20 py: PDF — reason moved into Notes and assumptions"] = [
    (GR, PY_EXPLAIN, "", 1),
    (GR, "        notes.append(\"VRAM estimates include ~1.5 GB CUDA context overhead per device.\")\n",
     "        notes.append(\"VRAM estimates include ~1.5 GB CUDA context overhead per device.\")\n        if not c[\"throughput_modelled\"]:\n            notes.append(f\"No measured utilisation is published for {gpu['name']}; borrowed constants do not transfer.\")\n", 1)]
S["E21 py: PDF — throughput section omitted entirely for the unknown card"] = [
    (GR, "            story.append(Paragraph(\"Throughput\", self.styles[\"SectionHead\"]))\n            story.append(self._make_kv_table([\n                [\"Throughput and TTFT\", \"Not modelled\"],\n                max_batch_row,\n            ]))\n" + PY_EXPLAIN, "            pass\n", 1)]

# ---- F. suppress a constants-independent figure as well ----
S["F1 js: weightsGB null without constants"] = [(IDX, JS_RET_VRAM, "    isMoE, weightsGB: throughputModelled ? weightsGB : null, kvCacheGB, activationsGB, totalOverhead, totalGB,\n", 1)]
S["F2 js: maxBatchByKV null without constants"] = [(IDX, JS_RET_BATCH, "    maxBatchByKV: throughputModelled ? maxBatchByKV : null, batchLimitedByKV, computeBound, ttftMs, ttftColdMs, ttftWarmMs,\n", 1)]
S["F3 js: tp/dp null without constants"] = [(IDX, JS_RET_TP, "    tp: throughputModelled ? tp : null, dp: throughputModelled ? dp : null,\n    freeForKVCache,", 1)]
S["F4 js: hourlyHyper null without constants"] = [(IDX, JS_RET_COST, "    hourlyHyper: throughputModelled ? hourlyHyper : null, hourlySpec, hourlySpot,\n", 1)]
S["F5 js: fits null without constants"] = [(IDX, JS_RET_FITS, "    fits: throughputModelled ? perGPU.total <= deviceGB : null,\n", 1)]
S["F6 js: capacity panel hides max context without constants (renderer only)"] = [(IDX, JS_CAP_MAXCTX, "<p class=\"value\">${computed.throughputModelled ? formatContext(computed.maxContextSingleUser) + ' tokens' : 'not modelled'}</p>", 1)]
S["F7 js: metrics panel hides weights without constants (renderer only)"] = [(IDX, JS_METRICS_W, "<p class=\"metric-label\">${weightsLabel}</p><p class=\"metric-value\">${computed.throughputModelled ? formatGB(clusterWeights) : 'not modelled'}</p>", 1)]
S["F8 js: markdown export drops the per-device fit line without constants"] = [(IDX, JS_MD_PERDEV, "- Per-device: ${computed.throughputModelled ? formatGB(computed.perGPU.total) + ' / ' + capacityLabel(computed.deviceGB) + ' (' + (computed.fits ? 'FITS' : 'DOES NOT FIT') + ')' : 'not modelled'}\\n`;", 1)]
S["F9 js: markdown export drops the cost section without constants"] = [(IDX, JS_MD_COST, "  if (computed.throughputModelled) report += `\\n## Cost (${state.gpuCount}× ${state.gpuName})\\n", 1)]
S["F10 py: weights_gb None without constants"] = [(GR, PY_RET_W, "        \"weights_gb\": weights_gb if throughput_modelled else None, \"kv_gb\": kv_gb, \"act_gb\": act_gb,\n", 1)]
S["F11 py: max_batch_kv None without constants"] = [(GR, PY_RET_BATCH, "        \"eff_batch\": eff_batch, \"max_batch_kv\": max_batch_kv if throughput_modelled else None, \"batch_limited\": batch_limited,\n", 1)]
S["F12 py: tp/dp None without constants"] = [(GR, PY_RET_TP, "        \"tp\": tp if throughput_modelled else None, \"dp\": dp if throughput_modelled else None,\n", 1)]
S["F13 py: fits None without constants"] = [(GR, PY_RET_FITS, "        \"fits\": fits if throughput_modelled else None, \"comfortable\": comfortable,\n", 1)]
S["F14 py: PDF max-context row hidden without constants (renderer only)"] = [(GR, PY_MAXCTX_ROW, "            [\"Max context (1 user, 90% util)\", fmt_k(c[\"max_ctx_1\"]) + \" tokens\" if c[\"throughput_modelled\"] else \"not modelled\"],\n", 1)]
S["F15 py: PDF cost section skipped without constants (renderer only)"] = [(GR, PY_COST_HEAD + "        cost_data = [", "        if c[\"throughput_modelled\"]:\n          story.append(Paragraph(\"Cost estimate\", self.styles[\"SectionHead\"]))\n        cost_data = [", 1),
    (GR, "        story.append(cost_table)\n", "        if c[\"throughput_modelled\"]: story.append(cost_table)\n", 1)]
S["F16 both: weights null/None in both engines"] = S["F1 js: weightsGB null without constants"] + S["F10 py: weights_gb None without constants"]
S["F17 both: max batch null/None in both engines"] = S["F2 js: maxBatchByKV null without constants"] + S["F11 py: max_batch_kv None without constants"]

# ---- G. plumbing that drops perfKey ----
S["G1 js: getGpuSpec drops perfKey"] = [(IDX, JS_GETSPEC, "           vendor: g.vendor, devices: g.devices, form: g.form, caps: g.caps };\n", 1)]
S["G2 js: readInputState drops perfKey"] = [(IDX, JS_STATE_PK, "", 1)]
S["G3 js: readInputState fills perfKey from vendor"] = [(IDX, JS_STATE_PK, "    perfKey: gpu.vendor,\n", 1)]
S["G4 js: getGpuSpec fills perfKey from vendor"] = [(IDX, JS_GETSPEC, "           vendor: g.vendor, perfKey: g.vendor, devices: g.devices, form: g.form, caps: g.caps };\n", 1)]
S["G5 js: getGpuSpec hardcodes perfKey 'nvidia'"] = [(IDX, JS_GETSPEC, "           vendor: g.vendor, perfKey: 'nvidia', devices: g.devices, form: g.form, caps: g.caps };\n", 1)]
S["G6 py: interactive_mode drops perfKey"] = [(GR, PY_B_INTERACTIVE, "        \"kv_bpp\": kv_bpp,", 1)]
S["G7 py: from_json preset branch drops perfKey"] = [(GR, PY_B_JSON, "            \"kv_bpp\": raw.get(\"kv_bpp\", 2),", 1)]
S["G8 py: from_json raw branch drops perfKey"] = [(GR, PY_B_RAW, "", 1)]
S["G9 py: from_cli_args drops perfKey"] = [(GR, PY_B_CLI, "            \"kv_bpp\": 1 if args.fp8_kv else 2,", 1)]
S["G10 py: from_cli_args fills perfKey from vendor"] = [(GR, PY_B_CLI, "            \"perfKey\": gpu[\"vendor\"],\n            \"kv_bpp\": 1 if args.fp8_kv else 2,", 1)]
S["G11 py: from_json lets the JSON override perfKey"] = [(GR, PY_B_JSON, "            \"perfKey\": raw.get(\"perfKey\", gpu[\"perfKey\"]),\n            \"kv_bpp\": raw.get(\"kv_bpp\", 2),", 1)]
S["G12 py: from_json raw branch lets the JSON override perfKey"] = [(GR, PY_B_RAW, "    cfg.setdefault(\"perfKey\", gpu[\"perfKey\"])\n", 1)]
S["G13 py: from_cli_args hardcodes perfKey 'nvidia'"] = [(GR, PY_B_CLI, "            \"perfKey\": \"nvidia\",\n            \"kv_bpp\": 1 if args.fp8_kv else 2,", 1)]
S["G14 sync: perfKey no longer required by tools/sync_data.py"] = [(SY, SYNC_FIELDS, "              \"vendor\", \"devices\", \"form\", \"caps\")\n", 1)]
S["G15 sync: a missing perfKey defaults to 'nvidia' in the generated blocks"] = [
    (SY, "    missing = [f for f in fields if f not in row]\n", "    row = dict(row); row.setdefault(\"perfKey\", \"nvidia\")\n    missing = [f for f in fields if f not in row]\n", 1)]
S["G16 both: readInputState and from_cli_args both fill perfKey from vendor"] = S["G3 js: readInputState fills perfKey from vendor"] + S["G10 py: from_cli_args fills perfKey from vendor"]

# ---- H. surfaces that still print figures or speak of speed ----
S["H1 js: throughput panel prints figures for the unknown card (gate removed)"] = [(IDX, JS_TP_GATE, "  if (false) {\n    /* And no benchmark panel.", 1)]
S["H2 js: comparison card prints figures for the unknown card"] = [(IDX, JS_CMP_GATE, JS_CMP_GATE.replace("c.throughputModelled", "true"), 1)]
S["H3 js: exec summary prints figures for the unknown card"] = [(IDX, JS_EXEC_GATE, JS_EXEC_GATE.replace("computed.throughputModelled", "true"), 1)]
S["H4 js: markdown export prints figures for the unknown card"] = [(IDX, JS_MD_GATE, JS_MD_GATE.replace("computed.throughputModelled", "true"), 1)]
S["H5 js: notes always say 'memory-bandwidth-bound decode estimate'"] = [(IDX, JS_NOTES_GATE, JS_NOTES_GATE.replace("computed.throughputModelled", "true"), 1)]
S["H6 js: notes always quote the 45-60% PCIe decode loss"] = [(IDX, JS_NOTES_PCIE, "'PCIe 64-128 GB/s — 45-60% decode loss, growing with device count. '", 1)]
S["H7 js: exec dp>1 note always points at 'the throughput figures'"] = [(IDX, JS_EXEC_DP, ", so treat the throughput figures as a ceiling to test, not a quote", 1)]
S["H8 js: queue warning always points at 'aggregate throughput above'"] = [(IDX, JS_QUEUE, " — aggregate throughput above reflects the batch that actually runs", 1)]
S["H9 py: PDF prints the modelled table for the unknown card (gate removed)"] = [(GR, PY_TP_GATE, PY_TP_GATE.replace("if c[\"throughput_modelled\"]:", "if True:"), 1)]
S["H10 py: PDF notes always quote the 30-50% PCIe decode loss"] = [(GR, PY_NOTES_PCIE, "'PCIe (64-128 GB/s) loses 30-50% decode throughput vs NVLink.'", 1)]
S["H11 py: PDF FP8 note printed without constants"] = [(GR, PY_FP8_NOTE, "        if c.get(\"fp8_no_tensor_cores\"):\n", 1)]
S["H12 py: PDF dp>1 note always mentions 'the throughput and TTFT figures'"] = [(GR, PY_DP_NOTE, "                         + \", so the throughput and TTFT figures inherit that uncertainty.\")\n", 1)]
S["H13 py: PDF keeps the 'Basis' row for the unknown card"] = [(GR, "                [\"Throughput and TTFT\", \"Not modelled\"],\n                max_batch_row,\n", "                [\"Throughput and TTFT\", \"Not modelled\"],\n                max_batch_row,\n                [\"Basis\", f\"Memory-bandwidth bound — {c['device_bw']:g} GB/s x {c['device_count']} device(s)\"],\n", 1)]
S["H14 js: unknown card's tile says 'Bandwidth-bound' in words"] = [(IDX, JS_TP_TILE, JS_TP_TILE.replace("<p class=\"value\">Not modelled</p>", "<p class=\"value\">Not modelled</p><p class=\"sub\">Bandwidth-bound decode would apply here</p>"), 1)]
S["H15 js: unknown card's tile keeps 'observed range' wording"] = [(IDX, JS_TP_TILE, JS_TP_TILE.replace("<p class=\"value\">Not modelled</p>", "<p class=\"value\">Not modelled</p><p class=\"sub\">observed range unavailable</p>"), 1)]

# ---- I. structural / result-shape ----
S["I1 js: singleStreamTokS omitted from the result (undefined) without constants"] = [(IDX, "    singleStreamTokS, aggregateTokS, perUserAtLoadTokS, effectiveBatch,\n", "    ...(throughputModelled ? { singleStreamTokS } : {}), aggregateTokS, perUserAtLoadTokS, effectiveBatch,\n", 1)]
S["I2 py: single_tok omitted from the result without constants"] = [(GR, "        \"single_tok\": single_tok, \"agg_tok\": agg_tok, \"per_user_load\": per_user_load,\n", "        **({\"single_tok\": single_tok} if throughput_modelled else {}), \"agg_tok\": agg_tok, \"per_user_load\": per_user_load,\n", 1)]
S["I3 js: unknown key throws instead of nulling"] = [(IDX, "  const throughputModelled = P !== null;\n", "  const throughputModelled = P !== null;\n  if (!throughputModelled) throw new Error('no constants');\n", 1)]
S["I4 py: unknown key raises instead of None"] = [(GR, "    throughput_modelled = P is not None\n", "    throughput_modelled = P is not None\n    if not throughput_modelled:\n        raise KeyError(perf_key)\n", 1)]

if __name__ == "__main__":
    pats = sys.argv[1:]
    if pats and pats[0] == "--from":
        # Everything from the named sabotage onward, in definition order.
        keys = list(S)
        names = keys[keys.index(next(k for k in keys if k.startswith(pats[1]))):]
    else:
        names = [n for n in S if not pats or any(p.lower() in n.lower() for p in pats)]
    print(f"{len(names)} sabotage(s)")
    require_green_baseline()
    survived, unapplied = [], []
    for name in names:
        try:
            touched = apply(S[name])
        except Exception as e:
            print(f"  !! {name}: could not apply: {e}")
            unapplied.append(name)
            restore([IDX, GR, SY])
            continue
        try:
            res = run_suites()
        finally:
            restore(touched)
        red = {k: v for k, v in res.items() if v[0] != 0}
        if not red:
            survived.append(name)
            print(f"  GREEN  {name}   <-- SURVIVED")
        else:
            bits = []
            for k, (rc, fails, errs, tally) in red.items():
                first = (fails or errs or ["(no FAIL line)"])[0]
                bits.append(f"{k}[{tally[1] if tally else '?'} failed: {first[:110]}]")
            print(f"  red    {name}\n         " + "\n         ".join(bits))
    print(f"\n{len(names) - len(survived) - len(unapplied)} caught, "
          f"{len(survived)} survived"
          + (f", {len(unapplied)} COULD NOT BE APPLIED" if unapplied else ""))
    for s in survived:
        print("  SURVIVED: " + s)
    if unapplied:
        # A sabotage that never reached the tree judged nothing. Counting it as
        # caught is how a drifted driver reports a clean run forever.
        sys.exit(2)
