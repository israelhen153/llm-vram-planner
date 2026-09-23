"""What engine sabotages are built from: the text they anchor on, and the parts
they assemble their payloads out of.

Most constants here are verbatim excerpts of index.html (JS_*),
generate_report.py (PY_*) or tools/sync_data.py (SYNC_*). A sabotage finds one
and replaces it with a broken version. If the engine is edited and an excerpt
stops matching, every sabotage anchored on it refuses to apply — which is the
point: a stale anchor fails loudly, rather than letting a sabotage quietly miss
its target and count as caught.

Six are NOT excerpts and will never be found in the engine, by design:
JS_FIELDS and PY_FIELDS list the throughput fields a sabotage is generated for,
JS_PERF and PY_PERF map constant names onto their PERF keys, and JS_UNSUPPRESS
and PY_UNSUPPRESS are payload templates — a block that re-derives throughput
from NVIDIA's constants, with FIELD replaced per sabotage before it is inserted.

tests/corpus.test.py holds this file to that description, in the suite: every
excerpt must still occur in the file its prefix names, and every payload part
must occur in none — one found in the engine would mean the engine already
contains the defect its sabotage inserts.
"""

# The six constants above that are not excerpts. Declared rather than inferred,
# so tests/corpus.test.py can check the declaration in both directions.
PAYLOAD_PARTS = ("JS_FIELDS", "JS_PERF", "JS_UNSUPPRESS", "PY_FIELDS", "PY_PERF", "PY_UNSUPPRESS")
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
JS_GETSPEC = "           vendor: g.vendor, perfKey: g.perfKey, devices: g.devices, form: g.form, caps: g.caps,\n"
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
JS_RET_VRAM = "    isMoE, weightsGB, kvCacheGB, activationsGB, totalOverhead, peerBufferGB, totalGB,\n"
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
# The two clocks the report prints, which tests/report.test.py holds still —
# see engine_r8_held_clock.py.
PY_CLOCK_COVER = "datetime.now().strftime('%B %d, %Y at %H:%M')"
PY_CLOCK_FOOTER = "datetime.now().strftime(\"%Y-%m-%d\")"
SYNC_FIELDS = "              \"vendor\", \"perfKey\", \"devices\", \"form\", \"caps\")\n"

# fix/cost-provenance: priceSourceLabel()/price_source_label() and every
# surface that calls them — see engine_r4_cost_provenance.py.
JS_PROVIDER_NAMES = ("const PROVIDER_NAMES = { azure: 'Azure', aws: 'AWS', lambda: 'Lambda', "
                     "coreweave: 'CoreWeave', vast: 'Vast.ai' };\n")
JS_PRICE_LABEL_RET = "  return `${provider} · ${src.sku} · ${src.region} · read ${src.date}`;\n"
JS_COST_HYPER_SUBLABEL = ("<span style=\"font-size:11px;color:var(--text-muted)\">"
                          "${priceSourceLabel(state, 'hyper')}</span>")
JS_CMP_COST_SOURCE = ("    html += `<div class=\"row\"><span class=\"label\">Cost source</span>"
                      "<span class=\"val\" style=\"font-size:10px;line-height:1.5;text-align:right\">"
                      "Hyper: ${priceSourceLabel(s, 'hyper')}<br>Spec: ${priceSourceLabel(s, 'spec')}"
                      "<br>Spot: ${priceSourceLabel(s, 'spot')}</span></div></div>`;\n")
JS_EXEC_COST_SOURCE = ("Cost source</span><span class=\"exec-value\" style=\"font-size:11px;"
                       "font-weight:400;color:var(--text-muted)\">Hyper: "
                       "${priceSourceLabel(state, 'hyper')} · Spec: ${priceSourceLabel(state, 'spec')} "
                       "· Spot: ${priceSourceLabel(state, 'spot')}</span>")
JS_CMP_RANGE = ("`$${Math.min(...cmpCosts).toFixed(2)}"
                "–$${Math.max(...cmpCosts).toFixed(2)}`")
JS_EXEC_RANGE = "const monthlyCheapest = Math.round(Math.min(...execCosts) * 730);\n"
PY_PRICE_LABEL_RET = "    return f\"{provider} · {src['sku']} · {src['region']} · read {src['date']}\"\n"
PY_TIER_HYPERSCALER = "            [\"Hyperscaler\",\n"
PY_SOURCE_HYPER_LINE = "            f\"Source — Hyperscaler: {price_source_label(gpu, 'hyper')}. \"\n"
PY_NOTES_COMPOSITE_OPEN = ("        notes.append(\"GPU prices are mid-2026 per-board/hr figures "
                           "across 3 tiers: hyperscaler, \"\n")
SYNC_OPTIONAL = "GPU_OPTIONAL = (\"default\", \"priceSource\", \"priceRecord\", \"priceNote\", \"priceLead\", \"gfx\")\n"
PRICE_CROSS_CHECK_THRESHOLD = "CROSS_CHECK_FLAG_THRESHOLD = 0.20\n"
PRICE_APPLY_LOOP = "    new_text = raw_text\n    for slug, row in gpus_data.items():\n"
PRICE_FIELD_LINE = "            \"price\": round(oc.reading.price_per_gpu, 2),\n"
GPUS_ADA_ROW = ("\"rtx6000ada-48\": { \"gb\": 48, \"bw\": 960, \"hyper\": null, \"spec\": 0.9, \"spot\": 0.6, \"tflops\": 364, "
                "\"name\": \"RTX 6000 Ada 48 GB\", \"vendor\": \"nvidia\", \"perfKey\": \"nvidia\", \"devices\": 1, \"form\": \"pc"
                "ie\", \"caps\": { \"fp8\": true }, \"priceNote\": { \"hyper\": { \"reason\": \"No hyperscaler rents this car"
                "d: AWS, Azure, Google Cloud and Oracle list no offer for it.\", \"checked\": \"2026-09-23\" }, \"spec\""
                ": { \"reason\": \"In the catalog since 2026-07-27 as a mid-2026 list price, with no provider record"
                "ed. Lambda's and CoreWeave's A6000 is the older Ampere RTX A6000, not this card.\", \"checked\": \"2"
                "026-09-16\" }, \"spot\": { \"reason\": \"In the catalog since 2026-07-27 as a mid-2026 list price, wit"
                "h no provider recorded. The weekly check can read Vast.ai, but its name for the RTX 6000 Ada has"
                "n't been confirmed against a live query yet.\", \"checked\": \"2026-09-23\" } } }")
GPUS_H100_HYPER_PREFIX = "\"h100-80\": { \"gb\": 80, \"bw\": 3352, \"hyper\": 12.3,"
GPUS_H100_SRC_BLOCK = ("\"priceSource\": { \"hyper\": { \"provider\": \"azure\", "
                      "\"sku\": \"Standard_ND96isr_H100_v5\", \"region\": \"eastus\", "
                      "\"date\": \"2026-09-22\", \"price\": 12.29 }")

# ---- feat/rocm-guidance (engine_r10_rocm_guidance.py): the FP8 gate, the ROCm command and
# its lines, the notes, the leads, and the overhead's words ----
JS_R10_FP8_LINE = "  if (!arch.fp8Weights) out.push(L.fp8Weights);\n"
JS_R10_BLOCKED = "  if (!gpu || gpu.vendor !== 'amd' || (ROCM.arch[gpu.gfx] || {}).fp8Weights) return '';\n"
JS_R10_BUILD_REFUSE = "  if ((state.quantMethod === 'fp8' || state.bytesPerParam === 1) && fp8WeightsBlocked({ vendor: state.vendor, gfx: state.gfx, name: state.gpuName }))\n"
JS_R10_ARCH = "  const arch = state.vendor === 'amd' ? (ROCM.arch[state.gfx] || {}) : null;\n"
JS_R10_AITER = "      + (arch.aiter ? '    --env VLLM_ROCM_USE_AITER=1 \\\\\\n' : '')\n"
JS_R10_IMAGE = "  \"image\": \"vllm/vllm-openai-rocm:v0.30.0\",\n"
JS_R10_GCD = "  if ((state.gpuDevices || 1) > 1) out.push(L.gcd);\n"
JS_R10_PANEL_ROCM = "  const rocm = computed.fits ? rocmGuidance(state) : [];\n"
JS_R10_EXPORT_ROCM = "  const rocmLines = computed.fits ? rocmGuidance(state) : [];\n"
JS_R10_PEER = "  const peerBufferGB = deviceCount > 1 ? (hasNVLink ? 0.3 : 0.2) : 0;\n"
JS_R10_PEER_TEXT = "    + `${state.vendor === 'amd' ? 'RCCL' : 'NCCL'} buffers ~${computed.peerBufferGB} GB/peer. `;\n"
JS_R10_SYNC_FORCE = "  if (blocked && fp8.selected) {\n"
JS_R10_SYNC_GIVE = "    if (bf16.selected) fp8.selected = true;\n"
JS_R10_SYNC_CALL = "  syncInterconnect();\n  syncPrecision();\n"
JS_R10_NOTE_TEXT = "  return note ? `${note.reason} Checked ${note.checked}.` : '';\n"
JS_R10_NOTE_LINE = "    + (priceNoteText(state, tier) ? `  - ${priceNoteText(state, tier)}\\n` : '')\n"
JS_R10_LEAD_LINE = "    + priceLeads(state, tier).map(lead => `  - ${priceLeadText(lead)}\\n`).join('');\n"
JS_R10_LEAD_GATE = "  if (state[TIER_COST_FIELD[tier]] !== null) return [];\n"
PY_R10_FP8_LINE = "    if not arch.get(\"fp8Weights\"):\n        out.append(lines[\"fp8Weights\"])\n"
PY_R10_BLOCKED = "    if gpu.get(\"vendor\") != \"amd\" or ROCM[\"arch\"].get(gpu.get(\"gfx\"), {}).get(\"fp8Weights\"):\n"
PY_R10_REFUSE_RAISE = "            raise PlanRefused(f\"{reason} Choose --prec bf16, awq or gptq.\")\n"
PY_R10_EXIT2 = "        print(f\"error: {refused}\", file=sys.stderr)\n        sys.exit(2)\n"
PY_R10_MENU_FILTER = "        prec_opts = [o for o in prec_opts if o[1] != \"FP8\"]\n"
PY_R10_MENU_QUANT = "    bpp, quant = prec_opts[prec_choice][0], prec_opts[prec_choice][2]\n"
PY_R10_JSON_FP8 = "    if cfg.get(\"bpp\") == 1 and not cfg.get(\"quant\"):\n        cfg[\"quant\"] = \"fp8\"\n"
PY_R10_ARCH = "    arch = ROCM[\"arch\"].get(gpu.get(\"gfx\"), {}) if gpu.get(\"vendor\") == \"amd\" else None\n"
PY_R10_LIBRARY = "            library = \"RCCL\" if cfg[\"gpu\"].get(\"vendor\") == \"amd\" else \"NCCL\"\n"
PY_R10_PEER = "    peer_buffer_gb = (0.3 if nvlink else 0.2) if device_count > 1 else 0\n"
PY_R10_WHY = "            story.append(Paragraph(\"Why — \" + \" \".join(why), self.styles[\"Small\"]))\n"
PY_R10_LEADS = "                story.append(Paragraph(f\"{name} — {price_lead_text(lead)}\", self.styles[\"Small\"]))\n"
PY_R10_PDF_ROCM = "        rocm = rocm_guidance(cfg) if c[\"fits\"] else []\n"
PY_R10_NOTE_TEXT = "    return f\"{note['reason']} Checked {note['checked']}.\" if note else \"\"\n"
PRICE_R10_NOTE_DROP = "        if note and oc.tier in note:\n            del note[oc.tier]\n            if not note:\n                del row[\"priceNote\"]\n"

JS_R10_HIP_SUB = "HIP_VISIBLE_DEVICES=0,1"
PY_R10_HIP_SUB = "HIP_VISIBLE_DEVICES=0,1"
PY_R10_IMAGE = "'image': 'vllm/vllm-openai-rocm:v0.30.0'"
JS_R10_COST_SPOT = "${priceNoteLine(state, 'spot')}${priceLeadLines(state, 'spot')}</td>"
JS_R10_HOURLY_SPEC = "  const hourlySpec = tierCost(gpuSpecCost);\n"
PY_R10_HOURLY_SPEC = "    hourly_spec = None if gpu[\"spec\"] is None else gpu[\"spec\"] * n_gpu\n"
PY_R10_CONTEXT_AMD = "        if cfg[\"gpu\"].get(\"vendor\") == \"amd\":\n"
JS_R10_GFX90A = "\"gfx90a\": {\n      \"fp8Weights\": false,"
PY_R10_GFX90A = "'gfx90a': {'fp8Weights': False,"
PY_R10_GFX942 = "'gfx942': {'fp8Weights': True, 'aiter': True,"

# The engine files a sabotage edits.

# feat/amd-gpus: what engine_r6_amd_rows.py attacks — the null price tier, the
# interconnect's name, the vendor sections, and price_check's null-tier rules.
JS_TIER_COST = "  const tierCost = (perBoard) => (perBoard == null ? null : perBoard * gpuCount);\n"
JS_HOURLY_HYPER = "  const hourlyHyper = tierCost(gpuHyperCost);\n"
JS_NULL_LABEL = "  if (state[TIER_COST_FIELD[tier]] === null) return NO_PRICE;\n"
JS_CMP_COSTS = "    const cmpCosts = [c.hourlyHyper, c.hourlySpec, c.hourlySpot].filter(v => v != null);\n"
JS_EXEC_COSTS = ("  const execCosts = [computed.hourlyHyper, computed.hourlySpec, computed.hourlySpot]"
                 ".filter(v => v != null);\n")
JS_USD_DASH = "  if (v == null) return '—';\n"
JS_INTERCONNECT_NAME = "  return state.hasNVLink ? 'NVLink' : state.gpuForm === 'oam' ? 'Infinity Fabric' : 'PCIe';\n"
JS_SHARDING_ONE = ("    : (fabric !== 'PCIe' ? fabric + ' — sharded ' + computed.deviceCount + '-way, each device "
                   "holds 1/' + computed.deviceCount + ' of model' : 'PCIe — same sharding, 5-12x slower all-reduce')}"
                   "</div>`;\n")
JS_SYNC_OAM_LABEL = "  sel.options[1].textContent = gpu && gpu.form === 'oam' ? 'Infinity Fabric' : 'PCIe only';\n"
JS_VENDORS = "  const vendors = [...new Set(rows.map(([, gpu]) => gpu.vendor))];\n"
JS_STATE_FORM = "    gpuForm: gpu.form,\n"
PY_HOURLY_HYPER = "    hourly_hyper = None if gpu[\"hyper\"] is None else gpu[\"hyper\"] * n_gpu\n"
PY_NULL_LABEL = "    if tier in gpu and gpu[tier] is None:\n        return NO_PRICE\n"
PY_INTERCONNECT_NAME = "    return \"Infinity Fabric\" if cfg[\"gpu\"].get(\"form\") == \"oam\" else \"PCIe\"\n"
PY_INTERCONNECT_ROW = "            [\"Interconnect\", interconnect_name(cfg)],\n"
PY_MENU_SPAN = "        if v[\"spot\"] is not None and v[\"hyper\"] is not None:\n"
PRICE_SPOT_SKU = "    recorded_sku = f\"{sku} ({kind})\" if kind else sku\n"
PRICE_NULL_GUARD = "            if row.get(tier) is None:\n"
PRICE_NULL_REFUSAL = "    on_null = automated_null_tiers(catalog[\"data\"], SOURCE_MAP)\n    if on_null:\n"
# The first cold check of feat/amd-gpus (engine_r7_amd_cold_check.py): the page's
# catalog-to-state path, the hand record's tier lookup, the fabric's gate, and
# price_check's writer.
JS_GETSPEC_PRICES = "  return { key: el.value, gb: g.gb, bw: g.bw, hyper: g.hyper, spec: g.spec, spot: g.spot,\n"
JS_STATE_SPOT = "    gpuSpotCost: gpu.spot,\n"
JS_REC_LOOKUP = "  const rec = !src && state.priceRecord && state.priceRecord[tier];\n"
JS_FABRIC = "  const fabric = interconnectName(state);\n"
PY_REC_LOOKUP = "    rec = None if src else (gpu.get(\"priceRecord\") or {}).get(tier)\n"
PRICE_DUMP_ROW = "    parts = \", \".join(f\"{json.dumps(k)}: {_dump_json_value(v)}\" for k, v in row.items())\n"
PRICE_DUMP_VALUE = ("        return \"{ \" + \", \".join(f\"{json.dumps(k)}: {_dump_json_value(x)}\" for k, x in v.items())"
                    " + \" }\"\n")
PRICE_APPLY_MOVED = "        if oc.status == \"MOVED\":\n            row[oc.tier] = oc.proposed\n"

INDEX_HTML = "index.html"
REPORT_PY = "generate_report.py"
SYNC_PY = "tools/sync_data.py"
PRICE_PY = "tools/price_check.py"
GPUS_JSON = "data/gpus.json"
