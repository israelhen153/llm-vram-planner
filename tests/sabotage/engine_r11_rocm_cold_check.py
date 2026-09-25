#!/usr/bin/env python3
"""Round 11: what the first cold check of feat/rocm-guidance found.

A checker with no prior context attacked 8405742 with 40 sabotages. Seven broke
a claim on inputs the tool documents and left every behavioural suite green,
six were caught only by a golden, four more were unreachable with today's
catalog, and two bugs turned up without any sabotage at all:

1. **Two bugs.** "quant": "FP8" in a JSON config got past the FP8 refusal, and
   any --prec the CLI didn't know was planned as AWQ 4-bit. Both are put back
   here, A1 to A3.
2. **Inputs the tests never used** (B1 to B6, B13): a local model outside /opt,
   a quantized choice made before switching to a card that can't run FP8, a
   MOVED reading, from_json()'s own-fields branch, and a --prec table the tests
   read their expectations from.
3. **Board counts the lead sweep skipped** (B7, B8): exactly two, and four and more.
4. **Contracts held only by the goldens** (B9 to B12): the lead sentence's gate,
   the per-device buffer off NVLink, and a price in a note without a "$".
5. **Shapes the catalog refuses** (C1 to C4): a note beside a hand record, a lead
   on a priced tier or on the hyperscaler's, and an AMD target the table lacks.

Each is kept here so the next change starts where this check ended.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import INDEX_HTML, REPORT_PY, PRICE_PY, GPUS_JSON

S = {
    # ---- 1. the two bugs, put back ----
    "A1 py: a config's quantization is compared as written, so 'FP8' gets past the refusal": [
        (REPORT_PY, '    if isinstance(cfg.get("quant"), str):\n        cfg["quant"] = cfg["quant"].lower()\n', "", 1)],
    "A2 py: from_cli_args() plans a --prec it doesn't know as AWQ": [
        (REPORT_PY, '    if args.prec not in PRECISIONS:\n'
                    '        raise ValueError(f"Unknown precision: {args.prec}. Available: {\', \'.join(PRECISIONS)}")\n', "", 1),
        (REPORT_PY, "        bpp, quant = PRECISIONS[args.prec]\n",
                    '        bpp, quant = PRECISIONS.get(args.prec, (0.5, "awq"))\n', 1)],
    "A3 py: the command line takes any --prec, not only the ones it knows": [
        (REPORT_PY, 'parser.add_argument("--prec", default="awq", choices=list(PRECISIONS),',
                    'parser.add_argument("--prec", default="awq",', 1)],

    # ---- 2. inputs the tests never used ----
    "B1 both: a local model is mounted into the ROCm container only under /opt": [
        (INDEX_HTML, "(modelPath.startsWith('/') ? `", "(modelPath.startsWith('/opt') ? `", 1),
        (REPORT_PY, 'if str(model or "").startswith("/"):', 'if str(model or "").startswith("/opt"):', 1)],
    "B2 js: a card that can't run FP8 sends every precision to BF16, not only FP8": [
        (INDEX_HTML, "  if (blocked && fp8.selected) {\n    precisionForcedFromFp8 = true;\n    bf16.selected = true;\n  }",
                     "  if (blocked) {\n    if (fp8.selected) precisionForcedFromFp8 = true;\n    bf16.selected = true;\n  }", 1)],
    "B3 price: --apply keeps a tier's note beside a MOVED reading": [
        (PRICE_PY, "        if note and oc.tier in note:",
                   '        if oc.status == "CONFIRMED" and note and oc.tier in note:', 1)],
    "B4 py: the one-byte FP8 fill only in from_json()'s preset branch": [
        (REPORT_PY, '    if cfg.get("bpp") == 1 and not cfg.get("quant"):\n        cfg["quant"] = "fp8"\n', "", 1),
        (REPORT_PY, '            "model_name": raw.get("model_name", pd["name"]),\n        })\n        return validate_arch(cfg)\n',
                    '            "model_name": raw.get("model_name", pd["name"]),\n        })\n'
                    '        if cfg.get("bpp") == 1 and not cfg.get("quant"):\n            cfg["quant"] = "fp8"\n'
                    '        return validate_arch(cfg)\n', 1)],
    "B5 py: --prec q8 names no quantization": [
        (REPORT_PY, '"q8": (1.1, "gguf")', '"q8": (1.1, "")', 1)],
    "B6 py: from_json()'s own-fields branch swallows the FP8 refusal": [
        (REPORT_PY, "    cfg.setdefault(\"model_name\", f\"{cfg['params']}B model\")\n    return validate_arch(cfg)\n",
                    "    cfg.setdefault(\"model_name\", f\"{cfg['params']}B model\")\n"
                    "    try:\n        return validate_arch(cfg)\n    except PlanRefused:\n        return cfg\n", 1)],
    "B13 py: --prec fp8 sized at two bytes a parameter, as BF16": [
        (REPORT_PY, '"fp8": (1, "fp8")', '"fp8": (2, "fp8")', 1)],

    # ---- 3. board counts the lead sweep skipped ----
    "B7 js: a lead's price becomes the per-board figure at four boards and more": [
        (INDEX_HTML, "${usd(state.gpuSpecCost, 2)}",
                     "${usd(state.gpuSpecCost ?? (state.gpuCount >= 4 && priceLeads(state, 'spec').length ? priceLeads(state, 'spec')[0].price : null), 2)}", 1)],
    "B7 py: a lead's price becomes the per-board figure at four boards and more": [
        (REPORT_PY, "            [\"Specialized\",\n             usd(gpu['spec']),",
                    "            [\"Specialized\",\n             usd(gpu['spec'] if gpu['spec'] is not None or cfg['n_gpu'] < 4 "
                    "or not price_leads(gpu, 'spec') else price_leads(gpu, 'spec')[0]['price']),", 1)],
    "B8 js: a lead's price becomes the per-board figure at exactly two boards": [
        (INDEX_HTML, "${usd(state.gpuSpecCost, 2)}",
                     "${usd(state.gpuSpecCost ?? (state.gpuCount === 2 && priceLeads(state, 'spec').length ? priceLeads(state, 'spec')[0].price : null), 2)}", 1)],
    "B8 py: a lead's price becomes the per-board figure at exactly two boards": [
        (REPORT_PY, "            [\"Specialized\",\n             usd(gpu['spec']),",
                    "            [\"Specialized\",\n             usd(gpu['spec'] if gpu['spec'] is not None or cfg['n_gpu'] != 2 "
                    "or not price_leads(gpu, 'spec') else price_leads(gpu, 'spec')[0]['price']),", 1)],

    # ---- 4. contracts only the goldens held ----
    "B9 js: the notes explain leads only when the specialized tier has one": [
        (INDEX_HTML, "if (['hyper', 'spec', 'spot'].some(tier => priceLeads(state, tier).length))",
                     "if (['spec'].some(tier => priceLeads(state, tier).length))", 1)],
    "B9 py: the PDF's notes explain leads only when the specialized tier has one": [
        (REPORT_PY, 'if any(price_leads(gpu, t) for t in ("hyper", "spec", "spot")):',
                    'if any(price_leads(gpu, t) for t in ("spec",)):', 1)],
    "B10 both: 0.3 GB per extra device off NVLink too": [
        (INDEX_HTML, "const peerBufferGB = deviceCount > 1 ? (hasNVLink ? 0.3 : 0.2) : 0;",
                     "const peerBufferGB = deviceCount > 1 ? (hasNVLink ? 0.3 : 0.3) : 0;", 1),
        (REPORT_PY, "peer_buffer_gb = (0.3 if nvlink else 0.2) if device_count > 1 else 0",
                    "peer_buffer_gb = (0.3 if nvlink else 0.3) if device_count > 1 else 0", 1)],
    "B11 both: 0.3 GB per extra device over Infinity Fabric": [
        (INDEX_HTML, "const peerBufferGB = deviceCount > 1 ? (hasNVLink ? 0.3 : 0.2) : 0;",
                     "const peerBufferGB = deviceCount > 1 ? (hasNVLink || state.gpuForm === 'oam' ? 0.3 : 0.2) : 0;", 1),
        (REPORT_PY, "peer_buffer_gb = (0.3 if nvlink else 0.2) if device_count > 1 else 0",
                    'peer_buffer_gb = (0.3 if nvlink or gpu.get("form") == "oam" else 0.2) if device_count > 1 else 0', 1)],
    "B12 data: a catalog note carries a price without a dollar sign": [
        (GPUS_JSON, "\"reason\": \"The one hourly rate found, Runcrate's, could not be confirmed; it is shown below as a lead.\"",
                    "\"reason\": \"Runcrate lists 0.82 an hour, which could not be confirmed; it is shown below as a lead.\"", 1)],

    # ---- 5. shapes the catalog refuses ----
    "C1 both: a note shows beside a hand record": [
        (INDEX_HTML, "  if ((state.priceSource && state.priceSource[tier]) || (state.priceRecord && state.priceRecord[tier])) return '';",
                     "  if (state.priceSource && state.priceSource[tier]) return '';", 1),
        (REPORT_PY, '    if (gpu.get("priceSource") or {}).get(tier) or (gpu.get("priceRecord") or {}).get(tier):\n        return ""',
                    '    if (gpu.get("priceSource") or {}).get(tier):\n        return ""', 1)],
    "C2 py: the PDF lists a lead on a priced tier": [
        (REPORT_PY, "    if gpu.get(tier) is not None:\n        return []\n", "", 1)],
    "C3 js: the cost table drops the hyperscaler tier's leads": [
        (INDEX_HTML, "${priceLeadLines(state, 'hyper')}", "", 1)],
    "C3 py: the PDF drops the hyperscaler tier's leads": [
        (REPORT_PY, '        for name, tier in (("Hyperscaler", "hyper"), ("Specialized", "spec"), ("Spot", "spot")):\n'
                    '            for lead in price_leads(gpu, tier):',
                    '        for name, tier in (("Specialized", "spec"), ("Spot", "spot")):\n'
                    '            for lead in price_leads(gpu, tier):', 1)],
    "C4 both: FP8 allowed on an AMD target the ROCm table doesn't know": [
        (INDEX_HTML, "if (!gpu || gpu.vendor !== 'amd' || (ROCM.arch[gpu.gfx] || {}).fp8Weights) return '';",
                     "if (!gpu || gpu.vendor !== 'amd' || (ROCM.arch[gpu.gfx] || {}).fp8Weights || !Object.hasOwn(ROCM.arch, gpu.gfx || '')) return '';", 1),
        (REPORT_PY, 'if gpu.get("vendor") != "amd" or ROCM["arch"].get(gpu.get("gfx"), {}).get("fp8Weights"):',
                    'if gpu.get("vendor") != "amd" or ROCM["arch"].get(gpu.get("gfx"), {}).get("fp8Weights") '
                    'or gpu.get("gfx") not in ROCM["arch"]:', 1)],
}

if __name__ == "__main__":
    run_driver(S)
