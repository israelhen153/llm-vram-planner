#!/usr/bin/env python3
"""Round 37: the cold check of the FP8 default and the pre-quantized stand-in at 953ecd9.

The requirement: the default plan's printed command loads. The default weight precision
is FP8 wherever vLLM v0.30.0 loads FP8 weights and BF16 on the cards where it cannot (the
MI210, MI250X and RX 7900 XTX, by fp8WeightsBlocked / fp8_weights_blocked), in all three
places a default is chosen: the page's control, the CLI's --prec and the interactive menu.
An explicit choice always wins, and FP8 on a blocked card is still refused. On a preset's
own repo, AWQ, GPTQ and every GGUF level name a stand-in path under one line saying a
pre-quantized checkpoint goes there, GGUF with the base repo as --tokenizer, on every
surface and in both engines; a path the user gave is never replaced; the stand-in is
absolute with no space or <>, and the line survives being copied out of the PDF.

Every sabotage below brings back one bug that requirement forbids, in the place it would
appear, and the ones a cross-engine parity test could hide (a one-sided change) are made in
both engines at once.

Run against 953ecd9: 41 caught, 1 survived, none caught by the golden comparisons alone.

Survived:
  K44 js: a link's precision is not restored. The share-and-reload test in
  tests/model.test.js resolves the token with its own copy of what loadURLHash() does,
  and the "checked, not assumed" test looks for the label's text inside loadURLHash(),
  which an `if (false)` round the branch leaves in place. No test runs loadURLHash() on
  a precision parameter, so a link's choice losing to the default goes unseen.

Caught: every other sabotage, each by at least one test that is not a golden comparison.
Nine hang on a single test each: K4 and K8 on the page's default-precision and
recalculate() tests, K16 and K19 to K22 on the CLI-default and menu tests, K36 and K40
on the never-replaced test.

Not a sabotage, found by reading the PDF back with pdftotext: on an AMD card the stand-in
is also the `-v path:path` mount line, and for mistral-sm4-24b, the longest preset repo,
that line is 108 to 110 characters and the PDF wraps it; 104-character lines fit. Pasted
into a shell, the copied command runs docker without its image, then the wrapped tail as
a command of its own. No test reads the rendered PDF: story_strings() reads the
paragraphs before layout, so a wrapped line is invisible to the suite.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver

INDEX, REPORT_PY = "index.html", "generate_report.py"

# ---- index.html: the control, its gate, the builder and the stand-in ----
JS_OPT_FP8 = '        <option value="1" selected data-q="fp8">FP8 (1.0 B/param)</option>\n'
JS_OPT_AWQ = '        <option value="0.5" data-q="awq">AWQ 4-bit (0.50 B/param)</option>\n'
JS_SYNC_FALLBACK = ("  if (blocked && fp8.selected) {\n    precisionForcedFromFp8 = true;\n    bf16.selected = true;\n"
                    "  } else if (!blocked && precisionForcedFromFp8) {\n    if (bf16.selected) fp8.selected = true;\n"
                    "    precisionForcedFromFp8 = false;\n  }\n")
JS_CHOOSE = "function choosePrecision() {\n  precisionForcedFromFp8 = false;\n  recalculate();\n}\n"
JS_BLOCKED = "  const blocked = !!fp8WeightsBlocked(gpu);\n  fp8.disabled = blocked;\n"
JS_BF16_OPT = "  const bf16 = [...sel.options].find(o => o.dataset.q === '');\n"
JS_GATE = "  if (!gpu || gpu.vendor !== 'amd' || (ROCM.arch[gpu.gfx] || {}).fp8Weights) return '';\n"
JS_STANDIN = "  const standIn = fromPreset ? prequantizedStandIn(state.quantMethod, baseRepo) : null;\n"
JS_NOTE_FIRST = "  if (standIn) cmd = `${standIn.note}\\n` + cmd;\n"
JS_TOKENIZER = "  if (standIn && state.quantMethod === 'gguf') cmd += `    --tokenizer ${baseRepo} \\\\\\n`;\n"
JS_PATH = "  const path = quantMethod === 'gguf' ? `/opt/models/${name}.gguf` : `/opt/models/${name}-${method}`;\n"
JS_PREQ = "const PREQUANTIZED = { awq: 'AWQ', gptq: 'GPTQ', gguf: 'GGUF' };\n"
JS_RENDER = "  const cmd = buildVllmCommand(state, computed, modelPath, !!preset);\n"
JS_NOTE = ("  return { path, note: `# ${method} needs a pre-quantized checkpoint, which ${baseRepo} is not: "
           "replace ${path} with ${forms}.` };\n")
JS_ARCH_GFX90A = '    "gfx90a": {\n      "fp8Weights": false,\n'
JS_BUILDER_REFUSAL = ("  if ((state.quantMethod === 'fp8' || state.bytesPerParam === 1) && fp8WeightsBlocked({ vendor: state.vendor, gfx: state.gfx, name: state.gpuName }))\n"
                      "    return `# ${fp8WeightsBlocked({ vendor: state.vendor, gfx: state.gfx, name: state.gpuName })} Choose BF16, AWQ or GPTQ.`;\n")
JS_RECALC_SYNC = "  syncPrecision();\n  renderRestoreNotice();\n"
JS_TAIL = "  cmd += `    --max-model-len ${Math.min(computed.maxContextSingleUser, state.contextLength)}`;\n  return cmd;\n"
JS_RESTORE_PREC = "    if (params.has('bpp') || params.has('pr')) {\n"

# ---- generate_report.py: --prec, the menu, the JSON reader, the builder and the PDF ----
PY_DEFAULT = '    return "bf16" if fp8_weights_blocked(gpu) else "fp8"\n'
PY_CLI_PREC = '    prec = default_precision(gpu) if args.prec is None else args.prec\n'
PY_PARSER_PREC = '    parser.add_argument("--prec", default=None, choices=list(PRECISIONS),\n'
PY_MENU_FILTER = '    if blocked:\n        prec_opts = [o for o in prec_opts if o[1] != "FP8"]\n'
PY_MENU_DEFAULT = ('    default_label = "BF16" if default_precision(gpu) == "bf16" else "FP8"\n'
                   '    default_prec = next(i for i, (_, l, _q) in enumerate(prec_opts, 1) if l == default_label)\n')
PY_MENU_FROM_PRESET = '        "model_from_preset": choice.lower() != "custom",\n'
PY_JSON_FROM_PRESET = '            "model_from_preset": "hf_model" not in raw,\n'
PY_CLI_FROM_PRESET = '            "hf_model": preset["hf"], "model_name": preset["name"], "model_from_preset": True,\n'
PY_RAW_FROM_PRESET = '    cfg["model_from_preset"] = False\n'
PY_STANDIN = '    stand_in = prequantized_stand_in(cfg.get("quant"), base_repo) if cfg.get("model_from_preset") else None\n'
PY_NOTE_FIRST = '    if stand_in:\n        parts.insert(0, stand_in[1])\n'
PY_TOKENIZER = '    if stand_in and cfg.get("quant") == "gguf":\n        parts.append(f"    --tokenizer {shlex.quote(base_repo)} \\\\")\n'
PY_PATH = '    path = f"/opt/models/{name}.gguf" if quant == "gguf" else f"/opt/models/{name}-{method}"\n'
PY_PREQ = 'PREQUANTIZED = {"awq": "AWQ", "gptq": "GPTQ", "gguf": "GGUF"}\n'
PY_PDF_LINES = '        for line in cmd.split("\\n"):\n            story.append(Paragraph(line, self.styles["CmdCode"]))\n'
PY_GATE = ('    if gpu.get("vendor") != "amd" or ROCM["arch"].get(gpu.get("gfx"), {}).get("fp8Weights"):\n'
           '        return ""\n')
PY_ARCH_GFX90A = " 'arch': {'gfx90a': {'fp8Weights': False, 'aiter': False, 'fp8KvUnverified': False},\n"
PY_NOTE = ('    return path, f"# {method} needs a pre-quantized checkpoint, which {base_repo} is not: '
           'replace {path} with {forms}."\n')
PY_BUILDER_REFUSAL = ('    if cfg.get("quant") == "fp8" or cfg.get("bpp") == 1:\n'
                      '        reason = fp8_weights_blocked(cfg.get("gpu"))\n'
                      '        if reason:\n'
                      '            return f"# {reason} Choose BF16, AWQ or GPTQ."\n')
PY_TAIL = '    parts.append(f"    --max-model-len {min(comp[\'max_ctx_1\'], cfg[\'ctx\'])}")\n'

S = {
    # ---- the page's control: where its default comes from ----
    "K1 js: the markup starts on AWQ, the default that did not load": [
        (INDEX, JS_OPT_FP8, JS_OPT_FP8.replace(" selected", ""), 1),
        (INDEX, JS_OPT_AWQ, JS_OPT_AWQ.replace('"0.5" data-q', '"0.5" selected data-q'), 1)],
    "K2 js: a blocked card disables FP8 but leaves it selected, so the box prints the refusal": [
        (INDEX, JS_SYNC_FALLBACK, JS_SYNC_FALLBACK.replace("    bf16.selected = true;\n", ""), 1)],
    "K3 js: the fallback to BF16 is never given back on a card that runs FP8": [
        (INDEX, JS_SYNC_FALLBACK,
         "  if (blocked && fp8.selected) {\n    precisionForcedFromFp8 = true;\n    bf16.selected = true;\n  }\n", 1)],
    "K4 js: a reader's own pick does not end the fallback, so their BF16 comes back as FP8": [
        (INDEX, JS_CHOOSE, "function choosePrecision() {\n  recalculate();\n}\n", 1)],
    "K5 js: the control withholds FP8 on every AMD card, the MI300X included": [
        (INDEX, JS_BLOCKED, "  const blocked = !!gpu && gpu.vendor === 'amd';\n  fp8.disabled = blocked;\n", 1)],
    "K6 js: the control withholds FP8 nowhere, so the MI210 loads on FP8": [
        (INDEX, JS_BLOCKED, "  const blocked = false;\n  fp8.disabled = blocked;\n", 1)],
    "K7 js: the fallback on a blocked card is AWQ, not BF16": [
        (INDEX, JS_BF16_OPT, JS_BF16_OPT.replace("=== ''", "=== 'awq'"), 1)],
    "K8 js: recalculate() no longer syncs the control, so the gate has no call site": [
        (INDEX, JS_RECALC_SYNC, "  renderRestoreNotice();\n", 1)],
    "K44 js: a link's precision is not restored, so every shared link opens on the default": [
        (INDEX, JS_RESTORE_PREC, "    if (false) {\n", 1)],
    # ---- the predicate both engines share, changed in both so parity sees nothing ----
    "K9 both: gfx90a is marked as loading FP8 weights, so the MI210 and MI250X default to FP8": [
        (INDEX, JS_ARCH_GFX90A, JS_ARCH_GFX90A.replace("false", "true"), 1),
        (REPORT_PY, PY_ARCH_GFX90A, PY_ARCH_GFX90A.replace("'fp8Weights': False", "'fp8Weights': True"), 1)],
    "K10 both: an AMD target the table does not know passes the gate": [
        (INDEX, JS_GATE, "  if (!gpu || gpu.vendor !== 'amd' || !Object.hasOwn(ROCM.arch, gpu.gfx || '') "
                         "|| (ROCM.arch[gpu.gfx] || {}).fp8Weights) return '';\n", 1),
        (REPORT_PY, PY_GATE, '    if gpu.get("vendor") != "amd" or gpu.get("gfx") not in ROCM["arch"] or '
                             'ROCM["arch"].get(gpu.get("gfx"), {}).get("fp8Weights"):\n        return ""\n', 1)],
    "K11 both: the gate follows the silicon's FP8 tensor cores, not vLLM's kernels": [
        (INDEX, JS_GATE, "  if (!gpu || (gpu.caps || {}).fp8) return '';\n", 1),
        (REPORT_PY, PY_GATE, '    if (gpu.get("caps") or {}).get("fp8"):\n        return ""\n', 1)],
    "K12 both: the command builders print FP8 on a blocked card": [
        (INDEX, JS_BUILDER_REFUSAL, "", 1),
        (REPORT_PY, PY_BUILDER_REFUSAL, "", 1)],
    # ---- the CLI's --prec ----
    "K13 py: the default is FP8 on every card, so the MI210's default plan is refused": [
        (REPORT_PY, PY_DEFAULT, '    return "fp8"\n', 1)],
    "K14 py: the default is BF16 on every card": [
        (REPORT_PY, PY_DEFAULT, '    return "bf16"\n', 1)],
    "K15 py: the default where FP8 is blocked is AWQ, the default that did not load": [
        (REPORT_PY, PY_DEFAULT, '    return "awq" if fp8_weights_blocked(gpu) else "fp8"\n', 1)],
    "K16 py: argparse fixes --prec to fp8, so from_cli_args never sees a missing one": [
        (REPORT_PY, PY_PARSER_PREC, PY_PARSER_PREC.replace("default=None", 'default="fp8"'), 1)],
    "K17 py: an explicit --prec bf16 is read as no choice and becomes FP8": [
        (REPORT_PY, PY_CLI_PREC, '    prec = default_precision(gpu) if args.prec in (None, "bf16") else args.prec\n', 1)],
    "K18 py: an explicit --prec fp8 on a blocked card is quietly planned as BF16, not refused": [
        (REPORT_PY, PY_CLI_PREC, '    prec = default_precision(gpu) if args.prec is None or '
                                 '(args.prec == "fp8" and fp8_weights_blocked(gpu)) else args.prec\n', 1)],
    # ---- the interactive menu ----
    "K19 py: the menu's default is option 1, BF16, on every card": [
        (REPORT_PY, PY_MENU_DEFAULT, "    default_prec = 1\n", 1)],
    "K20 py: the menu's default is option 2, which is INT4/AWQ once FP8 is left out": [
        (REPORT_PY, PY_MENU_DEFAULT, "    default_prec = 2\n", 1)],
    "K21 py: the menu keeps FP8 on a blocked card": [
        (REPORT_PY, PY_MENU_FILTER, "    if blocked:\n        pass\n", 1)],
    "K22 py: the menu's default follows the silicon's FP8 tensor cores, not vLLM's kernels": [
        (REPORT_PY, PY_MENU_DEFAULT,
         '    default_label = "FP8" if (gpu.get("caps") or {}).get("fp8") else "BF16"\n'
         '    default_prec = next(i for i, (_, l, _q) in enumerate(prec_opts, 1) if l == default_label)\n', 1)],
    # ---- the stand-in, in both engines at once ----
    "K23 both: a path the user gave is replaced too": [
        (INDEX, JS_STANDIN, "  const standIn = prequantizedStandIn(state.quantMethod, baseRepo);\n", 1),
        (REPORT_PY, PY_STANDIN, '    stand_in = prequantized_stand_in(cfg.get("quant"), base_repo)\n', 1)],
    "K24 both: GGUF gets no stand-in, so its command loads the base repo as a GGUF": [
        (INDEX, JS_PREQ, "const PREQUANTIZED = { awq: 'AWQ', gptq: 'GPTQ' };\n", 1),
        (REPORT_PY, PY_PREQ, 'PREQUANTIZED = {"awq": "AWQ", "gptq": "GPTQ"}\n', 1)],
    "K33 both: FP8 gets a stand-in too, though the preset's repo loads it": [
        (INDEX, JS_PREQ, "const PREQUANTIZED = { awq: 'AWQ', gptq: 'GPTQ', gguf: 'GGUF', fp8: 'FP8' };\n", 1),
        (REPORT_PY, PY_PREQ, 'PREQUANTIZED = {"awq": "AWQ", "gptq": "GPTQ", "gguf": "GGUF", "fp8": "FP8"}\n', 1)],
    "K25 both: GGUF gets no --tokenizer": [
        (INDEX, JS_TOKENIZER, "", 1),
        (REPORT_PY, PY_TOKENIZER, "", 1)],
    "K26 both: every stand-in gets --tokenizer, AWQ and GPTQ included": [
        (INDEX, JS_TOKENIZER, JS_TOKENIZER.replace(" && state.quantMethod === 'gguf'", ""), 1),
        (REPORT_PY, PY_TOKENIZER, PY_TOKENIZER.replace(' and cfg.get("quant") == "gguf"', ""), 1)],
    "K42 both: --tokenizer names the stand-in, not the base repo": [
        (INDEX, JS_TOKENIZER, JS_TOKENIZER.replace("${baseRepo}", "${modelPath}"), 1),
        (REPORT_PY, PY_TOKENIZER, PY_TOKENIZER.replace("{shlex.quote(base_repo)}", "{hf}"), 1)],
    "K27 both: the stand-in path is relative": [
        (INDEX, JS_PATH, JS_PATH.replace("`/opt/", "`opt/"), 1),
        (REPORT_PY, PY_PATH, PY_PATH.replace('f"/opt/', 'f"opt/'), 1)],
    "K28 both: the stand-in path holds a space": [
        (INDEX, JS_PATH, JS_PATH.replace("${name}-${method}", "${name} ${method}"), 1),
        (REPORT_PY, PY_PATH, PY_PATH.replace("{name}-{method}", "{name} {method}"), 1)],
    "K29 both: the stand-in's line is dropped, so the path stands unexplained": [
        (INDEX, JS_NOTE_FIRST, "", 1),
        (REPORT_PY, PY_NOTE_FIRST, "", 1)],
    "K30 both: the stand-in's line comes after the command instead of before it": [
        (INDEX, JS_NOTE_FIRST, "", 1),
        (INDEX, JS_TAIL, JS_TAIL.replace("  return cmd;\n", "  if (standIn) cmd += `\\n${standIn.note}`;\n  return cmd;\n"), 1),
        (REPORT_PY, PY_NOTE_FIRST, "", 1),
        (REPORT_PY, PY_TAIL, PY_TAIL + "    if stand_in:\n        parts.append(stand_in[1])\n", 1)],
    "K31 both: an apostrophe in the line, which a shell reads once the PDF wraps it": [
        (INDEX, JS_NOTE, JS_NOTE.replace("is not: replace", "isn't: replace"), 1),
        (REPORT_PY, PY_NOTE, PY_NOTE.replace("is not: replace", "isn't: replace"), 1)],
    "K32 both: angle brackets round the path in the line, which the PDF reads as markup": [
        (INDEX, JS_NOTE, JS_NOTE.replace("replace ${path} with", "replace <${path}> with"), 1),
        (REPORT_PY, PY_NOTE, PY_NOTE.replace("replace {path} with", "replace <{path}> with"), 1)],
    # ---- one engine or one surface: parity and the surface tests should see these ----
    "K34 py: the PDF prints the command without its first line, the stand-in's": [
        (REPORT_PY, PY_PDF_LINES, PY_PDF_LINES.replace('split("\\n"):', 'split("\\n")[1:]:'), 1)],
    "K35 py: a JSON config's own hf_model is replaced on the preset branch": [
        (REPORT_PY, PY_JSON_FROM_PRESET, '            "model_from_preset": True,\n', 1)],
    "K40 py: a JSON config with no preset that claims model_from_preset keeps the claim": [
        (REPORT_PY, PY_RAW_FROM_PRESET, "", 1)],
    "K36 py: the menu's own typed model id is replaced": [
        (REPORT_PY, PY_MENU_FROM_PRESET, '        "model_from_preset": True,\n', 1)],
    "K37 py: the CLI never names a stand-in on a preset": [
        (REPORT_PY, PY_CLI_FROM_PRESET, PY_CLI_FROM_PRESET.replace('"model_from_preset": True', '"model_from_preset": False'), 1)],
    "K38 js: the page never names a stand-in on a preset": [
        (INDEX, JS_RENDER, JS_RENDER.replace("!!preset)", "false)"), 1)],
    "K39 js: the page replaces an imported model id too": [
        (INDEX, JS_RENDER, JS_RENDER.replace("!!preset)", "true)"), 1)],
}

if __name__ == "__main__":
    run_driver(S)
