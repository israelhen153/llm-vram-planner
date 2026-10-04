#!/usr/bin/env python3
"""The second cold check of `fix/fp8-default`, at `cab6341`, against the requirement alone
(the default plan's command loads: FP8 by default where vLLM v0.30.0 loads it and BF16 on
the three cards it cannot; a stand-in path and its line for AWQ, GPTQ and GGUF on a preset's
own repo; every PDF command pasting as itself).

Found by probing, not by sabotage — real PDFs, `pdftotext` in both modes, the block pasted
into bash with `docker` and `vllm` stubbed:

  * a tab or a no-break space inside a JSON config's `hf_model` pastes out of the PDF as a
    plain space, in both pdftotext modes: reportlab's Paragraph prints either as a space;
  * a path with characters outside Courier's WinAnsi set (CJK, an emoji) prints as ■ and
    pastes as ■; Latin-1 (`résumé-ça`) and an em dash paste correctly;
  * under `pdftotext -layout`, every in-word break (a `\\` continuation, the next line
    starting with the word's next character) is split by the one-space indent the layout
    mode gives every line of the command box; a break between words (` \\`) is immune.
    The default mode pastes every swept command correctly;
  * a JSON config `{"quant": "gguf", "bpp": 1}` on an FP8-blocked card is refused with
    the FP8 reason (`bpp == 1` stands for FP8 in both gates), while the same config on an
    H100 plans GGUF at one byte.

Run at cab6341 (33da351 carries the driver): 40 caught, 1 survived.

SURVIVED (a test gap):

  * E7  the escape dropped from command_paragraphs(), so reportlab's Paragraph reads the
        command as markup. The paste test's odd paths carry `&` followed by a space, which
        the parser tolerates, and no `<`, `>` or `&amp;`: a path such as
        `/opt/models/<org>/m` is what the escape exists for, and no test supplies one.

Caught (the other forty): the page's markup starting on AWQ or BF16; the fallback dropped,
landing on AWQ, never given back, forgotten by choosePrecision(), or recalculate() without
syncPrecision(); a link's refused FP8 left out of the notice, given no reason, or never
restored; the command box printing FP8 on a blocked card (page only, and both builders);
--prec's default FP8 or BF16 everywhere, AWQ where blocked, fixed by argparse, an explicit
fp8 quietly planned as the default; the menu's default pinned to option 1 or 2, FP8 kept on
a blocked card; the stand-in in both engines at once — GGUF out of the set, a user's path
replaced, --tokenizer dropped, the path relative or holding a space, the line after the
command or respelled; the PDF skipping comment lines; the menu, the JSON path and the page
mis-flagging the preset's repo; and the PDF's breaks — no wrapping, a run of spaces in
quotes left to collapse (reportlab's Paragraph cleans its text as it is built, so the paste
test does see this), a single-quoted part broken open, a comment's tail without #, a
between-words break without ` \\`, the column measured without the frame's padding, an
in-word break without `\\`. Every catch's first failing test is a real one, none a golden.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver

JS, PY = "index.html", "generate_report.py"

# ---- the page's markup, control and notice ---------------------------------------------
JS_FP8_OPT = '        <option value="1" selected data-q="fp8">FP8 (1.0 B/param)</option>\n'
JS_AWQ_OPT = '        <option value="0.5" data-q="awq">AWQ 4-bit (0.50 B/param)</option>\n'
JS_BF16_OPT = '        <option value="2" data-q="">BF16 / FP16 (2.0 B/param)</option>\n'
JS_FALLBACK = "  if (blocked && fp8.selected) {\n    precisionForcedFromFp8 = true;\n    bf16.selected = true;\n"
JS_GIVE_BACK = ("  } else if (!blocked && precisionForcedFromFp8) {\n    if (bf16.selected) fp8.selected = true;\n"
                "    precisionForcedFromFp8 = false;\n  }\n")
JS_CHOOSE = "function choosePrecision() {\n  precisionForcedFromFp8 = false;\n  recalculate();\n}\n"
JS_RECALC = "  syncInterconnect();\n  syncPrecision();\n  renderRestoreNotice();\n"
JS_LINK_PICK = "      if (match) match.selected = true;\n"
JS_LINK_REFUSED = "      if (refused && bf16) urlRestoreLost.push({ id: 'weight-precision', label: 'the weight precision', raw: val,\n"
JS_LINK_WHY = "                                                 fallback: bf16.value, why: refused, shown: 'BF16' });\n"
JS_BOX_GATE = ("  if ((state.quantMethod === 'fp8' || (state.bytesPerParam === 1 && !state.quantMethod)) && fp8WeightsBlocked({ vendor: state.vendor, "
               "gfx: state.gfx, name: state.gpuName }))\n    return `# ${fp8WeightsBlocked(")
PY_BOX_GATE = ('    if asks_for_fp8_weights(cfg):\n        reason = fp8_weights_blocked(cfg.get("gpu"))\n'
               '        if reason:\n            return f"# {reason} Choose BF16, AWQ or GPTQ."\n')

# ---- the CLI and the menu ----------------------------------------------------------------
PY_DEFAULT = '    return "bf16" if fp8_weights_blocked(gpu) else "fp8"\n'
PY_RESOLVE = "    prec = default_precision(gpu) if args.prec is None else args.prec\n"
PY_ARGPARSE = '    parser.add_argument("--prec", default=None, choices=list(PRECISIONS),\n'
PY_MENU_FILTER = '        prec_opts = [o for o in prec_opts if o[1] != "FP8"]\n'
PY_MENU_DEFAULT = "    default_prec = next(i for i, (_, l, _q) in enumerate(prec_opts, 1) if l == default_label)\n"

# ---- the stand-in, both engines ----------------------------------------------------------
JS_SET = "const PREQUANTIZED = { awq: 'AWQ', gptq: 'GPTQ', gguf: 'GGUF' };\n"
PY_SET = 'PREQUANTIZED = {"awq": "AWQ", "gptq": "GPTQ", "gguf": "GGUF"}\n'
JS_FROM_PRESET = "  const standIn = fromPreset ? prequantizedStandIn(state.quantMethod, baseRepo) : null;\n"
PY_FROM_PRESET = '    stand_in = prequantized_stand_in(cfg.get("quant"), base_repo) if cfg.get("model_from_preset") else None\n'
JS_TOKENIZER = r"  if (standIn && state.quantMethod === 'gguf') cmd += `    --tokenizer ${baseRepo} \\\n`;" + "\n"
PY_TOKENIZER = ('    if stand_in and cfg.get("quant") == "gguf":\n'
                r'        parts.append(f"    --tokenizer {shlex.quote(base_repo)} \\")' + "\n")
JS_PATH = "  const path = quantMethod === 'gguf' ? `/opt/models/${name}.gguf` : `/opt/models/${name}-${method}`;\n"
PY_PATH = '    path = f"/opt/models/{name}.gguf" if quant == "gguf" else f"/opt/models/{name}-{method}"\n'
JS_NOTE_FIRST = "  if (standIn) cmd = `${standIn.note}\\n` + cmd;\n"
PY_NOTE_FIRST = "    if stand_in:\n        parts.insert(0, stand_in[1])\n"
PY_RETURN = "    parts.append(f\"    --max-model-len {min(comp['max_ctx_1'], cfg['ctx'])}\")\n    return \"\\n\".join(parts)\n"
JS_NOTE_TEXT = "needs a pre-quantized checkpoint, which ${baseRepo} is not"
PY_NOTE_TEXT = "needs a pre-quantized checkpoint, which {base_repo} is not"
PY_MENU_FLAG = '        "model_from_preset": choice.lower() != "custom",\n'
PY_JSON_FLAG = '            "model_from_preset": "hf_model" not in raw,\n'
PY_RAW_FLAG = '    cfg["model_from_preset"] = False\n'
JS_RENDER = "  const cmd = buildVllmCommand(state, computed, modelPath, !!preset);\n"

# ---- the PDF's breaks --------------------------------------------------------------------
PY_PARAS = "        return [Paragraph(escape(line), style) for line in pdf_command_lines(cmd, fits)]\n"
PY_WIDTH = "        return Frame(0, 0, A4[0] - 2 * self.margin, A4[1])._aW\n"
PY_RUN = '            if ch == " " and prev == " ":\n                atoms.append((quote + quote, quote))\n'
PY_COMMENT_TAIL = '        else:\n            out.append(cur)\n            cur = f"# {word}"\n'
PY_BETWEEN = '        if cur.strip() and fits(f"{word} \\\\"):\n            out.append(f"{cur} \\\\")\n            cur = word\n'
PY_OK = r'''                  if fits(cur + "".join(t for t, _ in rest[:k + 1]) + ("'\\" if rest[k][1] == "'" else "\\"))]''' + "\n"
PY_BREAK = r'''            out.append(cur + "".join(t for t, _ in rest[:k + 1]) + ("'\\" if inside == "'" else "\\"))''' + "\n"
PY_REOPEN = '''            cur, rest = ("'" if inside == "'" else ""), rest[k + 1:]\n'''

S = {}
# ---- (1) the page ----
S["A1 page: the markup starts on AWQ"] = [
    (JS, JS_FP8_OPT + JS_AWQ_OPT, JS_FP8_OPT.replace(" selected", "") + JS_AWQ_OPT.replace('"0.5" data-q="awq"', '"0.5" selected data-q="awq"'), 1)]
S["A2 page: the markup starts on BF16"] = [
    (JS, JS_BF16_OPT + JS_FP8_OPT, JS_BF16_OPT.replace('"2" data-q=""', '"2" selected data-q=""') + JS_FP8_OPT.replace(" selected", ""), 1)]
S["A3 page: FP8 stays selected, disabled, on a blocked card (the fallback dropped)"] = [
    (JS, JS_FALLBACK, JS_FALLBACK.replace("if (blocked && fp8.selected) {", "if (false && blocked && fp8.selected) {"), 1)]
S["A4 page: the fallback lands on AWQ, not BF16"] = [
    (JS, JS_FALLBACK, JS_FALLBACK.replace("    bf16.selected = true;\n", "    [...sel.options].find(o => o.dataset.q === 'awq').selected = true;\n"), 1)]
S["A5 page: a fallback is never given back on a card that runs FP8"] = [
    (JS, JS_GIVE_BACK, "  }\n", 1)]
S["A6 page: a reader's pick after the fallback is forgotten (choosePrecision keeps the flag)"] = [
    (JS, JS_CHOOSE, "function choosePrecision() {\n  recalculate();\n}\n", 1)]
S["A7 page: recalculate() no longer calls syncPrecision()"] = [
    (JS, JS_RECALC, "  syncInterconnect();\n  renderRestoreNotice();\n", 1)]
S["A8 page: a link's FP8 on a blocked card left out of the restore notice"] = [
    (JS, JS_LINK_REFUSED, JS_LINK_REFUSED.replace("if (refused && bf16)", "if (false && refused && bf16)"), 1)]
S["A9 page: the notice calls a link's refused FP8 'not in this version of the tool' (no why)"] = [
    (JS, JS_LINK_WHY, "                                                 fallback: bf16.value });\n", 1)]
S["A10 page: a link's FP8 is never restored, on any card"] = [
    (JS, JS_LINK_PICK, "      if (match && match.dataset.q !== 'fp8') match.selected = true;\n", 1)]
S["A11 page: the command box prints FP8 on a blocked card (gate dropped, page only)"] = [
    (JS, JS_BOX_GATE, JS_BOX_GATE.replace("if ((state.quantMethod", "if (false && (state.quantMethod"), 1)]
S["A12 both: the command printed on a blocked card by both builders"] = [
    (JS, JS_BOX_GATE, JS_BOX_GATE.replace("if ((state.quantMethod", "if (false && (state.quantMethod"), 1),
    (PY, PY_BOX_GATE, PY_BOX_GATE.replace('    if asks_for_fp8_weights(cfg):\n', "    if False:\n"), 1)]
# ---- (1) the CLI ----
S["B1 cli: the default is FP8 on every card"] = [(PY, PY_DEFAULT, '    return "fp8"\n', 1)]
S["B2 cli: the default is BF16 on every card"] = [(PY, PY_DEFAULT, '    return "bf16"\n', 1)]
S["B3 cli: the default is AWQ where FP8 is blocked"] = [(PY, PY_DEFAULT, '    return "awq" if fp8_weights_blocked(gpu) else "fp8"\n', 1)]
S["B4 cli: argparse fixes --prec to fp8"] = [
    (PY, PY_ARGPARSE, '    parser.add_argument("--prec", default="fp8", choices=list(PRECISIONS),\n', 1)]
S["B5 cli: an explicit --prec fp8 on a blocked card quietly planned as the default"] = [
    (PY, PY_RESOLVE, '    prec = default_precision(gpu) if args.prec in (None, "fp8") else args.prec\n', 1)]
# ---- (1) the menu ----
S["C1 menu: the default pinned to option 1"] = [(PY, PY_MENU_DEFAULT, "    default_prec = 1\n", 1)]
S["C2 menu: the default pinned to option 2"] = [(PY, PY_MENU_DEFAULT, "    default_prec = 2\n", 1)]
S["C3 menu: FP8 offered on a blocked card"] = [(PY, PY_MENU_FILTER, "", 1)]
# ---- (2) the stand-in, in both engines at once so parity sees nothing ----
S["D1 both: GGUF out of the stand-in set"] = [
    (JS, JS_SET, "const PREQUANTIZED = { awq: 'AWQ', gptq: 'GPTQ' };\n", 1),
    (PY, PY_SET, 'PREQUANTIZED = {"awq": "AWQ", "gptq": "GPTQ"}\n', 1)]
S["D2 both: a path the user gave is replaced too"] = [
    (JS, JS_FROM_PRESET, "  const standIn = prequantizedStandIn(state.quantMethod, baseRepo);\n", 1),
    (PY, PY_FROM_PRESET, '    stand_in = prequantized_stand_in(cfg.get("quant"), base_repo)\n', 1)]
S["D3 both: --tokenizer dropped from the GGUF command"] = [
    (JS, JS_TOKENIZER, "", 1), (PY, PY_TOKENIZER, "", 1)]
S["D4 both: the stand-in path is relative"] = [
    (JS, JS_PATH, JS_PATH.replace("`/opt/models/", "`models/"), 1),
    (PY, PY_PATH, PY_PATH.replace('f"/opt/models/', 'f"models/'), 1)]
S["D5 both: the stand-in's line after the command"] = [
    (JS, JS_NOTE_FIRST, "  if (standIn) cmd = cmd + `\\n${standIn.note}`;\n", 1),
    (PY, PY_NOTE_FIRST, "", 1),
    (PY, PY_RETURN, PY_RETURN.replace('    return "\\n".join(parts)\n', '    return "\\n".join(parts + ([stand_in[1]] if stand_in else []))\n'), 1)]
S["D6 both: the line respelled identically (prequantized)"] = [
    (JS, JS_NOTE_TEXT, JS_NOTE_TEXT.replace("pre-quantized", "prequantized"), 1),
    (PY, PY_NOTE_TEXT, PY_NOTE_TEXT.replace("pre-quantized", "prequantized"), 1)]
S["D7 py: the PDF skips the command's comment lines"] = [
    (PY, PY_PARAS, '        return [Paragraph(escape(line), style) for line in pdf_command_lines(cmd, fits) if not line.startswith("#")]\n', 1)]
S["D8 py: the menu never marks a preset's repo as the preset's"] = [
    (PY, PY_MENU_FLAG, '        "model_from_preset": False,\n', 1)]
S["D9 py: a JSON config's own hf_model is replaced (flag always on)"] = [
    (PY, PY_JSON_FLAG, '            "model_from_preset": True,\n', 1)]
S["D10 py: a no-preset JSON config may claim the preset's repo"] = [
    (PY, PY_RAW_FLAG, '    cfg.setdefault("model_from_preset", False)\n', 1)]
S["D11 page: renderCommand never says the path is a preset's"] = [
    (JS, JS_RENDER, "  const cmd = buildVllmCommand(state, computed, modelPath, false);\n", 1)]
S["D12 page: renderCommand says every path is a preset's"] = [
    (JS, JS_RENDER, "  const cmd = buildVllmCommand(state, computed, modelPath, true);\n", 1)]
S["D13 both: a stand-in path with a space in it"] = [
    (JS, JS_PATH, JS_PATH.replace("${name}-${method}", "${name} ${method}"), 1),
    (PY, PY_PATH, PY_PATH.replace("{name}-{method}", "{name} {method}"), 1)]
# ---- (3) the PDF's breaks ----
S["E1 pdf: the command's own lines handed to reportlab (no wrapping)"] = [
    (PY, PY_PARAS, '        return [Paragraph(escape(line), style) for line in cmd.split("\\n")]\n', 1)]
S["E2 pdf: a run of spaces in quotes left for the PDF to collapse"] = [(PY, PY_RUN, "", 1)]
S["E3 pdf: a single-quoted part broken without closing it"] = [
    (PY, PY_OK, PY_OK.replace('''("'\\\\" if rest[k][1] == "'" else "\\\\")''', '"\\\\"'), 1),
    (PY, PY_BREAK, PY_BREAK.replace('''("'\\\\" if inside == "'" else "\\\\")''', '"\\\\"'), 1),
    (PY, PY_REOPEN, '            cur, rest = "", rest[k + 1:]\n', 1)]
S["E4 pdf: a wrapped comment's tail without #"] = [
    (PY, PY_COMMENT_TAIL, PY_COMMENT_TAIL.replace('cur = f"# {word}"', 'cur = f"  {word}"'), 1)]
S["E5 pdf: a break between words without ' \\'"] = [
    (PY, PY_BETWEEN, PY_BETWEEN.replace('out.append(f"{cur} \\\\")', "out.append(cur)"), 1)]
S["E6 pdf: the column measured without the frame's padding"] = [
    (PY, PY_WIDTH, "        return A4[0] - 2 * self.margin\n", 1)]
S["E7 pdf: the escape dropped"] = [
    (PY, PY_PARAS, "        return [Paragraph(line, style) for line in pdf_command_lines(cmd, fits)]\n", 1)]
S["E8 pdf: an in-word break without its backslash"] = [
    (PY, PY_BREAK, PY_BREAK.replace('''("'\\\\" if inside == "'" else "\\\\")''', '''("'" if inside == "'" else "")'''), 1)]

if __name__ == "__main__":
    run_driver(S)
