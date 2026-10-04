#!/usr/bin/env python3
"""Round 27: fix/fp8-default. Sabotages against the branch that made the default
precision FP8, or BF16 on a card vLLM v0.30.0 has no FP8 weight kernel for, in the
page, in --prec and in the interactive menu, and that made a command on a preset's
own repo name a stand-in path for AWQ, GPTQ and GGUF, under one line saying a
pre-quantized checkpoint goes there, with the base repo as --tokenizer for GGUF.

Each edit puts back one way the default plan's command could fail to load again, or
the stand-in could go wrong: AWQ the default in one of the three places, FP8 the
default on a card that can't run it, the stand-in on one surface or in one engine
only, a path the user gave replaced, the line missing for one method, and the
tokenizer missing from GGUF. The ones marked "both" edit the two engines alike, so
the parity check that holds them equal cannot be what catches them.

Usage: python3 tests/sabotage/engine_r27_fp8_default.py [name-substring ...]
       python3 tests/sabotage/engine_r27_fp8_default.py --from S1
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import (INDEX_HTML, REPORT_PY, JS_R10_SYNC_FORCE, PY_R10_MENU_FILTER,
                     JS_R27_DEFAULT_OPTIONS, JS_R27_RENDER_CALL, JS_R27_STAND_IN, JS_R27_NOTE,
                     JS_R27_TOKENIZER, JS_R27_PATH, JS_R27_REPORT_CMD, PY_R27_PREC_ARG,
                     PY_R27_MENU_DEFAULT, PY_R27_DEFAULT_RULE, PY_R27_CLI_FLAG, PY_R27_JSON_FLAG,
                     PY_R27_JSON_OWN, PY_R27_MENU_FLAG, PY_R27_STAND_IN, PY_R27_NOTE,
                     PY_R27_TOKENIZER, PY_R27_PATH, PY_R27_PDF_LINES, PY_R27_PDF_SPLIT,
                     PY_R27_COMMENT_TAIL, PY_R27_WORD_BREAK, PY_R27_WORD_REOPEN, PY_R27_FRAME_WIDTH,
                     PY_R27_QUOTED_RUN, PY_R27_BETWEEN_WORDS)

# The command box, the one place the page writes the command. Used once, so kept here.
JS_BOX = "<code>${cmd}</code></div>${rocmLines}`);\n"

S = {}

# ---- D: AWQ the default again, in each of the three places ----
S["D1 js: the markup selects AWQ again"] = [
    (INDEX_HTML, JS_R27_DEFAULT_OPTIONS,
     JS_R27_DEFAULT_OPTIONS.replace(' selected data-q="fp8"', ' data-q="fp8"')
                           .replace(' data-q="awq"', ' selected data-q="awq"'), 1)]
S["D2 py: --prec defaults to awq again"] = [
    (REPORT_PY, PY_R27_PREC_ARG, PY_R27_PREC_ARG.replace("default=None", 'default="awq"'), 1)]
S["D3 py: the interactive menu defaults to INT4/AWQ again"] = [
    (REPORT_PY, PY_R27_MENU_DEFAULT, '    default_label = "INT4/AWQ"\n', 1)]

# ---- B: FP8 the default on a card vLLM has no FP8 weight kernel for ----
S["B1 js: the page moves a chosen FP8 off a blocked card, but not the default"] = [
    (INDEX_HTML, JS_R10_SYNC_FORCE, "  if (blocked && fp8.selected && !fp8.defaultSelected) {\n", 1)]
S["B2 py: --prec defaults to fp8 on every card"] = [
    (REPORT_PY, PY_R27_DEFAULT_RULE, '    return "fp8"\n', 1)]
S["B3 py: the menu keeps FP8 on a blocked card, as its default"] = [
    (REPORT_PY, PY_R10_MENU_FILTER, "        pass\n", 1),
    (REPORT_PY, PY_R27_MENU_DEFAULT, '    default_label = "FP8"\n', 1)]

# ---- S: the stand-in on one surface, or in one engine, only ----
S["S1 js only: the page names the preset's repo for AWQ, GPTQ and GGUF again"] = [
    (INDEX_HTML, JS_R27_RENDER_CALL, JS_R27_RENDER_CALL.replace(", !!preset", ""), 1)]
S["S2 py only: from_cli_args() does not mark the preset's repo as the preset's"] = [
    (REPORT_PY, PY_R27_CLI_FLAG, PY_R27_CLI_FLAG.replace('"model_from_preset": True', '"model_from_preset": False'), 1)]
S["S3 py only: build_vllm_cmd() never names the stand-in"] = [
    (REPORT_PY, PY_R27_STAND_IN, "    stand_in = None\n", 1)]
S["S4 js: the copied report drops the stand-in's line from the command it quotes"] = [
    (INDEX_HTML, JS_R27_REPORT_CMD,
     JS_R27_REPORT_CMD.replace("emitted.textContent :", "emitted.textContent.replace(/^# [^\\n]*\\n/, '') :"), 1)]
S["S5 py: the PDF drops the stand-in's line from the command it prints"] = [
    (REPORT_PY, PY_R27_PDF_LINES,
     '        story += [p for p in self.command_paragraphs(cmd) if not p.text.startswith("#")]\n', 1)]
S["S6 js: the command box hides the stand-in's line, so the copied report loses it too"] = [
    (INDEX_HTML, JS_BOX, JS_BOX.replace("<code>${cmd}</code>", "<code>${cmd.replace(/^# [^\\n]*pre-quantized[^\\n]*\\n/, '')}</code>"), 1)]
S["S7 js: the stand-in only in the vllm serve command, never in the ROCm one"] = [
    (INDEX_HTML, JS_R27_STAND_IN, JS_R27_STAND_IN.replace("fromPreset ?", "fromPreset && state.vendor !== 'amd' ?"), 1)]

# ---- U: a path the user gave, replaced ----
S["U1 js: an imported model id is replaced like a preset's repo"] = [
    (INDEX_HTML, JS_R27_RENDER_CALL, JS_R27_RENDER_CALL.replace("!!preset", "true"), 1)]
S["U2 py: a JSON config's hf_model beside a preset is replaced"] = [
    (REPORT_PY, PY_R27_JSON_FLAG, '            "model_from_preset": True,\n', 1)]
S["U3 py: a JSON config with no preset can claim a preset's repo, and its path is replaced"] = [
    (REPORT_PY, PY_R27_JSON_OWN, '    cfg.setdefault("model_from_preset", False)\n', 1)]
S["U4 py: the menu's typed model id is replaced"] = [
    (REPORT_PY, PY_R27_MENU_FLAG, '        "model_from_preset": True,\n', 1)]
S["U5 py: build_vllm_cmd() replaces any model path, not only a preset's repo"] = [
    (REPORT_PY, PY_R27_STAND_IN, PY_R27_STAND_IN.replace('if cfg.get("model_from_preset")', 'if cfg.get("hf_model")'), 1)]

# ---- L: the line saying what goes there, missing for one method ----
for n, method in enumerate(("awq", "gptq", "gguf"), 1):
    S[f"L{n} js: no line over the stand-in for {method}"] = [
        (INDEX_HTML, JS_R27_NOTE, JS_R27_NOTE.replace("if (standIn)", f"if (standIn && state.quantMethod !== '{method}')"), 1)]
    S[f"L{n + 3} py: no line over the stand-in for {method}"] = [
        (REPORT_PY, PY_R27_NOTE, PY_R27_NOTE.replace("    if stand_in:", f'    if stand_in and cfg.get("quant") != "{method}":'), 1)]
S["L7 both: no line over the stand-in for gptq, in both engines alike"] = [
    (INDEX_HTML, JS_R27_NOTE, JS_R27_NOTE.replace("if (standIn)", "if (standIn && state.quantMethod !== 'gptq')"), 1),
    (REPORT_PY, PY_R27_NOTE, PY_R27_NOTE.replace("    if stand_in:", '    if stand_in and cfg.get("quant") != "gptq":'), 1)]

# ---- T: GGUF without the base repo's tokenizer ----
S["T1 js: no --tokenizer on the GGUF command"] = [(INDEX_HTML, JS_R27_TOKENIZER, "", 1)]
S["T2 py: no --tokenizer on the GGUF command"] = [(REPORT_PY, PY_R27_TOKENIZER, "", 1)]
S["T3 both: no --tokenizer on the GGUF command, in both engines alike"] = [
    (INDEX_HTML, JS_R27_TOKENIZER, "", 1), (REPORT_PY, PY_R27_TOKENIZER, "", 1)]

# ---- P: a stand-in the planner's own path rules, or the PDF, would refuse ----
S["P1 both: the stand-in is a relative path, in both engines alike"] = [
    (INDEX_HTML, JS_R27_PATH, JS_R27_PATH.replace("`/opt/models/", "`opt/models/"), 1),
    (REPORT_PY, PY_R27_PATH, PY_R27_PATH.replace('f"/opt/models/', 'f"opt/models/'), 1)]
S["P2 both: the stand-in carries <...>, in both engines alike"] = [
    (INDEX_HTML, JS_R27_PATH, JS_R27_PATH.replace("${name}-${method}", "<${name}>-${method}"), 1),
    (REPORT_PY, PY_R27_PATH, PY_R27_PATH.replace("{name}-{method}", "<{name}>-{method}"), 1)]

# ---- Q: a command copied out of the PDF pasting as another command (round 37's cold check) ----
S["Q1 py: the PDF hands reportlab the command's own lines, which it wraps"] = [
    (REPORT_PY, PY_R27_PDF_SPLIT, PY_R27_PDF_SPLIT.replace("pdf_command_lines(cmd, fits)", 'cmd.split("\\n")'), 1)]
S["Q2 py: a comment broken for the PDF continues without its #"] = [
    (REPORT_PY, PY_R27_COMMENT_TAIL, "            cur = word\n", 1)]
S["Q3 py: a word broken for the PDF loses its backslash"] = [
    (REPORT_PY, PY_R27_WORD_BREAK, PY_R27_WORD_BREAK.replace('("\'\\\\" if inside == "\'" else "\\\\")', '("\'" if inside == "\'" else "")'), 1)]
S["Q4 py: a single-quoted part broken for the PDF without closing and reopening it"] = [
    (REPORT_PY, PY_R27_WORD_BREAK, PY_R27_WORD_BREAK.replace('("\'\\\\" if inside == "\'" else "\\\\")', '"\\\\"'), 1),
    (REPORT_PY, PY_R27_WORD_REOPEN, '            cur, rest = "", rest[k + 1:]\n', 1)]
S["Q5 py: the command column measured without the frame's padding, so lines still wrap"] = [
    (REPORT_PY, PY_R27_FRAME_WIDTH, "        return A4[0] - 2 * self.margin\n", 1)]
S["Q6 py: a run of spaces inside quotes printed as is, which the PDF prints as one"] = [
    (REPORT_PY, PY_R27_QUOTED_RUN, "", 1)]
S["Q7 py: a line broken between words without the \\ that joins it back"] = [
    (REPORT_PY, PY_R27_BETWEEN_WORDS, "            out.append(cur)\n            cur = word\n", 1)]

run_driver(S)
