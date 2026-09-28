#!/usr/bin/env python3
"""Round 16: the cold check of fix/json-precision-width.

Round 15 brought each of the owner's rules back one at a time, on the axes its
own test sweeps: every card, a board count per card, both of from_json()'s
branches, the methods --prec knows, and the widths the report can label. This
round attacks what that sweep holds fixed — the preset (always the dense
llama31-8b), the KV cache's width (always 2), a width of 0, the --prec tokens
that are not methods (int4, bf16, q4km, q6k, q8) — and the one path no test
drives to a width refusal: the command line itself, `--json` to exit status 2.

The rest are weaker fixes the sweep should see: the bug put back at either
builder, the method table written by hand or given a no-method entry, each
refusal narrowed, a disagreeing width overridden by the method's, the FP8 gate
moved ahead of the width rules, and the command's --quantization read off the
width instead of the method.

X4 is not kept: once a width of 0 or less is refused before the width is
resolved, `if not bpp` and `if bpp is None` behave the same (a bool is a type
error before either), so no test can tell them apart.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import (REPORT_PY, PY_WIDTH_IF_MISSING, PY_WIDTH_NO_SINGLE, PY_WIDTH_DEFAULT,
                     PY_WIDTH_CONTRADICTION, PY_WIDTH_NO_METHOD, PY_WIDTH_JSON_GIVEN)

# The own-fields branch's first request default, where the old guess sat.
PY_OWN_CTX_DEFAULT = '    cfg.setdefault("ctx", 8192)\n'
PY_WIDTH_READ = '    quant, bpp = cfg.get("quant") or "", cfg.get("bpp")\n'
PY_WIDTH_RAISE_CONTRADICTION = '        width = METHOD_WIDTH[quant]\n        raise PlanRefused('
PY_METHOD_WIDTH_TABLE = 'METHOD_WIDTH = method_widths()\n'
PY_METHOD_WIDTHS_SKIP_NONE = '        if quant:\n            widths.setdefault(quant, set()).add(bpp)\n'
PY_GGUF_LEVELS = ('    levels = sorted((bpp, label.split()[-1]) for bpp, label in PREC_LABELS.items() '
                  'if label.startswith("GGUF"))\n')
PY_MAIN_JSON = '        if args.json:\n            cfg = from_json(args.json)\n'
PY_CMD_QUANT = ('    if cfg.get("quant"):\n'
                '        parts.append(f"    --quantization {shlex.quote(cfg[\'quant\'])} \\\\")\n')

# The fix's own addition, from the round's findings.
WIDTH_AT_OR_BELOW_ZERO = '    if bpp is not None and bpp <= 0:\n'

S = {
    # ---- the bug put back at the builders ----
    "X1 py: the bug itself, both builders guess 0.5 again": [
        (REPORT_PY, PY_WIDTH_JSON_GIVEN, '        cfg["bpp"] = raw.get("bpp", 0.5)\n', 1),
        (REPORT_PY, PY_OWN_CTX_DEFAULT, '    cfg.setdefault("bpp", 0.5)\n' + PY_OWN_CTX_DEFAULT, 1)],
    "X2 py: the own-fields branch guesses 0.5 again, the preset branch fixed": [
        (REPORT_PY, PY_OWN_CTX_DEFAULT, '    cfg.setdefault("bpp", 0.5)\n' + PY_OWN_CTX_DEFAULT, 1)],
    "X3 py: the own-fields branch defaults to BF16 before the method is consulted": [
        (REPORT_PY, PY_OWN_CTX_DEFAULT, '    cfg.setdefault("bpp", 2)\n' + PY_OWN_CTX_DEFAULT, 1)],
    # ---- axes the sweep holds fixed ----
    "X5 py: the preset branch copies the JSON's width only when it is truthy": [
        (REPORT_PY, PY_WIDTH_JSON_GIVEN, '        if raw.get("bpp"):\n            cfg["bpp"] = raw["bpp"]\n', 1)],
    "X6 py: a missing width is resolved on dense models only": [
        (REPORT_PY, PY_WIDTH_IF_MISSING, '    if bpp is None and cfg.get("active", 100) >= 100:\n', 1)],
    "X7 py: a config naming nothing is sized at its KV cache's width": [
        (REPORT_PY, PY_WIDTH_DEFAULT, '        cfg["bpp"] = METHOD_WIDTH.get(quant, cfg.get("kv_bpp", 2))\n', 1)],
    "X8 py: the width looked up by --prec's token, not by method (int4, bf16 and q4km planned)": [
        (REPORT_PY, PY_WIDTH_NO_SINGLE, "        if quant and quant not in PRECISIONS:\n", 1),
        (REPORT_PY, PY_WIDTH_DEFAULT, '        cfg["bpp"] = PRECISIONS[quant][0] if quant else 2\n', 1)],
    "X9 py: a --json refusal exits 1 with the reason on stdout": [
        (REPORT_PY, PY_MAIN_JSON,
         '        if args.json:\n'
         '            try:\n'
         '                cfg = from_json(args.json)\n'
         '            except PlanRefused as refused:\n'
         '                print(f"error: {refused}")\n'
         '                sys.exit(1)\n', 1)],
    # ---- the method table ----
    "X10 py: only GGUF is refused without a width; an unknown method is BF16": [
        (REPORT_PY, PY_WIDTH_NO_SINGLE, '        if quant == "gguf":\n', 1)],
    "X11 py: the method table written by hand, without gptq": [
        (REPORT_PY, PY_METHOD_WIDTH_TABLE, 'METHOD_WIDTH = {"fp8": 1, "awq": 0.5}\n', 1)],
    "X12 py: the table gives the no-method entry a width": [
        (REPORT_PY, PY_METHOD_WIDTHS_SKIP_NONE, '        widths.setdefault(quant, set()).add(bpp)\n', 1)],
    # ---- each refusal narrowed ----
    "X13 py: 0.5 without a method is planned again, as INT4": [
        (REPORT_PY, PY_WIDTH_NO_METHOD, "    elif not quant and bpp not in (0.5, 1, 2):\n", 1)],
    "X14 py: BF16 alone is refused too": [
        (REPORT_PY, PY_WIDTH_NO_METHOD, "    elif not quant and bpp not in (1,):\n", 1)],
    "X15 py: a method and a width disagree only when the width is larger": [
        (REPORT_PY, PY_WIDTH_CONTRADICTION, "    elif quant in METHOD_WIDTH and bpp > METHOD_WIDTH[quant]:\n", 1)],
    "X16 py: a disagreeing width is overridden by the method's, silently": [
        (REPORT_PY, PY_WIDTH_RAISE_CONTRADICTION,
         '        width = METHOD_WIDTH[quant]\n        cfg["bpp"] = width\n        if False: raise PlanRefused(', 1)],
    "X17 py: the GGUF refusal names only the levels --prec takes": [
        (REPORT_PY, PY_GGUF_LEVELS,
         '    levels = sorted((bpp, PREC_LABELS[bpp].split()[-1]) for bpp, q in PRECISIONS.values() if q == "gguf")\n', 1)],
    # ---- order, and the other builder ----
    "X18 py: the FP8 gate runs before the width rules": [
        (REPORT_PY, PY_WIDTH_READ, "    refuse_fp8_where_vllm_cannot(cfg)\n" + PY_WIDTH_READ, 1)],
    "X19 py: the preset branch resolves a missing width itself, planning GGUF at BF16": [
        (REPORT_PY, PY_WIDTH_JSON_GIVEN,
         '        cfg["bpp"] = raw["bpp"] if "bpp" in raw else METHOD_WIDTH.get(str(raw.get("quant") or "").strip().lower(), 2)\n', 1)],
    # ---- the command ----
    "X20 py: the command's --quantization is read off the width, not the method": [
        (REPORT_PY, PY_CMD_QUANT,
         '    if cfg.get("quant"):\n'
         '        parts.append(f"    --quantization {shlex.quote({1: \'fp8\', 0.5: \'awq\'}.get(cfg[\'bpp\'], cfg[\'quant\']))} \\\\")\n', 1)],
    # ---- round 2: the fix's own addition, a width of 0 or less refused ----
    "Z1 py: a width of 0 or less is planned": [
        (REPORT_PY, WIDTH_AT_OR_BELOW_ZERO, "    if False:\n", 1)],
    "Z2 py: only a negative width is refused (0 planned)": [
        (REPORT_PY, WIDTH_AT_OR_BELOW_ZERO, "    if bpp is not None and bpp < 0:\n", 1)],
}

if __name__ == "__main__":
    run_driver(S)
