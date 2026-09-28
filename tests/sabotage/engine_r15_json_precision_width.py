#!/usr/bin/env python3
"""Round 15: a JSON config's width (fix/json-precision-width).

A JSON config that gave no "bpp" was sized at 0.5 bytes per parameter whatever
it named: FP8 at half the weights its command loads, and a config naming
nothing at a quarter of its BF16 checkpoint. The owner's rules (decision desk
A2, 2026-09-27): a method with one width in --prec's table gives the width, a
config naming nothing is BF16, and GGUF or an unknown method without "bpp", a
width with no method, and a method and width that disagree are refused.

Each sabotage brings one guess back: the old default, a method's width
ignored, a refusal turned into a plan, the width a JSON gave dropped in one
branch, GGUF treated as having one width, the rules kept to one vendor or one
board, and the GGUF refusal without its levels.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import (REPORT_PY, PY_WIDTH_IF_MISSING, PY_WIDTH_NO_SINGLE, PY_WIDTH_GGUF_HINT, PY_WIDTH_DEFAULT,
                     PY_WIDTH_CONTRADICTION, PY_WIDTH_NO_METHOD, PY_WIDTH_JSON_GIVEN, PY_WIDTH_ONE_ONLY)

S = {
    # ---- a guess brought back ----
    "W1 py: a config naming nothing is sized at 0.5 again": [
        (REPORT_PY, PY_WIDTH_DEFAULT, PY_WIDTH_DEFAULT.replace("METHOD_WIDTH.get(quant, 2)", "METHOD_WIDTH.get(quant, 0.5)"), 1)],
    "W2 py: every named method is sized at 0.5, FP8 included": [
        (REPORT_PY, PY_WIDTH_DEFAULT, '        cfg["bpp"] = 0.5 if quant else 2\n', 1)],
    "W3 py: GGUF or an unknown method without a width is planned, not refused": [
        (REPORT_PY, PY_WIDTH_NO_SINGLE, "        if False:\n", 1)],
    "W4 py: GGUF without a width is planned; only an unknown method is refused": [
        (REPORT_PY, PY_WIDTH_NO_SINGLE, '        if quant and quant not in METHOD_WIDTH and quant != "gguf":\n', 1)],
    # ---- a refusal turned into a plan ----
    "W5 py: a method and a width that disagree are planned": [
        (REPORT_PY, PY_WIDTH_CONTRADICTION, "    elif False:\n", 1)],
    "W6 py: a width with no method is planned": [
        (REPORT_PY, PY_WIDTH_NO_METHOD, "    elif False:\n", 1)],
    # ---- the width a JSON gave, and the table the widths come from ----
    "W7 py: the preset branch drops the width a JSON gave": [
        (REPORT_PY, PY_WIDTH_JSON_GIVEN, "", 1)],
    "W8 py: GGUF is treated as having one width (whichever its set yields first)": [
        (REPORT_PY, PY_WIDTH_ONE_ONLY, PY_WIDTH_ONE_ONLY.replace(" if len(ws) == 1", ""), 1)],
    # ---- the rules kept to one vendor or one board ----
    "W9 py: a missing width is resolved on NVIDIA cards only": [
        (REPORT_PY, PY_WIDTH_IF_MISSING, '    if bpp is None and (cfg.get("gpu") or {}).get("vendor") == "nvidia":\n', 1)],
    "W10 py: a missing width is resolved at one board only": [
        (REPORT_PY, PY_WIDTH_IF_MISSING, '    if bpp is None and cfg.get("n_gpu", 1) == 1:\n', 1)],
    "W11 py: contradictions are refused on NVIDIA cards only": [
        (REPORT_PY, PY_WIDTH_CONTRADICTION,
         '    elif quant in METHOD_WIDTH and bpp != METHOD_WIDTH[quant] and (cfg.get("gpu") or {}).get("vendor") == "nvidia":\n', 1)],
    # ---- the refusal's words ----
    "W12 py: the GGUF refusal names no levels": [
        (REPORT_PY, PY_WIDTH_GGUF_HINT, PY_WIDTH_GGUF_HINT.replace('{gguf_widths() if quant == "gguf" else ""}', ""), 1)],
}

if __name__ == "__main__":
    run_driver(S)
