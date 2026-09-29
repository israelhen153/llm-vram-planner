#!/usr/bin/env python3
"""Round 17: the quantizations a JSON config may name (fix/json-known-quantizations).

A JSON config's quant went straight into --quantization, so a name vLLM doesn't
know, or a CUDA-only one on AMD, printed a command that stops at startup; and
only the literal "fp8" was FP8. The owner's decisions (decision desk A1 and A3,
2026-09-27): only a quantization vLLM v0.30.0 serves on the card's vendor, and
every FP8 method on that list fixed at one byte.

Each sabotage lets one of those slip: any name accepted, AMD given NVIDIA's
list, the list checked on one vendor or only without a width, gguf dropped
from a vendor, a CUDA-only method accepted on AMD, the FP8 methods reduced to
"fp8" or sized at 0.5 or missing one, and the refusal without its names.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import (REPORT_PY, PY_QUANT_SERVED, PY_QUANT_CHECK, PY_QUANT_HINT, PY_FP8_WIDTHS, PY_FP8_METHODS_TAIL,
                     PY_QUANT_NVIDIA_TAIL, PY_QUANT_AMD_TAIL)

S = {
    # ---- the list ----
    "Q1 py: any name is planned": [
        (REPORT_PY, PY_QUANT_CHECK, "    if False:\n", 1)],
    "Q2 py: AMD takes NVIDIA's list": [
        (REPORT_PY, PY_QUANT_SERVED, '    served = VLLM_QUANTIZATIONS["nvidia"]\n', 1)],
    "Q3 py: the list is checked on NVIDIA cards only": [
        (REPORT_PY, PY_QUANT_CHECK, '    if cfg.get("quant") and cfg["quant"] not in served and vendor == "nvidia":\n', 1)],
    "Q4 py: the list is checked only when no width is given": [
        (REPORT_PY, PY_QUANT_CHECK,
         '    if cfg.get("quant") and cfg["quant"] not in served and cfg.get("bpp") is None:\n', 1)],
    "Q5 py: gguf is dropped from NVIDIA's list": [
        (REPORT_PY, PY_QUANT_NVIDIA_TAIL, '               "nvfp4_per_token", "mxfp8"),\n', 1)],
    "Q6 py: gguf is dropped from AMD's list": [
        (REPORT_PY, PY_QUANT_AMD_TAIL, '            ),\n', 1)],
    "Q7 py: a CUDA-only method is accepted on AMD": [
        (REPORT_PY, PY_QUANT_AMD_TAIL, '            "gptq_marlin", "gguf"),\n', 1)],
    # ---- the FP8 methods ----
    "Q8 py: only the literal fp8 is fixed at one byte": [
        (REPORT_PY, PY_FP8_WIDTHS, "METHOD_WIDTH = method_widths()\n", 1)],
    "Q9 py: the FP8 methods are fixed at 0.5": [
        (REPORT_PY, PY_FP8_WIDTHS, PY_FP8_WIDTHS.replace("{method: 1 for", "{method: 0.5 for"), 1)],
    "Q10 py: an FP8 method is missing from the FP8 list": [
        (REPORT_PY, PY_FP8_METHODS_TAIL, '               "modelopt_mxfp8")\n', 1)],
    # ---- the refusal's words ----
    "Q11 py: the refusal names no quantizations to use instead": [
        (REPORT_PY, PY_QUANT_HINT, "                          f'startup.')\n", 1)],
}

if __name__ == "__main__":
    run_driver(S)
