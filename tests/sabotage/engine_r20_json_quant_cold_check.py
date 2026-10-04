#!/usr/bin/env python3
"""Round 20: the cold check of fix/json-known-quantizations at 6a3a7f9.

Rounds 15 to 17 brought each of the owner's rules back one at a time on the axes
the report suite's sweep varies: every card, a board count per card, both of
from_json()'s branches, the smallest dense and MoE presets, both KV widths, every
name on either vendor's list and every width the report can label. This round
attacks what that sweep holds fixed — the model's size and attention regime (8B
standard and 26B SWA only; no 100B+, no MLA), the request fields (ctx 8192, conc 1),
the model id (a preset's, never one that names its own quantization), widths given
as floats (1 and 2 arrive as ints; 0 and -1 too), and a JSON null for "quant".

The rest are weaker fixes the sweep should see: the vendor's list chosen by the
card's LLVM target, the list consulted before the spelling is read or only with a
width, the FP8 gate seeing the literal fp8 only, one byte overriding every FP8
method's name, the GGUF levels named on every no-width refusal, a GGUF level's
width passing without its method, the zero refusal or the contradiction check
narrowed, the list refusal raised as a TypeError, and the command dropping
--quantization for a method with no fixed width.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import (REPORT_PY, PY_WIDTH_IF_MISSING, PY_WIDTH_NO_SINGLE, PY_WIDTH_DEFAULT, PY_WIDTH_CONTRADICTION,
                     PY_WIDTH_NO_METHOD, PY_WIDTH_GGUF_HINT, PY_QUANT_SERVED, PY_QUANT_CHECK)

PY_NORMALISE = '    if isinstance(cfg.get("quant"), str):\n        cfg["quant"] = cfg["quant"].strip().lower()\n'
PY_WIDTH_READ = '    quant, bpp = cfg.get("quant") or "", cfg.get("bpp")\n'
PY_AT_OR_BELOW_ZERO = '    if bpp is not None and bpp <= 0:\n'
PY_FP8_FILL = '    if cfg.get("bpp") == 1 and not cfg.get("quant"):\n        cfg["quant"] = "fp8"\n'
PY_FP8_GATE = '    return quant in FP8_METHODS or (not quant and cfg.get("bpp") == 1)\n'
PY_QUANT_RAISE = "        raise PlanRefused(f'\"quant\": {cfg[\"quant\"]!r} isn\\'t a quantization vLLM "
PY_CMD_QUANT = ('    if cfg.get("quant"):\n'
                '        parts.append(f"    --quantization {shlex.quote(cfg[\'quant\'])} \\\\")\n')
HF_SAYS_FP8 = '"fp8" in str(cfg.get("hf_model") or "").lower()'

S = {
    # ---- axes the sweep holds fixed: the model ----
    "C1 py: the width rules run for models under 100B only": [
        (REPORT_PY, PY_WIDTH_IF_MISSING, '    if bpp is None and cfg.get("params", 0) < 100:\n', 1)],
    "C2 py: the width rules skip a model with latent attention": [
        (REPORT_PY, PY_WIDTH_IF_MISSING, '    if bpp is None and cfg.get("attn", "standard") != "mla":\n', 1)],
    "C3 py: the width rules run at one concurrent request only": [
        (REPORT_PY, PY_WIDTH_IF_MISSING, '    if bpp is None and cfg.get("conc", 1) == 1:\n', 1)],
    "C4 py: the width rules run at the default context only": [
        (REPORT_PY, PY_WIDTH_IF_MISSING, '    if bpp is None and cfg.get("ctx", 8192) <= 8192:\n', 1)],
    "C5 py: a method without a fixed width takes the width its model id suggests": [
        (REPORT_PY, PY_WIDTH_NO_SINGLE,
         f'        if quant and quant not in METHOD_WIDTH and not {HF_SAYS_FP8}:\n', 1),
        (REPORT_PY, PY_WIDTH_DEFAULT,
         f'        cfg["bpp"] = METHOD_WIDTH.get(quant, 1 if quant and {HF_SAYS_FP8} else 2)\n', 1)],
    # ---- axes the sweep holds fixed: the width's type, and a null ----
    "C6 py: a width of 0 or less is refused as an integer only (-0.5 and 0.0 planned)": [
        (REPORT_PY, PY_AT_OR_BELOW_ZERO, '    if isinstance(bpp, int) and bpp <= 0:\n', 1)],
    "C7 py: a fixed width given as a float contradicts its method (fp8 with 1.0 refused)": [
        (REPORT_PY, PY_WIDTH_CONTRADICTION,
         '    elif quant in METHOD_WIDTH and (bpp != METHOD_WIDTH[quant] or type(bpp) is not type(METHOD_WIDTH[quant])):\n', 1)],
    "C8 py: one byte as a float names no quantization (1.0 alone sized FP8, commanded BF16)": [
        (REPORT_PY, PY_FP8_FILL,
         '    if cfg.get("bpp") == 1 and isinstance(cfg.get("bpp"), int) and not cfg.get("quant"):\n'
         '        cfg["quant"] = "fp8"\n', 1)],
    "C9 py: a null quantization is read as the name 'none' and refused": [
        (REPORT_PY, PY_NORMALISE, '    if "quant" in cfg:\n        cfg["quant"] = str(cfg["quant"]).strip().lower()\n', 1)],
    # ---- weaker fixes on the list ----
    "C10 py: the vendor's list is chosen by the card's LLVM target (gfx90a and RDNA take NVIDIA's)": [
        (REPORT_PY, PY_QUANT_SERVED,
         '    served = VLLM_QUANTIZATIONS["amd" if (cfg.get("gpu") or {}).get("gfx") == "gfx942" else "nvidia"]\n', 1)],
    "C11 py: the list is checked before the spelling is read": [
        (REPORT_PY, PY_NORMALISE, "", 1),
        (REPORT_PY, PY_WIDTH_READ, PY_NORMALISE + PY_WIDTH_READ, 1)],
    "C16 py: the list is consulted only when a width is given": [
        (REPORT_PY, PY_QUANT_CHECK,
         '    if cfg.get("quant") and cfg["quant"] not in served and cfg.get("bpp") is not None:\n', 1)],
    "C19 py: the list refusal is a TypeError, like validate_arch's other refusals": [
        (REPORT_PY, PY_QUANT_RAISE, PY_QUANT_RAISE.replace("PlanRefused(", "TypeError("), 1)],
    # ---- weaker fixes on the widths ----
    "C12 py: one byte per parameter overrides every FP8 method's name with fp8": [
        (REPORT_PY, PY_FP8_FILL, '    if cfg.get("bpp") == 1:\n        cfg["quant"] = "fp8"\n', 1)],
    "C13 py: the FP8 gate sees the literal fp8 only": [
        (REPORT_PY, PY_FP8_GATE, PY_FP8_GATE.replace("quant in FP8_METHODS", 'quant == "fp8"'), 1)],
    "C14 py: every no-width refusal names the GGUF levels": [
        (REPORT_PY, PY_WIDTH_GGUF_HINT,
         PY_WIDTH_GGUF_HINT.replace('{gguf_widths() if quant == "gguf" else ""}', "{gguf_widths()}"), 1)],
    "C15 py: a width with no method is planned when it is a GGUF level": [
        (REPORT_PY, PY_WIDTH_NO_METHOD, "    elif not quant and bpp not in (1, 2) and bpp not in PREC_LABELS:\n", 1)],
    "C17 py: a width of 0 or less is refused only when no method is named": [
        (REPORT_PY, PY_AT_OR_BELOW_ZERO, "    if bpp is not None and bpp <= 0 and not quant:\n", 1)],
    "C18 py: the contradiction check knows --prec's widths only, not the FP8 methods'": [
        (REPORT_PY, PY_WIDTH_CONTRADICTION, "    elif quant in method_widths() and bpp != method_widths()[quant]:\n", 1)],
    # ---- the command ----
    "C20 py: the command drops --quantization for a method with no fixed width": [
        (REPORT_PY, PY_CMD_QUANT, PY_CMD_QUANT.replace('if cfg.get("quant"):', 'if cfg.get("quant") in METHOD_WIDTH:'), 1)],
}

if __name__ == "__main__":
    run_driver(S)
