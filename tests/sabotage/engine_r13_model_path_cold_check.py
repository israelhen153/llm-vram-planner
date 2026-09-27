#!/usr/bin/env python3
"""Cold check D5-B of fix/refuse-relative-model-paths (e8865f0), beyond engine_r12.

Round 12's corpus weakens the path rules and the entry points. This round varies
what its tests hold fixed: the JSON test plans on the first card of each vendor
(t4-16, rx7900xtx-24: single-device, no CDNA3, no NVLink) with n_gpu 1, bpp 2 and
one dense preset; the menu is driven in-process, never through the real CLI, and
only with paths that carry no whitespace; the CLI test sends one tilde path. So
each gate below keys the refusal to one of those fixed values, and each entry
sabotage moves it to where only the real CLI, or a whitespace-bearing answer,
would notice.

Fourteen of the round's 23 survived e8865f0's tests; the tests that catch them
now generate their paths from parts and their plans from the catalog. One more
survivor is not kept: T1 moved the path refusal after the FP8 refusal, so a plan
with both problems names the FP8 one first. The path is never printed either
way, so the order is left free.

Then the fix's own addition, a JSON config's model path read with the spaces
round it trimmed, gets two sabotages of its own (S1, S2).
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import (REPORT_PY, PY_PATH_VARIABLE, PY_PATH_ABSOLUTE, PY_PATH_TILDE, PY_PATH_RELATIVE,
                     PY_PATH_REASON, PY_PATH_REFUSE_JSON, PY_PATH_REFUSE_MENU)

# Lines this round anchors on that anchors.py does not carry.
MENU_INPUT = '        hf_model = input("  HuggingFace model ID: ").strip() or "/opt/models/YourModel"\n'
MAIN_MENU = '        else:\n            cfg = interactive_mode()\n        refuse_fp8_where_vllm_cannot(cfg)\n'
MAIN_OUTPUT = '    output = args.output\n'
MAIN_PRINT_ERR = '        print(f"error: {refused}", file=sys.stderr)\n'
MAIN_EXIT = '        sys.exit(2)\n'
MAIN_GENERATED = '    print(f"\\nReport generated: {path}")\n'
JSON_PATH_TRIM = ('    if isinstance(cfg.get("hf_model"), str):\n'
                  '        cfg["hf_model"] = cfg["hf_model"].strip()\n')
VALIDATE_FP8_RETURN = ('    # weight kernel for: refused, not planned.\n'
                       '    return refuse_fp8_where_vllm_cannot(cfg)\n')


def gated(cond):
    """The refusal computed only when `cond` holds on the cfg it is given. The menu
    passes {"hf_model": ...} alone, so every gate reads as true there."""
    return (REPORT_PY, PY_PATH_REASON,
            f'    reason = unresolvable_model_path(cfg.get("hf_model")) if {cond} else ""\n', 1)


S = {
    # ---- the bug itself ----
    "B0 py: neither entry path checks the model path (the pre-fix code)": [
        (REPORT_PY, PY_PATH_REFUSE_JSON, "", 1),
        (REPORT_PY, PY_PATH_REFUSE_MENU, "", 1)],
    # ---- gates on what the tests hold fixed ----
    "G1 py: refusal only on single-device boards (mi250x plans ~/models)": [
        gated('((cfg.get("gpu") or {}).get("devices") or 1) == 1')],
    "G2 py: refusal only at n_gpu 1 (a two-GPU plan keeps ~/models)": [
        gated('cfg.get("n_gpu", 1) == 1')],
    "G3 py: refusal skipped for FP8 plans (bpp 1 or quant fp8)": [
        gated('not (cfg.get("bpp") == 1 or cfg.get("quant") == "fp8")')],
    "G4 py: refusal skipped on CDNA3 cards (mi300x, mi325x)": [
        gated('(cfg.get("gpu") or {}).get("gfx") != "gfx942"')],
    "G5 py: refusal only for dense models (a MoE config keeps ~/models)": [
        gated('cfg.get("active", 100) >= 100')],
    "G6 py: refusal skipped when NVLink is on": [
        gated('not cfg.get("nvlink")')],
    # ---- the CLI's handling of the refusal ----
    "C1 py: the menu's refusal escapes main's handler (traceback, exit 1)": [
        (REPORT_PY, MAIN_MENU, '        else:\n            cfg = {}\n        refuse_fp8_where_vllm_cannot(cfg)\n', 1),
        (REPORT_PY, MAIN_OUTPUT, '    if not cfg:\n        cfg = interactive_mode()\n    output = args.output\n', 1)],
    "C2 py: the reason goes to stdout": [
        (REPORT_PY, MAIN_PRINT_ERR, '        print(f"error: {refused}")\n', 1)],
    "C3 py: a refused plan exits 1": [
        (REPORT_PY, MAIN_EXIT, '        sys.exit(1)\n', 1)],
    "C4 py: the stderr line says refused: instead of error:": [
        (REPORT_PY, MAIN_PRINT_ERR, '        print(f"refused: {refused}", file=sys.stderr)\n', 1)],
    "C5 py: from_json warns and plans the placeholder path instead of refusing": [
        (REPORT_PY, PY_PATH_REFUSE_JSON,
         '    try:\n        refuse_model_path_the_server_cannot_resolve(cfg)\n'
         '    except PlanRefused as refused:\n        print(f"warning: {refused}", file=sys.stderr)\n'
         '        cfg["hf_model"] = "/opt/models/YourModel"\n', 1)],
    # ---- the check made later ----
    "T2 py: the PDF is written before the path is refused": [
        (REPORT_PY, PY_PATH_REFUSE_JSON, "", 1),
        (REPORT_PY, MAIN_GENERATED,
         '    refuse_model_path_the_server_cannot_resolve(cfg)\n    print(f"\\nReport generated: {path}")\n', 1)],
    # ---- the opposite failure: a path the server resolves is refused ----
    "O3 py: ~ anywhere refuses an absolute path (/opt/models/~backup/llama)": [
        (REPORT_PY, PY_PATH_ABSOLUTE, '    if not model or (model.startswith("/") and "~" not in model):\n', 1),
        (REPORT_PY, PY_PATH_TILDE, '    if "~" in model:\n', 1)],
    "O4 py: a space anywhere is refused (/opt/my models/llama)": [
        (REPORT_PY, PY_PATH_VARIABLE,
         '    if " " in model:\n        return "contains a space, which the GPU server cannot resolve"\n'
         '    if "$" in model:\n', 1)],
    "O5 py: a one-part name (gpt2) is refused as relative": [
        (REPORT_PY, PY_PATH_RELATIVE,
         '    if model in (".", "..") or model.startswith(("./", "../")) or model.count("/") != 1:\n', 1)],
    "O6 py: a hub id with a dot (org/model.v2) is refused": [
        (REPORT_PY, PY_PATH_RELATIVE,
         '    if model in (".", "..") or model.startswith(("./", "../")) or model.count("/") >= 2 or "." in model:\n', 1)],
    # ---- a kind of path let through, one step past round 12 ----
    "U1 py: a bare ~ is planned": [
        (REPORT_PY, PY_PATH_TILDE, '    if model.startswith("~") and len(model) > 1:\n', 1)],
    "U3 py: a shell variable is refused only beside a / ($MODEL is planned)": [
        (REPORT_PY, PY_PATH_VARIABLE, '    if "$" in model and "/" in model:\n', 1)],
    "U4 py: the menu checks the answer before stripping it ('  ~/llama' is planned)": [
        (REPORT_PY, MENU_INPUT + PY_PATH_REFUSE_MENU,
         '        typed = input("  HuggingFace model ID: ")\n'
         '        refuse_model_path_the_server_cannot_resolve({"hf_model": typed})\n'
         '        hf_model = typed.strip() or "/opt/models/YourModel"\n', 1)],
    "U5 py: the three-part rule skips paths with a dot (models/llama/x.gguf is planned)": [
        (REPORT_PY, PY_PATH_RELATIVE,
         '    if model in (".", "..") or model.startswith(("./", "../")) or (model.count("/") >= 2 and "." not in model):\n', 1)],
    # ---- the refusal's words, on the menu alone ----
    "M1 py: the menu's refusal drops the fix": [
        (REPORT_PY, PY_PATH_REFUSE_MENU,
         '        if unresolvable_model_path(hf_model):\n'
         '            raise PlanRefused(f"The model path {hf_model!r} {unresolvable_model_path(hf_model)}.")\n', 1)],
    # ---- round 2: the trim the fix added ----
    "S1 py: a JSON config's model path is read untrimmed": [
        (REPORT_PY, JSON_PATH_TRIM, "", 1)],
    "S2 py: a JSON config's model path is trimmed on the left only": [
        (REPORT_PY, JSON_PATH_TRIM, JSON_PATH_TRIM.replace(".strip()", ".lstrip()"), 1)],
}

if __name__ == "__main__":
    run_driver(S)
