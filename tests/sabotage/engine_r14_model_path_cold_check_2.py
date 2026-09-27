#!/usr/bin/env python3
"""The second cold check of fix/refuse-relative-model-paths (c0c770c), beyond
engine_r12 and engine_r13.

Round 13's tests sweep every generated path through one JSON configuration
(the first card, one board, BF16, one dense preset) and every configuration
through one sample path per KIND of refusal — the sorted set's first variable,
tilde and relative paths, which are "$HOME/.hidden/m", "~" and ".". So a rule
INSIDE a kind (three-part paths, ~name/, a $ inside an absolute path) that is
keyed to the configuration is judged only where the configuration is fixed, and
a gate on an axis the sweep never varies at all — the KV cache's precision, a
4-bit or GGUF weight precision, the context length, four boards and more — is
judged nowhere. The same holds for the trim: the sweep pads paths only on the
fixed configuration. Each K sabotage keys one such rule to one such axis.

The generated paths are made from fixed starts and bodies, so the shapes a
normalisation would change are absent: no relative path of three parts ends
with a /, none doubles a /, none ends with a $ (N1-N3). The menu's test pads its
answers with spaces only, and the JSON test with a space and a tab, so a trim of
a narrower character set passes each (W1, W2). And no path both starts with ~
and names a variable, so which reason wins is unpinned (P1, informational: the
requirement orders no precedence).

The V sabotages are controls along new axes that the tests are expected to
catch: a whole-kind gate on the vendor, the tilde rule skipped on a dotted path,
a four-part threshold, the trim kept in one from_json branch, the menu's refusal
exiting 3, and the path expanded on the planner's machine before judging.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import (REPORT_PY, PY_PATH_VARIABLE, PY_PATH_TILDE, PY_PATH_RELATIVE, PY_PATH_REASON)

# Lines this round anchors on that anchors.py does not carry.
MENU_INPUT = '        hf_model = input("  HuggingFace model ID: ").strip() or "/opt/models/YourModel"\n'
JSON_PATH_TRIM = ('    if isinstance(cfg.get("hf_model"), str):\n'
                  '        cfg["hf_model"] = cfg["hf_model"].strip()\n')
MAIN_EXIT = '        sys.exit(2)\n'
PATH_FN_HEAD = '    model = str(model or "")\n    if "$" in model:\n'
PRESET_RETURN = '        return validate_arch(cfg)\n'
TILDE_REASON = '"starts with ~, which the printed command quotes, so the GPU server never expands it"'


def dropped_when(cond):
    """The refusal computed, then dropped when `cond` holds on the cfg and the path.
    The menu passes {"hf_model": ...} alone, so every gate reads as false there;
    only a JSON plan on a configuration the sweep never builds sees the difference."""
    return (REPORT_PY, PY_PATH_REASON,
            '    model = str(cfg.get("hf_model") or "")\n'
            '    reason = unresolvable_model_path(model)\n'
            f'    if reason and ({cond}):\n'
            '        reason = ""\n', 1)


S = {
    # ---- a rule inside a kind, keyed to an axis the sweep fixes for all but one sample ----
    "K1 py: the three-part rule is skipped on AMD cards (models/llama/x.gguf plans on mi300x)": [
        dropped_when('reason.startswith("is relative") and model not in (".", "..") '
                     'and not model.startswith(("./", "../")) and (cfg.get("gpu") or {}).get("vendor") == "amd"')],
    "K2 py: ~name/ is planned above one board (~user/m at n_gpu 2)": [
        dropped_when('reason.startswith("starts with ~") and model != "~" and not model.startswith("~/") '
                     'and cfg.get("n_gpu", 1) > 1')],
    "K3 py: a $ inside an absolute path is planned on FP8 weights (/data/$USER/m at bpp 1)": [
        dropped_when('model.startswith("/") and cfg.get("bpp") == 1')],
    "K4 py: a JSON config's model path is trimmed only for dense models (' ~/m' plans on a MoE preset)": [
        (REPORT_PY, JSON_PATH_TRIM,
         '    if isinstance(cfg.get("hf_model"), str) and cfg.get("active", 100) >= 100:\n'
         '        cfg["hf_model"] = cfg["hf_model"].strip()\n', 1)],
    # ---- the whole refusal, keyed to an axis the sweep never varies ----
    "K5 py: the refusal is skipped with an FP8 KV cache (kv_bpp 1)": [
        dropped_when('cfg.get("kv_bpp", 2) == 1')],
    "K6 py: the refusal is skipped above the default context length": [
        dropped_when('cfg.get("ctx", 8192) != 8192')],
    "K7 py: the refusal is skipped at four boards and more": [
        dropped_when('cfg.get("n_gpu", 1) >= 4')],
    # "bpp" in cfg, not cfg.get("bpp", 0.5): the menu's cfg has no bpp key, and a gate
    # that fires on the missing key is caught by the menu's test for that reason alone.
    "K8 py: the refusal is skipped for 4-bit and GGUF weights (bpp 0.5, the JSON default)": [
        dropped_when('"bpp" in cfg and cfg["bpp"] not in (1, 2)')],
    # ---- a normalisation the generated shapes cannot see ----
    "N1 py: a trailing slash is dropped before counting parts (models/llama/ is planned)": [
        (REPORT_PY, PY_PATH_RELATIVE, PY_PATH_RELATIVE.replace('model.count("/") >= 2', 'model.rstrip("/").count("/") >= 2'), 1)],
    "N2 py: doubled slashes are collapsed before counting parts (a//b is planned)": [
        (REPORT_PY, PY_PATH_RELATIVE, PY_PATH_RELATIVE.replace('model.count("/") >= 2', 'model.replace("//", "/").count("/") >= 2'), 1)],
    "N3 py: a $ at the end of the path is planned (x$)": [
        (REPORT_PY, PY_PATH_VARIABLE, '    if "$" in model and model.split("$", 1)[1][:1] != "":\n', 1)],
    # ---- whitespace the tests never send ----
    "W1 py: the menu trims spaces only (a tab round the answer stays)": [
        (REPORT_PY, MENU_INPUT, MENU_INPUT.replace(".strip()", '.strip(" ")'), 1)],
    "W2 py: a JSON config's model path is trimmed of spaces and tabs only (a newline stays)": [
        (REPORT_PY, JSON_PATH_TRIM, JSON_PATH_TRIM.replace(".strip()", '.strip(" \\t")'), 1)],
    # ---- precedence of reasons (informational: the requirement orders none) ----
    "P1 py: ~ is judged before $ (~/$HOME/m gets the tilde reason)": [
        (REPORT_PY, PATH_FN_HEAD,
         '    model = str(model or "")\n    if model.startswith("~"):\n'
         f'        return {TILDE_REASON}\n    if "$" in model:\n', 1)],
    # ---- controls along new axes, expected caught ----
    "V1 py: the ~ rule is skipped on a dotted path (~/model.gguf is planned)": [
        (REPORT_PY, PY_PATH_TILDE, '    if model.startswith("~") and "." not in model:\n', 1)],
    "V2 py: the refusal applies only on NVIDIA cards": [
        dropped_when('(cfg.get("gpu") or {}).get("vendor") == "amd"')],
    "V3 py: the parts rule needs four parts (models/llama/x.gguf is planned)": [
        (REPORT_PY, PY_PATH_RELATIVE, PY_PATH_RELATIVE.replace('>= 2', '>= 3'), 1)],
    "V5 py: the trim is kept in from_json's preset branch only": [
        (REPORT_PY, JSON_PATH_TRIM, "", 1),
        (REPORT_PY, PRESET_RETURN, '        cfg["hf_model"] = cfg["hf_model"].strip()\n' + PRESET_RETURN, 1)],
    "V7 py: the menu's refusal exits 3, a JSON config's 2": [
        (REPORT_PY, MAIN_EXIT, '        sys.exit(2 if args.json else 3)\n', 1)],
    "V8 py: ~ and $ are expanded on the planner's machine before judging": [
        (REPORT_PY, PATH_FN_HEAD,
         '    model = os.path.expandvars(os.path.expanduser(str(model or "")))\n    if "$" in model:\n', 1)],
}

if __name__ == "__main__":
    run_driver(S)
