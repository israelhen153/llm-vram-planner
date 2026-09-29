#!/usr/bin/env python3
"""Round 19: the cold check of fix/hf-model-input, along what round 18's tests hold fixed.

Round 18's tests judge the new kinds well where they are generated: every dash-led and
empty shape goes through a JSON config on every plan axis and through the menu with every
whitespace round it. The sabotages here go where the generation stops.

- The wrong-type test runs one card, one board count, one weight width and one dense
  preset, so a type check keyed to any of them survives (A1-A3).
- It feeds one truthy instance of each wrong type, so a check that waves a false, 0, []
  or {} through as "no model given" survives (B1).
- No test drives a wrong-typed path through the real CLI, so main() can turn the
  TypeError into a warning and exit 0 (F2).
- The menu is driven on one architecture, a dense one (E1).
- Every whitespace padding is a space, a tab or a newline, so a trim that leaves a
  carriage return survives on both routes (T1, T2).

The rest are controls expected caught: the dash rule narrowed or judged after the
relative one, a whitespace-only path planned untrimmed or given the placeholder, the
preset branch's `or` fallback, the menu's default applied before the trim, a wrong type
coerced to text, and the empty path raised as a TypeError.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import REPORT_PY, PY_HF_TYPE, PY_PATH_EMPTY, PY_PATH_DASH, PY_PATH_REFUSE_MENU

# The engine text this round attacks that no earlier round anchored on.
STRIP = ('    if isinstance(cfg.get("hf_model"), str):\n'
         '        cfg["hf_model"] = cfg["hf_model"].strip()\n')
MENU_LINE = '        hf_model = input("  HuggingFace model ID: ").strip() or "/opt/models/YourModel"\n'
DASH_RULE = ('    if model.startswith("-"):\n'
             '        return "starts with -, so vLLM would read it as an option"\n')
RELATIVE_TAIL = ('                "the ROCm container that is /vllm-workspace")\n'
                 '    return ""\n')
PRESET_HF = '            "hf_model": raw.get("hf_model", pd["hf"]),\n'
MAIN_JSON = '        if args.json:\n            cfg = from_json(args.json)\n'

# A hand-written type check in place of the generic entry, run only under COND: the
# shape of a fix keyed to an axis its test never varies. It names the key the way the
# real one does, so the message check passes whenever the check runs at all.
TYPE_CHECK = '''    if COND and not isinstance(cfg.get("hf_model"), str):
        raise TypeError(f"cfg['hf_model'] must be str, got {type(cfg.get('hf_model')).__name__}: "
                        f"{cfg.get('hf_model')!r}")
'''


def typed_only_when(condition):
    return [(REPORT_PY, PY_HF_TYPE, "", 1),
            (REPORT_PY, STRIP, TYPE_CHECK.replace("COND", condition) + STRIP, 1)]


S = {
    # The wrong-type test: one card (h100-80), one board, bpp 2, one dense preset.
    "A1 py: a wrong-typed model path is named only on NVIDIA cards":
        typed_only_when('(cfg.get("gpu") or {}).get("vendor") != "amd"'),
    "A2 py: a wrong-typed model path is named only below two boards":
        typed_only_when('cfg.get("n_gpu", 1) < 2'),
    "A3 py: a wrong-typed model path is named only at two bytes per parameter":
        typed_only_when('cfg.get("bpp") == 2'),
    # One truthy instance of each wrong type: true, but never false, 0, [] or {}.
    "B1 py: a wrong-typed model path is named only when truthy, null aside (false, 0, [] and {} are refused as empty)":
        typed_only_when('(cfg.get("hf_model") is None or cfg.get("hf_model"))'),
    # The dash rule, narrowed or reordered: controls.
    "C1 py: a path starting with - is refused only before a letter or digit (-.hidden/m and --m are planned)": [
        (REPORT_PY, PY_PATH_DASH, '    if model.startswith("-") and (len(model) == 1 or model[1].isalnum()):\n', 1)],
    "C2 py: the dash rule is judged after the relative one (-a/b/c gets the relative reason)": [
        (REPORT_PY, DASH_RULE, "", 1),
        (REPORT_PY, RELATIVE_TAIL, RELATIVE_TAIL.replace('    return ""\n', DASH_RULE + '    return ""\n'), 1)],
    # The empty path: controls.
    "D1 py: a whitespace-only JSON path is planned untrimmed": [
        (REPORT_PY, STRIP, STRIP.replace('.strip()\n', '.strip() or cfg["hf_model"]\n'), 1)],
    "D2 py: an empty JSON path is given the placeholder path instead of refused": [
        (REPORT_PY, STRIP, STRIP.replace('.strip()\n', '.strip() or "/opt/models/YourModel"\n'), 1)],
    "D3 py: the preset branch falls back to the preset's id for an empty or null path": [
        (REPORT_PY, PRESET_HF, '            "hf_model": raw.get("hf_model") or pd["hf"],\n', 1)],
    # The menu: driven on one architecture, and its default's order with the trim.
    "E1 py: the menu's refusal is skipped for MoE models": [
        (REPORT_PY, PY_PATH_REFUSE_MENU, '        if arch["active"] >= 100:\n    ' + PY_PATH_REFUSE_MENU, 1)],
    "E2 py: the menu's default is applied before the answer is trimmed (a whitespace-only answer is refused as empty)": [
        (REPORT_PY, MENU_LINE,
         '        hf_model = (input("  HuggingFace model ID: ") or "/opt/models/YourModel").strip()\n', 1)],
    # The trim: every padding the tests type is a space, a tab or a newline.
    "T1 py: a JSON config's path is trimmed of spaces, tabs and newlines only (a carriage return stays)": [
        (REPORT_PY, STRIP, STRIP.replace('.strip()', '.strip(" \\t\\n")'), 1)],
    "T2 py: the menu trims spaces, tabs and newlines only (a carriage return stays, so a CRLF answer is not the default)": [
        (REPORT_PY, MENU_LINE, MENU_LINE.replace('.strip()', '.strip(" \\t\\n")'), 1)],
    # The wrong type's handling: controls, then the CLI.
    "G1 py: a wrong-typed model path is coerced to text (123 plans the hub id '123')": [
        (REPORT_PY, PY_HF_TYPE, "", 1),
        (REPORT_PY, STRIP, '    cfg["hf_model"] = str(cfg.get("hf_model") or "").strip()\n', 1)],
    "F1 py: an empty path is a TypeError rather than a refusal": [
        (REPORT_PY, PY_PATH_EMPTY, '    if not model:\n        raise TypeError("cfg[\'hf_model\'] is empty")\n', 1)],
    "F2 py: main() turns a wrong-typed model path's TypeError into a warning and exit 0": [
        (REPORT_PY, MAIN_JSON,
         '        if args.json:\n'
         '            try:\n'
         '                cfg = from_json(args.json)\n'
         '            except TypeError as wrong:\n'
         '                print(f"warning: {wrong}; nothing planned", file=sys.stderr)\n'
         '                sys.exit(0)\n', 1)],
}

if __name__ == "__main__":
    run_driver(S)
