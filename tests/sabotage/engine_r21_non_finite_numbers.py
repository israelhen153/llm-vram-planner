#!/usr/bin/env python3
"""Round 21: a NaN or Infinity in a JSON config or the menu (fix/json-precision-width).

Python's json reads NaN, Infinity and -Infinity, and reads 1e400 as Infinity; the
menu's float() takes "nan" and "inf". Each crashed the sizing with a traceback, or, as
"nvlink", planned NVLink. The fix refuses every non-finite value as soon as the file is
read, and the menu's as soon as it is typed. These bring the bug back at either route,
and weaker fixes: the check narrowed to NaN, to Infinity or to +Infinity, to the fields
validate_arch() types or to the request fields, to configs naming a preset or to NVIDIA
cards; a non-finite value dropped for its default; the refusal without its field or
raised as a TypeError; and the menu checking one of its two float answers.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import REPORT_PY, PY_NON_FINITE_TEST, PY_NON_FINITE_RAISE, PY_NON_FINITE_JSON, PY_NON_FINITE_MENU

JSON_CHECK = "        refuse_non_finite({json.dumps(key): value for key, value in raw.items()})\n"

S = {
    # ---- the bug put back ----
    "N1 py: from_json reads NaN and Infinity again": [
        (REPORT_PY, PY_NON_FINITE_JSON, "", 1)],
    "N2 py: the menu takes nan and inf again": [
        (REPORT_PY, PY_NON_FINITE_MENU, "", 1)],
    # ---- the test narrowed ----
    "N3 py: NaN refused, Infinity planned": [
        (REPORT_PY, PY_NON_FINITE_TEST, "        if isinstance(value, float) and math.isnan(value):\n", 1)],
    "N4 py: Infinity refused, NaN planned": [
        (REPORT_PY, PY_NON_FINITE_TEST, "        if isinstance(value, float) and math.isinf(value):\n", 1)],
    "N5 py: -Infinity passes (NaN and +Infinity refused)": [
        (REPORT_PY, PY_NON_FINITE_TEST,
         "        if isinstance(value, float) and (value != value or value == math.inf):\n", 1)],
    # ---- the keys, branches and cards narrowed ----
    "N6 py: only the fields validate_arch types are checked (kv_bpp, nvlink, preset, gpu pass)": [
        (REPORT_PY, JSON_CHECK,
         "        refuse_non_finite({json.dumps(key): value for key, value in raw.items() if key in ARCH_TYPES})\n", 1)],
    "N7 py: only the request fields are checked": [
        (REPORT_PY, JSON_CHECK,
         "        refuse_non_finite({json.dumps(key): value for key, value in raw.items() if key in REQUEST_KEYS})\n", 1)],
    "N8 py: the check runs only when the config names a preset": [
        (REPORT_PY, PY_NON_FINITE_JSON, PY_NON_FINITE_JSON.replace("isinstance(raw, dict):", "isinstance(raw, dict) and raw.get(\"preset\"):"), 1)],
    "N9 py: the check skips AMD cards": [
        (REPORT_PY, PY_NON_FINITE_JSON,
         PY_NON_FINITE_JSON.replace("isinstance(raw, dict):",
                                    "isinstance(raw, dict) and GPUS.get(raw.get(\"gpu\"), {}).get(\"vendor\") != \"amd\":"), 1)],
    # ---- the refusal weakened ----
    "N10 py: a non-finite value is dropped, so the preset's or the default is planned": [
        (REPORT_PY, PY_NON_FINITE_JSON,
         "    if isinstance(raw, dict):\n"
         "        raw = {k: v for k, v in raw.items() if not (isinstance(v, float) and not math.isfinite(v))}\n", 1)],
    "N11 py: the refusal doesn't name the field": [
        (REPORT_PY, PY_NON_FINITE_RAISE, PY_NON_FINITE_RAISE.replace('f"{label}: {json.dumps(value)}', 'f"A value: {json.dumps(value)}'), 1)],
    "N12 py: a non-finite value is a TypeError, exit 1, like a wrong type": [
        (REPORT_PY, PY_NON_FINITE_RAISE, PY_NON_FINITE_RAISE.replace("raise PlanRefused(", "raise TypeError("), 1)],
    "N13 py: the menu checks the parameter count only": [
        (REPORT_PY, PY_NON_FINITE_MENU, '        refuse_non_finite({"Parameters (B)": arch["params"]})\n', 1)],
}

if __name__ == "__main__":
    run_driver(S)
