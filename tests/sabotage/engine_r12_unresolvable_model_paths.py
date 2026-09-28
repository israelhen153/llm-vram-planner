#!/usr/bin/env python3
"""Round 12: the model-path refusal (fix/refuse-relative-model-paths).

The printed command runs on the GPU server. A model path starting with ~, or
naming a shell variable, stays literal there because the command quotes it,
and a relative one depends on the directory the command runs from; inside the
ROCm container that is /vllm-workspace. The owner chose to refuse them with
the reason and the fix (decision desk D5-B, 2026-09-27).

Each sabotage brings back one way the refusal could quietly weaken: a kind of
path let through, an entry path left unchecked, a check made too late or on
one vendor only, a refusal without its fix, and the opposite failure, a path
the server can resolve being refused.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import (REPORT_PY, PY_PATH_VARIABLE, PY_PATH_ABSOLUTE, PY_PATH_TILDE, PY_PATH_RELATIVE,
                     PY_PATH_REASON, PY_PATH_REFUSAL, PY_PATH_REFUSE_JSON, PY_PATH_REFUSE_MENU, PY_MENU_NAME)

S = {
    # ---- a kind of path let through ----
    "P1 py: a path naming a shell variable is planned": [
        (REPORT_PY, PY_PATH_VARIABLE, "    if False:\n", 1)],
    "P2 py: a shell variable inside an absolute path is planned": [
        (REPORT_PY, PY_PATH_VARIABLE, '    if "$" in model and not model.startswith("/"):\n', 1)],
    "P3 py: a path starting with ~ is planned": [
        (REPORT_PY, PY_PATH_TILDE, "    if False:\n", 1)],
    "P4 py: only ~/ is refused, so ~user/ and ~name pass": [
        (REPORT_PY, PY_PATH_TILDE, '    if model.startswith("~/"):\n', 1)],
    "P5 py: ./, ../, . and .. are planned": [
        (REPORT_PY, PY_PATH_RELATIVE, '    if model.count("/") >= 2:\n', 1)],
    "P6 py: a relative path of three or more parts is planned": [
        (REPORT_PY, PY_PATH_RELATIVE, '    if model in (".", "..") or model.startswith(("./", "../")):\n', 1)],
    "P7 py: . and .. are planned": [
        (REPORT_PY, PY_PATH_RELATIVE, '    if model.startswith(("./", "../")) or model.count("/") >= 2:\n', 1)],
    # ---- an entry path left unchecked, or checked too late, or on one vendor ----
    "E1 py: a JSON config's model path is not checked": [
        (REPORT_PY, PY_PATH_REFUSE_JSON, "", 1)],
    "E2 py: the menu's model path is not checked": [
        (REPORT_PY, PY_PATH_REFUSE_MENU, "", 1)],
    "E3 py: the menu checks the model path only after asking for a display name": [
        (REPORT_PY, PY_PATH_REFUSE_MENU, "", 1),
        (REPORT_PY, PY_MENU_NAME, PY_MENU_NAME + PY_PATH_REFUSE_MENU, 1)],
    "E4 py: the refusal applies only on AMD cards": [
        (REPORT_PY, PY_PATH_REASON,
         '    reason = unresolvable_model_path(cfg.get("hf_model")) if cfg.get("vendor") == "amd" else ""\n', 1)],
    # ---- the refusal's words ----
    "W1 py: the refusal gives the reason but not the fix": [
        (REPORT_PY, PY_PATH_REFUSAL, "        raise PlanRefused(f\"The model path {cfg['hf_model']!r} {reason}.\")\n", 1)],
    # ---- the opposite failure: a path the server can resolve is refused ----
    "O1 py: a Hugging Face id (org/name) is refused as relative": [
        (REPORT_PY, PY_PATH_RELATIVE, PY_PATH_RELATIVE.replace('model.count("/") >= 2', 'model.count("/") >= 1'), 1)],
    "O2 py: an absolute path is refused as relative": [
        (REPORT_PY, PY_PATH_ABSOLUTE, "    if not model:\n", 1)],
}

if __name__ == "__main__":
    run_driver(S)
