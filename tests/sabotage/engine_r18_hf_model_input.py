#!/usr/bin/env python3
"""Round 18: a JSON config's model path that isn't a path (fix/hf-model-input).

"hf_model": null, "" or "   " planned `vllm serve ''`; 123 or a list crashed
inside shlex.quote with a TypeError that named nothing; and "-8b" printed a
command vLLM reads as an option. The owner's decision (decision desk HF,
2026-09-27): refuse an empty path and one starting with -, and name a wrong
type the way quant and bpp already are.

Each sabotage lets one back in: the type check dropped or widened to allow
null, an empty path planned, a leading - planned, or only -- refused.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import REPORT_PY, PY_HF_TYPE, PY_PATH_EMPTY, PY_PATH_DASH

S = {
    "H1 py: a wrong-typed model path is not checked": [
        (REPORT_PY, PY_HF_TYPE, "", 1)],
    "H2 py: a null model path is accepted": [
        (REPORT_PY, PY_HF_TYPE, '    "hf_model": (str, type(None)),\n', 1)],
    "H3 py: an empty model path is planned": [
        (REPORT_PY, PY_PATH_EMPTY, "", 1)],
    "H4 py: a model path starting with - is planned": [
        (REPORT_PY, PY_PATH_DASH, "    if False:\n", 1)],
    "H5 py: only a path starting with -- is refused": [
        (REPORT_PY, PY_PATH_DASH, '    if model.startswith("--"):\n', 1)],
}

if __name__ == "__main__":
    run_driver(S)
