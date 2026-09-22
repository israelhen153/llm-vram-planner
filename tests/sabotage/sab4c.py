#!/usr/bin/env python3
"""Views fed from outside the harness: the copied report's command block (read
from a DOM node the JS harness stubs to empty), a leak gated on a selected
preset (probes never select one), and stdout from generate()."""
import os, sys
sys.dont_write_bytecode = True   # see the note in sab.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sab
from harness import run_driver
from sab import INDEX_HTML, REPORT_PY, JS_EXEC_UNKNOWN, PY_EXPLAIN

JS_DECODE_ESTIMATE = "Math.round(computed.deviceBandwidth * 0.7 / (state.params * state.bytesPerParam))"
MD_CMD = "  report += `\\n## vLLM command\\n\\`\\`\\`\\n${emitted ? emitted.textContent : ''}\\n\\`\\`\\`\\n`;\n"

S = {}
S["S1 js: the copied report withholds the vLLM command for a card without constants"] = [
    (INDEX_HTML, MD_CMD, MD_CMD.replace("${emitted ? emitted.textContent : ''}", "${emitted && computed.throughputModelled ? emitted.textContent : ''}"), 1)]
S["S2 js: exec summary prints '~<borrowed> tok/s' only when a preset is selected"] = [
    (INDEX_HTML, JS_EXEC_UNKNOWN, JS_EXEC_UNKNOWN + f"    if (state.presetKey) html += `<div class=\"exec-row\"><span class=\"exec-label\">Speed per user</span><span class=\"exec-value\">~${{{JS_DECODE_ESTIMATE}}} tok/s</span></div>`;\n", 1)]
S["S3 py: generate() prints 'Rough decode: ~<n> tok/s per user' to stdout for a card without constants"] = [
    (REPORT_PY, PY_EXPLAIN, PY_EXPLAIN + "            print(f\"Rough decode: ~{c['single_tok'] or 0} tok/s per user\")\n", 1)]

if __name__ == "__main__":
    run_driver(S)
