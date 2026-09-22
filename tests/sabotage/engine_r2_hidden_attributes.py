#!/usr/bin/env python3
"""Loose ends: an attribute figure on an element the CARD split does not expose,
and a no-number speed claim inside the PDF explanation paragraph."""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import INDEX_HTML, REPORT_PY, JS_EXEC_UNKNOWN, JS_CMP_UNKNOWN

JS_DECODE_ESTIMATE = "Math.round(computed.deviceBandwidth * 0.7 / (state.params * state.bytesPerParam))"

S = {}
S["N11b js: data/title attributes with the borrowed figure on the exec row's inner span"] = [
    (INDEX_HTML, JS_EXEC_UNKNOWN, JS_EXEC_UNKNOWN.replace("<span class=\"exec-value\">Not modelled", f"<span class=\"exec-value\" data-estimate=\"${{{JS_DECODE_ESTIMATE}}}\" title=\"about ${{{JS_DECODE_ESTIMATE}}} per user\">Not modelled"), 1)]
S["N11c js: the same attributes on the comparison card's value span"] = [
    (INDEX_HTML, JS_CMP_UNKNOWN, JS_CMP_UNKNOWN.replace("<span class=\"val\">not modelled", "<span class=\"val\" title=\"about ${Math.round(c.deviceBandwidth * 0.7 / (s.params * s.bytesPerParam))} per user\">not modelled"), 1)]
S["Q17 py: explanation ends 'This card should be fast enough for interactive chat.'"] = [
    (REPORT_PY, "                \"depend on them.\",\n                self.styles[\"Small\"]\n", "                \"depend on them. This card should be fast enough for interactive chat.\",\n                self.styles[\"Small\"]\n", 1)]

if __name__ == "__main__":
    run_driver(S)
