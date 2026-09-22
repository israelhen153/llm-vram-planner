#!/usr/bin/env python3
"""Two more: a borrowed-constant estimate shown identically for both cards
inside a MAY_DIFFER element (so it is 'text the card with constants shows'),
and a figure carried only in an href."""
import os, sys
sys.dont_write_bytecode = True   # see the note in sab.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sab
from harness import run_driver
from sab import IDX, GR, JS_TP_TILE, JS_UNMODELLED_NOTE

EST = "Math.round(computed.deviceBandwidth * 0.7 / (state.params * state.bytesPerParam))"
# The known path's first tile, so the same sentence can be added to it.
KNOWN_FIRST = "<p class=\"label\">Single-stream decode (1 user)</p>"
QUEUE_ANCHOR = "  html += queueWarning;\n\n  /* Benchmark panel."

S = {}
S["X2 js: 'Bandwidth-only guess: ~<borrowed> tokens each second' shown for every card, in the throughput panel"] = [
    # after the known path's tiles and inside the unknown branch alike
    (IDX, QUEUE_ANCHOR, f"  html += queueWarning + `<div class=\"reverse-card\"><p class=\"label\">Bandwidth-only guess</p><p class=\"value\">~${{{EST}}} tokens each second</p></div>`;\n\n  /* Benchmark panel.", 1),
    (IDX, "${maxBatchCard}</div>${queueWarning}`);\n    return;\n", f"${{maxBatchCard}}</div>${{queueWarning}}<div class=\"reverse-card\"><p class=\"label\">Bandwidth-only guess</p><p class=\"value\">~${{{EST}}} tokens each second</p></div>`);\n    return;\n", 1)]
S["W11 js: the figure rides in an href on the reason's own words"] = [
    (IDX, JS_UNMODELLED_NOTE, JS_UNMODELLED_NOTE.replace("which do not transfer</span>`;", f"which do not <a href=\"https://example.com/why?est=${{{EST}}}\" style=\"color:inherit\">transfer</a></span>`;"), 1)]

if __name__ == "__main__":
    run_driver(S)
