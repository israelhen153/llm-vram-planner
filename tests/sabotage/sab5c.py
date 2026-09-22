#!/usr/bin/env python3
"""W11 again with the link on a word the reason-piece filter does not match on:
was the href read, or did the <a> merely split 'do not transfer' in the raw markup?"""
import os, sys
sys.dont_write_bytecode = True   # see the note in sab.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sab
from harness import run_driver
from sab import IDX, GR, JS_UNMODELLED_NOTE

EST = "Math.round(computed.deviceBandwidth * 0.7 / (state.params * state.bytesPerParam))"
S = {"W11b js: the figure rides in an href on 'memory-bandwidth' inside the reason": [
    (IDX, JS_UNMODELLED_NOTE, JS_UNMODELLED_NOTE.replace("has no published memory-bandwidth or compute utilisation", f"has no published <a href=\"https://example.com/why?est=${{{EST}}}\" style=\"color:inherit\">memory-bandwidth</a> or compute utilisation"), 1)]}

if __name__ == "__main__":
    run_driver(S)
