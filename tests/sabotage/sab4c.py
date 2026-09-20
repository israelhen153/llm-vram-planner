#!/usr/bin/env python3
"""Views fed from outside the harness: the copied report's command block (read
from a DOM node the JS harness stubs to empty), a leak gated on a selected
preset (probes never select one), and stdout from generate()."""
import os, sys
sys.dont_write_bytecode = True   # see the note in sab.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sab
from sab import IDX, GR, JS_EXEC_UNKNOWN, PY_EXPLAIN

EST = "Math.round(computed.deviceBandwidth * 0.7 / (state.params * state.bytesPerParam))"
MD_CMD = "  report += `\\n## vLLM command\\n\\`\\`\\`\\n${emitted ? emitted.textContent : ''}\\n\\`\\`\\`\\n`;\n"

S = {}
S["S1 js: the copied report withholds the vLLM command for a card without constants"] = [
    (IDX, MD_CMD, MD_CMD.replace("${emitted ? emitted.textContent : ''}", "${emitted && computed.throughputModelled ? emitted.textContent : ''}"), 1)]
S["S2 js: exec summary prints '~<borrowed> tok/s' only when a preset is selected"] = [
    (IDX, JS_EXEC_UNKNOWN, JS_EXEC_UNKNOWN + f"    if (state.presetKey) html += `<div class=\"exec-row\"><span class=\"exec-label\">Speed per user</span><span class=\"exec-value\">~${{{EST}}} tok/s</span></div>`;\n", 1)]
S["S3 py: generate() prints 'Rough decode: ~<n> tok/s per user' to stdout for a card without constants"] = [
    (GR, PY_EXPLAIN, PY_EXPLAIN + "            print(f\"Rough decode: ~{c['single_tok'] or 0} tok/s per user\")\n", 1)]

if __name__ == "__main__":
    names = list(S)
    print(f"{len(names)} sabotage(s)")
    sab.require_green_baseline()
    survived, unapplied = [], []
    for name in names:
        try:
            touched = sab.apply(S[name])
        except Exception as e:
            print(f"  !! {name}: could not apply: {e}")
            unapplied.append(name)
            sab.restore([IDX, GR])
            continue
        try:
            res = sab.run_suites()
        finally:
            sab.restore(touched)
        red = {k: v for k, v in res.items() if v[0] != 0}
        if not red:
            survived.append(name)
            print(f"  GREEN  {name}   <-- SURVIVED")
        else:
            bits = []
            for k, (rc, fails, errs, tally) in red.items():
                first = (fails or errs or ["(no FAIL line)"])[0]
                bits.append(f"{k}[{tally[1] if tally else '?'} failed: {first[:120]}]")
            print(f"  red    {name}\n         " + "\n         ".join(bits))
    print(f"\n{len(names) - len(survived) - len(unapplied)} caught, "
          f"{len(survived)} survived"
          + (f", {len(unapplied)} COULD NOT BE APPLIED" if unapplied else ""))
    for s in survived:
        print("  SURVIVED: " + s)
    if unapplied:
        # A sabotage that never reached the tree judged nothing. Counting it as
        # caught is how a drifted driver reports a clean run forever.
        sys.exit(2)
