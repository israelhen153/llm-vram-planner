#!/usr/bin/env python3
"""Two more: a borrowed-constant estimate shown identically for both cards
inside a MAY_DIFFER element (so it is 'text the card with constants shows'),
and a figure carried only in an href."""
import os, sys
sys.dont_write_bytecode = True   # see the note in sab.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sab
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
