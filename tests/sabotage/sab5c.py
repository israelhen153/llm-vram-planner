#!/usr/bin/env python3
"""W11 again with the link on a word the reason-piece filter does not match on:
was the href read, or did the <a> merely split 'do not transfer' in the raw markup?"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sab
from sab import IDX, GR, JS_UNMODELLED_NOTE

EST = "Math.round(computed.deviceBandwidth * 0.7 / (state.params * state.bytesPerParam))"
S = {"W11b js: the figure rides in an href on 'memory-bandwidth' inside the reason": [
    (IDX, JS_UNMODELLED_NOTE, JS_UNMODELLED_NOTE.replace("has no published memory-bandwidth or compute utilisation", f"has no published <a href=\"https://example.com/why?est=${{{EST}}}\" style=\"color:inherit\">memory-bandwidth</a> or compute utilisation"), 1)]}

if __name__ == "__main__":
    # Same shape as every other driver on purpose. This loop was written separately
    # and drifted: it had no baseline guard, no summary line and no exit code, so
    # whether a property held depended on which driver you happened to run.
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
            for k, (rc, fails, errs, tally) in red.items():
                print(f"  red    {name}\n         {k}[{tally[1] if tally else '?'} failed: {(fails or errs or ['?'])[0][:150]}]")
    print(f"\n{len(names) - len(survived) - len(unapplied)} caught, "
          f"{len(survived)} survived"
          + (f", {len(unapplied)} COULD NOT BE APPLIED" if unapplied else ""))
    if unapplied:
        sys.exit(2)
