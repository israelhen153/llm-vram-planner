#!/usr/bin/env python3
"""Loose ends: an attribute figure on an element the CARD split does not expose,
and a no-number speed claim inside the PDF explanation paragraph."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sab
from sab import IDX, GR, JS_EXEC_UNKNOWN, JS_CMP_UNKNOWN

EST = "Math.round(computed.deviceBandwidth * 0.7 / (state.params * state.bytesPerParam))"

S = {}
S["N11b js: data/title attributes with the borrowed figure on the exec row's inner span"] = [
    (IDX, JS_EXEC_UNKNOWN, JS_EXEC_UNKNOWN.replace("<span class=\"exec-value\">Not modelled", f"<span class=\"exec-value\" data-estimate=\"${{{EST}}}\" title=\"about ${{{EST}}} per user\">Not modelled"), 1)]
S["N11c js: the same attributes on the comparison card's value span"] = [
    (IDX, JS_CMP_UNKNOWN, JS_CMP_UNKNOWN.replace("<span class=\"val\">not modelled", "<span class=\"val\" title=\"about ${Math.round(c.deviceBandwidth * 0.7 / (s.params * s.bytesPerParam))} per user\">not modelled"), 1)]
S["Q17 py: explanation ends 'This card should be fast enough for interactive chat.'"] = [
    (GR, "                \"depend on them.\",\n                self.styles[\"Small\"]\n", "                \"depend on them. This card should be fast enough for interactive chat.\",\n                self.styles[\"Small\"]\n", 1)]

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
