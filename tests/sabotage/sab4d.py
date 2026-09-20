#!/usr/bin/env python3
"""The identity test exempts a piece that changes when the engine's throughput
figures are moved. Give the max-batch card an invisible dependency on one
(a trailing space keyed on computeBound), then withhold the card for the
unknown card."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sab
from sab import IDX, GR, JS_TP_RETURN

MAXBATCH = "  const maxBatchCard = `<div class=\"reverse-card\"><p class=\"label\">Max batch at this context</p><p class=\"value\">${computed.maxBatchByKV} seq</p><p class=\"sub\">${computed.batchLimitedByKV ? `KV cache caps you below ${state.concurrency} requested` : 'KV cache has room for your concurrency'}</p></div>`;\n"

S = {}
S["T1 js: max-batch card withheld for the unknown card, exempted by an invisible computeBound dependency"] = [
    (IDX, MAXBATCH, MAXBATCH.replace("</p></div>`;\n", "</p>${computed.computeBound ? ' ' : ''}</div>`;\n"), 1),
    (IDX, JS_TP_RETURN, JS_TP_RETURN.replace("</div>${maxBatchCard}</div>${queueWarning}", "</div></div>${queueWarning}"), 1)]
S["T1b js (control): max-batch card withheld without the exemption trick"] = [
    (IDX, JS_TP_RETURN, JS_TP_RETURN.replace("</div>${maxBatchCard}</div>${queueWarning}", "</div></div>${queueWarning}"), 1)]

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
