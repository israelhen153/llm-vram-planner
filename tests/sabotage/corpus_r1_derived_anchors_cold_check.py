#!/usr/bin/env python3
"""Corpus round 1: the cold check of chore/derive-catalog-anchors at 23fac94.

tests/corpus.test.py promises that a sabotage quoting a value the weekly price
job rewrites goes red in the pull request that adds it, that a sabotage whose
target is gone is refused by name rather than dropped or allowed to stop its
driver loading, and that the check refreshes the catalog the way the job does.
A checker with no prior context attacked those promises with sixteen edits to
the drivers and to the check itself. Ten were caught. These are the six the
suite stayed green on, kept so the next change to the check starts where this
one ended:

- Q1: a price quoted with nothing after it. The check's `moved` refresh adds one
  cent, and 12.3 is a substring of 12.31, so the quote still counts once. Six of
  the nineteen read tiers had such a price on 2026-09-29.
- Q2: the price of a tier held under a note (h100-80/spot). No refresh kind moves
  it: `confirmed` records a first reading at the catalog's own price, where the
  real job's first reading is MOVED as often as not — that tier's note says its
  Vast read came back 42.5% higher.
- Q3: a second precondition in the shell driver. The check reads only the first
  `assert line.count(...)`, so a price quoted in a second one is invisible.
- M1, M2: a driver's Missing handling regressed to an assert, or a sabotage
  dropped instead of refused. The check can only see a target vanish when a
  refresh takes it out, and the only such target is the one held note, so every
  other driver's handling of a gone row, tier or reading is unchecked.
- B1: the check blinded so `moved` moves nothing. Its own guard is that the
  refreshed file differs, which the re-read dates satisfy on their own.

What these sabotages edit is the corpus itself: a driver, the shell driver, or
tests/corpus.test.py. The suites that judge an engine sabotage cannot see any of
that, and corpus.test.py, the one that can, is excluded from them because every
engine sabotage turns it red. So this driver names its own judge, corpus.test.py
alone, through run_driver()'s `judges`. A sabotage here reads GREEN until the
check learns to see it, and that is the finding, not a pass. Every quoted value
is read from data/gpus.json as this driver loads, so the sabotages keep saying
what they said after the job's next run; a tier they need and cannot find is a
harness.Missing, refused by name.

Usage: python3 tests/sabotage/corpus_r1_derived_anchors_cold_check.py [name-substring ...]
"""
import json, os, re, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
from harness import run_driver, Missing, ROOT
from anchors import GPUS_JSON
import price_check

R4 = "tests/sabotage/engine_r4_cost_provenance.py"
R10 = "tests/sabotage/engine_r10_rocm_guidance.py"
SHELL = "tests/sabotage/engine_r1_perfkey_typo.sh"
CHECK = "tests/corpus.test.py"
JUDGES = [("corpus", ["python3", "tests/corpus.test.py"])]

with open(os.path.join(ROOT, GPUS_JSON), encoding="utf-8") as f:
    TEXT = f.read()
ROWS = json.loads(TEXT)["data"]
# The tiers the job reads, from its own SOURCE_MAP, split as tests/corpus.test.py splits
# them: read (carrying a reading) and held (under a note, no reading yet).
AUTOMATED = [(s, t) for s, tiers in price_check.SOURCE_MAP.items() for t, cfg in tiers.items() if "primary" in cfg]
READ = [(s, t) for s, t in AUTOMATED if t in (ROWS.get(s, {}).get("priceSource") or {})]
HELD = [(s, t) for s, t in AUTOMATED if (s, t) not in READ and t in (ROWS.get(s, {}).get("priceNote") or {})]
H100 = ROWS.get("h100-80") or {}

# The line S5 reads its price on, and where a sabotage is added to engine_r4.
S5_READ = '(GPUS_JSON, row + json.dumps(price) + ",", row + json.dumps(moved) + ",", 1)]'
R4_ADD_AT = "S = {}\n"


def undelimited_quote():
    """A read tier's price quoted with nothing after it — preferring one that a cent up
    still begins with, since that is the quote the check's moved refresh misses."""
    candidates = []
    for s, t in READ:
        price = ROWS[s][t]
        quote = f'"{t}": {json.dumps(price)}'
        if isinstance(price, (int, float)) and TEXT.count(quote) == 1:
            stable = json.dumps(round(price + 0.01, 2)).startswith(json.dumps(price))
            candidates.append((not stable, s, t, quote))
    if not candidates:
        return None
    _, s, t, quote = sorted(candidates)[0]
    return s, t, quote


S = {}

# ---- Q: a value the job rewrites, quoted where the refresh check cannot see it ----
found = undelimited_quote()
if found:
    s, t, quote = found
    doubled = quote[:quote.index(":") + 2] + json.dumps(round(ROWS[s][t] * 2, 2))
    S[f"Q1 r4: a sabotage quotes {s}/{t}'s price ({quote}) with nothing after it, and a cent up it is still a prefix"] = [
        (R4, R4_ADD_AT, R4_ADD_AT + f'S["X1 data: {s}/{t} moved"] = [(GPUS_JSON, {quote!r}, {doubled!r}, 1)]\n', 1)]
else:
    S["Q1 r4: a sabotage quotes a read tier's price with nothing after it"] = [
        (R4, Missing("no automated tier carries a reading whose price occurs once in the catalog"), "", 1)]

# h100-80/spot, the tier held under a note on 2026-09-29, by name: once its note is confirmed
# away this has no held price to quote and says so, with the note the refusal is about, as
# engine_r5's C7 does. Another tier held later is not this sabotage's.
if ("h100-80", "spot") in HELD and isinstance(H100.get("spot"), (int, float)):
    quote = f'"spot": {json.dumps(H100["spot"])}, "tflops": {json.dumps(H100["tflops"])}'
    S["Q2 r4: a sabotage quotes h100-80/spot's price, held under a note, which no refresh kind moves"] = [
        (R4, R4_ADD_AT, R4_ADD_AT + 'S["X2 data: h100-80/spot moved while its note still says the read is held"] = '
         f'[(GPUS_JSON, {quote!r}, {quote.replace(json.dumps(H100["spot"]), "9.99", 1)!r}, 1)]\n', 1)]
else:
    S["Q2 r4: a sabotage quotes h100-80/spot's price, held under a note, which no refresh kind moves"] = [
        (R4, Missing("h100-80/spot is no longer held under a note (priceNote.spot), so there is no held "
                     "price to quote", note=("h100-80", "spot")), "", 1)]

shell_src = open(os.path.join(ROOT, SHELL), encoding="utf-8").read()
loop = re.search(r"^for slug in ([\w .-]+); do$", shell_src, re.M)
FIRST_ASSERT = "assert line.count('\"perfKey\": \"nvidia\"') == 1, f\"{slug}: expected exactly one perfKey\"\n"
if loop and all(isinstance(ROWS.get(slug, {}).get("hyper"), (int, float)) for slug in loop.group(1).split()):
    prices = " or ".join(f"line.count('\"hyper\": {json.dumps(ROWS[slug]['hyper'])},') == 1" for slug in loop.group(1).split())
    S["Q3 sh: the shell driver gains a second precondition quoting its rows' prices, which the check never reads"] = [
        (SHELL, FIRST_ASSERT, FIRST_ASSERT + f'assert {prices}, f"{{slug}}: price precondition"\n', 1)]
else:
    S["Q3 sh: the shell driver gains a second precondition quoting its rows' prices"] = [
        (SHELL, Missing("the shell driver's loop, or a hyperscaler price on one of its rows, is gone"), "", 1)]

# ---- M: a target gone, and the driver dropping the sabotage or failing to load ----
S["M1 r10: row_edit() asserts instead of carrying Missing, so a gone row stops the driver loading"] = [
    (R10, "    if line.count(old) != 1:\n        return (GPUS_JSON, Missing(f\"{slug}: {old!r} occurs {line.count(old)} times in its row\"), \"\", 1)\n",
     "    assert line.count(old) == 1, f\"{slug}: {old!r} occurs {line.count(old)} times in its row\"\n", 1)]
S["M2 r4: S5 dropped, not refused, when h100-80 carries no hyperscaler reading"] = [
    (R4, "else:\n    S[\"S5 data: h100-80/hyper moved with priceSource.price left where it was\"] = [NO_READING]\n",
     "else:\n    pass\n", 1)]

# ---- B: the check blinded ----
if isinstance(H100.get("hyper"), (int, float)):
    price = json.dumps(H100["hyper"])
    S["B1 check: the moved refresh moves every price by nothing, and S5 quotes h100-80's price with its comma"] = [
        (CHECK, 'price = row[tier] if kind == "reread" else round(row[tier] + 0.01, 2)',
         'price = row[tier] if kind == "reread" else round(row[tier] + 0.00, 2)', 1),
        (R4, S5_READ, f'(GPUS_JSON, row + "{price},", row + "{json.dumps(round(H100["hyper"] + 0.2, 2))},", 1)]', 1)]
else:
    S["B1 check: the moved refresh moves every price by nothing, and S5 quotes h100-80's price with its comma"] = [
        (R4, Missing("h100-80 carries no hyperscaler price to quote"), "", 1)]

if __name__ == "__main__":
    run_driver(S, judges=JUDGES)
