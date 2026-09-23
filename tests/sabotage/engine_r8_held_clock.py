#!/usr/bin/env python3
"""The report suite's held clock (fix/report-test-clock).

tests/report.test.py builds one report more than once and compares the builds,
and the report prints the wall clock twice: the cover's "Generated <date> at
<time>" and the footer's date. Until the suite held its clock, a run that
crossed a minute between two builds went red with nothing wrong. On 2026-09-23
that turned a full corpus run's baseline red, and in a sabotage run a check
that goes red on its own reads as a catch.

Each sabotage here undoes the hold one way. The checks that compare two builds
would notice only once a minute, by chance; what has to catch these every time
is the check that the printed clocks are the held instant, and the golden's
scrub keyed on that instant rather than on the machine's clock.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import REPORT_PY, PY_CLOCK_COVER, PY_CLOCK_FOOTER

# The suite, not the engine, so anchored here the way the workflow drivers
# anchor on the workflow file: anchors.py holds excerpts of the engine.
REPORT_TEST = "tests/report.test.py"
HOLD = "gr.datetime = HeldClock\n"
SCRUB = '    today = HELD.strftime("%Y-%m-%d")\n'

S = {
    "K1 test: the hold is gone, so the report reads the machine's clock again":
        [(REPORT_TEST, HOLD, "", 1)],
    "K2 test: the golden scrubs the machine's date instead of the held one":
        [(REPORT_TEST, SCRUB, '    today = datetime.now().strftime("%Y-%m-%d")\n', 1)],
    "K3 py: the cover reads the machine's clock around the hold":
        [(REPORT_PY, PY_CLOCK_COVER,
          PY_CLOCK_COVER.replace("datetime.now()", '__import__("datetime").datetime.now()'), 1)],
    "K4 py: the footer reads the machine's date around the hold":
        [(REPORT_PY, PY_CLOCK_FOOTER,
          PY_CLOCK_FOOTER.replace("datetime.now()", '__import__("datetime").date.today()'), 1)],
}

if __name__ == "__main__":
    run_driver(S)
