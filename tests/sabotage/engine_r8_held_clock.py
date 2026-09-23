#!/usr/bin/env python3
"""The report suite's held clock (fix/report-test-clock).

tests/report.test.py builds one report more than once and compares the builds,
and the report prints the wall clock twice: the cover's "Generated <date> at
<time>" and the footer's date. Until the suite held its clock, a run that
crossed a minute between two builds went red with nothing wrong. On 2026-09-23
that turned a full corpus run's baseline red, and in a sabotage run a check
that goes red on its own reads as a catch.

Each sabotage here undoes the hold one way. The checks that compare two builds
would notice only once a minute, by chance. What catches these every time:
- the check that every report the suite built printed the held instant;
- the golden's scrubs, keyed on that instant's value rather than on a shape;
- the check that the scrubs hide nothing else.

K5 to K10 come from the cold check of the first version, which checked the
clock on one probe config and scrubbed the cover by its shape. A clock read
around the hold for FP8 reports only (K5) passed all 120 tests.
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
HELD_AT = "HELD = datetime(2001, 2, 3, 4, 5)\n"
COVER_SCRUB = '        (re.compile(re.escape(HELD_COVER)), "Generated [date] at [time]"),\n'
DATE_SCRUB = '        (re.compile(r"\\A" + re.escape(HELD_DATE) + r"\\Z"), "[today]"),\n'
EVERY_BUILD = ('test("every report this suite built printed the held clock, so no check depends on when it runs",\n'
               '     check_every_report_built_here_read_the_held_clock)\n')


def cover_around_hold(gate="True"):
    """The cover's clock read around the hold whenever `gate` holds in generate()."""
    return PY_CLOCK_COVER.replace(
        "datetime.now()", f"(__import__('datetime').datetime.now() if {gate} else datetime.now())")


def footer_around_hold(gate="True"):
    """The footer's date read around the hold whenever `gate` holds in _header_footer()."""
    return PY_CLOCK_FOOTER.replace(
        "datetime.now()", f"(__import__('datetime').datetime.now() if {gate} else datetime.now())")


S = {
    "K1 test: the hold is gone, so the report reads the machine's clock again":
        [(REPORT_TEST, HOLD, "", 1)],
    "K2 test: the golden scrubs the machine's date instead of the held one":
        [(REPORT_TEST, DATE_SCRUB, DATE_SCRUB.replace(
            "re.escape(HELD_DATE)", 're.escape(datetime.now().strftime("%Y-%m-%d"))'), 1)],
    "K3 py: the cover reads the machine's clock around the hold":
        [(REPORT_PY, PY_CLOCK_COVER, cover_around_hold(), 1)],
    "K4 py: the footer reads the machine's date around the hold":
        [(REPORT_PY, PY_CLOCK_FOOTER,
          PY_CLOCK_FOOTER.replace("datetime.now()", "__import__('datetime').date.today()"), 1)],
    # ---- the cold check of the first version ----
    "K5 py: the cover reads the machine's clock only for FP8 weights (the survivor)":
        [(REPORT_PY, PY_CLOCK_COVER, cover_around_hold("cfg.get('quant') == 'fp8'"), 1)],
    "K6 py: the footer reads the machine's clock only for FP8 weights":
        [(REPORT_PY, PY_CLOCK_FOOTER, footer_around_hold("self.cfg.get('quant') == 'fp8'"), 1)],
    "K7 py: the cover reads the machine's clock only on PCIe":
        [(REPORT_PY, PY_CLOCK_COVER, cover_around_hold("not cfg.get('nvlink')"), 1)],
    "K8 both: the cover reads around the hold and the every-build check is unregistered":
        [(REPORT_PY, PY_CLOCK_COVER, cover_around_hold(), 1),
         (REPORT_TEST, EVERY_BUILD, "", 1)],
    "K9 test: the hold is taken at the real time, so a read around it prints the same minute":
        [(REPORT_TEST, HELD_AT, "HELD = datetime.now()\n", 1)],
    "K10 test: the golden scrubs the cover by its shape again, hiding any clock":
        [(REPORT_TEST, COVER_SCRUB, COVER_SCRUB.replace(
            "re.compile(re.escape(HELD_COVER))", 're.compile(r"Generated \\w+ \\d+, \\d{4} at \\d{2}:\\d{2}")'), 1)],
}

if __name__ == "__main__":
    run_driver(S)
