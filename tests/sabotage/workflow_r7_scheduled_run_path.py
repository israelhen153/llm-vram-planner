#!/usr/bin/env python3
"""Round 7 against tests/workflow.test.py: the shapes the cold check's fixes must
catch beyond its four survivors (workflow_r6_schedule_cold_check.py).

The check found a schedule that never fires and one that fires monthly, both at a
good minute, and a step-level `if:` on the fetch and on the gate that makes every
scheduled run do nothing. The fixes closed classes, not instances: the price job's
one cron must fire once a week at one fixed time, and every step's condition is
pinned by its position, none up to the gate and the gate's own from there through
the suite. These are the neighbours of the four the check found. A schedule that
runs too often (weekdays, every six hours, a second weekly cron beside the first,
which force-pushes over goldens committed after the first run), and a condition on
the steps the check did not try (the checkout, the re-sync, and the install, image
and suite steps between the gate and the suite).
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver

WF = ".github/workflows/price-refresh.yml"
CRON = '    - cron: "17 6 * * 1"'
CRON_LINE = '    - cron: "17 6 * * 1" # Monday 06:17 UTC\n'
CHECKOUT = "      - uses: actions/checkout@v4\n"
RESYNC = "      - name: Re-sync generated blocks from data/gpus.json\n"
GATE_IF = "        if: steps.diff.outputs.changed == 'true'\n"
INSTALL = "      - name: Install what the image step and the test suite need\n" + GATE_IF
ASSETS = "      - name: Regenerate published images (index.html changed)\n" + GATE_IF
SUITE = "        id: suite\n" + GATE_IF


def cron(expr):
    return [(WF, CRON, f'    - cron: "{expr}"', 1)]


def after(anchor, line):
    return [(WF, anchor, anchor + line, 1)]


def swap(anchor, old, new):
    return [(WF, anchor, anchor.replace(old, new), 1)]


S = {
 # ---- W: the schedule runs more often than once a week, at a good minute ----
 'W1 every weekday rather than once a week':            cron("17 6 * * 1-5"),
 'W2 every six hours on Mondays':                       cron("17 */6 * * 1"),
 'W3 a second weekly cron beside the first':            after(CRON_LINE, '    - cron: "17 6 * * 4"\n'),

 # ---- P: a condition on a step the check did not try ----
 'P1 the checkout skipped on scheduled runs':           after(CHECKOUT, "        if: github.event_name != 'schedule'\n"),
 'P2 the re-sync runs only when dispatched by hand':    after(RESYNC, "        if: github.event_name == 'workflow_dispatch'\n"),
 'P3 the install step also asks how the run started':   swap(INSTALL, "if: steps", "if: github.event_name == 'workflow_dispatch' && steps"),
 'P4 the image step runs whether or not anything moved': swap(ASSETS, GATE_IF, ""),
 'P5 the suite runs always(), whatever the gate said':  swap(SUITE, "steps.diff.outputs.changed == 'true'", "always()"),
}

if __name__ == "__main__":
    run_driver(S)
