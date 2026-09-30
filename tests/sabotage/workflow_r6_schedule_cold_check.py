#!/usr/bin/env python3
"""The first cold check of fix/price-refresh-off-the-hour (fc0c70e): the four
sabotages that survived it, kept so the next change starts where the check ended.

The schedule rule asks two things of the price workflow -- no cron that can fire
at minute 0, and a schedule at all -- and answers both by reading the minute field
and counting the crons. Every shape of minute 0 the check tried was refused (a
range from 0, a step over a star, 0/17, */60, 000, a second cron beside the good
one, a folded scalar, a new .yaml file), and so was every way of emptying the
schedule. What passed sits on either side of that rule:

- A cron whose minute is fine and which never fires (S1: February 31st), or fires
  monthly rather than weekly (S2). GitHub validates each field's range, not the
  calendar, so it accepts both; the rule sees one fixed minute and a schedule.
- A step-level `if:` on a step BEFORE the gate (S3 the fetch, S4 the diff), keyed
  to workflow_dispatch. The workflow is still scheduled, at minute 17, and every
  scheduled run does nothing: the fetch is skipped so nothing moves, or the diff
  is skipped so `changed` is never set and every delivery step is skipped. The
  tail rules pin the conditions of every step AFTER the suite; the head is
  unguarded, and the job-level rule sees only job-level conditions.

tests/corpus.test.py does go red on S1 and S2, because the cron line is an anchor
of workflow_r5_off_the_hour.py. That is drift, not detection: a contributor who
moves the cron moves the anchors with it, and the corpus check is not a judge.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver

WF = ".github/workflows/price-refresh.yml"
CRON = '    - cron: "17 6 * * 1"'
FETCH = "      - name: Fetch, validate, and (where confirmed or moved) apply\n        run: |"
DIFF = "      - name: Check whether there is anything to propose\n        id: diff\n        run: |"


def minute_kept(expr):
    return [(WF, CRON, f'    - cron: "{expr}"', 1)]


def gated(step):
    """The step carries a condition that is false on every scheduled run."""
    head, run = step.rsplit("\n", 1)
    return [(WF, step, f"{head}\n        if: github.event_name == 'workflow_dispatch'\n{run}", 1)]


S = {
 'S1 minute 17, on a day that never comes (February 31st)':  minute_kept("17 6 31 2 *"),
 'S2 minute 17, monthly rather than weekly':                 minute_kept("17 6 1 * *"),
 'S3 the fetch step runs only when dispatched by hand':      gated(FETCH),
 'S4 the diff step runs only when dispatched by hand':       gated(DIFF),
}

if __name__ == "__main__":
    run_driver(S)
