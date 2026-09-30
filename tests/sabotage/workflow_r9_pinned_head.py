#!/usr/bin/env python3
"""Round 9 against tests/workflow.test.py: the neighbours of the second cold
check's survivors (workflow_r8_schedule_cold_check_2.py), which its fixes close as
classes.

The check left the job scheduled and every rule green while each scheduled run
proposed nothing: through the fetch's own command, a step before the gate, the
runner or a job the price job needs, a second trigger, and `on:` given twice. The
fixes pinned the head of the job whole, the shape around it, the triggers, and a
reader that refuses a key given twice. These are the shapes the check did not try
that the same fixes must catch: a new step before the gate, a `shell:` or `env:`
on the fetch or a `shell:` on the gate, a checkout of another ref, another Python,
`defaults:`, `env:` or `container:` around the job, another trigger, no hand run,
and a key given twice inside one step.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver

WF = ".github/workflows/price-refresh.yml"
CHECKOUT = "      - uses: actions/checkout@v4\n"
PYTHON = '          python-version: "3.x"\n'
FETCH = "      - name: Fetch, validate, and (where confirmed or moved) apply\n"
GATE = "      - name: Check whether there is anything to propose\n        id: diff\n"
RUNS_ON = "    runs-on: ubuntu-latest\n"
JOBS_HEAD = "jobs:\n  price-check:\n"
DISPATCH = "  workflow_dispatch: {} # manual run, e.g. to check a single source sooner\n"


def after(anchor, text):
    return [(WF, anchor, anchor + text, 1)]


def before(anchor, text):
    return [(WF, anchor, text + anchor, 1)]


def swap(old, new):
    return [(WF, old, new, 1)]


S = {
 # ---- H: the head of the job, which is pinned whole ----
 'H1 a new step before the gate discards the moves on schedule':
     before(GATE, "      - name: Tidy\n        run: '[ \"$GITHUB_EVENT_NAME\" = schedule ] && git checkout -- . || true'\n\n"),
 'H2 the fetch runs under a shell that ignores its script':     after(FETCH, '        shell: "true {0}"\n'),
 'H3 the fetch carries an env: that changes what it does':      after(FETCH, "        env:\n          PYTHONPATH: /tmp/stub\n"),
 'H4 the gate runs under a shell that ignores its script':      after(GATE, '        shell: "true {0}"\n'),
 'H5 the checkout takes another ref':                           after(CHECKOUT, "        with:\n          ref: v1.0\n"),
 'H6 another Python':                                           swap(PYTHON, '          python-version: "3.8"\n'),

 # ---- E: the shape around the job ----
 'E1 defaults: gives every step a shell that ignores its script':
     before(JOBS_HEAD, "defaults:\n  run:\n    shell: 'true {0}'\n\n"),
 'E2 the job carries an env: that reaches every step':          after(RUNS_ON, "    env:\n      PYTHONPATH: /tmp/stub\n"),
 'E3 the job runs in a container with another Python':          after(RUNS_ON, "    container: python:3.8\n"),

 # ---- T: the triggers ----
 'T1 workflow_run starts it after every Tests run':
     after(DISPATCH, "  workflow_run:\n    workflows: [Tests]\n    types: [completed]\n"),
 'T2 no hand run, so a dropped Monday cannot be rescued':        swap(DISPATCH, ""),

 # ---- Y: a key given twice, which GitHub refuses and pyyaml keeps the last of ----
 'Y1 the fetch step given two run: keys, the first a no-op':    after(FETCH, "        run: echo skipped\n"),
}

if __name__ == "__main__":
    run_driver(S)
