#!/usr/bin/env python3
"""Round 3 against tests/workflow.test.py. Four sabotages, every one of which
passed the round-2 guard.

All four are the same defect in different clothes: a rule that pointed at a
name instead of at the thing. It bound to the literal job key `price-check`
rather than to the job that opens a pull request; it read `id: suite` off
whichever step mentioned the suite's path rather than the one that runs it; it
never looked at job-level keys at all; and it read filenames with a regex that
a documented expression walks straight past.

The decoy (R03) is the one to remember. `echo "the suite lives at
tests/run.sh"` carrying `id: suite`, with the real step invoking "${RUNNER}.sh"
so it matches nothing — and in one edit the PR body starts reporting an echo's
outcome (always success) on every red run, while the pipefail rule goes vacuous
because an echo has no pipe. Two prior rounds' findings reinstated by
misdirecting identity rather than by touching either rule.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver

WF = ".github/workflows/price-refresh.yml"

SECOND_JOB = """
  price-check-live:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
      - name: Fetch, validate, and apply
        run: python3 tools/price_check.py --apply --report-out "$RUNNER_TEMP/r.md"
      - name: Run the suite
        id: suite
        run: ./tests/run.sh
      - name: Open a pull request only if the suite passed
        uses: peter-evans/create-pull-request@v6
        with:
          branch: automation/price-refresh
          title: "Price refresh: catalog prices moved"
"""

S = {
    "R01 a second job in the same file carries the original bug, unguarded":
        [(WF, "jobs:\n  price-check:", "jobs:" + SECOND_JOB + "  price-check:", 1)],
    "R02 a job-level if retires the whole job, reported as 'skipped'":
        [(WF, "  price-check:\n    runs-on: ubuntu-latest",
              "  price-check:\n    if: ${{ github.repository_owner == 'nobody' }}\n    runs-on: ubuntu-latest", 1)],
    "R03 a decoy echo takes id: suite, so the body reports its outcome forever":
        [(WF, "      - name: Run the full suite — recorded, not obeyed\n        id: suite\n",
              '      - name: Locate the suite entrypoint\n'
              "        id: suite\n"
              "        continue-on-error: true\n"
              '        run: echo "the suite lives at tests/run.sh"\n\n'
              "      - name: Run the full suite — recorded, not obeyed\n", 1),
         (WF, '          set -o pipefail\n          ./tests/run.sh 2>&1 | tee "$RUNNER_TEMP/suite.log"',
              '          RUNNER="./tests/run"\n          "${RUNNER}.sh" 2>&1 | tee "$RUNNER_TEMP/suite.log"', 1)],
    "R04 format() hides a filename nothing writes from the path rule":
        [(WF, "            ${{ runner.temp }}/suite.log",
              "            ${{ format('{0}/suite-output.log', runner.temp) }}", 1)],
    # Renaming price-check is NOT a defect any more — the guard finds the job by
    # what it does, so the key is free. The name that IS load-bearing is the other
    # one: a status check is named after its job, and master's protection requires
    # `suite`. That coupling lives in a GitHub setting, invisible from the tree.
    "R05 tests.yml's job renamed, retiring the required status check silently":
        [(".github/workflows/tests.yml", "jobs:\n  suite:", "jobs:\n  full-suite:", 1)],
}

if __name__ == "__main__":
    run_driver(S)
