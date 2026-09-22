#!/usr/bin/env python3
"""Round 2 against .github/workflows/price-refresh.yml and tests/workflow.test.py.

Every sabotage here passed the round-1 guard. They are why that guard was
rewritten on a real YAML parser and why three of its rules changed scope.

Two classes, and the first is the one worth remembering. The round-1 reader
split the file on fixed indentation to keep the suite at one third-party
package. But a plain scalar continued onto a second line folds into ONE
expression for YAML and for GitHub, while the reader saw only the first
physical line — so a delivery condition could be gated on the suite in its
second line and pass. And a step spelled `-   name:` was invisible to the
reader and to the count check meant to catch that, because both were the same
regex: a completeness check derived from the parser it checks can never see a
shape that parser drops. A guard whose job is to know what GitHub will do with
a file cannot read it differently from YAML.

The second class is scope rather than parsing: a rule aimed at the tail while
the real gate sat upstream of it, a rule that said "an id" where it meant a
particular id, one that said "mentioned" where it meant "written", and one
word — conclusion for outcome — that makes the PR body report every red suite
as a pass with nothing anywhere to contradict it.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in sab.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sab
from harness import run_driver

WF = ".github/workflows/price-refresh.yml"
PR_IF = "        if: ${{ !cancelled() && steps.diff.outputs.changed == 'true' }}\n        uses: peter-evans/create-pull-request@v6"
UP_IF = "        if: always()\n        uses: actions/upload-artifact@v4"
BODY = "      - name: Tell the PR body what the suite did\n        id: body\n        if: ${{ !cancelled() && steps.diff.outputs.changed == 'true' }}"

S = {
    # --- the reader read the file differently from YAML ---------------------
    "S01 the PR condition is gated on the suite in a folded second line":
        [(WF, PR_IF, "        if: ${{ !cancelled() && steps.diff.outputs.changed == 'true'\n"
                     "          && steps.suite.outcome == 'success' }}\n"
                     "        uses: peter-evans/create-pull-request@v6", 1)],
    "S02 the upload reverts to quiet-runs-only in a folded second line":
        [(WF, UP_IF, "        if: always()\n"
                     "          && steps.diff.outputs.changed == 'false'\n"
                     "        uses: actions/upload-artifact@v4", 1)],
    "S09 a comment carries the words the rule looked for":
        [(WF, PR_IF, "        if: steps.diff.outputs.changed == 'true' # ${{ !cancelled() }}\n"
                     "        uses: peter-evans/create-pull-request@v6", 1)],
    "S08 the gated step is respelled so the reader drops it and its own count agrees":
        [(WF, BODY, "      -   name: Tell the PR body what the suite did\n"
                    "          if: ${{ !cancelled() && steps.diff.outputs.changed == 'true'\n"
                    "            && steps.suite.outcome == 'success' }}", 1)],
    "S14 !cancelled() unwrapped, which YAML rejects outright":
        [(WF, PR_IF, "        if: !cancelled() && steps.diff.outputs.changed == 'true'\n"
                     "        uses: peter-evans/create-pull-request@v6", 1)],

    # --- the rule was aimed at the wrong thing ------------------------------
    "S03 delivery gated through an object filter that never names the suite":
        [(WF, PR_IF, "        if: ${{ !cancelled() && steps.diff.outputs.changed == 'true'"
                     " && !contains(steps.*.outcome, 'failure') }}\n"
                     "        uses: peter-evans/create-pull-request@v6", 1)],
    "S16 delivery gated on job.status, which names no step at all":
        [(WF, PR_IF, "        if: ${{ !cancelled() && steps.diff.outputs.changed == 'true'"
                     " && job.status == 'success' }}\n"
                     "        uses: peter-evans/create-pull-request@v6", 1)],
    "S04 a test run inserted upstream of the gate, outside the tail entirely":
        [(WF, "      - name: Check whether there is anything to propose",
              "      - name: Sanity-check the golden before diffing\n"
              "        run: node tests/model.test.js\n\n"
              "      - name: Check whether there is anything to propose", 1)],
    "S05 the body is fed .conclusion, so a red suite reports as a pass":
        [(WF, "          SUITE: ${{ steps.suite.outcome }}",
              "          SUITE: ${{ steps.suite.conclusion }}", 1)],
    "S15 the suite step keeps an id, but not the one the body reads":
        [(WF, "        id: suite\n", "        id: tests\n", 1)],
    "S10 the PR step swallows its own failure":
        [(WF, PR_IF, PR_IF + "\n        continue-on-error: true", 1)],
    "S11 the artifact's suite log is read by a step but written by none":
        [(WF, '| tee "$RUNNER_TEMP/suite.log"', '| tee "$RUNNER_TEMP/suite-output.log"', 1)],

    # --- pipefail: present, but not in effect -------------------------------
    "S06 pipefail set after the pipe it was meant to govern":
        [(WF, '          set -o pipefail\n          ./tests/run.sh 2>&1 | tee "$RUNNER_TEMP/suite.log"',
              '          ./tests/run.sh 2>&1 | tee "$RUNNER_TEMP/suite.log"\n          set -o pipefail', 1)],
    "S06b the suite's exit status discarded outright":
        [(WF, '| tee "$RUNNER_TEMP/suite.log"', '| tee "$RUNNER_TEMP/suite.log" || true', 1)],

    # --- the bot approving its own change -----------------------------------
    "S12 the golden reached add-paths through a wildcard, with a step to regenerate it":
        [(WF, "            assets/\n", "            assets/\n            tests/*/page.json\n", 1),
         (WF, "          ./tests/run.sh 2>&1", "          UPDATE_GOLDEN=1 node tests/model.test.js\n          ./tests/run.sh 2>&1", 1)],
    "S13 the golden committed by hand before the PR action runs":
        [(WF, "      - name: Open a pull request (never pushes to master)",
              "      - name: Record the new display strings\n"
              "        if: ${{ !cancelled() && steps.diff.outputs.changed == 'true' }}\n"
              "        run: |\n"
              "          git add tests/golden\n"
              "          git commit -m 'regen golden'\n\n"
              "      - name: Open a pull request (never pushes to master)", 1)],
}

if __name__ == "__main__":
    run_driver(S)
