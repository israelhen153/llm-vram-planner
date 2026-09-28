#!/usr/bin/env python3
"""The third cold check of the schedule fix (011e8f2): the ten sabotages that
survived it, kept so the next change starts where the check ended.

The rules the last two rounds added hold where they look. Refused, by name of the
test that refused it: a new workflow file (.yml or .yaml) scheduled at minute 0
and tests.yml on `*/15` ("no workflow is scheduled at the start of the hour",
which also refuses `@weekly`, a block-scalar cron, a schedule written as a
mapping, a bare number and a plain string); `*/7` in day-of-month, Mondays in
June only, Sunday as 7 and `1-31` in day-of-month ("once a week, at one fixed
time"); a new step after the gate and a second checkout keyed to the event
("the same path as a manual one", "one of the two permitted conditions");
`timeout-minutes`, `environment:` or a workflow-level `env:` ("runs alone ... with
nothing around it"); an empty matrix on schedule ("retired by a condition");
`--cached` in the gate and `changed=true ` with a trailing space ("the gate still
asks"); `changed == true` unquoted (GitHub coerces the string to NaN); `add-paths`
behind an expression or narrowed; `workflow_call`; and a discard appended to the
suite step, whose run block is read. What passed sits outside what the rules read:

- The COMMANDS of the steps between the gate and the PR step (D1-D5). The head
  is pinned whole and the suite step's script is checked, but the install, image
  and body steps' scripts are not: `git checkout -- .` or `git stash` after the
  install or the image step — on scheduled runs, or on every run — restores the
  tree before peter-evans/create-pull-request looks, and "if there are no changes
  ... no pull request will be created and the action exits silently" (its
  README). The body step can `rm` the report instead, and the PR step then fails
  on its `body-path`: "throw new Error(`File '${inputs.bodyPath}' does not
  exist.`)" (src/main.ts, v6).
- A NEW step after the suite wearing one of the two permitted conditions (S1-S3).
  The condition rule checks what a step's `if:` says, not whether the step was
  there: a run step under `!cancelled() && changed == 'true'` that discards the
  moves on schedule, a second actions/checkout under the same condition — whose
  `clean` input "execute[s] `git clean -ffdx && git reset --hard HEAD` before
  fetching" (default true) — or a step after the PR step under `always()` that
  closes the PR on scheduled runs with `gh pr close automation/price-refresh
  --delete-branch` (gh ships on ubuntu-latest, and the job's token has
  pull-requests: write). Every scheduled run then proposes nothing, green.
- A key GitHub's parser does not define (K1, K2). pyyaml reads it and the rules
  never look at it; GitHub's TemplateReader refuses a mapping key that is not
  one of a definition's properties when the definition has no loose-key-type
  ("m_context.Error(nextKey, TemplateStrings.UnexpectedValue(nextKey.Value))",
  actions/runner, src/Sdk/DTObjectTemplating/ObjectTemplating/TemplateReader.cs),
  and the workflow is invalid: it fires neither on its schedule nor by hand.
  `concurrency-mapping` (src/Sdk/WorkflowParser/workflow-v1.0.json) lists `group`
  and `cancel-in-progress` and no loose key under every reader in that file, so
  K2 is refused for certain. The schedule item is `cron-mapping` (strict) under
  `workflow-root-strict` and `schedule-item-relaxed` (loose) under
  `workflow-root-new`; which of the two the trigger evaluator applies is not
  verified, so K1 is either a workflow that never fires or a key that is ignored.

Also found, and not sabotages:

- Legitimate edits the guard refuses: `python-version: "3.13"`, `node-version:
  "22"` and the fetch's `run: |` block folded to one line, all by "the head of
  the job, up to the gate, is pinned whole"; `17 6 * * mon` and `17 6 * 1-12 1`
  by "once a week, at one fixed time" (GitHub documents SUN-SAT; lowercase and
  a full-range month are unverified against GitHub, so these two are arguable).
- Moving every action to a newer major version (checkout@v5, setup-python@v6,
  setup-node@v5, create-pull-request@v7, upload-artifact@v5), or pinning one to
  a commit SHA, passes tests/workflow.test.py but turns tests/corpus.test.py red:
  workflow_r2, r3, r7, r8 and r9 quote the `@v4`/`@v6` lines. That is the corpus
  test's design, and it means the edit is not green until the anchors move with
  it. The same drift, not the guard, is what caught S2 when it was first tried as
  a second `actions/checkout@v4` line; it is kept here as `@v4.2.2`.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver

WF = ".github/workflows/price-refresh.yml"
CRON_LINE = '    - cron: "17 6 * * 1" # Monday 06:17 UTC\n'
CANCEL_LINE = "  cancel-in-progress: false\n"
INSTALL_RUN = "        run: pip install pillow websockets reportlab pyyaml\n"
IMAGE_RUN = "        run: python3 tools/make_assets.py\n"
BODY_END = '          } >> "$REPORT"\n'
BODY_STEP = "      - name: Tell the PR body what the suite did\n"
UPLOAD_STEP = "      - name: Upload the report, whatever happened\n"
PERMITTED = "        if: ${{ !cancelled() && steps.diff.outputs.changed == 'true' }}\n"
# Exits 0 always; discards every applied move on scheduled runs only.
DISCARD_ON_SCHEDULE = '[ "$GITHUB_EVENT_NAME" != schedule ] || git checkout -- .'


def after(anchor, text):
    return [(WF, anchor, anchor + text, 1)]


def before(anchor, text):
    return [(WF, anchor, text + anchor, 1)]


def swap(old, new):
    return [(WF, old, new, 1)]


S = {
 # ---- D: the commands between the gate and the PR step, which no rule reads ----
 'D1 the install step discards the moves on scheduled runs':
     swap(INSTALL_RUN, "        run: pip install pillow websockets reportlab pyyaml && { " + DISCARD_ON_SCHEDULE + "; }\n"),
 'D2 the image step discards the moves on scheduled runs':
     swap(IMAGE_RUN, "        run: python3 tools/make_assets.py && { " + DISCARD_ON_SCHEDULE + "; }\n"),
 'D3 the install step discards the moves on every run':
     swap(INSTALL_RUN, "        run: pip install pillow websockets reportlab pyyaml && git checkout -- .\n"),
 'D4 the install step stashes the moves on scheduled runs':
     swap(INSTALL_RUN, '        run: pip install pillow websockets reportlab pyyaml && { [ "$GITHUB_EVENT_NAME" != schedule ] || git stash; }\n'),
 'D5 the body step removes the report on schedule, so body-path names no file':
     after(BODY_END, '          [ "$GITHUB_EVENT_NAME" != schedule ] || rm -f "$REPORT"\n'),

 # ---- S: a new step after the suite, wearing a permitted condition ----
 'S1 a run step after the suite, permitted condition, discards the moves on schedule':
     before(BODY_STEP, "      - name: Tidy\n" + PERMITTED + "        run: '" + DISCARD_ON_SCHEDULE + "'\n\n"),
 'S2 a second checkout after the suite, permitted condition, wipes the moves on every run':
     before(BODY_STEP, "      - uses: actions/checkout@v4.2.2\n" + PERMITTED + "\n"),
 'S3 a step after the PR step closes the PR on scheduled runs':
     before(UPLOAD_STEP, "      - name: Close\n        if: always()\n        env:\n          GH_TOKEN: ${{ github.token }}\n"
            "        run: '[ \"$GITHUB_EVENT_NAME\" != schedule ] || gh pr close automation/price-refresh --delete-branch'\n\n"),

 # ---- K: a key GitHub's parser does not define, which pyyaml reads ----
 'K1 a key beside cron (timezone), refused under the strict reader':
     after(CRON_LINE, "      timezone: Europe/London\n"),
 'K2 a key inside concurrency, refused under every reader':
     after(CANCEL_LINE, "  timezone: UTC\n"),
}

if __name__ == "__main__":
    run_driver(S)
