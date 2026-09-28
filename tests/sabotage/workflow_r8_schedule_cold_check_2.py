#!/usr/bin/env python3
"""The second cold check of the schedule fix (514f3e7): the ten sabotages that
survived it, kept so the next change starts where the check ended.

The rules the fix added hold where they look. Every shape of minute 0 from the
earlier rounds, a value/step minute (17/20), a range/step minute (1-59/17), both
calendar fields restricted (17 6 1 * 1), an hour of 24 and a six-field cron were
all refused; so were a job-level `if:` in a new spelling (github.event.schedule),
the gate moved above the fetch, a gate that writes changed=yes, the job's
permissions narrowed, and a second workflow file carrying the price job. What
passed sits outside what the rules read:

- The fetch step's COMMAND is not pinned (F1-F4). Its position rule is satisfied
  because the step carries no `if:`; the run text can still choose `--apply` by
  an expression on the event name, wrap the whole command in a shell `if` on
  GITHUB_EVENT_NAME, drop `--apply`, or `--slug` a scheduled run down to one
  card. A scheduled run then fetches and never proposes.
- The re-sync's command is not pinned either (F5): `git checkout -- .` after it,
  on scheduled runs only, discards every applied move before the gate looks.
- A NAMED step before the fetch, with no `if:`, whose shell exits 1 on scheduled
  runs (F6). Every later step without `!cancelled()` is skipped, `changed` is
  never set, and the delivery steps' own gate skips them. Unnamed, the step-name
  rule stops the same step, which is a reader precondition, not a guard.
- The job's envelope (F7, F8): `runs-on` chosen by the event name, so a scheduled
  run waits for a label no runner carries; or `needs:` a job that fails on
  scheduled runs, so the price job is skipped with no `if:` anywhere. In the
  ordinary suite only tests/corpus.test.py notices these, because the lines are
  anchors of workflow_r3_name_bindings.py; that is drift, not detection.
- A second trigger beside the schedule (F9): `push:` to master, so the price job
  runs on every merge as well as every Monday. GitHub's docs: a workflow with both
  `schedule` and `push` triggers on either.
- `on:` defined twice (F10). pyyaml keeps the last mapping and reads a complete
  schedule; GitHub's parser refuses a workflow whose mapping repeats a key
  ("'on' is already defined"), so the job never fires on its own.

Also found, and not a sabotage: `17 6 * * MON` is refused by the once-a-week rule,
although GitHub's cron accepts SUN-SAT and JAN-DEC names. A rule that decodes the
day-of-week field must accept the names or the refusal is the bug.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver

WF = ".github/workflows/price-refresh.yml"
CHECKOUT = "      - uses: actions/checkout@v4\n"
FETCH_CMD = "python3 tools/price_check.py --apply --report-out"
FETCH_RUN = '          python3 tools/price_check.py --apply --report-out "$RUNNER_TEMP/price-check-report.md"\n'
RESYNC_RUN = "        run: python3 tools/sync_data.py\n"
RUNS_ON = "    runs-on: ubuntu-latest\n"
JOBS_HEAD = "jobs:\n  price-check:\n"
DISPATCH_LINE = "  workflow_dispatch: {} # manual run, e.g. to check a single source sooner\n"
ON_HEAD = "\non:\n  schedule:\n"
# Exits 1 on a scheduled run and 0 on any other, with no `if:` in sight.
BY_HAND_ONLY = "        run: '[ \"$GITHUB_EVENT_NAME\" != schedule ] || exit 1'\n"


def fetch(cmd):
    return [(WF, FETCH_CMD, cmd, 1)]


def after(anchor, text):
    return [(WF, anchor, anchor + text, 1)]


S = {
 # ---- F1-F5: the commands before the gate, which no rule reads ----
 'F1 --apply chosen by an expression inside run:, on dispatch only':
     fetch("python3 tools/price_check.py ${{ github.event_name == 'workflow_dispatch' && '--apply' || '' }} --report-out"),
 'F2 the fetch wrapped in a shell if on GITHUB_EVENT_NAME': [
     (WF, FETCH_RUN,
      '          if [ "$GITHUB_EVENT_NAME" = workflow_dispatch ]; then\n'
      '            python3 tools/price_check.py --apply --report-out "$RUNNER_TEMP/price-check-report.md"\n'
      '          fi\n', 1)],
 'F3 --apply dropped from the fetch':
     fetch("python3 tools/price_check.py --report-out"),
 'F4 --slug restricts a scheduled run to one card':
     fetch("python3 tools/price_check.py --apply ${{ github.event_name == 'schedule' && '--slug h100-80' || '' }} --report-out"),
 'F5 the re-sync discards the applied moves on scheduled runs': [
     (WF, RESYNC_RUN,
      "        run: |\n"
      "          python3 tools/sync_data.py\n"
      '          if [ "$GITHUB_EVENT_NAME" = schedule ]; then git checkout -- .; fi\n', 1)],

 # ---- F6: a step before the gate that fails on schedule, with no if: ----
 'F6 a named unconditioned first step that exits 1 on scheduled runs':
     after(CHECKOUT, "      - name: Preflight\n" + BY_HAND_ONLY),

 # ---- F7, F8: the job's envelope ----
 'F7 runs-on a label no runner carries, on scheduled runs only': [
     (WF, RUNS_ON, "    runs-on: ${{ github.event_name == 'schedule' && 'self-hosted' || 'ubuntu-latest' }}\n", 1)],
 'F8 the price job needs: a job that fails on schedule': [
     (WF, JOBS_HEAD,
      "jobs:\n"
      "  preflight:\n"
      "    runs-on: ubuntu-latest\n"
      "    steps:\n"
      "      - name: Preflight\n"
      + BY_HAND_ONLY +
      "  price-check:\n"
      "    needs: preflight\n", 1)],

 # ---- F9, F10: the triggers ----
 'F9 push: to master beside the schedule, so the price job also runs on every merge':
     after(DISPATCH_LINE, "  push:\n    branches: [master]\n"),
 'F10 on: defined twice, the schedule only in the second (pyyaml keeps the last, GitHub refuses the file)': [
     (WF, ON_HEAD, "\non:\n  workflow_dispatch: {}\n\non:\n  schedule:\n", 1)],
}

if __name__ == "__main__":
    run_driver(S)
