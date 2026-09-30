#!/usr/bin/env python3
"""Round 11 against the price watchdog (tools/price_watchdog.py, judged by
tests/watchdog.test.py).

Three cold checks of the price workflow's guard showed that reading the workflow cannot
prove a scheduled run does its work (the third's survivors are kept as accepted risk,
see the README), so a watchdog checks the outcome a day later, from GitHub's record.
These put back each way the watchdog could pass a week it should fail: a run
unfinished, failed or without its report taken as delivered; the report's own words
ignored; a pull request closed unmerged, or carrying only last week's commit or only a
person's, taken as the week's delivery; the slot judged with no grace, on Python's
weekday instead of cron's, or a week early; a problem found and the run left green, no
issue opened, or a new issue every week; and the watchdog's own workflow run inside
the grace, unable to open its issue, or reading another artifact than the job uploads.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver

WD = "tools/price_watchdog.py"
WF = ".github/workflows/price-watchdog.yml"


def swap(f, old, new):
    return [(f, old, new, 1)]


S = {
 # ---- V: the verdict on a run ----
 'V1 an unfinished run is judged as if it had ended':
     swap(WD, '    if run.get("status") != "completed":\n'
              '        return [f"{name} has not finished ({run.get(\'status\')}): {url}"]\n', ''),
 'V2 a failed run counts as delivered':
     swap(WD, '    if run.get("conclusion") != "success":\n'
              '        return [f"{name} ended {run.get(\'conclusion\')}: {url}"]\n', ''),
 'V3 a run that left no report is read as if it had one':
     swap(WD, '    if report is None:\n', '    if False:\n'),
 "V4 the report's words are ignored, so no pull request is ever expected":
     swap(WD, '    if not any(marker in report for marker in PROPOSED_MARKERS):\n', '    if True:\n'),
 'V5 a pull request closed unmerged counts as delivery':
     swap(WD, '        if pr.get("state") != "open" and not pr.get("merged_at"):\n            continue\n', ''),
 "V6 last week's commit by the job counts as this week's":
     swap(WD, ' and utc(c["date"]) >= started\n', '\n'),
 "V7 a person's commit counts as the job's":
     swap(WD, 'c["message"].split("\\n", 1)[0].strip() == job_message and ', ''),

 # ---- T: which slot is judged ----
 'T1 no grace: a slot is judged minutes after it passes':
     swap(WD, '    return slot - datetime.timedelta(days=7) if now - slot < GRACE else slot\n', '    return slot\n'),
 "T2 the weekday counted as Python counts it, not as cron does":
     swap(WD, '(weekday - 1) % 7', 'weekday % 7'),
 "T3 last week's run is taken for this week's":
     swap(WD, '    due = [r for r in runs if utc(r["created_at"]) >= slot]\n',
              '    due = [r for r in runs if utc(r["created_at"]) >= slot - datetime.timedelta(days=7)]\n'),

 # ---- R: what a problem does ----
 'R1 the problems are printed and the run stays green':
     swap(WD, '    report_issue(args.repo, found, slot, now)\n    return 1\n',
              '    report_issue(args.repo, found, slot, now)\n    return 0\n'),
 'R2 no issue is opened':
     swap(WD, '    report_issue(args.repo, found, slot, now)\n', ''),
 'R3 a new issue every week instead of adding to the open one':
     swap(WD, '        same = [i for i in open_issues if i.get("title") == ISSUE_TITLE]\n', '        same = []\n'),

 # ---- W: the watchdog's own workflow, and what it reads ----
 'W1 the watchdog runs inside the grace, an hour and a half after the price job':
     swap(WF, '    - cron: "43 7 * * 2"', '    - cron: "43 7 * * 1"'),
 'W2 the watchdog cannot open its issue':
     swap(WF, '      issues: write # ', '      issues: read # '),
 'W3 the watchdog reads another artifact than the job uploads':
     swap(WD, 'REPORT_ARTIFACT = "price-check-report"\n', 'REPORT_ARTIFACT = "price-report"\n'),
}

if __name__ == "__main__":
    run_driver(S)
