#!/usr/bin/env python3
"""Round 13: the second cold check of the price watchdog, at 4719776 (tools/price_watchdog.py,
.github/workflows/price-watchdog.yml, judged by tests/watchdog.test.py).

Round 12 closed the questions the watchdog asks GitHub, its own workflow and the words it
holds the price job to. This round went where tests/watchdog.test.py still does not look:
main(), which the verdict test never calls and the end-to-end run exercises on three
outcomes only (a delivered week, a failed run, a pull request closed unmerged); the
reading of the price workflow, checked for what it returns but never for whether main()
uses it; the two constants no case holds in place; and what the price job's own actions
do with what the watchdog reads. Eighteen sabotages, fourteen survived and are kept here.

What survived, and why each matters:

  K: main(), before the verdict. Every other outcome of a scheduled run is decided by
     problems(), which the verdict test calls directly; main() short-circuits ahead of
     it and no end-to-end world has a run still going, no run at all, a cancelled run,
     a skipped one, or a successful run without its artifact. So main() returning 0 for
     any of those passes, and `skipped` (what a job-level `if:` leaves behind, round 3's
     and round 6's survivor) accepted beside `success` passes too; GitHub's list-runs
     filter names fourteen status and conclusion values, and the verdict test holds two.
     A report_of() that answers '' for a missing artifact turns "left no report" into
     "found nothing", and a hand run standing in for a dropped scheduled one is the
     very case the docstring refuses.
  R: what main() reads. The read test compares schedule_of(), branch_of() and
     commit_message_of() with the parsed workflow, and the end-to-end run uses the real
     workflow, whose values equal any literal copied from it. main() judging a
     hardcoded `0 6 * * 1` (the schedule before the off-the-hour fix), or asking GitHub
     about "price-refresh.yml", "automation/price-refresh" and the commit message by
     name, passes until the day one of them moves.
  T: two tolerances. GRACE is held between one minute and twelve hours by the slot
     cases, and by the workflow's offset from above only; at one hour a watchdog run
     by hand after a slot GitHub is still late on (5 h 50 min and 6 h 45 min, the
     docstring's own record) opens an issue for a run that is merely late. The commit
     that counts must be dated after the run's start, and the verdict test's hand-run
     commit is 4 h 59 min before it, so a tolerance under that passes.
  Q: a question the fake answers as asked. `status=completed` on the runs query is a
     field the fake ignores; on GitHub a run still going is filtered out and the issue
     says "No scheduled run ... run it by hand", the wrong way round, while the
     concurrency group would queue that hand run behind the one still going.
  P: the price job's actions, which do more than the words the watchdog reads. A
     third path in the artifact from the workspace: actions/upload-artifact roots the
     artifact at the least common ancestor of all its paths (its README), so the
     report lands under `_temp/`, never at the path read: "left no report", every week,
     while the test asks only whether the report's name is a substring of `path:`.
     A `branch-suffix` on the pull-request step (`random`, `timestamp` or
     `short-commit-hash`: the action's alternative strategy, its README): the head ref
     is no longer the `branch:` the watchdog reads, no pull request answers its query,
     and a found week is "no open or merged pull request", every time, while the read
     test compares branch_of() with `branch:` alone. A `timezone:` under the price
     job's schedule entry, which GitHub's parser schema allows (actions/languageservices,
     workflow-v1.0.json, `cron-mapping`: `cron` and `timezone`): `17 6 * * 1` in
     Pacific/Auckland is Sunday 17:17 or 18:17 UTC by the season, so the run starts
     before the Monday 06:17 UTC slot the watchdog computes and nothing has started
     since it: "No scheduled run", every week. The workflow suite reads `cron` alone
     from each entry, as the watchdog does.

Caught: `on:` given twice (the workflow suite's StrictLoader reads every file), the body
step's condition widened around the gate, and the upload gated on `changed == 'false'`
again (both by the workflow suite's literal DELIVERY_IF). A `timezone:` under the
watchdog's own schedule entry was tried and survived, and is not kept: the same schema
allows it, so the file runs.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver

WD = "tools/price_watchdog.py"
PW = ".github/workflows/price-refresh.yml"


def swap(f, old, new):
    return [(f, old, new, 1)]


OK_LINE = '    ok = run is not None and run.get("status") == "completed" and run.get("conclusion") == "success"\n'


def before_verdict(guard, message):
    """main() returning 0 on `guard`, ahead of the line that feeds problems()."""
    return swap(WD, OK_LINE, f'    if {guard}:\n        print("price watchdog: {message}")\n        return 0\n' + OK_LINE)


S = {
 # ---- K: main(), before the verdict ----
 "K1 a skipped run counts as a success (what a job-level if leaves behind)":
     [(WD, 'run.get("conclusion") == "success"', 'run.get("conclusion") in ("success", "skipped")', 1),
      (WD, '    if run.get("conclusion") != "success":\n',
           '    if run.get("conclusion") not in ("success", "skipped"):\n', 1)],
 "K2 main() returns 0 while the scheduled run is still going":
     before_verdict('run is not None and run.get("status") != "completed"',
                    "the scheduled run is still going; judged next week."),
 "K3 main() returns 0 when no scheduled run started since the slot":
     before_verdict("run is None", "GitHub dropped the scheduled run; nothing to judge."),
 "K4 main() returns 0 when the scheduled run was cancelled":
     before_verdict('run is not None and run.get("conclusion") == "cancelled"',
                    "the scheduled run was cancelled by a person; nothing to judge."),
 "K5 a run that left no report reads as a run that found nothing (report_of answers '')":
     [(WD, 'check=False).returncode != 0:\n            return None\n',
           'check=False).returncode != 0:\n            return ""\n', 1),
      (WD, '        if not os.path.exists(path):\n            return None\n',
           '        if not os.path.exists(path):\n            return ""\n', 1)],
 "K6 a hand run since the slot stands in when GitHub dropped the scheduled one":
     swap(WD, '    run = pick(slot, scheduled_runs(args.repo, workflow_file))\n',
          '    run = pick(slot, scheduled_runs(args.repo, workflow_file))\n'
          '    if run is None:  # GitHub dropped the scheduled run: a run started by hand since the slot stands in\n'
          '        run = pick(slot, gh_json("api", "-X", "GET", f"repos/{args.repo}/actions/workflows/{workflow_file}/runs",\n'
          '                                 "-f", "per_page=30").get("workflow_runs", []))\n'),

 # ---- R: what main() reads, and whether it uses it ----
 "R1 main() judges the slot of a hardcoded `0 6 * * 1`, not the schedule it read":
     swap(WD, '    slot = slot_to_judge(now, schedule_of(text))\n',
          '    slot = slot_to_judge(now, (0, 6, 1))  # Monday 06:00 UTC\n'),
 "R2 main() asks GitHub about literals: the file, the branch and the commit message it read go unused":
     [(WD, 'scheduled_runs(args.repo, workflow_file)', 'scheduled_runs(args.repo, "price-refresh.yml")', 1),
      (WD, 'pull_requests(args.repo, branch_of(text))', 'pull_requests(args.repo, "automation/price-refresh")', 1),
      (WD, 'problems(slot, run, report, prs, commit_message_of(text))',
           'problems(slot, run, report, prs, "price-refresh: update catalog prices and provenance")', 1)],

 # ---- T: two tolerances ----
 "T1 the grace is one hour: a watchdog run by hand two hours after the slot reports a run GitHub is merely late on":
     swap(WD, 'GRACE = datetime.timedelta(hours=12)', 'GRACE = datetime.timedelta(hours=1)'),
 "T2 a commit made up to an hour before the scheduled run started counts as its own":
     swap(WD, 'utc(c["date"]) >= started', 'utc(c["date"]) >= started - datetime.timedelta(hours=1)'),

 # ---- Q: a question the fake answers as asked ----
 "Q1 the runs query adds status=completed: a run still going reads as never started":
     swap(WD, '"-f", "event=schedule", "-f", "per_page=30")',
          '"-f", "event=schedule", "-f", "status=completed", "-f", "per_page=30")'),

 # ---- P: the price job's actions ----
 "P1 a workspace file joins the artifact, so its root moves up and the report is never at the path read":
     swap(PW, '            ${{ runner.temp }}/price-check-report.md\n            ${{ runner.temp }}/suite.log\n',
          '            ${{ runner.temp }}/price-check-report.md\n            ${{ runner.temp }}/suite.log\n'
          '            data/gpus.json\n'),
 "P2 branch-suffix on the pull-request step: the head ref is not the branch the watchdog reads":
     swap(PW, '          branch: automation/price-refresh\n',
          '          branch: automation/price-refresh\n          branch-suffix: timestamp\n'),
 "P3 the price job's schedule gains timezone: Pacific/Auckland, so its run starts before the UTC slot the watchdog judges":
     swap(PW, '    - cron: "17 6 * * 1" # Monday 06:17 UTC\n',
          '    - cron: "17 6 * * 1" # Monday 06:17 UTC\n      timezone: Pacific/Auckland\n'),
}

if __name__ == "__main__":
    run_driver(S)
