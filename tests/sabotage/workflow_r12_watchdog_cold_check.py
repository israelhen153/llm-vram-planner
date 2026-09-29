#!/usr/bin/env python3
"""Round 12: the cold check of the price watchdog at c9c4a06 (tools/price_watchdog.py,
.github/workflows/price-watchdog.yml, judged by tests/watchdog.test.py).

Round 11 put back each way the watchdog's VERDICT could pass a week it should fail, and
tests/watchdog.test.py catches every one. This round went where that suite does not
look: the questions the watchdog asks GitHub, the workflow that runs it, and the words
it holds the price job to. Nineteen sabotages, sixteen survived; the three caught were
REPORT_FILE as a substring of the real name (the end-to-end test), the watchdog's cron
at minute 0 (the workflow suite reads every workflow's minute) and the body step
writing its markers to another file (the workflow suite runs the body script).

What survived, and why each matters:

  G: the gh calls. The end-to-end test's fake gh answers from a file whatever the
     arguments say, so a query that asks GitHub the wrong question is never seen.
     Without `event=schedule` a run started by hand is the scheduled run (2026-09-28's
     06:00 slot was run by hand at 07:44 and by GitHub at 12:44). Without `state=all`
     GET /pulls lists open pull requests only, so a pull request merged before Tuesday
     07:43 is invisible and the week is a false alarm. Without the head filter any of
     the ten newest pull requests may carry the commit, and the price one may not be
     among them. Listing /actions/runs instead of the price workflow's lists every
     scheduled run, and the watchdog's own run, in progress while it asks, is the
     latest since the slot: "has not finished", every week. Without `--name`, gh
     extracts each artifact under a directory of its own name, so the report is never
     at the path read: "left no report", every week. With `--state all` the problem is
     added to an issue the owner already closed.
  V: the verdict. A commit dated after the SLOT rather than after the run's start is
     a hand run's commit standing in for the scheduled run's, the very case the
     docstring describes.
  C: the clock. Every test passes --now, so main()'s real clock is never exercised: a
     naive datetime.now() there crashes the first comparison with a run's created_at.
  L: latent. WEEKDAYS reordered changes nothing while the cron says `1`, and the test
     computes its expectation from wd.WEEKDAYS itself, so it would pass a wrong table
     the day the cron is written as `MON`, which the workflow suite explicitly allows.
  W: the workflow that runs the watchdog. tests/watchdog.test.py finds it by a step
     whose shell mentions tools/price_watchdog.py and pins its cron and permissions,
     nothing else: `|| true` keeps the run green, a step-level or job-level `if:` keyed
     to workflow_dispatch retires the scheduled run (round 6's survivor, on the new
     file), the token dropped leaves gh unauthenticated on the runner (a crash, no
     issue), and another repository can be watched.
  P: the price workflow. The words test asks whether each marker is a substring of
     the body step's SOURCE, which a shell comment satisfies while the report the step
     writes says something else, so a found week reads as nothing to propose. And an
     artifact kept one day expires before a watchdog that runs 25 h after an on-time
     slot: "left no report".
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver

WD = "tools/price_watchdog.py"
WF = ".github/workflows/price-watchdog.yml"
PW = ".github/workflows/price-refresh.yml"


def swap(f, old, new):
    return [(f, old, new, 1)]


S = {
 # ---- G: what the watchdog asks GitHub ----
 "G1 a run started by hand counts as the scheduled run (event=schedule dropped)":
     swap(WD, '"-f", "event=schedule", "-f", "per_page=30")', '"-f", "per_page=30")'),
 "G2 a merged pull request is invisible (state=all dropped; GET /pulls lists open ones by default)":
     swap(WD, 'f"repos/{repo}/pulls", "-f", "state=all",', 'f"repos/{repo}/pulls",'),
 "G3 pull requests from any branch are read (head filter dropped)":
     swap(WD, '"-f", f"head={owner}:{branch}", ', ''),
 "G4 every workflow's scheduled runs are listed, the watchdog's own among them":
     swap(WD, 'f"repos/{repo}/actions/workflows/{workflow_file}/runs"', 'f"repos/{repo}/actions/runs"'),
 "G5 every artifact of the run is downloaded, each under its own directory (--name dropped)":
     swap(WD, '"--repo", repo, "--name", REPORT_ARTIFACT,', '"--repo", repo,'),
 "G6 the problem is added to an issue already closed (--state all)":
     swap(WD, '"--state", "open",', '"--state", "all",'),

 # ---- V: the verdict ----
 "V1 a commit made by a hand run since the slot counts as the scheduled run's":
     swap(WD, '    started, url = utc(run["created_at"]), run.get("html_url", "")\n',
              '    started, url = slot, run.get("html_url", "")\n'),

 # ---- C: the clock nothing passes ----
 "C1 the real clock is naive: comparing it with a run's created_at crashes":
     swap(WD, 'datetime.datetime.now(datetime.timezone.utc)', 'datetime.datetime.now()'),

 # ---- L: latent until the cron names its day ----
 "L1 WEEKDAYS starts at MON, and the test's expectation is computed from WEEKDAYS":
     swap(WD, 'WEEKDAYS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]',
              'WEEKDAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]'),

 # ---- W: the watchdog's own workflow ----
 "W1 the watchdog's run stays green whatever it found (|| true)":
     swap(WF, '        run: python3 tools/price_watchdog.py --repo "$GITHUB_REPOSITORY"\n',
              '        run: python3 tools/price_watchdog.py --repo "$GITHUB_REPOSITORY" || true\n'),
 "W2 the step runs only by hand (a step-level if keyed to workflow_dispatch)":
     swap(WF, "      - name: Check the price job's last scheduled run\n",
              "      - name: Check the price job's last scheduled run\n"
              "        if: github.event_name == 'workflow_dispatch'\n"),
 "W3 the job runs only by hand (a job-level if, reported as skipped)":
     swap(WF, '  watch:\n    runs-on: ubuntu-latest\n',
              "  watch:\n    if: github.event_name == 'workflow_dispatch'\n    runs-on: ubuntu-latest\n"),
 "W4 no GH_TOKEN, so gh is unauthenticated on the runner: a crash, and no issue":
     swap(WF, '        env:\n          GH_TOKEN: ${{ github.token }}\n', ''),
 "W5 another repository is watched":
     swap(WF, '--repo "$GITHUB_REPOSITORY"', '--repo "octocat/hello-world"'),

 # ---- P: the price workflow, as the watchdog reads it ----
 "P1 the RED marker survives only in a shell comment; the report the step writes says 'red'":
     swap(PW, "              printf '**The full suite is RED on this branch, and that is expected.**\\n\\n'\n",
              "              # **The full suite is RED on this branch, and that is expected.**\n"
              "              printf '**The full suite is red on this branch, and that is expected.**\\n\\n'\n"),
 "P2 the report artifact expires a day after upload, before a watchdog 25 h after an on-time slot":
     swap(PW, '          retention-days: 90\n', '          retention-days: 1\n'),
}

if __name__ == "__main__":
    run_driver(S)
