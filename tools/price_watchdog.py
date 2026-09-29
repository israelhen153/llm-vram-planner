#!/usr/bin/env python3
"""Check that the price job's last scheduled run happened and delivered what it found.

tests/workflow.test.py reads the price workflow and refuses the edits that would let
a scheduled run do nothing. Three rounds of cold checks (PR #40) showed that reading
the file cannot close that class: a script after the gate that throws the moves away,
a new step wearing a permitted condition, a key GitHub's parser rejects so the file
never runs. GitHub also delays and drops scheduled runs outright: 2026-09-28's
06:00 run started at 12:44 UTC, after it had been run by hand.

So this checks the outcome instead, from GitHub's own record, a day after the price
job's slot. Did a scheduled run start since the slot? Did it succeed? Did it leave its
report? And when the report says it found something to propose, does a pull request
from the price branch carry a commit the job made during that run? Any "no" opens, or
adds to, one issue, and this exits 1 so its own run shows red as well.

Everything it needs to know about the price job, it reads from the price workflow:
which file it is (the one whose job opens the pull request, as the workflow suite
finds it), its schedule, its branch and its commit message.

Run:  python3 tools/price_watchdog.py --repo owner/name [--now 2026-09-29T07:43:00Z]
      (needs the gh CLI, and GH_TOKEN with actions:read, pull-requests:read, issues:write)
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = os.path.join(ROOT, ".github", "workflows")
PR_ACTION = "peter-evans/create-pull-request"
REPORT_ARTIFACT = "price-check-report"
REPORT_FILE = "price-check-report.md"
# The price job's body step appends one of these to its report only when the gate
# found something to propose; tests/watchdog.test.py holds them against that step.
PROPOSED_MARKERS = ("**The full suite passed on this branch.**",
                    "**The full suite is RED on this branch")
# GitHub starts scheduled runs late (5 h 50 min on 2026-09-21, 6 h 45 min on
# 2026-09-28), so a slot younger than this is not judged yet.
GRACE = datetime.timedelta(hours=12)
ISSUE_TITLE = "The weekly price check did not deliver"
WEEKDAYS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]


class WatchdogError(Exception):
    """The watchdog could not read what it needs. Loud, never a pass."""


def utc(stamp):
    return datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(datetime.timezone.utc)


# ---------------------------------------------------------------- the price job, read from its workflow

def price_workflow():
    """(file name, text) of the one workflow whose job opens the price pull request."""
    hits = []
    for name in sorted(os.listdir(WORKFLOWS)):
        if name.endswith((".yml", ".yaml")):
            with open(os.path.join(WORKFLOWS, name), encoding="utf-8") as fh:
                text = fh.read()
            if PR_ACTION in text:
                hits.append((name, text))
    if len(hits) != 1:
        raise WatchdogError(f"expected one workflow to open the price pull request, found {[h[0] for h in hits]}")
    return hits[0]


def one(pattern, text, what):
    found = re.findall(pattern, text, re.M)
    if len(found) != 1:
        raise WatchdogError(f"expected one {what} in the price workflow, found {found}")
    return found[0]


def schedule_of(text):
    """(minute, hour, weekday) of the price job's one cron, weekday counted from
    Sunday as cron counts it. The workflow suite holds the cron to this form."""
    cron = one(r'^\s*-\s*cron:\s*["\']([^"\']+)["\']', text, "cron")
    m = re.fullmatch(r"([0-9]{1,2}) ([0-9]{1,2}) \* \* ([0-6]|SUN|MON|TUE|WED|THU|FRI|SAT)", cron.strip())
    if not m:
        raise WatchdogError(f"the price job's cron {cron!r} is not `minute hour * * weekday`")
    day = m.group(3)
    return int(m.group(1)), int(m.group(2)), int(day) if day.isdigit() else WEEKDAYS.index(day)


def branch_of(text):
    return one(r"^\s*branch:\s*(\S+)\s*$", text, "pull-request branch")


def commit_message_of(text):
    return one(r'^\s*commit-message:\s*["\']?(.+?)["\']?\s*$', text, "commit message")


# ---------------------------------------------------------------- which run is due

def last_slot(now, minute, hour, weekday):
    """The latest moment at or before `now` that the cron names, in UTC. Cron counts
    Sunday as 0; Python's weekday() counts Monday as 0."""
    slot = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    slot -= datetime.timedelta(days=(slot.weekday() - (weekday - 1) % 7) % 7)
    if slot > now:
        slot -= datetime.timedelta(days=7)
    return slot


def slot_to_judge(now, schedule):
    """The slot whose run should have finished by `now`: the last one, or the one
    before it when the last is younger than GRACE."""
    slot = last_slot(now, *schedule)
    return slot - datetime.timedelta(days=7) if now - slot < GRACE else slot


def pick(slot, runs):
    """The latest scheduled run that started at or after the slot, or None."""
    due = [r for r in runs if utc(r["created_at"]) >= slot]
    return max(due, key=lambda r: utc(r["created_at"])) if due else None


# ---------------------------------------------------------------- the verdict

def problems(slot, run, report, prs, job_message):
    """What went wrong with the scheduled run due at `slot`, as sentences; empty
    when nothing did.

    run:    the scheduled run pick() chose, or None
    report: the text of that run's report artifact, or None when it left none
    prs:    the pull requests from the price branch, each with its commits
            ({"state", "merged_at", "html_url", "commits": [{"message", "date"}]})"""
    when = f"{slot:%Y-%m-%d %H:%M} UTC"
    if run is None:
        return [f"No scheduled run of the price job started since its slot at {when}. GitHub delays "
                f"scheduled runs and drops some; run it by hand (Actions > Price refresh > Run workflow)."]
    started, url = utc(run["created_at"]), run.get("html_url", "")
    name = f"The scheduled run of {started:%Y-%m-%d %H:%M} UTC"
    if run.get("status") != "completed":
        return [f"{name} has not finished ({run.get('status')}): {url}"]
    if run.get("conclusion") != "success":
        return [f"{name} ended {run.get('conclusion')}: {url}"]
    if report is None:
        return [f"{name} left no report ({REPORT_ARTIFACT}), so what it found cannot be told: {url}"]
    if not any(marker in report for marker in PROPOSED_MARKERS):
        return []
    for pr in prs:
        if pr.get("state") != "open" and not pr.get("merged_at"):
            continue
        if any(c["message"].split("\n", 1)[0].strip() == job_message and utc(c["date"]) >= started
               for c in pr.get("commits", [])):
            return []
    return [f"{name} found prices to propose, and no open or merged pull request from the price "
            f"branch carries a commit it made: none was opened or updated, or it was closed "
            f"unmerged. {url}"]


# ---------------------------------------------------------------- GitHub, through gh

def gh(*args, check=True):
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=check)


def gh_json(*args):
    return json.loads(gh(*args).stdout)


def scheduled_runs(repo, workflow_file):
    got = gh_json("api", "-X", "GET", f"repos/{repo}/actions/workflows/{workflow_file}/runs",
                  "-f", "event=schedule", "-f", "per_page=30")
    return got.get("workflow_runs", [])


def report_of(repo, run):
    with tempfile.TemporaryDirectory() as d:
        if gh("run", "download", str(run["id"]), "--repo", repo, "--name", REPORT_ARTIFACT,
              "--dir", d, check=False).returncode != 0:
            return None
        path = os.path.join(d, REPORT_FILE)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return fh.read()


def pull_requests(repo, branch):
    owner = repo.split("/")[0]
    prs = gh_json("api", "-X", "GET", f"repos/{repo}/pulls", "-f", "state=all",
                  "-f", f"head={owner}:{branch}", "-f", "per_page=10")
    for pr in prs:
        commits = gh_json("api", "-X", "GET", f"repos/{repo}/pulls/{pr['number']}/commits", "-f", "per_page=100")
        pr["commits"] = [{"message": c["commit"]["message"], "date": c["commit"]["committer"]["date"]}
                         for c in commits]
    return prs


def report_issue(repo, found, slot, now):
    body = ("The price watchdog checked the price job's scheduled run due at "
            f"{slot:%Y-%m-%d %H:%M} UTC, at {now:%Y-%m-%d %H:%M} UTC:\n\n"
            + "".join(f"- {line}\n" for line in found)
            + "\nChecked by `.github/workflows/price-watchdog.yml` (`tools/price_watchdog.py`).\n")
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as fh:
        fh.write(body)
    try:
        open_issues = gh_json("issue", "list", "--repo", repo, "--state", "open",
                              "--search", f'in:title "{ISSUE_TITLE}"', "--json", "number,title")
        same = [i for i in open_issues if i.get("title") == ISSUE_TITLE]
        if same:
            gh("issue", "comment", str(same[0]["number"]), "--repo", repo, "--body-file", fh.name)
        else:
            gh("issue", "create", "--repo", repo, "--title", ISSUE_TITLE, "--body-file", fh.name)
    finally:
        os.unlink(fh.name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--now", help="judge as if it were this UTC time (ISO 8601)")
    args = ap.parse_args(argv)
    now = utc(args.now) if args.now else datetime.datetime.now(datetime.timezone.utc)
    workflow_file, text = price_workflow()
    slot = slot_to_judge(now, schedule_of(text))
    run = pick(slot, scheduled_runs(args.repo, workflow_file))
    ok = run is not None and run.get("status") == "completed" and run.get("conclusion") == "success"
    report = report_of(args.repo, run) if ok else None
    prs = pull_requests(args.repo, branch_of(text)) if report is not None else []
    found = problems(slot, run, report, prs, commit_message_of(text))
    if not found:
        print(f"price watchdog: the scheduled run due at {slot:%Y-%m-%d %H:%M} UTC delivered.")
        return 0
    for line in found:
        print(f"price watchdog: {line}")
    report_issue(args.repo, found, slot, now)
    return 1


if __name__ == "__main__":
    sys.exit(main())
