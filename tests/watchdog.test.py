#!/usr/bin/env python3
"""The price job's watchdog (tools/price_watchdog.py): its verdicts, the moment it
judges, and what it reads from the price workflow.

Three cold checks of the price workflow's guard (PR #40) ended in the same place:
reading a workflow file cannot prove that a scheduled run will do its work. The
watchdog checks the outcome instead, a day later, from GitHub's own record. It is only
as good as two things this file holds: its verdict on every outcome a scheduled run can
have, and its reading of the price workflow, which must stay the price job's own. A
watchdog that reads another schedule, another branch or words the job no longer writes
passes every week and watches nothing.

Run:  python3 tests/watchdog.test.py
"""
import datetime
import json
import os
import stat
import subprocess
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = os.path.join(ROOT, ".github", "workflows")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import price_watchdog as wd  # noqa: E402

pass_ct = fail_ct = 0


def test(name, fn):
    global pass_ct, fail_ct
    try:
        fn()
        print(f"  ok   {name}")
        pass_ct += 1
    except Exception as e:
        print(f"  FAIL {name}\n       {type(e).__name__}: {e}")
        fail_ct += 1


def at(stamp):
    return datetime.datetime.fromisoformat(stamp).replace(tzinfo=datetime.timezone.utc)


SLOT = at("2026-09-28T06:17:00")     # a Monday, the price job's slot
MESSAGE = "price-refresh: update catalog prices and provenance"
FOUND_RED = "## Proposed updates\n- x\n\n---\n\n**The full suite is RED on this branch, and that is expected.**\n"
FOUND_GREEN = "## Proposed updates\n- x\n\n---\n\n**The full suite passed on this branch.** No recorded surface moved,\n"
NOTHING = "## Proposed updates\n\n## Could not validate this run\n- every source failed\n"


def run(created="2026-09-28T12:44:59Z", status="completed", conclusion="success"):
    return {"id": 1, "created_at": created, "status": status, "conclusion": conclusion,
            "html_url": "https://github.com/o/r/actions/runs/1"}


def pr(state="open", merged_at=None, commits=((MESSAGE, "2026-09-28T12:45:33Z"),)):
    return {"state": state, "merged_at": merged_at, "html_url": "https://github.com/o/r/pull/39",
            "commits": [{"message": m, "date": d} for m, d in commits]}


print("\nThe verdict on each outcome of a scheduled run")


def check_every_outcome_gets_its_verdict():
    """Each way a scheduled run can end, and whether it is a problem. A run that
    found prices is delivered only when an open or merged pull request from the price
    branch carries a commit the job made during that run: the third cold check threw
    the moves away after the gate (no new commit), and closed the pull request in the
    same run (closed unmerged), and a person's later commit on last week's pull request
    is not this week's delivery. Each problem must also say which it is: an issue that
    calls an unfinished run failed, or a failed one reportless, sends its reader the
    wrong way, and a check with its own branch gone would still read as a problem."""
    delivered = None
    cases = {
        "no scheduled run since the slot": (None, None, [], "No scheduled run"),
        "a run still going": (run(status="in_progress", conclusion=None), None, [], "has not finished"),
        "a run that failed": (run(conclusion="failure"), None, [], "ended failure"),
        "a run that failed, with a report and a pull request": (run(conclusion="failure"), FOUND_RED, [pr()], "ended failure"),
        "a run that was cancelled": (run(conclusion="cancelled"), None, [], "ended cancelled"),
        "a run that left no report": (run(), None, [], "left no report"),
        "a run that found nothing to propose": (run(), NOTHING, [], delivered),
        "found, suite red, an open pull request carries it": (run(), FOUND_RED, [pr()], delivered),
        "found, suite green, an open pull request carries it": (run(), FOUND_GREEN, [pr()], delivered),
        "found, and a merged pull request carries it": (
            run(), FOUND_RED, [pr(state="closed", merged_at="2026-09-29T06:00:00Z")], delivered),
        "found, suite red, no pull request": (run(), FOUND_RED, [], "no open or merged pull request"),
        "found, suite green, no pull request": (run(), FOUND_GREEN, [], "no open or merged pull request"),
        "found, and the pull request was closed unmerged": (
            run(), FOUND_RED, [pr(state="closed")], "no open or merged pull request"),
        "found, and the job's newest commit is last week's": (
            run(), FOUND_RED, [pr(commits=((MESSAGE, "2026-09-21T11:51:00Z"),))], "no open or merged pull request"),
        "found, and only a person's commit is newer than the run": (
            run(), FOUND_RED, [pr(commits=((MESSAGE, "2026-09-21T11:51:00Z"),
                                           ("Record the goldens", "2026-09-28T14:00:00Z")))],
            "no open or merged pull request"),
    }
    wrong = {}
    for case, (r, report, prs, want) in cases.items():
        got = wd.problems(SLOT, r, report, prs, MESSAGE)
        if (got == []) != (want is delivered) or (want and not (len(got) == 1 and want in got[0])):
            wrong[case] = got
    assert not wrong, wrong

test("every outcome of a scheduled run gets its verdict", check_every_outcome_gets_its_verdict)


def check_it_judges_the_latest_run_since_the_slot():
    """Only runs that started at or after the slot are this week's; the latest of them
    is the one judged. Last week's run is never this week's delivery."""
    runs = [run(created="2026-09-21T11:49:59Z", conclusion="failure"),
            run(created="2026-09-28T07:44:18Z", conclusion="failure"),
            run(created="2026-09-28T12:44:59Z")]
    assert wd.pick(SLOT, runs)["created_at"] == "2026-09-28T12:44:59Z", wd.pick(SLOT, runs)
    assert wd.pick(SLOT, runs[:1]) is None, "last week's run was taken for this week's"

test("the latest scheduled run since the slot is the one judged", check_it_judges_the_latest_run_since_the_slot)


def check_it_judges_the_slot_whose_run_should_be_done():
    """GitHub starts scheduled runs hours late, so a slot younger than the grace is not
    judged yet and the one before it is. Cron counts Sunday as 0 and Python counts
    Monday as 0; every weekday is checked, derived, not listed."""
    monday = (17, 6, 1)
    cases = {
        "a day after the slot": ("2026-09-29T07:43:00", "2026-09-28T06:17:00"),
        "a minute after the slot": ("2026-09-28T06:18:00", "2026-09-21T06:17:00"),
        "just past the grace": ("2026-09-28T18:18:00", "2026-09-28T06:17:00"),
        "the Sunday before the slot": ("2026-09-27T23:00:00", "2026-09-21T06:17:00"),
    }
    for case, (now, want) in cases.items():
        got = wd.slot_to_judge(at(now), monday)
        assert got == at(want), (case, got)
    now = at("2026-09-30T10:00:00")
    for day in range(7):
        slot = wd.last_slot(now, 17, 6, day)
        assert (slot.weekday() + 1) % 7 == day and datetime.timedelta(0) <= now - slot < datetime.timedelta(days=7), (day, slot)

test("the slot judged is the last one past the grace, on every weekday", check_it_judges_the_slot_whose_run_should_be_done)


print("\nWhat the watchdog reads from the price workflow")


def workflow_steps(doc):
    return [s for j in doc["jobs"].values() for s in j.get("steps") or []]


def price_doc():
    """The price workflow as the workflow suite finds it: parsed, by the job that opens
    the pull request, not by the text the watchdog searches."""
    hits = []
    for name in sorted(os.listdir(WORKFLOWS)):
        if name.endswith((".yml", ".yaml")):
            with open(os.path.join(WORKFLOWS, name), encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
            if any(str(s.get("uses", "")).startswith(wd.PR_ACTION) for s in workflow_steps(doc)):
                hits.append((name, doc))
    assert len(hits) == 1, [h[0] for h in hits]
    return hits[0]


def check_it_reads_the_price_workflow_github_runs():
    """The watchdog searches the workflow text; GitHub parses it. The file, the
    schedule, the branch and the commit message the watchdog reads must be the ones
    the parsed workflow gives, or it watches a job that is not there."""
    name, doc = price_doc()
    found, text = wd.price_workflow()
    assert found == name, (found, name)
    on = doc.get(True, doc.get("on"))
    (cron,) = [e["cron"] for e in on["schedule"]]
    minute, hour, _, _, dow = cron.split()
    want = (int(minute), int(hour), int(dow) if dow.isdigit() else wd.WEEKDAYS.index(dow))
    assert wd.schedule_of(text) == want, (wd.schedule_of(text), want)
    (step,) = [s for s in workflow_steps(doc) if str(s.get("uses", "")).startswith(wd.PR_ACTION)]
    assert wd.branch_of(text) == step["with"]["branch"], (wd.branch_of(text), step["with"]["branch"])
    assert wd.commit_message_of(text) == step["with"]["commit-message"], wd.commit_message_of(text)

test("the watchdog reads the file, schedule, branch and commit message GitHub runs",
     check_it_reads_the_price_workflow_github_runs)


def check_it_reads_the_words_the_body_step_writes():
    """The watchdog decides whether a run found prices from the words the price job's
    body step appends to its report, and that step writes them only when the gate found
    a change. Each marker must be in that step's script, the step must be gated on the
    gate, and the file it writes must be the one the job uploads and the watchdog reads."""
    _, doc = price_doc()
    steps = workflow_steps(doc)
    (body,) = [s for s in steps if s.get("id") == "body"]
    for marker in wd.PROPOSED_MARKERS:
        assert marker in body["run"], f"the body step no longer writes {marker!r}"
    assert "steps.diff.outputs.changed == 'true'" in str(body.get("if", "")), body.get("if")
    assert wd.REPORT_FILE in body["run"], "the body step writes another report file"
    (upload,) = [s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    assert upload["with"]["name"] == wd.REPORT_ARTIFACT, upload["with"]["name"]
    assert wd.REPORT_FILE in upload["with"]["path"], upload["with"]["path"]

test("the watchdog reads the words, file and artifact the price job writes",
     check_it_reads_the_words_the_body_step_writes)


def check_the_watchdog_runs_a_day_after_the_price_job():
    """Exactly one workflow runs the watchdog. It runs once a week, far enough after the
    price job's slot that the run is past the grace, and before the next slot; and it
    holds the permissions it reads and writes with."""
    hits = []
    for name in sorted(os.listdir(WORKFLOWS)):
        if name.endswith((".yml", ".yaml")):
            with open(os.path.join(WORKFLOWS, name), encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
            if any("tools/price_watchdog.py" in str(s.get("run", "")) for s in workflow_steps(doc)):
                hits.append(doc)
    assert len(hits) == 1, f"{len(hits)} workflows run the watchdog"
    doc = hits[0]
    on = doc.get(True, doc.get("on"))
    (cron,) = [e["cron"] for e in on["schedule"]]
    minute, hour, dom, month, dow = cron.split()
    assert dom == month == "*", cron
    watch = (int(dow) if dow.isdigit() else wd.WEEKDAYS.index(dow)) * 1440 + int(hour) * 60 + int(minute)
    _, text = wd.price_workflow()
    p_minute, p_hour, p_day = wd.schedule_of(text)
    offset = datetime.timedelta(minutes=(watch - (p_day * 1440 + p_hour * 60 + p_minute)) % (7 * 1440))
    assert wd.GRACE <= offset < datetime.timedelta(days=7), f"the watchdog runs {offset} after the price job's slot"
    (job,) = doc["jobs"].values()
    perms = job.get("permissions") or {}
    for scope, level in {"actions": "read", "pull-requests": "read", "issues": "write"}.items():
        assert perms.get(scope) == level, f"the watchdog's job has {scope}: {perms.get(scope)!r}, and needs {level}"

test("the watchdog runs once a week, a day after the price job, with what it needs",
     check_the_watchdog_runs_a_day_after_the_price_job)


print("\nThe watchdog through gh, end to end")

FAKE_GH = """#!/usr/bin/env python3
import json, os, sys
with open(os.environ["FAKE_GH_STATE"]) as fh:
    state = json.load(fh)
with open(os.environ["FAKE_GH_LOG"], "a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
a = sys.argv[1:]
if a[:1] == ["api"]:
    path = a[3]
    key = "runs" if path.endswith("/runs") else "commits" if path.endswith("/commits") else "pulls"
    print(json.dumps({"workflow_runs": state["runs"]} if key == "runs" else state[key]))
elif a[:2] == ["run", "download"]:
    if state.get("report") is None:
        sys.exit(1)
    with open(os.path.join(a[a.index("--dir") + 1], "price-check-report.md"), "w") as fh:
        fh.write(state["report"])
elif a[:2] == ["issue", "list"]:
    print(json.dumps(state.get("issues", [])))
elif a[:2] not in (["issue", "create"], ["issue", "comment"]):
    sys.exit("fake gh: unexpected " + " ".join(a))
"""


def check_the_watchdog_end_to_end():
    """The real script, with a gh that answers from a file. A run that found prices,
    carried by an open pull request, passes and opens no issue. The same run with its
    pull request closed unmerged exits 1 and opens one issue; with that issue already
    open, it comments on it instead of opening another."""
    with tempfile.TemporaryDirectory() as d:
        gh, state_path, log = os.path.join(d, "gh"), os.path.join(d, "state.json"), os.path.join(d, "calls.log")
        with open(gh, "w") as fh:
            fh.write(FAKE_GH)
        os.chmod(gh, os.stat(gh).st_mode | stat.S_IEXEC)

        def call(state):
            with open(state_path, "w") as fh:
                json.dump(state, fh)
            open(log, "w").close()
            p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "price_watchdog.py"), "--repo", "o/r",
                                "--now", "2026-09-29T07:43:00Z"], capture_output=True, text=True, timeout=60,
                               env=dict(os.environ, PATH=d + os.pathsep + os.environ.get("PATH", ""),
                                        FAKE_GH_STATE=state_path, FAKE_GH_LOG=log))
            with open(log) as fh:
                calls = [json.loads(line) for line in fh]
            return p.returncode, [c for c in calls if c[:1] == ["issue"] and c[1] != "list"], p.stdout + p.stderr

        carried = {"runs": [run()], "report": FOUND_RED,
                   "pulls": [{"number": 39, "state": "open", "merged_at": None, "html_url": "u"}],
                   "commits": [{"commit": {"message": MESSAGE, "committer": {"date": "2026-09-28T12:45:33Z"}}}]}
        rc, issues, out = call(carried)
        assert rc == 0 and not issues, (rc, issues, out[-300:])
        closed = dict(carried, pulls=[{"number": 39, "state": "closed", "merged_at": None, "html_url": "u"}])
        rc, issues, out = call(closed)
        assert rc == 1 and [c[:2] for c in issues] == [["issue", "create"]], (rc, issues, out[-300:])
        rc, issues, out = call(dict(closed, issues=[{"number": 7, "title": wd.ISSUE_TITLE}]))
        assert rc == 1 and [c[:3] for c in issues] == [["issue", "comment", "7"]], (rc, issues, out[-300:])

test("the watchdog, through gh: delivered passes; closed unmerged opens one issue, then comments",
     check_the_watchdog_end_to_end)


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
