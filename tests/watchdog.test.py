#!/usr/bin/env python3
"""The price job's watchdog (tools/price_watchdog.py): its verdicts, the moment it
judges, what it asks GitHub, what it reads from the price workflow, and its own
workflow.

Three cold checks of the price workflow's guard (PR #40) ended in the same place:
reading a workflow file cannot prove that a scheduled run will do its work. The
watchdog checks the outcome instead, a day later, from GitHub's own record. It is only
as good as what this file holds: its verdict on every outcome a scheduled run can have,
the questions it puts to GitHub, its reading of the price workflow, which must stay the
price job's own, and its own workflow, which must run it and let it fail. A watchdog
that asks the wrong question, reads another schedule or words the job no longer
prints, or swallows its own exit, passes every week and watches nothing. Its first
cold check found each of those, and the fake gh below answers only the question asked.

Run:  python3 tests/watchdog.test.py
"""
import datetime
import json
import os
import re
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


# A contract, so it is a literal, and not the watchdog's own table: cron's weekday
# names, Sunday first ("0 - 6 or SUN-SAT", GitHub's cron table).
CRON_DAYS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]
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
    same run (closed unmerged); a person's later commit, and the job's commit from an
    earlier hand run, are not this run's delivery. Each problem must also say which it
    is: an issue that calls an unfinished run failed, or a failed one reportless, sends
    its reader the wrong way."""
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
        "found, and the job's newest commit came from an earlier hand run": (
            run(), FOUND_RED, [pr(commits=((MESSAGE, "2026-09-28T07:45:33Z"),))], "no open or merged pull request"),
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
    Monday as 0; every weekday is checked, derived, not listed, and every weekday name
    GitHub accepts is read against cron's own table, not the watchdog's."""
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
    for day, name in enumerate(CRON_DAYS):
        for spelled in (name, str(day)):
            got = wd.schedule_of(f'    - cron: "17 6 * * {spelled}"\n')
            assert got == (17, 6, day), f"{spelled!r} read as weekday {got[2]}, where cron means {day}"

test("the slot judged is the last one past the grace, on every weekday and weekday name",
     check_it_judges_the_slot_whose_run_should_be_done)


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


def cron_of(doc):
    on = doc.get(True, doc.get("on"))
    (cron,) = [e["cron"] for e in on["schedule"]]
    minute, hour, dom, month, dow = cron.split()
    assert dom == month == "*", cron
    return int(minute), int(hour), int(dow) if dow.isdigit() else CRON_DAYS.index(dow)


def check_it_reads_the_price_workflow_github_runs():
    """The watchdog searches the workflow text; GitHub parses it. The file, the
    schedule, the branch and the commit message the watchdog reads must be the ones
    the parsed workflow gives, or it watches a job that is not there."""
    name, doc = price_doc()
    found, text = wd.price_workflow()
    assert found == name, (found, name)
    assert wd.schedule_of(text) == cron_of(doc), (wd.schedule_of(text), cron_of(doc))
    (step,) = [s for s in workflow_steps(doc) if str(s.get("uses", "")).startswith(wd.PR_ACTION)]
    assert wd.branch_of(text) == step["with"]["branch"], (wd.branch_of(text), step["with"]["branch"])
    assert wd.commit_message_of(text) == step["with"]["commit-message"], wd.commit_message_of(text)

test("the watchdog reads the file, schedule, branch and commit message GitHub runs",
     check_it_reads_the_price_workflow_github_runs)


def body_output(steps, outcome):
    """What the price job's body step actually appends to its report, run as the
    workflow suite runs it: a fake RUNNER_TEMP, the suite's outcome in SUITE."""
    (body,) = [s for s in steps if s.get("id") == "body"]
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "suite.log"), "w") as fh:
            fh.write("FAIL a golden\n154 passed, 1 failed\n")
        done = subprocess.run(["bash", "-e", "-c", body["run"]], env=dict(os.environ, RUNNER_TEMP=tmp, SUITE=outcome),
                              capture_output=True, text=True, cwd=tmp, timeout=60)
        assert done.returncode == 0, f"the body script failed on SUITE={outcome}: {done.stderr[-300:]}"
        with open(os.path.join(tmp, wd.REPORT_FILE), encoding="utf-8") as fh:
            return fh.read()


def check_it_reads_the_words_the_body_step_writes():
    """The watchdog decides whether a run found prices from words the price job's body
    step appends to its report, and that step appends them only when the gate found a
    change. So the words must be in what the step writes, on a green suite and on a
    red one, not merely in its source: the cold check moved one into a shell comment
    and printed it differently, and a check of the source passed. The step must be
    gated on the gate, and the file it writes must be the one the job uploads (kept
    long enough for the watchdog to read it: see the watchdog's own workflow below)."""
    _, doc = price_doc()
    steps = workflow_steps(doc)
    for outcome in ("success", "failure"):
        written = body_output(steps, outcome)
        assert any(marker in written for marker in wd.PROPOSED_MARKERS), (
            f"on SUITE={outcome} the body step writes none of the watchdog's words:\n{written[-400:]}")
    for marker in wd.PROPOSED_MARKERS:
        assert any(marker in body_output(steps, o) for o in ("success", "failure")), f"nothing writes {marker!r}"
    (body,) = [s for s in steps if s.get("id") == "body"]
    assert "steps.diff.outputs.changed == 'true'" in str(body.get("if", "")), body.get("if")
    (upload,) = [s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    assert upload["with"]["name"] == wd.REPORT_ARTIFACT, upload["with"]["name"]
    assert wd.REPORT_FILE in upload["with"]["path"], upload["with"]["path"]

test("the watchdog reads the words the price job writes, from the artifact it keeps",
     check_it_reads_the_words_the_body_step_writes)


print("\nThe watchdog's own workflow")


def watch_doc():
    hits = []
    for name in sorted(os.listdir(WORKFLOWS)):
        if name.endswith((".yml", ".yaml")):
            with open(os.path.join(WORKFLOWS, name), encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
            if any("tools/price_watchdog.py" in str(s.get("run", "")) for s in workflow_steps(doc)):
                hits.append(doc)
    assert len(hits) == 1, f"{len(hits)} workflows run the watchdog"
    return hits[0]


def watch_offset():
    """How long after the price job's slot the watchdog's own slot falls, in a week."""
    minute, hour, day = cron_of(watch_doc())
    p_minute, p_hour, p_day = cron_of(price_doc()[1])
    return datetime.timedelta(minutes=((day - p_day) * 1440 + (hour - p_hour) * 60 + minute - p_minute) % (7 * 1440))


# A contract, so it is a literal: the watchdog's workflow, whole. An action may move to
# a newer version; nothing else changes without this.
WATCH_TOP_KEYS = {"name", "on", "permissions", "jobs"}
WATCH_JOB_KEYS = {"runs-on", "permissions", "steps"}
WATCH_PERMISSIONS = {"contents": "read", "actions": "read", "pull-requests": "read", "issues": "write"}
WATCH_STEPS = [
    {"uses": "actions/checkout"},
    {"uses": "actions/setup-python", "with": {"python-version": "3.x"}},
    {"name": "Check the price job's last scheduled run",
     "env": {"GH_TOKEN": "${{ github.token }}"},
     "run": 'python3 tools/price_watchdog.py --repo "$GITHUB_REPOSITORY"'},
]
ACTION_VERSION = r"@(v[0-9]+(\.[0-9]+){0,2}|[0-9a-f]{40})"


def check_the_watchdog_runs_a_day_after_the_price_job_and_can_fail():
    """The watchdog runs once a week, far enough after the price job's slot that the
    run is past the grace, and before the next slot. And its workflow is pinned whole,
    as the head of the price job is: the cold check made its step end `|| true`, gated
    the step and then the job on workflow_dispatch, took its token away, and pointed
    it at another repository, and each passed. A watchdog that cannot fail, or does not
    run on its schedule, watches nothing."""
    doc = watch_doc()
    after = watch_offset()
    assert wd.GRACE <= after < datetime.timedelta(days=7), f"the watchdog runs {after} after the slot"
    (upload,) = [s for s in workflow_steps(price_doc()[1]) if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    days = int(upload["with"].get("retention-days", 90))
    assert datetime.timedelta(days=days) >= after + datetime.timedelta(days=1), (
        f"the price job keeps its report {days} day(s) and the watchdog reads it {after} after the "
        f"slot, and a run can start hours late: the report can expire before it is read")
    assert {"on" if k is True else k for k in doc} == WATCH_TOP_KEYS, sorted(map(str, doc))
    on = doc.get(True, doc.get("on"))
    assert isinstance(on, dict) and set(on) == {"schedule", "workflow_dispatch"}, on
    (job,) = doc["jobs"].values()
    assert set(job) == WATCH_JOB_KEYS and job["runs-on"] == "ubuntu-latest", sorted(job)
    assert job["permissions"] == WATCH_PERMISSIONS, job["permissions"]
    assert len(job["steps"]) == len(WATCH_STEPS), [s.get("name") or s.get("uses") for s in job["steps"]]
    for want, got in zip(WATCH_STEPS, job["steps"]):
        if "uses" in want:
            assert re.fullmatch(re.escape(want["uses"]) + ACTION_VERSION, str(got.get("uses", ""))), got
            assert {k: v for k, v in got.items() if k != "uses"} == {k: v for k, v in want.items() if k != "uses"}, got
        else:
            assert got == want, f"the watch step is {got!r}, where the workflow has {want!r}"

test("the watchdog runs a day after the price job, pinned whole, and can fail",
     check_the_watchdog_runs_a_day_after_the_price_job_and_can_fail)


print("\nThe watchdog through gh, end to end")

# A gh that answers the question asked, as GitHub would: the runs of the workflow named
# in the path, filtered by event only when asked; open pull requests unless asked for
# all, from any branch unless one is named; an artifact extracted into the directory
# only when named, into one of its own name otherwise; open issues unless asked for all.
FAKE_GH = """#!/usr/bin/env python3
import json, os, sys
with open(os.environ["FAKE_GH_STATE"]) as fh:
    state = json.load(fh)
with open(os.environ["FAKE_GH_LOG"], "a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
a = sys.argv[1:]
fields = dict(a[i + 1].split("=", 1) for i, x in enumerate(a) if x == "-f")
if a[:1] == ["api"]:
    path = a[3]
    if path == "repos/o/r/actions/runs" or path.startswith("repos/o/r/actions/workflows/"):
        if path == "repos/o/r/actions/runs":
            runs = [r for rs in state["runs"].values() for r in rs]
        else:
            runs = state["runs"].get(path.split("/")[5], [])
        print(json.dumps({"workflow_runs": [r for r in runs if fields.get("event") in (None, r["event"])]}))
    elif path.startswith("repos/o/r/pulls/") and path.endswith("/commits"):
        print(json.dumps(state["commits"][path.split("/")[4]]))
    elif path == "repos/o/r/pulls":
        want = fields.get("state", "open")
        print(json.dumps([p for p in state["pulls"]
                          if want == "all" or ("merged" if p["merged_at"] else p["state"]) == want or p["state"] == want
                          if "head" not in fields or fields["head"] == "o:" + p["head"]["ref"]]))
    else:
        sys.exit("fake gh: unexpected api path " + path)
elif a[:2] == ["run", "download"]:
    report = state["reports"].get(a[2])
    if report is None:
        sys.exit(1)
    where = a[a.index("--dir") + 1]
    if "--name" not in a:
        where = os.path.join(where, "price-check-report")
        os.makedirs(where)
    with open(os.path.join(where, "price-check-report.md"), "w") as fh:
        fh.write(report)
elif a[:2] == ["issue", "list"]:
    want = a[a.index("--state") + 1] if "--state" in a else "open"
    print(json.dumps([i for i in state.get("issues", []) if want == "all" or i["state"] == want]))
elif a[:2] not in (["issue", "create"], ["issue", "comment"]):
    sys.exit("fake gh: unexpected " + " ".join(a))
"""


def gh_run(created, event, conclusion="success", status="completed", run_id=1):
    return {"id": run_id, "created_at": created, "event": event, "status": status, "conclusion": conclusion,
            "html_url": f"https://github.com/o/r/actions/runs/{run_id}"}


def gh_commit(date, message=MESSAGE):
    return {"commit": {"message": message, "committer": {"date": date}}}


def world(**over):
    """GitHub on Tuesday 2026-09-29, a week that delivered, with every trap a wrong
    question falls into: a hand run after the scheduled one, the watchdog's own run in
    progress, the price pull request already merged, and another branch's pull request
    carrying a commit with the job's message."""
    price_file, _ = wd.price_workflow()
    state = {
        "runs": {price_file: [gh_run("2026-09-28T12:44:59Z", "schedule", run_id=1),
                              gh_run("2026-09-28T14:00:00Z", "workflow_dispatch", run_id=2)],
                 "price-watchdog.yml": [gh_run("2026-09-29T07:43:00Z", "schedule", conclusion=None,
                                               status="in_progress", run_id=3)]},
        "reports": {"1": FOUND_RED, "2": FOUND_RED},
        "pulls": [{"number": 39, "state": "closed", "merged_at": "2026-09-29T06:00:00Z",
                   "head": {"ref": wd.branch_of(wd.price_workflow()[1])}, "html_url": "u39"},
                  {"number": 41, "state": "open", "merged_at": None, "head": {"ref": "fix/other"}, "html_url": "u41"}],
        "commits": {"39": [gh_commit("2026-09-28T12:45:33Z")], "41": [gh_commit("2026-09-28T13:00:00Z")]},
        "issues": [],
    }
    state.update(over)
    return state


def check_the_watchdog_end_to_end():
    """The real script, with a gh that answers the question asked. A delivered week
    passes and opens no issue. Each wrong question the cold check put back changes that:
    runs of any event (the later hand run is judged), of every workflow (the watchdog's
    own run is judged), open pull requests only (the merged one is missed), of any
    branch (another branch's commit counts), an artifact not named (its file lands in
    a directory of its own), and issues of any state (a closed one is commented on).
    A failed scheduled run is a problem even when a hand run delivered after it."""
    with tempfile.TemporaryDirectory() as d:
        gh, state_path, log = os.path.join(d, "gh"), os.path.join(d, "state.json"), os.path.join(d, "calls.log")
        with open(gh, "w") as fh:
            fh.write(FAKE_GH)
        os.chmod(gh, os.stat(gh).st_mode | stat.S_IEXEC)

        def call(state, now="2026-09-29T07:43:00Z"):
            with open(state_path, "w") as fh:
                json.dump(state, fh)
            open(log, "w").close()
            done = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "price_watchdog.py"), "--repo", "o/r",
                                   *(["--now", now] if now else [])], capture_output=True, text=True, timeout=60,
                                  env=dict(os.environ, PATH=d + os.pathsep + os.environ.get("PATH", ""),
                                           FAKE_GH_STATE=state_path, FAKE_GH_LOG=log))
            with open(log) as fh:
                calls = [json.loads(line) for line in fh]
            issue_calls = [c[:3] if c[1] == "comment" else c[:2] for c in calls if c[:1] == ["issue"] and c[1] != "list"]
            return done.returncode, issue_calls, done.stdout + done.stderr

        price_file, _ = wd.price_workflow()
        rc, issues, out = call(world())
        assert rc == 0 and not issues, f"a delivered week failed: {rc} {issues} {out[-300:]}"

        failed = world(runs=dict(world()["runs"], **{price_file: [
            gh_run("2026-09-28T12:44:59Z", "schedule", conclusion="failure", run_id=1),
            gh_run("2026-09-28T14:00:00Z", "workflow_dispatch", run_id=2)]}),
            commits=dict(world()["commits"], **{"39": [gh_commit("2026-09-28T14:01:00Z")]}))
        rc, issues, out = call(failed)
        assert rc == 1 and issues == [["issue", "create"]], f"a failed scheduled run passed: {rc} {issues} {out[-300:]}"

        closed = world(pulls=[dict(world()["pulls"][0], merged_at=None), world()["pulls"][1]])
        rc, issues, out = call(closed)
        assert rc == 1 and issues == [["issue", "create"]], f"closed unmerged passed: {rc} {issues} {out[-300:]}"
        rc, issues, out = call(dict(closed, issues=[{"number": 7, "title": wd.ISSUE_TITLE, "state": "open"}]))
        assert rc == 1 and issues == [["issue", "comment", "7"]], f"the open issue was not added to: {issues}"
        rc, issues, out = call(dict(closed, issues=[{"number": 6, "title": wd.ISSUE_TITLE, "state": "closed"}]))
        assert rc == 1 and issues == [["issue", "create"]], f"a closed issue was written to: {issues}"

        # On the real clock, as the workflow runs it: a scheduled run just after the slot
        # due now, which found nothing, is a delivered week and nothing crashes on the way.
        slot = wd.slot_to_judge(datetime.datetime.now(datetime.timezone.utc), wd.schedule_of(wd.price_workflow()[1]))
        started = (slot + datetime.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rc, issues, out = call(world(runs={price_file: [gh_run(started, "schedule")]}, reports={"1": NOTHING}), now=None)
        assert rc == 0 and not issues, f"on the real clock: {rc} {issues} {out[-400:]}"

test("the watchdog, through a gh that answers the question asked, on a fixed and on the real clock",
     check_the_watchdog_end_to_end)


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
