#!/usr/bin/env python3
"""The price job's watchdog (tools/price_watchdog.py): its verdicts, the moment it
judges, what it asks GitHub, what it reads from the price workflow, and its own
workflow.

Three cold checks of the price workflow's guard (PR #40) ended in the same place:
reading a workflow file cannot prove that a scheduled run will do its work. The
watchdog checks the outcome instead, a day later, from GitHub's own record. It is only
as good as what this file holds: its verdict on every outcome a scheduled run can have,
taken through main() as the workflow runs it; the questions it puts to GitHub, which a
fake gh answers only as asked; its reading of the price workflow, which must stay the
price job's own and be the one main() uses; and its own workflow, which must run it
and let it fail. Two cold checks of the watchdog found each of those open.

Run:  python3 tests/watchdog.test.py
"""
import contextlib
import datetime
import io
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


# Contracts, so they are literals, and none of them the watchdog's own tables.
# Cron's weekday names, Sunday first ("0 - 6 or SUN-SAT", GitHub's cron table).
CRON_DAYS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]
# What GitHub's API lists for a run's status and conclusion (the list-runs `status`
# filter's values, OpenAPI `workflow-run-status`, as the second cold check read them).
GITHUB_CONCLUSIONS = ["action_required", "cancelled", "failure", "neutral", "skipped", "stale", "success", "timed_out"]
GITHUB_UNFINISHED = ["in_progress", "queued", "requested", "waiting", "pending"]
# How late GitHub has started the price job's scheduled runs: 2026-09-21's and
# 2026-09-28's, each against its 06:00 slot.
GITHUB_DELAYS = [datetime.timedelta(hours=5, minutes=49, seconds=59),
                 datetime.timedelta(hours=6, minutes=44, seconds=59)]

SLOT = at("2026-09-28T06:17:00")     # a Monday, the price job's slot
NOW = "2026-09-29T07:43:00Z"         # the Tuesday the watchdog runs
MESSAGE = "price-refresh: update catalog prices and provenance"
FOUND_RED = "## Proposed updates\n- x\n\n---\n\n**The full suite is RED on this branch, and that is expected.**\n"
FOUND_GREEN = "## Proposed updates\n- x\n\n---\n\n**The full suite passed on this branch.** No recorded surface moved,\n"
NOTHING = "## Proposed updates\n\n## Could not validate this run\n- every source failed\n"
DELIVERED = None


def run(created="2026-09-28T12:44:59Z", status="completed", conclusion="success"):
    return {"id": 1, "created_at": created, "status": status, "conclusion": conclusion,
            "html_url": "https://github.com/o/r/actions/runs/1"}


def pr(state="open", merged_at=None, commits=((MESSAGE, "2026-09-28T12:45:33Z"),)):
    return {"state": state, "merged_at": merged_at, "html_url": "https://github.com/o/r/pull/39",
            "commits": [{"message": m, "date": d} for m, d in commits]}


# Every outcome a scheduled run can have: (run, its report, the price branch's pull
# requests, the words its problem must carry, or DELIVERED). A run that found prices is
# delivered only when an open or merged pull request from the price branch carries a
# commit the job made during that run: not one made before it started, even by a
# second, not a person's, and not last week's.
VERDICTS = {
    "no scheduled run since the slot": (None, None, [], "No scheduled run"),
    **{f"a run still {s}": (run(status=s, conclusion=None), None, [], "has not finished") for s in GITHUB_UNFINISHED},
    **{f"a run that ended {c}": (run(conclusion=c), None, [], f"ended {c}")
       for c in GITHUB_CONCLUSIONS if c != "success"},
    "a run that failed, with a report and a pull request": (run(conclusion="failure"), FOUND_RED, [pr()], "ended failure"),
    "a run that left no report": (run(), None, [], "left no report"),
    "a run that found nothing to propose": (run(), NOTHING, [], DELIVERED),
    "found, suite red, an open pull request carries it": (run(), FOUND_RED, [pr()], DELIVERED),
    "found, suite green, an open pull request carries it": (run(), FOUND_GREEN, [pr()], DELIVERED),
    "found, and a merged pull request carries it": (
        run(), FOUND_RED, [pr(state="closed", merged_at="2026-09-29T06:00:00Z")], DELIVERED),
    "found, suite red, no pull request": (run(), FOUND_RED, [], "no open or merged pull request"),
    "found, suite green, no pull request": (run(), FOUND_GREEN, [], "no open or merged pull request"),
    "found, and the pull request was closed unmerged": (
        run(), FOUND_RED, [pr(state="closed")], "no open or merged pull request"),
    "found, and the job's newest commit is last week's": (
        run(), FOUND_RED, [pr(commits=((MESSAGE, "2026-09-21T11:51:00Z"),))], "no open or merged pull request"),
    "found, and the job's newest commit came from an earlier hand run": (
        run(), FOUND_RED, [pr(commits=((MESSAGE, "2026-09-28T07:45:33Z"),))], "no open or merged pull request"),
    "found, and the job's newest commit is a second older than the run": (
        run(), FOUND_RED, [pr(commits=((MESSAGE, "2026-09-28T12:44:58Z"),))], "no open or merged pull request"),
    "found, and only a person's commit is newer than the run": (
        run(), FOUND_RED, [pr(commits=((MESSAGE, "2026-09-21T11:51:00Z"),
                                       ("Record the goldens", "2026-09-28T14:00:00Z")))],
        "no open or merged pull request"),
}


def verdict_agrees(got, want):
    return (got == []) == (want is DELIVERED) and (want is DELIVERED or (len(got) == 1 and want in got[0]))


print("\nThe verdict on each outcome of a scheduled run")


def check_every_outcome_gets_its_verdict():
    """Each outcome in VERDICTS, judged by problems() alone. Each problem must also
    say which it is: an issue that calls an unfinished run failed, or a failed one
    reportless, sends its reader the wrong way. Every conclusion and unfinished status
    GitHub lists is a case, derived, not a sample: a run GitHub concludes `skipped`
    passed as delivered in the watchdog's second cold check."""
    wrong = {case: got for case, (r, report, prs, want) in VERDICTS.items()
             if not verdict_agrees(got := wd.problems(SLOT, r, report, prs, MESSAGE), want)}
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
    judged yet and the one before it is, and the grace must outlast every delay GitHub
    has shown, with an hour to spare: at one hour, a watchdog run by hand after a slot
    reports a run GitHub is merely late on. Cron counts Sunday as 0 and Python counts
    Monday as 0; every weekday is checked, derived, and every weekday name GitHub
    accepts is read against cron's own table, not the watchdog's."""
    assert wd.GRACE >= max(GITHUB_DELAYS) + datetime.timedelta(hours=1), (
        f"the grace is {wd.GRACE}, and GitHub has started a scheduled run {max(GITHUB_DELAYS)} late")
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

test("the slot judged is the last one past a grace that outlasts GitHub's delays, on every weekday",
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


def watch_doc():
    """The one workflow that runs the watchdog."""
    hits = []
    for name in sorted(os.listdir(WORKFLOWS)):
        if name.endswith((".yml", ".yaml")):
            with open(os.path.join(WORKFLOWS, name), encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
            if any("tools/price_watchdog.py" in str(s.get("run", "")) for s in workflow_steps(doc)):
                hits.append(doc)
    assert len(hits) == 1, f"{len(hits)} workflows run the watchdog"
    return hits[0]


def schedule_entries(doc):
    on = doc.get(True, doc.get("on"))
    return on["schedule"]


def cron_of(doc):
    (entry,) = schedule_entries(doc)
    minute, hour, dom, month, dow = entry["cron"].split()
    assert dom == month == "*", entry
    return int(minute), int(hour), int(dow) if dow.isdigit() else CRON_DAYS.index(dow)


def check_it_reads_the_price_workflow_github_runs():
    """The watchdog searches the workflow text; GitHub parses it. The file, the
    schedule, the branch and the commit message the watchdog reads must be the ones
    the parsed workflow gives. And each must mean what the watchdog takes it to: a
    schedule entry is its cron alone, in UTC (a `timezone:` beside it moves the run to
    before the slot the watchdog computes), and the pull request's head is the branch
    as written (a `branch-suffix` makes every week's head another ref)."""
    name, doc = price_doc()
    found, text = wd.price_workflow()
    assert found == name, (found, name)
    assert wd.schedule_of(text) == cron_of(doc), (wd.schedule_of(text), cron_of(doc))
    (step,) = [s for s in workflow_steps(doc) if str(s.get("uses", "")).startswith(wd.PR_ACTION)]
    assert wd.branch_of(text) == step["with"]["branch"], (wd.branch_of(text), step["with"]["branch"])
    assert wd.commit_message_of(text) == step["with"]["commit-message"], wd.commit_message_of(text)
    assert "branch-suffix" not in step["with"], "the pull request's head is not the branch the watchdog reads"
    for which, entries in (("the price job's", schedule_entries(doc)), ("the watchdog's", schedule_entries(watch_doc()))):
        for entry in entries:
            assert set(entry) == {"cron"}, f"{which} schedule entry is {entry}: the watchdog reads cron alone, in UTC"

test("the watchdog reads the file, schedule, branch and commit message GitHub runs, as GitHub means them",
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
    red one, not merely in its source. The step must be gated on the gate, and the file
    it writes must be the one the job uploads, at the artifact's root: upload-artifact
    roots it at the paths' deepest common folder, so every path must sit directly in
    the runner's temp folder, or the report is never where the watchdog reads it."""
    _, doc = price_doc()
    steps = workflow_steps(doc)
    written = {o: body_output(steps, o) for o in ("success", "failure")}
    for outcome, text in written.items():
        assert any(marker in text for marker in wd.PROPOSED_MARKERS), (
            f"on SUITE={outcome} the body step writes none of the watchdog's words:\n{text[-400:]}")
    for marker in wd.PROPOSED_MARKERS:
        assert any(marker in text for text in written.values()), f"nothing writes {marker!r}"
    (body,) = [s for s in steps if s.get("id") == "body"]
    assert "steps.diff.outputs.changed == 'true'" in str(body.get("if", "")), body.get("if")
    (upload,) = [s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    assert upload["with"]["name"] == wd.REPORT_ARTIFACT, upload["with"]["name"]
    paths = [p.strip() for p in str(upload["with"]["path"]).splitlines() if p.strip()]
    assert f"${{{{ runner.temp }}}}/{wd.REPORT_FILE}" in paths, paths
    assert all(re.fullmatch(r"\$\{\{ runner\.temp \}\}/[^/]+", p) for p in paths), (
        f"every uploaded path must sit directly in the runner's temp folder: {paths}")

test("the watchdog reads the words the price job writes, from the root of the artifact it keeps",
     check_it_reads_the_words_the_body_step_writes)


print("\nThe watchdog's own workflow")


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
    run is past the grace, and before the next slot; the price job keeps its report at
    least a day longer than that. And the watchdog's workflow is pinned whole, as the
    head of the price job is: the first cold check made its step end `|| true`, gated
    the step and then the job on workflow_dispatch, took its token away, and pointed it
    at another repository, and each passed."""
    doc = watch_doc()
    after = watch_offset()
    assert wd.GRACE <= after < datetime.timedelta(days=7), f"the watchdog runs {after} after the slot"
    (upload,) = [s for s in workflow_steps(price_doc()[1]) if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    days = int(upload["with"].get("retention-days", 90))
    assert datetime.timedelta(days=days) >= after + datetime.timedelta(days=1), (
        f"the price job keeps its report {days} day(s) and the watchdog reads it {after} after the slot")
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


print("\nThe watchdog through gh: main() on every outcome, and the real process")

# GitHub, as a gh would answer it, from a state: the runs of the workflow named in the
# path, filtered by event and status only when asked; open pull requests unless asked
# for all, from any branch unless one is named; an artifact extracted into the directory
# only when named, into one of its own name otherwise; open issues unless asked for all.
# Used in-process for every outcome, and as the gh executable for the real process.
FAKE_GH_LOGIC = '''
import json, os


def answer(state, a):
    """(exit code, stdout) of `gh *a`, as GitHub would answer it from state."""
    fields = dict(a[i + 1].split("=", 1) for i, x in enumerate(a) if x == "-f")
    if a[:1] == ["api"]:
        path = a[3]
        if path == "repos/o/r/actions/runs" or path.startswith("repos/o/r/actions/workflows/"):
            runs = ([r for rs in state["runs"].values() for r in rs] if path == "repos/o/r/actions/runs"
                    else state["runs"].get(path.split("/")[5], []))
            runs = [r for r in runs if fields.get("event") in (None, r["event"])
                    and fields.get("status") in (None, r["status"], r["conclusion"])]
            return 0, json.dumps({"workflow_runs": runs})
        if path.startswith("repos/o/r/pulls/") and path.endswith("/commits"):
            return 0, json.dumps(state["commits"][path.split("/")[4]])
        if path == "repos/o/r/pulls":
            want = fields.get("state", "open")
            return 0, json.dumps([p for p in state["pulls"] if (want == "all" or p["state"] == want)
                                  and ("head" not in fields or fields["head"] == "o:" + p["head"]["ref"])])
        return 1, "unexpected api path " + path
    if a[:2] == ["run", "download"]:
        report = state["reports"].get(a[2])
        if report is None:
            return 1, ""
        where = a[a.index("--dir") + 1]
        if "--name" not in a:
            where = os.path.join(where, "price-check-report")
            os.makedirs(where)
        with open(os.path.join(where, "price-check-report.md"), "w") as fh:
            fh.write(report)
        return 0, ""
    if a[:2] == ["issue", "list"]:
        want = a[a.index("--state") + 1] if "--state" in a else "open"
        return 0, json.dumps([i for i in state.get("issues", []) if want == "all" or i["state"] == want])
    if a[:2] in (["issue", "create"], ["issue", "comment"]):
        return 0, ""
    return 1, "unexpected " + " ".join(a)
'''
_fake = {}
exec(FAKE_GH_LOGIC, _fake)
FAKE_GH = "#!/usr/bin/env python3" + FAKE_GH_LOGIC + '''
import sys
with open(os.environ["FAKE_GH_STATE"]) as fh:
    state = json.load(fh)
with open(os.environ["FAKE_GH_LOG"], "a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
code, out = answer(state, sys.argv[1:])
print(out)
sys.exit(code)
'''


def gh_run(created, event, conclusion="success", status="completed", run_id=1):
    return {"id": run_id, "created_at": created, "event": event, "status": status, "conclusion": conclusion,
            "html_url": f"https://github.com/o/r/actions/runs/{run_id}"}


def gh_commit(date, message=MESSAGE):
    return {"commit": {"message": message, "committer": {"date": date}}}


def world(the_run, report, prs, workflow_file, branch):
    """GitHub on Tuesday 2026-09-29 for one outcome, with every trap a wrong question
    falls into: last week's scheduled run, a hand run after this week's that found
    prices and delivered nothing, the watchdog's own run in progress, another branch's
    pull request carrying a commit with the job's message after the run, and a closed
    issue under the watchdog's title."""
    runs = [gh_run("2026-09-21T11:49:59Z", "schedule", run_id=9),
            gh_run("2026-09-28T14:00:00Z", "workflow_dispatch", run_id=2)]
    if the_run is not None:
        runs.append(gh_run(the_run["created_at"], "schedule", conclusion=the_run["conclusion"],
                           status=the_run["status"], run_id=1))
    pulls = [{"number": 50, "state": "open", "merged_at": None, "head": {"ref": "fix/other"}, "html_url": "u50"}]
    commits = {"50": [gh_commit("2026-09-28T13:00:00Z")]}
    for n, p in enumerate(prs, start=39):
        pulls.append({"number": n, "state": p["state"], "merged_at": p["merged_at"], "head": {"ref": branch},
                      "html_url": f"u{n}"})
        commits[str(n)] = [gh_commit(c["date"], c["message"]) for c in p["commits"]]
    return {"runs": {workflow_file: runs, "price-watchdog.yml": [
                gh_run("2026-09-29T07:43:00Z", "schedule", conclusion=None, status="in_progress", run_id=3)]},
            "reports": {"9": FOUND_RED, "2": FOUND_RED, **({"1": report} if report is not None else {})},
            "pulls": pulls, "commits": commits,
            "issues": [{"number": 6, "title": wd.ISSUE_TITLE, "state": "closed"}]}


def main_in_process(state, now=NOW, workflows=None):
    """wd.main() in this process, gh answered from state: (exit code, gh calls, printed)."""
    calls = []

    def gh(*args, check=True):
        calls.append(list(args))
        code, out = _fake["answer"](state, list(args))
        if check and code:
            raise subprocess.CalledProcessError(code, ["gh", *args], out)
        return subprocess.CompletedProcess(["gh", *args], code, out + "\n", "")

    saved = wd.gh, wd.WORKFLOWS
    wd.gh = gh
    if workflows:
        wd.WORKFLOWS = workflows
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            code = wd.main(["--repo", "o/r", *(["--now", now] if now else [])])
    finally:
        wd.gh, wd.WORKFLOWS = saved
    return code, calls, printed.getvalue()


def issue_calls(calls):
    return [c[:3] if c[1] == "comment" else c[:2] for c in calls if c[:1] == ["issue"] and c[1] != "list"]


def check_main_gives_every_outcome_its_verdict():
    """Every outcome in VERDICTS, through main() as the workflow runs it, against a
    GitHub that answers only the question asked and holds every trap a wrong one falls
    into. A delivered week exits 0 and opens no issue; any other exits 1, says which
    problem it is, and opens one issue (the only one under the title is closed). The
    second cold check made main() return 0 on a run that never started, one still going
    and one cancelled, read a missing report as an empty one, let a hand run stand in
    for a dropped scheduled one, and asked only for finished runs; problems() alone
    could see none of it."""
    price_file, text = wd.price_workflow()
    wrong = {}
    for case, (r, report, prs, want) in VERDICTS.items():
        code, calls, printed = main_in_process(world(r, report, prs, price_file, wd.branch_of(text)))
        good = (code == 0 and not issue_calls(calls)) if want is DELIVERED else (
            code == 1 and want in printed and issue_calls(calls) == [["issue", "create"]])
        if not good:
            wrong[case] = (code, issue_calls(calls), printed.strip()[-160:])
    assert not wrong, wrong

test("main() gives every outcome its verdict, against a GitHub that answers only what is asked",
     check_main_gives_every_outcome_its_verdict)


SYNTHETIC_WORKFLOW = """name: Weekly prices
on:
  schedule:
    - cron: "41 5 * * 3"
  workflow_dispatch: {}
jobs:
  prices:
    runs-on: ubuntu-latest
    steps:
      - name: Open a pull request
        uses: peter-evans/create-pull-request@v6
        with:
          branch: bots/prices
          commit-message: "prices: weekly refresh"
"""


def check_main_uses_what_it_reads():
    """main() must ask about the file, schedule, branch and commit message it read, not
    the ones the price workflow has today: the second cold check swapped each for its
    literal and every test passed, because the literals were the workflow's. So here the
    price job is another file, run Wednesdays at 05:41, on another branch with another
    message. A delivered week passes; a week whose only scheduled run came before this
    Wednesday's slot is a week with no scheduled run."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "weekly-prices.yml"), "w") as fh:
            fh.write(SYNTHETIC_WORKFLOW)
        now = "2026-10-01T18:00:00Z"   # Thursday; Wednesday 2026-09-30 05:41 is the slot

        def synthetic(runs, prs):
            state = world(None, None, [], "weekly-prices.yml", "bots/prices")
            state["runs"]["weekly-prices.yml"] = runs
            state["reports"]["1"] = FOUND_RED
            for n, (st, merged, date) in enumerate(prs, start=60):
                state["pulls"].append({"number": n, "state": st, "merged_at": merged,
                                       "head": {"ref": "bots/prices"}, "html_url": f"u{n}"})
                state["commits"][str(n)] = [gh_commit(date, "prices: weekly refresh")]
            return state

        delivered = synthetic([gh_run("2026-09-30T06:10:00Z", "schedule", run_id=1)],
                              [("open", None, "2026-09-30T06:11:00Z")])
        code, calls, printed = main_in_process(delivered, now=now, workflows=d)
        assert code == 0 and not issue_calls(calls), f"a delivered week of another price job: {code} {printed[-200:]}"
        early = synthetic([gh_run("2026-09-29T10:00:00Z", "schedule", run_id=1)],
                          [("open", None, "2026-09-29T10:01:00Z")])
        code, calls, printed = main_in_process(early, now=now, workflows=d)
        assert code == 1 and "No scheduled run" in printed, f"a run before this week's slot passed: {code} {printed[-200:]}"

test("main() asks about the file, schedule, branch and commit message it read",
     check_main_uses_what_it_reads)


def check_the_watchdog_as_a_process():
    """The real script, as the workflow starts it, with gh on PATH answering from a
    file: a week whose pull request was closed unmerged exits 1 and opens one issue,
    comments on the one already open instead of opening another, and never writes to a
    closed one. And one run goes through on the real clock, which no test did before the
    first cold check: a clock without a timezone would have crashed every real run."""
    with tempfile.TemporaryDirectory() as d:
        gh, state_path, log = os.path.join(d, "gh"), os.path.join(d, "state.json"), os.path.join(d, "calls.log")
        with open(gh, "w") as fh:
            fh.write(FAKE_GH)
        os.chmod(gh, os.stat(gh).st_mode | stat.S_IEXEC)

        def call(state, now=NOW):
            with open(state_path, "w") as fh:
                json.dump(state, fh)
            open(log, "w").close()
            done = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "price_watchdog.py"), "--repo", "o/r",
                                   *(["--now", now] if now else [])], capture_output=True, text=True, timeout=60,
                                  env=dict(os.environ, PATH=d + os.pathsep + os.environ.get("PATH", ""),
                                           FAKE_GH_STATE=state_path, FAKE_GH_LOG=log))
            with open(log) as fh:
                calls = [json.loads(line) for line in fh]
            return done.returncode, issue_calls(calls), done.stdout + done.stderr

        price_file, text = wd.price_workflow()
        closed = world(run(), FOUND_RED, [pr(state="closed")], price_file, wd.branch_of(text))
        rc, issues, out = call(closed)
        assert rc == 1 and issues == [["issue", "create"]], f"closed unmerged: {rc} {issues} {out[-300:]}"
        rc, issues, out = call(dict(closed, issues=[{"number": 7, "title": wd.ISSUE_TITLE, "state": "open"}]))
        assert rc == 1 and issues == [["issue", "comment", "7"]], f"the open issue was not added to: {issues}"

        slot = wd.slot_to_judge(datetime.datetime.now(datetime.timezone.utc), wd.schedule_of(text))
        started = (slot + datetime.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rc, issues, out = call(world(run(created=started), NOTHING, [], price_file, wd.branch_of(text)), now=None)
        assert rc == 0 and not issues, f"on the real clock: {rc} {issues} {out[-400:]}"

test("the watchdog as a process: one issue, added to once open, and the real clock",
     check_the_watchdog_as_a_process)


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
