#!/usr/bin/env python3
"""Guard the CI workflows against the shape that made the price job useless.

The price job ran for days unable to do the one thing it exists for.
tests/golden/page.json records what every card displays, prices included,
verbatim, so any real price move turns the suite red — and the pull-request
step was gated on that suite. The first scheduled run applied its moves,
changed 32 recorded surfaces, failed the golden, skipped the PR step, and
discarded the report with it. Nothing in the tree could see that, because no
suite read the workflow files: a job that could only succeed by finding nothing
looked exactly like a job that worked, and only a Monday could tell them apart.

This file reads the workflows with a real YAML parser, and that is a correction
rather than a preference. It was first written with a hand-rolled reader that
split on fixed indentation, to keep the suite's dependencies at reportlab. A
cold check took that reader apart: a plain scalar continued onto a second line
folds into one expression for YAML and for GitHub, while the reader saw only
the first line — so `if: ${{ !cancelled() && ... 'true'` / `  && steps.suite.
outcome == 'success' }}` reinstated the original bug and passed. Worse, a step
spelled `-   name:` was invisible to the reader AND to the count check meant to
catch exactly that, because both were the same regex. A guard whose whole job
is to know what GitHub will do with this file cannot read it differently from
YAML. pyyaml is the sixth dependency and it is cheaper than being wrong.

Run:  python3 tests/workflow.test.py
"""
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = os.path.join(ROOT, ".github", "workflows")

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


def load(path):
    """Parse, and fail loudly on anything that is not a workflow. A file GitHub
    would reject must fail here too rather than be skipped."""
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    assert isinstance(doc, dict), f"{os.path.basename(path)}: not a mapping"
    assert isinstance(doc.get("jobs"), dict), f"{os.path.basename(path)}: no jobs"
    return doc


PRICE_PATH = os.path.join(WORKFLOWS, "price-refresh.yml")
PR_ACTION = "peter-evans/create-pull-request"
_cache = {}


def price():
    """Loaded on demand, never at import. Renaming the job used to raise KeyError
    at module level, outside any test() — a crash rather than a named failure,
    which is how a suite stops reporting what it found."""
    if "doc" not in _cache:
        _cache["doc"] = load(PRICE_PATH)
    return _cache["doc"]


def JOBS():
    return price()["jobs"]


def JOB():
    """The job that opens the pull request, found by what it DOES. Binding to the
    literal key `price-check` meant a second job in the same file — same triggers,
    "jobs ... run in parallel by default" — could carry the original bug verbatim
    with every rule below passing, because none of them ever looked at it."""
    hits = [(n, j) for n, j in JOBS().items()
            if any(str(s.get("uses", "")).startswith(PR_ACTION) for s in j.get("steps") or [])]
    assert len(hits) == 1, (
        f"expected exactly one job to open a pull request, found {[n for n, _ in hits]}. "
        f"Every rule in this file binds to that job; a second one is unguarded, and "
        f"two of them would race on the same branch.")
    return hits[0][1]


def STEPS():
    return JOB()["steps"]


def all_steps():
    """Every step of every job in this workflow. The rules that must hold
    workflow-wide — nothing else runs the suite, nothing pushes — use this."""
    for name, j in JOBS().items():
        for s in j.get("steps") or []:
            yield name, s


def label(step):
    return step.get("name") or step.get("uses") or "<unnamed>"


def only(pred, what):
    """Exactly one. Never zero, never two — both mean the file moved under a
    test that would otherwise still pass."""
    hits = [s for s in STEPS() if pred(s)]
    assert len(hits) == 1, f"expected exactly 1 step {what}, found {len(hits)}"
    return hits[0]


def shell(step):
    """The step's script with shell comments removed, so a sentence describing a
    safeguard can never stand in for the safeguard."""
    src = step.get("run") or ""
    return "\n".join(l for l in src.split("\n") if not l.strip().startswith("#"))


# A contract, so it is a literal. Matching "a step whose shell mentions
# tests/run.sh" let an `echo "the suite lives at tests/run.sh"` carry `id: suite`
# while the real step invoked "${RUNNER}.sh" — the PR body then reported the
# echo's outcome, always success, on every red run. It also made the pipefail
# rule vacuous in the same stroke, because the decoy has no pipe to check.
SUITE_SHELL = 'set -o pipefail\n./tests/run.sh 2>&1 | tee "$RUNNER_TEMP/suite.log"'


def suite_step():
    return only(lambda s: s.get("id") == "suite", "with id: suite")


def gate_step():
    return only(lambda s: s.get("id") == "diff", "with id: diff (the real gate)")


def uses_in(job, action):
    hits = [s for s in job["steps"] if str(s.get("uses", "")).startswith(action)]
    assert len(hits) == 1, f"expected exactly 1 step using {action}, found {len(hits)}"
    return hits[0]


def uses_step(action):
    return only(lambda s: str(s.get("uses", "")).startswith(action), f"using {action}")


def delivery_steps():
    """Everything after the suite. Derived from position: once the suite has run,
    all that is left is carrying the result out."""
    return STEPS()[STEPS().index(suite_step()) + 1:]


print("\nThe workflow reader understands the files it judges")
present = sorted(f for f in os.listdir(WORKFLOWS) if f.endswith((".yml", ".yaml")))


def check_every_workflow_parses():
    assert present, "no workflow files found"
    for f in present:
        doc = load(os.path.join(WORKFLOWS, f))
        for name, job in doc["jobs"].items():
            assert isinstance(job.get("steps"), list) and job["steps"], f"{f}:{name} has no steps"
            for st in job["steps"]:
                assert st.get("name") or st.get("uses"), f"{f}:{name} has a step with neither name nor uses"

test(f"every workflow parses and every step is named ({len(present)} files)",
     check_every_workflow_parses)


print("\nThe rules bind to the job that actually does the work")


def check_exactly_one_job_opens_a_pull_request():
    """JOB() asserts this, but as a named test rather than a side effect: it is
    the binding every other rule in this file depends on. A second job appended
    to this file inherits the same `schedule` and `workflow_dispatch` triggers —
    "A workflow run is made up of one or more `jobs`, which run in parallel by
    default" — so it needs no wiring at all, and until this rule existed it was
    entirely unguarded. It could carry the original deadlock verbatim."""
    JOB()

test("exactly one job in this workflow opens the pull request",
     check_exactly_one_job_opens_a_pull_request)


def check_no_job_can_be_skipped_without_anyone_noticing():
    """A job-level `if:` that is never true makes the whole job — suite, delivery,
    everything — stop running, and GitHub reports it as "skipped", not as a
    failure: "You can use the `jobs.<job_id>.if` conditional to prevent a job from
    running unless a condition is met ... Otherwise, the job will be marked as
    skipped". Nothing alarms anyone, which is the same silence this file exists to
    end. `strategy` is here for the same reason: a matrix that expands to nothing
    runs nothing."""
    for name, j in JOBS().items():
        assert "if" not in j, (
            f"job {name!r} carries a job-level if: {j['if']!r}. A condition that "
            f"stops being true retires the job silently — GitHub marks it skipped, "
            f"not failed. Gate inside the steps, where the tail rules can see it.")
        assert "strategy" not in j, (
            f"job {name!r} has a strategy/matrix; one that expands to no instances "
            f"runs nothing and reports nothing")

test("no job can be retired by a condition that reports as 'skipped'",
     check_no_job_can_be_skipped_without_anyone_noticing)


print("\nThe price job can deliver a result its own suite goes red on")

# The tail's conditions are a contract, not a pattern, so they are literals. A
# regex over these was how `!contains(steps.*.outcome, 'failure')` and a second
# folded line both passed while gating delivery on the suite after all.
DELIVERY_IF = {
    "${{ !cancelled() && steps.diff.outputs.changed == 'true' }}":
        "runs unless the workflow was cancelled",
    "always()":
        "runs no matter what, including on cancellation",
}


def check_the_suite_still_runs_and_is_identified():
    s = suite_step()
    assert s.get("id"), f"the suite step needs an id for other steps to read: {label(s)}"
    assert s.get("continue-on-error") is True, (
        f"the suite step is not continue-on-error, so a red suite marks the job "
        f"failed and every success()-gated step after it is skipped: {label(s)}")

test("the price job still runs the full suite, and says which step it is",
     check_the_suite_still_runs_and_is_identified)


def check_every_delivery_condition_is_one_of_the_two_permitted():
    """The whole tail, by literal. `!cancelled()` suppresses the implicit
    `success()` GitHub otherwise applies ("A default status check of `success()`
    is applied unless you include one of these functions"), so nothing failing
    above can skip delivery. Anything else here is a new condition nobody has
    reasoned about — including one that merely looks like these."""
    tail = delivery_steps()
    assert tail, "no steps follow the suite — nothing reports its result"
    for st in tail:
        cond = st.get("if")
        assert cond in DELIVERY_IF, (
            f"step {label(st)!r} has if: {cond!r}, which is not one of the two "
            f"conditions delivery is allowed to carry:\n"
            + "".join(f"    {k!r}\n      -> {v}\n" for k, v in DELIVERY_IF.items())
            + "  Adding a third is a decision, not a detail: a bare condition is "
              "implicitly success()-gated and skipped by any earlier failure.")

test("every step after the suite carries one of the two permitted conditions",
     check_every_delivery_condition_is_one_of_the_two_permitted)


def check_no_delivery_step_swallows_its_own_failure():
    """continue-on-error on the PR step turns "no pull request was opened" into
    a green run. The one step allowed to fail quietly is the suite."""
    for st in delivery_steps():
        assert not st.get("continue-on-error"), (
            f"step {label(st)!r} is continue-on-error, so its failure — including "
            f"the repository setting that forbids Actions from creating pull "
            f"requests — would report as a clean run that delivered nothing")

test("no delivery step reports success when it failed to deliver",
     check_no_delivery_step_swallows_its_own_failure)


def check_nothing_upstream_of_the_gate_runs_the_tests():
    """The tail rule starts after the suite, but the real gate is before it: the
    `diff` step carries no `if:`, so it is implicitly success()-gated, and
    anything that fails above it leaves `changed` unset and delivers nothing.
    Moving a test run up there rebuilds the deadlock outside the tail entirely."""
    upstream = STEPS()[:STEPS().index(gate_step())]
    for st in upstream:
        body = shell(st) + " " + str(st.get("uses", ""))
        assert "tests/" not in body, (
            f"step {label(st)!r} runs something under tests/ before the gate that "
            f"decides whether there is anything to propose. A failure there skips "
            f"the gate itself, so `changed` is never set and nothing is delivered "
            f"— the original deadlock, rebuilt upstream of every rule below.")

test("nothing upstream of the gate can redden the job with a test",
     check_nothing_upstream_of_the_gate_runs_the_tests)


def check_the_body_reports_the_suites_real_result():
    """outcome, never conclusion. The docs are explicit that "when a
    `continue-on-error` step fails, the `outcome` is `failure`, but the final
    `conclusion` is `success`" — so one word turns the PR body into a claim that
    the suite passed on every red run, with pipefail intact and nothing else
    disturbed. The PR's own check is not a backstop: a PR opened with
    GITHUB_TOKEN starts its runs in an approval-required state, so this sentence
    is the only channel that reports without a human clicking first."""
    want = "${{ steps.suite.outcome }}"
    carriers = [st for st in STEPS() if want in str(st.get("env") or "")]
    assert carriers, (
        f"no step reads {want} into its environment. Reading .conclusion instead "
        f"reports every red suite as a pass, and there is no backstop: a PR opened "
        f"with GITHUB_TOKEN starts its checks in an approval-required state, so the "
        f"PR body is the only channel that speaks without a human clicking first.")

test("the PR body is fed the suite's outcome, not its conclusion",
     check_the_body_reports_the_suites_real_result)


def check_the_suite_step_is_the_step_that_runs_the_suite():
    """Pinned whole, because every other rule about the suite trusts this binding.
    `id: suite` is what the PR body reads, what continue-on-error is asserted on,
    and what the delivery tail is measured from — so a step carrying that id and
    not running the tests turns all of them at once into statements about an echo.

    The pipefail is inside this literal rather than checked separately: the
    default shell is `bash -e {0}`, fail-fast but no pipefail, so a pipe hands the
    step tee's exit code and a red suite records as a pass."""
    s = suite_step()
    got = shell(s).strip()
    assert got == SUITE_SHELL, (
        f"the suite step's shell is not what this file guarantees.\n"
        f"  expected: {SUITE_SHELL!r}\n"
        f"  found:    {got!r}\n"
        f"Change it deliberately here if the change is deliberate — everything "
        f"else about the suite is asserted through this one binding.")
    others = [f"{j}:{label(st)}" for j, st in all_steps()
              if st is not s and "tests/run.sh" in shell(st)]
    assert not others, (
        f"another step names tests/run.sh: {others}. Only one step may, or "
        f"identifying the suite by what it runs stops being possible.")


test("the step carrying id: suite is the one that runs the suite",
     check_the_suite_step_is_the_step_that_runs_the_suite)


print("\nThe delivery steps have what they need to deliver")

WRITE = re.compile(r"(?:>>?|\btee\b(?:\s+-\w+)*|--report-out)\s*\"?(?:\$RUNNER_TEMP|\$\{\{\s*runner\.temp\s*\}\})/([\w.\-]+)")
EXPR = re.compile(r"\$\{\{(.*?)\}\}", re.S)
ANY_TEMP = re.compile(r"(?:\$RUNNER_TEMP|\$\{\{\s*runner\.temp\s*\}\})/([\w.\-]+)")


def check_the_job_can_push_a_branch_and_open_a_pr():
    """The workflow-level default is `contents: read`. Without the job-level
    block, create-pull-request has a token that can neither push the branch nor
    open the PR — and nothing about the steps looks any different."""
    perms = JOB().get("permissions") or {}
    for key in ("contents", "pull-requests"):
        assert perms.get(key) == "write", (
            f"the price-check job no longer grants {key}: write (found "
            f"{perms.get(key)!r}). The PR step would fail on its token.")

test("the job still grants itself the permissions the PR step needs",
     check_the_job_can_push_a_branch_and_open_a_pr)


def check_every_temp_file_handed_to_an_action_is_one_the_job_writes():
    """Written, not merely mentioned. A path that only ever appears in a `tail`
    is read by the body step and produced by nobody: the artifact then retains
    nothing and reports success, because `if-no-files-found` is `warn` — which it
    must stay, since suite.log legitimately does not exist on a quiet run."""
    written = set()
    for st in STEPS():
        written.update(WRITE.findall(shell(st)))
    assert written, "no step writes anything under RUNNER_TEMP any more"
    handed = {}
    for st in STEPS():
        if not st.get("uses"):
            continue
        for f in ANY_TEMP.findall(yaml.safe_dump(st.get("with") or {})):
            handed.setdefault(f, label(st))
    assert handed, "no action is handed a RUNNER_TEMP path — is the report still carried?"
    for f, who in sorted(handed.items()):
        assert f in written, (
            f"{who!r} is handed {f!r}, which no step in this job writes (written: "
            f"{sorted(written)}). The PR step throws on a missing body-path, and "
            f"the upload step reports success with an empty artifact.")

test("every RUNNER_TEMP path handed to an action is one the job's own shell writes",
     check_every_temp_file_handed_to_an_action_is_one_the_job_writes)


def check_the_report_is_retained_by_an_upload_that_runs_after_the_suite():
    """The requirement stated directly: the file the PR body is built from must
    also reach an artifact. Every other rule here constrains the CONDITIONS of
    whatever steps happen to be present, so replacing the upload step with an
    `echo` satisfied all of them — its `if: always()` was still permitted, the PR
    step still handed over a body-path, and the report was simply never retained.
    Found by re-running round 1's corpus against this rewrite, which is what a
    corpus is for."""
    pr = uses_step("peter-evans/create-pull-request")
    body = ANY_TEMP.findall(str((pr.get("with") or {}).get("body-path", "")))
    assert body, "the PR step no longer carries a body-path"
    up = uses_step("actions/upload-artifact")        # exactly one, or this fails
    assert up in delivery_steps(), (
        "the upload does not run after the suite, so every rule about the tail "
        "stops covering it")
    kept = ANY_TEMP.findall(yaml.safe_dump(up.get("with") or {}))
    assert body[0] in kept, (
        f"the report the PR body is built from ({body[0]}) is not among the files "
        f"the artifact retains ({kept}). On a run whose PR step fails, that report "
        f"exists nowhere afterwards — which is how 2026-09-21's was lost.")

test("the report the PR body is built from is also retained as an artifact",
     check_the_report_is_retained_by_an_upload_that_runs_after_the_suite)


def check_no_with_block_hides_a_path_behind_an_expression():
    """The RUNNER_TEMP rule reads filenames with a regex that expects
    `runner.temp` to be followed immediately by `/name`. `${{ format('{0}/x.log',
    runner.temp) }}` is valid, documented syntax — "Replaces values in the
    `string` ... Variables in the `string` are specified using the `{N}` syntax"
    — and resolves at runtime to a real path the regex never sees, so a wrong
    filename passes unchecked and the artifact silently retains nothing under
    `if-no-files-found: warn`.

    So rather than teach the regex more syntax it can be outrun by, this rejects
    any expression in a `with:` block that this file cannot evaluate. That is the
    rule the round-1 reader broke by guessing instead of failing."""
    for jobname, st in all_steps():
        if not st.get("uses"):
            continue
        for e in EXPR.findall(yaml.safe_dump(st.get("with") or {})):
            assert e.strip() == "runner.temp", (
                f"{label(st)!r} passes an expression this file cannot evaluate: "
                f"${{{{{e}}}}}. Only a bare `runner.temp` is permitted in a with: "
                f"block, because the path rules read these values literally — "
                f"anything else can name a file nothing writes and pass.")

test("no with: block hides a path behind an expression",
     check_no_with_block_hides_a_path_behind_an_expression)


print("\nThe job proposes; a person decides")


def check_the_bot_never_commits_the_golden():
    """Regenerating the golden is the review — it is how a person says the moved
    display strings moved for the right reason. A bot that regenerates it is a
    bot approving its own price changes."""
    pr = uses_step("peter-evans/create-pull-request")
    paths = [p for p in str((pr.get("with") or {}).get("add-paths", "")).split("\n") if p.strip()]
    assert paths, "the PR step no longer lists add-paths, so it commits everything"
    bad = [p for p in paths if "golden" in p or re.search(r"tests/[*]", p)]
    assert not bad, f"the price PR would carry {bad} — the bot regenerating its own record"
    # Command position, not the substring. The body step *prints*
    # `UPDATE_GOLDEN=1 node tests/model.test.js` into the PR as the instruction a
    # reviewer follows; matching the text condemned it. That is the third time in
    # this file a check has read prose where it meant an instruction — after the
    # step comment naming tests/golden and the one explaining pipefail — so the
    # rule is the same each time: shell() drops comments, and a claim about what
    # the shell DOES must be anchored to where a command can actually start.
    for st in STEPS():
        assert not re.search(r"^\s*UPDATE_GOLDEN=", shell(st), re.M), (
            f"step {label(st)!r} regenerates the golden inside the job")

test("the price PR never carries a regenerated golden", check_the_bot_never_commits_the_golden)


def check_it_still_never_pushes_or_commits_by_hand():
    pr = uses_step("peter-evans/create-pull-request")
    branch = (pr.get("with") or {}).get("branch")
    assert branch and branch not in ("master", "main"), f"the price job targets {branch!r}"
    for jobname, st in all_steps():
        body = shell(st)
        assert not re.search(r"\bgit\s+push\b", body), (
            f"a step pushes directly: {jobname}:{label(st)}")
        assert not re.search(r"\bgit\s+commit\b", body), (
            f"a step commits by hand: {jobname}:{label(st)} — the PR action is what "
            f"commits, and it is what add-paths constrains")

test("the price job still opens a pull request rather than pushing",
     check_it_still_never_pushes_or_commits_by_hand)


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
