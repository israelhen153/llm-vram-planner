#!/usr/bin/env python3
"""Guard the CI workflows against the shape that made the price job useless.

tests/run.sh promises node + python3 + reportlab and nothing else, and
.github/workflows/tests.yml installs exactly that, so this file may not reach
for a YAML library. What it uses instead is not a YAML parser and does not try
to be one: it splits a file whose shape this repo owns into step blocks and
reads the keys at one fixed indent. The rule that makes that safe is that it
must FAIL on anything it cannot account for, never shrug and pass — a checker
that quietly judges nothing is the exact defect tests/sabotage was built after.

What is pinned here, and why it is worth pinning: the price-refresh job ran for
days unable to do the one thing it exists for. tests/golden/page.json records
what every card displays, prices included, verbatim, so any real price move
turns the suite red — and the pull-request step was gated on that suite. The
first scheduled run applied its moves, changed 32 recorded surfaces, failed the
golden, skipped the PR step, and discarded the report with it. Nothing was
wrong with the code it ran. The job was simply built so that finding something
and reporting something were mutually exclusive, and only a calendar could
reveal it. These assertions are the thing that notices next time.

Run:  python3 tests/workflow.test.py
"""
import os
import re
import sys

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


# --- the reader -------------------------------------------------------------
# Steps live at a fixed depth: `      - key: value`, continued by `        key:`.
# Anything deeper is a step's own content (a shell `if`, a printf) and is never
# mistaken for a key, because the depth is matched exactly rather than stripped.
STEP_START = re.compile(r"^      - (\S[^:]*):(.*)$")
STEP_KEY = re.compile(r"^        (\S[^:]*):(.*)$")
DEEPER = re.compile(r"^         ")


def read_steps(path):
    """Every step in every job of one workflow file, as {key: first-line-value}
    dicts with the raw block kept alongside. Raises if the file does not have
    the shape this repo writes."""
    text = open(path, encoding="utf-8").read()
    lines = text.split("\n")
    steps, cur, block = [], None, []

    def flush():
        if cur is not None:
            cur["_block"] = "\n".join(block)
            steps.append(cur)

    for line in lines:
        m = STEP_START.match(line)
        if m:
            flush()
            cur, block = {m.group(1).strip(): m.group(2).strip()}, [line]
            continue
        if cur is None:
            continue
        if DEEPER.match(line) or not line.strip():
            block.append(line)
            continue
        m = STEP_KEY.match(line)
        if m:
            block.append(line)
            cur[m.group(1).strip()] = m.group(2).strip()
            continue
        # Dedented out of the steps list entirely (a new job, a top-level key).
        flush()
        cur, block = None, []
    flush()

    # The reader must not silently return a short list. Every step in these
    # files is introduced by `- name:` or `- uses:`; if the count it found
    # disagrees with the count in the raw text, it misread the file.
    expected = len(re.findall(r"^      - (?:name|uses):", text, re.M))
    assert steps, f"{os.path.basename(path)}: no steps found at all"
    assert len(steps) == expected, (
        f"{os.path.basename(path)}: read {len(steps)} steps but the file starts "
        f"{expected} — the reader no longer understands this file's shape")
    for s in steps:
        assert "name" in s or "uses" in s, f"a step has neither name nor uses: {s}"
    return steps


def job_keys(path, job):
    """The job's own keys — `permissions:` and friends — which the steps reader
    above never sees. A cold check removed the whole block and every assertion
    in this file still passed, because all of them were about steps."""
    text = open(path, encoding="utf-8").read()
    m = re.search(rf"^  {re.escape(job)}:$", text, re.M)
    assert m, f"{os.path.basename(path)}: no job named {job}"
    body, out, key = text[m.end():], {}, None
    for line in body.split("\n"):
        if line.strip() and not line.startswith("    "):
            break                       # dedented out of this job
        k = re.match(r"^    (\S[^:]*):(.*)$", line)
        if k:
            key = k.group(1).strip()
            out[key] = k.group(2).strip()
            continue
        sub = re.match(r"^      (\S[^:]*):\s*([^#]*)", line)
        if sub and key:
            out[f"{key}.{sub.group(1).strip()}"] = sub.group(2).strip()
    return out


def label(step):
    return step.get("name") or step.get("uses")


def find(steps, **want):
    """The one step matching every predicate, or an assertion. Never zero, never
    two: both mean the file moved under a test that would otherwise still pass."""
    hits = [s for s in steps
            if all(pred(s.get(k, "") if k != "_block" else s["_block"])
                   for k, pred in want.items())]
    assert len(hits) == 1, f"expected exactly 1 step matching {list(want)}, found {len(hits)}"
    return hits[0]


# --- the reader reads both files --------------------------------------------
print("\nThe workflow reader understands the files it judges")
present = sorted(f for f in os.listdir(WORKFLOWS) if f.endswith((".yml", ".yaml")))


def check_every_workflow_parses():
    assert present, "no workflow files found"
    for f in present:
        steps = read_steps(os.path.join(WORKFLOWS, f))
        assert steps, f"{f}: parsed to no steps"

test(f"every workflow in .github/workflows parses into named steps ({len(present)} files)",
     check_every_workflow_parses)

PRICE = os.path.join(WORKFLOWS, "price-refresh.yml")
price_steps = read_steps(PRICE)


# --- the invariant this file exists for -------------------------------------
print("\nThe price job can deliver a result its own suite goes red on")


def suite_step():
    return find(price_steps, _block=lambda b: "tests/run.sh" in b)


def run_script(step):
    """The step's shell, and only the shell — no comments, no surrounding prose.
    The `# pipefail is load-bearing` comment sitting beside this very script is
    why: a check that reads the whole block lets a sentence about a safeguard
    stand in for the safeguard, and passes after it is deleted."""
    out, inside = [], False
    for line in step["_block"].split("\n"):
        if re.match(r"^        run: \|\s*$", line):
            inside = True
            continue
        if not inside:
            continue
        if line.strip() and not line.startswith("          "):
            break
        if line.strip().startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


def add_paths():
    """The files the bot is allowed to commit — the `add-paths:` block itself,
    not the step's prose. Reading the whole block would let a comment that names
    a file stand in for permission to commit it."""
    pr = find(price_steps, uses=lambda v: v.startswith("peter-evans/create-pull-request"))
    out, inside = [], False
    for line in pr["_block"].split("\n"):
        if re.match(r"^          add-paths: \|\s*$", line):
            inside = True
            continue
        if not inside:
            continue
        if not re.match(r"^            \S", line):
            break
        if line.strip().startswith("#"):
            continue
        out.append(line.strip())
    return out


def check_the_suite_still_runs():
    """Everything below is about the suite not blocking delivery. Deleting the
    suite would satisfy all of it, so pin that it still runs first."""
    s = suite_step()
    assert "id" in s, f"the suite step needs an id for the other steps to read: {label(s)}"

test("the price job still runs the full suite on what it proposes", check_the_suite_still_runs)


def check_a_red_suite_does_not_abort_the_job():
    """Without this, a failing step skips every later step that carries an
    ordinary `if:` — which is the entire delivery path below."""
    s = suite_step()
    assert s.get("continue-on-error") == "true", (
        f"the suite step is not continue-on-error, so a red suite aborts the job "
        f"before it can open the PR or upload the report: {label(s)}")

test("a red suite does not abort the job before it reports anything",
     check_a_red_suite_does_not_abort_the_job)


def delivery_steps():
    """Every step from the suite onwards except the suite itself. Derived from
    position, not named: once the suite has run, everything left exists to carry
    the result out, and a list written by hand would miss the step someone adds
    next. The PR step and the upload step are checked by identity below as well,
    so this cannot go vacuous by the steps being renamed."""
    idx = price_steps.index(suite_step())
    return price_steps[idx + 1:]


# A status check function suppresses the implicit success(); these are the two
# that still run after something above has failed. cancelled() and failure() do
# not qualify — they would make delivery conditional on the wrong thing.
SURVIVES_FAILURE = re.compile(r"always\(\)|!\s*cancelled\(\)")


def check_delivery_cannot_be_skipped_by_an_earlier_failure():
    """The one that matters, and the one the original bug came in through.

    GitHub applies "a default status check of `success()` ... unless you include
    one of these functions", so `if: steps.diff.outputs.changed == 'true'` really
    means `success() && steps.diff.outputs.changed == 'true'`. Any step that fails
    anywhere above therefore skips it. Gating the suite was only the *first* way to
    build that deadlock; a later step doing `exit 1` over anything at all rebuilds
    it without touching the PR step's own condition, which is what a check that
    reads only that condition cannot see.

    So the rule is positional and applies to the whole tail: nothing after the
    suite may be skippable by a failure before it."""
    tail = delivery_steps()
    assert tail, "no steps follow the suite — nothing reports its result"
    for st in tail:
        cond = st.get("if", "")
        assert SURVIVES_FAILURE.search(cond), (
            f"step {label(st)!r} runs only on success() (if: {cond or '<none>'}). "
            f"A failure anywhere above it — including one added later, guarding "
            f"something unrelated — will silently skip it, which is the deadlock "
            f"this job was rebuilt to escape. Use `!cancelled() && ...`.")

test("nothing after the suite can be skipped by a failure before it",
     check_delivery_cannot_be_skipped_by_an_earlier_failure)


def check_nothing_downstream_conditions_on_the_suite_result():
    """Reporting the result is not the same as obeying it. A step that *reads*
    steps.suite.outcome to choose its wording is the point; one that reads it in
    its `if:` is a gate wearing a different hat — and gating only the step that
    explains the red suite is the quietest version, because the PR still opens
    and simply arrives with no explanation of why it is red."""
    sid = suite_step()["id"]
    for st in delivery_steps():
        assert f"steps.{sid}." not in st.get("if", ""), (
            f"step {label(st)!r} is conditioned on the suite ({st['if']}). A price "
            f"move always reddens the golden, so this step would never run on the "
            f"runs this job exists for.")

test("no step downstream of the suite is conditioned on the suite's result",
     check_nothing_downstream_conditions_on_the_suite_result)


def check_the_pr_step_and_the_upload_step_are_in_that_tail():
    """Keeps the positional rule above honest: if either moved before the suite,
    every assertion about the tail would still pass and mean nothing."""
    tail = delivery_steps()
    for what in ("peter-evans/create-pull-request", "actions/upload-artifact"):
        assert any(st.get("uses", "").startswith(what) for st in tail), (
            f"{what} no longer runs after the suite, so the tail rule no longer covers it")

test("the pull request and the upload are both in that tail",
     check_the_pr_step_and_the_upload_step_are_in_that_tail)


def check_the_report_uploads_whatever_happened():
    """Narrower than the tail rule above and kept beside it on purpose: this one
    names the incident. `always()` rather than `!cancelled()` here because a
    report from a cancelled run is still worth keeping."""
    up = find(price_steps, uses=lambda v: v.startswith("actions/upload-artifact"))
    assert up.get("if") == "always()", (
        f"the report upload is conditional ({up.get('if')!r}). It was once "
        f"`changed == 'false'`, which kept the report only on the runs that had "
        f"nothing to say and lost it on both runs that did — 2026-09-17's survives "
        f"only in closed PR #12's body, and 2026-09-21's not at all")

test("the report is uploaded on every run, not only the quiet ones",
     check_the_report_uploads_whatever_happened)


def check_the_piped_suite_cannot_report_a_false_pass():
    """The step pipes the suite into tee so the log can reach the PR body. The
    default shell is `bash -e`, not pipefail, so without pipefail tee's 0 is the
    step's status and every red suite records as a pass — the PR body would then
    say the suite passed while the branch is red. Classifying on an exit code
    that was never the suite's is how the sabotage corpus reported hundreds of
    false catches."""
    script = run_script(suite_step())
    assert "tests/run.sh" in script, "the suite step's shell no longer runs the suite"
    if "|" not in script:
        return  # not piped; nothing stands between the suite and the exit code
    assert "pipefail" in script, (
        "the suite's output is piped without pipefail, so the pipe's last command "
        "supplies the exit code and every red suite records as a pass")

test("piping the suite's output cannot turn a red suite into a reported pass",
     check_the_piped_suite_cannot_report_a_false_pass)


print("\nThe job proposes; a person decides")


def check_the_bot_never_commits_the_golden():
    """Regenerating the golden is the review — it is how a person says the moved
    display strings moved for the right reason. A bot that regenerates it is a
    bot that approves its own price changes."""
    paths = add_paths()
    assert paths, "the pull-request step no longer lists add-paths, so it commits everything"
    carried = [p for p in paths if "golden" in p]
    assert not carried, (
        f"the price PR now carries {carried} — the bot would be regenerating "
        f"the record that exists to make its own changes visible")

test("the price PR never carries a regenerated golden", check_the_bot_never_commits_the_golden)


def check_it_still_never_pushes_to_master():
    pr = find(price_steps, uses=lambda v: v.startswith("peter-evans/create-pull-request"))
    m = re.search(r"^          branch: (\S+)", pr["_block"], re.M)
    assert m, "the pull-request step no longer names the branch it pushes to"
    assert m.group(1) not in ("master", "main"), f"the price job targets {m.group(1)}"
    for s in price_steps:
        assert not re.search(r"git push", s["_block"]), (
            f"a step pushes directly: {label(s)}")

test("the price job still opens a pull request rather than pushing",
     check_it_still_never_pushes_to_master)


print("\nThe delivery steps have what they need to deliver")

TEMP_REF = re.compile(r"(?:\$RUNNER_TEMP|\$\{\{\s*runner\.temp\s*\}\})/([\w.\-]+)")


def check_the_job_can_push_a_branch_and_open_a_pr():
    """The workflow-level default is `contents: read`. Without the job-level
    block, create-pull-request has a token that cannot push the branch or open
    the PR — and nothing about the steps looks any different."""
    keys = job_keys(PRICE, "price-check")
    for need, why in (("permissions.contents", "write"),
                      ("permissions.pull-requests", "write")):
        assert keys.get(need) == why, (
            f"the price-check job no longer grants {need}: {why!r} (found "
            f"{keys.get(need)!r}). The workflow-level default is contents: read, "
            f"so the PR step would fail on its token, not on its inputs.")

test("the job still grants itself the permissions the PR step needs",
     check_the_job_can_push_a_branch_and_open_a_pr)


def check_every_temp_file_handed_to_an_action_is_one_the_job_writes():
    """body-path and the artifact's path: are filenames, and nothing validates a
    filename. A typo in body-path fails create-pull-request outright on every
    run with something to propose; a typo in the artifact path uploads an empty
    artifact and reports success, because if-no-files-found is `warn` — which it
    must stay, since suite.log legitimately does not exist on a quiet run.

    Derived both ways rather than pinned: the shell's filenames are whatever the
    scripts name, and the actions' filenames must be drawn from that set."""
    in_shell, handed_over = set(), {}
    for st in price_steps:
        for f in TEMP_REF.findall(run_script(st)):
            in_shell.add(f)
    assert in_shell, "no step writes anything under RUNNER_TEMP any more"
    for st in price_steps:
        if not st.get("uses"):
            continue
        for f in TEMP_REF.findall(st["_block"]):
            handed_over.setdefault(f, label(st))
    assert handed_over, "no action is handed a RUNNER_TEMP path — is the report still carried?"
    for f, who in sorted(handed_over.items()):
        assert f in in_shell, (
            f"{who!r} is handed {f!r}, which no step in this job ever writes. "
            f"Nothing validates these filenames: the PR step throws on a missing "
            f"body-path, and the upload step reports success with an empty artifact.")

test("every RUNNER_TEMP path handed to an action is one the job's own shell writes",
     check_every_temp_file_handed_to_an_action_is_one_the_job_writes)


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
