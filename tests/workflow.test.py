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


def check_the_pr_is_not_gated_on_the_suite():
    sid = suite_step()["id"]
    pr = find(price_steps, uses=lambda v: v.startswith("peter-evans/create-pull-request"))
    assert f"steps.{sid}" not in pr.get("if", ""), (
        f"the pull-request step is conditioned on the suite again ({pr.get('if')}) — "
        f"a price move always reddens the golden, so this means it can never open one")

test("the pull request is not conditioned on the suite passing",
     check_the_pr_is_not_gated_on_the_suite)


def check_the_report_uploads_whatever_happened():
    up = find(price_steps, uses=lambda v: v.startswith("actions/upload-artifact"))
    assert up.get("if") == "always()", (
        f"the report upload is conditional ({up.get('if')!r}). It was once "
        f"`changed == 'false'`, which kept the report only on the runs that had "
        f"nothing to say and lost it on both runs that did")

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


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
