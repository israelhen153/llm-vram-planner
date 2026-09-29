#!/usr/bin/env python3
"""The sabotage corpus can still reach the engine text it attacks.

tests/sabotage/ reintroduces fixed bugs by finding verbatim excerpts of the
engine files — tests/sabotage/anchors.py — and replacing them with broken
versions. When the engine changes an excerpted line, every sabotage anchored on
it refuses to apply. That is by design and it is loud, but only when someone
runs the corpus, which takes twenty minutes and is not part of this suite. So
any pull request could move engine text out from under the corpus and its own
CI would stay green.

That is what happened to fix/cost-provenance: extending getSpec to carry
priceSource broke JS_GETSPEC, and four sabotages with it, and nothing said so.
This runs in under a second, and it fails in the pull request that moved the
text — which is when whoever moved it still knows what it became.

The excerpts are not the only way a driver reaches the engine: a driver can
quote it inline, and three did (E20, Q5 and Q6), so a note rewritten on
feat/rocm-guidance took them out while every excerpt here still matched. The
last checks therefore apply every sabotage itself, in memory, as the harness
would.

Deliberately NOT one of the suites that judge a sabotage (harness.JUDGING_SUITES):
every sabotage edits the engine, so this would go red under all of them and
report each one as caught whatever the real suites said — the same reason
tests/assets.test.py is excluded.

Run:  python3 tests/corpus.test.py
"""
import importlib.util
import os
import re
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "tests", "sabotage"))
import anchors  # noqa: E402

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


# The prefix says which file an excerpt was taken from. PRICE_/GPUS_ added by
# fix/cost-provenance's engine_r4_cost_provenance.py, which anchors sabotages
# on tools/price_check.py (the writer) and data/gpus.json (the catalog) as
# well as the three engine files above.
PREFIX_FILE = {"JS_": "index.html", "PY_": "generate_report.py", "SYNC_": "tools/sync_data.py",
               "PRICE_": "tools/price_check.py", "GPUS_": "data/gpus.json"}
PATHS = {"INDEX_HTML", "REPORT_PY", "SYNC_PY", "PRICE_PY", "GPUS_JSON"}
ENGINE = {f: open(os.path.join(ROOT, f), encoding="utf-8").read() for f in PREFIX_FILE.values()}
NAMES = sorted(n for n in vars(anchors) if n.isupper() and n != "PAYLOAD_PARTS")
PARTS = set(anchors.PAYLOAD_PARTS)


def excerpts():
    return [n for n in NAMES if n not in PATHS and n not in PARTS]


print("\nEvery constant in tests/sabotage/anchors.py is accounted for")


def check_every_constant_is_classified():
    """An excerpt, a payload part, or one of the three paths — nothing else. A
    constant this file cannot place would be skipped by every check below."""
    for n in NAMES:
        if n in PATHS or n in PARTS:
            continue
        assert any(n.startswith(p) for p in PREFIX_FILE), (
            f"{n} has no prefix naming its source file ({', '.join(PREFIX_FILE)}) and is not "
            f"declared in PAYLOAD_PARTS, so nothing can check it")
        assert isinstance(getattr(anchors, n), str), (
            f"{n} is a {type(getattr(anchors, n)).__name__}, not text — if it is a part sabotages "
            f"are built from rather than an excerpt, declare it in PAYLOAD_PARTS")
    missing = PARTS - set(NAMES)
    assert not missing, f"PAYLOAD_PARTS names constants that do not exist: {sorted(missing)}"

test("every constant is an excerpt, a declared payload part, or a path",
     check_every_constant_is_classified)


def check_the_paths_are_the_engine_files():
    for n in PATHS:
        path = getattr(anchors, n)
        assert path in PREFIX_FILE.values() and os.path.exists(os.path.join(ROOT, path)), (
            f"{n} = {path!r}, which is not one of the engine files sabotages edit")

test("the path constants name the engine files", check_the_paths_are_the_engine_files)


print("\nThe corpus can still reach what it attacks")


def check_every_excerpt_still_occurs_in_its_file():
    """The one this file exists for."""
    stale = []
    for n in excerpts():
        f = next(PREFIX_FILE[p] for p in PREFIX_FILE if n.startswith(p))
        if getattr(anchors, n) not in ENGINE[f]:
            first = getattr(anchors, n).strip().split("\n")[0][:70]
            stale.append(f"{n} (in {f}; begins {first!r})")
    assert not stale, (
        f"{len(stale)} excerpt(s) no longer occur in the engine, so every sabotage anchored on "
        f"them refuses to apply:\n         " + "\n         ".join(stale) +
        "\n       Update the excerpt in tests/sabotage/anchors.py to the engine's new text, in the "
        "same pull request that changed it.")

test(f"every excerpt still occurs verbatim in the file its prefix names ({len(excerpts())} excerpts)",
     check_every_excerpt_still_occurs_in_its_file)


def check_no_payload_part_is_already_in_the_engine():
    """A payload part is what a sabotage inserts. Found in the engine, the engine
    already carries the defect — and the sabotage built from it proves nothing."""
    found = [n for n in PARTS if isinstance(getattr(anchors, n), str)
             and any(getattr(anchors, n) in text for text in ENGINE.values())]
    assert not found, (
        f"payload parts already present in the engine: {found}. Either the engine now "
        f"contains a defect a sabotage exists to insert, or these are excerpts misdeclared "
        f"as payload parts to get past the check above")

test("no payload part is already present in the engine", check_no_payload_part_is_already_in_the_engine)


print("\nEvery sabotage still applies, wherever its excerpt is written")

# The drivers, found the way chain.sh finds them.
SABOTAGE_DIR = os.path.join(ROOT, "tests", "sabotage")
DRIVERS = sorted(f for f in os.listdir(SABOTAGE_DIR) if re.fullmatch(r"(engine|workflow)_\w+\.(py|sh)", f))
# A shell driver can't be read as data. The one there is gets a check of its own
# below, and a second one fails that check until it has one too.
SHELL_DRIVERS = ["engine_r1_perfkey_typo.sh"]


def sabotages_of(driver):
    """A driver's sabotages, read as data. Every driver builds S at import and ends
    by handing it to run_driver(), which does nothing while this runs: nothing is
    applied and no suite runs, for a driver another driver imports as well."""
    import harness
    real, harness.run_driver = harness.run_driver, lambda sabotages: None
    try:
        spec = importlib.util.spec_from_file_location(driver[:-3], os.path.join(SABOTAGE_DIR, driver))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        harness.run_driver = real
    return module.S


def why_it_would_not_apply(spec):
    """What harness.apply_edits() would refuse this sabotage for on the tree as it
    is, or None. The same count check, edit by edit and in order, on copies."""
    import harness
    files = {}
    for f, old, new, count in harness.edits_of(spec):
        if f not in files:
            try:
                files[f] = open(os.path.join(ROOT, f), encoding="utf-8").read()
            except OSError as e:
                return f"{f}: {e.strerror}"
        n = files[f].count(old)
        if n != count:
            return f"{f}: expected {count} occurrence(s) of {old[:70]!r}, found {n}"
        files[f] = files[f].replace(old, new)
    return None


PY_DRIVERS = {d: sabotages_of(d) for d in DRIVERS if d.endswith(".py")}


def check_every_sabotage_still_applies():
    """The check the excerpts can't give. Only a sabotage's edits are checked: the
    catalog re-sync some of them run afterwards is not."""
    empty = [d for d, sabotages in PY_DRIVERS.items() if not sabotages]
    assert not empty, f"drivers with no sabotages, so nothing here checks them: {empty}"
    stale = []
    for d, sabotages in PY_DRIVERS.items():
        for name, spec in sabotages.items():
            why = why_it_would_not_apply(spec)
            if why:
                stale.append(f"{d}: {name}: {why}")
    assert not stale, (
        f"{len(stale)} sabotage(s) no longer apply, so the corpus would skip them:\n         "
        + "\n         ".join(stale) +
        "\n       Point each at the engine's new text in the pull request that moved it. An "
        "excerpt more than one sabotage shares belongs in tests/sabotage/anchors.py.")

test(f"every sabotage still applies to the tree as it is "
     f"({sum(map(len, PY_DRIVERS.values()))} sabotages in {len(PY_DRIVERS)} Python drivers)",
     check_every_sabotage_still_applies)


def check_the_shell_driver_still_applies():
    """engine_r1_perfkey_typo.sh edits data/gpus.json from a heredoc. What it needs
    is read from the script: each catalog row its loop names carries the perfKey it
    corrupts exactly once."""
    shell = [d for d in DRIVERS if d.endswith(".sh")]
    assert shell == SHELL_DRIVERS, (
        f"shell drivers {shell}, where this file checks {SHELL_DRIVERS}: the new one's sabotages "
        f"are invisible to the check above. Write it on run_driver() in Python, or check it here")
    src = open(os.path.join(SABOTAGE_DIR, SHELL_DRIVERS[0]), encoding="utf-8").read()
    loop = re.search(r"^for slug in ([\w .-]+); do$", src, re.M)
    needs = re.search(r"assert line\.count\('([^']+)'\) == 1", src)
    assert loop and needs, f"{SHELL_DRIVERS[0]} no longer has the loop and the precondition this reads"
    rows = open(os.path.join(ROOT, "data", "gpus.json"), encoding="utf-8").read().splitlines()
    for slug in loop.group(1).split():
        line = [r for r in rows if r.strip().startswith(f'"{slug}":')]
        assert len(line) == 1 and line[0].count(needs.group(1)) == 1, (
            f"{SHELL_DRIVERS[0]}: catalog row {slug} does not carry {needs.group(1)} exactly once")

test("the shell driver's catalog rows still carry what it corrupts", check_the_shell_driver_still_applies)


def check_a_worker_judges_only_at_the_commit_the_run_proved():
    """parallel.py proves a commit once per run, on the first worker; every other
    worker starts from that proof, so each must refuse unless it is at that commit and
    clean. Each case gets a throwaway repository of two commits: at the second and
    clean, a worker judges; at the first, or with a file changed or added, it refuses."""
    import subprocess, tempfile
    spec = importlib.util.spec_from_file_location("parallel", os.path.join(ROOT, "tests", "sabotage", "parallel.py"))
    parallel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parallel)

    def two_commits(tree):
        def git(*args):
            return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                                   *args], cwd=tree, check=True, capture_output=True, text=True).stdout.strip()
        git("init", "-q", "--template=")  # no hooks from a global template
        with open(os.path.join(tree, "a.txt"), "w") as f:
            f.write("one\n")
        git("add", "a.txt")
        git("commit", "-q", "-m", "one")
        first = git("rev-parse", "HEAD")
        git("commit", "-q", "--allow-empty", "-m", "two")
        return first, git("rev-parse", "HEAD")

    refused = {}
    with tempfile.TemporaryDirectory() as d:
        for case in ("proved", "another commit", "a changed file", "an added file"):
            tree = os.path.join(d, case.replace(" ", "-"))
            os.makedirs(tree)
            first, second = two_commits(tree)
            sha = first if case == "another commit" else second
            if case == "a changed file":
                with open(os.path.join(tree, "a.txt"), "a") as f:
                    f.write("changed\n")
            if case == "an added file":
                with open(os.path.join(tree, "b.txt"), "w") as f:
                    f.write("added\n")
            try:
                parallel.refuse_unless_proven(tree, sha)
            except RuntimeError as why:
                refused[case] = str(why)
    assert set(refused) == {"another commit", "a changed file", "an added file"}, refused

test("a corpus worker judges only at the commit the run proved, and clean", check_a_worker_judges_only_at_the_commit_the_run_proved)


FAKE_HARNESS = """import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JUDGING_SUITES = [("fake", ["true"])]


def require_green_baseline():
    with open(os.environ["FAKE_HARNESS_LOG"], "a") as f:
        f.write("baseline\\n")


def run_judging_suites():
    with open(os.path.join(ROOT, "tests", "golden", "g.json")) as f:
        emptied = f.read().strip() == "{}"
    return {"fake": (1, ["FAIL the golden"], [], "") if emptied else (0, [], [], "")}
"""


def check_a_worker_starts_from_the_run_s_proof_or_refuses():
    """A corpus worker's start, through the real `parallel.py --worker` process, on a
    throwaway repository whose harness only records what it is asked. The first worker
    proves the commit: its baseline runs, and the golden failures are found by emptying
    the goldens. Another worker trusts that proof: no baseline, and the golden failures
    it judges with are the file's. And a worker at another commit refuses. Round 23
    dropped the proof's check, the first worker's baseline, and the golden file's
    contents from a worker, and the unit test of the check alone saw none of them."""
    import json, subprocess, tempfile
    runner = os.path.join(ROOT, "tests", "sabotage", "parallel.py")
    with tempfile.TemporaryDirectory() as d:
        tree, log, handed = os.path.join(d, "tree"), os.path.join(d, "harness.log"), os.path.join(d, "golden.json")
        os.makedirs(os.path.join(tree, "tests", "sabotage"))
        os.makedirs(os.path.join(tree, "tests", "golden"))
        with open(os.path.join(tree, "tests", "sabotage", "harness.py"), "w") as f:
            f.write(FAKE_HARNESS)
        with open(os.path.join(tree, "tests", "golden", "g.json"), "w") as f:
            f.write('{"a": 1}\n')
        with open(handed, "w") as f:
            json.dump(["FAIL handed on by the first worker"], f)

        def git(*args):
            return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                                   *args], cwd=tree, check=True, capture_output=True, text=True).stdout.strip()
        git("init", "-q", "--template=")  # no hooks from a global template
        git("add", ".")
        git("commit", "-q", "-m", "one")
        first = git("rev-parse", "HEAD")
        git("commit", "-q", "--allow-empty", "-m", "two")
        second = git("rev-parse", "HEAD")

        def start(sha, golden_file=None):
            open(log, "w").close()
            done = subprocess.run([sys.executable, "-B", runner, "--worker", tree, "--sha", sha,
                                   *(["--golden", golden_file] if golden_file else [])],
                                  input="", capture_output=True, text=True, timeout=60,
                                  env=dict(os.environ, FAKE_HARNESS_LOG=log))
            lines = done.stdout.splitlines()
            assert lines, f"the worker said nothing: {done.stderr[-400:]}"
            with open(log) as f:
                return json.loads(lines[0]), f.read().split()

        ready, calls = start(second)
        assert ready.get("ready") and ready["golden"] == ["FAIL the golden"], ready
        assert calls == ["baseline"], f"the first worker proved nothing green: {calls}"
        ready, calls = start(second, handed)
        assert ready.get("ready") and ready["golden"] == ["FAIL handed on by the first worker"], ready
        assert calls == [], f"a worker given the proof proved the commit again: {calls}"
        ready, calls = start(first, handed)
        assert not ready.get("ready") and "not at" in ready.get("error", ""), ready

test("a corpus worker starts from the run's proof, or proves it first, or refuses at another commit",
     check_a_worker_starts_from_the_run_s_proof_or_refuses)


# The runner's verdicts. The second cold check of the speed-ups changed each of these
# in tests/sabotage/parallel.py and the suite stayed green: nothing tested how the
# runner classifies a catch, stops a run early, reads a driver, reports a driver's
# exit, keeps a worker clean, or keeps stale bytecode out (round 24's RUNNER R8-R15).
_runner = {}


def runner():
    if "module" not in _runner:
        spec = importlib.util.spec_from_file_location("parallel", os.path.join(ROOT, "tests", "sabotage", "parallel.py"))
        _runner["module"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_runner["module"])
    return _runner["module"]


GOLDEN_LINE = "FAIL the page golden"
REAL_LINE = "FAIL a real check"


def check_a_catch_is_golden_only_only_when_every_red_suite_is():
    """A catch is golden-only when regenerating the goldens would hide it: every suite
    that went red did so on golden failures alone. One real failure anywhere, a golden
    and a real failure in one suite, or a red suite with no FAIL line (a crash) is a
    catch that survives regenerating the goldens. The check made the runner call a
    catch golden-only when any red suite was (R8), and whenever it was caught at all
    (R9), and no test noticed either."""
    parallel = runner()
    golden = {GOLDEN_LINE}
    cases = {
        "a real failure": ({"model": (1, [REAL_LINE], [], "")}, False),
        "a golden failure alone": ({"model": (1, [GOLDEN_LINE], [], "")}, True),
        "a golden failure beside a green suite": ({"model": (1, [GOLDEN_LINE], [], ""), "sync": (0, [], [], "")}, True),
        "a golden failure beside a real one elsewhere": (
            {"model": (1, [GOLDEN_LINE], [], ""), "report": (1, [REAL_LINE], [], "")}, False),
        "a golden and a real failure in one suite": ({"model": (1, [GOLDEN_LINE, REAL_LINE], [], "")}, False),
        "a red suite that printed no FAIL line": ({"model": (1, [], ["Traceback"], "")}, False),
    }
    wrong = {}
    for case, (results, want) in cases.items():
        last = {}

        def run_driver(sabotages, results=results):
            last.update(results)
            print(f"  red    {next(iter(sabotages))}")
            print("")
            print("1 caught, 0 survived")

        got = parallel.judge_one(types.SimpleNamespace(run_driver=run_driver), "a_driver", case, [], last, golden)
        if got["golden_only"] is not want:
            wrong[case] = got["golden_only"]
    assert not wrong, f"golden-only called wrongly: {wrong}"

test("the runner calls a catch golden-only only when every red suite is", check_a_catch_is_golden_only_only_when_every_red_suite_is)


def check_early_exit_runs_on_past_a_suite_red_on_golden_failures_alone():
    """With --early-exit, a suite red on golden failures alone does not end the run: the
    next suite may catch the sabotage for real. The check made a red suite of any kind
    end it (R10), which reports such a sabotage golden-only when a later suite would
    have caught it, and no test noticed. A real catch does end it, and the suites are
    restored afterwards."""
    parallel = runner()
    per_suite = {"workflow": (1, [GOLDEN_LINE], [], ""), "price": (1, [REAL_LINE], [], ""), "parity": (0, [], [], "")}
    harness, ran = types.SimpleNamespace(JUDGING_SUITES=[(n, ["true"]) for n in ("parity", "price", "workflow")]), []

    def one_suite():
        (name, _), = harness.JUDGING_SUITES
        ran.append(name)
        return {name: per_suite[name]}

    harness.run_judging_suites = one_suite
    last = parallel.install_judging(harness, {GOLDEN_LINE}, early_exit=True)
    harness.run_judging_suites()
    assert ran[:2] == ["workflow", "price"] and "parity" not in ran, (
        f"early exit ran {ran}: it must run on past the golden-only suite and stop at the real catch")
    assert set(last) == {"workflow", "price"}, sorted(last)
    assert [s[0] for s in harness.JUDGING_SUITES] == ["parity", "price", "workflow"], harness.JUDGING_SUITES

test("early exit runs on past a suite red on golden failures alone, and stops at a real catch",
     check_early_exit_runs_on_past_a_suite_red_on_golden_failures_alone)


def check_a_worker_reads_a_driver_from_the_commit_under_test():
    """A worker judges the sabotages of the commit under test, read from that commit's
    tree, not the ones in the runner's own checkout: the two differ whenever a branch
    changes a driver. The check pointed the runner at its own checkout (R12) and no test
    noticed. The driver here has a name the runner's checkout also has, and different
    sabotages; the harness's run_driver is the real one, silenced while the driver
    loads, as the runner silences it."""
    import tempfile, harness
    parallel = runner()
    name = "engine_r1_throughput_leaks"
    assert os.path.exists(os.path.join(SABOTAGE_DIR, name + ".py")), f"{name} is gone; pick another driver both checkouts have"
    with tempfile.TemporaryDirectory() as tree:
        os.makedirs(os.path.join(tree, "tests", "sabotage"))
        with open(os.path.join(tree, "tests", "sabotage", name + ".py"), "w") as f:
            f.write('S = {"the commit under test\'s own": [("a.txt", "one", "two", 1)]}\n')
        got = parallel.sabotages_of(harness, tree, name)
    assert list(got) == ["the commit under test's own"], f"read another checkout's driver: {list(got)[:3]}"

test("a worker reads a driver from the commit under test, not the runner's checkout",
     check_a_worker_reads_a_driver_from_the_commit_under_test)


def check_a_driver_log_ends_with_how_its_run_ended():
    """A driver's log ends the way its own run under chain.sh ends: exit 1 when a
    sabotage errored, 2 when one could not be applied, 0 only when every one was
    judged. The check made every log say exit 0 (R15) and no test noticed."""
    parallel = runner()
    caught = {"status": "caught", "block": ["  red    a"]}
    cases = {
        "every sabotage judged": ({("d", "a"): caught}, 0),
        "one could not be applied": ({("d", "a"): caught, ("d", "b"): {"status": "unapplied", "block": ["  !! b: could not apply: x"]}}, 2),
        "one errored": ({("d", "a"): caught, ("d", "b"): {"status": "error", "error": "boom"}}, 1),
    }
    for case, (results, want) in cases.items():
        text, rc = parallel.driver_log("d", [n for _, n in results], results)[:2]
        assert rc == want and text.rstrip().endswith(f"driver exit {want}"), (case, rc, text.splitlines()[-1])

test("a driver's log ends with how its run ended: 1 errored, 2 unapplied, 0 judged",
     check_a_driver_log_ends_with_how_its_run_ended)


FAKE_JUDGING_HARNESS = """import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JUDGING_SUITES = [("fake", ["true"])]


def require_green_baseline():
    pass


def run_judging_suites():
    with open(os.environ["FAKE_HARNESS_LOG"], "a") as f:
        f.write(f"{os.environ.get('PYTHONDONTWRITEBYTECODE')} {os.path.getmtime(os.path.abspath(__file__))}\\n")
    return {"fake": (1, ["FAIL a real catch"], [], "")}


def run_driver(sabotages):
    name = next(iter(sabotages))
    with open(os.path.join(ROOT, "a.txt"), "a") as f:
        f.write("left behind\\n")
    run_judging_suites()
    print(f"  red    {name}")
    print("")
    print("1 caught, 0 survived")
"""


def check_a_worker_restores_a_tree_left_dirty_and_trusts_no_old_bytecode():
    """Through the real `parallel.py --worker` process, on a throwaway repository whose
    one sabotage leaves a tracked file changed. The worker must report that sabotage as
    an error, restore the file, and stay clean for the next: the check dropped the
    dirty check between sabotages (R11) and every later sabotage would have been judged
    on the changed tree. And no suite may run a stale .pyc: the worker turns bytecode
    off for its suites and gives every tracked .py a fresh timestamp, which the check
    removed (R14), putting back the 90 false catches seen at d99f1b6."""
    import json, subprocess, tempfile
    parallel_py = os.path.join(ROOT, "tests", "sabotage", "parallel.py")
    with tempfile.TemporaryDirectory() as d:
        tree, log, handed = os.path.join(d, "tree"), os.path.join(d, "harness.log"), os.path.join(d, "golden.json")
        fake = os.path.join(tree, "tests", "sabotage", "harness.py")
        os.makedirs(os.path.dirname(fake))
        with open(fake, "w") as f:
            f.write(FAKE_JUDGING_HARNESS)
        with open(os.path.join(tree, "tests", "sabotage", "engine_fake.py"), "w") as f:
            f.write('S = {"leaves the tree dirty": [("a.txt", "one", "two", 1)]}\n')
        with open(os.path.join(tree, "a.txt"), "w") as f:
            f.write("one\n")
        with open(handed, "w") as f:
            json.dump(["FAIL the golden"], f)

        def git(*args):
            return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                                   *args], cwd=tree, check=True, capture_output=True, text=True).stdout.strip()
        git("init", "-q", "--template=")  # no hooks from a global template
        git("add", ".")
        git("commit", "-q", "-m", "one")
        old = 1_000_000_000
        os.utime(fake, (old, old))
        env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
        env["FAKE_HARNESS_LOG"] = log
        task = json.dumps({"kind": "py", "driver": "engine_fake", "name": "leaves the tree dirty"})
        done = subprocess.run([sys.executable, "-B", parallel_py, "--worker", tree, "--sha", git("rev-parse", "HEAD"),
                               "--golden", handed], input=task + "\n", capture_output=True, text=True, timeout=60, env=env)
        lines = done.stdout.splitlines()
        assert len(lines) == 2, f"expected a ready line and one result: {done.stdout[-400:]} {done.stderr[-400:]}"
        ready, result = json.loads(lines[0]), json.loads(lines[1])
        assert ready.get("ready"), ready
        assert result.get("status") == "error" and "left the worker dirty" in result.get("error", ""), result
        assert not result.get("fatal"), f"the worker could not restore itself: {result}"
        assert git("status", "--porcelain") == "", "the worker was left dirty"
        with open(log) as f:
            bytecode, mtime = f.read().split()
    assert bytecode == "1", f"the suites ran with bytecode on (PYTHONDONTWRITEBYTECODE={bytecode})"
    assert float(mtime) > old + 1, "the commit's .py files kept the timestamps a stale .pyc could match"

test("a corpus worker restores a tree a sabotage left dirty, and trusts no old bytecode",
     check_a_worker_restores_a_tree_left_dirty_and_trusts_no_old_bytecode)


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
