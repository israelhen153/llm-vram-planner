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


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
