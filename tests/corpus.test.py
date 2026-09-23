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

Deliberately NOT one of the suites that judge a sabotage (harness.JUDGING_SUITES):
every sabotage edits the engine, so this would go red under all of them and
report each one as caught whatever the real suites said — the same reason
tests/assets.test.py is excluded.

Run:  python3 tests/corpus.test.py
"""
import os
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


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
