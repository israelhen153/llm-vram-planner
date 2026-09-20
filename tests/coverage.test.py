#!/usr/bin/env python3
"""Hold the benchmark dataset's coverage visible, and stop it shrinking.

The tool's whole claim is that its numbers are auditable. That claim rests on the
measured entries in benchmarks/data.json -- and on 2026-09-20 there were five of
them, covering three of twelve catalogued cards, with nothing added since
2026-07-31. "478 tests green" says nothing about that. This file makes the suite
say it out loud on every run, and fails if the number goes down.

It is a ratchet, not a deadline. A test keyed on elapsed time goes red on a
calendar boundary with no code change, on an unrelated pull request, and is
switched off within a fortnight. This one can only be tripped by a real
regression: data removed, an entry downgraded, or a card losing its last
measurement. Raise the floors when data is added.

Two things it deliberately does NOT claim. It does not check that any number is
correct -- only that it is present and classified honestly. And the dataset has
no `engine` field, so "is this vLLM" is read out of prose.

That prose check asks for POSITIVE evidence, and the first version did not. It
carried a denylist of engine names instead, which was wrong in both directions
and a cold check broke it both ways: TGI, TRT-LLM, LMDeploy and llama-cpp-python
are not in any such list, so an entry re-attributed to one still counted; while
a perfectly good entry whose note said "faster than TensorRT-LLM in the same
post" was thrown out, and "MLCommons" matched the substring "mlc". Comparison
posts are the normal source of these numbers, so both were realistic.

Requiring "vllm" somewhere in src, note or url keeps all five entries that count
today -- three of them prove it only in the url, which the denylist version never
read. It is still prose sniffing. **The real fix is an explicit `engine` field
in benchmarks/data.json**, and until that exists this file is the argument for it.

Run:  python3 tests/coverage.test.py
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Floors, counted from the data at 1207b9a on 2026-09-20. Raise them when data
# lands; never lower one to make a change pass.
MIN_ENTRIES = 13
MIN_MEASURED_VLLM_BATCH = 5
MIN_CARDS_WITH_MEASURED = 3

# The dataset records its engine in prose (see its own _meta: "check each entry's
# `note` for the engine"), so this is a text check -- for positive evidence, never
# against a list of other engines' names. See the note in the docstring.
# "vllm" appearing in prose is not evidence when the prose is denying it: the note
# "TGI 2.0 (not vLLM)" contains the substring. Negations are removed before looking.
VLLM_DENIED = re.compile(r"\b(?:not|non-?|isn't|rather than|instead of)\s*(?:a\s+)?vllm\b", re.I)

# Vocabulary every estimate in this dataset already uses. An entry that talks like
# an estimate while carrying no `estimated` flag is either mislabelled data or a
# misspelled key, and both would be counted as a measurement.
#
# Anchored, because unanchored substrings are what went wrong twice here already:
# "mlc" matched "MLCommons" in the denylist this replaced, and then "rough" matched
# "throughput" in the first draft of this very tuple, condemning four real entries.
ESTIMATE_PATTERNS = re.compile(
    r"\b(?:extrapolat\w*|interpolat\w*|unverified|estimat\w*|guess\w*)\b"
    r"|not measured|no published source", re.I)

# Anything else is a typo. `"estimted": true` counted as measured until this existed.
KNOWN_FIELDS = {"tokS", "mode", "src", "note", "prec", "date", "url", "estimated"}

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


def load(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return json.load(fh)


def prose_of(entry):
    """Everything an entry says about itself, including the url -- three of the five
    entries that count today name vLLM nowhere else."""
    return f"{entry.get('src', '')} {entry.get('note', '')} {entry.get('url', '')}".lower()


def is_measured_vllm_batch(entry):
    """A batch-throughput number actually measured on vLLM.

    Four ways an entry fails to be one, and each has bitten a public claim: it is
    an estimate, it is a single-stream number, it carries no usable number, or
    nothing about it says vLLM. The launch post said "7 sourced" while only five
    were this.
    """
    if entry.get("estimated"):
        return False
    if entry.get("mode") != "batch":
        return False
    tok = entry.get("tokS")
    # bool is an int in Python, and `"12500"` is not a measurement.
    if isinstance(tok, bool) or not isinstance(tok, (int, float)) or tok <= 0:
        return False
    return "vllm" in VLLM_DENIED.sub(" ", prose_of(entry)).lower()


def card_of(key):
    """Keys are '<model>-<card>'; the card is everything after the first dash."""
    return "-".join(key.split("-")[1:])


try:
    bench = load("benchmarks", "data.json")["data"]
    catalog = load("data", "gpus.json")["data"]
    assert isinstance(bench, dict) and isinstance(catalog, dict), "both `data` blocks must be objects"
    bad = [k for k, v in bench.items() if not isinstance(v, dict)]
    assert not bad, f"benchmark entries must be objects: {bad}"
except Exception as exc:
    # Outside a test() this is a traceback with no summary and no named failure,
    # which is how a malformed dataset reads as an infrastructure problem instead
    # of a data problem.
    print("\nBenchmark coverage — could not be read")
    print(f"  FAIL the benchmark data loads at all\n       {type(exc).__name__}: {exc}")
    print("\n0 passed, 1 failed\n")
    raise SystemExit(1)

measured = {k: v for k, v in bench.items() if is_measured_vllm_batch(v)}
cards_measured = {card_of(k) for k in measured}
cards_any = {card_of(k) for k in bench}

print("\nBenchmark coverage — what the auditable claim actually rests on")
print(f"       {len(bench)} entries | {len(measured)} measured vLLM batch | "
      f"{len(cards_measured)} of {len(catalog)} catalogued cards have one")
print(f"       cards with a measured number: {', '.join(sorted(cards_measured)) or 'none'}")
print(f"       cards with no entry at all:   "
      f"{', '.join(sorted(set(catalog) - cards_any)) or 'none'}")


def check_entries():
    assert len(bench) >= MIN_ENTRIES, (
        f"benchmarks/data.json has {len(bench)} entries, floor is {MIN_ENTRIES}. "
        f"Entries were removed; restore them or justify the removal by lowering the floor.")


def check_measured():
    assert len(measured) >= MIN_MEASURED_VLLM_BATCH, (
        f"{len(measured)} measured vLLM batch entries, floor is {MIN_MEASURED_VLLM_BATCH}. "
        f"An entry was removed, downgraded to estimated, or re-attributed to another engine.")


def check_cards():
    assert len(cards_measured) >= MIN_CARDS_WITH_MEASURED, (
        f"{len(cards_measured)} cards have a measured number, floor is "
        f"{MIN_CARDS_WITH_MEASURED}. A card lost its last measurement.")


def check_keys_name_real_cards():
    """A key naming a card the catalog does not have is a benchmark nothing can use,
    and it would inflate every count above."""
    unknown = sorted({card_of(k) for k in bench} - set(catalog))
    assert not unknown, f"benchmark keys name cards not in data/gpus.json: {unknown}"


def check_keys_are_reachable():
    """The card half is only half the key. A model half that is empty, uppercase or
    otherwise unreachable is dead data the tool can never match -- and it still
    inflates the counts printed above, which is what this file exists to report."""
    bad = []
    for k in bench:
        model = k.split("-")[0]
        if not model or model != model.lower() or not card_of(k):
            bad.append(k)
    assert not bad, f"benchmark keys the tool cannot match: {sorted(bad)}"


def check_no_unknown_fields():
    """`"estimted": true` counted as a measurement until this existed: the flag was
    never read, because the key was not the one anything looks for."""
    offenders = {k: sorted(set(v) - KNOWN_FIELDS) for k, v in bench.items()
                 if set(v) - KNOWN_FIELDS}
    assert not offenders, (
        f"benchmark entries carry fields nothing reads, which is how a misspelled "
        f"`estimated` turns an estimate into a measurement: {offenders}")


def check_estimates_are_flagged():
    """Every estimate in this dataset says so in prose. One that says so and carries
    no flag is counted as a measurement, and no floor above would notice."""
    unflagged = sorted(k for k, v in bench.items()
                       if not v.get("estimated")
                       and ESTIMATE_PATTERNS.search(prose_of(v)))
    assert not unflagged, (
        f"these read as estimates but carry no `estimated: true`, so they count as "
        f"measurements: {unflagged}")


def check_classifier_rejects_what_it_must():
    """The floors are only as honest as this predicate. Pinned with fixtures rather
    than by inspection, because a classifier that drifts to counting everything
    keeps every assertion above green while meaning nothing."""
    good = {"mode": "batch", "tokS": 1234, "src": "Some vLLM benchmark",
            "note": "BF16 batch throughput", "url": ""}
    assert is_measured_vllm_batch(good), "a plain measured vLLM batch entry must count"

    assert not is_measured_vllm_batch({**good, "estimated": True}), \
        "an estimated entry must not count as measured"
    assert not is_measured_vllm_batch({**good, "mode": "single"}), \
        "a single-stream entry must not count as a batch measurement"
    assert not is_measured_vllm_batch({"mode": "batch", "tokS": 1}), \
        "an entry naming no engine must not be assumed to be vLLM"
    assert not is_measured_vllm_batch({**good, "tokS": "1234"}), \
        "a number that is not a number must not count"
    assert not is_measured_vllm_batch({**good, "tokS": 0}), \
        "a zero measurement must not count"
    assert not is_measured_vllm_batch({k: v for k, v in good.items() if k != "tokS"}), \
        "an entry with no tokS must not count"

    # Positive evidence, wherever it lives -- three of the five that count today
    # name vLLM only in the url.
    url_only = {"mode": "batch", "tokS": 1, "src": "DatabaseMart A100 benchmark",
                "note": "BF16 batch throughput",
                "url": "https://www.databasemart.com/blog/vllm-gpu-benchmark-a100-80gb"}
    assert is_measured_vllm_batch(url_only), "vLLM evidence in the url must count"

    # And the two directions a denylist got wrong, which is why there is not one.
    assert is_measured_vllm_batch(
        {**good, "note": "vLLM 0.9 batch throughput, faster than TensorRT-LLM in the same post"}), \
        "naming another engine in a comparison must not disqualify a vLLM measurement"
    assert not is_measured_vllm_batch(
        {**good, "src": "HuggingFace TGI benchmarks", "note": "TGI 2.0 (not vLLM)",
         "url": "https://example.com/tgi"}), \
        "an engine nobody thought to denylist must still fail for lack of evidence"


def check_floors_are_not_above_reality():
    """A floor raised past the data is a test that can never pass, which gets deleted
    rather than fixed. Keep them honest in both directions."""
    assert MIN_ENTRIES <= len(bench) and MIN_MEASURED_VLLM_BATCH <= len(measured) \
        and MIN_CARDS_WITH_MEASURED <= len(cards_measured), \
        "a floor is above the current data — raise data, not the floor"


test("the dataset has not lost entries", check_entries)
test("the measured vLLM batch numbers have not been reduced", check_measured)
test("no card has lost its last measured number", check_cards)
test("every benchmark key names a card the catalog has", check_keys_name_real_cards)
test("every benchmark key is one the tool can match", check_keys_are_reachable)
test("no entry carries a field nothing reads", check_no_unknown_fields)
test("every entry that reads as an estimate is flagged as one", check_estimates_are_flagged)
test("the measured/estimated classifier rejects what it must", check_classifier_rejects_what_it_must)
test("the floors still describe data that exists", check_floors_are_not_above_reality)

print(f"\n{pass_ct} passed, {fail_ct} failed\n")
raise SystemExit(1 if fail_ct else 0)
