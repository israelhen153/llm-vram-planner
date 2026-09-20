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
no `engine` field, so "is this vLLM" is read out of prose, which is why the
classifier is pinned by its own fixtures below: a classifier that quietly starts
counting estimates would make every floor here meaningless while staying green.

Run:  python3 tests/coverage.test.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Floors, counted from the data at 1207b9a on 2026-09-20. Raise them when data
# lands; never lower one to make a change pass.
MIN_ENTRIES = 13
MIN_MEASURED_VLLM_BATCH = 5
MIN_CARDS_WITH_MEASURED = 3

# The dataset records its engine in prose (see its own _meta: "check each entry's
# `note` for the engine"), so this is a text check. Adding a real `engine` field
# would be better than lengthening this tuple.
NON_VLLM_ENGINES = ("llama.cpp", "ollama", "tensorrt", "sglang", "exllama", "mlc")

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


def is_measured_vllm_batch(entry):
    """A batch-throughput number actually measured on vLLM.

    Three ways an entry fails to be one, and each has bitten a public claim:
    it is an estimate, it is a single-stream number, or it was measured on a
    different engine. The launch post said "7 sourced" while only five were this.
    """
    if entry.get("estimated"):
        return False
    if entry.get("mode") != "batch":
        return False
    prose = f"{entry.get('src', '')} {entry.get('note', '')}".lower()
    return not any(e in prose for e in NON_VLLM_ENGINES)


def card_of(key):
    """Keys are '<model>-<card>'; the card is everything after the first dash."""
    return "-".join(key.split("-")[1:])


bench = load("benchmarks", "data.json")["data"]
catalog = load("data", "gpus.json")["data"]

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


def check_classifier_rejects_what_it_must():
    """The floors are only as honest as this predicate. Pinned with fixtures rather
    than by inspection, because a classifier that drifts to counting everything
    keeps every assertion above green while meaning nothing."""
    good = {"mode": "batch", "src": "Some vLLM benchmark", "note": "BF16 batch throughput"}
    assert is_measured_vllm_batch(good), "a plain measured vLLM batch entry must count"

    assert not is_measured_vllm_batch({**good, "estimated": True}), \
        "an estimated entry must not count as measured"
    assert not is_measured_vllm_batch({**good, "mode": "single"}), \
        "a single-stream entry must not count as a batch measurement"
    assert not is_measured_vllm_batch(
        {**good, "note": "Q4_K_XL, single-user decode, llama.cpp (not vLLM)"}), \
        "an entry measured on another engine must not count as vLLM"
    assert not is_measured_vllm_batch({"src": "x", "note": "y"}), \
        "an entry with no mode must not count"


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
test("the measured/estimated classifier rejects what it must", check_classifier_rejects_what_it_must)
test("the floors still describe data that exists", check_floors_are_not_above_reality)

print(f"\n{pass_ct} passed, {fail_ct} failed\n")
raise SystemExit(1 if fail_ct else 0)
