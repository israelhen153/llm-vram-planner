#!/usr/bin/env python3
"""Round 26: the cold check of chore/derive-catalog-anchors (97a2189), which has to
guarantee four things about the corpus and the weekly price job:

1. No sabotage quotes a value the job rewrites (a tier's price, a reading's date, price
   or offer count, the note a first reading takes out), and tests/corpus.test.py goes red
   in the pull request that adds one.
2. A sabotage whose target is taken out of the catalog keeps its name and is refused by
   name (harness.Missing): never dropped, never a driver that fails to load, and
   tests/corpus.test.py goes red when a driver breaks this.
3. The pull request that confirms the held tier, h100-80/spot, fails nothing but the
   pinned HELD_UNDER_A_NOTE in tests/price_check.test.py, which it edits on purpose:
   tests/model.test.js and tests/corpus.test.py stay green on it.
4. A reading invented on a held tier is caught by a real test, not only by the goldens.

Two dicts, two judges. S holds sabotages of the catalog and of two test files, judged by
the six suites as every driver is; `--full` judges them with tests/corpus.test.py as well
and prints every FAIL line, so a catch only a golden makes reads as "gold", not "red".
G3a-G3c are the confirming pull request itself, built with the job's own writers
(price_check.apply_outcomes and apply_to_text, then tools/sync_data.py), and G3c releases
the pinned tier as that pull request does. G4b-G4f invent readings the ways the pinned set
and the catalog rules must catch. CORPUS holds sabotages of the corpus: a sabotage added to
engine_r5_provenance_fields.py, an excerpt added to anchors.py, an assert added to the shell
driver, or a rule in tests/corpus.test.py itself, judged with `--corpus` by
tests/corpus.test.py alone, as engine_r23's RUNNER is. Q1-Q21 attack guarantees 1 and 2,
Q19-Q21 the rules corpus.test.py holds itself to.

At 97a2189, judged this way: G3a and G3b fail the pinned set and the two goldens and
nothing else, G3c only the two goldens, and corpus.test.py is green on all three, which is
guarantee 3 holding; G3d turns a real model test red, so the derived mixed row is what
keeps it green; G4b-G4e are each caught by a real price test. Ten survived or were caught
by the goldens alone: G3e (a test that will break the confirming pull request can be added
today and nothing notices until then), G4f (the pinned set made a tautology, and the
invented reading is then the goldens' alone to catch), Q6 (a quoted date with its count
taken from the catalog applies vacuously once the count is 0), Q7 (a sabotage with no edits
applies), Q12 (a renamed sabotage is not a dropped one: only counts are compared), Q13 (a
sabotage that retargets is not refused), Q14 (the shell driver's reader sees line.count()
and nothing else), Q15 (an offer count re-read as one more keeps its leading digit), Q16
(the simulated confirming tree edits the catalog and the engines, not the pinned line the
real one edits, so a driver quoting that line is red on the real pull request and green
here; Q16b shows it) and Q18 (a Missing naming any tier with a reading is excused, so a
sabotage can be parked forever and only the listing says so). Q17 is a probe, not a
sabotage: a correct driver that reads the catalog through pathlib fails the moved refresh,
because reading_from() rebinds builtins.open alone.

Nothing here quotes a value the job rewrites: every catalog edit is a row read as this
driver loads and rewritten by the job's own row writer, the confirming pull request is the
writers' own output diffed line by line, the pinned line is found by pattern, and each is
refused by name (with its note, where the price job is what takes the note out) once its
target is gone. The CORPUS snippets do quote such values on purpose, as the literals a
driver author would type today, and each literal is taken from the catalog as this driver
loads so the probe stays the same probe after any refresh.

Usage: python3 tests/sabotage/engine_r26_derive_anchors_cold_check.py [name-substring ...]
       python3 tests/sabotage/engine_r26_derive_anchors_cold_check.py --full [name-substring ...]
       python3 tests/sabotage/engine_r26_derive_anchors_cold_check.py --corpus [name-substring ...]
"""
import ast, copy, difflib, json, os, re, subprocess, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (run_driver, apply_edits, restore_files, edits_of, resyncs, Missing, ROOT,
                     JUDGING_SUITES)
from anchors import GPUS_JSON
sys.path.insert(0, os.path.join(ROOT, "tools"))
import price_check as pc

PRICE_TEST = "tests/price_check.test.py"
MODEL_TEST = "tests/model.test.js"
CORPUS_TEST = "tests/corpus.test.py"
R5 = "tests/sabotage/engine_r5_provenance_fields.py"
ANCHORS = "tests/sabotage/anchors.py"
SHELL = "tests/sabotage/engine_r1_perfkey_typo.sh"

# The catalog as text and as data, read as this driver loads, so nothing below quotes it.
TEXT = open(os.path.join(ROOT, GPUS_JSON), encoding="utf-8").read()
ROWS = json.loads(TEXT)["data"]
LINE = {slug: next((l for l in TEXT.splitlines() if l.lstrip().startswith(f'"{slug}":')), None) for slug in ROWS}
H100 = ROWS.get("h100-80") or {}
NOTE = (H100.get("priceNote") or {}).get("spot")
HYPER = (H100.get("priceSource") or {}).get("hyper")
# Refused by name, naming the note, once the price job's first reading has taken it out.
HELD_NOTE_GONE = (GPUS_JSON, Missing("h100-80/spot carries no note (priceNote.spot): the tier is no longer held, "
                                     "so there is no first reading to confirm on it", note=("h100-80", "spot")), "", 1)


def row_rewrite(slug, mutate):
    """One catalog row, as the file holds it, replaced by the same row rewritten by the price
    job's own row writer after mutate(row). The row is read, never quoted, and a row the
    writer would render differently from the file is refused by name, as a drifted anchor is."""
    line = LINE.get(slug)
    if line is None:
        return (GPUS_JSON, Missing(f"no row {slug!r} in the catalog"), "", 1)
    row = copy.deepcopy(ROWS[slug])
    old = line.rstrip().rstrip(",")
    if pc._dump_row_compact(slug, row) != old:
        return (GPUS_JSON, Missing(f"{slug}'s row is not written in the price job's own format"), "", 1)
    mutate(row)
    return (GPUS_JSON, old, pc._dump_row_compact(slug, row), 1)


def held_note_rewrite(mutate):
    """A rewrite of h100-80's row that needs its spot tier still held under a note."""
    if not NOTE or not isinstance(H100.get("spot"), (int, float)):
        return HELD_NOTE_GONE
    return row_rewrite("h100-80", mutate)


def invented(price, sku="H100 SXM (median of 9 verified offers)"):
    return {"provider": "vast", "sku": sku, "region": "global", "date": "2026-09-28", "price": price}


def note_replaced(tier, reading):
    """mutate(row): the tier's note taken out and reading put where the job would put one."""
    def mutate(row):
        del row["priceNote"][tier]
        if not row["priceNote"]:
            del row["priceNote"]
        row.setdefault("priceSource", {})[tier] = reading
    return mutate


def confirming_pr(status, factor, day="2026-09-29", sku="H100 SXM (median of 41 verified offers)"):
    """The lines the price job's own writers change when h100-80/spot's first reading lands:
    apply_outcomes() on the catalog, apply_to_text() on the file, one edit per changed line.
    MOVED at factor times the tier's price, or CONFIRMED at the price itself."""
    if not NOTE or not isinstance(H100.get("spot"), (int, float)):
        return [HELD_NOTE_GONE]
    data = json.loads(TEXT)["data"]
    spot = data["h100-80"]["spot"]
    price = round(spot * factor, 2)
    reading = pc.Reading(provider="vast", sku=sku, region="global", price_per_gpu=price, date=day, evidence="")
    outcome = pc.Outcome("h100-80", "spot", status, current=spot, proposed=price if status == "MOVED" else spot,
                         reading=reading)
    pc.apply_outcomes(data, [outcome])
    try:
        new = pc.apply_to_text(TEXT, data, ["h100-80"], day)
    except SystemExit as e:
        return [(GPUS_JSON, Missing(f"the price job's writer refuses the catalog: {e}"), "", 1)]
    before, after = TEXT.splitlines(keepends=True), new.splitlines(keepends=True)
    edits = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag != "replace" or i2 - i1 != j2 - j1:
            return [(GPUS_JSON, Missing(f"the writers changed the file's shape ({tag}), not lines in place"), "", 1)]
        for old, changed in zip(before[i1:i2], after[j1:j2]):
            if TEXT.count(old) != 1:
                return [(GPUS_JSON, Missing(f"a line the writers change occurs {TEXT.count(old)} times: {old[:60]!r}"), "", 1)]
            edits.append((GPUS_JSON, old, changed, 1))
    return edits


# The line pinning the held tiers, found by pattern: it is the line the confirming pull
# request edits, so a sabotage quoting it stops applying on exactly that pull request.
PRICE_TEST_SRC = open(os.path.join(ROOT, PRICE_TEST), encoding="utf-8").read()
PINNED = re.search(r"^HELD_UNDER_A_NOTE = (\{.*\})\n", PRICE_TEST_SRC, re.M)


def pinned_held():
    return ast.literal_eval(PINNED.group(1)) if PINNED else None


def pinned_edit(new_line):
    held = pinned_held()
    if held is None:
        return (PRICE_TEST, Missing("tests/price_check.test.py no longer pins HELD_UNDER_A_NOTE on one line"), "", 1)
    if "spot" not in held.get("h100-80", []):
        return (PRICE_TEST, Missing("h100-80/spot is no longer pinned as held under a note", note=("h100-80", "spot")), "", 1)
    return (PRICE_TEST, PINNED.group(0), new_line, 1)


def released():
    """The pinned set with h100-80/spot taken out, as the confirming pull request writes it."""
    rest = {s: [t for t in ts if (s, t) != ("h100-80", "spot")] for s, ts in (pinned_held() or {}).items()}
    return "HELD_UNDER_A_NOTE = " + json.dumps({s: ts for s, ts in rest.items() if ts}) + "\n"


# The pinned set derived from the catalog: held and pinned are then the same expression.
DERIVED = ('HELD_UNDER_A_NOTE = {s: [t for t, cfg in tiers.items() if "primary" in cfg\n'
           '                         and t in ((rows.get(s) or {}).get("priceNote") or {})]\n'
           '                     for rows in [json.load(open(os.path.join(ROOT, "data", "gpus.json")))["data"]]\n'
           '                     for s, tiers in pc.SOURCE_MAP.items()}\n')

MODEL_MIXED = "  const MIXED = realRows(s => s.includes('sourced') && s.includes('not recorded'));\n"
MODEL_NAMED = "  const MIXED = [['h100-80', GPU_TABLE['h100-80']]];\n"


def a100_spot_invented():
    """a100-80/spot: held under a note, and a tier the job never reads (SOURCE_MAP manual)."""
    row = ROWS.get("a100-80") or {}
    if "primary" in (pc.SOURCE_MAP.get("a100-80", {}).get("spot") or {}):
        return (GPUS_JSON, Missing("a100-80/spot is read by the price job now, so a reading there is not invented"), "", 1)
    if "spot" not in (row.get("priceNote") or {}) or not isinstance(row.get("spot"), (int, float)):
        return (GPUS_JSON, Missing("a100-80/spot carries no note (priceNote.spot) to put an invented reading in place of"), "", 1)
    return row_rewrite("a100-80", note_replaced("spot", invented(row["spot"], "A100 SXM (median of 12 verified offers)")))


def hyper_back_under_a_note():
    if not HYPER:
        return (GPUS_JSON, Missing("h100-80 carries no hyperscaler reading (priceSource.hyper) to take out"), "", 1)

    def mutate(row):
        del row["priceSource"]["hyper"]
        if not row["priceSource"]:
            del row["priceSource"]
        row.setdefault("priceNote", {})["hyper"] = {"reason": "Held for a re-read.", "checked": "2026-09-23"}
    return row_rewrite("h100-80", mutate)


S = {
    # ---- guarantee 3: the confirming pull request, built with the job's own writers ----
    "G3a data: h100-80/spot's first reading lands, MOVED within the band, as the price job's pull request leaves the catalog":
        confirming_pr("MOVED", 1.29),
    "G3b data: h100-80/spot's first reading lands at the tier's own price, CONFIRMED: the note goes, the price stays":
        confirming_pr("CONFIRMED", 1.0),
    "G3c data+price: G3a with h100-80/spot released from HELD_UNDER_A_NOTE, the confirming pull request as it should be":
        confirming_pr("MOVED", 1.29) + [pinned_edit(released())],
    "G3d data+price+model: G3c with model.test.js naming h100-80 as its mixed row again":
        confirming_pr("MOVED", 1.29) + [pinned_edit(released()), (MODEL_TEST, MODEL_MIXED, MODEL_NAMED, 1)],
    "G3e model: model.test.js names h100-80 as its mixed row again, on the tree as it is":
        [(MODEL_TEST, MODEL_MIXED, MODEL_NAMED, 1)],
    # ---- guarantee 4: readings invented on held tiers ----
    "G4b data: a reading invented on h100-80/spot at the tier's own price, its note kept beside it":
        [held_note_rewrite(lambda row: row.setdefault("priceSource", {}).__setitem__("spot", invented(H100.get("spot"))))],
    "G4c data: a reading invented on a100-80/spot, a held tier the job never reads, in place of its note":
        [a100_spot_invented()],
    "G4d data: a reading invented on h100-80/spot in place of its note, at a price 5% off the tier's own":
        [held_note_rewrite(note_replaced("spot", invented(round((H100.get("spot") or 0) * 1.05, 2))))],
    "G4e data: h100-80/hyper's reading taken out and a note put in its place, a read tier held again":
        [hyper_back_under_a_note()],
    "G4f price+data: HELD_UNDER_A_NOTE derived from the catalog instead of pinned, and a reading invented on h100-80/spot at its own price":
        [pinned_edit(DERIVED), held_note_rewrite(note_replaced("spot", invented(H100.get("spot"))))],
}


# ---- CORPUS: sabotages of the corpus, judged by tests/corpus.test.py ----
R5_TAIL = 'if __name__ == "__main__":\n    run_driver(S)\n'
SHELL_LINE = "line = [l for l in s.splitlines() if l.strip().startswith(f'\"{slug}\":')][0]\n"
ANCHORS_TAIL = 'GPUS_JSON = "data/gpus.json"\n'
CORPUS_EXCUSED = '    stale, listed, excused = [], [], replaced_by_a_reading(base["data/gpus.json"])\n'
CORPUS_TAKEN = '    return files, (lambda missing: True) if kind == "gone" else replaced_by_a_reading(catalog)\n'
CORPUS_MOVED = '            price = row[tier] if kind == "reread" else moved_from(row[tier])\n'


def literal(v):
    """A Python literal of v, for a snippet inserted into a driver."""
    return repr(v)


def most_common_read_date():
    dates = [src["date"] for row in ROWS.values() for src in (row.get("priceSource") or {}).values()]
    return max(set(dates), key=dates.count) if dates else None


def offer_count_prefix():
    """rtx4090-24's spot SKU text through the first digit of its offer count."""
    src = (ROWS.get("rtx4090-24", {}).get("priceSource") or {}).get("spot")
    m = re.search(r"\(median of (\d+)", src["sku"]) if src else None
    return ('"sku": ' + json.dumps(src["sku"])[:m.start(1) + 2]) if m else None


def r5_added(snippet):
    """A sabotage added to engine_r5_provenance_fields.py, before its run_driver() tail."""
    return (R5, R5_TAIL, snippet.rstrip("\n") + "\n\n\n" + R5_TAIL, 1)


def r5_or_missing(needed, why, snippet):
    return [r5_added(snippet) if needed else (R5, Missing(why), "", 1)]


READ_DATE = most_common_read_date()
PREFIX = offer_count_prefix()
L40S_HYPER = ROWS.get("l40s-48", {}).get("hyper")

CORPUS = {
    # ---- guarantee 1: quotes of values the job rewrites (Q1, Q3, Q5 are controls) ----
    "Q1 r5: h100-80's hyperscaler price quoted, the value the job moves":
        r5_or_missing(isinstance(H100.get("hyper"), (int, float)), "h100-80 has no hyperscaler price to quote",
            'S["Q1 data: h100-80/hyper moved with its reading left where it was"] = [\n'
            '    (GPUS_JSON, \'"h100-80": { "gb": 80, "bw": 3352, "hyper": \' + ' + literal(json.dumps(H100.get("hyper"))) + ' + ",",\n'
            '     \'"h100-80": { "gb": 80, "bw": 3352, "hyper": \' + ' + literal(json.dumps(round((H100.get("hyper") or 0) + 0.2, 2))) + ' + ",", 1)]\n'),
    "Q3 r5: h100-80/spot's held note quoted, the note the first reading takes out":
        r5_or_missing(bool(NOTE), "h100-80/spot has no note to quote",
            'S["Q3 data: h100-80/spot\'s note reworded"] = [\n'
            '    (GPUS_JSON, ' + literal(json.dumps((NOTE or {}).get("reason"))) + ',\n'
            '     ' + literal(json.dumps(((NOTE or {}).get("reason") or ".")[:-1] + ", still.")) + ', 1)]\n'),
    "Q5 anchors: an excerpt of h100-80's row through its hyperscaler price":
        [(ANCHORS, ANCHORS_TAIL, ANCHORS_TAIL + "GPUS_H100_ROW = "
          + literal('"h100-80": { "gb": 80, "bw": 3352, "hyper": ' + json.dumps(H100.get("hyper"))) + "\n", 1)],
    "Q6 r5: a reading's date quoted as a literal, its count taken from the catalog as the driver loads":
        r5_or_missing(READ_DATE is not None, "no reading carries a date to quote",
            'S["Q6 data: every reading of ' + str(READ_DATE) + ' dated 2026-01-01"] = [\n'
            '    (GPUS_JSON, \'"date": \' + ' + literal(json.dumps(READ_DATE)) + ', \'"date": "2026-01-01"\',\n'
            '     open(os.path.join(ROOT, GPUS_JSON), encoding="utf-8").read().count(\'"date": \' + ' + literal(json.dumps(READ_DATE)) + '))]\n'),
    "Q14 shell: the shell driver asserts l40s-48's hyperscaler price with `in`, not line.count()":
        [(SHELL, SHELL_LINE, SHELL_LINE + 'if slug == "l40s-48":\n    assert \'"hyper": '
          + json.dumps(L40S_HYPER) + '\' in line, "l40s-48: the hyperscaler price moved"\n', 1)]
        if isinstance(L40S_HYPER, (int, float)) else [(SHELL, Missing("l40s-48 has no hyperscaler price"), "", 1)],
    "Q15 r5: rtx4090-24's spot SKU quoted through the first digit of its offer count, refused by name once the reading is gone":
        r5_or_missing(PREFIX is not None, "rtx4090-24/spot carries no reading counting its offers",
            '_spot = ((json.load(open(os.path.join(ROOT, GPUS_JSON), encoding="utf-8"))["data"].get("rtx4090-24") or {})\n'
            '         .get("priceSource") or {}).get("spot")\n'
            'S["Q15 data: rtx4090-24/spot\'s offer count given a digit"] = [\n'
            '    (GPUS_JSON, ' + literal(PREFIX) + ', ' + literal(PREFIX) + ' + "0", 1) if _spot\n'
            '    else (GPUS_JSON, Missing("rtx4090-24/spot carries no reading (priceSource.spot) to edit"), "", 1)]\n'),
    "Q16 r5: a sabotage quoting the pinned HELD_UNDER_A_NOTE line of tests/price_check.test.py":
        r5_or_missing(PINNED is not None, "tests/price_check.test.py no longer pins HELD_UNDER_A_NOTE on one line",
            'S["Q16 price: the pinned held set emptied"] = [\n'
            '    ("tests/price_check.test.py", ' + literal(PINNED.group(0) if PINNED else "") + ', "HELD_UNDER_A_NOTE = {}\\n", 1)]\n'),
    "Q16b r5+data+price: Q16's quote, on the tree the confirming pull request leaves (the consequence)":
        r5_or_missing(PINNED is not None, "tests/price_check.test.py no longer pins HELD_UNDER_A_NOTE on one line",
            'S["Q16 price: the pinned held set emptied"] = [\n'
            '    ("tests/price_check.test.py", ' + literal(PINNED.group(0) if PINNED else "") + ', "HELD_UNDER_A_NOTE = {}\\n", 1)]\n')
        + confirming_pr("MOVED", 1.29) + [pinned_edit(released())],
    "Q21 corpus: the moved refresh keeps every price where it is":
        [(CORPUS_TEST, CORPUS_MOVED, "            price = row[tier]\n", 1)],
    # ---- guarantee 2: a target taken out of the catalog (Q8, Q11 are controls) ----
    "Q7 r5: h100-80/hyper's provider swapped, or no edit at all once the reading is gone":
        [r5_added('S["Q7 data: h100-80/hyper\'s provider swapped"] = (\n'
                  '    [reading_edit(\'"provider": "azure", "sku": "Standard_ND96isr_H100_v5"\', \'"provider": "aws", "sku": "Standard_ND96isr_H100_v5"\')]\n'
                  '    if READING else [])\n')],
    "Q8 r5: h100-80/hyper's region swapped, dropped from the corpus once the reading is gone":
        [r5_added('if READING:\n'
                  '    S["Q8 data: h100-80/hyper\'s region swapped"] = [\n'
                  '        reading_edit(\'"sku": "Standard_ND96isr_H100_v5", "region": "eastus"\', \'"sku": "Standard_ND96isr_H100_v5", "region": "westus2"\')]\n')],
    "Q11 r5: a reading invented on h100-80/spot, refused without naming its note once the note is gone":
        [r5_added('S["Q11 data: a reading invented on h100-80/spot"] = [\n'
                  '    invented_spot_reading() if (H100.get("priceNote") or {}).get("spot")\n'
                  '    else (GPUS_JSON, Missing("h100-80/spot has no note to put a reading in place of"), "", 1)]\n')],
    "Q12 r5: a sabotage named after h100-80's hyperscaler price":
        [r5_added('S["Q12 data: h100-80/hyper\'s reading at " + json.dumps(H100.get("hyper")) + " given a blank region"] = [\n'
                  '    reading_edit(\'"sku": "Standard_ND96isr_H100_v5", "region": "eastus"\', \'"sku": "Standard_ND96isr_H100_v5", "region": ""\')]\n')],
    "Q13 r5: a sabotage that attacks mi300x-192's hand record instead once h100-80's reading is gone":
        [r5_added('S["Q13 data: a provenance region blanked"] = [\n'
                  '    reading_edit(\'"sku": "Standard_ND96isr_H100_v5", "region": "eastus"\', \'"sku": "Standard_ND96isr_H100_v5", "region": ""\')\n'
                  '    if READING else\n'
                  '    (GPUS_JSON, \'"sku": "MI300X (Secure Cloud)", "region": "global"\', \'"sku": "MI300X (Secure Cloud)", "region": ""\', 1)]\n')],
    "Q18 r5: a sabotage claiming h100-80/hyper's note was replaced by its reading, with nothing of it missing":
        [r5_added('S["Q18 data: built on h100-80/hyper\'s note"] = [\n'
                  '    (GPUS_JSON, Missing("h100-80/hyper\'s note is gone", note=("h100-80", "hyper")), "", 1)]\n')],
    # ---- guarantee 3: the rules tests/corpus.test.py holds itself to on the confirming tree ----
    "Q19 corpus: the check of the tree as it is excuses no refusal by name":
        [(CORPUS_TEST, CORPUS_EXCUSED, "    stale, listed, excused = [], [], (lambda missing: False)\n", 1)],
    "Q20 corpus: a refresh excuses only the notes it replaced itself, not one an earlier reading replaced":
        [(CORPUS_TEST, CORPUS_TAKEN,
          '    return files, (lambda missing: True) if kind == "gone" else (lambda missing: missing.note in tiers_of(rows)[1])\n', 1)],
    # ---- a probe, not a sabotage: a correct driver that reads the catalog through pathlib ----
    "Q17 r5: h100-80's hyperscaler price moved by a cent, the catalog read through pathlib rather than open()":
        [r5_added('import pathlib as _pathlib\n'
                  '_disk = json.loads(_pathlib.Path(os.path.join(ROOT, GPUS_JSON)).read_text(encoding="utf-8"))["data"].get("h100-80") or {}\n'
                  'S["Q17 data: h100-80/hyper moved by a cent"] = [\n'
                  '    (GPUS_JSON, \'"h100-80": { "gb": 80, "bw": 3352, "hyper": \' + json.dumps(_disk.get("hyper")) + ",",\n'
                  '     \'"h100-80": { "gb": 80, "bw": 3352, "hyper": \' + json.dumps(round((_disk.get("hyper") or 0) + 0.01, 2)) + ",", 1)\n'
                  '    if isinstance(_disk.get("hyper"), (int, float)) else (GPUS_JSON, Missing("h100-80 has no hyperscaler price"), "", 1)]\n')],
}

CORPUS_SUITE = [("corpus", ["python3", "tests/corpus.test.py"])]
# The two tests that compare against tests/golden/: a catch made by them alone is hidden
# by regenerating the goldens, so it is reported apart from a real one.
GOLDEN = re.compile(r"golden records")
# This driver, by the name tests/corpus.test.py reports a driver's problems under. Several
# sabotages in S move text this driver's own sabotages anchor on (the pinned line, the
# mixed-row line, a row of the catalog), so on their trees corpus.test.py lists this
# driver's own anchors as stale. That is the sabotage meeting its own driver, not a catch,
# and a corpus verdict made of nothing else is set aside and said so.
ME = os.path.basename(__file__)
PROBLEM = re.compile(r"^\s+((?:engine|workflow)_\w+\.(?:py|sh)|tests/sabotage/anchors\.py)(?:: | no longer loads)")


def run_suites(suites):
    res = {}
    for name, cmd in suites:
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = p.stdout + p.stderr
        fails = [l.strip() for l in out.splitlines() if l.lstrip().startswith("FAIL")]
        errs = [l.strip() for l in out.splitlines() if re.search(r"Error|Traceback|node runner failed", l)]
        tally = re.findall(r"(\d+) passed, (\d+) failed", out)
        res[name] = (p.returncode, fails, errs, tally[-1] if tally else None, out)
    return res


def corpus_problems(out):
    """The sabotages and excerpts tests/corpus.test.py's failures list, by driver: the lines
    under a FAIL naming a driver or anchors.py, not the refusals it lists as excused."""
    lines = [l for l in out.splitlines() if PROBLEM.match(l) and "refused by name," not in l]
    return [l.strip() for l in lines if PROBLEM.match(l).group(1) == ME], \
           [l.strip() for l in lines if PROBLEM.match(l).group(1) != ME]


def pick(sabotages, patterns):
    return [n for n in sabotages if not patterns or any(p.lower() in n.lower() for p in patterns)]


def judge(sabotages, suites, names):
    """run_driver()'s loop with the suites given, every FAIL line printed, and a catch made
    by the golden comparisons alone reported as "gold" rather than "red"."""
    sys.stdout.reconfigure(line_buffering=True)
    touchable = sorted({edit[0] for spec in sabotages.values() for edit in edits_of(spec)})
    red = [k for k, v in run_suites(suites).items() if v[0] != 0]
    if red:
        sys.exit(f"refusing to judge: {', '.join(red)} already red on the unmodified tree")
    print(f"{len(names)} sabotage(s), judged by {', '.join(n for n, _ in suites)}")
    survived, unapplied, golden_only = [], [], []
    for name in names:
        spec = sabotages[name]
        try:
            touched = apply_edits(edits_of(spec))
            if resyncs(spec, touched):
                r = subprocess.run(["python3", "tools/sync_data.py"], capture_output=True, text=True)
                if r.returncode:
                    raise RuntimeError(f"tools/sync_data.py failed after the edit: {(r.stderr or r.stdout).strip()[:200]}")
        except Exception as e:
            print(f"  !! {name}: could not apply: {e}")
            unapplied.append(name)
            restore_files(touchable)
            continue
        try:
            results = run_suites(suites)
        finally:
            restore_files(touched)
        reds = {s: r for s, r in results.items() if r[0] != 0}
        own, others = corpus_problems(reds["corpus"][4]) if "corpus" in reds else ([], [])
        aside = "corpus" in reds and own and not others
        if aside:
            reds.pop("corpus")
        if not reds:
            survived.append(name)
            print(f"  GREEN  {name}   <-- SURVIVED")
        else:
            gold = all(fails and all(GOLDEN.search(f) for f in fails) for rc, fails, errs, _, _ in reds.values())
            if gold:
                golden_only.append(name)
            print(f"  {'gold ' if gold else 'red  '}  {name}")
            for suite, (rc, fails, errs, tally, _) in reds.items():
                print(f"         {suite}[{tally[1] if tally else '?'} failed]")
                for line in (fails or errs or ["(no FAIL line)"])[:12]:
                    print(f"           {line[:170]}")
                if suite == "corpus":
                    print(f"           problems listed: {len(own)} in this driver, {len(others)} in others")
                    for line in others[:4]:
                        print(f"             {line[:170]}")
        if aside:
            print(f"         corpus red on {len(own)} of this driver's own anchors, which the sabotage moved: set aside")
    print(f"\n{len(names) - len(survived) - len(unapplied) - len(golden_only)} caught, "
          f"{len(golden_only)} caught by the goldens alone, {len(survived)} survived"
          + (f", {len(unapplied)} COULD NOT BE APPLIED" if unapplied else ""))
    for name in golden_only:
        print("  GOLDEN ONLY: " + name)
    for name in survived:
        print("  SURVIVED: " + name)
    if unapplied:
        sys.exit(2)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--corpus":
        judge(CORPUS, CORPUS_SUITE, pick(CORPUS, args[1:]))
    elif args and args[0] == "--full":
        judge(S, JUDGING_SUITES + CORPUS_SUITE, pick(S, args[1:]))
    else:
        run_driver(S)
