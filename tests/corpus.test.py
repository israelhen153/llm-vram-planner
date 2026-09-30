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
checks after these therefore apply every sabotage itself, in memory, as the
harness would.

Nobody has to touch the engine for a sabotage to go stale, either. The weekly
price job (tools/price_check.py --apply, then tools/sync_data.py) rewrites
prices, read dates, offer counts and held notes in data/gpus.json and in both
engines' generated blocks. A sabotage quoting one of them applies today and
stops at the job's next run, in a pull request a bot opened, far from whoever
wrote it. So the last checks refresh the catalog in memory the ways that job
does, with its own writers, and apply every sabotage again.

Deliberately NOT one of the suites that judge a sabotage (harness.JUDGING_SUITES):
every sabotage edits the engine, so this would go red under all of them and
report each one as caught whatever the real suites said — the same reason
tests/assets.test.py is excluded.

Run:  python3 tests/corpus.test.py
"""
import builtins
import contextlib
import importlib.util
import io
import json
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
# The tree the refreshes below start from: the engine files, and the one line the pull
# request that confirms a held tier edits by hand, the pinned held set in the price suite.
PRICE_TEST = "tests/price_check.test.py"
TREE = {**ENGINE, PRICE_TEST: open(os.path.join(ROOT, PRICE_TEST), encoding="utf-8").read()}
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


def stale_excerpts(files):
    """Every excerpt that no longer occurs in the file its prefix names, as files has it."""
    stale = []
    for n in excerpts():
        f = next(PREFIX_FILE[p] for p in PREFIX_FILE if n.startswith(p))
        if getattr(anchors, n) not in files[f]:
            first = getattr(anchors, n).strip().split("\n")[0][:70]
            stale.append(f"{n} (in {f}; begins {first!r})")
    return stale


def check_every_excerpt_still_occurs_in_its_file():
    """The one this file exists for."""
    stale = stale_excerpts(ENGINE)
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


def sabotages_of(driver, directory=SABOTAGE_DIR):
    """A driver's sabotages, read as data. Every driver builds S at import and ends
    by handing it to run_driver(), which does nothing while this runs: nothing is
    applied and no suite runs, for a driver another driver imports as well."""
    import harness
    real, harness.run_driver = harness.run_driver, lambda sabotages: None
    try:
        spec = importlib.util.spec_from_file_location(driver[:-3], os.path.join(directory, driver))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        harness.run_driver = real
    return module.S


def why_it_would_not_apply(spec, tree=None, excused=lambda missing: False):
    """What harness.apply_edits() would refuse this sabotage for on the tree as it
    is, or None. The same check, harness.refusal(), edit by edit and in order, on
    copies: a count that does not match, or a target its driver found gone.

    tree holds files as another state of the repository would have them, by path;
    a file it does not hold is read from disk. excused says of a target a driver
    found gone (a harness.Missing) whether that state had to take it away: a note
    a confirmed reading replaced, or anything a person removed. Such a sabotage is
    refused by name, as it has to be, and that refusal is not counted against it."""
    import harness
    if not harness.edits_of(spec):
        return harness.NO_EDIT
    files = {}
    for f, old, new, count in harness.edits_of(spec):
        if isinstance(old, harness.Missing) and excused(old):
            continue
        if f not in files:
            if tree is not None and f in tree:
                files[f] = tree[f]
            else:
                try:
                    files[f] = open(os.path.join(ROOT, f), encoding="utf-8").read()
                except OSError as e:
                    return f"{f}: {e.strerror}"
        why = harness.refusal(f, files[f], old, count)
        if why:
            return why
        files[f] = files[f].replace(old, new)
    return None


PY_DRIVERS = {d: sabotages_of(d) for d in DRIVERS if d.endswith(".py")}


def replaced_by_a_reading(catalog):
    """Whether a sabotage built on a price note may be refused by name on this catalog:
    only when the note's tier now carries a reading (priceSource), which is what takes a
    note out when the price job confirms one. A note gone with no reading in its place
    is a catalog the price suite refuses, not one to excuse here."""
    rows = json.loads(catalog)["data"]
    return lambda missing: missing.note is not None and missing.note[1] in (
        (rows.get(missing.note[0]) or {}).get("priceSource") or {})


def check_every_sabotage_still_applies(base=None):
    """The check the excerpts can't give. Only a sabotage's edits are checked: the
    catalog re-sync some of them run afterwards is not. A sabotage built on a held
    note is refused by name once a confirmed reading has replaced that note, as on
    the price pull request that confirms it, and is listed rather than counted: its
    cold check found this check red on exactly that pull request. base is another
    state of the tree, by path, as refresh() takes it: the drivers are loaded, and
    their sabotages checked, with open() reading it. Returns what it listed."""
    base = base or TREE
    changed = {f: text for f, text in base.items() if text != TREE.get(f)}
    stale, listed, excused = [], [], replaced_by_a_reading(base["data/gpus.json"])
    with reading_from(changed) if changed else contextlib.nullcontext():
        drivers = {d: sabotages_of(d) for d in PY_DRIVERS} if changed else PY_DRIVERS
        empty = [d for d, sabotages in drivers.items() if not sabotages]
        assert not empty, f"drivers with no sabotages, so nothing here checks them: {empty}"
        for d, sabotages in drivers.items():
            for name, spec in sabotages.items():
                why = why_it_would_not_apply(spec, excused=excused)
                if why:
                    stale.append(f"{d}: {name}: {why}")
                elif why_it_would_not_apply(spec):
                    listed.append(f"{d}: {name}: refused by name, its note replaced by a reading: "
                                  f"{why_it_would_not_apply(spec)}")
    for line in listed:
        print(f"       {line}")
    assert not stale, (
        f"{len(stale)} sabotage(s) no longer apply, so the corpus would skip them:\n         "
        + "\n         ".join(stale) +
        "\n       Point each at the engine's new text in the pull request that moved it. An "
        "excerpt more than one sabotage shares belongs in tests/sabotage/anchors.py.")
    return listed

test(f"every sabotage still applies to the tree as it is "
     f"({sum(map(len, PY_DRIVERS.values()))} sabotages in {len(PY_DRIVERS)} Python drivers)",
     check_every_sabotage_still_applies)


class _Wrote(Exception):
    """The shell driver's heredoc reached its write: all it requires of the row held."""


def shell_driver_problems(catalog, src=None):
    """What would stop engine_r1_perfkey_typo.sh applying to this text of
    data/gpus.json. Its heredoc is run for each catalog row its loop names, with open()
    serving this text and stopping it at its write: whatever it requires of the row,
    written in any form, held if it gets there. Reading the script for its
    requirements missed a second assert (its first cold check) and an `in` test (the
    second), so nothing reads it any more."""
    src = src if src is not None else open(os.path.join(SABOTAGE_DIR, SHELL_DRIVERS[0]), encoding="utf-8").read()
    loop = re.search(r"^for slug in ([\w .-]+); do$", src, re.M)
    body = re.search(r"<<'PY'[^\n]*\n(.*?)^PY$", src, re.S | re.M)
    if not (loop and body):
        return [f"{SHELL_DRIVERS[0]} no longer has the loop and the heredoc this runs"]
    code = compile(body.group(1), SHELL_DRIVERS[0], "exec")

    def served(file, mode="r", *args, **kwargs):
        if any(c in mode for c in "wax+"):
            raise _Wrote()
        if os.path.normpath(str(file)) == "data/gpus.json":
            return io.StringIO(catalog)
        return open(os.path.join(ROOT, file), mode, *args, **kwargs)

    problems, argv = [], sys.argv
    for slug in loop.group(1).split():
        sys.argv = [SHELL_DRIVERS[0], slug]
        try:
            exec(code, {"__name__": "__main__", "open": served})
            problems.append(f"{SHELL_DRIVERS[0]}: {slug}: its heredoc ended without writing the catalog")
        except _Wrote:
            pass
        except Exception as e:
            problems.append(f"{SHELL_DRIVERS[0]}: {slug}: its heredoc stops before its edit: {type(e).__name__}: {e}")
        finally:
            sys.argv = argv
    return problems


def check_the_shell_driver_still_applies():
    """engine_r1_perfkey_typo.sh edits data/gpus.json from a heredoc, so it cannot be
    read as data like the others; shell_driver_problems() runs the heredoc instead."""
    shell = [d for d in DRIVERS if d.endswith(".sh")]
    assert shell == SHELL_DRIVERS, (
        f"shell drivers {shell}, where this file checks {SHELL_DRIVERS}: the new one's sabotages "
        f"are invisible to the check above. Write it on run_driver() in Python, or check it here")
    problems = shell_driver_problems(ENGINE["data/gpus.json"])
    assert not problems, "\n       ".join(problems)

test("the shell driver's catalog rows still carry what it corrupts", check_the_shell_driver_still_applies)


print("\nEvery sabotage still applies after the weekly price job refreshes the catalog")

# The job is tools/price_check.py --apply and then tools/sync_data.py, and these are
# its own writers: what it does to a row, how it writes the file, and how the
# engines' blocks are generated from that file. A simulation writing them its own
# way would test its own way.
sys.path.insert(0, os.path.join(ROOT, "tools"))
import price_check  # noqa: E402
import sync_data  # noqa: E402

# A read date no row carries, so no quoted date survives a refresh by coincidence.
READ_ON = "2099-12-31"
# Every tier the job reads, from its own SOURCE_MAP rather than a list here: a tier it
# starts reading is refreshed here the day SOURCE_MAP says so.
AUTOMATED = [(slug, tier, cfg["primary"]) for slug, tiers in price_check.SOURCE_MAP.items()
             for tier, cfg in tiers.items() if "primary" in cfg]


def tiers_of(rows):
    """The automated tiers that carry a reading in these rows, and those still held under a note."""
    read = [(s, t) for s, t, _ in AUTOMATED if t in (rows.get(s, {}).get("priceSource") or {})]
    held = [(s, t) for s, t, _ in AUTOMATED if (s, t) not in read and t in (rows.get(s, {}).get("priceNote") or {})]
    return read, held


ROWS = json.loads(ENGINE["data/gpus.json"])["data"]
READ, HELD = tiers_of(ROWS)


def counted(sku, spec):
    """The numbers in a reading's SKU that the reading counted itself, as Vast.ai's
    "median of 43 verified offers" does: every number outside what SOURCE_MAP asks the
    source for (an instance type, a plan, a GPU name), as re.Match objects."""
    asked = [m.span() for v in spec.values() if isinstance(v, str) for m in re.finditer(re.escape(v), sku)]
    return [m for m in re.finditer(r"\d+", sku) if not any(a <= m.start() and m.end() <= b for a, b in asked)]


def reread_sku(sku, spec):
    """The SKU the next reading of a tier records. What SOURCE_MAP asks for comes back
    as it was, and a number the reading counted is a different number next time. Its
    first digit changes: a count moved by one kept it, and a quote through that digit
    survived (the second cold check's Q15)."""
    for m in reversed(counted(sku, spec)):
        sku = sku[:m.start()] + str(int(m.group()[0]) % 9 + 1) + m.group()[1:] + sku[m.end():]
    return sku


def moved_from(price):
    """A price a reading moves this one to, sharing no text with it: a cent was the move
    here, and 12.3 is still inside 12.31, so a quote with nothing after the price survived
    it (the cold check's first survivor). Leading digits change instead."""
    moved = round(price * 1.5 + 1.17, 2)
    while str(price) in str(moved) or str(moved) in str(price):
        moved = round(moved + 1.01, 2)
    return moved


def refresh(kind, base=None, on=READ_ON):
    """The three files the job rewrites, as one kind of run leaves them, in memory, and
    a test of which refusals by name that tree has to cause. base holds the files the run
    starts from, by path, the tree as it is by default, and on is the day it reads.

    reread     every automated tier with a reading is read again, on a new day, at the
               price it recorded
    moved      the same, with every one of those prices moved, the catalog's too
    confirmed  every automated tier still held under a note gets its first reading, at a
               price of its own, as a first reading comes (h100-80's came back 44% higher),
               which takes the note out and moves the catalog's price
    gone       every automated tier's reading and note is taken away, as a person could
               take them: a sabotage built on one has to be refused by name, and its
               driver has to load and keep every other sabotage

    Each kind checks it did what it says to every tier it names: its cold check blinded
    the moved kind and nothing noticed.

    The refusals it has to cause: after gone, any; after the others, a sabotage built on
    a note a reading has replaced, in this run or an earlier one. Only this run's were
    excused once, and the pull request confirming h100-80/spot, built with the job's own
    writers, then failed every refresh here on the refusal it had caused itself."""
    base = base or TREE
    rows = json.loads(base["data/gpus.json"])["data"]
    data = json.loads(base["data/gpus.json"])["data"]
    outcomes = []
    for slug, tier, spec in AUTOMATED:
        row = data[slug]
        source = (row.get("priceSource") or {}).get(tier)
        if source and kind in ("reread", "moved"):
            price = row[tier] if kind == "reread" else moved_from(row[tier])
            reading = price_check.Reading(
                provider=source["provider"], sku=reread_sku(source["sku"], spec), region=source["region"],
                price_per_gpu=source["price"] if kind == "reread" else price, date=on, evidence="")
            outcomes.append(price_check.Outcome(slug, tier, "CONFIRMED" if kind == "reread" else "MOVED",
                                                current=row[tier], proposed=price, reading=reading))
        elif not source and kind == "confirmed" and tier in (row.get("priceNote") or {}):
            price = moved_from(row[tier])
            reading = price_check.Reading(
                provider=spec["kind"], sku="(simulated)", region=spec.get("region", "global"),
                price_per_gpu=price, date=on, evidence="")
            outcomes.append(price_check.Outcome(slug, tier, "MOVED", current=row[tier],
                                                proposed=price, reading=reading))
    price_check.apply_outcomes(data, outcomes)
    changed = sorted({oc.slug for oc in outcomes})
    if kind == "gone":
        for slug, tier, _ in AUTOMATED:
            for field in ("priceSource", "priceNote"):
                if tier in (data[slug].get(field) or {}):
                    del data[slug][field][tier]
                    changed.append(slug)
                    if not data[slug][field]:
                        del data[slug][field]
    did_what_it_says(kind, data, rows, on)
    catalog = price_check.apply_to_text(base["data/gpus.json"], data, sorted(set(changed)), on)
    files = {"data/gpus.json": catalog}
    written = json.loads(catalog)["data"]   # what tools/sync_data.py reads: the file the job wrote
    for f, render in (("index.html", sync_data.render_gpu_js), ("generate_report.py", sync_data.render_gpu_py)):
        files[f] = sync_data.BLOCK_RES["GPU_TABLE"].sub(
            lambda m, render=render: render(written).rstrip("\n"), base[f], count=1)
    if tiers_of(written)[1] != tiers_of(rows)[1]:
        files[PRICE_TEST] = pinned(base[PRICE_TEST], tiers_of(written)[1])
    return files, (lambda missing: True) if kind == "gone" else replaced_by_a_reading(catalog)


PINNED_LINE = re.compile(r"^HELD_UNDER_A_NOTE = .*$", re.M)


def pinned(test_src, held):
    """The price suite with its pinned held set written as the pull request that changes
    the set writes it, the one line such a pull request edits by hand. Left as it was, a
    sabotage quoting that line applied here and not on the real pull request (the second
    cold check's Q16)."""
    tiers = {}
    for slug, tier in held:
        tiers.setdefault(slug, []).append(tier)
    line = "HELD_UNDER_A_NOTE = {" + ", ".join(f"{json.dumps(s)}: {json.dumps(t)}" for s, t in tiers.items()) + "}"
    new, n = PINNED_LINE.subn(lambda m: line, test_src)
    assert n == 1, f"{PRICE_TEST} pins HELD_UNDER_A_NOTE on {n} line(s), where the refresh rewrites one"
    return new


def did_what_it_says(kind, data, rows, on):
    """That a refresh changed, on every tier it names, what it says it changes, from rows."""
    read, held = tiers_of(rows)
    wrong = []
    for slug, tier in (read if kind in ("reread", "moved") else held if kind == "confirmed" else
                       [(s, t) for s, t, _ in AUTOMATED]):
        row, was = data[slug], rows[slug]
        source, note = (row.get("priceSource") or {}).get(tier), (row.get("priceNote") or {}).get(tier)
        if kind == "reread" and not (source and source["date"] == on != was["priceSource"][tier]["date"]):
            wrong.append(f"{slug}/{tier}: not re-read on a new day")
        if kind in ("moved", "confirmed") and (row[tier] == was[tier] or str(was[tier]) in str(row[tier])):
            wrong.append(f"{slug}/{tier}: its price {was[tier]} did not move to one sharing no text with it ({row[tier]})")
        if kind == "confirmed" and (note or not source):
            wrong.append(f"{slug}/{tier}: its note was not replaced by a reading")
        if kind == "gone" and (source or note):
            wrong.append(f"{slug}/{tier}: still carries a reading or a note")
    assert not wrong, f"the {kind} refresh did not do what it says, so it checks less than it claims: {wrong}"


@contextlib.contextmanager
def reading_from(files):
    """While this holds, open() hands a driver the text in files for each file it holds,
    as though the refresh had been written, and refuses to open anything for writing:
    this check writes nothing to disk. io.open too, which pathlib reads through: a driver
    reading the catalog that way was failed here for a price it read correctly (the
    second cold check's Q17). Drivers already imported are forgotten first, so one that
    another driver imports reads the refreshed files too."""
    served = {os.path.realpath(os.path.join(ROOT, f)): text for f, text in files.items()}
    real = builtins.open

    def refreshed_open(file, mode="r", *args, **kwargs):
        if any(c in mode for c in "wax+"):
            raise PermissionError(f"the refresh check writes nothing, and {file} was opened with {mode!r}")
        if isinstance(file, (str, os.PathLike)) and os.path.realpath(file) in served:
            text = served[os.path.realpath(file)]
            return io.BytesIO(text.encode("utf-8")) if "b" in mode else io.StringIO(text)
        return real(file, mode, *args, **kwargs)

    def forget_drivers():
        for name in [m for m in sys.modules if re.match(r"(engine|workflow)_", m)]:
            del sys.modules[name]

    forget_drivers()
    builtins.open = io.open = refreshed_open
    try:
        yield
    finally:
        builtins.open = io.open = real
        forget_drivers()


def check_after(kind, excused, base=None, extra=None):
    """Every sabotage in every driver, loaded against the refreshed files, still applies
    to them, and every one keeps its name: a sabotage has to be refused by name, never
    dropped, and a name that moves with the data reads as one dropped and one added (the
    second cold check's Q12). A refusal the refreshed tree has to cause (see refresh())
    goes in excused instead. base is the tree the refresh starts from, as refresh()
    takes it; extra maps drivers written against this check to their directory."""
    base = base or TREE
    try:
        files, taken = refresh(kind, base)
    except SystemExit as e:   # the job's writers stop, rather than write a catalog they cannot
        raise AssertionError(f"the price job's own writers refuse this catalog: {e}") from None
    if kind != "confirmed" or tiers_of(json.loads(base["data/gpus.json"])["data"])[1]:
        assert files["data/gpus.json"] != base["data/gpus.json"], "the refresh rewrote nothing, so this checks nothing"
    tree = {**base, **files}
    drivers = {**{d: SABOTAGE_DIR for d in PY_DRIVERS}, **(extra or {})}
    had = {d: set(PY_DRIVERS[d]) for d in PY_DRIVERS}
    changed = {f: text for f, text in base.items() if text != TREE.get(f)}
    with reading_from(changed) if changed else contextlib.nullcontext():
        had.update({d: set(sabotages_of(d, where)) for d, where in (extra or {}).items()})
    problems, loaded = [], {}
    # The base's own changes too, where the refresh leaves them: the confirming pull
    # request's pinned line, which a re-read after it does not touch. Serving only the
    # refresh's files, a driver reading that line as it loaded read the disk's.
    with reading_from({**changed, **files}):
        for d, where in drivers.items():
            try:
                loaded[d] = sabotages_of(d, where)
            except Exception as e:
                problems.append(f"{d} no longer loads: {type(e).__name__}: {e}")
    for d, sabotages in loaded.items():
        lost = sorted(had[d] - set(sabotages))
        if lost:
            problems.append(f"{d}: {len(lost)} sabotage(s) not found by name after the refresh, dropped or "
                            f"renamed where each had to keep its name: {lost[:3]}")
        for name, spec in sabotages.items():
            why = why_it_would_not_apply(spec, tree)
            if why and why_it_would_not_apply(spec, tree, taken) is None:
                excused.append(f"{d}: {name}: refused by name, as it has to be: {why}")
            elif why:
                problems.append(f"{d}: {name}: {why_it_would_not_apply(spec, tree, taken)}")
    problems += [f"tests/sabotage/anchors.py: {s}" for s in stale_excerpts(tree)]
    problems += shell_driver_problems(files["data/gpus.json"])
    assert not problems, (
        f"{len(problems)} thing(s) the job's next run would break:\n         " + "\n         ".join(problems) +
        "\n       A price, a reading's date, price or offer count, and a note a confirmed reading replaces "
        "are all rewritten by the price job. Read them from data/gpus.json with open() as the driver "
        "loads, and quote only what the job never rewrites: see tests/sabotage/README.md.")


def refreshes(read, held):
    return [
        ("reread", f"re-reads the {len(read)} automated tiers that carry a reading, each on a new day"),
        ("moved", f"moves each of those {len(read)} prices"),
        ("confirmed", "confirms a first reading, at a new price, on every automated tier still held under a note ("
                      + (", ".join(f"{s}/{t}" for s, t in held) or "none") + ")"),
        ("gone", f"takes away the reading or note of all {len(AUTOMATED)} automated tiers, as a person could"),
    ]


for kind, label in refreshes(READ, HELD):
    excused = []
    test(f"after a refresh that {label}, every sabotage in every driver still applies",
         lambda kind=kind, excused=excused: check_after(kind, excused))
    for line in excused:
        print(f"       {line}")

# And from the tree the pull request confirming the held tiers leaves: a sabotage built
# on a note that pull request's reading replaced is refused by name from then on, and
# only a note the refresh itself replaced was excused here.
if HELD:
    def confirming_pull_request():
        return {**TREE, **refresh("confirmed", on="2099-12-30")[0]}

    def refused_on_its_notes(base, refused):
        """That the tree a check read is base: every sabotage built on a note the pull
        request replaced is among those it refused by name, and each would apply to the
        tree as it is."""
        import harness
        with reading_from({f: text for f, text in base.items() if text != TREE.get(f)}):
            on_a_note = [f"{d}: {name}: " for d in PY_DRIVERS for name, spec in sabotages_of(d).items()
                         if any(isinstance(old, harness.Missing) and old.note in HELD
                                for _, old, _, _ in harness.edits_of(spec))]
        assert on_a_note, f"no sabotage is built on a note {confirming} carried, so this tree is checked for nothing"
        unrefused = [n for n in on_a_note if not any(e.startswith(n) for e in refused)]
        assert not unrefused, f"built on a note the pull request replaced, and not refused by name here: {unrefused}"

    def check_after_confirming(kind, excused):
        base = confirming_pull_request()
        check_after(kind, excused, base)
        refused_on_its_notes(base, excused)

    def check_the_confirming_tree():
        base = confirming_pull_request()
        refused_on_its_notes(base, check_every_sabotage_still_applies(base))

    confirming = ", ".join(f"{s}/{t}" for s, t in HELD)
    test(f"on the tree confirming {confirming} leaves, every sabotage still applies", check_the_confirming_tree)
    for kind, label in refreshes(READ + HELD, []):
        excused = []
        test(f"on the tree confirming {confirming} leaves, after a refresh that {label}, every sabotage still applies",
             lambda kind=kind, excused=excused: check_after_confirming(kind, excused))
        print(f"       {len(excused)} refused by name, as they have to be")
else:
    print("       no automated tier is held under a note today, so no pull request is left to confirm one")


print("\nThe refresh check sees what a sabotage could quote")


def row_through(catalog, slug, text):
    """slug's catalog row, from its start through text: a quote a sabotage could write
    with nothing after the value it ends on."""
    (line,) = [r for r in catalog.splitlines() if r.strip().startswith(f'"{slug}":')]
    start = line.index(f'"{slug}":')
    return line[start:line.index(text, start) + len(text)]


def check_the_refresh_check_sees_a_quoted_price():
    """A sabotage quoting a price the job moves has to stop applying after the refresh
    that moves it, even quoted with nothing after it. The cold check quoted h100-80's
    hyperscaler price that way and a move of a cent left it inside the new price; it
    quoted h100-80's held spot price and a first reading at the catalog's own price left
    it standing. Every read tier's and every held tier's price, each quoted from its
    row's start through the price, derived from the catalog."""
    survived = []
    for kind, tiers in (("moved", READ), ("confirmed", HELD)):
        files, _ = refresh(kind)
        tree = {**TREE, **files}
        for slug, tier in tiers:
            quote = row_through(ENGINE["data/gpus.json"], slug, f'"{tier}": {json.dumps(ROWS[slug][tier])}')
            spec = [("data/gpus.json", quote, quote + "0", 1)]
            assert why_it_would_not_apply(spec) is None, f"{slug}/{tier}: the quote does not apply before the refresh"
            if why_it_would_not_apply(spec, tree) is None:
                survived.append(f"{kind}: {slug}/{tier}: ...{quote[-32:]!r}")
    assert not survived, f"a quoted price survived the refresh that moves it: {survived}"

test("a price quoted with nothing after it stops applying after the refresh that moves it",
     check_the_refresh_check_sees_a_quoted_price)


def check_the_refresh_check_runs_the_shell_driver():
    """A precondition the shell driver sets on a price its rows carry has to stop it
    applying after the refresh that moves the price, however it is written: a second
    line.count() assert (its first cold check) and an `in` test (its second, Q14). The
    driver's own script, each added before its first assert, for one of its rows."""
    src = open(os.path.join(SABOTAGE_DIR, SHELL_DRIVERS[0]), encoding="utf-8").read()
    loop = re.search(r"^for slug in ([\w .-]+); do$", src, re.M).group(1).split()
    slug, tier = next((s, t) for s, t in READ if s in loop)
    quote = f'"{tier}": {json.dumps(ROWS[slug][tier])},'
    files, _ = refresh("moved")
    for form in (f"assert line.count({quote!r}) == 1", f"assert {quote!r} in line"):
        added = src.replace("assert line.count(", f"if slug == {slug!r}:\n    {form}\nassert line.count(", 1)
        assert shell_driver_problems(TREE["data/gpus.json"], src=added) == [], f"{form}: does not hold before the refresh"
        assert shell_driver_problems(files["data/gpus.json"], src=added), (
            f"{form}: the refresh moved {quote!r} in {slug}'s row, and nothing noticed")

test("a price the shell driver requires, in any form, stops it applying after the refresh",
     check_the_refresh_check_runs_the_shell_driver)


def check_the_refresh_check_sees_a_quoted_offer_count():
    """A reading's SKU quoted through the first digit of a number the reading counted
    itself (Vast.ai's "median of 46 verified offers") has to stop applying after the
    re-read: the count moved by one kept its first digit, and the quote survived (the
    second cold check's Q15). Every such number in every reading, derived."""
    files, _ = refresh("reread")
    tree, quoted, survived = {**TREE, **files}, 0, []
    for slug, tier, spec in AUTOMATED:
        source = (ROWS[slug].get("priceSource") or {}).get(tier)
        for m in counted(source["sku"], spec) if source else []:
            text = '"sku": ' + json.dumps(source["sku"])[:m.start() + 2]
            quote = row_through(TREE["data/gpus.json"], slug, text)
            edit = [("data/gpus.json", quote, quote + "0", 1)]
            assert why_it_would_not_apply(edit) is None, f"{slug}/{tier}: the quote does not apply before the re-read"
            quoted += 1
            if why_it_would_not_apply(edit, tree) is None:
                survived.append(f"{slug}/{tier}: ...{quote[-40:]!r}")
    assert quoted, "no reading counts anything of its own, so this checked nothing"
    assert not survived, f"a count quoted through its first digit survived the re-read: {survived}"

test("an offer count quoted through its first digit stops applying after the re-read",
     check_the_refresh_check_sees_a_quoted_offer_count)


def check_the_refresh_check_sees_the_pinned_line_change():
    """The pull request that confirms a held tier edits one line by hand, the pinned held
    set in the price suite, and a sabotage quoting that line has to stop applying on its
    tree as a quoted price does: the simulated pull request left the line as it was, and
    the quote passed here and failed on the real one (the second cold check's Q16)."""
    if not HELD:
        print("       no automated tier is held under a note today, so no pull request changes the pinned set")
        return
    (line,) = [l for l in TREE[PRICE_TEST].splitlines() if PINNED_LINE.fullmatch(l)]
    edit = [(PRICE_TEST, line, "HELD_UNDER_A_NOTE = {}", 1)]
    assert why_it_would_not_apply(edit) is None, "the quote of the pinned line does not apply to the tree as it is"
    files, _ = refresh("confirmed")
    assert why_it_would_not_apply(edit, {**TREE, **files}), (
        "the tree confirming the held tiers still carries the pinned line as it was")

test("a quote of the pinned held set stops applying on the tree that confirms a held tier",
     check_the_refresh_check_sees_the_pinned_line_change)


def check_an_edit_that_changes_nothing_is_refused():
    """A sabotage with no edit, or with an edit expecting its text 0 times, leaves the
    tree as it was, and judged, it reads as a survivor. The harness a run uses refuses
    both, and so does this check (the second cold check's Q6 and Q7)."""
    import harness
    assert why_it_would_not_apply([]) == harness.NO_EDIT, "a sabotage with no edit applies here"
    try:
        harness.apply_edits([])   # nothing to write either way
        raise AssertionError("harness.apply_edits() accepts a sabotage with no edit")
    except RuntimeError:
        pass
    assert harness.refusal("data/gpus.json", "text", "absent", 0), "an edit expecting its text 0 times is accepted"
    assert why_it_would_not_apply([("data/gpus.json", "absent from the catalog", "x", 0)]), (
        "an edit expecting its text 0 times applies here")

test("a sabotage that changes nothing is refused, by a run and by this check",
     check_an_edit_that_changes_nothing_is_refused)


# Drivers written against the refresh check, each the shape of a way the second cold
# check got a sabotage past it. {slug}, {tier}, {price} and {date} are one read tier's,
# from the catalog as it is.
AGAINST_THE_CHECK = {
    "engine_zz_count_at_load.py": ("reread", True, """
import json, os
from harness import run_driver, ROOT
TEXT = open(os.path.join(ROOT, "data/gpus.json"), encoding="utf-8").read()
QUOTE = '"date": ' + json.dumps({date!r})
S = {{"Z6 data: a reading's date quoted as it was, counted as the driver loads": [
    ("data/gpus.json", QUOTE, '"date": "2026-01-01"', TEXT.count(QUOTE))]}}
run_driver(S)
"""),
    "engine_zz_no_edit_once_gone.py": ("gone", True, """
import json, os
from harness import run_driver, ROOT
ROWS = json.load(open(os.path.join(ROOT, "data/gpus.json"), encoding="utf-8"))["data"]
READING = (ROWS[{slug!r}].get("priceSource") or {{}}).get({tier!r})
S = {{"Z7 data: a reading's provider swapped, and no edit once it is gone": [
    ("data/gpus.json", '"provider": ' + json.dumps(READING["provider"]) + ', "sku": ' + json.dumps(READING["sku"]),
     '"provider": "nobody", "sku": ' + json.dumps(READING["sku"]), 1)] if READING else []}}
run_driver(S)
"""),
    "engine_zz_named_after_a_price.py": ("moved", True, """
import json, os
from harness import run_driver, ROOT
ROWS = json.load(open(os.path.join(ROOT, "data/gpus.json"), encoding="utf-8"))["data"]
S = {{"Z12 data: " + {slug!r} + " at " + json.dumps(ROWS[{slug!r}][{tier!r}]) + ", its row's key padded": [
    ("data/gpus.json", json.dumps({slug!r}) + ": {{", json.dumps({slug!r}) + ":  {{", 1)]}}
run_driver(S)
"""),
    "engine_zz_reads_the_pin.py": ("moved on the confirming tree", False, """
import os, re
from harness import run_driver, ROOT
SRC = open(os.path.join(ROOT, "tests", "price_check.test.py"), encoding="utf-8").read()
PIN = re.search(r"^HELD_UNDER_A_NOTE = .*$", SRC, re.M).group(0)
S = {{"Z16 price: the pinned held set, read as the driver loads, given a comment": [
    ("tests/price_check.test.py", PIN, PIN + "  # z", 1)]}}
run_driver(S)
"""),
    "engine_zz_read_through_pathlib.py": ("moved", False, """
import json, pathlib
from harness import run_driver, ROOT
TEXT = pathlib.Path(ROOT, "data/gpus.json").read_text(encoding="utf-8")
AT = '"' + {tier!r} + '": ' + json.dumps(json.loads(TEXT)["data"][{slug!r}][{tier!r}])
LINE = next(l for l in TEXT.splitlines() if l.strip().startswith(json.dumps({slug!r}) + ":"))
QUOTE = LINE[LINE.index(json.dumps({slug!r})):LINE.index(AT) + len(AT)]
S = {{"Z17 data: a price moved, the catalog read through pathlib": [("data/gpus.json", QUOTE, QUOTE + "0", 1)]}}
run_driver(S)
"""),
}


def check_the_refresh_check_holds_drivers_written_against_it():
    """Each driver above, loaded from a scratch directory as the corpus's own are, and
    held to the refresh its shape breaks on. Three have to be reported: one whose count
    of a date was taken as it loaded, one with no edit once its reading is gone, one
    named after a price. Two read correctly and have to pass: one through pathlib, one
    reading the pinned held set as it loads, on the confirming pull request's tree
    (replaying the second cold check's driver found that one failed). Each applies to
    the tree as it is first, or this would check nothing."""
    import tempfile
    slug, tier = READ[0]
    fill = {"slug": slug, "tier": tier, "date": ROWS[slug]["priceSource"][tier]["date"]}
    with tempfile.TemporaryDirectory() as where:
        for name, (_, _, src) in AGAINST_THE_CHECK.items():
            with open(os.path.join(where, name), "w", encoding="utf-8") as f:
                f.write(src.format(**fill))
            for sabotage, spec in sabotages_of(name, where).items():
                assert why_it_would_not_apply(spec) is None, f"{name}: {sabotage} does not apply to the tree as it is"
        wrong = []
        for kind in dict.fromkeys(k for k, _, _ in AGAINST_THE_CHECK.values()):
            names = {n: flagged for n, (k, flagged, _) in AGAINST_THE_CHECK.items() if k == kind}
            base = None
            if kind.endswith(" on the confirming tree"):
                if not HELD:
                    continue   # no pull request is left to confirm a held tier
                kind, base = kind.split(" ")[0], {**TREE, **refresh("confirmed", on="2099-12-30")[0]}
            try:
                check_after(kind, [], base=base, extra={n: where for n in names})
                report = ""
            except AssertionError as e:
                report = str(e)
            wrong += [f"{kind}: {n} {'passed' if flagged else 'was reported'}"
                      for n, flagged in names.items() if (n in report) != flagged]
    assert not wrong, f"the refresh check misjudged drivers written against it: {wrong}"

test("a count taken at load, an empty sabotage and a name that moves are reported; correct reads pass",
     check_the_refresh_check_holds_drivers_written_against_it)


def check_a_note_a_reading_replaced_is_listed_not_counted():
    """On the price pull request that confirms a held tier, the tree as it is has that
    tier's note replaced by a reading, and a sabotage built on the note refuses by name.
    The check of the tree as it is lists that refusal and does not count it: its cold
    check found it counted, which would have turned that pull request red whatever its
    prices said. A note taken away with no reading in its place is still counted."""
    import harness
    if not HELD:
        print("       no automated tier is held under a note today, so no note can be confirmed")
        return
    files, _ = refresh("confirmed")
    tree, excused = {**TREE, **files}, replaced_by_a_reading(files["data/gpus.json"])
    with reading_from(files):
        loaded = {d: sabotages_of(d) for d in PY_DRIVERS}
    counted = [f"{d}: {name}: {why}" for d, sabotages in loaded.items() for name, spec in sabotages.items()
               if (why := why_it_would_not_apply(spec, tree, excused))]
    assert not counted, f"refusals counted on the tree a confirmed reading leaves: {counted}"
    gone, _ = refresh("gone")
    assert not replaced_by_a_reading(gone["data/gpus.json"])(harness.Missing("gone", note=HELD[0])), (
        "a note taken away with no reading in its place was excused")

test("a sabotage built on a note a reading replaced is listed, not counted, on that tree",
     check_a_note_a_reading_replaced_is_listed_not_counted)


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


RUNNER_HARNESS = """import fcntl, os, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JUDGING_SUITES = [("model", ["true"]), ("seventh", ["true"])]


def require_green_baseline():
    pass


def run_judging_suites():
    with open(os.path.join(os.path.dirname(ROOT), ".lock")) as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            held = "held"
        else:
            held = "free"
            fcntl.flock(fh, fcntl.LOCK_UN)
    with open(os.environ["RUNNER_TEST_LOG"], "a") as fh:
        fh.write(held + "\\n")
    with open(os.path.join(ROOT, "target.txt")) as fh:
        sabotaged = "sabotaged" in fh.read()
    with open(os.path.join(ROOT, "tests", "golden", "g.json")) as fh:
        emptied = fh.read().strip() == "{}"
    got = {"model": (1, ["FAIL the golden"], [], "") if emptied else (0, [], [], ""),
           "seventh": (1, ["FAIL a real catch"], [], "") if sabotaged else (0, [], [], "")}
    return {name: got[name] for name, _ in JUDGING_SUITES}


def run_driver(sabotages):
    caught = survived = 0
    for name, edits in sabotages.items():
        for f, old, new, count in edits:
            path = os.path.join(ROOT, f)
            with open(path) as fh:
                text = fh.read()
            with open(path, "w") as fh:
                fh.write(text.replace(old, new))
        judged = run_judging_suites()
        subprocess.run(["git", "checkout", "--", "."], cwd=ROOT, check=True)
        if any(r[0] for r in judged.values()):
            caught += 1
            print(f"  red    {name}")
        else:
            survived += 1
            print(f"  GREEN  {name}   <-- SURVIVED")
    print("")
    print(f"{caught} caught, {survived} survived")
"""

RUNNER_DRIVER = """S = {"caught by the seventh suite alone": [("target.txt", "clean", "sabotaged", 1)],
     "survives every suite": [("target.txt", "clean", "tidied", 1)]}
"""


def throwaway_repo(tree):
    """A git runner for a throwaway repository: no hooks, no signing, its own name."""
    import subprocess

    def git(*args):
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                               *args], cwd=tree, check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q", "--template=")  # no hooks from a global template
    return git


def check_a_run_judges_the_asked_commit_with_every_suite_under_its_lock():
    """A whole run of this checkout's parallel.py, copied into a throwaway repository
    whose own checkout sits at a commit without the driver, asked to judge the next
    commit, which has it. The harness there has a seventh judging suite that
    ORDER_HINT does not name and that alone catches one sabotage, and one sabotage
    that survives every suite. The run must judge the commit asked for, reading its
    drivers and their sabotages from that commit and not from its own checkout; run
    every suite under --early-exit, the seventh too; exit non-zero on the survivor; and
    hold the workers' lock for as long as any suite runs. The third cold check broke
    each of these in one line and the suite stayed green (R17, R18, R21, R23, R27), and
    found the lock already let go while judging, because its name was reused."""
    import glob, shutil, subprocess, tempfile
    with tempfile.TemporaryDirectory() as d:
        repo, log = os.path.join(d, "repo"), os.path.join(d, "judging.log")
        sab = os.path.join(repo, "tests", "sabotage")
        os.makedirs(sab)
        os.makedirs(os.path.join(repo, "tests", "golden"))
        shutil.copy(os.path.join(ROOT, "tests", "sabotage", "parallel.py"), sab)
        for path, text in ((os.path.join(sab, "harness.py"), RUNNER_HARNESS),
                           (os.path.join(repo, "target.txt"), "clean\n"),
                           (os.path.join(repo, "tests", "golden", "g.json"), '{"a": 1}\n'),
                           (os.path.join(repo, ".gitignore"), "tmp/\n")):
            with open(path, "w") as f:
                f.write(text)
        git = throwaway_repo(repo)
        git("add", ".")
        git("commit", "-q", "-m", "without the driver")
        without = git("rev-parse", "HEAD")
        with open(os.path.join(sab, "engine_fake.py"), "w") as f:
            f.write(RUNNER_DRIVER)
        git("add", ".")
        git("commit", "-q", "-m", "with the driver")
        asked = git("rev-parse", "HEAD")
        git("checkout", "-q", "--detach", without)
        done = subprocess.run([sys.executable, "-B", os.path.join(sab, "parallel.py"), "--ref", asked, "--jobs", "1",
                               "--early-exit", "engine_fake"], cwd=repo, capture_output=True, text=True, timeout=120,
                              env=dict(os.environ, SABOTAGE_INHIBITED="1", RUNNER_TEST_LOG=log))
        out = (done.stdout + done.stderr)[-800:]
        assert f"judging {asked[:7]}" in done.stdout, f"the run did not judge the commit asked for:\n{out}"
        logs = glob.glob(os.path.join(repo, "tmp", "sabotage", "parallel-*", "engine_fake.log"))
        assert len(logs) == 1, f"no log of the driver the asked commit has:\n{out}"
        with open(logs[0]) as f:
            judged = f.read()
        assert "  red    caught by the seventh suite alone" in judged, f"the seventh suite never ran:\n{judged}"
        assert "  GREEN  survives every suite   <-- SURVIVED" in judged, judged
        assert done.returncode != 0, f"a run with a survivor exited 0:\n{out}"
        with open(log) as f:
            states = f.read().split()
    assert states and set(states) == {"held"}, f"the workers' lock was free while suites ran: {states}"

test("a run judges the commit asked for, with every suite, under its lock, and fails on a survivor",
     check_a_run_judges_the_asked_commit_with_every_suite_under_its_lock)


def check_a_shell_driver_runs_from_the_commit_under_test():
    """The shell driver is run from the commit under test's tree, as the Python ones
    are read from it. The check pointed it at the runner's own checkout (R16), and a
    shell driver only the commit under test has would never have run."""
    import tempfile
    with tempfile.TemporaryDirectory() as tree:
        os.makedirs(os.path.join(tree, "tests", "sabotage"))
        with open(os.path.join(tree, "tests", "sabotage", "engine_fake_shell.sh"), "w") as f:
            f.write('echo "the commit under test\'s own shell driver"\n')
        got = runner().judge_shell(tree, "engine_fake_shell")
    assert got["rc"] == 0 and "the commit under test's own shell driver" in got["output"], got

test("the shell driver runs from the commit under test, not the runner's checkout",
     check_a_shell_driver_runs_from_the_commit_under_test)


def check_compare_reads_the_suites_under_each_status_and_every_driver():
    """--compare holds one run to another sabotage by sabotage, and a status alone is
    not enough: the same "red" can come from another suite or another failure, which is
    how 90 catches made by a stale .pyc once matched on status. And a driver judged in
    one run only is a difference, not a skip. The check made it compare statuses alone
    (R19) and pass a one-sided driver (R20), and no test noticed."""
    import contextlib, io, tempfile
    block = lambda failure: f"1 sabotage(s)\n  red    a\n         model[1 failed: {failure}]\n\n1 caught, 0 survived\n"
    with tempfile.TemporaryDirectory() as d:
        first, second = os.path.join(d, "first"), os.path.join(d, "second")
        os.makedirs(first)
        os.makedirs(second)
        for where, name, text in ((first, "engine_x", block("FAIL the real check")),
                                  (second, "engine_x", block("FAIL a stale .pyc")),
                                  (first, "engine_only_first", block("FAIL the real check"))):
            with open(os.path.join(where, name + ".log"), "w") as f:
                f.write(text)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            rc = runner().compare(first, second)
    said = printed.getvalue()
    assert rc == 1, said
    assert "engine_x: a:" in said, f"the same status from another failure was called the same:\n{said}"
    assert "engine_only_first: only in the first run" in said, f"a driver judged in one run only was passed:\n{said}"

test("--compare reads the suites under each status, and every driver on both sides",
     check_compare_reads_the_suites_under_each_status_and_every_driver)


def check_the_golden_failures_come_from_every_golden():
    """The FAIL lines that only a golden makes are found by emptying every golden file
    at once and running the suites, and every file must be emptied: the check emptied
    the first alone (R30), and a catch by the second golden would have been reported
    as a real one. Each file is restored afterwards."""
    import tempfile
    with tempfile.TemporaryDirectory() as tree:
        gdir = os.path.join(tree, "tests", "golden")
        os.makedirs(gdir)
        for name in ("a.json", "b.json"):
            with open(os.path.join(gdir, name), "w") as f:
                f.write('{"kept": 1}\n')
        git = throwaway_repo(tree)
        git("add", ".")
        git("commit", "-q", "-m", "goldens")

        def emptied(name):
            with open(os.path.join(gdir, name)) as f:
                return f.read().strip() == "{}"

        harness = types.SimpleNamespace(run_judging_suites=lambda: {
            "model": (1, ["FAIL golden a"], [], "") if emptied("a.json") else (0, [], [], ""),
            "report": (1, ["FAIL golden b"], [], "") if emptied("b.json") else (0, [], [], "")})
        got = runner().golden_failures(harness, tree)
        restored = [not emptied(n) for n in ("a.json", "b.json")]
    assert got == {"FAIL golden a", "FAIL golden b"}, f"golden failures found: {got}"
    assert all(restored), "a golden file was left emptied"

test("the golden failures come from every golden, each restored after",
     check_the_golden_failures_come_from_every_golden)


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
