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
    base = base or ENGINE
    changed = {f: text for f, text in base.items() if text != ENGINE[f]}
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


def shell_driver_problems(catalog, before=None, src=None):
    """What would stop engine_r1_perfkey_typo.sh applying to this text of
    data/gpus.json, read from the script: each catalog row its loop names has to
    carry the perfKey it corrupts exactly once. And, given the catalog before a
    refresh, every string the script counts in a row, in any assert, has to be
    counted there as often after it as before: a quoted price or date stops the
    driver applying after the job's next run. Its cold check added a second assert
    quoting a price, and only the first was read."""
    src = src if src is not None else open(os.path.join(SABOTAGE_DIR, SHELL_DRIVERS[0]), encoding="utf-8").read()
    loop = re.search(r"^for slug in ([\w .-]+); do$", src, re.M)
    needs = re.search(r"assert line\.count\('([^']+)'\) == 1", src)
    if not (loop and needs):
        return [f"{SHELL_DRIVERS[0]} no longer has the loop and the precondition this reads"]
    counted = re.findall(r"line\.count\('([^']+)'\)", src)

    def row(text, slug):
        return [r for r in text.splitlines() if r.strip().startswith(f'"{slug}":')]

    problems = []
    for slug in loop.group(1).split():
        line = row(catalog, slug)
        if not (len(line) == 1 and line[0].count(needs.group(1)) == 1):
            problems.append(f"{SHELL_DRIVERS[0]}: catalog row {slug} does not carry {needs.group(1)} exactly once")
            continue
        was = row(before, slug) if before is not None else []
        for quote in counted if len(was) == 1 else []:
            if was[0].count(quote) != line[0].count(quote):
                problems.append(f"{SHELL_DRIVERS[0]}: catalog row {slug} carries {quote!r} "
                                f"{was[0].count(quote)} time(s) before the refresh and {line[0].count(quote)} after")
    return problems


def check_the_shell_driver_still_applies():
    """engine_r1_perfkey_typo.sh edits data/gpus.json from a heredoc, so it cannot be
    read as data like the others; shell_driver_problems() reads what it needs."""
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


def reread_sku(sku, spec):
    """The SKU the next reading of a tier records. What SOURCE_MAP asks the source for
    (an instance type, a plan, a GPU name) comes back as it was. A number outside all
    of it was counted from the reading itself, as Vast.ai's "median of 43 verified
    offers" is, and is a different number next time."""
    asked = [m.span() for v in spec.values() if isinstance(v, str) for m in re.finditer(re.escape(v), sku)]
    return re.sub(r"\d+", lambda m: m.group() if any(a <= m.start() and m.end() <= b for a, b in asked)
                  else str(int(m.group()) + 1), sku)


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
    base = base or ENGINE
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
    return files, (lambda missing: True) if kind == "gone" else replaced_by_a_reading(catalog)


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
    this check writes nothing to disk. Drivers already imported are forgotten first, so
    one that another driver imports reads the refreshed files too."""
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
    builtins.open = refreshed_open
    try:
        yield
    finally:
        builtins.open = real
        forget_drivers()


def check_after(kind, excused, base=None):
    """Every sabotage in every driver, loaded against the refreshed files, still applies
    to them, and no driver has fewer than it has on the tree as it is: a sabotage has to
    be refused by name, never dropped. A refusal the refreshed tree has to cause (see
    refresh()) goes in excused instead. base is the tree the refresh starts from, as
    refresh() takes it."""
    base = base or ENGINE
    try:
        files, taken = refresh(kind, base)
    except SystemExit as e:   # the job's writers stop, rather than write a catalog they cannot
        raise AssertionError(f"the price job's own writers refuse this catalog: {e}") from None
    if kind != "confirmed" or tiers_of(json.loads(base["data/gpus.json"])["data"])[1]:
        assert files["data/gpus.json"] != base["data/gpus.json"], "the refresh rewrote nothing, so this checks nothing"
    tree = {**base, **files}
    problems, loaded = [], {}
    with reading_from(files):
        for d in PY_DRIVERS:
            try:
                loaded[d] = sabotages_of(d)
            except Exception as e:
                problems.append(f"{d} no longer loads: {type(e).__name__}: {e}")
    for d, sabotages in loaded.items():
        if len(sabotages) < len(PY_DRIVERS[d]):
            problems.append(f"{d}: {len(PY_DRIVERS[d])} sabotages before the refresh and {len(sabotages)} "
                            "after it, so one dropped out where it had to be refused by name")
        for name, spec in sabotages.items():
            why = why_it_would_not_apply(spec, tree)
            if why and why_it_would_not_apply(spec, tree, taken) is None:
                excused.append(f"{d}: {name}: refused by name, as it has to be: {why}")
            elif why:
                problems.append(f"{d}: {name}: {why_it_would_not_apply(spec, tree, taken)}")
    problems += [f"tests/sabotage/anchors.py: {s}" for s in stale_excerpts(tree)]
    problems += shell_driver_problems(files["data/gpus.json"], before=base["data/gpus.json"])
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
        return {**ENGINE, **refresh("confirmed", on="2099-12-30")[0]}

    def refused_on_its_notes(base, refused):
        """That the tree a check read is base: every sabotage built on a note the pull
        request replaced is among those it refused by name, and each would apply to the
        tree as it is."""
        import harness
        with reading_from({f: base[f] for f in ("data/gpus.json", "index.html", "generate_report.py")}):
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
        tree = {**ENGINE, **files}
        for slug, tier in tiers:
            quote = row_through(ENGINE["data/gpus.json"], slug, f'"{tier}": {json.dumps(ROWS[slug][tier])}')
            spec = [("data/gpus.json", quote, quote + "0", 1)]
            assert why_it_would_not_apply(spec) is None, f"{slug}/{tier}: the quote does not apply before the refresh"
            if why_it_would_not_apply(spec, tree) is None:
                survived.append(f"{kind}: {slug}/{tier}: ...{quote[-32:]!r}")
    assert not survived, f"a quoted price survived the refresh that moves it: {survived}"

test("a price quoted with nothing after it stops applying after the refresh that moves it",
     check_the_refresh_check_sees_a_quoted_price)


def check_the_refresh_check_reads_every_shell_quote():
    """A second assert in the shell driver, quoting a price one of its rows carries, has
    to count against it after the refresh that moves that price: only the first assert
    was read. The driver's own script, with that assert added before its first."""
    src = open(os.path.join(SABOTAGE_DIR, SHELL_DRIVERS[0]), encoding="utf-8").read()
    loop = re.search(r"^for slug in ([\w .-]+); do$", src, re.M).group(1).split()
    slug, tier = next((s, t) for s, t in READ if s in loop)
    quote = f'"{tier}": {json.dumps(ROWS[slug][tier])},'
    added = src.replace("assert line.count(", f"assert line.count('{quote}') <= 1\nassert line.count(", 1)
    files, _ = refresh("moved")
    assert shell_driver_problems(ENGINE["data/gpus.json"], before=ENGINE["data/gpus.json"], src=added) == [], (
        "the added assert does not hold before the refresh")
    assert shell_driver_problems(files["data/gpus.json"], before=ENGINE["data/gpus.json"], src=added), (
        f"the shell driver counts {quote!r} in {slug}'s row, the refresh moved it, and nothing noticed")

test("a price the shell driver counts in any assert stops it applying after the refresh",
     check_the_refresh_check_reads_every_shell_quote)


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
    tree, excused = {**ENGINE, **files}, replaced_by_a_reading(files["data/gpus.json"])
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


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
