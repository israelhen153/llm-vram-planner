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


def why_it_would_not_apply(spec, tree=None, confirmed=()):
    """What harness.apply_edits() would refuse this sabotage for on the tree as it
    is, or None. The same check, harness.refusal(), edit by edit and in order, on
    copies: a count that does not match, or a target its driver found gone.

    tree holds files as another state of the repository would have them, by path;
    a file it does not hold is read from disk. confirmed names the (slug, tier) of
    each note that state no longer has because a reading was confirmed there. A
    sabotage built on one of those notes is refused by name, as it has to be, and
    that refusal is not counted against it."""
    import harness
    files = {}
    for f, old, new, count in harness.edits_of(spec):
        if isinstance(old, harness.Missing) and old.note in confirmed:
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


def shell_driver_problems(catalog):
    """What would stop engine_r1_perfkey_typo.sh applying to this text of
    data/gpus.json, read from the script: each catalog row its loop names has to
    carry the perfKey it corrupts exactly once."""
    src = open(os.path.join(SABOTAGE_DIR, SHELL_DRIVERS[0]), encoding="utf-8").read()
    loop = re.search(r"^for slug in ([\w .-]+); do$", src, re.M)
    needs = re.search(r"assert line\.count\('([^']+)'\) == 1", src)
    if not (loop and needs):
        return [f"{SHELL_DRIVERS[0]} no longer has the loop and the precondition this reads"]
    rows = catalog.splitlines()
    problems = []
    for slug in loop.group(1).split():
        line = [r for r in rows if r.strip().startswith(f'"{slug}":')]
        if not (len(line) == 1 and line[0].count(needs.group(1)) == 1):
            problems.append(f"{SHELL_DRIVERS[0]}: catalog row {slug} does not carry {needs.group(1)} exactly once")
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
ROWS = json.loads(ENGINE["data/gpus.json"])["data"]
READ = [(s, t) for s, t, _ in AUTOMATED if t in (ROWS.get(s, {}).get("priceSource") or {})]
HELD = [(s, t) for s, t, _ in AUTOMATED if (s, t) not in READ
        and t in (ROWS.get(s, {}).get("priceNote") or {})]


def reread_sku(sku, spec):
    """The SKU the next reading of a tier records. What SOURCE_MAP asks the source for
    (an instance type, a plan, a GPU name) comes back as it was. A number outside all
    of it was counted from the reading itself, as Vast.ai's "median of 43 verified
    offers" is, and is a different number next time."""
    asked = [m.span() for v in spec.values() if isinstance(v, str) for m in re.finditer(re.escape(v), sku)]
    return re.sub(r"\d+", lambda m: m.group() if any(a <= m.start() and m.end() <= b for a, b in asked)
                  else str(int(m.group()) + 1), sku)


def refresh(kind):
    """The three files the job rewrites, as one kind of run leaves them, in memory, and
    the (slug, tier) of each note that run confirmed away.

    reread     every automated tier with a reading is read again, on a new day, at the
               price it recorded
    moved      the same, with every one of those prices a cent higher, the catalog's too
    confirmed  every automated tier still held under a note gets its first reading, at
               the catalog's price, which takes the note out"""
    data = json.loads(ENGINE["data/gpus.json"])["data"]
    outcomes, confirmed = [], set()
    for slug, tier, spec in AUTOMATED:
        row = data[slug]
        source = (row.get("priceSource") or {}).get(tier)
        if source and kind in ("reread", "moved"):
            price = row[tier] if kind == "reread" else round(row[tier] + 0.01, 2)
            reading = price_check.Reading(
                provider=source["provider"], sku=reread_sku(source["sku"], spec), region=source["region"],
                price_per_gpu=source["price"] if kind == "reread" else price, date=READ_ON, evidence="")
            outcomes.append(price_check.Outcome(slug, tier, "CONFIRMED" if kind == "reread" else "MOVED",
                                                current=row[tier], proposed=price, reading=reading))
        elif not source and kind == "confirmed" and tier in (row.get("priceNote") or {}):
            reading = price_check.Reading(
                provider=spec["kind"], sku="(simulated)", region=spec.get("region", "global"),
                price_per_gpu=row[tier], date=READ_ON, evidence="")
            outcomes.append(price_check.Outcome(slug, tier, "CONFIRMED", current=row[tier],
                                                proposed=row[tier], reading=reading))
            confirmed.add((slug, tier))
    price_check.apply_outcomes(data, outcomes)
    catalog = price_check.apply_to_text(ENGINE["data/gpus.json"], data,
                                        sorted({oc.slug for oc in outcomes}), READ_ON)
    files = {"data/gpus.json": catalog}
    rows = json.loads(catalog)["data"]   # what tools/sync_data.py reads: the file the job wrote
    for f, render in (("index.html", sync_data.render_gpu_js), ("generate_report.py", sync_data.render_gpu_py)):
        files[f] = sync_data.BLOCK_RES["GPU_TABLE"].sub(
            lambda m, render=render: render(rows).rstrip("\n"), ENGINE[f], count=1)
    return files, confirmed


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


def check_after(kind, excused):
    """Every sabotage in every driver, loaded against the refreshed files, still applies
    to them, and no driver has fewer than it had: a sabotage has to be refused by name,
    never dropped. A refusal the refresh has to cause, a sabotage built on a note it
    confirmed away, goes in excused instead."""
    try:
        files, confirmed = refresh(kind)
    except SystemExit as e:   # the job's writers stop, rather than write a catalog they cannot
        raise AssertionError(f"the price job's own writers refuse this catalog: {e}") from None
    if kind != "confirmed" or HELD:
        assert files["data/gpus.json"] != ENGINE["data/gpus.json"], "the refresh rewrote nothing, so this checks nothing"
    tree = {**ENGINE, **files}
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
            if why and why_it_would_not_apply(spec, tree, confirmed) is None:
                excused.append(f"{d}: {name}: refused by name, as it has to be: {why}")
            elif why:
                problems.append(f"{d}: {name}: {why_it_would_not_apply(spec, tree, confirmed)}")
    problems += [f"tests/sabotage/anchors.py: {s}" for s in stale_excerpts(tree)]
    problems += shell_driver_problems(files["data/gpus.json"])
    assert not problems, (
        f"{len(problems)} thing(s) the job's next run would break:\n         " + "\n         ".join(problems) +
        "\n       A price, a reading's date, price or offer count, and a note a confirmed reading replaces "
        "are all rewritten by the price job. Read them from data/gpus.json with open() as the driver "
        "loads, and quote only what the job never rewrites: see tests/sabotage/README.md.")


REFRESHES = [
    ("reread", f"re-reads the {len(READ)} automated tiers that carry a reading, each on a new day"),
    ("moved", f"moves each of those {len(READ)} prices a cent"),
    ("confirmed", "confirms a reading on every automated tier still held under a note ("
                  + (", ".join(f"{s}/{t}" for s, t in HELD) or "none today") + ")"),
]
for kind, label in REFRESHES:
    excused = []
    test(f"after a refresh that {label}, every sabotage in every driver still applies",
         lambda kind=kind, excused=excused: check_after(kind, excused))
    for line in excused:
        print(f"       {line}")


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
