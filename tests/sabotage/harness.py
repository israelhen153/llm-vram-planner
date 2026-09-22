"""The machinery every sabotage driver shares.

A sabotage is a list of exact-string edits to committed files. The harness
applies them — refusing if a target string is not found the expected number of
times, so a driver that has drifted from the code fails loudly instead of
silently testing nothing — runs the suites that can judge it, and restores every
touched file with `git checkout --`.

Not a driver, and not run directly. It was the top of sab.py until 2026-09-22,
which made that one file both the machinery and the biggest driver.
"""
import os, re, subprocess, sys

# No __pycache__. The drivers import each other, so a run used to leave one behind;
# it is gitignored, which means git cannot remove the directory on a branch switch,
# which leaves an unlisted tests/sabotage behind and turns the project-structure
# guard red on a branch that has nothing to do with this. It has cost two people a
# confusing red suite, so it stops being created.
sys.dont_write_bytecode = True
# The repo root, however deep this file is filed. Asking git rather than counting
# "..", because counting is what breaks silently when this directory moves: the
# sabotages would then be applied to whatever tree the wrong path happens to name.
ROOT = subprocess.check_output(
    ['git', 'rev-parse', '--show-toplevel'],
    cwd=os.path.dirname(os.path.abspath(__file__)), text=True).strip()
os.chdir(ROOT)

SUITES = [("model", ["node", "tests/model.test.js"]),
          ("parity", ["python3", "tests/parity.test.py"]),
          ("report", ["python3", "tests/report.test.py"]),
          ("sync", ["python3", "tests/sync.test.py"]),
          ("price", ["python3", "tests/price_check.test.py"]),
          ("workflow", ["python3", "tests/workflow.test.py"])]


def run_suites():
    res = {}
    for name, cmd in SUITES:
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = p.stdout + p.stderr
        fails = [l.strip() for l in out.splitlines() if l.lstrip().startswith("FAIL")]
        errs = [l.strip() for l in out.splitlines() if re.search(r"Error|Traceback|node runner failed", l)]
        tally = re.findall(r"(\d+) passed, (\d+) failed", out)
        res[name] = (p.returncode, fails, errs, tally[-1] if tally else None)
    return res


def require_green_baseline():
    """A sabotage is judged by a suite going red. If a suite is ALREADY red for an
    unrelated reason — a flaky test, a missing dependency, an unrelated regression in
    the same commit — then every sabotage in the run reads as caught, with no warning
    and no way to tell it from a genuine catch. The run is worthless and looks perfect.

    So prove the tree judges green before judging anything against it."""
    red = [k for k, v in run_suites().items() if v[0] != 0]
    if red:
        sys.exit(f"refusing to judge: {', '.join(red)} already red on the unmodified "
                 f"tree, so every sabotage would read as caught")


def apply(edits):
    """edits: list of (file, old, new, count). Returns the files touched."""
    touched = []
    for f, old, new, count in edits:
        path = os.path.join(ROOT, f)
        src = open(path).read()
        n = src.count(old)
        if n != count:
            raise RuntimeError(f"{f}: expected {count} occurrence(s) of {old[:70]!r}, found {n}")
        open(path, "w").write(src.replace(old, new))
        touched.append(f)
    return touched


def restore(files):
    # Every file a suite can rewrite, not only the ones edited: tests/sync.test.py
    # runs the real sync against the real engines, so a sabotaged tools/sync_data.py
    # regenerates both GPU_TABLE blocks under the test.
    subprocess.run(["git", "checkout", "--", "index.html", "generate_report.py",
                    "tools/sync_data.py", "data/gpus.json", *sorted(set(files))], check=True)
    st = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    assert st.strip() == "", f"tree not clean after restore:\n{st}"


def run_driver(sabotages):
    """Run a driver's sabotages one at a time and report which the suite missed.

        python3 <driver> [name-substring ...]    only the sabotages matching one
        python3 <driver> --from <name-prefix>    from that sabotage onwards

    Every driver used to carry its own copy of this loop — fifteen copies in six
    different shapes, each commented "same shape as every other driver, on
    purpose". Some filtered by name and five could not; some matched
    case-insensitively; one supported --from; each restored its own hand-written
    list of files when a sabotage failed to apply. The same corpus behaved
    differently depending on which file you ran. One loop now.

    A sabotage that cannot be applied is NOT a catch: the run exits 2, so a
    drifted driver cannot report a clean run forever. And the tree must judge
    green before anything is judged against it — see require_green_baseline.
    """
    # Line-buffered, so a log chain.sh is writing shows progress as it happens.
    # Redirected to a file, Python buffers by the block, and the first ten minutes
    # of a twenty-minute run read as an empty log — which looks exactly like a hang.
    sys.stdout.reconfigure(line_buffering=True)
    # Every file this driver could touch, derived from its own sabotages rather
    # than listed by hand: the hand-written lists were one of the six drifts.
    touchable = sorted({edit[0] for edits in sabotages.values() for edit in edits})
    patterns = sys.argv[1:]
    if patterns and patterns[0] == "--from":
        keys = list(sabotages)
        names = keys[keys.index(next(k for k in keys if k.startswith(patterns[1]))):]
    else:
        names = [n for n in sabotages
                 if not patterns or any(p.lower() in n.lower() for p in patterns)]
    print(f"{len(names)} sabotage(s)")
    require_green_baseline()
    survived, unapplied = [], []
    for name in names:
        try:
            touched = apply(sabotages[name])
        except Exception as e:
            print(f"  !! {name}: could not apply: {e}")
            unapplied.append(name)
            restore(touchable)
            continue
        try:
            results = run_suites()
        finally:
            restore(touched)
        red = {suite: r for suite, r in results.items() if r[0] != 0}
        if not red:
            survived.append(name)
            print(f"  GREEN  {name}   <-- SURVIVED")
        else:
            lines = []
            for suite, (rc, fails, errs, tally) in red.items():
                first = (fails or errs or ["(no FAIL line)"])[0]
                lines.append(f"{suite}[{tally[1] if tally else '?'} failed: {first[:120]}]")
            print(f"  red    {name}\n         " + "\n         ".join(lines))
    print(f"\n{len(names) - len(survived) - len(unapplied)} caught, "
          f"{len(survived)} survived"
          + (f", {len(unapplied)} COULD NOT BE APPLIED" if unapplied else ""))
    for name in survived:
        print("  SURVIVED: " + name)
    if unapplied:
        sys.exit(2)
