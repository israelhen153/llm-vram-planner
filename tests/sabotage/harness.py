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
