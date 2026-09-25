#!/usr/bin/env python3
"""Run the sabotage corpus in parallel, one git worktree per worker.

chain.sh runs every driver back to back in the one checkout. On 2026-09-24 that
was 567 sabotages at about 13.5 s each: 2 h 12 min of running, with the tree held
the whole time. This splits the same work sabotage by sabotage across worker
worktrees, each detached at the commit under test, and judges every sabotage with
that commit's own harness: its run_driver(), its six suites. What changes is where
and when a sabotage runs, not how it is judged, which is what lets a parallel run
be checked against a serial one, sabotage by sabotage (--compare).

    tests/sabotage/parallel.py [--jobs N] [--ref REF] [driver ...]
    tests/sabotage/parallel.py --compare SERIAL_LOGDIR PARALLEL_LOGDIR

--ref is the commit to judge (default: HEAD of the checkout this runs from), so the
runner on one branch can judge another branch's corpus without either merging.
Uncommitted work is never judged, and never touched: the workers check out the
commit. Drivers are named as chain.sh takes them; the default is every engine_*
and workflow_* driver in the commit under test, found the way chain.sh finds them.

Logs go to LOGDIR (default: tmp/sabotage/parallel-<sha>-<UTC time>/ in the main
checkout; a relative LOGDIR is taken from there too), one per driver in chain.sh's
format, plus results.json with every sabotage's result, worker and time.

The workers are tmp/corpus-workers/w1..wN in the main checkout, reused from run to
run. This script never deletes one: a worker that is dirty, locked, missing or not
a worktree stops the run before anything is touched, and is listed. Removing the
workers is the owner's call; see the sabotage-corpus skill.
"""
import argparse
import fnmatch
import importlib.util
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
# chain.sh's own globs, so the two runners discover the same drivers.
PATTERNS = ["engine_*.py", "engine_*.sh", "workflow_*.py", "workflow_*.sh"]
WORKERS_DIR = os.path.join("tmp", "corpus-workers")

# The lines run_driver() and the shell driver print for one sabotage. A block is a
# start line and the 9-space lines under it.
START = [("caught", re.compile(r"^  red    (.+)$")),
         ("survived", re.compile(r"^  GREEN  (.+?)   <-- SURVIVED$")),
         ("unapplied", re.compile(r"^  !! (.+?): could not apply: .*$")),
         ("unapplied", re.compile(r"^  ERROR  (.+) \(could not apply\)$"))]
CONTINUATION = " " * 9
SUMMARY = re.compile(r"(\d+) caught, (\d+) survived(?:, (\d+) COULD NOT BE APPLIED)?")


def git(*args, cwd, check=True):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check)


def natural(s):
    """sort -V, near enough for driver names: r2 before r10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def blocks_of(text):
    """{name: [status, start line, continuation lines]} for every sabotage in a log."""
    blocks, cur = {}, None
    for line in text.splitlines():
        for status, rx in START:
            m = rx.match(line)
            if m:
                name = m.group(1)
                if name in blocks:
                    raise ValueError(f"sabotage {name!r} appears twice")
                cur = blocks[name] = [status, line, []]
                break
        else:
            if cur is not None and line.startswith(CONTINUATION) and line.strip():
                cur[2].append(line)
            else:
                cur = None
    return blocks


# ---------------------------------------------------------------- worker side

def load_harness(tree):
    # Replace, not prepend: sys.path[0] is this script's own directory, which holds
    # the runner branch's harness.py. The commit under test is judged by its own.
    sys.path[0:1] = [os.path.join(tree, "tests", "sabotage")]
    import harness
    if os.path.realpath(harness.ROOT) != os.path.realpath(tree):
        raise RuntimeError(f"imported the harness of {harness.ROOT}, not of {tree}")
    return harness


def sabotages_of(harness, tree, driver):
    """A driver's sabotages, read as data, the way tests/corpus.test.py reads them:
    every driver builds S at import and hands it to run_driver(), which does
    nothing while this runs."""
    real, harness.run_driver = harness.run_driver, (lambda sabotages: None)
    try:
        spec = importlib.util.spec_from_file_location(
            driver, os.path.join(tree, "tests", "sabotage", driver + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        harness.run_driver = real
    sabotages = getattr(module, "S", None)
    if not isinstance(sabotages, dict) or not sabotages:
        raise RuntimeError(f"{driver}: no sabotages found in S")
    return sabotages


def judge_one(harness, driver, name, spec):
    """One sabotage through the commit's own run_driver(): applied, re-synced if
    it says so, judged by its six suites, restored. Only the green baseline is
    skipped, because this worker proved it once before its first sabotage."""
    buf = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", newline="\n")
    saved = sys.stdout, sys.argv
    sys.stdout, sys.argv = buf, [driver]
    code = 0
    try:
        harness.run_driver({name: spec})
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    finally:
        sys.stdout, sys.argv = saved
    buf.flush()
    text = buf.buffer.getvalue().decode("utf-8")
    summary = SUMMARY.search(text)
    blocks = blocks_of(text)
    if not summary or list(blocks) != [name]:
        raise RuntimeError(f"run_driver printed no result for {name!r}:\n{text[-800:]}")
    caught, survived, unapplied = (int(g or 0) for g in summary.groups())
    status = blocks[name][0]
    expected = {"caught": (1, 0, 0), "survived": (0, 1, 0), "unapplied": (0, 0, 1)}[status]
    if (caught, survived, unapplied) != expected or (code == 2) != (status == "unapplied"):
        raise RuntimeError(f"{name!r}: its line says {status}, its summary says "
                           f"{summary.group(0)!r}, exit {code}")
    return {"status": status, "block": [blocks[name][1], *blocks[name][2]]}


def judge_shell(tree, driver):
    """The one shell driver runs whole, as chain.sh runs it."""
    p = subprocess.run(["bash", os.path.join(tree, "tests", "sabotage", driver + ".sh")],
                       cwd=tree, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    return {"status": "shell", "rc": p.returncode, "output": p.stdout}


def worker_main(tree):
    # The task stream and the results travel on this process's own stdin and stdout.
    # Children inherit fd 0 and fd 1, so both are moved out of their reach first: a
    # suite reading stdin would eat tasks, and one printing would corrupt results.
    tasks_in = os.fdopen(os.dup(0), "r", encoding="utf-8")
    results_out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(1, "w", encoding="utf-8", closefd=False)

    def send(obj):
        results_out.write(json.dumps(obj) + "\n")
        results_out.flush()

    started = time.monotonic()
    try:
        harness = load_harness(tree)
        harness.require_green_baseline()
    except SystemExit as e:
        send({"ready": False, "error": str(e.code)})
        return 2
    except Exception:
        send({"ready": False, "error": traceback.format_exc()})
        return 2
    harness.require_green_baseline = lambda: None
    send({"ready": True, "baseline_s": round(time.monotonic() - started, 2)})

    cache = {}
    for line in tasks_in:
        task = json.loads(line)
        t0 = time.monotonic()
        try:
            if task["kind"] == "sh":
                result = judge_shell(tree, task["driver"])
            else:
                if task["driver"] not in cache:
                    cache[task["driver"]] = sabotages_of(harness, tree, task["driver"])
                sabotages = cache[task["driver"]]
                result = judge_one(harness, task["driver"], task["name"], sabotages[task["name"]])
        except Exception:
            result = {"status": "error", "error": traceback.format_exc()}
        # chain.sh checks between drivers; this checks between sabotages. A worker
        # left dirty is restored the way chain.sh restores, and if that is not
        # enough it stops taking work rather than judge on a changed tree.
        dirty = git("status", "--porcelain", cwd=tree).stdout
        if dirty.strip():
            git("checkout", "--", ".", cwd=tree, check=False)
            still = git("status", "--porcelain", cwd=tree).stdout
            result = {"status": "error", "error": f"left the worker dirty:\n{dirty}",
                      "fatal": bool(still.strip())}
        result["seconds"] = round(time.monotonic() - t0, 2)
        send(result)
        if result.get("fatal"):
            return 1
    return 0


def list_main(tree, drivers):
    # Anything a driver prints while it loads goes to stderr, so stdout carries only the JSON.
    out, sys.stdout = sys.stdout, sys.stderr
    try:
        harness = load_harness(tree)
        listing = {d: list(sabotages_of(harness, tree, d)) for d in drivers}
    finally:
        sys.stdout = out
    print(json.dumps(listing))
    return 0


# ---------------------------------------------------------- orchestrator side

def main_checkout():
    common = git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=HERE).stdout.strip()
    if os.path.basename(common) != ".git":
        sys.exit(f"cannot place the workers: git's common dir is {common}, not a checkout's .git")
    return os.path.dirname(common)


def inhibit_sleep():
    """A laptop that suspends mid-run stretches it without a word: a driver that
    takes 3.6 min took 86 on 2026-09-24. Hold a sleep lock for the whole run where
    systemd provides one, by re-running this under systemd-inhibit. Closing the lid
    still suspends: logind's default (LidSwitchIgnoreInhibited=yes) lets the lid
    ignore sleep locks."""
    if os.environ.get("SABOTAGE_INHIBITED") == "1" or not shutil.which("systemd-inhibit"):
        return
    probe = subprocess.run(["systemd-inhibit", "--what=sleep:idle", "--who=sabotage-corpus",
                            "--why=probe", "true"], capture_output=True, text=True)
    if probe.returncode != 0:
        print(f"warning: could not block sleep ({probe.stderr.strip()[:160]}); "
              f"a suspend will stretch this run", file=sys.stderr)
        return
    os.environ["SABOTAGE_INHIBITED"] = "1"
    os.execvp("systemd-inhibit", ["systemd-inhibit", "--what=sleep:idle",
                                  "--who=sabotage-corpus", "--why=sabotage corpus",
                                  "--mode=block", sys.executable, os.path.abspath(__file__),
                                  *sys.argv[1:]])


def registered_worktrees(root):
    trees, cur = {}, None
    for line in git("worktree", "list", "--porcelain", cwd=root).stdout.splitlines():
        if line.startswith("worktree "):
            cur = os.path.realpath(line[len("worktree "):])
            trees[cur] = set()
        elif cur and line.split(" ")[0] in ("locked", "prunable"):
            trees[cur].add(line.split(" ")[0])
    return trees


def prepare_workers(root, sha, jobs):
    """Reuse clean workers, create missing ones, and refuse on anything else, all
    before any worker is touched."""
    base = os.path.join(root, WORKERS_DIR)
    trees = registered_worktrees(root)
    paths, problems = [], []
    for i in range(1, jobs + 1):
        path = os.path.realpath(os.path.join(base, f"w{i}"))
        flags, exists = trees.get(path), os.path.isdir(path)
        if flags is not None and "locked" in flags:
            problems.append(f"{path}: locked")
        elif flags is not None and not exists:
            problems.append(f"{path}: registered as a worktree but missing (`git worktree prune` clears it)")
        elif flags is None and exists:
            problems.append(f"{path}: exists but is not a worktree")
        elif flags is not None:
            dirty = git("status", "--porcelain", cwd=path).stdout
            if dirty.strip():
                problems.append(f"{path}: not clean\n" + "".join("    " + l + "\n" for l in dirty.splitlines()))
        paths.append((path, flags is not None))
    if problems:
        print("refusing to run: these workers need the owner's decision first; nothing was touched",
              file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        sys.exit(2)
    os.makedirs(base, exist_ok=True)
    for path, registered in paths:
        if registered:
            git("checkout", "--quiet", "--detach", sha, cwd=path)
        else:
            git("worktree", "add", "--quiet", "--detach", path, sha, cwd=root)
    return [p for p, _ in paths]


def discover(tree, named):
    d = os.path.join(tree, "tests", "sabotage")
    found = {os.path.splitext(f)[0] for f in os.listdir(d)
             if any(fnmatch.fnmatch(f, p) for p in PATTERNS)}
    for n in sorted(found):
        if os.path.exists(os.path.join(d, n + ".py")) and os.path.exists(os.path.join(d, n + ".sh")):
            sys.exit(f"refusing to run: {n}.py and {n}.sh both exist — one would never execute")
    if named:
        unknown = [n for n in named if n not in found]
        if unknown:
            sys.exit(f"no such driver in the commit under test: {', '.join(unknown)}")
        found = set(named)
    return [(n, "py" if os.path.exists(os.path.join(d, n + ".py")) else "sh")
            for n in sorted(found, key=natural)]


def driver_log(driver, names, results):
    """One driver's log, in the shape its own run under chain.sh writes."""
    lines = [f"{len(names)} sabotage(s)"]
    caught = survived = unapplied = errors = 0
    survivors = []
    for name in names:
        r = results.get((driver, name), {"status": "error", "error": "never run"})
        if r["status"] == "error":
            errors += 1
            lines.append(f"  ERROR  {name}: " + (r["error"].strip().splitlines() or ["?"])[-1][:200])
            continue
        lines.extend(r["block"])
        caught += r["status"] == "caught"
        survived += r["status"] == "survived"
        unapplied += r["status"] == "unapplied"
        if r["status"] == "survived":
            survivors.append(name)
    lines.append("")
    lines.append(f"{caught} caught, {survived} survived"
                 + (f", {unapplied} COULD NOT BE APPLIED" if unapplied else "")
                 + (f", {errors} ERRORED" if errors else ""))
    lines.extend("  SURVIVED: " + n for n in survivors)
    rc = 1 if errors else 2 if unapplied else 0
    lines.append(f"driver exit {rc}")
    return "\n".join(lines) + "\n", rc, caught, survived, unapplied, errors


def run(args):
    inhibit_sleep()
    root = main_checkout()
    sha = git("rev-parse", "--verify", f"{args.ref}^{{commit}}", cwd=HERE).stdout.strip()
    if args.ref == "HEAD" and git("status", "--porcelain", cwd=HERE).stdout.strip():
        print(f"note: uncommitted changes in {git('rev-parse', '--show-toplevel', cwd=HERE).stdout.strip()} "
              f"are not judged; the workers check out {sha[:7]}", file=sys.stderr)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logdir = os.environ.get("LOGDIR") or os.path.join("tmp", "sabotage", f"parallel-{sha[:7]}-{stamp}")
    logdir = os.path.join(root, logdir)
    if os.path.isdir(logdir) and any(f.endswith(".log") for f in os.listdir(logdir)):
        sys.exit(f"refusing to overwrite the logs in {logdir}: name a new LOGDIR")
    os.makedirs(logdir, exist_ok=True)

    jobs = args.jobs or os.cpu_count() or 2
    workers = prepare_workers(root, sha, jobs)
    drivers = discover(workers[0], args.drivers)
    py = [d for d, kind in drivers if kind == "py"]
    listing = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), "--list", workers[0], *py],
                             capture_output=True, text=True)
    if listing.returncode:
        sys.exit(f"could not read the drivers' sabotages:\n{listing.stderr}")
    names = json.loads(listing.stdout)
    # The shell driver is one task of several sabotages, so it goes first.
    tasks = [{"kind": "sh", "driver": d} for d, kind in drivers if kind == "sh"]
    tasks += [{"kind": "py", "driver": d, "name": n} for d in py for n in names[d]]
    print(f"judging {sha[:7]}: {len(drivers)} drivers, {len(tasks)} tasks, {jobs} workers "
          f"in {os.path.join(root, WORKERS_DIR)}; logs in {logdir}", flush=True)

    t_start = time.monotonic()
    procs = []
    for i, w in enumerate(workers, 1):
        err = open(os.path.join(logdir, f"worker-w{i}.stderr"), "w")
        procs.append(subprocess.Popen([sys.executable, "-B", os.path.abspath(__file__), "--worker", w],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                      text=True, bufsize=1, start_new_session=True))
    baselines = []
    for i, p in enumerate(procs, 1):
        line = p.stdout.readline()
        msg = json.loads(line) if line else {"ready": False, "error": "exited before its baseline"}
        if not msg.get("ready"):
            for q in procs:
                q.stdin.close()
            sys.exit(f"w{i} refused to judge on {sha[:7]}: {msg.get('error')}")
        baselines.append(msg["baseline_s"])

    todo = queue.Queue()
    for t in tasks:
        todo.put(t)
    results, lock, stop = {}, threading.Lock(), threading.Event()
    done = [0]

    def key(t):
        return (t["driver"], t.get("name"))

    def pump(i, p):
        while not stop.is_set():
            try:
                t = todo.get_nowait()
            except queue.Empty:
                break
            p.stdin.write(json.dumps(t) + "\n")
            p.stdin.flush()
            line = p.stdout.readline()
            r = json.loads(line) if line else {"status": "error", "error": f"w{i} died", "fatal": True}
            r["worker"] = f"w{i}"
            with lock:
                results[key(t)] = r
                done[0] += 1
                n = done[0]
                if r["status"] not in ("caught", "shell") or n % 25 == 0 or n == len(tasks):
                    elapsed = time.monotonic() - t_start
                    eta = elapsed / n * (len(tasks) - n)
                    label = t["driver"] + (f": {t['name']}" if t.get("name") else "")
                    print(f"[{n}/{len(tasks)} {elapsed / 60:.1f} min, ~{eta / 60:.0f} min left] "
                          f"{r['status']:<9} {label[:110]}", flush=True)
            if r.get("fatal"):
                break
        try:
            p.stdin.close()
        except OSError:
            pass
        p.wait()

    threads = [threading.Thread(target=pump, args=(i, p), daemon=True) for i, p in enumerate(procs, 1)]
    for th in threads:
        th.start()
    try:
        for th in threads:
            th.join()
    except KeyboardInterrupt:
        # The workers run in their own sessions, so Ctrl-C reaches only this process.
        # Each finishes the sabotage it holds and restores it before it exits.
        stop.set()
        print("\ninterrupted: letting each worker finish and restore the sabotage it holds", flush=True)
        for th in threads:
            th.join()
    wall = time.monotonic() - t_start

    failed = False
    summary = []
    for d, kind in drivers:
        path = os.path.join(logdir, d + ".log")
        if kind == "sh":
            r = results.get((d, None), {"status": "error", "error": "never run"})
            if r["status"] == "error":
                text, rc = f"ERROR: {r['error']}\ndriver exit 1\n", 1
            else:
                text, rc = r["output"] + f"driver exit {r['rc']}\n", r["rc"]
            open(path, "w").write(text)
            s = SUMMARY.findall(text)
            survivors = text.count("<-- SURVIVED")
            if rc:
                summary.append(f"{d:<42} driver errored (exit {rc}) -> {path}")
                failed = True
            elif survivors or (s and s[-1][1] != "0"):
                summary.append(f"{d:<42} SURVIVOR(S) -> {path}")
                failed = True
            else:
                c, sv, _ = s[-1] if s else ("?", "?", "")
                summary.append(f"{d:<42} {c} caught, {sv} survived  -> {path}")
            continue
        text, rc, c, sv, un, er = driver_log(d, names[d], results)
        open(path, "w").write(text)
        if rc:
            summary.append(f"{d:<42} driver errored (exit {rc}) -> {path}")
            failed = True
        elif sv:
            summary.append(f"{d:<42} SURVIVOR(S) -> {path}")
            failed = True
        else:
            summary.append(f"{d:<42} {c} caught, {sv} survived  -> {path}")

    statuses = [r["status"] for r in results.values() if r["status"] != "shell"]
    judging = sum(r.get("seconds", 0) for r in results.values())
    json.dump({"sha": sha, "jobs": jobs, "workers": workers, "wall_s": round(wall, 1),
               "judging_s": round(judging, 1), "baselines_s": baselines,
               "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "results": [{"driver": d, "name": n, **r} for (d, n), r in results.items()]},
              open(os.path.join(logdir, "results.json"), "w"), indent=1)

    print("\n" + "\n".join(summary))
    print(f"\n{len(results)} of {len(tasks)} tasks judged at {sha[:7]}: "
          f"{statuses.count('caught')} caught, {statuses.count('survived')} survived, "
          f"{statuses.count('unapplied')} could not be applied, {statuses.count('error')} errored "
          f"(plus the shell driver above). {wall / 60:.1f} min wall clock on {jobs} workers, "
          f"{judging / 60:.1f} min of judging.")
    dirty = [w for w in workers if git("status", "--porcelain", cwd=w).stdout.strip()]
    if dirty:
        failed = True
        print("workers left dirty, which the next run will refuse: " + ", ".join(dirty))
    print(f"\nThe workers stay for the next run: {os.path.join(root, WORKERS_DIR)}/w1..w{jobs}. "
          f"This script never removes them; that is the owner's call.")
    if len(results) < len(tasks):
        failed = True
    return 1 if failed else 0


def compare(serial_dir, parallel_dir):
    """Whether two runs judged every sabotage the same: status, and the suites and
    first failures under it. Drivers missing on either side count as differences."""
    logs = lambda d: {f[:-4] for f in os.listdir(d) if f.endswith(".log") and not f.startswith("worker-")}
    a, b = logs(serial_dir), logs(parallel_dir)
    differences, same = [], 0
    for d in sorted(a | b, key=natural):
        if d not in a or d not in b:
            differences.append(f"{d}: only in {'the first' if d in a else 'the second'} run")
            continue
        x = blocks_of(open(os.path.join(serial_dir, d + ".log")).read())
        y = blocks_of(open(os.path.join(parallel_dir, d + ".log")).read())
        for name in sorted(set(x) | set(y)):
            if name not in x or name not in y:
                differences.append(f"{d}: {name}: only in {'the first' if name in x else 'the second'} run")
            elif x[name] != y[name]:
                differences.append(f"{d}: {name}:\n    first:  " + "\n            ".join([x[name][1], *x[name][2]])
                                   + "\n    second: " + "\n            ".join([y[name][1], *y[name][2]]))
            else:
                same += 1
    for line in differences:
        print(line)
    print(f"\n{same} sabotages judged identically, {len(differences)} difference(s), "
          f"across {len(a | b)} drivers")
    return 1 if differences else 0


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        return worker_main(sys.argv[2])
    if len(sys.argv) >= 3 and sys.argv[1] == "--list":
        return list_main(sys.argv[2], sys.argv[3:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", type=int, default=0, help="workers (default: the CPU count)")
    ap.add_argument("--ref", default="HEAD", help="the commit to judge (default: HEAD)")
    ap.add_argument("--compare", nargs=2, metavar=("FIRST_LOGDIR", "SECOND_LOGDIR"),
                    help="compare two runs' logs sabotage by sabotage, and run nothing")
    ap.add_argument("drivers", nargs="*", help="driver names, as chain.sh takes them")
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
