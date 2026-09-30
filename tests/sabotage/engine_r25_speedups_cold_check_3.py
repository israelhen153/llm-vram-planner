#!/usr/bin/env python3
"""Round 25: the third cold check of chore/corpus-speedups, at b520f07 against 3496a57.

By content, b520f07 differs from 3496a57 in tests/report.test.py, tests/sync.test.py,
tests/corpus.test.py, tests/sabotage/parallel.py, the three speed-up drivers and docs.
The engine, the data, the goldens, the harness and the other four judging suites are
byte-identical, so only the report and sync suites, and the runner, can change a verdict.

The report suite's parse memo compares a style by style_state(): every attribute the
style or its class carries, read through getattr, dropping callables. reportlab reads
the bullet as getattr(style, 'bulletText', None), which a __getattr__ also answers, and
a bullet that happens to be callable is read like any other. So a figure served by a
__getattr__ on ParagraphStyle (P9), or carried by a callable str on the style (P12), for
a card without constants only, is invisible to the state; the memo serves the first
card's parse (a card with constants, no bullet), and both SURVIVED at b520f07 (159
passed, 0 failed) where 3496a57 caught both (report: 3 failed, "the report prints no
throughput figure without constants, and says why in its place"). A bullet property on
the class (P11) is read by value and caught. The same __getattr__ figure for a card with
constants only (P13) is caught at both commits and at b520f07 fails one test more, "the
report prints no None, nan or throughput figure for a card without constants", because
the memo serves that first card's bullet to every later card: a failure naming a card
the sabotage never touched. K2 and K3 drop a js_str escape and are caught by the sync
suite's hostile-note round-trip under the shared node process.

RUNNER, judged by tests/corpus.test.py with --runner as rounds 23 and 24 were, and all
nine SURVIVED at b520f07: the shell driver run from the runner's own checkout (R16); the
driver list (R17) or the sabotage names (R18) read from the runner's checkout, so a
sabotage only the commit under test has is never judged; --compare reading statuses
alone (R19), the shape the README says passed 90 stale-bytecode catches, or passing a
driver one run lacks (R20); --early-exit running only the suites ORDER_HINT names, so a
seventh judging suite never runs (R21); a run with survivors exiting 0 (R23); --ref
ignored for the runner's own HEAD (R27); and the golden failures found by emptying the
first golden file only, so a catch by the report golden alone is reported real (R30).
R1, round 23's commit check, is the control. 3496a57 tests none of these either: gaps,
not regressions, as round 24's were.

Not a sabotage, found by running the corpus: run() binds the workers' lock file to
`lock` and later rebinds `lock` to a threading.Lock(), which drops the file's last
reference, closes it and releases the flock for the whole judging phase (3496a57 does
the same). A second run then passes the lock and finds the first run's workers dirty,
or races it between two sabotages.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import REPORT_PY, SYNC_PY, PY_COST_HEAD
from engine_r23_speedups_cold_check import RUNNER as ROUND_23_RUNNER, judge_runner
from engine_r24_speedups_cold_check_2 import named

RUNNER_PY = "tests/sabotage/parallel.py"
# Round 23's figure: it needs no PERF constant to print, so with_moved_figures cannot see it.
FIG = 'f"~{round(c[\'device_bw\'] * 0.7)} tokens/sec"'
JS_ESCAPE = '        ch if (ch >= " " and ch not in "\\u2028\\u2029") else "\\\\u%04x" % ord(ch)\n'


def around_cost_head(before, after):
    """An edit that wraps the Cost heading: `before` runs for a card without
    constants only, `after` always, so the next card starts from a clean class."""
    return [(REPORT_PY, PY_COST_HEAD,
             '        if not c["throughput_modelled"]:\n' + before + PY_COST_HEAD + after, 1)]


GETATTR_BULLET = around_cost_head(
    '            def _bullet_of(style, name):\n'
    '                if name == "bulletText":\n'
    f'                    return {FIG}\n'
    '                raise AttributeError(name)\n'
    '            ParagraphStyle.__getattr__ = _bullet_of\n',
    '        if "__getattr__" in vars(ParagraphStyle):\n'
    '            del ParagraphStyle.__getattr__\n')[0][2]

S = {
    "P9 py: the figure served by a __getattr__ on ParagraphStyle, for a card without constants only": [
        (REPORT_PY, PY_COST_HEAD, GETATTR_BULLET, 1)],
    "P12 py: the figure as a callable str bulletText on the SectionHead style, for a card without constants only":
        around_cost_head(
            '            class _Fig(str):\n'
            '                def __call__(self):\n'
            '                    return None\n'
            f'            self.styles["SectionHead"].bulletText = _Fig({FIG})\n',
            '        if "bulletText" in vars(self.styles["SectionHead"]):\n'
            '            del self.styles["SectionHead"].bulletText\n'),
    "P11 py (control): the figure as a bulletText property on ParagraphStyle, for a card without constants only":
        around_cost_head(
            f'            ParagraphStyle.bulletText = property(lambda style: {FIG})\n',
            '        if "bulletText" in vars(ParagraphStyle):\n'
            '            del ParagraphStyle.bulletText\n'),
    "P13 py (control): P9's __getattr__ figure for a card with constants only, which the memo serves to every later card": [
        (REPORT_PY, PY_COST_HEAD,
         GETATTR_BULLET.replace('        if not c["throughput_modelled"]:\n', '        if c["throughput_modelled"]:\n'), 1)],
    "K2 sync (control): js_str lets a raw carriage return through": [
        (SYNC_PY, JS_ESCAPE,
         '        ch if ((ch >= " " or ch == "\\r") and ch not in "\\u2028\\u2029") else "\\\\u%04x" % ord(ch)\n', 1)],
    "K3 sync (control): js_str no longer escapes the U+2028/U+2029 line terminators": [
        (SYNC_PY, JS_ESCAPE, '        ch if ch >= " " else "\\\\u%04x" % ord(ch)\n', 1)],
}

JUDGE_SHELL = '    p = subprocess.run(["bash", os.path.join(tree, "tests", "sabotage", driver + ".sh")],\n'
DISCOVER = '    drivers = discover(workers[0], args.drivers)\n'
LISTING = '    listing = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), "--list", workers[0], *py],\n'
COMPARE_SAME = ('            elif (x[name][0] != y[name][0] or not set(y[name][2]) <= set(x[name][2])) '
                'if subset else (x[name] != y[name]):\n')
COMPARE_MISSING = ('        if d not in a or d not in b:\n'
                   '            differences.append(f"{d}: only in {\'the first\' if d in a else \'the second\'} run")\n'
                   '            continue\n')
ORDER = '    order = sorted(suites, key=lambda s: ORDER_HINT.index(s[0]) if s[0] in ORDER_HINT else len(ORDER_HINT))\n'
SURVIVOR_FAILS = ('        elif sv:\n'
                  '            summary.append(f"{d:<42} SURVIVOR(S) -> {path}")\n'
                  '            failed = True\n')
REF = '    sha = git("rev-parse", "--verify", f"{args.ref}^{{commit}}", cwd=HERE).stdout.strip()\n'
EMPTY_GOLDENS = '        for f in files:\n            with open(os.path.join(tree, f), "w") as fh:\n'
OWN_ROOT = 'os.path.dirname(os.path.dirname(HERE))'

RUNNER = {
    "R16 runner: the shell driver is run from the runner's own checkout, not the commit under test": [
        (RUNNER_PY, JUDGE_SHELL, '    p = subprocess.run(["bash", os.path.join(HERE, driver + ".sh")],\n', 1)],
    "R17 runner: the drivers are discovered in the runner's own checkout, so one only the commit under test has is never judged": [
        (RUNNER_PY, DISCOVER, f'    drivers = discover({OWN_ROOT}, args.drivers)\n', 1)],
    "R18 runner: the sabotage names are listed from the runner's own checkout, so one only the commit under test has is never judged": [
        (RUNNER_PY, LISTING, LISTING.replace('"--list", workers[0]', f'"--list", {OWN_ROOT}'), 1)],
    "R19 runner: --compare compares statuses alone, blind to the suites and failures under them": [
        (RUNNER_PY, COMPARE_SAME, '            elif x[name][0] != y[name][0]:\n', 1)],
    "R20 runner: --compare passes a driver judged in one run only": [
        (RUNNER_PY, COMPARE_MISSING, '        if d not in a or d not in b:\n            continue\n', 1)],
    "R21 runner: --early-exit runs only the suites ORDER_HINT names, so a seventh judging suite never runs": [
        (RUNNER_PY, ORDER, '    order = [s for n in ORDER_HINT for s in suites if s[0] == n]\n', 1)],
    "R23 runner: a run with survivors exits 0": [
        (RUNNER_PY, SURVIVOR_FAILS, SURVIVOR_FAILS.replace('            failed = True\n', ''), 1)],
    "R27 runner: --ref is ignored and the runner's own HEAD is judged in its place": [
        (RUNNER_PY, REF, '    sha = git("rev-parse", "HEAD", cwd=HERE).stdout.strip()\n', 1)],
    "R30 runner: the golden failures are found by emptying the first golden file only, so a catch by the other golden is called real": [
        (RUNNER_PY, EMPTY_GOLDENS, EMPTY_GOLDENS.replace("for f in files:", "for f in files[:1]:"), 1)],
    "R1 runner (control): round 23's commit check dropped": named(ROUND_23_RUNNER, "R1"),
}

if __name__ == "__main__":
    if "--runner" in sys.argv:
        judge_runner(RUNNER)
    else:
        run_driver(S)
