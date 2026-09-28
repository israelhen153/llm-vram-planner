#!/usr/bin/env python3
"""Round 24: the second cold check of chore/corpus-speedups (1572e53), against 97a0309.

Round 23 found the report suite's parse memo serving a style's bulletText by the
style's NAME; the fix compares the style attribute by attribute — by vars(style).
reportlab reads the bullet through getattr(style, 'bulletText', None), which also
sees what vars() does not: a class attribute, a property, a __getattr__. The first
paragraph built with a given markup and style name in the suite is a card WITH
constants, so a figure carried by ParagraphStyle's class attribute for a card
without constants only is served that first card's parse. P3 and P3b SURVIVED at
1572e53 (156 passed, 0 failed); 97a0309 caught both (report: 3 failed, "the report
prints no throughput figure without constants, and says why in its place").

Controls, all caught at both commits: the Cost heading's own bulletText set after
construction for a card with constants only (P1) or without (P4), and a new style
of the same name carrying the figure as its own attribute (P8). P1 is kept for what
its third failure shows: the memo stores the first paragraph itself, so a paragraph
the engine changes after building it is served changed to every later card (97a0309
failed 2 tests, 1572e53 failed 3). The engine changes no paragraph after building it
today, so nothing is judged wrongly yet. K1 is a sync-suite control: the shared node
process still sees an escaping regression in js_str. G1 puts round 3's every-card
guess (X2, which only the page golden catches) beside round 23's E1 (a real report
catch) — one sabotage the runner must report as caught for real, not golden-only,
and the fixture RUNNER's R8 and R10 misreport.

RUNNER, judged by tests/corpus.test.py with --runner as round 23's was: the runner
reporting a sabotage as golden-only when any red suite is (R8) or whenever it is
caught (R9); --early-exit stopping at a suite red on golden failures alone, so a
later real catch is never run and the sabotage is reported golden-only (R10); the
dirty check between sabotages dropped (R11); a driver's sabotages read from the
runner's own checkout rather than the commit under test (R12) and the harness too
(R13, the one the corpus test catches); the bytecode guard dropped, which is the 90
false catches at d99f1b6 put back (R14); and a driver's log saying exit 0 with an
errored or unapplied sabotage in it (R15). 97a0309 had no runner test at all, so
these are gaps, not regressions: only R13 goes red at 1572e53.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import REPORT_PY, SYNC_PY, PY_COST_HEAD
from engine_r3_identical_estimate_and_href import S as ROUND_3
from engine_r23_speedups_cold_check import S as ROUND_23, judge_runner

RUNNER_PY = "tests/sabotage/parallel.py"
PY_NOTES_HEAD = '        story.append(Paragraph("Notes and assumptions", self.styles["SectionHead"]))\n'
# Round 23's figure: it needs no PERF constant to print, so with_moved_figures cannot see it.
FIG = 'f"~{round(c[\'device_bw\'] * 0.7)} tokens/sec"'


def named(sabotages, prefix):
    """One sabotage of another round, by the prefix of its name — derived, so a renamed
    sabotage errors here instead of silently judging a copy."""
    return next(v for k, v in sabotages.items() if k.startswith(prefix + " "))


def class_bullet(head):
    """The figure on ParagraphStyle itself around one heading, for a card without
    constants only, and taken off the class again after the heading is built."""
    return [(REPORT_PY, head,
             '        if not c["throughput_modelled"]:\n'
             f'            ParagraphStyle.bulletText = {FIG}\n' + head +
             '        ParagraphStyle.bulletText = None\n', 1)]


def own_bullet(condition):
    return [(REPORT_PY, PY_COST_HEAD,
             '        head = Paragraph("Cost estimate", self.styles["SectionHead"])\n'
             f'        if {condition}:\n'
             f'            head.bulletText = {FIG}\n'
             '        story.append(head)\n', 1)]


S = {
    "P3 py: the figure rides on ParagraphStyle's class bulletText around the Cost heading, for a card without constants only":
        class_bullet(PY_COST_HEAD),
    "P3b py: the same class bulletText around the Notes heading, for a card without constants only":
        class_bullet(PY_NOTES_HEAD),
    "P1 py (control): the Cost heading's own bulletText set after it is built, for a card with constants only":
        own_bullet('c["throughput_modelled"]'),
    "P4 py (control): the Cost heading's own bulletText set after it is built, for a card without constants only":
        own_bullet('not c["throughput_modelled"]'),
    "P8 py (control): a new SectionHead style of the same name carrying the figure as its own bulletText, without constants only": [
        (REPORT_PY, PY_COST_HEAD,
         '        head_style = self.styles["SectionHead"]\n'
         '        if not c["throughput_modelled"]:\n'
         f'            head_style = ParagraphStyle("SectionHead", parent=head_style, bulletText={FIG})\n'
         '        story.append(Paragraph("Cost estimate", head_style))\n', 1)],
    "K1 sync (control): js_str no longer escapes the apostrophe": [
        (SYNC_PY, '.replace("\'", "\\\\\'")', '', 1)],
    "G1 js+py: round 3's X2 (a page-golden-only catch) beside round 23's E1 (a real report catch)":
        named(ROUND_3, "X2 js:") + named(ROUND_23, "E1 py:"),
}

GOLDEN_ONLY = '            "golden_only": status == "caught" and bool(red) and all(r[1] and set(r[1]) <= golden for r in red)}\n'
EARLY_BREAK = '                    if rc != 0 and not (fails and set(fails) <= golden):\n'
DIRTY = '        dirty = git("status", "--porcelain", cwd=tree).stdout\n        if dirty.strip():\n'
DRIVER_PATH = '            driver, os.path.join(tree, "tests", "sabotage", driver + ".py"))\n'
LOAD = ('    sys.path[0:1] = [os.path.join(tree, "tests", "sabotage")]\n'
        '    import harness\n'
        '    if os.path.realpath(harness.ROOT) != os.path.realpath(tree):\n'
        '        raise RuntimeError(f"imported the harness of {harness.ROOT}, not of {tree}")\n')
BYTECODE = ('    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"\n'
            '    now = time.time()\n'
            '    for rel in git("ls-files", "*.py", cwd=tree).stdout.split():\n'
            '        os.utime(os.path.join(tree, rel), (now, now))\n')
DRIVER_RC = '    rc = 1 if errors else 2 if unapplied else 0\n'

RUNNER = {
    "R8 runner: a sabotage is golden-only when any red suite is, not when every one is": [
        (RUNNER_PY, GOLDEN_ONLY, GOLDEN_ONLY.replace("and all(", "and any("), 1)],
    "R9 runner: every caught sabotage is reported golden-only": [
        (RUNNER_PY, GOLDEN_ONLY, '            "golden_only": status == "caught" and bool(red)}\n', 1)],
    "R10 runner: --early-exit stops at a suite that is red on golden failures alone": [
        (RUNNER_PY, EARLY_BREAK, '                    if rc != 0:\n', 1)],
    "R11 runner: a worker left dirty by one sabotage judges the next on that tree": [
        (RUNNER_PY, DIRTY, '        dirty = ""\n        if dirty.strip():\n', 1)],
    "R12 runner: a driver's sabotages are read from the runner's own checkout, not the commit under test": [
        (RUNNER_PY, DRIVER_PATH, '            driver, os.path.join(HERE, driver + ".py"))\n', 1)],
    "R13 runner (control): the harness is imported from the runner's own checkout": [
        (RUNNER_PY, LOAD, '    sys.path.append(os.path.join(tree, "tests", "sabotage"))\n    import harness\n', 1)],
    "R14 runner: a worker writes bytecode again and trusts whatever .pyc it finds": [
        (RUNNER_PY, BYTECODE, '', 1)],
    "R15 runner: a driver's log says exit 0 with an errored or unapplied sabotage in it": [
        (RUNNER_PY, DRIVER_RC, '    rc = 0\n', 1)],
}

if __name__ == "__main__":
    if "--runner" in sys.argv:
        judge_runner(RUNNER)
    else:
        run_driver(S)
