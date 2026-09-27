#!/usr/bin/env python3
"""Round 23: the cold check of chore/corpus-speedups (688a490), against dcf6c60.

The report suite's parse memo is keyed on a paragraph's markup and its style's NAME,
and serves the first paragraph's whole __dict__ to the next. reportlab's Paragraph
reads more off the style than its name: with no bulletText passed, the bullet is
`style.bulletText`, and story_strings() harvests bulletText. So a figure carried by
a style's bulletText is read from whichever config first built that markup — and
the first story_strings() in the suite is a card WITH constants (DUAL, perfKey
"nvidia"). E1 puts the figure there for a card without constants only; E2 for a card
with them only; E3/E3k pass the same figure positionally and by keyword, which the
memo's gates route around the memo, as controls; E4 rides the "Small" style, whose
later paragraphs include text only a card without constants shows.

Then the memo's own gates, one at a time (M3-M5) and each with the control it lets
through (M6, M7); the sync suite's shared node process weakened five ways (N5-N9);
and, in RUNNER, judged by tests/corpus.test.py with --runner since no judging suite
reads parallel.py, the corpus runner's once-per-run proof weakened six ways.
"""
import os, subprocess, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver, apply_edits, restore_files, edits_of
from anchors import REPORT_PY, PY_COST_HEAD

REPORT_TEST = "tests/report.test.py"
SYNC_TEST = "tests/sync.test.py"
RUNNER_PY = "tests/sabotage/parallel.py"

# A speed figure that needs no PERF constant to print, so it survives with_moved_figures:
# the point is where it rides, not what it is.
FIG = 'f"~{round(c[\'device_bw\'] * 0.7)} tokens/sec"'
COST_SECTION = "        # ---- Cost ----\n"

MEMO_GATE = '        if _reading and isinstance(text, str) and len(args) == 1 and not kwargs:\n'
MEMO_KEY = '            key = (text, args[0].name)\n'
MEMO_MISS = '                super().__init__(text, *args)\n'

NODE_VM = '         "const vm = require(\'vm\');"\n'
NODE_CONTEXT = ('         "  try { return {ok: vm.runInNewContext("\n'
                '         f"\'(new Function(b + \\"; return {var};\\"))()\', {{b}})}}; }}"\n')
NODE_CATCH = '         "  catch (e) { return {error: String(e && e.stack || e)}; }"\n'
NODE_COUNT = '    assert len(results) == len(blocks), f"{len(blocks)} blocks, {len(results)} results"\n'
NODE_RETURN = '    return [result["ok"] for result in results]\n'

E3 = ('        story.append(Paragraph("Cost estimate", self.styles["SectionHead"],\n'
      f'                               None if c["throughput_modelled"] else {FIG}))\n')
E3K = E3.replace("                               None", "                               bulletText=None")

S = {
    "E1 py: the figure rides on the SectionHead style's bulletText, for a card without constants only": [
        (REPORT_PY, PY_COST_HEAD,
         '        if not c["throughput_modelled"]:\n'
         f'            self.styles["SectionHead"].bulletText = {FIG}\n' + PY_COST_HEAD, 1)],
    "E2 py: the figure rides on the SectionHead style's bulletText, for a card with constants only": [
        (REPORT_PY, PY_COST_HEAD,
         '        if c["throughput_modelled"]:\n'
         f'            self.styles["SectionHead"].bulletText = {FIG}\n' + PY_COST_HEAD, 1)],
    "E3 py: the same figure passed as the paragraph's positional bulletText (control)": [
        (REPORT_PY, PY_COST_HEAD, E3, 1)],
    "E3k py: the same figure passed as the paragraph's keyword bulletText (control)": [
        (REPORT_PY, PY_COST_HEAD, E3K, 1)],
    "E4 py: the figure rides on the Small style's bulletText from the cost section on, without constants only": [
        (REPORT_PY, COST_SECTION,
         '        if not c["throughput_modelled"]:\n'
         f'            self.styles["Small"].bulletText = {FIG}\n' + COST_SECTION, 1)],
    "M3 report: the memo's key drops the style's name": [
        (REPORT_TEST, MEMO_KEY, '            key = text\n', 1)],
    "M4 report: the memo no longer requires exactly one positional argument": [
        (REPORT_TEST, MEMO_GATE, '        if _reading and isinstance(text, str) and args and not kwargs:\n', 1)],
    "M5 report: the memo no longer refuses keyword arguments": [
        (REPORT_TEST, MEMO_GATE, '        if _reading and isinstance(text, str) and len(args) == 1:\n', 1),
        (REPORT_TEST, MEMO_MISS, '                super().__init__(text, *args, **kwargs)\n', 1)],
    "M6 report+py: M5, and the figure as a keyword bulletText (E3k) with it": [
        (REPORT_TEST, MEMO_GATE, '        if _reading and isinstance(text, str) and len(args) == 1:\n', 1),
        (REPORT_TEST, MEMO_MISS, '                super().__init__(text, *args, **kwargs)\n', 1),
        (REPORT_PY, PY_COST_HEAD, E3K, 1)],
    "M7 report+py: M4, and the figure as a positional bulletText (E3) with it": [
        (REPORT_TEST, MEMO_GATE, '        if _reading and isinstance(text, str) and args and not kwargs:\n', 1),
        (REPORT_PY, PY_COST_HEAD, E3, 1)],
    "N5 sync: every block runs in node's own context": [
        (SYNC_TEST, NODE_CONTEXT,
         '         "  try { globalThis.b = b; return {ok: vm.runInThisContext("\n'
         '         f"\'(new Function(b + \\"; return {var};\\"))()\')}}; }}"\n', 1)],
    "N6 sync: every block runs in one shared vm context": [
        (SYNC_TEST, NODE_VM, '         "const vm = require(\'vm\'); const ctx = vm.createContext({});"\n', 1),
        (SYNC_TEST, NODE_CONTEXT,
         '         "  try { ctx.b = b; return {ok: vm.runInContext("\n'
         '         f"\'(new Function(b + \\"; return {var};\\"))()\', ctx)}}; }}"\n', 1)],
    "N7 sync: a block that can't be evaluated is answered with null": [
        (SYNC_TEST, NODE_CATCH, '         "  catch (e) { return {ok: null}; }"\n', 1)],
    "N8 sync: the blocks' objects come back in reverse order": [
        (SYNC_TEST, NODE_RETURN, '    return [result["ok"] for result in results][::-1]\n', 1)],
    "N9 sync: the evaluator no longer checks it got one result per block": [
        (SYNC_TEST, NODE_COUNT, '', 1)],
}

RUNNER_HEAD = ('    if head != sha:\n'
               '        raise RuntimeError(f"the worker is at {head[:7]}, not at {sha[:7]}")\n')
RUNNER_CLEAN = ('    if git("status", "--porcelain", cwd=tree).stdout.strip():\n'
                '        raise RuntimeError("the worker is not clean")\n')
RUNNER_CALL = '        refuse_unless_proven(tree, sha)\n'
RUNNER_SHARED = '                golden = set(json.load(fh))\n'
RUNNER_PROOF = ('            harness.require_green_baseline()\n'
                '            golden = golden_failures(harness, tree)\n')

RUNNER = {
    "R1 runner: a worker no longer checks it is at the proved commit": [
        (RUNNER_PY, RUNNER_HEAD, '', 1)],
    "R2 runner: a worker no longer checks it is clean": [
        (RUNNER_PY, RUNNER_CLEAN, '', 1)],
    "R3 runner: a worker's clean check ignores untracked files": [
        (RUNNER_PY, RUNNER_CLEAN, RUNNER_CLEAN.replace('"--porcelain", cwd', '"--porcelain", "--untracked-files=no", cwd'), 1)],
    "R4 runner: refuse_unless_proven() is never called": [
        (RUNNER_PY, RUNNER_CALL, '', 1)],
    "R5 runner: the other workers ignore the first worker's golden failures": [
        (RUNNER_PY, RUNNER_SHARED, '                golden = set()\n', 1)],
    "R6 runner: the first worker finds the golden failures without proving the tree green": [
        (RUNNER_PY, RUNNER_PROOF, '            golden = golden_failures(harness, tree)\n', 1)],
}


def judge_runner(sabotages):
    """RUNNER through tests/corpus.test.py, in run_driver()'s shape: no judging suite
    opens parallel.py, and corpus.test.py judges no engine sabotage, so the two are
    kept apart on purpose."""
    sys.stdout.reconfigure(line_buffering=True)
    suite = ["python3", "tests/corpus.test.py"]
    if subprocess.run(suite, capture_output=True, text=True).returncode:
        sys.exit("refusing to judge: tests/corpus.test.py already red on the unmodified tree")
    print(f"{len(sabotages)} sabotage(s)")
    survived = []
    for name, spec in sabotages.items():
        touched = apply_edits(edits_of(spec))
        try:
            p = subprocess.run(suite, capture_output=True, text=True)
        finally:
            restore_files(touched)
        fails = [l.strip() for l in (p.stdout + p.stderr).splitlines() if l.lstrip().startswith("FAIL")]
        if p.returncode == 0:
            survived.append(name)
            print(f"  GREEN  {name}   <-- SURVIVED")
        else:
            print(f"  red    {name}\n         corpus[{(fails or ['(no FAIL line)'])[0][:120]}]")
    print(f"\n{len(sabotages) - len(survived)} caught, {len(survived)} survived")
    for name in survived:
        print("  SURVIVED: " + name)


if __name__ == "__main__":
    if "--runner" in sys.argv:
        judge_runner(RUNNER)
    else:
        run_driver(S)
