#!/usr/bin/env python3
"""Round 22: the suites' speed-ups (chore/corpus-speedups), undone the ways that would
change a verdict.

The report suite parses each distinct paragraph once while story_strings() reads a
story, and the sync suite evaluates each test's blocks in one node process. Both were
shown to change no verdict by judging the whole corpus before and after. These bring
back what would: the memo keeping a parse that failed, serving a build outside
story_strings(), comparing styles by their own attributes only (blind to a bullet on
their class, as the second cold check showed), or keeping the first paragraph's live
attributes rather than a copy; the node evaluator running every block in one context, dropping a
failure, naming no block, or answering every block with the first one's object.

Not here: the corpus runner's own check that a worker is at the proved commit and
clean. Its test is in tests/corpus.test.py, which by design judges no sabotage.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver

REPORT_TEST = "tests/report.test.py"
SYNC_TEST = "tests/sync.test.py"

MEMO_KEEP = '            _PARSED[key] = (dict(self.__dict__), state)\n'
MEMO_MISS = '            super().__init__(text, *args)\n' + MEMO_KEEP
MEMO_GATE = '        if _reading and isinstance(text, str) and len(args) == 1 and not kwargs:\n'
STYLE_STATE = ('    names = set(vars(style))\n'
               '    for cls in type(style).__mro__[:-1]:\n'
               '        names.update(k for k in vars(cls) if not k.startswith("_"))\n')
NODE_CONTEXT = '         "  try { return {ok: vm.runInNewContext("\n         f"\'(new Function(b + \\"; return {var};\\"))()\', {{b}})}}; }}"\n'
NODE_FAILURE = '        assert "ok" in result, f"node could not evaluate block {i}:\\n{result[\'error\']}"\n'
NODE_RETURN = '    return [result["ok"] for result in results]\n'

S = {
    "M1 report: the memo keeps a parse that failed, so the markup parses the second time": [
        (REPORT_TEST, MEMO_MISS, MEMO_KEEP + '            super().__init__(text, *args)\n', 1)],
    "M2 report: the memo serves every build, not only a story being read": [
        (REPORT_TEST, MEMO_GATE, '        if isinstance(text, str) and len(args) == 1 and not kwargs:\n', 1)],
    "M8 report: styles are compared by their own attributes only, blind to a bullet on their class": [
        (REPORT_TEST, STYLE_STATE, '    names = set(vars(style))\n', 1)],
    "M9 report: the memo keeps the first paragraph's live attributes, so a later change reaches every later build": [
        (REPORT_TEST, MEMO_KEEP, '            _PARSED[key] = (self.__dict__, state)\n', 1)],
    "N1 sync: every block runs in one context": [
        (SYNC_TEST, NODE_CONTEXT, '         "  try { return {ok: "\n         f"(new Function(b + \\"; return {var};\\"))()}}; }}"\n', 1)],
    "N2 sync: a block that can't be evaluated is dropped": [
        (SYNC_TEST, NODE_FAILURE, '        pass\n', 1),
        (SYNC_TEST, NODE_RETURN, '    return [result.get("ok") for result in results if "ok" in result]\n', 1)],
    "N3 sync: a failure doesn't name its block": [
        (SYNC_TEST, NODE_FAILURE, '        assert "ok" in result, f"node could not evaluate a block:\\n{result[\'error\']}"\n', 1)],
    "N4 sync: every block is answered with the first block's object": [
        (SYNC_TEST, NODE_RETURN, '    return [results[0]["ok"] for result in results]\n', 1)],
}

if __name__ == "__main__":
    run_driver(S)
