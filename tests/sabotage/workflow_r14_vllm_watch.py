#!/usr/bin/env python3
"""Round 14 against the vLLM release watch (tools/vllm_watch.py, judged by
tests/vllm_watch.test.py).

The watch tells the owner when vLLM ships past the release the planner pins, and what
to re-check. These put back each way it could stay silent, or say the wrong thing:
versions compared as text; pre-releases, dev releases or yanked releases counted;
post-releases counted again, in the release number, beside their release, or in its
place, or dropped where one is all of a release that can be installed; the pin
hard-coded, or taken from one engine when the two disagree; the files to re-check given
as today's list rather than searched; a new issue every week, a release announced
again, or told again under its post-release's name; a failure to read PyPI passed off as
a quiet week, three ways, or gh's answer read as an empty issue list, whether it is not
JSON or JSON of another shape; a contract nobody defines dropped from the notice without
a word, one both engines define searched in one engine only, or the FP8 refusal named at
its Python gate and not at the rule in both engines; and its workflow at minute 0, given
`contents: write`, or kept green with `|| true` or continue-on-error.

The hard-coded pin and file list are read from the tree as this driver loads, so each
is the one that is right today, which is what makes it a sabotage the real tree alone
cannot tell apart.
"""
import os, re, subprocess, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver, ROOT

VW = "tools/vllm_watch.py"
WF = ".github/workflows/vllm-watch.yml"

with open(os.path.join(ROOT, "index.html"), encoding="utf-8") as _fh:
    PIN = re.search(r'^const ROCM = \{[\s\S]*?"vllm":\s*"([^"]+)"', _fh.read(), re.M).group(1)
LISTED = [(path, 1) for path in subprocess.run(["git", "-C", ROOT, "grep", "-l", "-I", "-F", PIN],
                                               capture_output=True, text=True, check=True).stdout.split()]


def swap(f, old, new):
    return [(f, old, new, 1)]


S = {
 # ---- V: which releases count, and in what order ----
 'V1 versions compared as strings':
     swap(VW, '        if release > pin_release:\n', '        if version > pin.lstrip("vV"):\n'),
 'V2 pre-releases counted':
     swap(VW, '    return not (m.group("pre") or m.group("dev")), ', '    return not m.group("dev"), '),
 'V3 dev releases counted':
     swap(VW, '    return not (m.group("pre") or m.group("dev")), ', '    return not m.group("pre"), '),
 'V4 yanked releases counted':
     swap(VW, '        installable = [f for f in files if not f["yanked"]]\n', '        installable = files\n'),
 'V5 post-releases counted again, the post segment back in the release number':
     swap(VW, '(int(m.group("epoch") or 0), tuple(release)), post_n\n',
              '(int(m.group("epoch") or 0), tuple(release), post_n), post_n\n'),
 'V6 a release returned beside its own post-releases, one version per release dropped':
     swap(VW, '            found[release] = min(found.get(release, (post, version)), (post, version))',
              '            found[(release, post)] = (post, version)'),
 'V7 post-releases dropped, so a release whose final is yanked is never told':
     swap(VW, '        if not is_final:\n', '        if not is_final or post >= 0:\n'),
 'V8 a release named by its post-release when its final can be installed too':
     swap(VW, '            found[release] = min(found.get(', '            found[release] = max(found.get('),

 # ---- P: the pin, and what the notice names ----
 'P1 a hard-coded pin':
     swap(VW, '    return pin\n\n\ndef mentions(', f'    return {PIN!r}\n\n\ndef mentions('),
 "P2 the engines' disagreement ignored, index.html's pin taken":
     swap(VW, '    if len(set(pins.values())) != 1:\n',
              '    pins = {"index.html": pins["index.html"]}\n    if len(set(pins.values())) != 1:\n'),
 'P3 a hard-coded file list':
     swap(VW, '    root = root or ROOT\n    bare = ', f'    return {LISTED!r}\n    root = root or ROOT\n    bare = '),
 'P4 a contract nobody defines dropped from the notice without a word':
     swap(VW, '    if missing:\n', '    if False:\n'),
 'P5 a contract both engines define searched in generate_report.py only':
     swap(VW, '    found = [(name, what, [(engine, n) for engine in ENGINES\n',
              '    found = [(name, what, [(engine, n) for engine in ENGINES[-1:]\n'),
 'P6 the FP8 refusal named at its Python gate, and not at the rule in both engines':
     swap(VW, r'r"(?:def\s+fp8_weights_blocked|function\s+fp8WeightsBlocked)\("', r'r"def refuse_fp8_where_vllm_cannot\("'),

 # ---- R: what it tells GitHub ----
 'R1 a new issue every week, the open one ignored':
     swap(VW, '    open_issues = sorted(i["number"] for i in ours if str(i.get("state", "")).lower() == "open")\n',
              '    open_issues = []\n'),
 'R2 an announced release announced again':
     swap(VW, '    fresh = [v for v in newer if release_of(v) not in told]\n', '    fresh = list(newer)\n'),
 "R3 a release told again under its post-release's name, the dedupe by exact string":
     [(VW, '    told = {release_of(version) for issue in ours\n', '    told = {version for issue in ours\n', 1),
      (VW, '    fresh = [v for v in newer if release_of(v) not in told]\n',
           '    fresh = [v for v in newer if v not in told]\n', 1)],

 # ---- F: when PyPI or GitHub cannot be read ----
 'F1 a fetch error treated as no news, exit 0':
     swap(VW, '        print(f"vllm watch: {e}", file=sys.stderr)\n        return 1\n',
              '        print(f"vllm watch: {e}", file=sys.stderr)\n        return 0\n'),
 'F2 a fetch error read as a week with nothing newer':
     swap(VW, "        raise WatchError(f\"could not read vLLM's releases from {url}: {type(e).__name__}: {e}\")\n",
              '        return {"releases": {"0": [{"yanked": False}]}}\n'),
 'F3 the exit code dropped on the way out of the process':
     swap(VW, '    sys.exit(main())\n', '    main()\n'),
 "F4 gh's non-JSON answer read as an empty issue list":
     swap(VW, "        raise WatchError(f\"gh {' '.join(args[:2])} did not answer JSON: {e}\")\n", '        return []\n'),
 "F5 the issue list's shape unchecked, so {} reads as no issues and a second issue is created":
     swap(VW, '    check_issue_list(listed)\n', '    pass\n'),
 'F6 an issue missing a field the watch reads taken as it is':
     [(VW, '            if field not in item:\n', '            if False:\n', 1),
      (VW, '            if type(item[field]) is not kind:\n', '            if False:\n', 1)],

 # ---- W: the watch's own workflow ----
 'W1 the schedule moved to minute 0':
     swap(WF, '    - cron: "29 5 * * 3"', '    - cron: "0 5 * * 3"'),
 'W2 permissions widened to contents: write':
     swap(WF, '      contents: read # to check out', '      contents: write # to check out'),
 'W3 the step kept green with || true':
     swap(WF, '        run: python3 tools/vllm_watch.py --repo "$GITHUB_REPOSITORY"\n',
              '        run: python3 tools/vllm_watch.py --repo "$GITHUB_REPOSITORY" || true\n'),
 'W4 the step kept green with continue-on-error':
     swap(WF, "      - name: Compare vLLM's releases with the pinned one\n",
              "      - name: Compare vLLM's releases with the pinned one\n        continue-on-error: true\n"),
}

if __name__ == "__main__":
    run_driver(S)
