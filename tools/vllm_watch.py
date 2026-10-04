#!/usr/bin/env python3
"""Notice when vLLM ships a final release newer than the one the planner pins.

The planner prints `vllm serve` commands and, on AMD, a `docker run` of vLLM's ROCm
image pinned at one release. What it says about that release was read against it:
which quantization names vLLM accepts, which methods are FP8, where FP8 is refused,
what GGUF needs, the ROCm lines and the date they were checked. All of it can go stale
the week vLLM ships again, and nothing in the tree notices.

So once a week this reads the pin from both engines' ROCm table (index.html and
generate_report.py, which tests/parity.test.py holds equal; when they differ it stops
there), asks PyPI which final releases of vLLM exist, and when any is newer than the
pin it opens one issue, or adds to the one open, saying what to re-check: every tracked
file that mentions the pin, searched at run time, and where each contract a release
decides lives, found at run time. It changes nothing in the tree.

A release counts by its number alone. A post-release (1.20.9.post1) is a fix to a
release, not a newer one: small patches do not count (the owner, 2026-10-04). A final
release and its post-releases are one release, named by the final release when any of
its files can be installed, and otherwise by its lowest post-release that can.

Each release is announced once. The notice carries a marker per version, and a release
an issue under the title already carries a marker for, open or closed, is not announced
again, whether that marker spells it 1.21.0 or 1.21.0.post1. A closed issue is never
written to: a release after it is closed opens a new one.

A failure to fetch PyPI's answer, or to read it, exits 1, never "no news". So do
engines that disagree on the pin, a contract it cannot find, a gh call that fails, and
an issue list that is not the list of issues it reads, which must never pass for an
empty one.

Run:  python3 tools/vllm_watch.py --repo owner/name
      (needs the gh CLI, and GH_TOKEN with issues:write)
"""
import argparse
import ast
import http.client
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINES = ("index.html", "generate_report.py")
PYPI_URL = "https://pypi.org/pypi/vllm/json"
ISSUE_TITLE = "vLLM has shipped a release newer than the pinned one"
MARK = "<!-- vllm-watch announced {} -->"
MARKED = re.compile(r"<!-- vllm-watch announced (\S+) -->")
urlopen = urllib.request.urlopen   # tests answer in its place

# The contracts a vLLM release decides, by the names the engines give them. Where each
# lives is found at run time in both engines; one found in neither is an error, because
# a notice that quietly drops a contract sends its reader past it. The ROCm table's
# `checked` date is found inside that table, in contracts() below.
CONTRACTS = [
    ("`VLLM_QUANTIZATIONS`", "the quantization names each vendor's vLLM accepts",
     r"(?:const\s+)?VLLM_QUANTIZATIONS\s*="),
    ("`FP8_METHODS`", "the methods vLLM loads as FP8", r"(?:const\s+)?FP8_METHODS\s*="),
    ("`GGUF_GUIDANCE`", "what the planner says vLLM needs to serve GGUF", r"(?:const\s+)?GGUF_GUIDANCE\s*="),
    ("the FP8 refusal", "no command where vLLM cannot run FP8 weights", r"def refuse_fp8_where_vllm_cannot\("),
]
CHECKED_KEY = r"\s*[\"']checked[\"']\s*:"


class WatchError(Exception):
    """The watch could not read what it needs. Loud, never "no news"."""


# ---------------------------------------------------------------- versions

# PEP 440's public version, as packaging spells it (no local segment: PyPI refuses
# those). Written out rather than imported, so the job needs nothing pip installs.
VERSION = re.compile(r"""
    v?
    (?:(?P<epoch>[0-9]+)!)?
    (?P<release>[0-9]+(?:\.[0-9]+)*)
    (?P<pre>[-_.]?(?:alpha|a|beta|b|preview|pre|c|rc)[-_.]?[0-9]*)?
    (?P<post>-[0-9]+|[-_.]?(?:post|rev|r)[-_.]?[0-9]*)?
    (?P<dev>[-_.]?dev[-_.]?[0-9]*)?
    """, re.VERBOSE | re.IGNORECASE)


def parse(version):
    """(final, release, post) for a version. `final` is False for an alpha, beta, rc or
    dev release. `release` is its release number, the epoch and the numeric parts, which
    order releases as numbers, never as text: PyPI lists vLLM's releases sorted as
    strings (on 2026-10-02 its last key was 0.9.2 and its newest release 0.30.0), and
    vLLM ships four-part versions too (0.9.0.1, 0.10.1.1), which are releases. `post`
    is the post-release number, or -1 for a version that is not one, so a release sorts
    before its fixes.

    A post-release is not a newer release: 1.20.9.post1 has the release number of
    1.20.9, and counts for nothing beside it (the owner, 2026-10-04: small patches do
    not count). It is still a build pip installs without --pre, and the only one when
    its release was yanked: on 2026-10-02 PyPI listed 10, from 0.2.1.post1 to
    0.8.5.post1, and every file of 0.2.1 itself is yanked, so its post-release was the
    only 0.2.1 anyone could install, which newer_releases() still counts, under that
    name. A post-release of a pre-release is still a pre-release."""
    m = VERSION.fullmatch(str(version).strip())
    if not m:
        raise WatchError(f"{version!r} is not a version PEP 440 allows, so it cannot be ordered")
    release = [int(part) for part in m.group("release").split(".")]
    while len(release) > 1 and release[-1] == 0:   # 0.30 and 0.30.0 are one release
        release.pop()
    post = m.group("post")
    post_n = -1 if post is None else int(re.sub(r"[^0-9]", "", post) or 0)
    return not (m.group("pre") or m.group("dev")), (int(m.group("epoch") or 0), tuple(release)), post_n


def release_of(version):
    """The release number of a final version, whether it is spelled 1.21.0 or
    1.21.0.post1, or None for a string that is no final version. A marker in an issue
    is read with this, and one the watch did not write may say anything."""
    try:
        final, release, _ = parse(version)
    except WatchError:
        return None
    return release if final else None


def newer_releases(pin, data):
    """One version for each final release in PyPI's answer `data` that is newer than
    `pin`, oldest first. A release is newer by its number alone (see parse()), so a
    release and its post-releases are one, named by the final release when any of its
    files can be installed, and otherwise by its lowest post-release that can. A
    release none of whose versions has a file that can be installed is skipped: every
    file yanked, or none."""
    releases = data.get("releases") if isinstance(data, dict) else None
    if not isinstance(releases, dict) or not releases:
        raise WatchError("PyPI's answer lists no releases, so it says nothing about what vLLM shipped")
    final, pin_release, _ = parse(pin)
    if not final:
        raise WatchError(f"the pin {pin!r} is not a final release, and only final releases are compared")
    found = {}   # release number -> (post, version) of the version that names it
    for version, files in releases.items():
        if not isinstance(files, list) or not all(isinstance(f, dict) and isinstance(f.get("yanked"), bool)
                                                  for f in files):
            raise WatchError(f"PyPI's files for {version!r} are not a list of files each marked yanked or not")
        is_final, release, post = parse(version)
        if not is_final:
            continue
        installable = [f for f in files if not f["yanked"]]
        if not installable:
            continue
        if release > pin_release:
            found[release] = min(found.get(release, (post, version)), (post, version))   # a final's post is -1
    return [version for _, (_, version) in sorted(found.items())]


def fetch(url=PYPI_URL):
    """PyPI's JSON for vllm. Failing to fetch it or to read it raises WatchError."""
    try:
        with urlopen(url, timeout=60) as answer:
            return json.loads(answer.read().decode("utf-8"))
    except (OSError, ValueError, http.client.HTTPException) as e:
        raise WatchError(f"could not read vLLM's releases from {url}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- the tree

def rocm_tables(root=None):
    """{engine: (its ROCm table, first line, last line)}, read from each engine as
    tests/model.test.js and tests/parity.test.py find it."""
    root = root or ROOT
    with open(os.path.join(root, "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    blocks = list(re.finditer(r"^const ROCM = (\{.*?\n\});$", html, re.M | re.S))
    if len(blocks) != 1:
        raise WatchError(f"index.html: expected one `const ROCM = {{...}};`, found {len(blocks)}")
    try:
        js = json.loads(blocks[0].group(1))
    except ValueError as e:
        raise WatchError(f"index.html: its ROCM table is not the JSON this reads: {e}")
    with open(os.path.join(root, "generate_report.py"), encoding="utf-8") as fh:
        source = fh.read()
    try:
        nodes = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "ROCM" for t in n.targets)]
        if len(nodes) != 1:
            raise WatchError(f"generate_report.py: expected one `ROCM = {{...}}`, found {len(nodes)}")
        py = ast.literal_eval(nodes[0].value)
    except (SyntaxError, ValueError, TypeError) as e:
        raise WatchError(f"generate_report.py: its ROCM table is not a literal this reads: {e}")
    return {"index.html": (js, html.count("\n", 0, blocks[0].start()) + 1, html.count("\n", 0, blocks[0].end()) + 1),
            "generate_report.py": (py, nodes[0].lineno, nodes[0].end_lineno)}


def pinned(root=None):
    """The vLLM release both engines pin, as their ROCm table spells it."""
    pins = {engine: table.get("vllm") if isinstance(table, dict) else None
            for engine, (table, _, _) in rocm_tables(root).items()}
    if len(set(pins.values())) != 1:
        raise WatchError("the engines disagree on the vLLM release they pin: "
                         + ", ".join(f"{engine} says {pin!r}" for engine, pin in pins.items())
                         + ". tests/parity.test.py holds them equal; settle that first.")
    (pin,) = set(pins.values())
    if not isinstance(pin, str):
        raise WatchError(f"the engines' ROCm table pins {pin!r}, which is not a version")
    return pin


def mentions(pin, root=None):
    """[(path, lines)] for every tracked file that mentions the pin, with or without its
    `v`, found by searching the tree as it stands: never a list, which is the thing that
    goes stale."""
    root = root or ROOT
    bare = pin[1:] if pin[:1] in ("v", "V") else pin
    pattern = re.compile(r"(?<![0-9.])[vV]?" + re.escape(bare) + r"(?![0-9])")
    try:
        listed = subprocess.run(["git", "-C", root, "ls-files", "-z"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise WatchError(f"could not list the tracked files to search for {pin}: {e}")
    found = []
    for path in sorted(p for p in listed.decode("utf-8", "surrogateescape").split("\0") if p):
        try:
            with open(os.path.join(root, path), "rb") as fh:
                data = fh.read()
        except OSError:   # tracked, but not in this working tree
            continue
        if b"\0" in data:   # an image or another binary file
            continue
        hits = sum(1 for line in data.decode("utf-8", "replace").splitlines() if pattern.search(line))
        if hits:
            found.append((path, hits))
    if not found:
        raise WatchError(f"no tracked file mentions {pin}, not even the engines it was read from")
    return found


def contracts(root=None):
    """[(name, what, [(engine, line)])] for each contract a release decides, found at
    run time. Raises WatchError for one neither engine defines."""
    root = root or ROOT
    text = {}
    for engine in ENGINES:
        with open(os.path.join(root, engine), encoding="utf-8") as fh:
            text[engine] = fh.read().split("\n")
    found = [(name, what, [(engine, n) for engine in ENGINES
                           for n, line in enumerate(text[engine], start=1) if re.match(pattern, line)])
             for name, what, pattern in CONTRACTS]
    checked, dates = [], set()
    for engine, (table, first, last) in rocm_tables(root).items():
        hits = [n for n in range(first, last + 1) if re.match(CHECKED_KEY, text[engine][n - 1])]
        if len(hits) != 1:
            raise WatchError(f"{engine}: expected one `checked` in its ROCm table, found {len(hits)}")
        checked.append((engine, hits[0]))
        dates.add(str(table.get("checked")))
    found.insert(2, ("the ROCm table's `checked`",
                     f"the date its lines were checked against the pin ({', '.join(sorted(dates))})", checked))
    missing = [name for name, _, sites in found if not sites]
    if missing:
        raise WatchError(f"cannot say where {', '.join(missing)} lives: neither engine defines it")
    return found


def notice(pin, newer, fresh, files, where):
    """The issue body or comment that announces `fresh`."""
    lines = [f"vLLM has shipped {', '.join(f'`{v}`' for v in fresh)}, newer than the planner's pin, `{pin}`."]
    if newer != fresh:
        lines.append(f"Every final release newer than the pin: {', '.join(f'`{v}`' for v in newer)}.")
    lines += ["", "Nothing has been changed. Before the pin moves, re-check:", "",
              f"**Every tracked file that mentions `{pin}`**, as a search of the tree found them:"]
    lines += [f"- `{path}` ({n} line{'' if n == 1 else 's'})" for path, n in files]
    lines += ["", "**Where each contract the release decides lives:**"]
    lines += [f"- {name}, {what}: " + ", ".join(f"`{engine}:{n}`" for engine, n in sites)
              for name, what, sites in where]
    lines += ["", "Posted by `.github/workflows/vllm-watch.yml` (`tools/vllm_watch.py`)."]
    lines += [MARK.format(v) for v in fresh]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- GitHub, through gh

def gh(*args, check=True):
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=check)


def gh_json(*args):
    out = gh(*args).stdout
    try:
        return json.loads(out)
    except ValueError as e:
        raise WatchError(f"gh {' '.join(args[:2])} did not answer JSON: {e}")


# What announce() reads of each issue `gh issue list` answers, and the type JSON gives it.
ISSUE_FIELDS = (("number", int), ("title", str), ("state", str), ("body", str), ("comments", list))


def check_issue_list(listed):
    """Raise WatchError unless `listed`, the JSON `gh issue list` answered, is the list
    announce() reads: objects each with a number, title, state and body, and comments
    that are a list of objects with a body. A list it cannot read must never pass for an
    empty one: with no issue seen, the week would open a second issue beside the one
    that is open, and an answer of another shape would crash with a traceback instead."""
    if not isinstance(listed, list):
        raise WatchError(f"gh issue list answered {listed!r:.60}, not a list of issues")
    for at, item in enumerate(listed):
        if not isinstance(item, dict):
            raise WatchError(f"gh issue list answered {item!r:.60} as its item {at}, not an issue")
        for field, kind in ISSUE_FIELDS:
            if field not in item:
                raise WatchError(f"gh issue list answered its item {at} without its {field!r}")
            if type(item[field]) is not kind:
                raise WatchError(f"gh issue list answered its item {at} with {field!r} as {item[field]!r:.60}, "
                                 f"not {kind.__name__}")
        for comment in item["comments"]:
            if not isinstance(comment, dict) or type(comment.get("body")) is not str:
                raise WatchError(f"gh issue list answered its item {at} with the comment {comment!r:.60}, "
                                 f"not an object with a body")


def announce(repo, pin, newer, files, where):
    """Announce each release in `newer` that no issue under the title has announced, by
    its release number and not the spelling of its version: a release told as 1.21.0 is
    not told again as 1.21.0.post1, nor the reverse. The announcement is a comment on
    the open issue, or a new issue when none is open. Returns the versions it
    announced."""
    listed = gh_json("issue", "list", "--repo", repo, "--state", "all", "--search", f'in:title "{ISSUE_TITLE}"',
                     "--json", "number,title,state,body,comments", "--limit", "200")
    check_issue_list(listed)
    ours = [i for i in listed if i.get("title") == ISSUE_TITLE]
    told = {release_of(version) for issue in ours
            for text in [issue.get("body") or ""] + [c.get("body") or "" for c in issue.get("comments") or []]
            for version in MARKED.findall(text)}   # None for a marker that names no final release
    fresh = [v for v in newer if release_of(v) not in told]
    if not fresh:
        return []
    open_issues = sorted(i["number"] for i in ours if str(i.get("state", "")).lower() == "open")
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as fh:
        fh.write(notice(pin, newer, fresh, files, where))
    try:
        if open_issues:
            gh("issue", "comment", str(open_issues[0]), "--repo", repo, "--body-file", fh.name)
        else:
            gh("issue", "create", "--repo", repo, "--title", ISSUE_TITLE, "--body-file", fh.name)
    finally:
        os.unlink(fh.name)
    return fresh


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--pypi-url", default=PYPI_URL,
                    help="where to read vLLM's releases; the tests point it at a file:// fixture")
    args = ap.parse_args(argv)
    try:
        pin = pinned()
        files, where = mentions(pin), contracts()
        newer = newer_releases(pin, fetch(args.pypi_url))
        if not newer:
            print(f"vllm watch: no final release of vLLM is newer than the pinned {pin}.")
            return 0
        fresh = announce(args.repo, pin, newer, files, where)
    except WatchError as e:
        print(f"vllm watch: {e}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(f"vllm watch: {' '.join(map(str, e.cmd[:3]))} failed: {(e.stderr or '').strip()[-300:]}",
              file=sys.stderr)
        return 1
    if fresh:
        print(f"vllm watch: announced {', '.join(fresh)}, newer than the pinned {pin}.")
    else:
        print(f"vllm watch: {', '.join(newer)} newer than the pinned {pin}, every one already announced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
