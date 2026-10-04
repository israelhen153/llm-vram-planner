#!/usr/bin/env python3
"""The vLLM release watch (tools/vllm_watch.py): the pin it reads, the releases it
counts and how it orders them, what it tells GitHub and how often, what it does when
it cannot read PyPI, and what its notice sends the owner to re-check.

The watch is only as good as what this file holds. It must read the pin from both
engines and refuse when they disagree; count final, installable releases only, in
numeric order, a release and its post-releases as one; announce each release once,
on one open issue, whichever way its version is spelled; exit 1 whenever it could
not read the answer, because "no news" is what a broken watch looks like; and find
the files and contracts its notice names by searching the tree it runs in.

No test here reaches the network or GitHub. PyPI is answered from fixtures, a proxy
that refuses every connection is set for anything that slips past, and a fake gh
comes first on PATH for the whole run.

Run:  python3 tests/vllm_watch.test.py
"""
import contextlib
import http.client
import io
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
import urllib.error
import uuid

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import vllm_watch as vw  # noqa: E402

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


# GitHub, as gh would answer it, from a state it updates: open issues unless asked for
# all, the body of a new issue or comment read from its --body-file. A state with
# "list_says" has `issue list` answer that text, whatever it is, for gh answering
# something that is not JSON. Used in-process, and as the gh executable on PATH for the
# real process and anything that slips past.
FAKE_GH_LOGIC = '''
import json


def answer(state, a):
    """(exit code, stdout) of `gh *a` from state."""
    def flag(name, default=None):
        return a[a.index(name) + 1] if name in a else default
    if a[:2] == ["issue", "list"]:
        if "list_says" in state:
            return 0, state["list_says"]
        want = flag("--state", "open")
        return 0, json.dumps([i for i in state["issues"] if want == "all" or i["state"].lower() == want])
    if a[:2] in (["issue", "create"], ["issue", "comment"]):
        with open(flag("--body-file")) as fh:
            body = fh.read()
        if a[1] == "create":
            number = max([i["number"] for i in state["issues"]] + [0]) + 1
            state["issues"].append({"number": number, "title": flag("--title"), "state": "OPEN",
                                    "body": body, "comments": []})
        else:
            (issue,) = [i for i in state["issues"] if str(i["number"]) == a[2]]
            issue["comments"].append({"body": body})
        return 0, ""
    return 1, "unexpected " + " ".join(a)
'''
_fake = {}
exec(FAKE_GH_LOGIC, _fake)
FAKE_GH = "#!/usr/bin/env python3" + FAKE_GH_LOGIC + '''
import os, sys
path = os.environ["FAKE_GH_STATE"]
with open(path) as fh:
    state = json.load(fh)
code, out = answer(state, sys.argv[1:])
state.setdefault("calls", []).append(sys.argv[1:])
with open(path, "w") as fh:
    json.dump(state, fh)
print(out)
sys.exit(code)
'''

BASE = tempfile.TemporaryDirectory()
os.makedirs(os.path.join(BASE.name, "bin"))
with open(os.path.join(BASE.name, "bin", "gh"), "w") as _fh:
    _fh.write(FAKE_GH)
os.chmod(os.path.join(BASE.name, "bin", "gh"), stat.S_IRWXU)
os.environ["PATH"] = os.path.join(BASE.name, "bin") + os.pathsep + os.environ.get("PATH", "")
for _var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ[_var] = "http://127.0.0.1:9"
for _var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(_var, None)


def engine_pins(root=ROOT):
    """Each engine's pin, read here without the watch: the `vllm` entry of its ROCm table."""
    with open(os.path.join(root, "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    with open(os.path.join(root, "generate_report.py"), encoding="utf-8") as fh:
        py = fh.read()
    table = re.search(r"^const ROCM = \{[\s\S]*?\n\};$", html, re.M).group(0)
    return {"index.html": re.search(r'"vllm":\s*"([^"]+)"', table).group(1),
            "generate_report.py": re.search(r"'vllm':\s*'([^']+)'", py[py.index("\nROCM = {"):]).group(1)}


def real_pin():
    pins = engine_pins()
    assert len(set(pins.values())) == 1, f"the engines in this tree disagree: {pins}"
    return pins["index.html"]


def later(pin, n=1):
    """The release n patch releases after `pin`, so no test depends on today's pin."""
    parts = [int(p) for p in pin.lstrip("vV").split(".")]
    parts[-1] += n
    return ".".join(map(str, parts))


def pypi(releases):
    """PyPI's JSON for vllm from {version: [whether each file is yanked]}."""
    return {"info": {"name": "vllm"},
            "releases": {v: [{"filename": f"vllm-{v}-{n}.whl", "yanked": y} for n, y in enumerate(flags)]
                         for v, flags in releases.items()}}


def tree(edit=None, extra=None):
    """A git work tree holding both engines, copied from this one through
    edit(engine, text), and `extra` {path: text}, every file tracked."""
    d = tempfile.mkdtemp(dir=BASE.name)
    files = dict(extra or {})
    for engine in vw.ENGINES:
        with open(os.path.join(ROOT, engine), encoding="utf-8") as fh:
            files[engine] = edit(engine, fh.read()) if edit else fh.read()
    for path, text in files.items():
        os.makedirs(os.path.dirname(os.path.join(d, path)), exist_ok=True)
        with open(os.path.join(d, path), "w", encoding="utf-8") as fh:
            fh.write(text)
    subprocess.run(["git", "init", "-q", "--template=", d], check=True)
    subprocess.run(["git", "-C", d, "add", "-A"], check=True)
    return d


def issue(number, state, body="", comments=(), title=vw.ISSUE_TITLE):
    return {"number": number, "title": title, "state": state, "body": body,
            "comments": [{"body": c} for c in comments]}


def run_main(state, answer, root=None, argv=()):
    """vw.main() in this process: PyPI answers `answer` (JSON, raw bytes, or an
    exception it raises), gh answers from `state`. (exit code, printed, the issue
    writes it made, the URLs it fetched)."""
    calls, urls = [], []

    def gh(*args, check=True):
        calls.append(list(args))
        code, out = _fake["answer"](state, list(args))
        if check and code:
            raise subprocess.CalledProcessError(code, ["gh", *args], out, out)
        return subprocess.CompletedProcess(["gh", *args], code, out + "\n", "")

    def urlopen(url, timeout=None):
        urls.append(url)
        if isinstance(answer, BaseException):
            raise answer
        return io.BytesIO(answer if isinstance(answer, bytes) else json.dumps(answer).encode())

    saved = vw.gh, vw.urlopen, vw.ROOT
    vw.gh, vw.urlopen, vw.ROOT = gh, urlopen, root or vw.ROOT
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
            code = vw.main(["--repo", "o/r", *argv])
    finally:
        vw.gh, vw.urlopen, vw.ROOT = saved
    writes = [c[:3] if c[1] == "comment" else c[:2] for c in calls if c[:1] == ["issue"] and c[1] != "list"]
    return code, printed.getvalue(), writes, urls


print("\nThe pin, read from both engines")


def check_the_pin_is_read_from_both_engines():
    """The pin is the `vllm` entry of each engine's ROCm table, read where it is set:
    in this tree, and in a tree where both engines pin another release."""
    pin = real_pin()
    assert vw.pinned() == pin, f"the engines pin {pin}, and the watch read {vw.pinned()}"
    other = "v" + later(pin, 7)
    moved = tree(lambda engine, text: text.replace(pin, other))
    assert vw.pinned(moved) == other, f"both engines pin {other}, and the watch read {vw.pinned(moved)}"

test("the pin is read from both engines' ROCm table, whatever it is", check_the_pin_is_read_from_both_engines)


def check_engines_that_disagree_are_refused():
    """Either engine moved alone is refused, by name, and main() exits 1 having
    written nothing: a pin taken from one engine plans the other's commands against a
    release nobody checked."""
    pin = real_pin()
    other = "v" + later(pin, 7)
    for engine in vw.ENGINES:
        d = tree(lambda e, text: text.replace(pin, other) if e == engine else text)
        try:
            got = vw.pinned(d)
        except vw.WatchError as e:
            assert pin in str(e) and other in str(e), f"the refusal does not name both pins: {e}"
        else:
            raise AssertionError(f"{engine} pins {other} and the other engine {pin}, and the watch read {got}")
        state = {"issues": []}
        code, printed, writes, _ = run_main(state, pypi({later(pin): [False]}), root=d)
        assert code == 1 and "disagree" in printed and not writes, (engine, code, writes, printed[-200:])

test("engines that disagree on the pin are refused, and main() exits 1", check_engines_that_disagree_are_refused)


print("\nThe releases it counts, and their order")


def check_only_final_installable_releases_count():
    """Alpha, beta, rc and dev releases are not final, and neither is a post-release
    of one. A release whose files are all yanked, or which has none, cannot be
    installed. A post-release of the pin's own release is not newer (see parse()), and
    a release with one file yanked and another not counts."""
    data = pypi({"1.21.0rc1": [False], "1.21.0a2": [False], "1.21.0b1": [False], "1.21.0.dev0": [False],
                 "1.21.0rc1.post1": [False], "1.21.1.dev2": [False],
                 "1.21.1": [True, True], "1.21.2": [],
                 "1.21.3": [True, False], "1.20.9.post1": [False], "1.20.9": [False], "1.20.8": [False]})
    got = vw.newer_releases("v1.20.9", data)
    assert got == ["1.21.3"], f"newer than v1.20.9: {got}"

test("only final, installable releases count: no pre-release, dev release or yanked release",
     check_only_final_installable_releases_count)


def check_versions_are_compared_as_numbers():
    """1.20.10 is newer than 1.20.9 and 1.9.30 is not, which text compares the other
    way round; trailing zeros and a fourth part are numbers too. PyPI lists vLLM's
    releases as strings sort them, so the order it gives proves nothing."""
    data = pypi({v: [False] for v in ("1.20.10", "1.9.30", "2.0", "1.20.9", "1.100.0", "1.3.0.post2",
                                      "1.20", "1.20.9.1", "1.20.9.0")})
    got = vw.newer_releases("v1.20.9", data)
    assert got == ["1.20.9.1", "1.20.10", "1.100.0", "2.0"], f"newer than v1.20.9, oldest first: {got}"

test("versions are compared as numbers, never as text", check_versions_are_compared_as_numbers)


def in_both_orders(releases):
    """PyPI's JSON for {version: [whether each file is yanked]}, listed as given and
    in reverse: PyPI sorts its list as strings, so no result may depend on its order."""
    return [pypi(releases), pypi(dict(reversed(list(releases.items()))))]


def check_a_release_and_its_post_releases_are_one_release():
    """Small patches do not count (the owner, 2026-10-04): a post-release has its
    release's number, so it is never newer than the pin's own release, and beside a
    newer release it adds nothing: the release is named, not its fixes. A four-part
    version is a release, with post-releases of its own; and a pin that is itself a
    post-release is compared by its release number."""
    cases = [("v1.20.9", {"1.21.0": [False], "1.21.0.post1": [False], "1.21.0.post2": [False],
                          "1.20.9": [False], "1.20.9.post1": [False]}, ["1.21.0"]),
             ("v0.9.0", {"0.9.0": [False], "0.9.0.post1": [False], "0.9.0.1": [False],
                         "0.9.0.1.post1": [False]}, ["0.9.0.1"]),
             ("v1.20.9.post1", {"1.20.9": [False], "1.20.9.post1": [False], "1.20.9.post2": [False],
                                "1.21.0": [False]}, ["1.21.0"])]
    wrong = {(pin, tuple(data["releases"])): got for pin, releases, want in cases
             for data in in_both_orders(releases) for got in [vw.newer_releases(pin, data)] if got != want}
    assert not wrong, f"newer than the pin, wrongly: {wrong}"

test("a release and its post-releases are one release: the release is named, its fixes are not",
     check_a_release_and_its_post_releases_are_one_release)


def check_a_release_known_only_by_a_post_release_still_counts():
    """vLLM's 0.2.1 had every file yanked, and 0.2.1.post1 was the only 0.2.1 anyone
    could install: such a release still counts, under the name of its lowest
    post-release that can be installed. A release with nothing installable under any
    of its names does not count, and one whose final release can be installed is
    named by it, with a post-release listed beside it."""
    releases = {"1.21.0": [True, True], "1.21.0.post1": [True], "1.21.0.post2": [False], "1.21.0.post3": [False],
                "1.21.1": [True], "1.21.1.post1": [True],
                "1.21.2": [True], "1.21.2.post1": [False],
                "1.21.3": [False], "1.21.3.post1": [False],
                "1.20.9": [False]}
    want = ["1.21.0.post2", "1.21.2.post1", "1.21.3"]
    wrong = {tuple(data["releases"]): got for data in in_both_orders(releases)
             for got in [vw.newer_releases("v1.20.9", data)] if got != want}
    assert not wrong, f"wanted {want}, and in these orders it gave: {wrong}"

test("a release whose final is yanked, with an installable post-release, counts under the post-release's name",
     check_a_release_known_only_by_a_post_release_still_counts)


print("\nWhat it tells GitHub, and how often")


def check_no_newer_release_writes_nothing():
    """The pin itself, a newer pre-release, a newer yanked release and an older one:
    nothing to say, so no issue, no comment, and a green run. It asked PyPI's real
    address for vllm, a contract, so a literal."""
    assert vw.PYPI_URL == "https://pypi.org/pypi/vllm/json", vw.PYPI_URL
    pin = real_pin()
    state = {"issues": [issue(4, "OPEN", "earlier")]}
    data = pypi({pin.lstrip("vV"): [False], later(pin) + "rc1": [False], later(pin, 2): [True], "0.0.1": [False]})
    code, printed, writes, urls = run_main(state, data)
    assert code == 0 and not writes, f"nothing newer, and it exited {code} having made {writes}: {printed[-200:]}"
    assert urls == [vw.PYPI_URL], urls

test("no newer release: no issue, no comment, exit 0", check_no_newer_release_writes_nothing)


def check_an_open_issue_gets_a_comment():
    """With the issue open, a new release is a comment on it, never a second issue;
    not on a closed one, and not on an open one under another title."""
    pin = real_pin()
    state = {"issues": [issue(3, "CLOSED", "old"), issue(7, "OPEN", "earlier"),
                        issue(9, "OPEN", "someone else's", title="vLLM questions")]}
    code, printed, writes, _ = run_main(state, pypi({later(pin): [False]}))
    assert code == 0 and writes == [["issue", "comment", "7"]], f"exit {code}, wrote {writes}: {printed[-200:]}"
    (said,) = state["issues"][1]["comments"]
    assert f"`{later(pin)}`" in said["body"] and pin in said["body"], said["body"][:300]

test("an open issue gets a comment, never a second issue", check_an_open_issue_gets_a_comment)


def check_a_release_is_announced_once():
    """Week by week: a release opens the issue; the same answer again says nothing; a
    second release is one comment, naming it; once the issue is closed the same answer
    still says nothing, and a third release opens a new issue, leaving the closed one
    alone. A marker under another title announces nothing."""
    pin = real_pin()
    first, second, third = later(pin), later(pin, 2), later(pin, 3)
    state = {"issues": [issue(1, "OPEN", vw.MARK.format(first), title="vLLM questions")]}
    weeks = [({first: [False]}, [["issue", "create"]]),
             ({first: [False]}, []),
             ({first: [False], second: [False]}, [["issue", "comment", "2"]]),
             ({first: [False], second: [False]}, [])]
    for week, (releases, want) in enumerate(weeks, start=1):
        code, printed, writes, _ = run_main(state, pypi(releases))
        assert code == 0 and writes == want, f"week {week}: exit {code}, wrote {writes}, wanted {want}: {printed[-200:]}"
    ours = state["issues"][1]
    assert second in ours["comments"][-1]["body"] and vw.MARK.format(first) not in ours["comments"][-1]["body"], (
        f"the comment does not announce {second} alone: {ours['comments'][-1]['body'][-300:]}")
    ours["state"] = "CLOSED"
    code, _, writes, _ = run_main(state, pypi({first: [False], second: [False]}))
    assert code == 0 and writes == [], f"after the issue was closed, the same releases wrote {writes}"
    code, _, writes, _ = run_main(state, pypi({first: [False], second: [False], third: [False]}))
    assert code == 0 and writes == [["issue", "create"]] and len(ours["comments"]) == 1, (
        f"a third release with the issue closed wrote {writes}, and the closed issue has {len(ours['comments'])} comments")

test("each release is announced once, open issue or closed", check_a_release_is_announced_once)


def check_a_release_is_told_once_under_either_name():
    """A release and its post-release are one release, so one told as X is not told
    again as X.post1, nor the reverse: the final yanked after it was announced, leaving
    its post-release; or the post-release announced while the final was yanked, and the
    final restored. Asserted on the writes it makes, from the markers it wrote itself.
    A newer release is still told, and a marker that names no final release (a
    pre-release, or text that is no version) tells nothing and breaks nothing."""
    pin = real_pin()
    rel, post, newest = later(pin), later(pin) + ".post1", later(pin, 2)
    cases = {"announced as the release, then only its post-release installs":
                 (rel, {rel: [False]}, {rel: [True], post: [False]}),
             "announced as the post-release, then the release installs":
                 (post, {rel: [True], post: [False]}, {rel: [False], post: [False]})}
    for case, (named, first, then) in cases.items():
        state = {"issues": []}
        code, printed, writes, _ = run_main(state, pypi(first))
        assert code == 0 and writes == [["issue", "create"]] and vw.MARK.format(named) in state["issues"][0]["body"], (
            f"{case}: week 1 exit {code}, wrote {writes}, named {named}: {printed[-200:]}")
        code, printed, writes, _ = run_main(state, pypi(then))
        assert code == 0 and writes == [] and "already announced" in printed, (
            f"{case}: week 2 exit {code}, wrote {writes}: {printed[-200:]}")
        code, printed, writes, _ = run_main(state, pypi({**then, newest: [False]}))
        said = (state["issues"][0]["comments"] or [{"body": ""}])[-1]["body"]
        assert code == 0 and writes == [["issue", "comment", "1"]] and vw.MARK.format(newest) in said and (
            vw.MARK.format(named) not in said), f"{case}: week 3 exit {code}, wrote {writes}: {said[-300:]}"
    stray = [vw.MARK.format("latest"), vw.MARK.format(rel + "rc1")]
    state = {"issues": [issue(3, "OPEN", "earlier", comments=stray)]}
    code, printed, writes, _ = run_main(state, pypi({rel: [False]}))
    assert code == 0 and writes == [["issue", "comment", "3"]], (
        f"markers that name no final release: exit {code}, wrote {writes}: {printed[-200:]}")

test("a release is told once, under either name: the release, or its post-release",
     check_a_release_is_told_once_under_either_name)


print("\nWhen PyPI cannot be read")


def check_an_unread_answer_exits_1():
    """Every way the answer can fail to arrive or fail to read exits 1 and writes
    nothing: an empty answer read as "no news" is the silence this job exists to end."""
    pin = real_pin()
    new = later(pin)
    cases = {
        "PyPI unreachable": urllib.error.URLError("no route to host"),
        "HTTP 503": urllib.error.HTTPError(vw.PYPI_URL, 503, "Service Unavailable", {}, None),
        "a timeout": TimeoutError("timed out"),
        "a reset connection": ConnectionResetError("reset by peer"),
        "a truncated answer": http.client.IncompleteRead(b'{"releases": {'),
        "an answer that is not JSON": b"<html>busy</html>",
        "an answer that is not UTF-8": b"\xff\xfe\x00",
        "JSON without releases": {"info": {"name": "vllm"}},
        "no releases at all": {"releases": {}},
        "releases that are not a mapping": {"releases": [new]},
        "a release whose files are not a list": {"releases": {new: {"yanked": False}}},
        "a file without its yanked flag": {"releases": {new: [{"filename": "x.whl"}]}},
        "a version PEP 440 does not allow": {"releases": {"latest": [{"yanked": False}], new: [{"yanked": False}]}},
    }
    wrong = {}
    for case, answer in cases.items():
        state = {"issues": [issue(5, "OPEN", "earlier")]}
        code, printed, writes, _ = run_main(state, answer)
        if code != 1 or writes:
            wrong[case] = (code, writes, printed.strip()[-120:])
    assert not wrong, wrong

test("a fetch or parse error exits 1, never 'no news'", check_an_unread_answer_exits_1)


print("\nWhen GitHub cannot be read")


def check_an_issue_list_that_is_not_json_exits_1():
    """gh's `issue list` answering something that is not JSON (an HTML error page,
    nothing at all, a list cut off, a line of text) is a list it could not read, not an
    empty one: main() exits 1 and writes nothing, no comment and no issue. Read as
    empty, the week would open a second issue beside the one already open."""
    new = later(real_pin())
    answers = {"an HTML error page": "<html><body><h1>502 Bad Gateway</h1></body></html>",
               "nothing at all": "",
               "a list cut off": '[{"number": 5, "title": "vLLM has',
               "a line of text": "HTTP 401: Bad credentials"}
    wrong = {}
    for case, text in answers.items():
        state = {"issues": [issue(5, "OPEN", "earlier")], "list_says": text}
        code, printed, writes, _ = run_main(state, pypi({new: [False]}))
        if code != 1 or writes or "did not answer JSON" not in printed:
            wrong[case] = (code, writes, printed.strip()[-120:])
    assert not wrong, wrong

test("an issue list that is not JSON exits 1 and writes nothing, never an empty list",
     check_an_issue_list_that_is_not_json_exits_1)


def check_an_issue_list_of_the_wrong_shape_exits_1():
    """JSON that is not the list of issues announce() reads is a list it cannot read,
    never an empty one: objects each with a number, title, state and body, and comments
    that are a list of objects with a body. {} and "" read as no issue open would open a
    second issue beside the one that is, and null, [1] and gh's own error object would
    crash with a traceback. Each of them, an issue missing any field the watch reads and
    a malformed comment must give exit 1, the watch's own message and no write. The
    fields are a literal here: they are what announce() reads, not what the check says."""
    new = later(real_pin())
    good = issue(5, "OPEN", "earlier", comments=["a comment"])
    cases = {"an empty object": "{}", "an empty string": '""', "null": "null", "a list of numbers": "[1]",
             "gh's error object": '{"message": "Bad credentials"}', "an issue that is a string": '["issue"]',
             "a number that is text": json.dumps([{**good, "number": "5"}]),
             "comments that are not a list": json.dumps([{**good, "comments": {"totalCount": 1}}]),
             "a comment that is not an object": json.dumps([{**good, "comments": ["text"]}]),
             "a comment without its body": json.dumps([{**good, "comments": [{"id": "IC_1"}]}])}
    for field in ("number", "title", "state", "body", "comments"):
        cases[f"an issue without its {field}"] = json.dumps([{k: v for k, v in good.items() if k != field}])
    wrong = {}
    for case, text in cases.items():
        try:
            code, printed, writes, _ = run_main({"issues": [good], "list_says": text}, pypi({new: [False]}))
        except Exception as e:   # a crash is not the watch's own message
            wrong[case] = f"crashed: {type(e).__name__}: {e}"
            continue
        if code != 1 or writes or "vllm watch: gh issue list answered" not in printed:
            wrong[case] = (code, writes, printed.strip()[-120:])
    assert not wrong, wrong
    rich = {**good, "url": "https://github.com/o/r/issues/5", "labels": [],
            "comments": [{"body": "a comment", "author": {"login": "someone"}, "createdAt": "2026-10-01T00:00:00Z"}]}
    code, printed, writes, _ = run_main({"issues": [rich]}, pypi({new: [False]}))
    assert code == 0 and writes == [["issue", "comment", "5"]], (
        f"an answer carrying more than the watch reads, as gh's does: exit {code}, wrote {writes}: {printed[-200:]}")

test("an issue list of the wrong shape exits 1 and writes nothing: {}, null, a missing field, a bad comment",
     check_an_issue_list_of_the_wrong_shape_exits_1)


print("\nWhat the notice sends the owner to re-check")


def check_the_files_are_found_by_search():
    """Every tracked file that mentions the pin, as git grep finds them in this tree;
    and in a tree holding a file no list could name, that file too, in the notice
    main() writes, and not a file that does not mention the pin."""
    pin = real_pin()
    bare = re.escape(pin.lstrip("vV"))
    grep = subprocess.run(["git", "-C", ROOT, "grep", "-l", "-I", "-E", f"(^|[^0-9.])[vV]?{bare}([^0-9]|$)"],
                          capture_output=True, text=True, check=True).stdout.split()
    got = [path for path, _ in vw.mentions(pin)]
    assert got == sorted(grep), f"the watch found {got}, git grep {sorted(grep)}"
    name = f"notes-{uuid.uuid4().hex[:8]}/{uuid.uuid4().hex[:8]}.md"
    d = tree(extra={name: f"Pinned at {pin} until the next check.\n", "unrelated.md": "Nothing pinned here.\n"})
    got = [path for path, _ in vw.mentions(pin, d)]
    assert got == sorted(["generate_report.py", "index.html", name]), got
    state = {"issues": []}
    code, printed, writes, _ = run_main(state, pypi({later(pin): [False]}), root=d)
    assert code == 0 and writes == [["issue", "create"]], (code, writes, printed[-200:])
    body = state["issues"][0]["body"]
    assert f"`{name}`" in body and "unrelated.md" not in body, body[:600]

test("the files to re-check are found by searching the tree, not listed",
     check_the_files_are_found_by_search)


# A contract, so it is a literal: what each contract is called in the notice, and the
# text its line must carry where the watch says it lives.
CONTRACT_LINES = {"`VLLM_QUANTIZATIONS`": "VLLM_QUANTIZATIONS", "`FP8_METHODS`": "FP8_METHODS",
                  "the ROCm table's `checked`": "checked", "`GGUF_GUIDANCE`": "GGUF_GUIDANCE",
                  "the FP8 refusal": "def refuse_fp8_where_vllm_cannot("}
# And in which engines each lives, written out from the engines as they are and never
# from the patterns the watch searches by: a watch that searched one engine for a
# contract both define would still find it, and send its reader to one copy. Today
# index.html defines neither VLLM_QUANTIZATIONS nor FP8_METHODS, and has no such def.
CONTRACT_ENGINES = {"`VLLM_QUANTIZATIONS`": {"generate_report.py"}, "`FP8_METHODS`": {"generate_report.py"},
                    "the ROCm table's `checked`": {"index.html", "generate_report.py"},
                    "`GGUF_GUIDANCE`": {"index.html", "generate_report.py"},
                    "the FP8 refusal": {"generate_report.py"}}


def check_each_contract_is_found_where_it_lives():
    """Each contract, at a file:line that carries it, and in exactly the engines that
    define it, which is both for GGUF_GUIDANCE and the `checked` date of the ROCm table
    and generate_report.py alone for the rest; lines that move when the engines grow; a
    contract that has gone refused; and every site in the notice main() writes."""
    def lines_of(root, engine):
        with open(os.path.join(root, engine), encoding="utf-8") as fh:
            return fh.read().split("\n")
    where = vw.contracts()
    assert [name for name, _, _ in where] == list(CONTRACT_LINES), [name for name, _, _ in where]
    for name, _, sites in where:
        assert sites, f"{name} was found nowhere"
        for engine, n in sites:
            assert CONTRACT_LINES[name] in lines_of(ROOT, engine)[n - 1], f"{name} is not at {engine}:{n}"
    sites = dict((name, s) for name, _, s in where)
    assert list(CONTRACT_ENGINES) == list(CONTRACT_LINES), "the two literals name different contracts"
    wrong = {name: (sorted({e for e, _ in sites[name]}), sorted(want)) for name, want in CONTRACT_ENGINES.items()
             if {e for e, _ in sites[name]} != want}
    assert not wrong, f"contract: (the engines it was found in, the engines that define it): {wrong}"
    grown =tree(lambda engine, text: "\n" * 5 + text)
    moved = vw.contracts(grown)
    assert [(name, [(e, n + 5) for e, n in s]) for name, _, s in where] == [(name, s) for name, _, s in moved], moved
    gone = tree(lambda engine, text: text.replace("\nFP8_METHODS = ", "\nFP8_KINDS = "))
    try:
        vw.contracts(gone)
    except vw.WatchError as e:
        assert "FP8_METHODS" in str(e), e
    else:
        raise AssertionError("FP8_METHODS is defined nowhere, and the watch did not refuse")
    state = {"issues": []}
    pin = real_pin()
    code, _, writes, _ = run_main(state, pypi({later(pin): [False]}))
    body = state["issues"][0]["body"] if state["issues"] else ""
    missing = [f"{e}:{n}" for _, _, s in where for e, n in s if f"`{e}:{n}`" not in body]
    assert code == 0 and not missing, f"exit {code}; the notice lacks {missing}"

test("each contract is found where it lives, at run time, and named in the notice",
     check_each_contract_is_found_where_it_lives)


print("\nThe watch as a process")


def check_the_watch_as_a_process():
    """The real script, as the workflow starts it, with gh on PATH answering from a file
    and PyPI from a file:// fixture: a newer release opens one issue and exits 0; an
    answer that cannot be fetched exits 1 and writes nothing."""
    pin = real_pin()
    fixture = os.path.join(BASE.name, "pypi.json")
    with open(fixture, "w") as fh:
        json.dump(pypi({later(pin): [False]}), fh)
    state_path = os.path.join(BASE.name, "state.json")
    with open(state_path, "w") as fh:
        json.dump({"issues": []}, fh)

    def call(url):
        done = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "vllm_watch.py"), "--repo", "o/r",
                               "--pypi-url", url], capture_output=True, text=True, timeout=60,
                              env=dict(os.environ, FAKE_GH_STATE=state_path))
        with open(state_path) as fh:
            return done.returncode, json.load(fh), done.stdout + done.stderr

    rc, state, out = call(pathlib.Path(fixture).as_uri())
    assert rc == 0 and len(state["issues"]) == 1 and f"`{later(pin)}`" in state["issues"][0]["body"], (rc, out[-300:])
    rc, state, out = call(pathlib.Path(fixture + ".missing").as_uri())
    writes = [c for c in state["calls"] if c[:2] in (["issue", "create"], ["issue", "comment"])]
    assert rc == 1 and len(writes) == 1, f"an unfetchable answer: exit {rc}, writes {writes}: {out[-300:]}"

test("the watch as a process: one issue for a newer release, exit 1 when PyPI cannot be read",
     check_the_watch_as_a_process)


print("\nThe watch's own workflow")

# A contract, so it is a literal: the watch's workflow, whole. An action may move to a
# newer version; nothing else changes without this.
WATCH_TOP_KEYS = {"name", "on", "permissions", "jobs"}
WATCH_JOB_KEYS = {"runs-on", "permissions", "steps"}
WORKFLOW_PERMISSIONS = {"contents": "read"}
WATCH_PERMISSIONS = {"contents": "read", "issues": "write"}
WATCH_STEPS = [
    {"uses": "actions/checkout"},
    {"uses": "actions/setup-python", "with": {"python-version": "3.x"}},
    {"name": "Compare vLLM's releases with the pinned one",
     "env": {"GH_TOKEN": "${{ github.token }}"},
     "run": 'python3 tools/vllm_watch.py --repo "$GITHUB_REPOSITORY"'},
]
ACTION_VERSION = r"@(v[0-9]+(\.[0-9]+){0,2}|[0-9a-f]{40})"
# Cron's weekday names, Sunday first ("0 - 6 or SUN-SAT", GitHub's cron table).
CRON_DAYS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]
WEEK = 7 * 24 * 60


def workflows():
    """{file: parsed} for every workflow file."""
    import yaml
    where = os.path.join(ROOT, ".github", "workflows")
    found = {}
    for name in sorted(os.listdir(where)):
        if name.endswith((".yml", ".yaml")):
            with open(os.path.join(where, name), encoding="utf-8") as fh:
                found[name] = yaml.safe_load(fh)
    return found


def schedule_of(doc):
    on = doc.get(True, doc.get("on"))
    return (on.get("schedule") or []) if isinstance(on, dict) else []


def minute_of_week(cron):
    """When a once-a-week cron fires, in minutes from Sunday 00:00 UTC."""
    minute, hour, dom, month, dow = cron.split()
    assert dom == month == "*" and minute.isdigit() and hour.isdigit(), f"{cron!r} is not `minute hour * * weekday`"
    return (int(dow) if dow.isdigit() else CRON_DAYS.index(dow)) * 1440 + int(hour) * 60 + int(minute)


def check_the_workflow_runs_the_watch_weekly_and_can_fail():
    """One workflow runs the watch: once a week at one fixed minute, never minute 0,
    where GitHub delays and drops scheduled runs, and at least an hour from every other
    workflow's slot; with no permission beyond reading the tree and writing its issue;
    and pinned whole, so nothing can keep its step green (`|| true`,
    continue-on-error), retire it, or give it more than it needs."""
    docs = workflows()
    mine = [name for name, doc in docs.items()
            if any("tools/vllm_watch.py" in str(s.get("run", "")) for j in doc["jobs"].values() for s in j.get("steps") or [])]
    assert len(mine) == 1, f"{len(mine)} workflows run the watch: {mine}"
    doc = docs[mine[0]]
    assert {"on" if k is True else k for k in doc} == WATCH_TOP_KEYS, sorted(map(str, doc))
    on = doc.get(True, doc.get("on"))
    assert isinstance(on, dict) and set(on) == {"schedule", "workflow_dispatch"}, on
    (entry,) = schedule_of(doc)
    assert set(entry) == {"cron"}, f"the schedule entry is {entry}: its cron alone, in UTC"
    minute = entry["cron"].split()[0]
    assert minute.isdigit() and 1 <= int(minute) <= 59, f"{entry['cron']!r}: one fixed minute from 1 to 59"
    slot = minute_of_week(entry["cron"])
    near = [f"{name} ({e['cron']})" for name, other in docs.items() if name != mine[0] for e in schedule_of(other)
            if min((slot - minute_of_week(e["cron"])) % WEEK, (minute_of_week(e["cron"]) - slot) % WEEK) < 60]
    assert not near, f"the watch runs within an hour of {near}"
    assert doc["permissions"] == WORKFLOW_PERMISSIONS, doc["permissions"]
    (job,) = doc["jobs"].values()
    assert set(job) == WATCH_JOB_KEYS and job["runs-on"] == "ubuntu-latest", sorted(job)
    assert job["permissions"] == WATCH_PERMISSIONS, job["permissions"]
    assert len(job["steps"]) == len(WATCH_STEPS), [s.get("name") or s.get("uses") for s in job["steps"]]
    for want, got in zip(WATCH_STEPS, job["steps"]):
        if "uses" in want:
            assert re.fullmatch(re.escape(want["uses"]) + ACTION_VERSION, str(got.get("uses", ""))), got
            assert {k: v for k, v in got.items() if k != "uses"} == {k: v for k, v in want.items() if k != "uses"}, got
        else:
            assert got == want, f"the watch step is {got!r}, where the workflow has {want!r}"

test("the watch runs once a week, off the hour and apart, pinned whole, and can fail",
     check_the_workflow_runs_the_watch_weekly_and_can_fail)


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
