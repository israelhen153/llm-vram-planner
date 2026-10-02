#!/usr/bin/env python3
"""The vLLM release watch (tools/vllm_watch.py): the pin it reads, the releases it
counts and how it orders them, what it tells GitHub and how often, what it does when
it cannot read PyPI, and what its notice sends the owner to re-check.

The watch is only as good as what this file holds. It must read the pin from both
engines and refuse when they disagree; count final, installable releases only, in
numeric order; announce each release once, on one open issue; exit 1 whenever it
could not read the answer, because "no news" is what a broken watch looks like; and
find the files and contracts its notice names by searching the tree it runs in.

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
# all, the body of a new issue or comment read from its --body-file. Used in-process,
# and as the gh executable on PATH for the real process and anything that slips past.
FAKE_GH_LOGIC = '''
import json


def answer(state, a):
    """(exit code, stdout) of `gh *a` from state."""
    def flag(name, default=None):
        return a[a.index(name) + 1] if name in a else default
    if a[:2] == ["issue", "list"]:
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
    installed. A post-release of a final release counts (see parse()), and so does a
    release with one file yanked and another not."""
    data = pypi({"1.21.0rc1": [False], "1.21.0a2": [False], "1.21.0b1": [False], "1.21.0.dev0": [False],
                 "1.21.0rc1.post1": [False], "1.21.1.dev2": [False],
                 "1.21.1": [True, True], "1.21.2": [],
                 "1.21.3": [True, False], "1.20.9.post1": [False], "1.20.9": [False], "1.20.8": [False]})
    got = vw.newer_releases("v1.20.9", data)
    assert got == ["1.20.9.post1", "1.21.3"], f"newer than v1.20.9: {got}"

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


def check_each_contract_is_found_where_it_lives():
    """Each contract, at a file:line that carries it, the refusal in generate_report.py,
    the `checked` date in both engines' ROCm table; lines that move when the engines
    grow; a contract that has gone refused; and every site in the notice main() writes."""
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
    assert {e for e, _ in sites["the FP8 refusal"]} == {"generate_report.py"}, sites["the FP8 refusal"]
    assert {e for e, _ in sites["the ROCm table's `checked`"]} == set(vw.ENGINES), sites["the ROCm table's `checked`"]
    grown = tree(lambda engine, text: "\n" * 5 + text)
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


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
