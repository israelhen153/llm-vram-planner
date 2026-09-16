#!/usr/bin/env python3
"""Verify the published images can still be regenerated from the tool.

assets/og-image.png and assets/vllm_planner.png show the tool's own output, so
they go stale every time that output changes -- silently, in the README and in
every link preview. The pair these replaced sat wrong for five weeks.

Re-rendering them needs a browser, which the suite deliberately does not, so
the staleness check is done by digest instead: the generator records the sha256
of the index.html it rendered, and if index.html has moved since, the images are
by definition no longer known to match it. Rendering is deterministic, so the
remedy is always the same -- re-run the generator -- and an edit that does not
reach the images reproduces them byte for byte, leaving only the manifest
changed. The rest of this file checks what can be checked without rendering:
that the pinned configuration still resolves, that the images the documents
point at exist, and that the social card is still the size a social card has to
be. Regenerating is `python3 tools/make_assets.py`.

Run:  python3 tests/assets.test.py
"""
import hashlib
import json
import os
import re
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def png_size(path):
    """Width and height out of the IHDR, without pulling in an image library."""
    with open(path, "rb") as fh:
        head = fh.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    return struct.unpack(">II", head[16:24])


print("\nThe published images can still be regenerated")

def check_pinned_state_resolves():
    # The generator's own check, rather than a second copy of it here: a preset key
    # or GPU key that stops existing makes the tool fall back to a default and render
    # its "could not be fully restored" notice into the screenshot, where it reads as
    # a feature. This is how that gets noticed before the image is published.
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_assets.py"),
                           "--check"], capture_output=True, text=True)
    assert proc.returncode == 0, f"make_assets.py --check failed:\n{proc.stdout}{proc.stderr}"

test("the configuration the screenshots are pinned to still resolves",
     check_pinned_state_resolves)


def check_state_is_complete():
    """Every parameter updateURLHash() emits must appear in the pinned state.

    A new input that reaches the URL but not the pinned state would restore
    partially, and the screenshot would carry the restore warning. Read off
    updateURLHash() rather than listed here, so a new parameter fails this
    instead of being discovered in a published image.
    """
    html = read("index.html")
    body = re.search(r"function updateURLHash\(\) \{(.*?)\n\}", html, re.S).group(1)
    emitted = set(re.findall(r"[?&`]([a-z]+)=", body))
    assert emitted, "could not read the parameters out of updateURLHash()"
    pinned = set(re.findall(r"([a-z]+)=", read("tools", "make_assets.py").split("STATE = (")[1]
                            .split(")")[0]))
    missing = emitted - pinned
    assert not missing, (f"updateURLHash() emits {sorted(missing)}, which the pinned "
                         f"screenshot state does not set")

test("the pinned state sets every parameter the tool puts in the URL",
     check_state_is_complete)


def check_images_match_the_tool():
    """The images are a function of index.html; if it moved, they are unverified.

    This is the failure the generator exists to prevent and the only one the other
    checks here cannot see: a label or a formula changes, every published image
    silently stops matching the tool, and nothing says so. A cold review changed
    "VRAM breakdown" to "VRAM Breakdown TEST" and the rest of this file stayed green.

    Deliberately strict -- it fires on any index.html change, not only the ones that
    reach the pixels. That is affordable because rendering is deterministic: re-running
    the generator on an unrelated edit rewrites the PNGs identically and leaves only
    assets/manifest.json in the diff.
    """
    path = os.path.join(ROOT, "assets", "manifest.json")
    assert os.path.exists(path), "assets/manifest.json is missing; run tools/make_assets.py"
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(os.path.join(ROOT, "index.html"), "rb") as fh:
        actual = hashlib.sha256(fh.read()).hexdigest()
    assert manifest.get("index_sha256") == actual, (
        f"index.html has changed since the images were generated "
        f"({str(manifest.get('index_sha256'))[:12]} -> {actual[:12]}), so they are no "
        f"longer known to show what the tool renders. Run: python3 tools/make_assets.py")
    for name in ("vllm_planner.png", "og-image.png"):
        assert manifest.get(name) == list(png_size(os.path.join(ROOT, "assets", name))), \
            f"assets/{name} is not the size the manifest recorded; run tools/make_assets.py"

test("the images were generated from this index.html", check_images_match_the_tool)


def check_referenced_images_exist():
    referenced = set()
    for doc in ("README.md", "index.html"):
        referenced |= set(re.findall(r"assets/([\w.-]+\.png)", read(doc)))
    assert referenced, "no image references found in README.md or index.html"
    for name in sorted(referenced):
        assert os.path.exists(os.path.join(ROOT, "assets", name)), \
            f"assets/{name} is referenced but does not exist"

test("every image the documents point at exists", check_referenced_images_exist)


def check_card_dimensions():
    # 1200x630 is what og:image wants; drift here degrades or crops the preview on
    # every platform that renders one, and nothing else in the repo would notice.
    size = png_size(os.path.join(ROOT, "assets", "og-image.png"))
    assert size == (1200, 630), f"og-image.png is {size[0]}x{size[1]}, expected 1200x630"

test("the social card is still 1200x630", check_card_dimensions)


print(f"\n{pass_ct} passed, {fail_ct} failed")
sys.exit(1 if fail_ct else 0)
