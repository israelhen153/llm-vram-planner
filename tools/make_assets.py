#!/usr/bin/env python3
"""Regenerate assets/vllm_planner.png and assets/og-image.png from index.html.

Both images show the tool's own output, so both go stale silently every time the
tool's output changes. That is what happened to the pair this replaces: they were
captured on 2026-07-23 and 2026-07-27 and still showed "GPU 0", a single decode
figure, and a benchmark comparison whose basis has since changed -- in the README
and in every link preview, for five weeks, with nothing watching.

So the images are generated, not drawn. Run this after any change to what the tool
renders, and at every release:

    python3 tools/make_assets.py            # rewrite both PNGs
    python3 tools/make_assets.py --check    # verify the pinned state still resolves

Needs: google-chrome (headless), pillow, websockets. None are needed to run the
tool, the suite, or the report generator -- this is a maintenance script, and the
offline guarantee is unaffected.
"""

import argparse
import asyncio
import hashlib
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, 'index.html')
ASSETS = os.path.join(ROOT, 'assets')

# The configuration both images depict. Kept identical to the pair this replaces --
# Llama 3.1 8B, BF16, one A100 80GB, 128K context -- so that the only difference
# between the old images and the new ones is what the tool does, which is the whole
# point of regenerating them. `pr` is precisionToken(): dataset.q + ':' + value, and
# BF16 carries an empty q, hence the bare ":2".
STATE = ('p=8&a=100&bpp=2&l=32&kv=8&hd=128&se=0&ctx=131072&cc=1&ng=1'
         '&gpu=a100-80&nv=0&kvp=2&pr=%3A2&pre=llama31-8b')

# Anything here that stops resolving makes the tool fall back to a default and show
# its "could not be fully restored" notice -- in the screenshot, where it would look
# like a feature. --check asserts each one still exists rather than waiting to see it.
PINNED = {
    'preset key': ("pre", 'llama31-8b', r"'llama31-8b':\s*\{"),
    'GPU key': ("gpu", 'a100-80', r"'a100-80':\s*\{|\"a100-80\":\s*\{"),
    'BF16 precision': ("pr", ':2', r'data-q=""[^>]*value="2"|value="2"[^>]*data-q=""'),
}

# Written beside the images and read by tests/assets.test.py. The images are a
# function of index.html, so the suite can tell they are stale without rendering
# anything: index.html moved, the recorded digest did not. Rendering is deterministic
# -- regenerating an unaffected image reproduces it byte for byte -- so the remedy is
# always just to re-run this script, and an edit that does not reach the images leaves
# only this file changed.
MANIFEST = 'manifest.json'

CARD_W, CARD_H = 1200, 630
SHOT_W = 672          # width of the README hero, unchanged from the image it replaces
BG, FG = (26, 26, 24), (255, 255, 255)
MUTED, DOT = (150, 148, 140), (110, 200, 165)

TITLE = 'LLM VRAM Planner'
SUBTITLE = 'Will your model fit? What flags do you pass?'
BULLETS = [
    'VRAM breakdown — weights, KV cache, activations',
    'Generates the runnable  vllm serve  command',
    'Cloud cost: hyperscaler / specialized / spot',
    'Inference + LoRA / QLoRA training estimates',
]
PILLS = ['Single HTML file', 'Works offline', 'No signup', 'MIT']

CHROME = next((c for c in ('google-chrome', 'google-chrome-stable', 'chromium',
                           'chromium-browser') if shutil.which(c)), None)


def check_state():
    """Fail loudly if the pinned configuration no longer resolves in index.html."""
    html = open(INDEX, encoding='utf-8').read()
    bad = [f'{what} ({param}={value})' for what, (param, value, pattern) in PINNED.items()
           if not re.search(pattern, html)]
    for what, (param, value, _) in PINNED.items():
        if f'{param}=' not in STATE:
            bad.append(f'{what}: STATE carries no {param}=')
    if bad:
        print('The pinned screenshot state no longer resolves:', file=sys.stderr)
        for b in bad:
            print(f'  - {b}', file=sys.stderr)
        print('\nThe tool would fall back to a default and render its '
              '"could not be fully restored" notice into the screenshot.', file=sys.stderr)
        return False
    print(f'pinned state resolves: {", ".join(PINNED)}')
    return True


async def _cdp(port, url):
    """Drive Chrome over the DevTools protocol: emulate light, load, measure, clip."""
    import websockets

    # Pick the document, not whatever came first: an installed extension's
    # background_page usually heads the list, and evaluating against it reports an
    # empty document with no sections.
    for _ in range(100):                                   # wait for the port to open
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list', timeout=1) as r:
                tabs = [t for t in json.load(r)
                        if t.get('type') == 'page' and t.get('url', '').startswith('file://')]
            if tabs:
                break
        except Exception:
            pass
        time.sleep(0.1)
    else:
        raise RuntimeError(f'Chrome never opened {url}')

    async with websockets.connect(tabs[0]['webSocketDebuggerUrl'],
                                  max_size=64 * 1024 * 1024) as ws:
        n = 0

        async def call(method, **params):
            nonlocal n
            n += 1
            await ws.send(json.dumps({'id': n, 'method': method, 'params': params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get('id') == n:
                    if 'error' in msg:
                        raise RuntimeError(f'{method}: {msg["error"]}')
                    return msg['result']

        await call('Page.enable')
        # The real fix for the theme, rather than an undocumented Blink flag: :root in
        # index.html is light and the dark palette is a prefers-color-scheme override,
        # so headless (which reports dark) would otherwise capture the wrong one.
        await call('Emulation.setEmulatedMedia',
                   features=[{'name': 'prefers-color-scheme', 'value': 'light'}])
        # Chrome is started on the target URL rather than navigated to it: navigating
        # away from about:blank leaves Runtime.evaluate bound to the old execution
        # context, which reports an empty document and no sections.
        await call('Page.reload', ignoreCache=True)
        await asyncio.sleep(3)                             # let the tool compute and render

        box = await call('Runtime.evaluate', returnByValue=True, expression="""
            (() => {
              const secs = [...document.querySelectorAll('div.section')]
                .filter(s => getComputedStyle(s).display !== 'none');
              const has = re => secs.find(s => re.test(s.textContent));
              // The inputs are cropped away and the run of result sections is kept.
              // It ends at the vllm command rather than at the last section: the README's
              // alt text has always promised "VRAM breakdown, vllm serve command, and cost
              // comparison" and the image it describes stopped at cost, while the section
              // after the command is the comparison panel, which is empty until a snapshot
              // is saved and photographs as a blank strip.
              const first = has(/VRAM breakdown/i), last = has(/vllm command/i);
              if (!first || !last) return null;
              const a = first.getBoundingClientRect(), b = last.getBoundingClientRect();
              return {x: a.left + scrollX, y: a.top + scrollY,
                      width: a.width, height: b.bottom + scrollY - (a.top + scrollY)};
            })()
        """)
        clip = box['result'].get('value')
        if not clip:
            raise RuntimeError('could not locate the VRAM breakdown section in index.html')
        clip['scale'] = 2                                  # capture at 2x, downscale later

        shot = await call('Page.captureScreenshot', format='png',
                          clip=clip, captureBeyondViewport=True)
        return base64.b64decode(shot['data'])


def render_panel():
    """Screenshot the results panel -- VRAM breakdown through cost -- at the pinned state."""
    if not CHROME:
        sys.exit('no chrome/chromium on PATH; cannot render')
    port, profile = 9222, tempfile.mkdtemp(prefix='mkassets-')
    proc = subprocess.Popen(
        [CHROME, '--headless', '--disable-gpu', '--no-sandbox', '--hide-scrollbars',
         f'--remote-debugging-port={port}', f'--user-data-dir={profile}',
         '--allow-file-access-from-files', '--disable-extensions',
         '--window-size=1100,3000',
         f'file://{INDEX}#{STATE}'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        return asyncio.run(_cdp(port, f'file://{INDEX}#{STATE}'))
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        shutil.rmtree(profile, ignore_errors=True)


def rounded(im, radius):
    from PIL import Image, ImageDraw
    mask = Image.new('L', im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([(0, 0), (im.width - 1, im.height - 1)],
                                           radius=radius, fill=255)
    out = Image.new('RGBA', im.size, (0, 0, 0, 0))
    out.paste(im, (0, 0), mask)
    return out


def build_card(panel):
    """Compose the 1200x630 social card around a scaled copy of the panel."""
    from PIL import Image, ImageDraw, ImageFont

    def font(size, bold=False):
        path = ('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf' if bold
                else '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf')
        return ImageFont.truetype(path, size) if os.path.exists(path) \
            else ImageFont.load_default()

    card = Image.new('RGB', (CARD_W, CARD_H), BG)
    d = ImageDraw.Draw(card)

    d.text((57, 88), TITLE, font=font(54, bold=True), fill=FG)
    d.text((57, 162), SUBTITLE, font=font(23), fill=MUTED)

    f = font(19)
    for i, line in enumerate(BULLETS):
        y = 240 + i * 46
        d.ellipse([(60, y + 7), (70, y + 17)], fill=DOT)
        d.text((88, y), line, font=f, fill=(228, 226, 219))

    fp, x = font(15), 57
    for pill in PILLS:
        w = d.textlength(pill, font=fp) + 34
        d.rounded_rectangle([(x, 541), (x + w, 577)], radius=18,
                            outline=(92, 90, 84), width=1)
        d.text((x + 17, 551), pill, font=fp, fill=(196, 194, 186))
        x += w + 12

    # The panel, scaled to the plate on the right and cropped where the plate ends.
    pw, ph, px, py = 372, 474, 772, 78
    scaled = panel.resize((pw, int(panel.height * pw / panel.width)), Image.LANCZOS)
    plate = Image.new('RGB', (pw, ph), (255, 255, 255))
    plate.paste(scaled.crop((0, 0, pw, min(ph, scaled.height))), (0, 0))
    card.paste(rounded(plate, 10), (px, py), rounded(plate, 10))
    return card


def index_digest():
    with open(INDEX, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true',
                    help='verify the pinned state still resolves; render nothing')
    args = ap.parse_args()

    if args.check:
        sys.exit(0 if check_state() else 1)
    if not check_state():
        sys.exit(1)

    from PIL import Image
    panel = Image.open(io.BytesIO(render_panel())).convert('RGB')

    hero = panel.resize((SHOT_W, int(panel.height * SHOT_W / panel.width)), Image.LANCZOS)
    hero.save(os.path.join(ASSETS, 'vllm_planner.png'), optimize=True)
    print(f'assets/vllm_planner.png  {hero.width}x{hero.height}')

    card = build_card(panel)
    card.save(os.path.join(ASSETS, 'og-image.png'), optimize=True)
    print(f'assets/og-image.png      {card.width}x{card.height}')

    with open(os.path.join(ASSETS, MANIFEST), 'w', encoding='utf-8') as fh:
        json.dump({'_comment': 'Written by tools/make_assets.py. If a test says this is '
                               'stale, re-run that script; do not edit this file.',
                   'index_sha256': index_digest(), 'state': STATE,
                   'vllm_planner.png': [hero.width, hero.height],
                   'og-image.png': [card.width, card.height]}, fh, indent=2)
        fh.write('\n')
    print(f'assets/{MANIFEST}       index.html @ {index_digest()[:12]}')


if __name__ == '__main__':
    main()
