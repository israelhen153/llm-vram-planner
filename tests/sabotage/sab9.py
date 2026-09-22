#!/usr/bin/env python3
"""What the first real run of the fixed price job found, 2026-09-22.

Not a cold check — the job ran on a runner, opened PR #24, and the PR failed
the asset gate. Three rounds of adversarial review had not found this, because
all three attacked what the guard CHECKED. This was a thing the guard did not
know existed: `add-paths` listed `assets/*.png`, and tools/make_assets.py
writes three files into assets/. The third is manifest.json, which records the
hash of the index.html the images were built from. The bot regenerated it
correctly every time and then left it behind, so every price PR failed against
a hash from before the prices moved.

Worth keeping as the reminder that a corpus grown only from review converges on
what review can see. The real run is a different oracle, and it found in forty
seconds what three rounds had missed.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in sab.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sab

WF = ".github/workflows/price-refresh.yml"

S = {
    "P01 add-paths back to *.png, which drops the manifest the gate reads":
        [(WF, "            assets/\n", "            assets/*.png\n", 1)],
    "P02 add-paths names one image and nothing else":
        [(WF, "            assets/\n", "            assets/og-image.png\n", 1)],
    "P03 assets dropped from add-paths entirely":
        [(WF, "            assets/\n", "", 1)],
    "P04 a second step that really does run the suite, not merely name it":
        [(WF, "      - name: Run the full suite — recorded, not obeyed\n        id: suite",
              "      - name: Locate the suite\n        id: suite\n"
              "        continue-on-error: true\n        run: ./tests/run.sh --help\n\n"
              "      - name: Run the full suite — recorded, not obeyed", 1)],
    "P05 the body names one golden again, when set -e hides the second":
        [(WF, "              printf 'If the moves are right, there are TWO goldens to regenerate on this\\n'",
              "              printf 'If the moves are right: UPDATE_GOLDEN=1 node tests/model.test.js\\n'", 1)],
}

if __name__ == "__main__":
    # Same shape as every other driver on purpose — see the note in sab5c.py.
    names = [n for n in S if not sys.argv[1:] or any(a in n for a in sys.argv[1:])]
    print(f"{len(names)} sabotage(s)")
    sab.require_green_baseline()
    survived, unapplied = [], []
    for name in names:
        try:
            touched = sab.apply(S[name])
        except Exception as e:
            print(f"  !! {name}: could not apply: {e}")
            unapplied.append(name)
            sab.restore([WF])
            continue
        try:
            res = sab.run_suites()
        finally:
            sab.restore(touched)
        red = {k: v for k, v in res.items() if v[0] != 0}
        if not red:
            survived.append(name)
            print(f"  GREEN  {name}   <-- SURVIVED")
        else:
            for k, (rc, fails, errs, tally) in red.items():
                print(f"  red    {name}\n         {k}[{tally[1] if tally else '?'} failed: {(fails or errs or ['?'])[0][:150]}]")
    print(f"\n{len(names) - len(survived) - len(unapplied)} caught, "
          f"{len(survived)} survived"
          + (f", {len(unapplied)} COULD NOT BE APPLIED" if unapplied else ""))
    if unapplied:
        sys.exit(2)
