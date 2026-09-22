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
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver

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
    # The body must not lose the second command. Deleting the printf that carries
    # it is the whole sabotage — the first version of this also deleted a line of
    # prose and left the command in place, so it read as a survivor when nothing
    # about the body had actually changed.
    "P05 the body names one golden again, when set -e hides the second":
        [(WF, "              printf 'UPDATE_GOLDEN=1 python3 tests/report.test.py  # what the PDF says\\n'\n", "", 1)],
}

if __name__ == "__main__":
    run_driver(S)
