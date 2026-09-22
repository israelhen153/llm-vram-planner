#!/usr/bin/env python3
"""Round 1 against .github/workflows/price-refresh.yml and tests/workflow.test.py.

The price job ran for days unable to do the one thing it exists for: the golden
records displayed prices verbatim, so any real price move reddens the suite, and
the pull-request step was gated on that suite. It applied its moves, went red,
skipped the PR, and discarded the report. Nothing in the tree could see it —
no suite read the workflow files, so a job that could only succeed by finding
nothing looked exactly like a job that worked, and only a Monday could tell.

The first eleven below are the reverts anyone would think of. The last seven are
the ones a cold check found AFTER the guard was written and passing: every one of
them leaves the guarded fields alone and breaks the job somewhere the fields do
not reach — a gate built out of a step that merely exits, a condition on the step
that explains the red, the job's permissions block, and filenames that no longer
name anything. They are the reason the guard is positional and derived rather than
a list of fields, and they are what the NEXT round should try to get past.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in sab.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sab

WF = ".github/workflows/price-refresh.yml"
DELIV = "        if: ${{ !cancelled() && steps.diff.outputs.changed == 'true' }}"
BODY = "      - name: Tell the PR body what the suite did\n" + DELIV

S = {
    # --- the original defects, reverted one at a time -----------------------
    "A1 the suite gates the job again (continue-on-error dropped)":
        [(WF, "        continue-on-error: true\n", "", 1)],
    "A2 the report uploads only on runs that found nothing":
        [(WF, "        if: always()\n        uses: actions/upload-artifact@v4",
             "        if: steps.diff.outputs.changed == 'false'\n        uses: actions/upload-artifact@v4", 1)],
    "A3 the PR step is conditioned on the suite outcome":
        [(WF, DELIV + "\n        uses: peter-evans/create-pull-request@v6",
             "        if: ${{ !cancelled() && steps.diff.outputs.changed == 'true' && steps.suite.outcome == 'success' }}\n        uses: peter-evans/create-pull-request@v6", 1)],
    "A4 pipefail dropped, so tee supplies the suite step's exit code":
        [(WF, "          set -o pipefail\n", "", 1)],
    "A5 the bot regenerates and commits the golden it is measured against":
        [(WF, "            assets/\n", "            assets/\n            tests/golden/page.json\n", 1)],

    # --- ways the guard could be satisfied while meaning nothing ------------
    "B6 the suite step stops running the suite":
        [(WF, '          ./tests/run.sh 2>&1 | tee "$RUNNER_TEMP/suite.log"', "          echo skipped", 1)],
    "B7 the upload step stops uploading":
        [(WF, "        uses: actions/upload-artifact@v4", '        run: echo "not uploading"', 1)],
    "B8 the upload is weakened from always() to success()":
        [(WF, "        if: always()\n        uses: actions/upload-artifact@v4",
             "        if: success()\n        uses: actions/upload-artifact@v4", 1)],
    "B9 the PR step is retargeted at master":
        [(WF, "          branch: automation/price-refresh", "          branch: master", 1)],
    "B10 a step pushes to master directly":
        [(WF, "      - name: Upload the report, whatever happened",
             "      - name: Ship it\n        run: git push origin HEAD:master\n\n      - name: Upload the report, whatever happened", 1)],
    "B11 the file is reindented until the reader cannot account for it":
        [(WF, "      - name: Run the full suite", "      -   name: Run the full suite", 1)],

    # --- what the cold check found after the guard was written and green ----
    # Each of these leaves every guarded field exactly as it is.
    "C12 a gate rebuilt out of a step that merely exits, touching no condition":
        [(WF, "      - name: Tell the PR body what the suite did",
             '      - name: Re-check the suite outcome for safety\n'
             "        if: steps.diff.outputs.changed == 'true'\n"
             '        run: |\n'
             '          if [ "${{ steps.suite.outcome }}" != "success" ]; then\n'
             "            exit 1\n"
             "          fi\n\n"
             "      - name: Tell the PR body what the suite did", 1)],
    "C13 only the step that explains the red suite is gated on the suite":
        [(WF, BODY, "      - name: Tell the PR body what the suite did\n"
             "        if: ${{ !cancelled() && steps.diff.outputs.changed == 'true' && steps.suite.outcome == 'success' }}", 1)],
    "C14 the same step reverted to a bare, implicitly success()-gated condition":
        [(WF, BODY, "      - name: Tell the PR body what the suite did\n"
             "        if: steps.diff.outputs.changed == 'true'", 1)],
    "C15 the job-level permissions block is removed":
        [(WF, "    permissions:\n      contents: write # to push the branch the PR step opens\n"
              "      pull-requests: write # to open (or update) that PR\n", "", 1)],
    "C16 contents: write quietly downgraded to read":
        [(WF, "      contents: write # to push the branch", "      contents: read # to push the branch", 1)],
    "C17 body-path names a file nothing writes":
        [(WF, "          body-path: ${{ runner.temp }}/price-check-report.md",
             "          body-path: ${{ runner.temp }}/price-report.md", 1)],
    "C18 the artifact path names a file nothing writes":
        [(WF, "            ${{ runner.temp }}/suite.log", "            ${{ runner.temp }}/suite-output.log", 1)],
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
