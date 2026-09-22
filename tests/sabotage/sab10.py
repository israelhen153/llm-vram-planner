#!/usr/bin/env python3
"""Round 4 against tests/workflow.test.py, and the round that changed the most.

Every sabotage here passed the guard, and six of them went through executes() —
the helper added one commit earlier AS the mechanism for the recurring "read the
thing, not its name" bug:

    node "tests/model.test.js"        the path hidden in quotes
    LOG="$(./tests/run.sh 2>&1)"      hidden in a substitution
    bash $'tests/run.sh'              a quoting form it did not know
    echo "x"  # don't skip this       an apostrophe opens a span that runs to
    node tests/model.test.js          the next quote ANYWHERE later, across
                                      newlines, swallowing this line whole

No regex over quote pairs tokenises shell. Round 2 had already taught that about
YAML; the lesson was generalised one file wide instead of one level up.

The rest are the same shape one layer out: a rule whose guard was a substring so
it `return`ed and passed by not looking (`python3 -m tools.make_assets`), an
extraction that knew one idiom (`open(f'{ASSETS}/...','w')`), a matcher that
checked something matched and never that something was excluded
(`:!assets/manifest.json`), a splitter that split on newlines where the action
splits on commas too, a body rule satisfied by a trailing comment or an unused
variable or a branch a red run never reaches, UPDATE_GOLDEN reached through
`env:` where no shell line exists to find, and — the headline — nothing at all
pinning the gate, so `echo "changed=false"` makes the job green every week and
silent forever.

What replaced them: literal contracts where there is a contract, a shell pinned
whole where behaviour matters, and for the PR body, running the script and
reading what it wrote.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in sab.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sab

WF = ".github/workflows/price-refresh.yml"
MA = "tools/make_assets.py"
GATE = "      - name: Check whether there is anything to propose"
BODYCMD = "              printf 'UPDATE_GOLDEN=1 python3 tests/report.test.py  # what the PDF says\\n'"


def ins(before, block):
    return (WF, before, block + before, 1)


def rep(f, a, b):
    return (f, a, b, 1)


STEP = lambda body: f"      - name: Sanity-check first\n        run: {body}\n\n"

S = {
 # 1. executes() quoting shapes — all inserted before the gate
 'X02 quoted path argument':            [ins(GATE, STEP('node "tests/model.test.js"'))],
 'X03 command substitution':            [ins(GATE, '      - name: Sanity-check first\n        run: |\n          LOG="$(./tests/run.sh 2>&1)"\n\n')],
 'X04 ANSI-C quoting':                  [ins(GATE, STEP("bash $'tests/run.sh'"))],
 'X05 apostrophe in a comment':         [ins(GATE, '      - name: Sanity-check first\n        run: |\n          echo "checking"  # don\'t skip this\n          node tests/model.test.js\n          echo \'done\'\n\n')],
 'X06 apostrophe in a heredoc':         [ins(GATE, '      - name: Sanity-check first\n        run: |\n          cat <<EOF\n          it\'s fine\n          EOF\n          node tests/model.test.js\n\n')],
 'X07 escaped quote':                   [ins(GATE, '      - name: Sanity-check first\n        run: |\n          echo "\\""\n          node tests/model.test.js\n\n')],
 # 2. add-paths vacuity
 'Y02 make_assets via -m':              [rep(WF,'        run: python3 tools/make_assets.py','        run: python3 -m tools.make_assets'), rep(WF,'            assets/\n','            assets/*.png\n')],
 'Y03 make_assets via cd':              [rep(WF,'        run: python3 tools/make_assets.py','        run: cd tools && python3 make_assets.py'), rep(WF,'            assets/\n','            assets/*.png\n')],
 'Y04 add-paths narrowed to png+manifest': [rep(WF,'            assets/\n','            assets/*.png\n            assets/manifest.json\n')],
 'Y05 a fourth output as an f-string':  [rep(MA,"    with open(os.path.join(ASSETS, MANIFEST), 'w', encoding='utf-8') as fh:","    open(f'{ASSETS}/sizes.json', 'w').write('{}')\n    with open(os.path.join(ASSETS, MANIFEST), 'w', encoding='utf-8') as fh:"), rep(WF,'            assets/\n','            assets/*.png\n            assets/manifest.json\n')],
 'Y08 negative pathspec':               [rep(WF,'            assets/\n','            assets/\n            :!assets/manifest.json\n')],
 'Y10 comma-separated add-paths':       [rep(WF,'            assets/\n','            assets/*.png, assets/manifest.json\n')],
 # 3. the body rule
 'Z02 commands become a trailing comment': [rep(WF, BODYCMD, "              printf 'see the tests\\n'  # tests/report.test.py")],
 'Z03 UPDATE_GOLDEN stripped':          [rep(WF,"              printf 'UPDATE_GOLDEN=1 node tests/model.test.js      # what the cards display\\n'\n"+BODYCMD+"\n","              printf 'node tests/model.test.js\\n'\n              printf 'python3 tests/report.test.py\\n'\n")],
 'Z05 names parked in an unused var':   [rep(WF, BODYCMD, "              GOLDENS='tests/model.test.js tests/report.test.py'")],
 # 4. the gate
 'V01 gate always says nothing moved':  [rep(WF,'          if git diff --quiet -- data/gpus.json index.html generate_report.py; then\n            echo "changed=false" >> "$GITHUB_OUTPUT"\n          else\n            echo "changed=true" >> "$GITHUB_OUTPUT"\n          fi','          echo "changed=false" >> "$GITHUB_OUTPUT"')],
 # 5. the golden via env
 'V03 UPDATE_GOLDEN through env:':      [rep(WF,"        id: suite\n","        id: suite\n        env:\n          UPDATE_GOLDEN: \"1\"\n"), rep(WF,'            assets/\n','            assets/\n            tests/?olden/*.json\n')],
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
            sab.restore([WF, MA])
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
