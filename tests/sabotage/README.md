# tests/sabotage

Reintroduce a bug that was fixed, run the suite, and see whether it goes red.
**A sabotage the suite does not notice is a test gap, and the gap is the finding.**

This is not part of `tests/run.sh`. It is run by hand, before a change ships, by a checker
that has not read the diff — see `.claude/skills/sabotage-corpus/`.

## Why it lives here

These drivers are coupled to the code they attack. `sab5.py` knows `perfKey`, the renderer
names and the PDF's `KeepTogether` / `bulletText` structure. A copy that drifts from the
engine is worse than no copy, because it still reports green. Versioned here, a refactor
that breaks a driver breaks it visibly, in the same commit.

They were in a gitignored scratch directory until 2026-09-20, which made them one `rm` from
gone and invisible to anyone who cloned the repo.

## Running it

```
tests/sabotage/chain.sh                 # every driver
tests/sabotage/chain.sh sab5 sab5b      # named drivers
LOGDIR=path tests/sabotage/chain.sh     # logs elsewhere (default: tmp/sabotage, gitignored)
python3 tests/sabotage/sab.py W11       # one sabotage, by name substring
```

**Commit before running.** The drivers restore with `git checkout -- <file>`, which restores
the index and not unsaved edits. Running one over work in progress has destroyed work on this
project twice. `chain.sh` refuses to start on a dirty tree for that reason; the individual
drivers assert the tree is clean when they finish.

## The corpus

Each driver applies a list of exact-string edits to committed files, refusing if a target
string is not found the expected number of times — so a driver that has drifted from the code
fails loudly rather than silently testing nothing.

| Driver | Round | What it attacks |
|---|---|---|
| `sab.py` | 1 | The machinery, and the bulk of the corpus: throughput figures leaking into each of the four JS surfaces and the PDF when a card has no measured constants |
| `sab2.py` | 1 | Leaks the suites' regexes may not see, and the benchmark panel drawn for a card without constants |
| `sab3.sh` | 1 | A `perfKey` typo on one catalog row, propagated through `tools/sync_data.py` into both generated blocks — the shape of a real contributor mistake |
| `sab4.py` | 2 | Attacks on the tests round 1 produced |
| `sab4b.py` | 2 | Leaks conditioned on state axes the probes never vary, and a figure in a reportlab attribute the spy never harvests |
| `sab4c.py` | 2 | Views fed from outside the harness: the copied report's command block, a leak gated on a selected preset, and stdout from `generate()` |
| `sab4d.py` | 2 | An invisible dependency on a throughput figure in the max-batch card, which the identity test exempts |
| `sab4e.py` | 2 | An attribute figure on an element the CARD split does not expose, and a no-number speed claim inside the PDF explanation |
| `sab5.py` | 3 | Attacks on the assumptions behind round 2's rules — fixed vendor, fixed key, fixed device count |
| `sab5b.py` | 3 | A borrowed-constant estimate shown identically for both cards, and a figure carried only in an `href` |
| `sab5c.py` | 3 | The same `href` attack on a word the reason-piece filter does not match — was the href read, or did the `<a>` merely split the raw markup? |

`suites.sh` is the shared judge: it runs the five suites that can rule on a sabotage and prints
one line each. `assets.test.py` is deliberately excluded — it hashes `index.html` and goes red on
any edit, so it would report every sabotage as caught regardless of what the sabotage did.

Rounds escalate: round 2 found gaps round 1 left, round 3 found gaps round 2 left. That is
normal and expensive, and it converges when probes are **derived from the data** rather than
enumerated. Round 3's own finding was that four renderers were never exercised because the
harness enumerated its renderer list instead of deriving it.

## What round 3 said about what comes next

Round 3 recorded that its probes fixed `vendor: 'nvidia'` and a single unknown key, so **a
leak gated on `vendor === 'amd'` or on a CDNA `perfKey` would have passed.** Any commit adding
a vendor must add probes that vary those axes, or this corpus does not cover it.

Equally: every probe ran at `devices: 2` while every catalog row is `devices: 1`, which hid a
leak gated on single-chip cards until round 2 found it. Derive probe parameters from
`data/gpus.json`, not from whichever fixture was convenient.

## compare/ — proving a change moved nothing

The drivers ask *would the suite notice if the bug came back*. `compare/` asks the other
question: **did this change move any number it was not supposed to move?**

```
node    tests/sabotage/compare/nochange.js  [base-ref]
python3 tests/sabotage/compare/nochange.py  [base-ref] [stride] [pdf_stride]
```

Both compare the working tree against `base-ref` (default `master`): the JS engine over
every catalog row and a grid of states, the Python engine over `compute()`, the command,
the advice, the PDF's story strings and real PDF bytes with the clock pinned, plus
`from_json` and `interactive_mode`. A commit that is meant to add rows without touching
existing figures should print zeros everywhere.

Both sides come from **git**, not from checked-in copies. The originals loaded 475 KB of
snapshotted engines, which go stale the moment either side moves — and a comparison against
a stale "master" reports zero differences for the wrong reason.

**No field is exempt.** The originals excused `perfKey`, because the commit they were
written for was the one adding it. That exemption was asymmetric, so it reported five false
differences as soon as both sides had the field — and it would have *hidden* a real
`perfKey` change in any later commit. The AMD rows are exactly such a commit: their keys are
not `nvidia`, and the old check asserted they would be. When a change legitimately adds a
field, read the diff this prints rather than silencing it.

## Accepted residual risk

Hardening stopped after round 3. These were accepted then and are **still open**:

- A gauge or meter that carries a figure with no text
- Invented prose that no rule recognises as a claim

These were accepted after round 3 and have since been **closed by the golden** (PR #16):

- A figure carried in an `href` — caught by `every card still displays exactly what the golden
  records` (`W11`, `W11b`)
- A claim added to *every* card at once, which the with/without comparison is blind to by
  construction (`X2`)
- A figure written to `document.title` — recorded by the golden in `e421faf`

## Reading a result

- `N caught, 0 survived` — no gap found. This is the common outcome once a commit is careful.
- `<-- SURVIVED` — the suite did not notice. Write the test that notices it, then re-run.
- `driver errored` — usually a target string that no longer exists, which means the driver has
  drifted from the code. Fix the driver in the commit that moved the code.

A driver that exits 0 having judged nothing is the worst outcome, because it reads as a pass.
`sab3.sh` did exactly that until 2026-09-20: it called a helper that no longer existed, ran no
suite, restored cleanly and returned success. Every driver now fails loudly when it cannot
judge, and `chain.sh` treats a non-zero exit as a finding rather than noise.
