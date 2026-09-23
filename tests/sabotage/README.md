# tests/sabotage

Reintroduce a bug that was fixed, run the suite, and see whether it goes red.
**A sabotage the suite does not notice is a test gap, and the gap is the finding.**

This is not part of `tests/run.sh`. It is run by hand, before a change ships, by a checker
that has not read the diff — see `.claude/skills/sabotage-corpus/`.

## Why it lives here

These drivers are coupled to the code they attack. `engine_r3_fixed_assumptions.py` knows `perfKey`, the renderer
names and the PDF's `KeepTogether` / `bulletText` structure. A copy that drifts from the
engine is worse than no copy, because it still reports green. Versioned here, a refactor
that breaks a driver breaks it visibly, in the same commit.

They were in a gitignored scratch directory until 2026-09-20, which made them one `rm` from
gone and invisible to anyone who cloned the repo.

## Running it

```
tests/sabotage/chain.sh                 # every driver
tests/sabotage/chain.sh engine_r3_fixed_assumptions engine_r3_identical_estimate_and_href   # named drivers
LOGDIR=path tests/sabotage/chain.sh     # logs elsewhere (default: tmp/sabotage, gitignored)
python3 tests/sabotage/engine_r1_throughput_leaks.py W11     # one sabotage, by name substring
python3 tests/sabotage/engine_r1_throughput_leaks.py --from H8   # from one sabotage onwards
```

**Commit before running.** The drivers restore with `git checkout -- <file>`, which restores
the index and not unsaved edits. Running one over work in progress has destroyed work on this
project twice. `chain.sh` refuses to start on a dirty tree for that reason; the individual
drivers assert the tree is clean when they finish.

## The corpus

Each driver applies a list of exact-string edits to committed files, refusing if a target
string is not found the expected number of times — so a driver that has drifted from the code
fails loudly rather than silently testing nothing.

### How the files are named

A driver's name says what it attacks and which review round produced it:
`engine_*` drivers attack the model in `index.html` and `generate_report.py` — and, from round 4,
the data and writer that feed it (`data/gpus.json`, `tools/sync_data.py`, `tools/price_check.py`),
since `engine_r4_cost_provenance.py` covers a feature that spans both — `workflow_*` drivers
attack `.github/workflows/price-refresh.yml` and the guard in `tests/workflow.test.py`, and `rN` is
the round. `ls` therefore lists each family in the order its rounds escalated. `workflow_live_*`
came from a real run rather than a review.

Two files are not drivers:

- **`harness.py`** — the machinery every driver shares: `apply_edits`, `restore_files`,
  `run_judging_suites`, `require_green_baseline`, and `run_driver`, the one loop every Python driver
  calls. There used to be fifteen copies of that loop in six different shapes.
- **`anchors.py`** — the exact excerpts of the engine files that `engine_*` sabotages anchor their
  edits on, plus `INDEX_HTML`, `REPORT_PY` and `SYNC_PY`, the paths they edit.

Until 2026-09-22 the drivers were `sab.py`, `sab2.py` … `sab10.py`, and `sab.py` was both the
harness and the biggest driver. The rename changed no sabotage: all 330 in the Python drivers were
compared as resolved values before and after, and every one was identical.

| Driver | Round | What it attacks |
|---|---|---|
| `engine_r1_throughput_leaks.py` | 1 | The bulk of the corpus: throughput figures leaking into each of the four JS surfaces and the PDF when a card has no measured constants |
| `engine_r1_regex_blind_spots.py` | 1 | Leaks the suites' regexes may not see, and the benchmark panel drawn for a card without constants |
| `engine_r1_perfkey_typo.sh` | 1 | A `perfKey` typo on one catalog row, propagated through `tools/sync_data.py` into both generated blocks — the shape of a real contributor mistake |
| `engine_r2_attacks_on_new_tests.py` | 2 | Attacks on the tests round 1 produced |
| `engine_r2_unvaried_state_axes.py` | 2 | Leaks conditioned on state axes the probes never vary, and a figure in a reportlab attribute the spy never harvests |
| `engine_r2_views_outside_harness.py` | 2 | Views fed from outside the JS test harness: the copied report's command block, a leak gated on a selected preset, and stdout from `generate()` |
| `engine_r2_invisible_dependency.py` | 2 | An invisible dependency on a throughput figure in the max-batch card, which the identity test exempts |
| `engine_r2_hidden_attributes.py` | 2 | An attribute figure on an element the CARD split does not expose, and a no-number speed claim inside the PDF explanation |
| `engine_r3_fixed_assumptions.py` | 3 | Attacks on the assumptions behind round 2's rules — fixed vendor, fixed key, fixed device count |
| `engine_r3_identical_estimate_and_href.py` | 3 | A borrowed-constant estimate shown identically for both cards, and a figure carried only in an `href` |
| `engine_r3_href_on_unmatched_word.py` | 3 | The same `href` attack on a word the reason-piece filter does not match — was the href read, or did the `<a>` merely split the raw markup? |
| `engine_r5_provenance_fields.py` | 2 | What a recorded provenance may say: a provider that did not supply the number, a region it was not read in, a date that is not a date or has not happened, and an extra key nothing renders — all of which rendered, and none of which the round-1 field checks noticed |
| `engine_r5_provenance_rules.py` | 2 | Round 2 against those tests: fourteen shapes that passed them once the goldens were regenerated — a label deleted so its tier was skipped, two tiers' labels swapped, and composites joined by ` and `, `&`, a newline and the label's own separator, lowercase, and by providers no list contains |
| `engine_r4_cost_provenance.py` | 4 | `fix/cost-provenance`: a composite provider list restored or appended anywhere near a price, `priceSource` dropped from the generated blocks, an invented source for a tier `SOURCE_MAP` marks manual, a blank date, a label that names the provider but drops the date or region, a price moved with its provenance left untouched, a cost range no longer bounded by the tier that is actually cheapest or priciest, the pre-fix catalog writer that lost a row's formatting the moment a different row changed, and a cross-check floor widened past the disagreement it exists to catch |
| `engine_r6_amd_rows.py` | 6 | `feat/amd-gpus`, along the axes the AMD rows introduce, every catalog probe derived from `data/gpus.json`: throughput leaks gated on the vendor, each real AMD `perfKey`, a two-device board, the form, each card's name in both spellings and a null tier; AMD borrowing NVIDIA's constants in either engine; every AMD row's FP8 flag, `perfKey`, TFLOPS, device count, form and vendor edited, a price put where none is confirmed, a hand record moved, dated ahead, over plain http or no longer backing its price, and a spot SKU losing its marker; a null tier costing $0 or borrowing a neighbour's price on every surface; an OAM board's fabric called PCIe; the dropdown losing its sections; and `price_check.py` fetching or automating a null tier |
| `workflow_r1_gate.py` | 1 | The gate that left the price job able to succeed only by finding nothing, and the seven ways a cold check rebuilt it afterwards without touching a guarded field |
| `workflow_r2_yaml_reader.py` | 2 | The round-1 guard's own reader and rules: a delivery condition gated on the suite in a FOLDED second line, a step respelled until the reader and its own count check both dropped it, an object filter that gates delivery while naming no step, a test run moved upstream of the gate, and `.conclusion` for `.outcome` |
| `workflow_r3_name_bindings.py` | 3 | The round-2 guard's bindings: a second job in the same file carrying the original bug where every rule read one literal job key, a job-level `if:` that retires the job as "skipped", a decoy `echo` that takes `id: suite` so the PR body reports its outcome forever, and `format()` hiding a filename from the path regex |
| `workflow_live_first_run.py` | live | Not from a review: what the first real run of the fixed price job found. `add-paths` listed `assets/*.png` and make_assets.py writes three files there, so the manifest recording index.html's hash was regenerated and then left behind, failing the asset gate on every price PR |
| `workflow_r4_shell_quoting.py` | 4 | The guard's own helper: six shell-quoting shapes that hid a test run from `executes()` (quoted argument, command substitution, `$'...'`, an apostrophe in a comment that opens a span across newlines), plus a rule that passed by not looking, a negative pathspec, `UPDATE_GOLDEN` through `env:`, and the gate — which nothing pinned at all |

The suites that judge a sabotage are listed once, in `harness.py`'s `JUDGING_SUITES`, and
`suites.sh` runs the same set for the one bash driver, printing one line each. `assets.test.py` is deliberately excluded — it hashes `index.html` and goes red on
any edit, so it would report every sabotage as caught regardless of what the sabotage did.
`tests/corpus.test.py` is excluded for the same reason — it checks the anchors still match the
engine, which every sabotage breaks. It runs in the ordinary suite instead, so a pull request
that moves anchored engine text goes red in its own CI rather than at the next corpus run.
`workflow.test.py` joined the judges with `workflow_r1_gate.py`: without it a workflow sabotage reads green,
because none of the other five opens `.github/`.

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
`engine_r1_perfkey_typo.sh` (then `sab3.sh`) did exactly that until 2026-09-20: it called a helper that no longer existed, ran no
suite, restored cleanly and returned success. Every driver now fails loudly when it cannot
judge, and `chain.sh` treats a non-zero exit as a finding rather than noise.
