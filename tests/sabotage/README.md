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
tests/sabotage/parallel.py              # every driver, split across worker worktrees
tests/sabotage/parallel.py --early-exit # the same, each sabotage stopped at its first real catch
tests/sabotage/parallel.py --ref <commit> engine_r10_rocm_guidance   # another commit's corpus, named drivers
tests/sabotage/parallel.py --compare <logdir> <logdir>   # two runs, sabotage by sabotage
tests/sabotage/parallel.py --subset --compare <serial logdir> <early-exit logdir>   # the same, for early exit
tests/sabotage/chain.sh                 # every driver, one after another, in this checkout
tests/sabotage/chain.sh engine_r3_fixed_assumptions engine_r3_identical_estimate_and_href   # named drivers
LOGDIR=path tests/sabotage/chain.sh     # logs elsewhere (default: tmp/sabotage, gitignored)
python3 tests/sabotage/engine_r1_throughput_leaks.py W11     # one sabotage, by name substring
python3 tests/sabotage/engine_r1_throughput_leaks.py --from H8   # from one sabotage onwards
```

**`parallel.py` changes where a sabotage runs, not how it is judged.** Each worker is a git
worktree detached at the commit under test, and every sabotage goes through that commit's own
`run_driver()` and its six suites. Only the green baseline is proved once per run instead of
once per driver: the first worker proves it, and finds the golden failures, alone; every other
worker checks that it is a clean checkout of the same commit and starts from that proof. Its
logs have `chain.sh`'s shape, so `--compare` can hold a parallel run to a
serial one. It never touches your checkout, so it needs no clean tree, and uncommitted work is
not judged. The workers live in `tmp/corpus-workers/` and are reused from run to run. **The
script never deletes one:** a worker that is dirty, locked or missing stops the run and is
listed, and removing them is the owner's call. One run holds the workers at a time.

**`--early-exit` stops at the first failure that isn't a golden's.** Suites run cheapest and most
often red first, and a sabotage whose failures so far are all golden comparisons keeps going, so
the run still reports it: every run lists the catches only a golden made in `golden-only.txt`,
since regenerating the goldens would hide them. Which failures are golden ones is found by
emptying the goldens and running the suites, not from test names, and kept in the run's
`golden-failures.json`. Workers write no bytecode and
refresh every tracked `.py`'s timestamp first: a sabotage restored inside a second once left a
`.pyc` that turned 90 later runs on one worker into false catches.

**Commit before running `chain.sh`.** The drivers restore with `git checkout -- <file>`, which restores
the index and not unsaved edits. Running one over work in progress has destroyed work on this
project twice. `chain.sh` refuses to start on a dirty tree for that reason; the individual
drivers assert the tree is clean when they finish.

Both runners hold a sleep lock for the whole run where systemd provides one. A suspend
stretched one driver from 3.6 min to 86 on 2026-09-24. Closing the lid still suspends.

## The corpus

Each driver applies a list of exact-string edits to committed files, refusing if a target
string is not found the expected number of times — so a driver that has drifted from the code
fails loudly rather than silently testing nothing.

**Never quote a value the weekly price job rewrites.** `tools/price_check.py --apply` rewrites a
tier's price, its reading's date and price, a Vast.ai reading's offer count, and drops the note
a confirmed reading replaces, and `tools/sync_data.py` carries all of it into both engines. A
sabotage that quotes one stops applying at the next refresh, so it reads them from
`data/gpus.json` as its driver loads (`engine_r4`'s S5 and C2, `engine_r5`, `engine_r10`'s P2),
and quotes only what the job leaves alone: a provider, a region, the part of a SKU that names the
product. When the row, tier or note it attacks is gone, it keeps its name and carries
`harness.Missing` where its anchor would be. The run refuses it by name, as it refuses a drifted
anchor, rather than dropping it or failing to load the driver. So which tiers it attacks comes
from `SOURCE_MAP`, which taking a reading out leaves alone, and not from the readings
(`engine_r6`'s C12, `engine_r10`'s N8): derived from the readings, C12 dropped out of the corpus
and N8 stopped its driver loading once they were gone.

### How the files are named

A driver's name says what it attacks and which review round produced it:
`engine_*` drivers attack the model in `index.html` and `generate_report.py` — and, from round 4,
the data and writer that feed it (`data/gpus.json`, `tools/sync_data.py`, `tools/price_check.py`),
since `engine_r4_cost_provenance.py` covers a feature that spans both — `workflow_*` drivers
attack `.github/workflows/price-refresh.yml` and the guard in `tests/workflow.test.py`, and `rN` is
the round. `ls` therefore lists each family in the order its rounds escalated. `workflow_live_*`
came from a real run rather than a review.

Two files are not drivers:

- **`harness.py`** — the machinery every driver shares: `apply_edits` and the one check it makes,
  `refusal`, which `tests/corpus.test.py` makes too; `Missing`; `restore_files`,
  `run_judging_suites`, `require_green_baseline`, and `run_driver`, the one loop every Python driver
  calls. There used to be fifteen copies of that loop in six different shapes.
- **`anchors.py`** — the exact excerpts of the engine files that `engine_*` sabotages anchor their
  edits on, plus `INDEX_HTML`, `REPORT_PY`, `SYNC_PY`, `PRICE_PY` and `GPUS_JSON`, the paths they edit.

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
| `engine_r6_amd_rows.py` | 6 | `feat/amd-gpus`, along the axes the AMD rows introduce, every catalog probe derived from `data/gpus.json`: throughput leaks gated on the vendor, each real AMD `perfKey`, a two-device board, the form, each card's name in the spelling each engine carries and a null tier; AMD borrowing NVIDIA's constants in either engine; every AMD row's FP8 flag, `perfKey`, TFLOPS, device count, form and vendor edited, a price put where none is confirmed, a hand record moved, dated ahead, over plain http or no longer backing its price, and a spot SKU losing its marker; a null tier costing $0 or borrowing a neighbour's price on every surface; an OAM board's fabric called PCIe; the dropdown losing its sections; and `price_check.py` fetching or automating a null tier |
| `engine_r7_amd_cold_check.py` | 7 | The first cold check of `feat/amd-gpus`: the eleven sabotages that survived it, kept so the next change starts where the check ended — a null tier filled or zeroed on the way from the catalog to the page's state, the hand record read off the spec tier whatever tier is asked (both engines), a multi-device board's fabric renamed above one board (both engines), `price_check.py --apply` dropping a hand record or its URL, and a record whose URL is a site's front page |
| `engine_r8_held_clock.py` | 8 | `fix/report-test-clock`: the report suite's held clock undone fifteen ways — the hold deleted or taken at the real time; the golden's scrubs keyed on the machine's date or on a shape again; the cover or the footer reading the machine's clock around the hold, always or only for one kind of report (FP8 weights, PCIe, 20 concurrent users, a 123B model); the cover read around the hold with the every-build check unregistered; and each of the three recorders that check reads uninstalled. Two cold-check rounds found the gated reads: a one-config check could not see the first, and a recorder inside one helper could not see builds another helper made |
| `engine_r9_gguf_plugin.py` | 9 | `fix/gguf-plugin`: the GGUF guidance dropped from the command panel, the copied report or the PDF one surface at a time, shown for a precision that is not GGUF, gated on one GGUF level, printed without its sources or under a command that is not printed, the plugin sentence, a source or the retired single-file claim changed in both engines alike so parity cannot see it, and the copied report quoting a banner's `<code>` span instead of the command box |
| `engine_r10_rocm_guidance.py` | 10 | `feat/rocm-guidance`: the FP8 gate opened on every card, keyed to one target, or taken out of the command builder, the precision control or its call site; the ROCm command given to NVIDIA, its image unpinned or swapped for `rocm/vllm`, AITER on everywhere or nowhere, and its lines dropped from one surface or changed in both engines alike; a price tier's note dropped, undated or given a figure, and kept beside a new reading; a lead's price reaching a figure or shown on a priced tier; the overhead text saying 0.3 or NCCL on AMD again; the nulled hyperscaler prices put back; and the two quantization paths C8 fixed |
| `engine_r11_rocm_cold_check.py` | 11 | The first cold check of `feat/rocm-guidance`: the two bugs it found without sabotaging anything, put back ("FP8" in a JSON config getting past the refusal, and an unknown `--prec` planned as AWQ); a local model mounted only under `/opt`; every precision sent to BF16 on a card that can't run FP8; a note kept beside a MOVED reading; the one-byte FP8 fill, or the refusal, lost in `from_json()`'s own-fields branch; `--prec q8` naming no quantization and `--prec fp8` sized as BF16; a lead's price in the per-board cell at two boards and at four and more; the lead sentence gated on one tier; 0.3 GB per device off NVLink or over Infinity Fabric; a price in a note without a `$`; a note beside a hand record; a lead on a priced tier or dropped from the hyperscaler's; and FP8 allowed on a target the ROCm table lacks. Then the second round, against the fixes: the same gaps one step over, where a fix had listed cases rather than derived them — `" fp8"`, wrong types, the refusal swallowed above one board, the per-device buffer, AMD's wording above three boards, leads at five boards and at seventeen and more, paths with a dot or deep nesting, `--apply` on the hyperscaler tier, a same-day reading or a row with a hand record, the menu above one board, an empty quantization, note wordings the rule missed, and a board of four devices |
| `engine_r15_json_precision_width.py` | 15 | `fix/json-precision-width`: a JSON config's width guessed again — the old 0.5 default, every method at 0.5 (FP8 included), GGUF or an unknown method planned without a width, a method and a width that disagree or a width with no method planned, the preset branch dropping the width a JSON gave, GGUF treated as having one width, the rules kept to NVIDIA or to one board, and the GGUF refusal without its levels |
| `engine_r16_json_width_cold_check.py` | 16 | The cold check of `fix/json-precision-width`, on what round 15's sweep holds fixed. Six survived it at `3d1e37e` (X4–X9), and the tests that catch them now vary each of those axes: a width of 0 treated as missing at either builder, the rule kept to dense models (the sweep's only preset is llama31-8b), a config naming nothing sized at its KV cache's width (always 2 in the sweep), the width looked up by `--prec`'s token so `int4`, `bf16` and `q4km` are planned with a `--quantization` vLLM rejects, and a `--json` refusal exiting 1 on stdout (the CLI tests refuse through `--preset` only). Caught: the bug put back at either builder, the method table written by hand or given a no-method entry, each refusal narrowed, a disagreeing width overridden by the method's, the FP8 gate ahead of the width rules, the preset branch resolving a width itself, and the command's `--quantization` read off the width; then the fix's own refusal of a width of 0 or less, dropped or narrowed to negatives. X4 is no longer kept: after that refusal, `if not bpp` behaves as `if bpp is None` does |
| `engine_r17_json_known_quantizations.py` | 17 | `fix/json-known-quantizations`: the quantizations a JSON config may name let slip — any name accepted, AMD given NVIDIA's list, the list checked on NVIDIA only or only without a width, gguf dropped from either vendor, a CUDA-only method accepted on AMD, the FP8 methods reduced to `fp8`, sized at 0.5 or missing one, and the refusal without its names |
| `engine_r20_json_quant_cold_check.py` | 20 | The cold check of `fix/json-known-quantizations` at `6a3a7f9`, on what the precision sweep holds fixed: the width rules kept to models under 100B, to models without latent attention, to one concurrent request or the default context; a no-width method sized by what its model id says; the zero refusal on integer widths only, a fixed width given as a float refused, one byte as a float left without its `--quantization`, and a JSON null read as the name `none`. Then weaker fixes: the vendor's list chosen by the card's LLVM target, the list consulted before the spelling is read or only with a width, its refusal raised as a TypeError, the FP8 gate seeing the literal `fp8` only, one byte overriding every FP8 method's name, the GGUF levels on every no-width refusal, a GGUF level's width passing without its method, the zero refusal or the contradiction check narrowed, and the command dropping `--quantization` for a method with no fixed width |
| `engine_r21_non_finite_numbers.py` | 21 | `fix/json-precision-width`: a JSON config's or the menu's NaN or Infinity planned or crashing again — the check dropped at `from_json()` or the menu, narrowed to NaN, to Infinity or to +Infinity, to the fields `validate_arch()` types or the request fields, to configs naming a preset or to NVIDIA cards, a non-finite value dropped for its default, the refusal without its field or raised as a TypeError, and the menu checking the parameter count only |
| `engine_r12_unresolvable_model_paths.py` | 12 | `fix/refuse-relative-model-paths`: the refusal of a model path the GPU server can't resolve, weakened fourteen ways — a shell variable let through anywhere or only inside an absolute path, `~` in any form or all but `~/`, `./`, `../`, `.` and `..`, and relative paths of three parts; a JSON config's path or the menu's left unchecked, or the menu's checked only after the display name; the refusal applied on AMD cards alone, or without its fix; and the opposite failure, a Hugging Face id or an absolute path refused |
| `engine_r13_model_path_cold_check.py` | 13 | The cold check of `fix/refuse-relative-model-paths`, along the axes round 12's tests hold fixed: the refusal gated on a single-device board, one GPU, a non-FP8 plan, a card that is not CDNA3, a dense model or the fabric; the menu's refusal escaping `main()`'s handler, the reason on stdout, exit 1, another prefix, or a warning and the placeholder path instead; the check after the FP8 refusal or after the PDF; an absolute path with `~` inside or a space refused, a one-part name or a dotted hub id refused; a bare `~`, `$MODEL` without a `/`, a menu answer checked before it is stripped, or a dotted three-part path planned; and the menu's refusal without its fix; then the fix's own trim of a JSON config's model path, dropped or made left-only. T1, the path refusal moved after the FP8 one, is not kept: the path is never printed either way |
| `engine_r14_model_path_cold_check_2.py` | 14 | The second cold check of `fix/refuse-relative-model-paths`, along what round 13's sweeps still hold fixed: every generated path runs on one configuration, and every configuration on one sample path per kind (`$HOME/.hidden/m`, `~`, `.`), so a rule inside a kind keyed to the card, the board count or the weight precision, the trim keyed to the model, and the whole refusal keyed to an axis the sweep never varies (FP8 KV cache, 4-bit or GGUF weights, the context length, four boards and more); the shapes the generated set lacks (a three-part path ending in `/`, a doubled `/`, a trailing `$`); the menu's trim of spaces only and the JSON trim of spaces and tabs only; the reason precedence for `~/$HOME/m`; and six controls expected caught (a vendor gate, the tilde rule skipped on a dot, a four-part threshold, the trim in one branch, exit 3 from the menu, `~` and `$` expanded on the planner's machine) |
| `engine_r15_json_precision_width.py` | 15 | `fix/json-precision-width`: a JSON config's width guessed again — the old 0.5 default, every method at 0.5 (FP8 included), GGUF or an unknown method planned without a width, a method and a width that disagree or a width with no method planned, the preset branch dropping the width a JSON gave, GGUF treated as having one width, the rules kept to NVIDIA or to one board, and the GGUF refusal without its levels |
| `engine_r16_json_width_cold_check.py` | 16 | The cold check of `fix/json-precision-width`, on what round 15's sweep holds fixed. Six survived it at `3d1e37e` (X4–X9), and the tests that catch them now vary each of those axes: a width of 0 treated as missing at either builder, the rule kept to dense models (the sweep's only preset is llama31-8b), a config naming nothing sized at its KV cache's width (always 2 in the sweep), the width looked up by `--prec`'s token so `int4`, `bf16` and `q4km` are planned with a `--quantization` vLLM rejects, and a `--json` refusal exiting 1 on stdout (the CLI tests refuse through `--preset` only). Caught: the bug put back at either builder, the method table written by hand or given a no-method entry, each refusal narrowed, a disagreeing width overridden by the method's, the FP8 gate ahead of the width rules, the preset branch resolving a width itself, and the command's `--quantization` read off the width; then the fix's own refusal of a width of 0 or less, dropped or narrowed to negatives. X4 is no longer kept: after that refusal, `if not bpp` behaves as `if bpp is None` does |
| `engine_r18_hf_model_input.py` | 18 | `fix/hf-model-input`: a JSON config's model path that isn't a path let back in — the type check dropped or widened to allow null, an empty path planned, a leading `-` planned, or only `--` refused |
| `engine_r19_hf_model_cold_check.py` | 19 | The cold check of `fix/hf-model-input`, along what round 18's tests hold fixed: the wrong-type check keyed to the card's vendor, the board count or the weight width its one test never varies, or run only for a truthy value so a `false`, `0`, `[]` or `{}` is refused as empty instead of named; `main()` turning the TypeError into a warning and exit 0, which no CLI run observes; the menu's refusal skipped for MoE models, where the menu is driven on one dense architecture; and a trim of spaces, tabs and newlines only on either route, where every padding typed is one of those. Then the controls expected caught: the dash rule narrowed to a letter or digit or judged after the relative one, a whitespace-only path planned untrimmed or given the placeholder, the preset branch's `or` fallback, the menu's default applied before the trim, a wrong type coerced to text, and the empty path raised as a TypeError |
| `engine_r21_non_finite_numbers.py` | 21 | `fix/json-precision-width`: a JSON config's or the menu's NaN or Infinity planned or crashing again — the check dropped at `from_json()` or the menu, narrowed to NaN, to Infinity or to +Infinity, to the fields `validate_arch()` types or the request fields, to configs naming a preset or to NVIDIA cards, a non-finite value dropped for its default, the refusal without its field or raised as a TypeError, and the menu checking the parameter count only |
| `engine_r22_suite_speedups.py` | 22 | `chore/corpus-speedups`: the suites' speed-ups undone the ways that would change a verdict — the report suite's parse memo keeping a parse that failed, or serving a build outside `story_strings()`; the sync suite's shared node process running every block in one context, dropping or not naming a failure, or answering every block with the first one's object |
| `engine_r23_speedups_cold_check.py` | 23 | The cold check of `chore/corpus-speedups`: a speed figure carried by a *style's* `bulletText`, which reportlab's Paragraph reads when none is passed and the parse memo copies from the first paragraph it saw with that markup and style name — set for a card without constants only, for one with constants only, and on the `Small` style; the same figure passed positionally and by keyword as controls; the memo's key without the style name and its two argument gates dropped, alone and each with the control it lets through; the shared node process run in node's own context, in one shared context, answering a failed block with null, in reverse order, and without its count check. Its `RUNNER` dict, judged by `tests/corpus.test.py` with `--runner`, weakens the corpus runner's once-per-run proof: the commit check, the clean check and its untracked files dropped, the check never called, the shared golden set ignored, and the green baseline skipped |
| `engine_r24_speedups_cold_check_2.py` | 24 | The second cold check of `chore/corpus-speedups`, at `1572e53` against `97a0309`: the figure carried by `ParagraphStyle`'s *class* `bulletText` for a card without constants only, around the Cost heading and the Notes heading — reportlab reads the bullet through `getattr`, the memo compares the style by `vars()`, and the first paragraph with that markup is a card with constants, so both survived at `1572e53` and were caught at `97a0309`; the heading's own bullet set after it is built, a same-named style with its own bullet, and a `js_str` escape dropped, as controls; and one sabotage a page golden and a real report test catch together, which the runner must not call golden-only. Its `RUNNER` dict, judged with `--runner`: golden-only claimed when any red suite is golden-only or whenever a sabotage is caught, `--early-exit` stopping at a suite red on golden failures alone, the dirty check between sabotages dropped, the sabotages or the harness read from the runner's own checkout, the bytecode guard dropped, and a driver's log saying exit 0 over an error |
| `engine_r25_speedups_cold_check_3.py` | 25 | The third cold check of `chore/corpus-speedups`, at `b520f07` against `3496a57`: the figure reaching a heading's bullet by the two routes `style_state()` cannot see — a `__getattr__` on `ParagraphStyle`, and a bullet that is callable — for a card without constants only, both of which survived at `b520f07` and were caught at `3496a57`; as controls, a bullet `property` on the class, the `__getattr__` figure for a card with constants only (which the memo then serves to every later card, failing a test for a card the sabotage never touched) and two `js_str` escapes dropped. Its `RUNNER` dict, judged with `--runner`: the shell driver, the driver list or the sabotage names taken from the runner's own checkout, `--compare` reading statuses alone or passing a driver one run lacks, `--early-exit` running only the suites `ORDER_HINT` names, a run with survivors exiting 0, and `--ref` ignored for the runner's own HEAD |
| `engine_r27_fp8_default.py` | 27 | `fix/fp8-default`: AWQ put back as the default in the page's markup, in `--prec` or in the interactive menu, one at a time; FP8 left the default on a card vLLM has no FP8 weight kernel for, in the page (only the default, a chosen FP8 still moved), in `--prec`, and in the menu; the stand-in path for AWQ, GPTQ and GGUF on a preset's own repo dropped from one engine, one cfg builder, the copied report, the PDF, the command box or the ROCm command only; a path the user gave (an imported id, a JSON `hf_model` with or without a preset, the menu's typed id) replaced; the line over the stand-in missing for one method in one engine, or for GPTQ in both alike; `--tokenizer` missing from the GGUF command in one engine or both; and a stand-in the path rules or the PDF would refuse (relative, or carrying `<...>`) in both |
| `engine_r37_fp8_default_cold_check.py` | 37 | The cold check of `fix/fp8-default` at `953ecd9`, against the requirement alone: the page's markup starting on AWQ, the control's BF16 fallback dropped, never given back, not ended by a reader's own pick, made on every AMD card or on none, falling back to AWQ, left without a call site, or a link's precision no longer restored; the FP8 predicate changed in both engines at once so parity sees nothing (gfx90a marked as loading FP8, an unknown target let through, the gate following the silicon's tensor cores) and both builders printing FP8 on a blocked card; `--prec`'s default FP8 or BF16 everywhere, AWQ where blocked, fixed by argparse, an explicit `bf16` read as no choice, an explicit `fp8` quietly downgraded; the menu's default pinned to option 1 or 2, FP8 kept on a blocked card, the default read off the tensor cores; the stand-in in both engines at once — a user's path replaced, GGUF or FP8 in or out of the set, `--tokenizer` missing, on every method or naming the stand-in, the path relative or holding a space, the line dropped, moved after the command, given an apostrophe or angle brackets; and on one engine or surface — the PDF dropping the first line, the JSON, menu or CLI path mis-flagging the preset's repo, the page never or always naming a stand-in |
| `workflow_r1_gate.py` | 1 | The gate that left the price job able to succeed only by finding nothing, and the seven ways a cold check rebuilt it afterwards without touching a guarded field |
| `workflow_r2_yaml_reader.py` | 2 | The round-1 guard's own reader and rules: a delivery condition gated on the suite in a FOLDED second line, a step respelled until the reader and its own count check both dropped it, an object filter that gates delivery while naming no step, a test run moved upstream of the gate, and `.conclusion` for `.outcome` |
| `workflow_r3_name_bindings.py` | 3 | The round-2 guard's bindings: a second job in the same file carrying the original bug where every rule read one literal job key, a job-level `if:` that retires the job as "skipped", a decoy `echo` that takes `id: suite` so the PR body reports its outcome forever, and `format()` hiding a filename from the path regex |
| `workflow_live_first_run.py` | live | Not from a review: what the first real run of the fixed price job found. `add-paths` listed `assets/*.png` and make_assets.py writes three files there, so the manifest recording index.html's hash was regenerated and then left behind, failing the asset gate on every price PR |
| `workflow_r4_shell_quoting.py` | 4 | The guard's own helper: six shell-quoting shapes that hid a test run from `executes()` (quoted argument, command substitution, `$'...'`, an apostrophe in a comment that opens a span across newlines), plus a rule that passed by not looking, a negative pathspec, `UPDATE_GOLDEN` through `env:`, and the gate — which nothing pinned at all |
| `workflow_r5_off_the_hour.py` | 5 | `fix/price-refresh-off-the-hour`: the price job's schedule put back at the start of the hour, where GitHub delays and drops scheduled runs — as `0`, `00`, `*/30`, a list holding `0` and `*` — the schedule emptied so the rule has nothing to judge, another workflow scheduled on the hour, and `on:` quoted so a reader keyed to YAML's boolean finds no triggers |
| `workflow_r6_schedule_cold_check.py` | 6 | The first cold check of `fix/price-refresh-off-the-hour`: the four sabotages that survived it — a cron that keeps minute 17 and never fires (February 31st) or fires monthly rather than weekly, and a step-level `if:` keyed to `workflow_dispatch` on the fetch step or on the diff step, so the job stays scheduled off the hour and every scheduled run delivers nothing |
| `workflow_r7_scheduled_run_path.py` | 7 | The neighbours of round 6's survivors, which its fixes close as classes: a schedule at a good minute that runs more often than once a week (every weekday, every six hours, a second weekly cron beside the first), and a condition on the steps the check did not try — the checkout and the re-sync before the gate, and the install, image and suite steps between the gate and the suite |
| `workflow_r8_schedule_cold_check_2.py` | 8 | The second cold check of the schedule fix (514f3e7): the ten sabotages that survived it. The fetch and re-sync commands, which no rule reads — `--apply` chosen by an expression on the event name, the fetch wrapped in a shell `if`, `--apply` dropped, `--slug` narrowing a scheduled run, the applied moves discarded after the re-sync; a named step before the gate whose shell exits 1 on scheduled runs; the job's envelope — `runs-on` chosen by the event name, `needs:` a job that fails on schedule; a `push:` trigger beside the schedule; and `on:` defined twice, which pyyaml reads and GitHub refuses. Also records that `17 6 * * MON` was wrongly refused (since fixed) |
| `workflow_r9_pinned_head.py` | 9 | The neighbours of round 8's survivors, which its fixes close as classes: a new step before the gate, a `shell:` or `env:` on the fetch and a `shell:` on the gate, a checkout of another ref, another Python, `defaults:`, `env:` or `container:` around the job, a `workflow_run` trigger, no hand run, and one step given two `run:` keys |
| `workflow_r11_price_watchdog.py` | 11 | The price watchdog, which checks a scheduled run's outcome because round 10 (below, accepted risk) showed reading the workflow cannot: an unfinished, failed or reportless run taken as delivered, the report's words ignored, a pull request closed unmerged or carrying only last week's commit or a person's taken as delivery, the slot judged with no grace, on Python's weekday or a week early, a problem left green, no issue or a new one each week, and the watchdog's own workflow run inside the grace, unable to open its issue, or reading another artifact |
| `workflow_r12_watchdog_cold_check.py` | 12 | The cold check of the price watchdog at `c9c4a06`: the sixteen of nineteen sabotages that survived it, where round 11 did not look. The questions it asks GitHub, which the end-to-end fake answers whatever the arguments say: a hand run taken for the scheduled one (`event=schedule` dropped), a merged pull request invisible (`state=all` dropped), pull requests from any branch, every workflow's scheduled runs so the watchdog's own in-progress run is judged, every artifact downloaded under its own directory so the report is never at the path read, and the problem added to a closed issue; a hand run's commit standing in for the scheduled run's (dated after the slot, not the run); `main()`'s real clock naive, which every test bypasses with `--now`; `WEEKDAYS` reordered, latent while the cron says `1` and the test derives its expectation from the table; the watchdog's workflow kept green with `\|\| true`, retired by a step-level or job-level `if:` on `workflow_dispatch`, run without `GH_TOKEN`, or pointed at another repository; and the price workflow's RED marker kept only in a shell comment, and its artifact expiring a day after upload. Caught: `REPORT_FILE` as a substring of the real name, the watchdog's cron at minute 0, the body step writing to another file |
| `workflow_r13_watchdog_cold_check_2.py` | 13 | The second cold check of the price watchdog, at `4719776`: the fourteen of eighteen sabotages that survived it, where `tests/watchdog.test.py` still does not look. `main()`, which the verdict test never calls and the end-to-end run exercises on three outcomes only: it returns 0 for a run still going, for no run at all and for a cancelled one, accepts `skipped` beside `success`, reads a missing report as nothing found, and lets a hand run stand in for a dropped scheduled one; what it read from the price workflow, checked for what the readers return and never for whether `main()` uses them (a hardcoded `0 6 * * 1`; the file, the branch and the commit message as literals); the grace at one hour, and a commit dated up to an hour before the run's start; `status=completed` on the runs query, a field the fake ignores; and the price job's own actions, which do more than the words the watchdog reads: a workspace path in the artifact (upload-artifact roots it at the least common ancestor of its paths, so the report is never at the path read), a `branch-suffix` on the pull-request step, and a `timezone:` under its schedule entry, which GitHub's schema allows and the watchdog reads as UTC. Caught: `on:` given twice, the body step's condition widened around the gate, and the upload gated on `changed == 'false'` again |

The suites that judge a sabotage are listed once, in `harness.py`'s `JUDGING_SUITES`, and
`suites.sh` runs the same set for the one bash driver, printing one line each. `assets.test.py` is deliberately excluded — it hashes `index.html` and goes red on
any edit, so it would report every sabotage as caught regardless of what the sabotage did.
`tests/corpus.test.py` is excluded for the same reason — it checks the anchors still match the
engine, and that every sabotage in every driver still applies, both of which every sabotage breaks.
It runs in the ordinary suite instead, so a pull request that moves engine text a sabotage quotes,
through `anchors.py` or inline, goes red in its own CI rather than at the next corpus run. It
also refreshes the catalog in memory four ways, with the price job's own writers and the tiers
its `SOURCE_MAP` reads: every reading re-read on a new day, every price moved to one sharing no
text with it, every tier still held under a note confirmed at a new price, and every automated
tier's reading and note taken away, as a person could. It does all four again from the tree the
pull request confirming the held tiers leaves, the one line that pull request edits by hand
included: the pinned held set in `tests/price_check.test.py`. Each time it loads every driver
against the result and applies every sabotage again, so one that quotes a value the job rewrites
goes red in the pull request that adds it, not in the bot's, and one whose target is gone has to be
refused by name, never dropped and never renamed: every name a driver has before a refresh, it has
after. The shell driver is run, not read: its heredoc executes against each refreshed catalog and
has to reach its write. A sabotage that makes no edit, or expects its text 0 times, changes nothing
and is refused, by a run and by this check alike. It writes nothing to disk. The refusals it lets pass are the ones the refreshed
tree has to cause: after the fourth, any; otherwise a sabotage built on a note a reading has
replaced, in that refresh or an earlier one, which says so with
`harness.Missing(..., note=(slug, tier))`. The check of the tree as it is lists such a refusal
and does not count it.
`workflow.test.py` joined the judges with `workflow_r1_gate.py`: without it a workflow sabotage reads green,
because none of the other five opens `.github/`. `watchdog.test.py` joined them with
`workflow_r11_price_watchdog.py`, for the same reason: no other suite runs `tools/price_watchdog.py`.

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

The price workflow's guard stopped hardening after its third cold check (round 10, 2026-09-28),
whose ten survivors are kept on `chore/schedule-cold-check-3` (`75c8a89`) rather than in the corpus,
where they would read as survivors on every run. They are **accepted in the file and caught at run
time** by the price watchdog (`tools/price_watchdog.py`, round 11), which checks a day later that
the scheduled run happened, succeeded, left its report and, when it found prices, that a pull
request carries a commit it made:

- A script between the gate and the pull request that throws the moves away (install, image or
  PR-body step): no commit from the run reaches a pull request
- A new step after the suite wearing one of the two permitted conditions, such as a second
  checkout (which cleans the tree) or a step that closes the pull request: the same
- A key GitHub's parser rejects (`timezone:` under `concurrency:`), so the file never runs: no
  scheduled run starts

Three rounds found 4, 10 and 10 survivors, each ring outside the last; reading a workflow cannot
prove a scheduled run will do its work, and watching the outcome can.

The corpus check's second cold check (`tests/corpus.test.py`, on chore/derive-catalog-anchors at
`97a2189`) left three ways past it that no test can close, accepted by the owner on 2026-09-30.
Its driver is kept on `chore/derive-anchors-cold-check-2` (`82177d9`) rather than in the corpus:

- A test that hard-codes a held tier's state, as `model.test.js` once named h100-80 its mixed row,
  is green on the tree as it is and red only on the pull request that confirms the tier (G3e).
  Catching it sooner would mean running `model.test.js` and `report.test.py` a second time, on the
  simulated confirming tree, in every suite run. It costs a confusing red on a pull request that is
  handled by hand anyway.
- A sabotage that switches to another target once its own is gone, keeping its name, still applies
  (Q13): nothing tells it from one written that way on purpose. The rule to refuse by name, above,
  is what stands against it, in review.
- A sabotage whose only edit claims a note was replaced by a reading is excused whenever that tier
  has one, so it can stay refused indefinitely (Q18). The check lists it among those refused by
  name, and a full run counts it under "could not be applied".

## Reading a result

- `N caught, 0 survived` — no gap found. This is the common outcome once a commit is careful.
- `<-- SURVIVED` — the suite did not notice. Write the test that notices it, then re-run.
- `driver errored` — usually a target string that no longer exists, which means the driver has
  drifted from the code. Fix the driver in the commit that moved the code.

A driver that exits 0 having judged nothing is the worst outcome, because it reads as a pass.
`engine_r1_perfkey_typo.sh` (then `sab3.sh`) did exactly that until 2026-09-20: it called a helper that no longer existed, ran no
suite, restored cleanly and returned success. Every driver now fails loudly when it cannot
judge, and `chain.sh` treats a non-zero exit as a finding rather than noise.
