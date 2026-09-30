# v1.1 Health Report — vllm_planner

Written 2026-09-30 against master @ `e523fd5` (2026-09-30T09:53:06+03:00), before the v1.1
release commit, with PR #40 and #41 open. One report per release; feeds one decision: does
the next stretch go to maintenance or features. Every number below names the command that
produced it. Where a number could not be measured cleanly, that is said instead of guessed.

**Since it was written:** #40 merged at 10:15 UTC (master `a5bd550`). It adds the watchdog suite
(9 tests) and 6 workflow tests, so the suite is 822 tests at master's tree (`./tests/run.sh` at
`466f50a`, whose tree master's merge commit carries). At `a5bd550` it took 10.11 s and 10.03 s
(`time ./tests/run.sh`, twice, 2026-09-30 ~11:57 UTC, load average 6.55 falling to 5.22 as a
corpus run wound down). Written by a no-context agent (Sonnet),
then checked by the main session: **six figures were corrected**, one updated, and the wording of eight claims
tightened before publishing, all listed at the end.

## Cost signals — what it now costs to change this code

**Suite size** (`./tests/run.sh`, e523fd5): 807 tests, 9 suites, 0 failures.

| suite | tests |
|---|---|
| benchmark coverage | 9 |
| model math (JS) | 182 |
| JS/Python parity | 301 |
| report generator | 159 |
| generated data blocks | 23 |
| price-refresh checks | 91 |
| CI workflows | 20 |
| sabotage corpus reaches engine | 17 |
| published images | 5 |

**Wall clock**, run twice back to back (`time ./tests/run.sh`):
- Run 1: real **9.232s** (user 8.778s, sys 1.066s). `uptime` before: load avg 0.09/0.12/0.09
  (11:56:31); after: 0.23/0.15/0.10 (11:56:40).
- Run 2: real **9.455s** (user 8.921s, sys 1.117s). `uptime` before: 0.26/0.16/0.11 (11:56:58);
  after: 0.37/0.18/0.12 (11:57:08).
- Load climbed across the two runs (0.09→0.37); nothing else was intentionally started, but
  per the task's own caveat this machine may not have been idle. Both runs still land within
  0.25s of each other.

**Sabotage corpus size** (`tests/sabotage/`, e523fd5): 39 driver files (34 `engine_r*`, 5
`workflow_r*`/`workflow_live*`; 38 Python + 1 shell). The last full run on that tree (`c797eab`,
below) judged **800 tasks: 799 Python sabotages and the shell driver**. A grep for `S["` misses the
sabotages built in loops or written as dict literals; the runner's count is the one to use.
At `ca1d573` this directory did not exist (`git ls-tree ca1d573 -- tests/sabotage` = empty). The
drivers were **moved in** from gitignored `tmp/`, where earlier review rounds had built them, on
2026-09-20 (`fd02cf6`, "Keep the sabotage corpus with the code it attacks, not in gitignored
scratch"). On 2026-09-22 (`1005900`) the directory held 16 drivers (`git ls-tree 1005900`).

**Wall time of recorded corpus runs** (`tmp/sabotage/parallel-*/results.json`, 26 runs on disk,
2026-09-27→2026-09-30; these are the maintainer's local run logs, not in the repository). The two most recent full runs against commits actually on master:
- `c797eab` (2026-09-29, on master — "Merge master into chore/corpus-speedups: #39 landed"):
  **wall_s 1213.1 (~20.2 min)**, finished 2026-09-29T13:20:03Z, 800 judged outcomes, 799
  `caught` + 1 `shell`, 0 `missed`, early-exit mode.
- `8670c3d` (2026-09-29, on master): wall_s 1151.4 (~19.2 min), 773 judged outcomes, finished
  2026-09-29T07:04:44Z, early-exit mode.
- 3 of the 26 recorded runs were not on master when this was written: `466f50a` is #40's
  merge commit (on master since #40 merged), and `442cebe` and `97a2189` are #41's.

**Growth since 2026-09-20** (`ca1d573`, 474 tests). Verified by `git archive ca1d573` into an
isolated scratch copy and running its own `tests/run.sh` there (no checkout of the main repo):
152+133+112+15+57+5 = **474**, matching exactly.
- Suite: 474 → 807 (**+333, +70%**). 46 of that is 3 whole suites that didn't exist at
  `ca1d573` (coverage 9, workflow 20, corpus 17); the remaining 287 is growth inside the 6
  suites that already existed (parity +168, report +47, price_check +34, model.test.js +30,
  sync +8, assets +0).
- Corpus: moved into the repo on 2026-09-20 with the rounds built before in `tmp/`; 16 drivers on
  2026-09-22 (`1005900`); 39 drivers and 800 judged tasks at master's tree on 2026-09-29.

**CI time** (`gh run list`, last 40 runs, 2026-09-28→2026-09-30): the `Tests` workflow's runs took
**~24–62 s**.
The one outlier is not CI time: the `Tests` run on `automation/price-refresh` created
2026-09-28T12:45:41Z has **no jobs** (`gh api .../actions/runs/36423909044/jobs` returns none), and
its `updated_at`, 2026-09-29T12:27:08Z, is one second after #39 merged. It never ran, so the gap
is not CI time.

**Merged PRs since 2026-09-20** (`gh pr list --state merged --search "merged:>=2026-09-20"`,
files via `gh pr view <n> --json files`): 24 PRs (#16–#39), 257 file changes across them, counted
by file, not by line: **50 (19%)** were to what users get (`index.html`, `generate_report.py`,
`data/`, `benchmarks/`, `README.md`, `ROADMAP.md`, `docs/research/`, `assets/`); **207 (81%)** were
to tests, tooling or CI. Both open PRs continue the pattern: #40 is 17 files (1 user-facing /
16 tests-tooling), #41 is 11 files (0 / 11).

## Value signals — what the tool gives its users now

**Benchmark coverage** (`tests/coverage.test.py`, via `./tests/run.sh`, e523fd5): **13 entries,
5 measured vLLM batches, 3 of 17 catalogued cards have a measured number** (a100-80, b200-192,
h100-80). 12 cards — including all 5 AMD cards — have no benchmark entry at all.

**Price tier provenance** (`data/gpus.json`, 17 rows × 3 tiers = 51 pairs, parsed with
`python3`/`json`):

| tier | read by the weekly job (priceSource) | recorded by hand (priceRecord) | priced, no source confirms it (a note says why) | no price at all (a note says why) |
|---|---|---|---|---|
| hyper | 10 | 0 | 0 | 7 |
| spec | 7 | 2 | 5 | 3 |
| spot | 2 | 0 | 11 | 4 |
| **total** | **19** | **2** | **16** | **14** |

Every one of the 51 (row, tier) pairs says how its figure was reached, or why there is none.
**21 carry a named, dated source (19 read by the weekly job, 2 recorded by hand), 16 show a price
no source confirms, and 14 show no price.**

**ROADMAP.md v1.1 vs master**: 7 subsections under v1.1.0. 6 are marked "shipped, not yet
released": "Correct above 8 GPUs", "A throughput ceiling that holds for FP8", "A weekly price
check", "Documents and images that match the tool", "Unmodelled hardware says so", and "The
documents say what the tool shows".

The 7th, **"AMD / ROCm — researched, not yet built," is stale**: master already carries 5 AMD
catalog rows, each with its own `perfKey` (`rx7900xtx-24`→rdna3, `mi210-64`/`mi250x-128`→cdna2,
`mi300x-192`/`mi325x-256`→cdna3; `python3`/`json` on `data/gpus.json`). No AMD key has measured
constants, so throughput shows as not modelled, which is what ROADMAP.md promises. They were shipped in PR #30
(`feat/amd-gpus`, merged 2026-09-27) and PR #32 (`feat/rocm-guidance`, merged 2026-09-27), and
hardened by at least 4 dedicated sabotage drivers (`engine_r6_amd_rows`,
`engine_r7_amd_cold_check`, `engine_r10_rocm_guidance`, `engine_r11_rocm_cold_check`).
`ROADMAP.md` itself was last edited 2026-09-23 (`git log -- ROADMAP.md`, commit `0c3e489`) —
4 days *before* the AMD PRs landed, and has not been touched since.

## What this shows

- **Cost side**: most of what changed since 2026-09-20 was the safety net. 81% of the merged
  pull requests' file changes were tests, tooling or CI, beside the product changes the roadmap
  lists, and the corpus moved into the repo and grew to 800 judged tasks, a run of about 20
  minutes. The suite runs in about 9–10 s and CI in under a minute; the corpus run is the expensive
  part.
- **Value side**: every v1.1 item in ROADMAP.md is on master. Six are marked "shipped, not yet
  released", and the seventh, AMD / ROCm, is live though still marked "not yet built": the
  documentation lags the code, not the other way around. Benchmark coverage is thin (3 of 17
  cards), and every AMD card is in the uncovered set. The price catalog is fully accounted for (no
  silent gaps): 21 of 51 tiers carry a named, dated source, 16 show a price no source confirms, and
  14 show none.
- Both of these are descriptions of what the evidence shows, not a recommendation. Whether the
  next stretch goes to maintenance (paying down/updating docs, closing benchmark coverage) or
  features (catalog import, benchmark ingestion in CI per the v1.2.0 roadmap entry) is the
  owner's call.

## Corrections by the main session (2026-09-30)

Checked against the repository after the agent finished. Items 1–6 were wrong or misleading as
first written; item 7 is what changed since:
1. **Sabotages:** the agent counted "305" by grepping `S["` lines. Most sabotages are built in
   loops, and the runner's own count on this tree is 800 judged tasks (the `c797eab` run).
2. **Corpus origin:** the corpus is not "entirely new since 2026-09-22". It moved into the repo
   from `tmp/` on 2026-09-20 (`fd02cf6`).
3. **Price tiers:** the table had no null tiers. 14 tiers carry no price, and 16 carry a price no
   source confirms. The agent had counted every note as one category.
4. **AMD:** the rows have their own `perfKey`s but no measured constants. The agent's "real
   `perfKey` constants" read as the opposite.
5. **The 23.7 h CI "anomaly":** a run with no jobs, closed a second after #39 merged. It was
   never CI time.
6. **Off-master runs:** `442cebe` and `97a2189` are #41's, not #40's.
7. **Since written:** #40 merged, and the suite is 822 tests at master's tree.

## Tightened before publishing (2026-09-30)

Each claim was checked against the repository before this file went into `docs/`, and eight were
worded past what their evidence shows:
1. "this release built the safety net, not the product": the product changed too (AMD rows, ROCm
   guidance, price provenance). It now says most of the file changes were the safety net.
2. "The net hasn't made everyday iteration slower": no measurement from before exists here to back it.
3. "Most sabotages are built in loops": not counted. It now says what a grep misses.
4. "regardless of PR size": from two pull requests. Dropped.
5. "runs no CI by design": more than the claim needs. It now says the run never ran.
6. "a confirmed source": two of the 21 are hand records, named and dated but not re-checked weekly.
7. "each has a corresponding passing suite": not established. The roadmap items are now quoted by
   their own headings.
8. The merged-PR split now names its unit, file changes, and the run logs are marked as local.
