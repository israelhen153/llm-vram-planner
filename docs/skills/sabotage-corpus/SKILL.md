---
name: sabotage-corpus
description: "Run this repo's sabotage corpus against a change before it ships — reintroduce each fixed bug and check the suite goes red. Use when cold-checking a commit here, when adding a GPU vendor or catalog field, or when a review round keeps finding cases the last one missed."
allowed-tools: Read, Bash, Grep, Glob
---

# Sabotage corpus — llm-vram-planner

Full reference: `tests/sabotage/README.md`. This is when and how to reach for it.

## Before you start

**Commit what you want judged.** `parallel.py` judges a commit (`--ref`, default `HEAD`) in its own
worker worktrees, so uncommitted work is neither judged nor touched, and your checkout stays free
while it runs. `chain.sh` judges the checkout it runs in, and its drivers restore with
`git checkout -- <file>`, which restores the index and not unsaved edits. That has destroyed work
here twice, the second time two hours after the lesson was written down. `chain.sh` refuses a
dirty tree; do not work around it.

Never `git stash` — the stack is shared across worktrees.

## Deleting anything is the owner's call

Nothing here deletes on its own, and neither does a session. `parallel.py` keeps its workers in
`tmp/corpus-workers/` for the next run; a dirty, locked or missing worker stops the run and is
listed, not cleaned. When a run or a check leaves anything behind (workers, a checker's worktree,
a branch, scratch logs):

1. **List it for the owner**, with why each item is safe to remove: the tree is clean, and every
   commit it holds is also on a branch that stays.
2. **Wait for the owner's approval.** A request to delete is not an approval.
3. **Delete, then move on:** `git worktree remove <path>` without `--force` and `git branch -d`,
   so a target that changed since you looked is refused rather than destroyed.

The rule is the owner's, set on 2026-09-25 when the runner was built.

## Running

```
tests/sabotage/parallel.py --early-exit          # every driver, stopping each sabotage at its first real catch
tests/sabotage/parallel.py                       # every driver, every suite for each sabotage
tests/sabotage/parallel.py --ref <commit> engine_r11_rocm_cold_check   # named drivers, another commit
tests/sabotage/parallel.py --subset --compare <serial logs> <early-exit logs>   # one run held to another
tests/sabotage/chain.sh                          # every driver in series, in this checkout: the reference
python3 tests/sabotage/engine_r1_throughput_leaks.py W11   # one sabotage by name
```

**What a full run costs**, measured at 567 sabotages on this 4-core, 8-thread laptop:

| Runner | Wall clock | Measured |
|---|---:|---|
| `chain.sh` | 2 h 12 min of running | 2026-09-24, at `d99f1b6` |
| `parallel.py`, 8 workers | 31.2 min | 2026-09-25, at `d99f1b6`, cold checks running alongside part of it |
| `parallel.py --early-exit`, 8 workers | 11.7 min | 2026-09-25, at `d99f1b6`, a cold check running alongside |

Estimate a run from its sabotage count and a pace you measured, never from a figure written down:
this file said ~15 min when the real figure was 2 h 12 min, and the suites keep growing. Both runners
hold a sleep lock where systemd provides one. Closing the lid still suspends.

`N caught, 0 survived` is the pass. `<-- SURVIVED` means the suite did not notice — write the
test that notices, then re-run. `driver errored` usually means a target string no longer exists,
so the driver has drifted from the code: fix it in the commit that moved the code.

**A catch only a golden makes is not a catch you can keep.** Regenerating the goldens hides it.
`parallel.py` lists them after every run, in `golden-only.txt`, and finds which failures are golden
ones by emptying the goldens rather than by reading test names. The six accepted after round 3
(W5, W6, W7, W11b, X2 and K8) are the only ones meant to stay; any other is a test to write.

**Before trusting a change to the runner, hold it to the serial run.** `--compare` checks a
parallel run sabotage by sabotage; `--subset` does it for an early-exit run, which runs fewer suites
by design. Statuses alone are not enough: an early-exit run once matched the serial one on every
status while 90 of its catches came from a stale `.pyc`.

## Adding to it — what this repo's history says you will miss

Three rounds of cold checks found **zero behaviour defects and forty-plus test gaps.** Every gap
had the same shape: a probe that held fixed an axis the real data varies.

- **Round 2** found every probe running at `devices: 2` while every catalog row is `devices: 1` —
  a leak gated on single-chip cards passed the whole suite.
- **Round 3** found every probe fixing `vendor: 'nvidia'` and a single unknown key, so **a leak
  behind `vendor === 'amd'` or a CDNA `perfKey` would have passed.**
- **Round 3** also found four renderers never exercised, because the harness enumerated its
  renderer list instead of deriving it.

So, when you add sabotages:

1. **Derive probe parameters from `data/gpus.json`**, never from whichever fixture was handy.
2. **Vary the axis your change introduces.** A vendor commit needs AMD-gated probes; a field
   commit needs probes where the field is absent, wrong-typed and unknown.
3. **Derive any list the harness walks**, and assert the derived list is complete.

## What is not covered

Accepted residual risk, still open: a gauge carrying a figure with no text, and invented prose no
rule recognises as a claim. The `href` and every-card and `document.title` routes were accepted
after round 3 and are now closed by the golden (PR #16) — do not re-file them as open.

## Related

The golden (`tests/golden/`) and this corpus answer different questions. The golden asks *did the
displayed output change*; the corpus asks *would the suite notice if the bug came back*. A change
that passes one and not the other is telling you something.
