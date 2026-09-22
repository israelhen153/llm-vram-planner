---
name: sabotage-corpus
description: "Run this repo's sabotage corpus against a change before it ships — reintroduce each fixed bug and check the suite goes red. Use when cold-checking a commit here, when adding a GPU vendor or catalog field, or when a review round keeps finding cases the last one missed."
allowed-tools: Read, Bash, Grep, Glob
---

# Sabotage corpus — llm-vram-planner

Full reference: `tests/sabotage/README.md`. This is when and how to reach for it.

## Before you start

**Commit.** The drivers restore with `git checkout -- <file>`, which restores the index and not
unsaved edits. This has destroyed work here twice, the second time two hours after the lesson was
written down. `chain.sh` refuses a dirty tree; do not work around it.

Never `git stash` — the stack is shared across worktrees.

## Running

```
tests/sabotage/chain.sh                 # every driver, ~15 min
tests/sabotage/chain.sh engine_r3_fixed_assumptions engine_r3_identical_estimate_and_href engine_r3_href_on_unmatched_word   # engine round 3, ~3 min
python3 tests/sabotage/engine_r1_throughput_leaks.py W11   # one sabotage by name
```

`N caught, 0 survived` is the pass. `<-- SURVIVED` means the suite did not notice — write the
test that notices, then re-run. `driver errored` usually means a target string no longer exists,
so the driver has drifted from the code: fix it in the commit that moved the code.

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
