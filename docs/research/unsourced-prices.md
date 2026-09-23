# NVIDIA prices with no recorded source

**Read:** 2026-09-23.

**Why this page exists.** The owner defined tier honesty on 2026-09-23: *"we don't hide anything and
allow users to audit us completely and see how we got to what we have."* Each price tier either:
- was read automatically by the weekly check (`priceSource`);
- was read by hand from a named page (`priceRecord`);
- or now says why it has neither (`priceNote`).

This page records the evidence behind the notes on the 20 NVIDIA tiers in the third group.

**How it was read.** Where each figure came from was traced in git. For every commit that touched `index.html` or
`data/gpus.json`, the tier's value was read; the table gives the start of the last unbroken run at today's value. Before
2026-08-19 the prices lived in the GPU `<option>` values of `index.html` (`gb|bw|hyper|spec|spot|tflops`). The reason no
weekly reader covers each tier is the note in `tools/price_check.py`'s `SOURCE_MAP`.

## Where the figures came from

- **Every one of the 20 arrived in one of two commits, and neither names a provider:**
  - `597df81`, "Initial commit", 2026-05-28;
  - `de26df2`, "Add four GPUs and correct two bandwidth figures", 2026-07-27. It says only "Prices are mid-2026 list and drift; the UI already says so."
- None has changed since.

| Row | Tier | Catalog | In the catalog since | Why no weekly reader covers it (`SOURCE_MAP`) |
|---|---|---|---|---|
| `t4-16` | spec | $0.35 | 2026-05-28, `597df81` | not on Lambda's or CoreWeave's current lineup (checked 2026-09-16) |
| `t4-16` | spot | $0.15 | 2026-05-28, `597df81` | Vast gpu_name for T4 not verified against a live query |
| `l4-24` | spec | $0.43 | 2026-07-27, `de26df2` | not on Lambda's or CoreWeave's current lineup (checked 2026-09-16) |
| `l4-24` | spot | $0.22 | 2026-07-27, `de26df2` | Vast gpu_name for L4 not verified against a live query |
| `rtx4090-24` | hyper | $0.75 | 2026-05-28, `597df81` | "no hyperscaler rents this card", re-checked below |
| `rtx4090-24` | spec | $0.45 | 2026-05-28, `597df81` | not on Lambda's or CoreWeave's current lineup (checked 2026-09-16) |
| `rtx5090-32` | hyper | $0.89 | 2026-07-27, `de26df2` | "no hyperscaler rents this card", re-checked below |
| `rtx5090-32` | spec | $0.65 | 2026-07-27, `de26df2` | not on Lambda's or CoreWeave's current lineup (checked 2026-09-16) |
| `rtx5090-32` | spot | $0.45 | 2026-07-27, `de26df2` | Vast gpu_name for RTX 5090 not verified against a live query |
| `a100-40` | spot | $0.63 | 2026-05-28, `597df81` | Vast gpu_name for A100 40GB not verified against a live query |
| `rtx6000ada-48` | hyper | $1.50 | 2026-07-27, `de26df2` | "no hyperscaler rents this card", re-checked below |
| `rtx6000ada-48` | spec | $0.90 | 2026-07-27, `de26df2` | Lambda's and CoreWeave's "A6000" is the older Ampere RTX A6000, not this card |
| `rtx6000ada-48` | spot | $0.60 | 2026-07-27, `de26df2` | Vast gpu_name for RTX 6000 Ada not verified against a live query |
| `l40s-48` | spot | $0.90 | 2026-05-28, `597df81` | Vast gpu_name for L40S not verified against a live query |
| `a100-80` | spot | $0.99 | 2026-05-28, `597df81` | Vast gpu_name for A100 80GB not verified against a live query |
| `h100-80` | spot | $2.25 | 2026-05-28, `597df81` | automated (Vast), but no read has been applied, see below |
| `rtxpro-96` | hyper | $5.00 | 2026-05-28, `597df81` | "no hyperscaler rents this card". **False: all four hyperscalers rent it**, see below |
| `rtxpro-96` | spot | $2.00 | 2026-05-28, `597df81` | Vast gpu_name for RTX PRO 6000 not verified against a live query |
| `h200-141` | spot | $2.50 | 2026-07-27, `de26df2` | Vast gpu_name for H200 not verified against a live query |
| `b200-192` | spot | $2.12 | 2026-05-28, `597df81` | Vast gpu_name for B200 not verified against a live query |

## h100-80's spot tier: read, and held for a second read

The weekly check reads this tier from Vast, and the reads so far disagree with the catalog:
- 2026-09-16: aborted, because Vast returned 4 qualifying offers against a minimum of 5 (PR #10).
- 2026-09-17: +34.2%. The report is in closed PR #12's body; the owner set it aside in favour of the 2026-09-22 run.
- 2026-09-22: +42.5% on the median of 5 verified offers (PR #24). That is past the 40% move a person has to approve. `80f5dba` held it: "Needs a second read before it should move."

The note says that, and gives the percentage rather than a second price. A figure beside the catalog's figure would read as
a price.

## The four hyperscaler tiers, re-checked

The price map said "no hyperscaler rents this card" for all four, with no date. Re-checked on 2026-09-23 by a
Fable research agent. Items marked "re-checked by hand" I re-read at their source; the rest is the agent's reading, with its sources named.

**Not offered: RTX 4090, RTX 5090, RTX 6000 Ada.** None of AWS, Azure, Google Cloud or Oracle rents any of the three.
The sweep:
- **AWS:**
  - the EC2 accelerated-instance spec page, whose GPUs are L4, L40S, A10G, T4, "RTX PRO Server 6000", RTX PRO 4500, H200, H100, B200, B300 and A100;
  - the us-east-1 Linux on-demand price feed, where no `4090`, `5090` or `A6000` appears anywhere, re-checked by hand.
- **Azure:** the Retail Prices API, with `contains()` on SKU and product names for `4090`, `5090` and `A6000`: 0 items each, re-checked by hand. Its GPU size families list none of them either.
- **Google Cloud:** the GPU catalog page lists GB300, GB200, B200, H200, H100, A100, RTX PRO 6000, L4, T4, P4 and V100.
- **Oracle:** the compute-shapes page and the price-list API name no GeForce or Ada card. "Ada Lovelace" appears only as the L40S's architecture.

Not evidence, but worth knowing: the L40S is the datacenter card on the RTX 6000 Ada's die, and AWS and Oracle do rent it.

**Offered by all four: RTX PRO 6000 Blackwell Server Edition (96 GB).** Linux, on demand, no commitment, the smallest
whole-GPU size:

| Provider | Instance | GPUs | Region | Per GPU, $/hr | Read from |
|---|---|---|---|---|---|
| AWS | `g7e.2xlarge` | 1 | us-east-1 | 3.36312 | the EC2 on-demand price feed the pricing page loads: `"Instance Type": "g7e.2xlarge", "price": "3.3631200000"` (re-checked by hand) |
| Google Cloud | `g4-standard-48` | 1 | us-central1 (the page's default region) | 4.49993 | [accelerator-optimized pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized): "g4-standard-48 … $4.49993 / 1 hour", GPU included (re-checked by hand) |
| Oracle | `BM.GPU.RTXPRO.8` | 8 | none: Oracle's price list has no region | 4.50 | price-list API, part B112613 "OCI - Compute - GPU - RTX PRO 6000", GPU Per Hour, pay as you go 4.5 (re-checked by hand) |
| Azure | `Standard_NC144lds_xl_RTXPRO6000BSE_v6` | 1 | westus2 | 5.50 | Retail Prices API, Consumption meter `NC144ldsxlRTX6kv6`, 5.5 per hour (re-checked by hand) |

AWS's larger G7e sizes cost more per GPU: $3.99816 for the one-GPU `g7e.4xlarge`, and $4.14304 per GPU for the 2-, 4- and 8-GPU sizes.

**What changes.**
- rtx4090-24, rtx5090-32 and rtx6000ada-48 lose their hyperscaler figures. Those tiers become "no confirmed hourly price", with the sweep above as the reason.
- rtxpro-96's hyperscaler tier is read automatically from AWS `g7e.2xlarge`, one GPU with divisor 1, the same shape as l40s-48's `g6e.xlarge`. The weekly check already reads that AWS feed.
  - Its unsourced $5.00 becomes the read price, about $3.36.
  - The other three providers stay as the cross-checks above. The weekly check has no Google Cloud or Oracle reader. Azure could be wired in as an automated cross-check, but at 64% above AWS it is past the check's 20% cross-check floor and would be flagged every week.

## What the planner shows

- **A tier with a figure** reads "not recorded" and then its note: where the figure came from, and why no reader covers the
  tier.
- **A tier found to have no hourly offer** becomes null, reads "no confirmed hourly price", and gives the reason.
- **A tier a hyperscaler does rent** gets that provider's price, recorded by hand with its page and date.
