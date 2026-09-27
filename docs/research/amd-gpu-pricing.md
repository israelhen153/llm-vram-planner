# AMD GPU Cloud Pricing

**Read:** 2026-09-23. Supersedes the 2026-07-31 snapshot this file used to hold (see "Drift"
below).

**Purpose:** the prices `data/gpus.json` carries for its five AMD rows, why each tier holds
what it holds, and every lead that was considered and not used.

## What counts as a price

- **One named SKU's hourly price, read on the provider's own pricing page or official pricing
  API.** Comparison sites (getdeploying, thundercompute, spheron, computeprices, gpus.io …)
  are leads for finding pages, never sources.
- **Per GPU.** A per-instance price is divided by the GPUs in the instance.
- **The tier's own price.** Hyperscaler and specialized tiers take the on-demand price; the
  spot tier takes a spot or preemptible price.
- **Excluded:** "starting at" figures, reserved, committed or contract prices, and plans that no
  region currently offers.
- **Where more than one provider qualifies,** the tier takes the lowest confirmed price and names
  that one provider. A tier never carries one number for several providers.
- **Where none qualifies,** the tier is `null`, which every surface shows as "no confirmed hourly
  price". Its `SOURCE_MAP` entry in `tools/price_check.py` records why.

## The catalog's prices

| Row | Tier | $/GPU/hr | Provider | SKU as the provider names it | Region | Recorded as | Page |
|---|---|---|---|---|---|---|---|
| `mi300x-192` | hyper | 6.00 | Azure | `Standard_ND96isr_MI300X_v5`, meter `ND96isrMI300Xv5`, $48.00/hr for 8 GPUs | eastus2 | `priceSource`, re-read weekly by `tools/price_check.py` | Azure Retail Prices API |
| `mi300x-192` | spec | 2.39 | RunPod | MI300X (Secure Cloud), on demand | global | `priceRecord`, recorded by hand | https://www.runpod.io/gpu-models/mi300x |
| `mi300x-192` | spot | 1.11 | Azure | `Standard_ND96isr_MI300X_v5` spot, meter `ND96isrMI300Xv5 Spot`, $8.8704/hr for 8 GPUs | eastus2 | `priceSource`, re-read weekly | Azure Retail Prices API |
| `mi325x-256` | spec | 3.80 | DigitalOcean | "AMD Instinct™ MI325X" GPU Droplet, on demand, 1 or 8 GPUs | global | `priceRecord`, recorded by hand | https://www.digitalocean.com/pricing/gpu-droplets |

Every other AMD tier is null.

One qualification on the MI325X figure. DigitalOcean's pricing page lists $3.80/GPU/hour as the
"On-Demand Price", but its Droplet documentation says the MI325X sizes (`gpu-mi325x1-256gb-contracted`,
`gpu-mi325x8-2048gb-contracted`) are not in the self-service list and are provisioned through a
sales agreement. It is a published hourly rate, not a quote, so it qualifies. The hand record
names the page it was read from.

The Azure query is
`https://prices.azure.com/api/retail/prices?api-version=2023-01-01-preview&$filter=armSkuName eq 'Standard_ND96isr_MI300X_v5' and armRegionName eq 'eastus2'`.
It returns one Linux Consumption row per meter: $48.00 for `ND96isrMI300Xv5` and $8.8704 for
`ND96isrMI300Xv5 Spot`. The Windows, DevTest, Low Priority and Reservation rows sit beside them.
The repo's fetcher ignores them because it selects by exact meter name, the `Consumption` type
and a non-Windows product name. The same SKU is listed in 17 Azure
regions. eastus2 and westus3 are the cheapest, at $48.00.

## Tiers with no confirmed price

| Row | Tier | Why it is null |
|---|---|---|
| `mi325x-256` | hyper | No hyperscaler offers it. The Azure Retail Prices API returns 0 items for any SKU containing MI325, MI355, MI250 or MI210. Oracle's price list has MI300X and MI355X only, and neither AWS nor GCP lists an Instinct instance. |
| `mi325x-256` | spot | Vultr lists a preemptible price for `vbm-256c-3072gb-8-mi325x-gpu`: $16.00/hr for 8 GPUs, $2.00 per GPU. The plan's `locations` is empty, so no region offers it. |
| `mi250x-128` | all | Cirrascale lists "4X AMD Instinct MI250", not the MI250X, and only monthly. Runcrate's MI250X page (https://www.runcrate.ai/pricing/gpu/mi250x) advertises "$1.35 /hr" but see the MI210 row. |
| `mi210-64` | all | Runcrate's MI210 page (https://www.runcrate.ai/pricing/gpu/mi210) advertises "$0.82/hr On-demand · per-second billing", "$0.70–$0.95/hr range" and "Available now · 4 regions". Runcrate's own pricing page (https://www.runcrate.ai/pricing) lists 20 GPU SKUs and no AMD GPU. The per-card pages misstate AMD's own figures: "47.9 TFLOPS FP16" for the MI210, where AMD says 181.0, and 95.7 for the MI250X, where AMD says 383.0. The offer cannot be confirmed without an account, so the tier stays null, and `SOURCE_MAP` records why. The cold check read the same pages and put the question to the owner; see "Owner decision". |
| `rx7900xtx-24` | all | Vast.ai: "No current offers". HOSTKEY rents a 4× RX 7900 XTX server monthly, as a pre-order, with no hourly price. |

The RunPod figure is also confirmed by RunPod's own GraphQL API (`api.runpod.io/graphql`, `gpuTypes`,
`securePrice: 2.39` for "AMD Instinct MI300X OAM"). RunPod's general pricing page, updated September 13,
2026, does not list the MI300X, so a reader re-checking by hand should use the card's page or the API.

## Owner decision, 2026-09-23: show them as leads, never as prices

The question was whether a provider's per-card marketing page counts as "a provider's own page"
when the provider's own pricing page does not list that card. The owner's answer is no, for
pricing, but the reader should see what was found. So the MI210 and MI250X specialized tiers
stay `null` and out of every cost figure. The planner will show Runcrate's advertised figures
beside those tiers as unconfirmed leads, with why they are unconfirmed, the page they came from,
and a link to [due diligence on the company](runcrate-due-diligence.md). That page shows the
figures are fixed numbers in the page source, and that nothing else of Runcrate's, nor the
independent tracker checked, backs an AMD offer.

Earlier notes called Runcrate a reseller. That was an inference. By its own account it is an
aggregator that does not host GPUs itself; see the due diligence.

## Other confirmed prices, not the catalog's choice

All read 2026-09-23 on the provider's own page or API. MI300X unless noted.

- **DigitalOcean**, `gpu-mi300x1-192gb`: $2.59/GPU/hr on demand (ATL1). The 12-month reserved
  price is $1.91; for MI325X it is $2.88.
- **Hot Aisle**, 1×/2×/4× MI300X VMs: $2.99/GPU/hr, billed per minute (Michigan).
- **Crusoe**, `mi300x-192gb-ib.8x`: $3.45/GPU/hr (us-east1-a).
- **Oracle**, `BM.GPU.MI300X.8`: $6.00/GPU/hr, pay as you go. This matches Azure's $6.00, a
  cross-check the weekly job cannot make, because it has no Oracle reader.
- **Vultr**, `vbm-256c-2048gb-8-mi300x-gpu`: $31.92/hr standard, $14.80/hr preemptible, for 8 GPUs
  ($3.99 and $1.85 per GPU). The plan is marked not deployable on demand and lists no locations.
- **Azure Low Priority**, `ND96isrMI300Xv5 Low Priority`: $9.60/hr for 8 GPUs in eastus2.

## Leads, unconfirmed

- **TensorWave:** "Starting at $1.71/GPU HR" (MI300X) and "$2.25" (MI325X). Terms not stated;
  `tensorwave.com/pricing` returns 404.
- **Runcrate:** MI210 and MI250X. See the table above. Its MI300X page
  (https://www.runcrate.ai/pricing/gpu/mi300x) quotes other providers' prices, so it can be checked.
  On 2026-09-23 it listed Runcrate at $2.50/hr as "Cheapest", against RunPod
  at $3.49, Oracle at $4.30, AWS at $4.95 and Azure at $5.20. The same day, RunPod's own API gave
  $2.39, Oracle's and Azure's price lists gave $6.00 per GPU, and AWS listed no Instinct instance.
  Its RX 7900 XTX page returned a server error each time it was read. What could be verified about the company is in
  [runcrate-due-diligence.md](runcrate-due-diligence.md).
- **Bentaus:** MI325X at $2.25. Found on a comparison site only; no provider page.
- **Cirrascale:** MI300X and MI325X by monthly commitment or on request only.

## Drift from the 2026-07-31 snapshot

- **DigitalOcean MI300X:** $1.99 then, $2.59 now.
- **Vultr's "$1.85" for MI300X** was the preemptible price of a plan no region offers.
- **Vultr's "$2.00 marketplace" for MI325X:** still a preemptible price, and still no region
  offers the plan.
- **Runcrate's MI210 and MI250X figures** cannot be confirmed as list prices.
- **Azure and Oracle MI300X, $6.00/GPU:** unchanged.
- **Sources:** the July table's figures came mostly from comparison-site blogs. None is used as
  a source now.

## Newer cards, out of scope

- **MI355X:** Oracle `BM.GPU.MI355X.8` at $8.60/GPU/hr (Oracle price list), and Vultr.
- **MI350X:** DigitalOcean spot at $3.00/GPU/hr, RunPod at $5.49, CloudRift at $3.65.

Both are candidates for a later catalog release.

## Who read what

A research pass read every price above on 2026-09-23. A second reader re-read these on the
same day:
- the Azure API rows for both MI300X tiers;
- DigitalOcean's MI300X and MI325X on-demand prices;
- RunPod's MI300X price;
- Vultr's plan list, including the empty `locations`;
- the Azure API's empty result for MI325, MI355, MI250 and MI210;
- Vast's "No current offers" for the RX 7900 XTX.

The release's cold check re-reads at least one automated price and one hand-recorded price
independently.
