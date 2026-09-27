# Runcrate: due diligence

**Read:** 2026-09-23.

**Why this page exists.** Runcrate's card pages advertise the only hourly prices found for two
cards in the catalog: the MI210 at $0.82/hr and the MI250X at $1.35/hr. The planner does not use them
as prices. The owner decided on 2026-09-23 that it will show them as unconfirmed leads, beside tiers
that read "no confirmed hourly price", with a link to this page. Few readers will have heard of the
company, so this page says what could be verified about it and what could not.

**How it was read.** A research pass read the sources, then every claim below was re-checked by
hand the same day. Claims that could not be re-checked are left out: LinkedIn, Trustpilot,
Crunchbase and the business registries were blocked or gated.

## Summary

- **Runcrate describes itself as a GPU aggregator.** It is run by Aeonmind, LLC, which says it is a Delaware LLC
  formed in July 2025 and self-funded. The LLC could not be confirmed in a public registry.
- **By its own account, it does not run its own hardware.** The account that posted Runcrate's launch on Hacker News:
  "We are an aggregator on top of a lot of providers, we do not host gpus ourselves."
- **Nothing backs its AMD offers.** The independent tracker checked (GetDeploying) lists no Runcrate
  AMD offer. Runcrate's own pricing page, docs and partner programme name only NVIDIA hardware, apart
  from one Intel Gaudi2 listing on the pricing page.
- **The AMD prices are fixed numbers in the page source**, not a live feed. They sit next to
  competitor prices that don't match what those competitors publish.
- **Its own pages disagree** on its prices, founding year and number of regions.

## Who runs it

- The terms of service name **"Aeonmind, LLC"** (https://www.runcrate.ai/terms). They:
  - say "Last updated: January 2025", but `runcrate.ai` was registered on 2025-05-30;
  - name no governing law, jurisdiction or address;
  - say "Billing is calculated hourly", while the card pages advertise "per-second billing".
- Aeonmind's own site says: "JUL · 2025 Aeonmind LLC incorporated — Delaware filing.
  Self-funded." (https://www.aeonmind.ai/)
- Runcrate's company page says "2024 Founded" (https://www.runcrate.ai/company).
- **Registries:** not confirmed. Delaware's entity search is behind a CAPTCHA, and OpenCorporates
  needs an API key.

## Where its GPUs come from

- **Hacker News, 2025-10-30**, a reply from the author of Runcrate's Show HN: "We are an
  aggregator on top of a lot of providers, we do not host gpus ourselves. More like Skyscanner or
  Airbnb for GPUs" (https://news.ycombinator.com/item?id=45754893).
- **The company page:** "aggregating GPU capacity across the world", "10K+ GPUs in network",
  "8 Global regions". The MI210 page says "4 regions". None of the pricing pages read here names a
  region, or the provider behind a price.
- **The partner programme:** "If you have NVIDIA GPUs, we want to talk"
  (https://www.runcrate.ai/partners). AMD appears on that page only in its metadata keywords.
- **The docs:** dedicated clusters run "NVIDIA H100, H200, B200, and B300 GPUs"
  (https://www.runcrate.ai/docs/dedicated/available-gpus).

## The AMD prices

- **The prices are fixed values in the page source:**
  - the MI210 page's source carries `"pricePerHour":{"min":0.7,"max":0.95,"avg":0.82}`
    (https://www.runcrate.ai/pricing/gpu/mi210);
  - the MI250X page's carries `{"min":1.2,"max":1.5,"avg":1.35}`
    (https://www.runcrate.ai/pricing/gpu/mi250x).

  These are the ranges and averages each page prints.
- **The competitor figures don't match what those competitors publish.** The same source carries
  `"competitorPricing":{"aws":4.95,"azure":5.2,"oracle":4.3,"runpod":3.49}`, the comparison table
  its MI300X page prints. The same day, the competitors' own sources said:
  - RunPod's API: $2.39;
  - Oracle's and Azure's price lists: $6.00 per GPU;
  - AWS: no MI300X instance listed.

  See [amd-gpu-pricing.md](amd-gpu-pricing.md).
- **The card pages give about a quarter of AMD's published speeds.** In TFLOPS:
  - MI210: 47.9, against AMD's 181.0;
  - MI250X: 95.7, against 383.0;
  - MI300X: 326, against 1307.4.
- **Its own pricing page lists no AMD GPU** (https://www.runcrate.ai/pricing).
- **The MI210 page first appears in the Wayback Machine on 2026-01-25.**
- **The independent tracker checked lists no Runcrate AMD offer.** GetDeploying follows Runcrate's prices
  ("last updated on Sept. 23, 2026") and lists no AMD model for it
  (https://getdeploying.com/runcrate). Its MI300X page doesn't list Runcrate
  (https://getdeploying.com/gpus/amd-mi300x).

## Its own pages disagree

- **H100 price:** $1.54/hr on the home and company pages; $3.00/hr on the pricing page.
- **Founding:** "2024 Founded" on the company page. The operating company says it was incorporated
  in July 2025, and the domain was registered in May 2025.
- **Regions:** "8 Global regions" on the company page; "4 regions" on the MI210 page.

## Public footprint

- **Domain:** `runcrate.ai` was registered 2025-05-30 through Cloudflare (RDAP).
- **GitHub:** the `aeonmindai` organisation was created 2025-06-02. It has 9 public repositories
  and 2 followers (https://api.github.com/orgs/aeonmindai).
- **SDKs:**
  - PyPI `runcrate-sdk`: first upload 2026-04-14, 5 releases, author "Runcrate
    <support@runcrate.ai>".
  - npm `@runcrate/sdk`: created 2026-04-14, 4 versions.
- **Community:** a Discord server named "Runcrate" with 229 members.
- **Hacker News:** "Show HN: Frustrated by GPU costs, so I built my own cloud", 2025-10-27, 3 points
  and 3 comments (https://news.ycombinator.com/item?id=45720013).
- **Funding:** "Self-funded", per Aeonmind's site.

## What this means for the planner (interpretation)

- **Runcrate looks like a real, small, young business.** It has working docs, SDKs, a community,
  and an independent tracker that follows its NVIDIA prices.
- **Its AMD prices are not evidence of a bookable AMD offer.** They are fixed numbers on marketing
  pages. Nothing else of Runcrate's backs them: not its pricing page, docs or partner programme.
  The tracker checked doesn't list them. And they sit beside competitor prices that don't match
  those competitors' own.
- **So the planner will not count them as a price.** It will show them only as leads, linked here. If
  Runcrate's pricing page or an independent tracker ever lists an AMD offer, that would be a
  source, and this page would change.
