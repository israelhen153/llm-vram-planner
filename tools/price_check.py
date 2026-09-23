#!/usr/bin/env python3
"""Re-read published GPU prices and report what moved against data/gpus.json.

This is repo-side automation: it runs on a schedule (see
.github/workflows/price-refresh.yml) or by hand, reads a handful of public
pricing sources over the network, and either leaves data/gpus.json alone or
(with --apply) writes back the prices and provenance that passed every
sanity check. It is never imported by index.html or generate_report.py, and
it has no bearing on the tool's offline guarantee: the *planner* still makes
zero network calls from file://. Only this maintenance job does, the same
way tools/make_assets.py needs a browser that the planner itself never does.

The design constraint this file exists to satisfy: a provider page that
changes shape must produce NO change rather than a WRONG one. Every source
below is fetch -> parse -> validate, and a validation failure aborts only
that source's slug/tier — it is logged, it is not fatal to the run, and it
never contributes a number. See SOURCE_MAP's docstring for the coverage
this build actually verified live vs. what is deliberately left manual.

Usage:
    python3 tools/price_check.py                        # fetch, print a report, change nothing
    python3 tools/price_check.py --report-out FILE       # also write the report to FILE
    python3 tools/price_check.py --apply                 # write confirmed/moved rows back to data/gpus.json
    python3 tools/price_check.py --slug h100-80 --slug b200-192   # restrict to given slugs
    python3 tools/price_check.py --only azure,aws         # restrict to given source kinds

Needs only the Python standard library — same constraint tests/run.sh holds
every other test to ("needs only node and python3"), so tests/price_check.test.py
can run in CI with no extra install step, using canned fixtures instead of the
network for anything that parses a live shape.
"""
import argparse
import gzip
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

USER_AGENT = ("llm-vram-planner-price-check/1.0 "
              "(+https://github.com/israelhen153/llm-vram-planner; repo-side price sanity job)")

# Absolute sanity band for a per-GPU $/hr figure, after any divisor is applied.
# Catches a unit change (per-minute, per-month), a currency change, or a wrong
# divisor — all of which land wildly outside this range rather than merely
# looking a bit off.
ABS_MIN, ABS_MAX = 0.05, 100.0

# A move at or below this fraction of the current catalog value is treated as
# noise: catalog prices are hand-rounded to 2-3 significant figures, so
# sub-1% differences are rounding, not a market move. Above FLAG_THRESHOLD,
# the move is real but too large to auto-apply blind — a 40%+ swing is
# exactly as likely to be a genuine price cut (verified today: AWS's H100
# instance really has moved this much since the catalog's number was set) as
# it is a parsing bug, so it is surfaced for a human instead of either
# extreme (silently apply / silently refuse forever).
CONFIRM_THRESHOLD = 0.01
FLAG_THRESHOLD = 0.40

# A secondary/cross-check reading is fetched from a different vendor than the
# primary and never drives a proposed value on its own — see SOURCE_MAP's
# docstring: two vendors legitimately charge different amounts for
# comparable-but-not-identical hardware, so a strict floor here would flag
# routine vendor variation. But primary and secondary are read in the same
# run, the same hour, for the same board: unlike a hand-set catalog value
# against a fresh fetch, there is no "the market moved since we looked"
# excuse for the two to disagree by a lot. Found live 2026-09-22:
# b200-192/spec applied Lambda's $6.69 while CoreWeave, fetched the same run,
# read $8.60 for what both call an 8x HGX B200 node — 28% apart — and the
# run applied the lower number without saying so anywhere. Set below
# FLAG_THRESHOLD on purpose: two live reads disagreeing by half of what it
# takes to flag a stale-catalog-vs-live move is more suspicious, not less,
# because staleness cannot explain it.
CROSS_CHECK_FLAG_THRESHOLD = 0.20


class SourceError(Exception):
    """A source failed validation. Caught per (slug, tier); never crashes the run."""


@dataclass(frozen=True)
class Reading:
    provider: str
    sku: str
    region: str
    price_per_gpu: float
    date: str
    evidence: str


@dataclass
class Outcome:
    slug: str
    tier: str
    status: str  # CONFIRMED | MOVED | FLAGGED | ABORTED | MANUAL
    current: object = None
    proposed: object = None
    reading: object = None            # Reading, when status came from a fetch
    secondary: list = field(default_factory=list)   # list[Reading | SourceError]
    note: str = ""


def today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def http_get(url, timeout=20, extra_headers=None):
    """GET a URL as this job, decompressing gzip by hand (urllib does not do
    it automatically just because Accept-Encoding was sent), and turning any
    network failure into a SourceError so callers never need their own
    try/except around a plain socket error."""
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            resp_headers = dict(resp.headers)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
        raise SourceError(f"HTTP {e.code} fetching {url}: {body or e.reason}") from e
    except urllib.error.URLError as e:
        raise SourceError(f"network error fetching {url}: {e.reason}") from e
    except TimeoutError as e:
        raise SourceError(f"timed out fetching {url}") from e
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace"), resp_headers


def _check_band(per_gpu, label):
    """Absolute band + zero-is-not-a-price, shared by every source."""
    if not isinstance(per_gpu, (int, float)) or isinstance(per_gpu, bool):
        raise SourceError(f"{label}: computed price is not numeric: {per_gpu!r}")
    if per_gpu <= 0:
        raise SourceError(f"{label}: computed price is {per_gpu} — zero or negative is never a price")
    if not (ABS_MIN <= per_gpu <= ABS_MAX):
        raise SourceError(
            f"{label}: ${per_gpu:.4f}/GPU/hr is outside the sane band "
            f"[${ABS_MIN}, ${ABS_MAX}] — likely a unit or divisor error")


# ---------------------------------------------------------------------------
# Azure Retail Prices API — hyper, primary.
#
# Documented, unauthenticated, versioned via api-version. Verified live
# 2026-09-16: Standard_ND96isr_H100_v5 in eastus is $98.32/hr for 8 GPUs =
# $12.29/GPU, which is where the catalog's h100-80 hyper=12.3 almost
# certainly came from.
#
# Two gotchas, both confirmed against the live response that day, and the
# second found independently while building this tool (not in the prior
# research):
#   1. One armSkuName returns rows spanning OTHER regions and Spot/Low
#      Priority meters even inside priceType eq 'Consumption'. armRegionName
#      and meterName must both be matched exactly (never substring).
#   2. The Windows-priced row is NOT distinguished by meterName at all in
#      every case seen — for ND96isr_H100_v5 the Linux and Windows rows
#      share the identical meterName ("ND96isrH100v5"); only productName
#      gains a " Windows" suffix. (For ND96amsr_A100_v4 productName instead
#      carries an explicit " Linux" / " Windows" suffix on both rows — the
#      two SKU families are not even consistent with each other.) Filtering
#      on meterName alone is not enough; productName must be checked too.
# ---------------------------------------------------------------------------
AZURE_API = "https://prices.azure.com/api/retail/prices"
AZURE_API_VERSION = "2023-01-01-preview"


def fetch_azure(sku, region, meter_name, divisor):
    filt = (f"armSkuName eq '{sku}' and armRegionName eq '{region}' "
            f"and priceType eq 'Consumption'")
    url = f"{AZURE_API}?api-version={AZURE_API_VERSION}&$filter={urllib.parse.quote(filt)}"
    body, _ = http_get(url)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        raise SourceError(f"azure {sku}: response was not JSON ({e})")
    for key in ("Items", "Count"):
        if key not in payload:
            raise SourceError(f"azure {sku}: envelope missing {key!r} — API shape changed")
    candidates = [
        it for it in payload["Items"]
        if it.get("armRegionName") == region
        and it.get("armSkuName") == sku
        and it.get("meterName") == meter_name
        and it.get("type") == "Consumption"
        and "windows" not in str(it.get("productName", "")).lower()
    ]
    if len(candidates) != 1:
        raise SourceError(
            f"azure {sku}/{region}/{meter_name!r}: resolved to {len(candidates)} rows, "
            f"expected exactly 1 (of {len(payload['Items'])} returned by the API)")
    row = candidates[0]
    if row.get("unitOfMeasure") != "1 Hour":
        raise SourceError(f"azure {sku}: unitOfMeasure is {row.get('unitOfMeasure')!r}, expected '1 Hour'")
    if row.get("currencyCode") != "USD":
        raise SourceError(f"azure {sku}: currencyCode is {row.get('currencyCode')!r}, expected 'USD'")
    price = row.get("retailPrice")
    if not isinstance(price, (int, float)) or isinstance(price, bool):
        raise SourceError(f"azure {sku}: retailPrice is not numeric: {price!r}")
    per_gpu = price / divisor
    _check_band(per_gpu, f"azure:{sku}")
    return Reading(provider="azure", sku=sku, region=region, price_per_gpu=per_gpu, date=today(),
                    evidence=f"${price:.4f}/hr / {divisor} GPUs, meterName={meter_name!r}")


# ---------------------------------------------------------------------------
# AWS — hyper, secondary/cross-check for h100-80/a100-40/a100-80, primary for
# everything else this build covers (h200-141, b200-192, t4-16, l4-24, l40s-48).
#
# UNDOCUMENTED SOURCE — read this before touching fetch_aws_feed/select_aws.
# This is not AWS's documented Price List API (the 304 MB bulk CSV/JSON, or
# the SigV4-signed GetProducts query). It is the small JSON feed the AWS
# pricing *calculator page itself* calls: 58 KB gzipped, AWS-owned, verified
# unauthenticated live 2026-09-16. It carries no api-version, no
# publication-date contract a caller can rely on structurally, and no
# deprecation policy — AWS could reshape or retire it without notice. That is
# exactly why this source fails closed on any surprise, more aggressively
# than a documented one would need to: every key this code reads is asserted
# present before use, and a shape it doesn't recognise aborts every AWS slug
# for this run rather than guessing.
#
# One correction found independently while building this tool, beyond what
# the prior research or the handoff described: the feed's shape is
#     regions -> "<region label>" -> "<compound key>" -> {price, "Instance Type", ...}
# and the inner dict's own KEYS are NOT the instance type — they are compound
# strings like "p5.48xlarge US East N. Virginia Linux". Looking a bare
# instance type up as region_map[instance_type] silently returns nothing
# (verified: it returns None, not a KeyError). The correct approach, used
# below, is to search the region's values for a row whose "Instance Type"
# field matches — verified live that exactly one such row exists per type in
# this feed (it is pre-filtered to Linux, on-demand, no secondary license
# dimension by the URL path itself, unlike the bulk CSV which needs a 6-way
# filter to avoid the same trap).
# ---------------------------------------------------------------------------
AWS_FEED = ("https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/"
            "ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json")
AWS_REGION_LABEL = "US East (N. Virginia)"


def fetch_aws_feed():
    """Fetch and shape-check the feed once; callers pull individual instance
    types out of the returned region map with select_aws(). One HTTP call
    serves every AWS-sourced slug in a run instead of one call each."""
    body, _ = http_get(AWS_FEED)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        raise SourceError(f"aws: response was not JSON ({e})")
    if "manifest" not in payload or "regions" not in payload:
        raise SourceError("aws: top-level 'manifest'/'regions' key missing — feed shape changed")
    region_map = payload["regions"].get(AWS_REGION_LABEL)
    if not isinstance(region_map, dict) or not region_map:
        raise SourceError(f"aws: region {AWS_REGION_LABEL!r} missing or empty in feed")
    return region_map, payload["manifest"]


def select_aws(region_map, instance_type, divisor):
    matches = [v for v in region_map.values()
               if isinstance(v, dict) and v.get("Instance Type") == instance_type]
    if len(matches) != 1:
        raise SourceError(f"aws {instance_type}: {len(matches)} rows matched, expected exactly 1")
    row = matches[0]
    if row.get("Operating System") != "Linux":
        raise SourceError(
            f"aws {instance_type}: Operating System is {row.get('Operating System')!r}, expected 'Linux'")
    price_raw = row.get("price")
    try:
        price = float(price_raw)
    except (TypeError, ValueError):
        raise SourceError(f"aws {instance_type}: non-numeric price {price_raw!r}")
    per_gpu = price / divisor
    _check_band(per_gpu, f"aws:{instance_type}")
    return Reading(provider="aws", sku=instance_type, region=AWS_REGION_LABEL,
                    price_per_gpu=per_gpu, date=today(), evidence=f"${price:.4f}/hr / {divisor} GPUs")


# ---------------------------------------------------------------------------
# Lambda — spec, primary. Server-rendered HTML; verified live 2026-09-16.
#
# Each pricing row is `<tr class="..." data-plan="NVIDIA H100 SXM">...<td
# data-label="PRICE/GPU/HR*">$3.99</td>`. The page repeats every card across
# four node scales (1x/2x/4x/8x) with no explicit scale label on the row —
# the catalog wants the 8x figure (that is what today's spec values track) —
# so the anchor pins the row's own vCPUs count as a schema fingerprint
# (verified live: 8x H100 SXM = 208 vCPUs) rather than trying to infer scale
# from row order, which is exactly the kind of positional assumption a page
# edit breaks silently.
#
# The page ALSO embeds an escaped JSON duplicate of the same table
# (`data-label=\"PRICE/GPU/HR...` inside a hydration payload) elsewhere on
# the page. The row regex below requires a literal `"` right after `=`, which
# a backslash-escaped duplicate never has, so the duplicate is excluded by
# construction rather than by extra filtering.
# ---------------------------------------------------------------------------
LAMBDA_URL = "https://lambda.ai/pricing"
LAMBDA_ANCHOR = 'data-label="PRICE/GPU/HR'
_LAMBDA_ROW_RE = re.compile(r'<tr class="[^"]*" data-plan="([^"]+)">(.*?)</tr>', re.DOTALL)
_LAMBDA_PRICE_RE = re.compile(r'data-label="PRICE/GPU/HR\*?">([^<]*)</td>')
_LAMBDA_VRAM_RE = re.compile(r'data-label="VRAM/GPU">([^<]*)</td>')
_LAMBDA_VCPU_RE = re.compile(r'data-label="vCPUs">([^<]*)</td>')


def fetch_lambda_page():
    body, _ = http_get(LAMBDA_URL)
    if LAMBDA_ANCHOR not in body:
        raise SourceError(f"lambda: anchor {LAMBDA_ANCHOR!r} not found — page shape changed")
    return body


def parse_lambda_rows(html):
    rows = []
    for m in _LAMBDA_ROW_RE.finditer(html):
        plan, block = m.group(1), m.group(2)
        price_m = _LAMBDA_PRICE_RE.search(block)
        if not price_m:
            continue  # not every <tr data-plan> is a pricing row (e.g. spec-only rows)
        vram_m = _LAMBDA_VRAM_RE.search(block)
        vcpu_m = _LAMBDA_VCPU_RE.search(block)
        rows.append({
            "plan": plan,
            "vram": vram_m.group(1).strip() if vram_m else None,
            "vcpus": vcpu_m.group(1).strip() if vcpu_m else None,
            "price_text": price_m.group(1).strip(),
        })
    if not rows:
        raise SourceError("lambda: zero pricing rows parsed — page structure changed")
    return rows


def select_lambda(rows, plan, vram, vcpus):
    matches = [r for r in rows if r["plan"] == plan and r["vram"] == vram and r["vcpus"] == str(vcpus)]
    if len(matches) != 1:
        raise SourceError(
            f"lambda {plan}/{vram}/{vcpus} vCPU: {len(matches)} rows matched, expected exactly 1")
    row = matches[0]
    price_text = row["price_text"]
    if not price_text or not price_text.startswith("$"):
        raise SourceError(f"lambda {plan}/{vram}: price cell is {price_text!r} — no price listed "
                           "(a reserved-only row, or the column changed)")
    try:
        price = float(price_text[1:].replace(",", ""))
    except ValueError:
        raise SourceError(f"lambda {plan}/{vram}: could not parse {price_text!r} as a number")
    # Lambda's column is already PRICE/GPU/HR — no divisor.
    _check_band(price, f"lambda:{plan}/{vram}")
    return Reading(provider="lambda", sku=f"{plan} ({vram}, {vcpus} vCPU tier)", region="global",
                    price_per_gpu=price, date=today(), evidence=f"{price_text}/GPU/hr, row vCPUs={vcpus}")


# ---------------------------------------------------------------------------
# CoreWeave — spec, secondary/cross-check (primary where Lambda does not
# carry the card: h200-141, l40s-48, rtxpro-96). Server-rendered Webflow
# HTML; verified live 2026-09-16.
#
# The page renders each GPU model's card MORE than once (a desktop grid copy
# and a labelled "meta" copy, at minimum). Verified live, independently of
# the prior research, on the exact same product: the On-Demand price agreed
# across every copy found for HGX H100 ($49.24), but the Spot price did NOT
# ($19.71 vs $19.51 for the same fetch) — confirming the prior finding fresh
# rather than assuming it still holds. This is why the code below (a) never
# uses the spot figure at all, and (b) treats disagreement among the
# On-Demand copies as an abort rather than picking one.
#
# GPU Count and VRAM are read from the explicitly labelled
# `<div class="table-meta-value">N</div><div>GPU Count</div>` blocks, not
# from the desktop grid's positional cells — the grid repeats the same
# `<div class="table-v2-cell"><div>N</div></div>` shape for GPU Count, VRAM,
# vCPUs and storage back to back with nothing to tell them apart by position
# alone, which is fragile in a way the labelled block is not.
# ---------------------------------------------------------------------------
COREWEAVE_URL = "https://www.coreweave.com/pricing"
COREWEAVE_ANCHOR_GPU_COUNT = "GPU Count"
COREWEAVE_ANCHOR_PRICE = "instance-price"


def fetch_coreweave_page():
    body, _ = http_get(COREWEAVE_URL)
    if COREWEAVE_ANCHOR_GPU_COUNT not in body or COREWEAVE_ANCHOR_PRICE not in body:
        raise SourceError("coreweave: expected anchors not found — page shape changed")
    return body


def select_coreweave(html, product_slug, expected_name, expected_vram, divisor):
    title_pat = re.compile(
        re.escape(f'data-product="{product_slug}"') + r'[^>]*class="table-model-name">'
        + re.escape(expected_name) + r'</h3>')
    title_positions = [m.end() for m in title_pat.finditer(html)]
    if not title_positions:
        raise SourceError(f"coreweave {product_slug}: no row titled {expected_name!r} found "
                           "(product slug or display name changed)")

    def collect(pattern):
        vals = set()
        for pos in title_positions:
            m = re.search(pattern, html[pos:pos + 1500])
            if m and m.group(1).strip():
                vals.add(m.group(1).strip())
        return vals

    vrams = collect(r'<div class="table-meta-value">([\d.]+)</div>\s*<div>VRAM</div>')
    gpu_counts = collect(r'<div class="table-meta-value">(\d+)</div>\s*<div>GPU Count</div>')
    on_demand = collect(r'instance-price">On-Demand Price: <span class="item-value">(\$[\d,.]+)</span>')
    per_gpu_listed = collect(
        r'inference-price">Inference Single CPU Price: <span class="item-value">(\$[\d,.]+)</span>')

    if vrams != {expected_vram}:
        raise SourceError(f"coreweave {product_slug}: VRAM read as {vrams or '(none)'}, "
                           f"expected {{{expected_vram!r}}}")
    if len(gpu_counts) != 1:
        raise SourceError(f"coreweave {product_slug}: GPU Count disagreed or was missing across "
                           f"the page's duplicated blocks: {gpu_counts!r}")
    gpu_count = int(next(iter(gpu_counts)))
    if gpu_count != divisor:
        raise SourceError(f"coreweave {product_slug}: page says GPU Count={gpu_count}, "
                           f"configured divisor is {divisor} — fix the divisor, not this assertion")
    if not on_demand:
        raise SourceError(f"coreweave {product_slug}: no On-Demand price found — "
                           "likely a 'Contact sales' row (missing price is absent, never 0)")
    if len(on_demand) > 1:
        raise SourceError(f"coreweave {product_slug}: On-Demand price disagreed across the page's "
                           f"duplicated blocks: {sorted(on_demand)} — never trust which copy is right")
    on_demand_price = float(next(iter(on_demand)).replace("$", "").replace(",", ""))
    per_gpu_from_division = on_demand_price / gpu_count

    if len(per_gpu_listed) > 1:
        raise SourceError(f"coreweave {product_slug}: per-GPU price disagreed across duplicated "
                           f"blocks: {sorted(per_gpu_listed)}")
    if per_gpu_listed:
        per_gpu_from_page = float(next(iter(per_gpu_listed)).replace("$", "").replace(",", ""))
        if abs(per_gpu_from_page - per_gpu_from_division) / per_gpu_from_division > 0.01:
            raise SourceError(
                f"coreweave {product_slug}: on-demand/GPU-count (${per_gpu_from_division:.2f}) "
                f"disagrees with the page's own per-GPU figure (${per_gpu_from_page:.2f}) by >1%")
        per_gpu = per_gpu_from_page
        evidence_extra = f", page's own per-GPU figure ${per_gpu_from_page:.2f} agrees within 1%"
    else:
        per_gpu = per_gpu_from_division
        evidence_extra = ""

    _check_band(per_gpu, f"coreweave:{product_slug}")
    return Reading(provider="coreweave", sku=expected_name, region="global", price_per_gpu=per_gpu,
                    date=today(),
                    evidence=f"On-Demand ${on_demand_price:.2f}/hr / {gpu_count} GPUs{evidence_extra} "
                              "(spot column never used — verified unreliable, see module docstring)")


# ---------------------------------------------------------------------------
# Vast.ai — spot, primary. Documented bundles search API; verified live
# 2026-09-16 against console.vast.ai (never the vast.ai website — its
# /pricing/tables/ is robots-disallowed for every agent; the API host is not).
#
# The response's own `verified` field on each offer came back null on every
# offer today even with verified:{eq:true} in the query, and the offer count
# changed (9 -> 5) when that filter was added or removed — so the *filter*
# is honoured server-side, it is just not echoed back in the field of the
# same name in the payload. This code trusts the query filter and never
# reads offer['verified'] back out of the response.
# ---------------------------------------------------------------------------
VAST_API = "https://console.vast.ai/api/v0/bundles/"


def fetch_vast(gpu_name, divisor=1, min_sample=5, num_gpus=1):
    query = {
        "gpu_name": {"eq": gpu_name}, "verified": {"eq": True}, "rentable": {"eq": True},
        "external": {"eq": False}, "num_gpus": {"eq": num_gpus}, "type": "ask",
        "limit": 64, "order": [["dph_total", "asc"]],
    }
    url = VAST_API + "?q=" + urllib.parse.quote(json.dumps(query))
    body, _ = http_get(url)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        raise SourceError(f"vast {gpu_name}: response was not JSON ({e})")
    if payload.get("success") is False:
        raise SourceError(f"vast {gpu_name}: API returned an error envelope: {payload.get('msg')}")
    offers = payload.get("offers")
    if not isinstance(offers, list):
        raise SourceError(f"vast {gpu_name}: 'offers' key missing or not a list — API shape changed")
    prices = sorted(o["dph_total"] / num_gpus for o in offers
                     if isinstance(o.get("dph_total"), (int, float)))
    if len(prices) < min_sample:
        raise SourceError(f"vast {gpu_name}: only {len(prices)} qualifying offers, "
                           f"need at least {min_sample} — thin sample, not trusted")
    mid = len(prices) // 2
    median = prices[mid] if len(prices) % 2 else (prices[mid - 1] + prices[mid]) / 2
    _check_band(median, f"vast:{gpu_name}")
    return Reading(provider="vast", sku=f"{gpu_name} (median of {len(prices)} verified offers)",
                    region="global", price_per_gpu=median, date=today(),
                    evidence=f"median ${median:.2f}/GPU/hr across {len(prices)} offers "
                              f"(range ${prices[0]:.2f}-${prices[-1]:.2f}, never the min — the cheapest "
                              "listing is routinely an unverified or broken host)")


# ---------------------------------------------------------------------------
# SOURCE_MAP — per slug, per tier: either {"manual": "<reason>"} or
# {"primary": <spec>, "secondary": [<spec>, ...]}. Secondary readings are
# fetched, validated and reported for context but NEVER drive a proposed
# value — Lambda and CoreWeave are different vendors with legitimately
# different prices for similar hardware (verified today: Lambda $3.99 vs
# CoreWeave $6.16 per GPU for what both call an 8x H100 SXM node), so a
# secondary silently standing in for a failed primary would let the
# catalog's number swing on which vendor happened to validate that run —
# exactly the un-documented-choice problem the owner already flagged for
# the hyper tier's provider picker. Only the designated primary ever writes.
#
# Absence must be explicit, never discovered by a parser: every slug/tier
# combination below is either configured or carries a manual reason, and the
# reasons distinguish two different kinds of "not automated" —
#   - no automatable source exists (a hyperscaler does not rent a consumer
#     card; a spec provider does not stock this model) — a structural fact,
#     unlikely to change;
#   - not verified against a live fetch in THIS build — a scope decision
#     under time, not a structural one; a future run can extend the map by
#     verifying one more query, following the pattern already proven below.
#
# Coverage verified live 2026-09-16 while building this tool (all providers
# in this map were queried for real; see the PR body for every number
# observed, including several that disagree with the catalog's current
# value — reported as findings, never applied here):
#   hyper : azure  h100-80, a100-40, a100-80 (SKU/GB mapping confirmed
#                  against Microsoft Learn, not just numeric coincidence)
#           aws    h100-80, a100-40, a100-80, h200-141, b200-192, t4-16,
#                  l4-24, l40s-48
#   spec  : lambda h100-80, a100-40, a100-80, b200-192
#           coreweave l40s-48, a100-80, h100-80, rtxpro-96, h200-141, b200-192
#   spot  : vast   h100-80, rtx4090-24
# Azure has no H200 SKU in eastus as of 2026-09-16 (Count: 0 for a live
# productName-contains-'H200' query) — a real, verified absence, not an
# oversight; do not "fix" this by guessing a SKU name.
# ---------------------------------------------------------------------------
SOURCE_MAP = {
    "t4-16": {
        "hyper": {"primary": {"kind": "aws", "instanceType": "g4dn.xlarge", "divisor": 1}},
        "spec": {"manual": "not on Lambda's or CoreWeave's current lineup (checked 2026-09-16)"},
        "spot": {"manual": "Vast gpu_name for T4 not verified against a live query in this build"},
    },
    "l4-24": {
        "hyper": {"primary": {"kind": "aws", "instanceType": "g6.xlarge", "divisor": 1}},
        "spec": {"manual": "not on Lambda's or CoreWeave's current lineup (checked 2026-09-16)"},
        "spot": {"manual": "Vast gpu_name for L4 not verified against a live query in this build"},
    },
    "rtx4090-24": {
        "hyper": {"manual": "no hyperscaler rents this card"},
        "spec": {"manual": "not on Lambda's or CoreWeave's current lineup (checked 2026-09-16)"},
        "spot": {"primary": {"kind": "vast", "gpuName": "RTX 4090", "divisor": 1,
                              "minSample": 5, "numGpus": 1}},
    },
    "rtx5090-32": {
        "hyper": {"manual": "no hyperscaler rents this card"},
        "spec": {"manual": "not on Lambda's or CoreWeave's current lineup (checked 2026-09-16)"},
        "spot": {"manual": "Vast gpu_name for RTX 5090 not verified against a live query in this build"},
    },
    "a100-40": {
        "hyper": {
            "primary": {"kind": "azure", "sku": "Standard_ND96asr_A100_v4", "region": "eastus",
                        "meterName": "ND96asrA100v4_NU", "divisor": 8},
            "secondary": [{"kind": "aws", "instanceType": "p4d.24xlarge", "divisor": 8}],
        },
        "spec": {"primary": {"kind": "lambda", "plan": "NVIDIA A100 SXM", "vram": "40 GB", "vcpus": "124"}},
        "spot": {"manual": "Vast gpu_name for A100 40GB not verified against a live query in this build"},
    },
    "rtx6000ada-48": {
        "hyper": {"manual": "no hyperscaler rents this card"},
        # Lambda's "A6000" and CoreWeave's "rtx-a6000" are both the Ampere-generation
        # RTX A6000, a different card from the Ada-generation RTX 6000 Ada this slug
        # names — checked live 2026-09-16, do not conflate on the name substring alone.
        "spec": {"manual": "Lambda/CoreWeave's 'A6000' is the older Ampere RTX A6000, "
                            "not the Ada-generation RTX 6000 Ada this slug names"},
        "spot": {"manual": "Vast gpu_name for RTX 6000 Ada not verified against a live query in this build"},
    },
    "l40s-48": {
        "hyper": {"primary": {"kind": "aws", "instanceType": "g6e.xlarge", "divisor": 1}},
        "spec": {"primary": {"kind": "coreweave", "productSlug": "nvidia-l40s", "name": "NVIDIA L40S",
                              "vram": "48", "divisor": 8}},
        "spot": {"manual": "Vast gpu_name for L40S not verified against a live query in this build"},
    },
    "a100-80": {
        "hyper": {
            "primary": {"kind": "azure", "sku": "Standard_ND96amsr_A100_v4", "region": "eastus",
                        "meterName": "ND96amsr A100 v4", "divisor": 8},
            "secondary": [{"kind": "aws", "instanceType": "p4de.24xlarge", "divisor": 8}],
        },
        "spec": {
            "primary": {"kind": "lambda", "plan": "NVIDIA A100 SXM", "vram": "80 GB", "vcpus": "240"},
            "secondary": [{"kind": "coreweave", "productSlug": "nvidia-a100", "name": "NVIDIA A100",
                            "vram": "80", "divisor": 8}],
        },
        "spot": {"manual": "Vast gpu_name for A100 80GB not verified against a live query in this build"},
    },
    "h100-80": {
        "hyper": {
            "primary": {"kind": "azure", "sku": "Standard_ND96isr_H100_v5", "region": "eastus",
                        "meterName": "ND96isrH100v5", "divisor": 8},
            "secondary": [{"kind": "aws", "instanceType": "p5.48xlarge", "divisor": 8}],
        },
        "spec": {
            "primary": {"kind": "lambda", "plan": "NVIDIA H100 SXM", "vram": "80 GB", "vcpus": "208"},
            "secondary": [{"kind": "coreweave", "productSlug": "hgx-h100", "name": "NVIDIA HGX H100",
                            "vram": "80", "divisor": 8}],
        },
        "spot": {"primary": {"kind": "vast", "gpuName": "H100 SXM", "divisor": 1,
                              "minSample": 5, "numGpus": 1}},
    },
    "rtxpro-96": {
        "hyper": {"manual": "no hyperscaler rents this card"},
        "spec": {"primary": {"kind": "coreweave", "productSlug": "nvidia-rtx-pro-6000-blackwell-server-edition",
                              "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition (High Memory)",
                              "vram": "96", "divisor": 8}},
        "spot": {"manual": "Vast gpu_name for RTX PRO 6000 not verified against a live query in this build"},
    },
    "h200-141": {
        # Azure: no H200 SKU in eastus today (Count=0, verified live) — see SOURCE_MAP's
        # own docstring above. AWS is the only automated hyper candidate found.
        "hyper": {"primary": {"kind": "aws", "instanceType": "p5en.48xlarge", "divisor": 8}},
        "spec": {"primary": {"kind": "coreweave", "productSlug": "hgx-h200", "name": "NVIDIA HGX H200",
                              "vram": "141", "divisor": 8}},
        "spot": {"manual": "Vast gpu_name for H200 not verified against a live query in this build"},
    },
    "b200-192": {
        "hyper": {"primary": {"kind": "aws", "instanceType": "p6-b200.48xlarge", "divisor": 8}},
        # Both Lambda and CoreWeave publish this board's VRAM as 180 GB, not the
        # catalog's 192 GB (gb is out of scope for this commit — see the PR body
        # finding; NVIDIA's own B200 SXM datasheet lists 180 GB HBM3e). The board
        # match itself (HGX B200, 8-GPU SXM node) is unambiguous on both pages, so
        # the anchor pins the source's own real value (180) — the anchor's job is
        # to fingerprint the SOURCE page's shape, not to re-validate the catalog.
        "spec": {
            "primary": {"kind": "lambda", "plan": "NVIDIA B200 SXM6", "vram": "180 GB", "vcpus": "208"},
            "secondary": [{"kind": "coreweave", "productSlug": "nvidia-b200", "name": "NVIDIA HGX B200",
                            "vram": "180", "divisor": 8}],
        },
        "spot": {"manual": "Vast gpu_name for B200 not verified against a live query in this build"},
    },
}


def _needed_kinds(source_map):
    kinds = set()
    for tiers in source_map.values():
        for cfg in tiers.values():
            if "primary" not in cfg:
                continue
            for spec in [cfg["primary"]] + cfg.get("secondary", []):
                kinds.add(spec["kind"])
    return kinds


def prime_shared_sources(needed_kinds):
    """Fetch and shape-check the sources that serve every slug from one page
    (aws/lambda/coreweave), once each, regardless of how many slugs use them.
    A failure here is stored, not raised — every slug/tier that needs this
    source aborts individually, with this reason, when it is looked up."""
    shared = {}
    if "aws" in needed_kinds:
        try:
            shared["aws"] = fetch_aws_feed()
        except SourceError as e:
            shared["aws"] = e
    if "lambda" in needed_kinds:
        try:
            shared["lambda_rows"] = parse_lambda_rows(fetch_lambda_page())
        except SourceError as e:
            shared["lambda_rows"] = e
    if "coreweave" in needed_kinds:
        try:
            shared["coreweave_html"] = fetch_coreweave_page()
        except SourceError as e:
            shared["coreweave_html"] = e
    return shared


def _fetch_one(spec, shared):
    kind = spec["kind"]
    if kind == "azure":
        return fetch_azure(spec["sku"], spec["region"], spec["meterName"], spec["divisor"])
    if kind == "aws":
        result = shared.get("aws")
        if result is None:
            raise SourceError("aws: source not primed for this run (see --only)")
        if isinstance(result, SourceError):
            raise result
        region_map, _manifest = result
        return select_aws(region_map, spec["instanceType"], spec["divisor"])
    if kind == "lambda":
        result = shared.get("lambda_rows")
        if result is None:
            raise SourceError("lambda: source not primed for this run (see --only)")
        if isinstance(result, SourceError):
            raise result
        return select_lambda(result, spec["plan"], spec["vram"], spec["vcpus"])
    if kind == "coreweave":
        result = shared.get("coreweave_html")
        if result is None:
            raise SourceError("coreweave: source not primed for this run (see --only)")
        if isinstance(result, SourceError):
            raise result
        return select_coreweave(result, spec["productSlug"], spec["name"], spec["vram"], spec["divisor"])
    if kind == "vast":
        return fetch_vast(spec["gpuName"], divisor=spec.get("divisor", 1),
                           min_sample=spec.get("minSample", 5), num_gpus=spec.get("numGpus", 1))
    raise SourceError(f"unknown source kind {kind!r}")


def _classify(current, fetched):
    if not isinstance(current, (int, float)) or isinstance(current, bool) or current <= 0:
        # Every catalog row carries hyper/spec/spot today (required fields in
        # tools/sync_data.py's GPU_FIELDS) — this is a defensive floor, not a
        # path exercised by the current catalog.
        return "MOVED", round(fetched, 2)
    delta = abs(fetched - current) / current
    if delta > FLAG_THRESHOLD:
        return "FLAGGED", None  # never propose a value across this line
    if delta <= CONFIRM_THRESHOLD:
        return "CONFIRMED", current
    return "MOVED", round(fetched, 2)


def _cross_check_disagreement(reading, secondary_readings):
    """None if every secondary reading that actually resolved agrees with the
    primary within CROSS_CHECK_FLAG_THRESHOLD; otherwise a human-readable note
    naming the worst offender. A SourceError secondary (an aborted cross-check,
    e.g. a 'Contact sales' row or a page shape change) is not disagreement —
    it already failed on its own terms and is reported separately as an
    aborted secondary, not folded into this."""
    worst = None
    for s in secondary_readings:
        if isinstance(s, SourceError):
            continue
        delta = abs(s.price_per_gpu - reading.price_per_gpu) / reading.price_per_gpu
        if delta > CROSS_CHECK_FLAG_THRESHOLD and (worst is None or delta > worst[0]):
            worst = (delta, s)
    if worst is None:
        return None
    delta, s = worst
    return (f"primary {reading.provider}:{reading.sku} (${reading.price_per_gpu:.4f}/GPU/hr) and "
            f"cross-check {s.provider}:{s.sku} (${s.price_per_gpu:.4f}/GPU/hr) disagree by {delta:+.1%}, "
            f"past the {CROSS_CHECK_FLAG_THRESHOLD:.0%} cross-check floor — flagged instead of applied, "
            "whatever _classify() made of the primary against the catalog")


# What this floor governs, and what it deliberately does not. Owner decision,
# 2026-09-22: it decides whether the tool may APPLY a new value on its own. It
# does not decide whether an already-attributed price is shown as sourced.
#
# Three tiers sit past it today — h100-80/hyper (Azure $12.29 against AWS
# $6.88), h100-80/spec (Lambda $3.99 against CoreWeave $6.16) and b200-192/spec
# (Lambda $6.69 against CoreWeave $8.60) — and all three keep their provenance.
# The disagreement is not a parsing error: those providers really do charge
# those amounts, and naming the provider is exactly what makes each price true.
# "The hyperscaler price for an H100" is not a quantity that exists, which is
# why the cost surfaces name a source at all.
#
# Stripping their provenance to satisfy this floor would leave the price on
# screen with nothing saying where it came from — the unattributed number this
# whole commit exists to remove. So the spread is reported to a person, above,
# and the label stays.


# A catalog tier recorded as null has no confirmed hourly price. An automated
# source cannot introduce one: there is nothing to confirm or move, and a price
# appearing where the catalog declared none is a decision for a person.
NULL_TIER_NOTE = ("the catalog records no confirmed hourly price for this tier (null), so an "
                  "automated source has nothing to confirm or move; a null tier needs a manual "
                  "SOURCE_MAP entry")


def automated_null_tiers(catalog_data, source_map):
    """Every slug/tier SOURCE_MAP would fetch although the catalog records it as null."""
    return sorted(f"{slug}/{tier}" for slug, tiers in source_map.items()
                  for tier, cfg in tiers.items()
                  if "primary" in cfg and tier in catalog_data.get(slug, {})
                  and catalog_data[slug][tier] is None)


def run(gpus_data, source_map, shared, only=None, slugs=None):
    outcomes = []
    for slug, row in gpus_data.items():
        if slugs is not None and slug not in slugs:
            continue
        for tier in ("hyper", "spec", "spot"):
            cfg = source_map.get(slug, {}).get(tier)
            if not cfg:
                outcomes.append(Outcome(slug, tier, "MANUAL", current=row.get(tier),
                                         note="no source configured for this slug/tier"))
                continue
            if "manual" in cfg:
                outcomes.append(Outcome(slug, tier, "MANUAL", current=row.get(tier), note=cfg["manual"]))
                continue
            if row.get(tier) is None:
                # main() refuses this map before any fetch; this is the same rule
                # for a caller that hands run() a map directly.
                outcomes.append(Outcome(slug, tier, "ABORTED", current=None, note=NULL_TIER_NOTE))
                continue
            primary = cfg["primary"]
            if only is not None and primary["kind"] not in only:
                outcomes.append(Outcome(slug, tier, "MANUAL", current=row.get(tier),
                                         note=f"excluded by --only (kind={primary['kind']!r})"))
                continue
            try:
                reading = _fetch_one(primary, shared)
            except SourceError as e:
                outcomes.append(Outcome(slug, tier, "ABORTED", current=row.get(tier), note=str(e)))
                continue
            secondary_readings = []
            for spec in cfg.get("secondary", []):
                if only is not None and spec["kind"] not in only:
                    continue
                try:
                    secondary_readings.append(_fetch_one(spec, shared))
                except SourceError as e:
                    secondary_readings.append(e)
            status, proposed = _classify(row.get(tier), reading.price_per_gpu)
            # A cross-check that disagrees hard with the primary overrides
            # CONFIRMED/MOVED the same way a >40% catalog move does: withhold
            # the value and surface it for a human, rather than let a primary
            # that merely agrees with a stale catalog sail through while a
            # live secondary is shouting that something is wrong.
            note = _cross_check_disagreement(reading, secondary_readings)
            if note and status in ("CONFIRMED", "MOVED"):
                status, proposed = "FLAGGED", None
            outcomes.append(Outcome(slug, tier, status, current=row.get(tier), proposed=proposed,
                                     reading=reading, secondary=secondary_readings, note=note or ""))
    return outcomes


def apply_outcomes(gpus_data, outcomes):
    """Mutate gpus_data (the catalog's "data" dict) in place for CONFIRMED and
    MOVED outcomes: MOVED rewrites the price, both rewrite priceSource for
    that tier. FLAGGED/ABORTED/MANUAL never touch the row. Returns whether
    anything changed, so the caller knows whether there is a file to write.

    priceSource[tier]["price"] is the reading this run actually compared
    against the catalog (oc.reading.price_per_gpu, rounded the same way a
    MOVED value is), not oc.proposed — _classify() sets proposed=current for
    CONFIRMED, which would make a CONFIRMED row's provenance simply echo the
    catalog back and defeat the point of recording what was read. For MOVED
    the two are the same number by construction (both round the same fetch),
    so this needs no CONFIRMED/MOVED branch of its own."""
    changed = False
    for oc in outcomes:
        if oc.status not in ("CONFIRMED", "MOVED"):
            continue
        row = gpus_data[oc.slug]
        if oc.status == "MOVED":
            row[oc.tier] = oc.proposed
        row.setdefault("priceSource", {})[oc.tier] = {
            "provider": oc.reading.provider, "sku": oc.reading.sku,
            "region": oc.reading.region, "date": oc.reading.date,
            "price": round(oc.reading.price_per_gpu, 2),
        }
        changed = True
    return changed


def _dump_json_value(v):
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{json.dumps(k)}: {_dump_json_value(x)}" for k, x in v.items()) + " }"
    return json.dumps(v)


def _dump_row_compact(slug, row):
    """Render one row the way data/gpus.json writes a GPU row: one line, no
    per-column padding. This used to describe itself as leaving the file's
    hand-tuned column alignment alone for every row this function did not
    touch — it did not, in practice: the first run that touched any row lost
    that row's padding for good, and every run since left the file a little
    more mixed, part padded, part compact, because a generic serializer
    cannot infer per-column padding to reproduce it. The fix is not a better
    serializer; it is one format for the whole file, produced here and
    applied to every row by apply_to_text() below, not only the row that
    changed."""
    parts = ", ".join(f"{json.dumps(k)}: {_dump_json_value(v)}" for k, v in row.items())
    return f'    {json.dumps(slug)}: {{ {parts} }}'


_META_LAST_UPDATED_RE = re.compile(r'("last_updated"\s*:\s*)"[^"]*"')


def apply_to_text(raw_text, gpus_data, changed_slugs, run_date):
    """Rewrite data/gpus.json's row block in one format, whenever there is at
    least one real change to write.

    An earlier version of this function replaced only the rows named in
    changed_slugs, on the theory that leaving every other byte alone
    preserved the file's hand-aligned column formatting for untouched rows.
    That theory was false in practice, as PR #24 demonstrated: eleven rows
    were touched over two runs and all eleven came out compact, while the
    rows no automated source has ever confirmed (rtx4090-24 before this
    commit, rtx5090-32, rtx6000ada-48) stayed in the old padded style
    forever — the file was, and would keep becoming, part one format, part
    the other. A generic serializer cannot reproduce hand-tuned per-column
    padding (see _dump_row_compact), so the fix is not a better serializer;
    it is to stop having two formats. Whenever there is at least one real
    change, every row present in gpus_data is rewritten in the one compact
    style, not only the rows in changed_slugs — which is what actually
    normalizes the stragglers, since a row with no automatable source (like
    the two above) is never itself in changed_slugs. A row whose rendering
    is already byte-identical to what's on disk produces no diff, so a
    normal run's diff is still confined to the rows that actually moved —
    the same property the old code was after, kept without a second format
    to drift back out of.

    changed_slugs still gates whether anything happens at all: no real
    change means the file, _meta.last_updated included, is untouched."""
    if not changed_slugs:
        return raw_text
    new_text = raw_text
    for slug, row in gpus_data.items():
        pattern = re.compile(r'^([ \t]*"' + re.escape(slug) + r'"\s*:\s*\{.*\},?)[ \t]*$', re.MULTILINE)
        m = pattern.search(new_text)
        if not m:
            raise SystemExit(
                f"apply: could not find {slug!r}'s row in data/gpus.json as a single line — "
                "the file's formatting has changed; update this row by hand instead")
        trailing_comma = "," if m.group(1).rstrip().endswith(",") else ""
        replacement = _dump_row_compact(slug, row) + trailing_comma
        new_text = new_text[:m.start()] + replacement + new_text[m.end():]
    new_text, n = _META_LAST_UPDATED_RE.subn(rf'\g<1>"{run_date}"', new_text, count=1)
    if n != 1:
        raise SystemExit("apply: could not find _meta.last_updated to update")
    return new_text


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_reading(r):
    if isinstance(r, SourceError):
        return f"(aborted: {r})"
    return f"{r.provider}:{r.sku} [{r.region}, {r.date}] — {r.evidence}"


def render_report(outcomes, generated_at):
    by_status = {"FLAGGED": [], "MOVED": [], "CONFIRMED": [], "ABORTED": [], "MANUAL": []}
    for oc in outcomes:
        by_status[oc.status].append(oc)

    lines = [f"# Price check — {generated_at}", ""]

    lines += ["## Flagged for manual review (>{:.0%} move, or cross-check disagreement — "
              "no value applied)".format(FLAG_THRESHOLD)]
    if by_status["FLAGGED"]:
        for oc in by_status["FLAGGED"]:
            delta = (oc.reading.price_per_gpu - oc.current) / oc.current if oc.current else float("nan")
            lines.append(f"- **{oc.slug}** / {oc.tier}: catalog {oc.current} -> live "
                         f"{oc.reading.price_per_gpu:.4f} ({delta:+.1%}) — {_fmt_reading(oc.reading)}")
            # Set only when this row was flagged (or downgraded from CONFIRMED/
            # MOVED) because a cross-check disagreed, not because the primary
            # itself moved >40% against the catalog — the delta above can be
            # small or even ~0% in that case, so the note is what explains why
            # a seemingly ordinary row still has no value applied.
            if oc.note:
                lines.append(f"    - {oc.note}")
            for s in oc.secondary:
                lines.append(f"    - cross-check: {_fmt_reading(s)}")
    else:
        lines.append("- none")
    lines.append("")

    lines += ["## Proposed updates (price + provenance)"]
    if by_status["MOVED"]:
        for oc in by_status["MOVED"]:
            delta = (oc.proposed - oc.current) / oc.current if oc.current else float("nan")
            lines.append(f"- **{oc.slug}** / {oc.tier}: {oc.current} -> {oc.proposed} "
                         f"({delta:+.1%}) — {_fmt_reading(oc.reading)}")
            for s in oc.secondary:
                lines.append(f"    - cross-check: {_fmt_reading(s)}")
    else:
        lines.append("- none")
    lines.append("")

    lines += ["## Confirmed unchanged (provenance refreshed only)"]
    if by_status["CONFIRMED"]:
        for oc in by_status["CONFIRMED"]:
            lines.append(f"- {oc.slug} / {oc.tier}: {oc.current} — {_fmt_reading(oc.reading)}")
    else:
        lines.append("- none")
    lines.append("")

    lines += ["## Could not validate this run (no PR content — see reasons)"]
    if by_status["ABORTED"]:
        for oc in by_status["ABORTED"]:
            lines.append(f"- {oc.slug} / {oc.tier}: {oc.note}")
    else:
        lines.append("- none")
    lines.append("")

    lines += ["## Not automated (manual by design)"]
    if by_status["MANUAL"]:
        for oc in sorted(by_status["MANUAL"], key=lambda o: (o.slug, o.tier)):
            lines.append(f"- {oc.slug} / {oc.tier}: {oc.note}")
    else:
        lines.append("- none")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_catalog():
    path = os.path.join(ROOT, "data", "gpus.json")
    with open(path) as f:
        return json.load(f), path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--report-out", help="also write the markdown report to this path")
    p.add_argument("--apply", action="store_true",
                    help="write confirmed/moved prices and provenance back to data/gpus.json")
    p.add_argument("--only", help="restrict to these comma-separated source kinds "
                                   "(azure,aws,lambda,coreweave,vast)")
    p.add_argument("--slug", action="append", help="restrict to this catalog slug (repeatable)")
    args = p.parse_args(argv)

    catalog, path = load_catalog()

    missing = set(SOURCE_MAP) - set(catalog["data"])
    if missing:
        raise SystemExit(f"SOURCE_MAP names slug(s) {sorted(missing)} that data/gpus.json no longer "
                          "has — the map has drifted from the catalog and needs fixing before this can run")

    on_null = automated_null_tiers(catalog["data"], SOURCE_MAP)
    if on_null:
        raise SystemExit(f"SOURCE_MAP automates {on_null}, which data/gpus.json records as null — "
                         f"{NULL_TIER_NOTE}")

    slugs = None
    if args.slug:
        unknown = set(args.slug) - set(catalog["data"])
        if unknown:
            raise SystemExit(f"--slug names {sorted(unknown)}, not in data/gpus.json")
        slugs = set(args.slug)

    only = set(k.strip() for k in args.only.split(",")) if args.only else None
    source_map = {k: v for k, v in SOURCE_MAP.items() if slugs is None or k in slugs}
    needed = _needed_kinds(source_map)
    if only is not None:
        needed &= only

    shared = prime_shared_sources(needed)
    run_date = today()
    outcomes = run(catalog["data"], source_map, shared, only=only, slugs=slugs)

    report = render_report(outcomes, generated_at=run_date)
    print(report)
    if args.report_out:
        with open(args.report_out, "w") as f:
            f.write(report)

    if args.apply:
        changed = apply_outcomes(catalog["data"], outcomes)
        if changed:
            changed_slugs = sorted({oc.slug for oc in outcomes if oc.status in ("CONFIRMED", "MOVED")})
            with open(path) as f:
                raw = f.read()
            new_raw = apply_to_text(raw, catalog["data"], changed_slugs, run_date)
            with open(path, "w") as f:
                f.write(new_raw)
            print(f"\n{path}: updated {len(changed_slugs)} row(s): {', '.join(changed_slugs)}")
            print("Re-run tools/sync_data.py (harmless no-op unless a row's price also "
                  "moved) and tests/run.sh before committing.")
        else:
            print(f"\n{path}: nothing confirmed or moved — no changes to write.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
