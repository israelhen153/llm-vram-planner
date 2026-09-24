#!/usr/bin/env python3
"""Verify tools/price_check.py's per-source parsing, validation and
classification — the design constraint it exists to satisfy is "a provider
page that changes shape must open NO PR rather than a wrong one", so most of
this file is deliberately shape-breaking fixtures that must all abort.

No test here makes a network call: tests/run.sh promises node + python3 and
nothing else, so every fixture below is a canned response standing in for a
live fetch — several are trimmed, real captures from the sources this tool
targets (2026-09-16), not invented shapes. The one thing this file cannot
exercise is "did I read the live schema correctly today" — that was done by
hand while building this tool and is reported in the PR, not asserted here.

Run:  python3 tests/price_check.test.py
"""
import contextlib
import copy
import json
import os
import re
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import price_check as pc

pass_ct = fail_ct = 0


def test(name, fn):
    global pass_ct, fail_ct
    try:
        fn()
        print(f"  ok   {name}")
        pass_ct += 1
    except Exception as e:
        print(f"  FAIL {name}\n       {type(e).__name__}: {e}")
        fail_ct += 1


def raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except pc.SourceError as e:
        return str(e)
    raise AssertionError(f"{fn.__name__} did not raise SourceError")


@contextlib.contextmanager
def fake_http_get(response):
    """Monkeypatch price_check.http_get for the duration of the with-block,
    then restore it. `response` is either a fixed (text, headers) tuple
    returned regardless of the URL, or a callable(url, **kwargs) -> that
    tuple for tests that need to branch on the URL."""
    fn = response if callable(response) else (lambda url, **kw: response)
    original = pc.http_get
    pc.http_get = fn
    try:
        yield
    finally:
        pc.http_get = original


# ===========================================================================
# Absolute band / zero-is-not-a-price
# ===========================================================================
print("\n_check_band: the absolute floor every source shares")

def check_band_rejects_zero():
    msg = raises(pc._check_band, 0, "test")
    assert "zero or negative" in msg

test("a zero price is refused, never treated as a value", check_band_rejects_zero)


def check_band_rejects_negative():
    raises(pc._check_band, -1.5, "test")

test("a negative price is refused", check_band_rejects_negative)


def check_band_rejects_too_low():
    msg = raises(pc._check_band, 0.001, "test")
    assert "sane band" in msg

test("a price below the absolute floor is refused (unit/divisor error)", check_band_rejects_too_low)


def check_band_rejects_too_high():
    raises(pc._check_band, 500.0, "test")

test("a price above the absolute ceiling is refused", check_band_rejects_too_high)


def check_band_accepts_in_range():
    pc._check_band(6.16, "test")  # must not raise

test("a sane per-GPU price passes the band check", check_band_accepts_in_range)


# ===========================================================================
# Azure — meterName exactness, the Windows-shares-meterName gotcha, zero rows
# ===========================================================================
print("\nfetch_azure: region+meterName+productName, all three, verified live 2026-09-16")

# Trimmed, real shape captured live for Standard_ND96isr_H100_v5/eastus: Linux
# and Windows share the SAME meterName, only productName differs.
AZURE_H100_ITEMS = [
    {"armRegionName": "eastus", "armSkuName": "Standard_ND96isr_H100_v5", "meterName": "ND96isrH100v5",
     "productName": "Virtual Machines NDsr H100 v5 Series", "type": "Consumption",
     "unitOfMeasure": "1 Hour", "currencyCode": "USD", "retailPrice": 98.32},
    {"armRegionName": "eastus", "armSkuName": "Standard_ND96isr_H100_v5", "meterName": "ND96isrH100v5",
     "productName": "Virtual Machines NDsr H100 v5 Series Windows", "type": "Consumption",
     "unitOfMeasure": "1 Hour", "currencyCode": "USD", "retailPrice": 102.736},
    {"armRegionName": "eastus", "armSkuName": "Standard_ND96isr_H100_v5", "meterName": "ND96isrH100v5 Spot",
     "productName": "Virtual Machines NDsr H100 v5 Series", "type": "Consumption",
     "unitOfMeasure": "1 Hour", "currencyCode": "USD", "retailPrice": 18.169536},
    {"armRegionName": "eastus", "armSkuName": "Standard_ND96isr_H100_v5", "meterName": "ND96isrH100v5 Low Priority",
     "productName": "Virtual Machines NDsr H100 v5 Series Windows", "type": "Consumption",
     "unitOfMeasure": "1 Hour", "currencyCode": "USD", "retailPrice": 41.094},
]


def _azure_body(items):
    return json.dumps({"BillingCurrency": "USD", "Items": items, "NextPageLink": None, "Count": len(items)})


def check_azure_isolates_linux_ondemand():
    with fake_http_get((_azure_body(AZURE_H100_ITEMS), {})):
        r = pc.fetch_azure("Standard_ND96isr_H100_v5", "eastus", "ND96isrH100v5", 8)
    assert abs(r.price_per_gpu - 98.32 / 8) < 1e-9, r.price_per_gpu
    assert r.provider == "azure" and r.region == "eastus"

test("Windows/Spot/Low-Priority rows are excluded even though Windows shares meterName with Linux",
     check_azure_isolates_linux_ondemand)


def check_azure_spot_meter_says_spot_in_its_sku():
    """Two tiers can be read off one Azure SKU — MI300X's hyperscaler price and
    its spot price are both ND96isr_MI300X_v5. The spot reading records "(Spot)"
    in its SKU, so the page never shows one label beside two prices; the
    on-demand reading keeps the SKU verbatim."""
    with fake_http_get((_azure_body(AZURE_H100_ITEMS), {})):
        spot = pc.fetch_azure("Standard_ND96isr_H100_v5", "eastus", "ND96isrH100v5 Spot", 8)
    assert abs(spot.price_per_gpu - 18.169536 / 8) < 1e-9, spot.price_per_gpu
    assert spot.sku == "Standard_ND96isr_H100_v5 (Spot)", spot.sku
    with fake_http_get((_azure_body(AZURE_H100_ITEMS), {})):
        on_demand = pc.fetch_azure("Standard_ND96isr_H100_v5", "eastus", "ND96isrH100v5", 8)
    assert on_demand.sku == "Standard_ND96isr_H100_v5", on_demand.sku

test("a spot meter's reading says Spot in its SKU; the on-demand reading keeps the SKU verbatim",
     check_azure_spot_meter_says_spot_in_its_sku)


def check_azure_zero_rows_aborts():
    with fake_http_get((_azure_body([]), {})):
        msg = raises(pc.fetch_azure, "Standard_ND97_Nonexistent_v5", "eastus", "X", 8)
    assert "expected exactly 1" in msg

test("zero rows (e.g. no H200 SKU in a region, verified live) aborts rather than guessing",
     check_azure_zero_rows_aborts)


def check_azure_missing_envelope_key_aborts():
    with fake_http_get((json.dumps({"NotItems": []}), {})):
        msg = raises(pc.fetch_azure, "sku", "eastus", "m", 8)
    assert "shape changed" in msg

test("a response missing the Items/Count envelope keys aborts as a shape change",
     check_azure_missing_envelope_key_aborts)


def check_azure_ambiguous_rows_abort():
    dup = [dict(AZURE_H100_ITEMS[0]), dict(AZURE_H100_ITEMS[0])]
    with fake_http_get((_azure_body(dup), {})):
        msg = raises(pc.fetch_azure, "Standard_ND96isr_H100_v5", "eastus", "ND96isrH100v5", 8)
    assert "resolved to 2 rows" in msg

test("two rows surviving the filter abort instead of picking one", check_azure_ambiguous_rows_abort)


def check_azure_non_usd_aborts():
    row = dict(AZURE_H100_ITEMS[0])
    row["currencyCode"] = "EUR"
    with fake_http_get((_azure_body([row]), {})):
        raises(pc.fetch_azure, "Standard_ND96isr_H100_v5", "eastus", "ND96isrH100v5", 8)

test("a non-USD currency aborts", check_azure_non_usd_aborts)


# ===========================================================================
# AWS — the compound-key trap this tool's own live check found
# ===========================================================================
print("\nfetch_aws_feed/select_aws: keys are compound strings, not the instance type")

AWS_FEED_FIXTURE = {
    "manifest": {"hawkFilePublicationDate": "2026-09-10T19:55:14Z"},
    "regions": {
        "US East (N. Virginia)": {
            # The real, live-verified key shape: "<type> <region-ish> Linux", not "p5.48xlarge".
            "p5 48xlarge US East N. Virginia Linux": {
                "Instance Type": "p5.48xlarge", "Operating System": "Linux", "price": "55.0400000000"},
            "p4d 24xlarge US East N. Virginia Linux": {
                "Instance Type": "p4d.24xlarge", "Operating System": "Linux", "price": "21.9576420000"},
        }
    },
}


def check_aws_feed_shape_and_lookup():
    with fake_http_get((json.dumps(AWS_FEED_FIXTURE), {})):
        region_map, manifest = pc.fetch_aws_feed()
    assert manifest["hawkFilePublicationDate"] == "2026-09-10T19:55:14Z"
    # Confirms the trap this tool's own live check found: a bare-instance-type
    # lookup into region_map would silently return nothing.
    assert region_map.get("p5.48xlarge") is None
    r = pc.select_aws(region_map, "p5.48xlarge", 8)
    assert abs(r.price_per_gpu - 55.04 / 8) < 1e-9

test("select_aws finds a row by its 'Instance Type' field, not by dict key",
     check_aws_feed_shape_and_lookup)


def check_aws_missing_top_level_keys_aborts():
    with fake_http_get((json.dumps({"nope": True}), {})):
        msg = raises(pc.fetch_aws_feed)
    assert "shape changed" in msg

test("a feed missing manifest/regions aborts as a shape change", check_aws_missing_top_level_keys_aborts)


def check_aws_zero_matches_abort():
    region_map, _ = AWS_FEED_FIXTURE["regions"]["US East (N. Virginia)"], None
    msg = raises(pc.select_aws, region_map, "p9.999xlarge", 8)
    assert "expected exactly 1" in msg

test("an instance type with zero matches aborts", check_aws_zero_matches_abort)


def check_aws_duplicate_matches_abort():
    region_map = {
        "a": {"Instance Type": "p5.48xlarge", "Operating System": "Linux", "price": "55.04"},
        "b": {"Instance Type": "p5.48xlarge", "Operating System": "Linux", "price": "99.99"},
    }
    msg = raises(pc.select_aws, region_map, "p5.48xlarge", 8)
    assert "2 rows matched" in msg

test("two rows matching the same instance type abort instead of picking one",
     check_aws_duplicate_matches_abort)


def check_aws_non_linux_aborts():
    region_map = {"a": {"Instance Type": "p5.48xlarge", "Operating System": "Windows", "price": "70.00"}}
    raises(pc.select_aws, region_map, "p5.48xlarge", 8)

test("a non-Linux row aborts", check_aws_non_linux_aborts)


def check_aws_non_numeric_price_aborts():
    region_map = {"a": {"Instance Type": "p5.48xlarge", "Operating System": "Linux", "price": "call us"}}
    msg = raises(pc.select_aws, region_map, "p5.48xlarge", 8)
    assert "non-numeric" in msg

test("a non-numeric price aborts", check_aws_non_numeric_price_aborts)


# ===========================================================================
# Lambda — data-plan rows, the escaped-duplicate exclusion, scale pinned by vCPUs
# ===========================================================================
print("\nLambda: PRICE/GPU/HR rows, JSON-escaped duplicates excluded by construction")

LAMBDA_HTML = '''
<table><tbody>
<tr class="_row_1" data-plan="NVIDIA H100 SXM"><th data-label="Plan">NVIDIA H100 SXM</th><td data-label="VRAM/GPU">80 GB</td><td data-label="vCPUs">208</td><td data-label="PRICE/GPU/HR*">$3.99</td></tr>
<tr class="_row_1" data-plan="NVIDIA H100 SXM"><th data-label="Plan">NVIDIA H100 SXM</th><td data-label="VRAM/GPU">80 GB</td><td data-label="vCPUs">26</td><td data-label="PRICE/GPU/HR*">$4.29</td></tr>
<tr class="_row_1" data-plan="NVIDIA V100"><th data-label="Plan">NVIDIA V100</th><td data-label="VRAM/GPU">16 GB</td><td data-label="vCPUs">90</td><td data-label="PRICE/GPU/HR*">\u2014</td></tr>
</tbody></table>
<script>var hydration = "...\\u003Ctr class=\\"_row_1\\" data-plan=\\"NVIDIA H100 SXM\\"\\u003E...data-label=\\"PRICE/GPU/HR*\\"\\u003E$99.99...";</script>
'''


def check_lambda_anchor_required():
    with fake_http_get(("<html>no pricing here</html>", {})):
        raises(pc.fetch_lambda_page)

test("a page missing the PRICE/GPU/HR anchor aborts before parsing", check_lambda_anchor_required)


def check_lambda_escaped_duplicate_excluded():
    rows = pc.parse_lambda_rows(LAMBDA_HTML)
    # Exactly the 3 plain rows — the JSON-escaped $99.99 duplicate contributes nothing.
    assert len(rows) == 3, rows
    assert all(r["price_text"] != "$99.99" for r in rows)

test("the JSON-escaped duplicate table never reaches parse_lambda_rows's output",
     check_lambda_escaped_duplicate_excluded)


def check_lambda_selects_by_plan_vram_and_vcpus():
    rows = pc.parse_lambda_rows(LAMBDA_HTML)
    r = pc.select_lambda(rows, "NVIDIA H100 SXM", "80 GB", "208")
    assert r.price_per_gpu == 3.99
    # The 1x row (26 vCPUs) for the same plan/vram must NOT match a query pinned to 208.
    r2 = pc.select_lambda(rows, "NVIDIA H100 SXM", "80 GB", "26")
    assert r2.price_per_gpu == 4.29

test("vCPUs pins the node-scale row (8x vs 1x) when plan+VRAM alone is ambiguous",
     check_lambda_selects_by_plan_vram_and_vcpus)


def check_lambda_missing_price_aborts():
    rows = pc.parse_lambda_rows(LAMBDA_HTML)
    msg = raises(pc.select_lambda, rows, "NVIDIA V100", "16 GB", "90")
    assert "no price listed" in msg

test("an em-dash price cell aborts rather than being read as zero", check_lambda_missing_price_aborts)


def check_lambda_zero_matches_abort():
    rows = pc.parse_lambda_rows(LAMBDA_HTML)
    msg = raises(pc.select_lambda, rows, "NVIDIA H100 SXM", "40 GB", "208")
    assert "expected exactly 1" in msg

test("a plan/VRAM/vCPUs combination with no row aborts", check_lambda_zero_matches_abort)


def check_lambda_ambiguous_matches_abort():
    dup_html = LAMBDA_HTML.replace(
        '<tr class="_row_1" data-plan="NVIDIA V100">',
        '<tr class="_row_1" data-plan="NVIDIA H100 SXM"><th data-label="Plan">NVIDIA H100 SXM</th>'
        '<td data-label="VRAM/GPU">80 GB</td><td data-label="vCPUs">208</td>'
        '<td data-label="PRICE/GPU/HR*">$3.79</td></tr>\n<tr class="_row_1" data-plan="NVIDIA V100">')
    rows = pc.parse_lambda_rows(dup_html)
    msg = raises(pc.select_lambda, rows, "NVIDIA H100 SXM", "80 GB", "208")
    assert "2 rows matched" in msg

test("two rows for the same plan/VRAM/vCPUs abort instead of picking one",
     check_lambda_ambiguous_matches_abort)


def check_lambda_zero_rows_total_aborts():
    msg = raises(pc.parse_lambda_rows, "<html>data-label=\"PRICE/GPU/HR*\" but no rows</html>")
    assert "zero pricing rows" in msg

test("an anchor present but zero parseable rows aborts (structure changed)",
     check_lambda_zero_rows_total_aborts)


# ===========================================================================
# CoreWeave — duplicated blocks, spot never trusted, GPU Count / divisor cross-check
# ===========================================================================
print("\nCoreWeave: duplicated DOM blocks, on-demand must agree, spot is never read")

def _coreweave_block(slug, name, gpu_count, vram, on_demand, spot, per_gpu=None):
    per_gpu_span = (f'<span class="inference-price">Inference Single CPU Price: '
                     f'<span class="item-value">${per_gpu:.2f}</span> / Hour</span>') if per_gpu is not None else \
                    '<span class="inference-price">Inference Single CPU Price: <span class="item-value"></span> / Hour</span>'
    od_span = (f'<span class="item-value">${on_demand:.2f}</span>' if on_demand is not None
               else '<span class="item-value"></span>')
    spot_span = (f'<span class="item-value">${spot:.2f}</span>' if spot is not None
                 else '<span class="item-value">N/A</span>')
    return f'''
<h3 data-product="{slug}" class="table-model-name">{name}</h3>
<div class="table-cell-column table-cell-column-left">
  <div class="table-meta-text"><div class="table-meta-value">
    <span class="instance-price">On-Demand Price: {od_span} / Hour<br/></span>
    <span class="spot-price">Spot Price: {spot_span} / Hour<br/></span>
    {per_gpu_span}
  </div></div>
</div>
<div class="table-cell-column table-cell-column-right"><div class="table-cell-column-contents">
  <div class="table-meta-text table-meta-text-right"><div class="table-meta-value">{gpu_count}</div><div>GPU Count</div></div>
  <div class="table-meta-text table-meta-text-right"><div class="table-meta-value">{vram}</div><div>VRAM</div></div>
</div></div>
'''


def check_coreweave_happy_path():
    html = _coreweave_block("hgx-h100", "NVIDIA HGX H100", 8, "80", 49.24, 19.71, per_gpu=6.16)
    r = pc.select_coreweave(html, "hgx-h100", "NVIDIA HGX H100", "80", 8)
    assert abs(r.price_per_gpu - 6.16) < 1e-9
    assert "spot" not in r.sku.lower()

test("the page's own per-GPU figure is used when it agrees with on-demand/count within 1%",
     check_coreweave_happy_path)


def check_coreweave_spot_disagreement_is_ignored():
    # Reproduces what this tool's own live check found today: the SAME product's
    # spot price differs between duplicated blocks in one fetch. On-demand agrees.
    html = (_coreweave_block("hgx-h100", "NVIDIA HGX H100", 8, "80", 49.24, 19.71, per_gpu=6.16)
            + _coreweave_block("hgx-h100", "NVIDIA HGX H100", 8, "80", 49.24, 19.51, per_gpu=6.16))
    r = pc.select_coreweave(html, "hgx-h100", "NVIDIA HGX H100", "80", 8)
    assert abs(r.price_per_gpu - 6.16) < 1e-9  # succeeds despite the spot mismatch

test("disagreeing spot figures across duplicated blocks never abort the run (spot is unused)",
     check_coreweave_spot_disagreement_is_ignored)


def check_coreweave_on_demand_disagreement_aborts():
    html = (_coreweave_block("hgx-h100", "NVIDIA HGX H100", 8, "80", 49.24, 19.71, per_gpu=6.16)
            + _coreweave_block("hgx-h100", "NVIDIA HGX H100", 8, "80", 51.00, 19.71, per_gpu=6.16))
    msg = raises(pc.select_coreweave, html, "hgx-h100", "NVIDIA HGX H100", "80", 8)
    assert "disagreed" in msg

test("disagreeing on-demand figures across duplicated blocks abort — never guess which copy is right",
     check_coreweave_on_demand_disagreement_aborts)


def check_coreweave_contact_sales_aborts():
    html = _coreweave_block("nvidia-gb300-nvl72", "NVIDIA GB300 NVL72", 4, "279", None, None, per_gpu=None)
    msg = raises(pc.select_coreweave, html, "nvidia-gb300-nvl72", "NVIDIA GB300 NVL72", "279", 4)
    assert "Contact sales" in msg or "no On-Demand price" in msg

test("a 'Contact sales' row (no price at all) aborts, never reads as zero",
     check_coreweave_contact_sales_aborts)


def check_coreweave_vram_mismatch_aborts():
    html = _coreweave_block("nvidia-b200", "NVIDIA HGX B200", 8, "180", 68.80, 34.11, per_gpu=8.60)
    # Configured for 192 (a catalog value) when the page says 180 — must abort,
    # not silently accept a board whose VRAM doesn't match what was pinned.
    msg = raises(pc.select_coreweave, html, "nvidia-b200", "NVIDIA HGX B200", "192", 8)
    assert "VRAM" in msg

test("a VRAM mismatch against the configured anchor aborts", check_coreweave_vram_mismatch_aborts)


def check_coreweave_divisor_mismatch_aborts():
    html = _coreweave_block("hgx-h100", "NVIDIA HGX H100", 8, "80", 49.24, 19.71, per_gpu=6.16)
    msg = raises(pc.select_coreweave, html, "hgx-h100", "NVIDIA HGX H100", "80", 4)
    assert "divisor" in msg

test("the page's own GPU Count disagreeing with the configured divisor aborts",
     check_coreweave_divisor_mismatch_aborts)


def check_coreweave_cross_check_mismatch_aborts():
    # On-demand/GPU-count = 49.24/8 = 6.155, more than 1% away from a page figure of 7.00.
    html = _coreweave_block("hgx-h100", "NVIDIA HGX H100", 8, "80", 49.24, 19.71, per_gpu=7.00)
    msg = raises(pc.select_coreweave, html, "hgx-h100", "NVIDIA HGX H100", "80", 8)
    assert "disagrees" in msg

test("on-demand/GPU-count disagreeing with the page's own per-GPU figure by >1% aborts",
     check_coreweave_cross_check_mismatch_aborts)


def check_coreweave_no_title_aborts():
    html = _coreweave_block("hgx-h100", "NVIDIA HGX H100", 8, "80", 49.24, 19.71, per_gpu=6.16)
    msg = raises(pc.select_coreweave, html, "hgx-h200", "NVIDIA HGX H200", "141", 8)
    assert "no row titled" in msg

test("a product slug that no longer appears on the page aborts", check_coreweave_no_title_aborts)


def check_coreweave_falls_back_to_division_without_page_figure():
    html = _coreweave_block("nvidia-l40s", "NVIDIA L40S", 8, "48", 18.00, 7.88, per_gpu=None)
    r = pc.select_coreweave(html, "nvidia-l40s", "NVIDIA L40S", "48", 8)
    assert abs(r.price_per_gpu - 18.00 / 8) < 1e-9

test("falls back to on-demand/GPU-count when the page carries no separate per-GPU figure",
     check_coreweave_falls_back_to_division_without_page_figure)


# ===========================================================================
# Vast.ai — sample floor, median (never min), structured error envelope
# ===========================================================================
print("\nVast.ai: sample-size floor, median over min, the error envelope")

def _vast_body(dph_list, success=True):
    if not success:
        return json.dumps({"success": False, "error": "bad_request", "msg": "q must be valid JSON"})
    return json.dumps({"offers": [{"dph_total": d} for d in dph_list], "truncated": False})


def check_vast_uses_median_not_min():
    with fake_http_get((_vast_body([2.74, 3.03, 3.35, 4.09, 4.57]), {})):
        r = pc.fetch_vast("H100 SXM")
    assert r.price_per_gpu == 3.35, r.price_per_gpu  # the median, not 2.74 (the min)

test("the median is used, never the minimum offer", check_vast_uses_median_not_min)


def check_vast_even_sample_averages_middle_two():
    with fake_http_get((_vast_body([1.0, 2.0, 3.0, 4.0]), {})):
        r = pc.fetch_vast("X", min_sample=4)
    assert r.price_per_gpu == 2.5

test("an even-sized sample averages the middle two", check_vast_even_sample_averages_middle_two)


def check_vast_thin_sample_aborts():
    with fake_http_get((_vast_body([2.74, 3.03, 3.35]), {})):
        msg = raises(pc.fetch_vast, "H100 SXM")  # default min_sample=5, only 3 offered
    assert "thin sample" in msg

test("fewer offers than the sample floor aborts rather than trusting a thin median",
     check_vast_thin_sample_aborts)


def check_vast_error_envelope_aborts():
    with fake_http_get((_vast_body([], success=False), {})):
        msg = raises(pc.fetch_vast, "H100 SXM")
    assert "error envelope" in msg

test("a {success: false} error envelope aborts with the API's own message",
     check_vast_error_envelope_aborts)


def check_vast_missing_offers_key_aborts():
    with fake_http_get((json.dumps({"nope": []}), {})):
        raises(pc.fetch_vast, "H100 SXM")

test("a response missing the 'offers' key aborts as a shape change",
     check_vast_missing_offers_key_aborts)


def check_vast_num_gpus_divides():
    with fake_http_get((_vast_body([8.0, 9.0, 10.0, 11.0, 12.0]), {})):
        r = pc.fetch_vast("8x thing", num_gpus=4)
    assert r.price_per_gpu == 2.5  # 10.0 (median dph_total) / 4

test("dph_total is divided by num_gpus before the median is taken",
     check_vast_num_gpus_divides)


# ===========================================================================
# Classification: CONFIRMED / MOVED / FLAGGED thresholds
# ===========================================================================
print("\n_classify: the 1% confirm floor and the 40% flag ceiling")

def check_classify_tiny_delta_confirms():
    status, proposed = pc._classify(12.3, 12.31)
    assert status == "CONFIRMED" and proposed == 12.3

test("a sub-1% delta confirms without changing the stored value", check_classify_tiny_delta_confirms)


def check_classify_mid_delta_moves():
    status, proposed = pc._classify(10.0, 11.0)  # +10%
    assert status == "MOVED" and proposed == 11.0

test("a delta between 1% and 40% proposes the new (rounded) value", check_classify_mid_delta_moves)


def check_classify_large_delta_flags_and_withholds():
    status, proposed = pc._classify(10.0, 20.0)  # +100%
    assert status == "FLAGGED" and proposed is None

test("a delta over 40% flags for human review and proposes nothing",
     check_classify_large_delta_flags_and_withholds)


def check_classify_boundary_is_exclusive_on_flag():
    status, _ = pc._classify(10.0, 14.0)  # exactly +40%
    assert status == "MOVED", "exactly the flag threshold must not itself flag"
    status, _ = pc._classify(10.0, 14.0001)
    assert status == "FLAGGED"

test("the 40% boundary itself still proposes a value; only strictly past it flags",
     check_classify_boundary_is_exclusive_on_flag)


def check_classify_rounds_proposed_value():
    status, proposed = pc._classify(1.0, 1.23456)
    assert proposed == 1.23

test("a proposed value is rounded to 2 decimal places, matching the catalog's own style",
     check_classify_rounds_proposed_value)


# ===========================================================================
# apply_outcomes / apply_to_text — never touches an untouched row
# ===========================================================================
print("\napply_outcomes / apply_to_text: only CONFIRMED and MOVED rows change, nothing else does")

def _reading(provider="azure", sku="SKU", region="eastus", price=1.0, date="2026-09-16"):
    return pc.Reading(provider=provider, sku=sku, region=region, price_per_gpu=price,
                       date=date, evidence="test")


def check_apply_confirmed_only_touches_price_source():
    gpus = {"x": {"hyper": 5.0, "spec": 2.0, "spot": 1.0}}
    oc = pc.Outcome("x", "hyper", "CONFIRMED", current=5.0, proposed=5.0, reading=_reading(price=5.0))
    changed = pc.apply_outcomes(gpus, [oc])
    assert changed is True
    assert gpus["x"]["hyper"] == 5.0  # unchanged
    assert gpus["x"]["priceSource"]["hyper"]["provider"] == "azure"

test("CONFIRMED refreshes priceSource but never rewrites the price",
     check_apply_confirmed_only_touches_price_source)


def check_apply_moved_rewrites_price_and_source():
    gpus = {"x": {"hyper": 5.0}}
    oc = pc.Outcome("x", "hyper", "MOVED", current=5.0, proposed=4.2, reading=_reading(price=4.2))
    pc.apply_outcomes(gpus, [oc])
    assert gpus["x"]["hyper"] == 4.2
    assert gpus["x"]["priceSource"]["hyper"]["sku"] == "SKU"

test("MOVED rewrites both the price and priceSource", check_apply_moved_rewrites_price_and_source)


def check_apply_writes_the_reading_not_the_proposed_value_as_price():
    """priceSource[tier].price has to be the figure the run actually read and
    compared, not oc.proposed: _classify() sets proposed=current for
    CONFIRMED, so a naive `"price": oc.proposed` would make a CONFIRMED row's
    provenance simply echo the catalog value back — recording nothing new and
    defeating the point of a later test asserting the two agree."""
    gpus = {"x": {"hyper": 5.0}}
    # CONFIRMED: reading (5.03) differs slightly from current/proposed (5.0),
    # inside the 1% floor. price must be the reading, not the echoed current.
    oc = pc.Outcome("x", "hyper", "CONFIRMED", current=5.0, proposed=5.0, reading=_reading(price=5.03))
    pc.apply_outcomes(gpus, [oc])
    assert gpus["x"]["priceSource"]["hyper"]["price"] == 5.03, gpus["x"]["priceSource"]["hyper"]

test("CONFIRMED records the reading it compared, not an echo of the untouched catalog value",
     check_apply_writes_the_reading_not_the_proposed_value_as_price)


def check_apply_rounds_price_like_every_other_catalog_figure():
    gpus = {"x": {"hyper": 5.0}}
    oc = pc.Outcome("x", "hyper", "MOVED", current=5.0, proposed=4.2, reading=_reading(price=4.19951))
    pc.apply_outcomes(gpus, [oc])
    assert gpus["x"]["priceSource"]["hyper"]["price"] == 4.2, gpus["x"]["priceSource"]["hyper"]["price"]

test("priceSource.price is rounded to 2 decimals, matching the catalog's own style",
     check_apply_rounds_price_like_every_other_catalog_figure)


def check_apply_flagged_aborted_manual_never_touch_row():
    gpus = {"x": {"hyper": 5.0}}
    original = copy.deepcopy(gpus)
    outcomes = [
        pc.Outcome("x", "hyper", "FLAGGED", current=5.0, proposed=None, reading=_reading(price=50.0)),
        pc.Outcome("x", "hyper", "ABORTED", current=5.0, note="boom"),
        pc.Outcome("x", "hyper", "MANUAL", current=5.0, note="no source"),
    ]
    changed = pc.apply_outcomes(gpus, outcomes)
    assert changed is False
    assert gpus == original

test("FLAGGED, ABORTED and MANUAL outcomes never mutate the row",
     check_apply_flagged_aborted_manual_never_touch_row)


def check_apply_to_text_only_changes_named_rows():
    """"Only changes" is about VALUES, not bytes: apply_to_text() now rewrites
    every row's line (see the next test), but a row already in the canonical
    compact style renders byte-identical, so its line does not move in a real
    diff even though the function touched it. gpus here happens to describe
    a100-80 in exactly that style already, which is what this test checks —
    the drift-normalizing case (a row NOT already compact) is the next one."""
    raw = (
        '{\n'
        '  "_meta": {\n    "last_updated": "2026-08-24"\n  },\n'
        '  "data": {\n'
        '    "a100-80": { "gb": 80, "hyper": 4.5 },\n'
        '    "h100-80": { "gb": 80, "hyper": 12.3 }\n'
        '  }\n'
        '}\n'
    )
    gpus = {"a100-80": {"gb": 80, "hyper": 4.5}, "h100-80": {"gb": 80, "hyper": 12.29}}
    new_text = pc.apply_to_text(raw, gpus, ["h100-80"], "2026-09-16")
    parsed = json.loads(new_text)
    assert parsed["data"]["h100-80"]["hyper"] == 12.29
    assert parsed["data"]["a100-80"] == {"gb": 80, "hyper": 4.5}, "untouched row's value must not move"
    assert parsed["_meta"]["last_updated"] == "2026-09-16"
    # The untouched row's exact original line must still be present, byte for byte.
    assert '"a100-80": { "gb": 80, "hyper": 4.5 },' in new_text

test("apply_to_text leaves an already-canonical row's value and bytes alone",
     check_apply_to_text_only_changes_named_rows)


def check_apply_to_text_normalizes_a_stale_formatted_row_too():
    """The bug PR #24 shipped: a row nobody's run has ever touched (no
    automatable source, so it never appears in changed_slugs) keeps its old
    hand-padded formatting forever, while every row a run does touch loses
    its padding the first time and never gets it back — the file ends up
    part one style, part the other, and it gets worse every run. The fix is
    that whenever ANYTHING changes, every row in gpus_data is rewritten
    compact — including a100-80 here, which is untouched by VALUE (not in
    changed_slugs, and its dict is identical to what's on disk) but is
    reformatted anyway because h100-80 changed elsewhere in the same file."""
    raw = (
        '{\n'
        '  "_meta": {\n    "last_updated": "2026-08-24"\n  },\n'
        '  "data": {\n'
        '    "a100-80":    { "gb": 80,    "hyper": 4.5 },\n'
        '    "h100-80": { "gb": 80, "hyper": 12.3 }\n'
        '  }\n'
        '}\n'
    )
    gpus = {"a100-80": {"gb": 80, "hyper": 4.5}, "h100-80": {"gb": 80, "hyper": 12.29}}
    new_text = pc.apply_to_text(raw, gpus, ["h100-80"], "2026-09-16")
    parsed = json.loads(new_text)
    # The value is untouched either way — this is a formatting check, not a value one.
    assert parsed["data"]["a100-80"] == {"gb": 80, "hyper": 4.5}
    # But the padded line is gone, replaced by the same compact style every
    # other row uses — the whole point of the fix.
    assert '"a100-80":    {' not in new_text, "a100-80 kept its stale hand-padding"
    assert '"a100-80": { "gb": 80, "hyper": 4.5 },' in new_text, "a100-80 was not normalized to compact"

test("apply_to_text normalizes a row's stale formatting even when only a different row's value changed",
     check_apply_to_text_normalizes_a_stale_formatted_row_too)


def check_apply_to_text_no_changes_leaves_last_updated_alone():
    raw = '{\n  "_meta": {\n    "last_updated": "2026-08-24"\n  },\n  "data": {\n    "x": { "hyper": 1.0 }\n  }\n}\n'
    new_text = pc.apply_to_text(raw, {"x": {"hyper": 1.0}}, [], "2026-09-16")
    assert new_text == raw

test("no changed slugs means the file is byte-identical, last_updated included",
     check_apply_to_text_no_changes_leaves_last_updated_alone)


def check_apply_to_text_missing_row_errors_clearly():
    raw = '{\n  "data": {\n    "x": { "hyper": 1.0 }\n  }\n}\n'
    try:
        pc.apply_to_text(raw, {"y": {"hyper": 1.0}}, ["y"], "2026-09-16")
    except SystemExit as e:
        assert "y" in str(e)
        return
    raise AssertionError("a slug missing from the raw text should have raised SystemExit")

test("a slug that can't be found as a single line fails loudly, naming the slug",
     check_apply_to_text_missing_row_errors_clearly)


# ===========================================================================
# run() orchestration
# ===========================================================================
print("\nrun(): manual/aborted/flagged wiring, secondary failures never blocking primary")

def check_run_manual_entry_produces_manual_outcome():
    gpus = {"x": {"hyper": 1.0, "spec": 1.0, "spot": 1.0}}
    source_map = {"x": {"hyper": {"manual": "no hyperscaler rents this card"}}}
    outcomes = pc.run(gpus, source_map, shared={})
    hyper = next(o for o in outcomes if o.tier == "hyper")
    assert hyper.status == "MANUAL" and "no hyperscaler" in hyper.note

test("a manual config entry produces a MANUAL outcome with its reason",
     check_run_manual_entry_produces_manual_outcome)


def check_run_unconfigured_tier_is_also_manual():
    gpus = {"x": {"hyper": 1.0, "spec": 1.0, "spot": 1.0}}
    outcomes = pc.run(gpus, {}, shared={})  # nothing configured for slug "x" at all
    assert all(o.status == "MANUAL" for o in outcomes)
    assert all("no source configured" in o.note for o in outcomes)

test("a slug/tier entirely absent from SOURCE_MAP is MANUAL, not a crash",
     check_run_unconfigured_tier_is_also_manual)


def check_run_primary_failure_aborts_and_skips_secondary():
    gpus = {"x": {"hyper": 1.0}}
    calls = []

    def boom(spec, shared):
        calls.append(spec["kind"])
        raise pc.SourceError("boom")

    original = pc._fetch_one
    pc._fetch_one = boom
    try:
        source_map = {"x": {"hyper": {"primary": {"kind": "azure"}, "secondary": [{"kind": "aws"}]}}}
        outcomes = pc.run(gpus, source_map, shared={})
    finally:
        pc._fetch_one = original
    assert outcomes[0].status == "ABORTED"
    assert calls == ["azure"], "secondary must never be fetched once the primary aborts"

test("a failed primary aborts the tier and never attempts the secondary",
     check_run_primary_failure_aborts_and_skips_secondary)


def check_run_secondary_failure_does_not_affect_primary_status():
    gpus = {"x": {"hyper": 5.0}}

    def fake_fetch(spec, shared):
        if spec["kind"] == "primary-kind":
            return _reading(price=5.0)
        raise pc.SourceError("secondary is down")

    original = pc._fetch_one
    pc._fetch_one = fake_fetch
    try:
        source_map = {"x": {"hyper": {"primary": {"kind": "primary-kind"},
                                        "secondary": [{"kind": "secondary-kind"}]}}}
        outcomes = pc.run(gpus, source_map, shared={})
    finally:
        pc._fetch_one = original
    oc = outcomes[0]
    assert oc.status == "CONFIRMED"
    assert len(oc.secondary) == 1 and isinstance(oc.secondary[0], pc.SourceError)

test("a failing secondary is recorded but never changes the primary's classification",
     check_run_secondary_failure_does_not_affect_primary_status)


def check_run_only_filter_excludes_a_kind():
    gpus = {"x": {"hyper": 5.0}}
    source_map = {"x": {"hyper": {"primary": {"kind": "azure"}}}}
    outcomes = pc.run(gpus, source_map, shared={}, only={"aws"})  # azure not in the allowed set
    assert outcomes[0].status == "MANUAL"
    assert "--only" in outcomes[0].note

test("--only excluding a tier's configured kind reports it as manual for this run",
     check_run_only_filter_excludes_a_kind)


def check_run_never_fetches_a_null_tier():
    """A catalog tier recorded as null has no confirmed hourly price, so an
    automated source has nothing to confirm or move — and a price appearing
    where the catalog declared none is a person's decision. run() never fetches
    it, whatever the map says; a manual entry keeps its own reason."""
    calls = []

    def fetch(spec, shared):
        calls.append(spec["kind"])
        return _reading(price=5.0)

    gpus = {"x": {"hyper": None, "spec": None, "spot": 2.0}}
    source_map = {"x": {"hyper": {"primary": {"kind": "azure"}},
                        "spec": {"manual": "no specialised cloud lists it"},
                        "spot": {"primary": {"kind": "vast"}}}}
    original = pc._fetch_one
    pc._fetch_one = fetch
    try:
        outcomes = {o.tier: o for o in pc.run(gpus, source_map, shared={})}
    finally:
        pc._fetch_one = original
    assert calls == ["vast"], f"fetched {calls}: a null tier was sent to its source"
    assert outcomes["hyper"].status == "ABORTED" and "null" in outcomes["hyper"].note, outcomes["hyper"]
    assert outcomes["spec"].status == "MANUAL" and outcomes["spec"].note == "no specialised cloud lists it"
    assert outcomes["spot"].status != "ABORTED"

test("run() never fetches a tier the catalog records as null", check_run_never_fetches_a_null_tier)


def check_main_refuses_to_automate_a_null_tier():
    """Before any fetch: a SOURCE_MAP that automates a null tier is a
    configuration error, named, rather than a run that quietly skips it."""
    catalog = {"_meta": {}, "data": {"x": {"hyper": None, "spec": 1.0, "spot": 1.0}}}
    assert pc.automated_null_tiers(catalog["data"], {"x": {"hyper": {"primary": {"kind": "azure"}}}}) == ["x/hyper"]
    assert pc.automated_null_tiers(catalog["data"], {"x": {"hyper": {"manual": "none"}}}) == []
    assert pc.automated_null_tiers(catalog["data"], {"x": {"spec": {"primary": {"kind": "azure"}}}}) == []
    saved = pc.load_catalog, pc.SOURCE_MAP
    pc.load_catalog = lambda: (catalog, "/dev/null")
    pc.SOURCE_MAP = {"x": {"hyper": {"primary": {"kind": "azure"}}}}
    try:
        try:
            pc.main([])
        except SystemExit as e:
            assert "x/hyper" in str(e) and "null" in str(e), f"the refusal does not name the tier: {e}"
        else:
            raise AssertionError("main() ran with an automated source on a null tier")
    finally:
        pc.load_catalog, pc.SOURCE_MAP = saved

test("main() refuses a SOURCE_MAP that automates a null tier, before fetching",
     check_main_refuses_to_automate_a_null_tier)


def check_every_real_null_tier_is_manual():
    """The real catalog and the real map, held to the same rule."""
    with open(os.path.join(ROOT, "data", "gpus.json")) as f:
        rows = json.load(f)["data"]
    assert pc.automated_null_tiers(rows, pc.SOURCE_MAP) == []
    bad = [f"{slug}/{tier}" for slug, row in rows.items() for tier in ("hyper", "spec", "spot")
           if row.get(tier) is None and tier in (row.get("priceSource") or {})]
    assert not bad, f"a null tier carries a priceSource: {bad}"

test("in the real catalog, no null tier is automated or carries a priceSource",
     check_every_real_null_tier_is_manual)


def check_apply_keeps_a_null_tier_null():
    """apply_to_text() re-serialises every row. A null tier must come back as
    JSON null — not 0, not "None", not dropped."""
    raw = ('{\n  "_meta": {\n    "last_updated": "2026-08-24"\n  },\n  "data": {\n'
           '    "x": { "hyper": null, "spec": 1.0, "spot": 2.0 }\n  }\n}\n')
    new_text = pc.apply_to_text(raw, {"x": {"hyper": None, "spec": 1.5, "spot": 2.0}}, ["x"], "2026-09-23")
    assert '"hyper": null' in new_text and json.loads(new_text)["data"]["x"]["hyper"] is None, new_text

test("apply_to_text() writes a null tier back as null", check_apply_keeps_a_null_tier_null)


def check_apply_keeps_a_hand_record_intact():
    """--apply refreshes provenance on the tiers it read and re-serialises every
    row. A hand record on another tier of the same row has to come back as it
    was, URL included: the writer carried no fixture with one, so dropping it —
    or its URL — passed every test (cold check, round 1)."""
    rec = {"provider": "RunPod", "sku": "MI300X (Secure Cloud)", "region": "global",
           "date": "2026-09-23", "price": 2.39, "url": "https://www.runpod.io/gpu-models/mi300x"}
    raw = ('{\n  "_meta": {\n    "last_updated": "2026-09-22"\n  },\n  "data": {\n'
           '    "x": { "hyper": 6.0, "spec": 2.39, "spot": 1.11, "priceRecord": { "spec": '
           + json.dumps(rec) + ' } }\n  }\n}\n')
    gpus = json.loads(raw)["data"]
    confirmed = pc.Outcome("x", "hyper", "CONFIRMED", current=6.0, proposed=6.0,
                           reading=_reading(price=6.0, date="2026-09-23"))
    assert pc.apply_outcomes(gpus, [confirmed]), "a CONFIRMED outcome changed nothing"
    row = json.loads(pc.apply_to_text(raw, gpus, ["x"], "2026-09-23"))["data"]["x"]
    assert row.get("priceRecord") == {"spec": rec}, f"the hand record did not survive --apply: {row.get('priceRecord')}"
    assert "hyper" in row.get("priceSource", {}), "the tier that was read gained no provenance"

test("--apply refreshes one tier's provenance and keeps another tier's hand record, URL included",
     check_apply_keeps_a_hand_record_intact)


def check_apply_replaces_a_note_with_the_reading():
    """A note says why a tier has no source. When --apply records a reading for
    that tier, the note has to go: a tier carries exactly one of priceSource,
    priceRecord and priceNote, and h100-80's spot tier is automated and noted
    today, so the first weekly run that confirms it would otherwise leave the
    catalog breaking that rule, and the page showing a reason that is no
    longer true beside a price that now has a source. A note on another tier
    stays, and a row's last note going takes the empty key with it."""
    note = lambda reason: {"reason": reason, "checked": "2026-09-23"}
    raw = ('{\n  "_meta": {\n    "last_updated": "2026-09-22"\n  },\n  "data": {\n'
           '    "x": { "hyper": 6.0, "spec": 2.39, "spot": 1.11, "priceNote": { "spot": '
           + json.dumps(note("Held for a second read.")) + ', "spec": ' + json.dumps(note("No reader for this provider.")) + ' } }\n  }\n}\n')
    gpus = json.loads(raw)["data"]
    read = pc.Outcome("x", "spot", "CONFIRMED", current=1.11, proposed=1.11,
                      reading=_reading(price=1.11, date="2026-09-28"))
    assert pc.apply_outcomes(gpus, [read]), "a CONFIRMED outcome changed nothing"
    row = json.loads(pc.apply_to_text(raw, gpus, ["x"], "2026-09-28"))["data"]["x"]
    assert "spot" in row.get("priceSource", {}), "the tier that was read gained no provenance"
    assert row.get("priceNote") == {"spec": note("No reader for this provider.")}, (
        f"the read tier's note should be gone and the other kept: {row.get('priceNote')}")
    last = pc.Outcome("x", "spec", "CONFIRMED", current=2.39, proposed=2.39,
                      reading=_reading(price=2.39, date="2026-09-28"))
    pc.apply_outcomes(gpus, [last])
    assert "priceNote" not in gpus["x"], f"an empty priceNote was left behind: {gpus['x'].get('priceNote')}"

test("--apply replaces a tier's note with the reading it records, and keeps the others",
     check_apply_replaces_a_note_with_the_reading)


# ===========================================================================
# Cross-check disagreement: two live sources for one run, not catalog-vs-live
# ===========================================================================
print("\n_cross_check_disagreement / run(): a disagreeing secondary withholds the value")

def check_cross_check_agreement_is_silent():
    r = _reading(provider="lambda", sku="P", price=6.69)
    s = _reading(provider="coreweave", sku="Q", price=6.69 * 1.10)  # 10%, under the floor
    assert pc._cross_check_disagreement(r, [s]) is None

test("a secondary within the cross-check floor produces no note",
     check_cross_check_agreement_is_silent)


def check_cross_check_disagreement_is_named():
    # The live case this was built for: Lambda $6.69 vs CoreWeave $8.60, 28% apart.
    r = _reading(provider="lambda", sku="NVIDIA B200 SXM6", price=6.69)
    s = _reading(provider="coreweave", sku="NVIDIA HGX B200", price=8.60)
    note = pc._cross_check_disagreement(r, [s])
    assert note is not None
    assert "lambda" in note and "coreweave" in note
    assert "28" in note  # the delta itself, not just that one exists

test("a secondary past the cross-check floor is named, with both sources and the delta",
     check_cross_check_disagreement_is_named)


def check_cross_check_ignores_an_aborted_secondary():
    r = _reading(price=6.69)
    assert pc._cross_check_disagreement(r, [pc.SourceError("boom")]) is None

test("an aborted secondary (a SourceError) is not disagreement — it already failed on its own",
     check_cross_check_ignores_an_aborted_secondary)


def check_cross_check_takes_the_worst_of_several():
    r = _reading(price=10.0)
    close = _reading(provider="a", sku="close", price=10.5)     # 5%, agrees
    far = _reading(provider="b", sku="far", price=15.0)          # 50%, disagrees
    note = pc._cross_check_disagreement(r, [close, far])
    assert "far" in note and "close" not in note

test("with several secondaries, only the worst disagreement is named",
     check_cross_check_takes_the_worst_of_several)


def check_run_downgrades_a_disagreeing_confirm_to_flagged():
    """The exact live shape found 2026-09-22: the primary alone would CONFIRM
    (it agrees with the stale catalog value), but a cross-check disagrees hard
    — and CONFIRMED would otherwise refresh provenance for a number a second
    live source is actively contradicting."""
    # tier is "hyper", the first tier run() iterates, so outcomes[0] below is
    # unambiguously this tier's outcome and not an unconfigured tier's MANUAL.
    gpus = {"x": {"hyper": 6.69}}

    def fake_fetch(spec, shared):
        return {"primary-kind": _reading(provider="lambda", sku="P", price=6.69),
                "secondary-kind": _reading(provider="coreweave", sku="Q", price=8.60)}[spec["kind"]]

    original = pc._fetch_one
    pc._fetch_one = fake_fetch
    try:
        source_map = {"x": {"hyper": {"primary": {"kind": "primary-kind"},
                                        "secondary": [{"kind": "secondary-kind"}]}}}
        outcomes = pc.run(gpus, source_map, shared={})
    finally:
        pc._fetch_one = original
    oc = outcomes[0]
    assert oc.tier == "hyper"
    assert oc.status == "FLAGGED", f"expected FLAGGED, got {oc.status}"
    assert oc.proposed is None
    assert "disagree" in oc.note

test("a cross-check disagreement downgrades what would otherwise CONFIRM to FLAGGED",
     check_run_downgrades_a_disagreeing_confirm_to_flagged)


def check_run_downgrades_a_disagreeing_move_to_flagged():
    gpus = {"x": {"hyper": 5.0}}

    def fake_fetch(spec, shared):
        return {"primary-kind": _reading(provider="lambda", sku="P", price=5.3),
                "secondary-kind": _reading(provider="coreweave", sku="Q", price=8.0)}[spec["kind"]]

    original = pc._fetch_one
    pc._fetch_one = fake_fetch
    try:
        source_map = {"x": {"hyper": {"primary": {"kind": "primary-kind"},
                                        "secondary": [{"kind": "secondary-kind"}]}}}
        outcomes = pc.run(gpus, source_map, shared={})
    finally:
        pc._fetch_one = original
    oc = outcomes[0]
    assert oc.tier == "hyper"
    assert oc.status == "FLAGGED", f"expected FLAGGED, got {oc.status}"
    assert oc.proposed is None

test("a cross-check disagreement downgrades what would otherwise MOVE to FLAGGED",
     check_run_downgrades_a_disagreeing_move_to_flagged)


def check_run_agreeing_secondary_never_touches_status():
    gpus = {"x": {"hyper": 5.0}}

    def fake_fetch(spec, shared):
        return {"primary-kind": _reading(price=5.3),
                "secondary-kind": _reading(price=5.35)}[spec["kind"]]

    original = pc._fetch_one
    pc._fetch_one = fake_fetch
    try:
        source_map = {"x": {"hyper": {"primary": {"kind": "primary-kind"},
                                        "secondary": [{"kind": "secondary-kind"}]}}}
        outcomes = pc.run(gpus, source_map, shared={})
    finally:
        pc._fetch_one = original
    oc = outcomes[0]
    assert oc.tier == "hyper"
    assert oc.status == "MOVED" and oc.proposed == 5.3 and oc.note == ""

test("an agreeing secondary leaves MOVED/CONFIRMED and their note untouched",
     check_run_agreeing_secondary_never_touches_status)


def check_apply_never_touches_a_row_flagged_for_disagreement():
    """FLAGGED already means "never touch the row" (see
    check_apply_flagged_aborted_manual_never_touch_row above) — this pins
    that the disagreement path produces a real FLAGGED outcome, not a
    look-alike status apply_outcomes doesn't recognise."""
    gpus = {"x": {"spec": 6.69}}
    oc = pc.Outcome("x", "spec", "FLAGGED", current=6.69, proposed=None,
                     reading=_reading(provider="lambda", price=6.69),
                     secondary=[_reading(provider="coreweave", price=8.60)],
                     note="primary and cross-check disagree")
    changed = pc.apply_outcomes(gpus, [oc])
    assert changed is False
    assert gpus == {"x": {"spec": 6.69}}

test("a row FLAGGED for cross-check disagreement is never written by apply_outcomes",
     check_apply_never_touches_a_row_flagged_for_disagreement)


# ===========================================================================
# SOURCE_MAP integrity — the config itself, checked the way main() checks it
# ===========================================================================
print("\nSOURCE_MAP: every slug matches the real catalog, every entry is well-formed")

def check_source_map_matches_real_catalog():
    with open(os.path.join(ROOT, "data", "gpus.json")) as f:
        catalog_slugs = set(json.load(f)["data"])
    assert set(pc.SOURCE_MAP) == catalog_slugs, (
        f"SOURCE_MAP and data/gpus.json have drifted: "
        f"map-only={set(pc.SOURCE_MAP) - catalog_slugs}, catalog-only={catalog_slugs - set(pc.SOURCE_MAP)}")

test("SOURCE_MAP covers exactly the slugs data/gpus.json has, no more and no fewer",
     check_source_map_matches_real_catalog)


def check_source_map_entries_well_formed():
    known_kinds = {"azure", "aws", "lambda", "coreweave", "vast"}
    for slug, tiers in pc.SOURCE_MAP.items():
        for tier, cfg in tiers.items():
            assert tier in ("hyper", "spec", "spot"), f"{slug}: unknown tier {tier!r}"
            is_manual = "manual" in cfg
            is_configured = "primary" in cfg
            assert is_manual != is_configured, (
                f"{slug}/{tier}: must be exactly one of manual or primary-configured")
            if is_manual:
                assert isinstance(cfg["manual"], str) and cfg["manual"], f"{slug}/{tier}: empty manual reason"
            else:
                for spec in [cfg["primary"]] + cfg.get("secondary", []):
                    assert spec["kind"] in known_kinds, f"{slug}/{tier}: unknown kind {spec['kind']!r}"

test("every SOURCE_MAP entry is exactly one of manual-with-a-reason or primary-configured",
     check_source_map_entries_well_formed)


def check_needed_kinds_reflects_the_map():
    kinds = pc._needed_kinds(pc.SOURCE_MAP)
    assert kinds == {"azure", "aws", "lambda", "coreweave", "vast"}, kinds

test("_needed_kinds sees every source kind actually used in SOURCE_MAP", check_needed_kinds_reflects_the_map)


# ===========================================================================
# The real catalog: every priceSource.price matches its own row's price
# ===========================================================================
# fix/cost-provenance's test 5: a later hand-edit to a price without updating
# its provenance must fail. Checked against the committed file, not a fixture,
# because that is the one place a stale price-vs-provenance pair would
# actually ship. Not exact equality: CONFIRMED refreshes provenance without
# rewriting a price that moved less than CONFIRM_THRESHOLD
# (check_apply_writes_the_reading_not_the_proposed_value_as_price above pins
# that priceSource.price is the reading, not an echo of the catalog value),
# so a legitimately-confirmed row can drift from its recorded reading by a
# little and still be exactly what it claims to be.
#
# Cold-check finding: using CONFIRM_THRESHOLD (1%) itself as this test's own
# floor left a gap a hand-edit can hide in. h100-80/hyper -- the one real row
# with any drift at all -- sits at 0.081% (12.30 vs. the recorded 12.29); the
# cold check's sabotage moved it to 12.40 and 12.41, 0.895% and 0.976%, both
# comfortably under a 1% floor and both a hand-edit with no fetch behind it
# at all. A relative floor tied to CONFIRM_THRESHOLD cannot both admit
# whatever a legitimate CONFIRMED run might someday produce (which the
# tool's own design allows up to 1%) and reject an adversarial edit that
# stays just under that same number -- the two are the same shape of gap by
# construction. This halves the tolerance instead of matching it exactly:
# still >6x the one real drift on record, so it does not fail real data, and
# comfortably below both of the cold check's edits, which is what actually
# matters here. It does not close the gap in principle -- a sufficiently
# careful edit stays under any fixed relative floor -- but it closes the
# specific one this run demonstrated, without fabricating precision this
# file does not have about what a future legitimate CONFIRMED drift could
# look like.
PRICE_DRIFT_TOLERANCE = pc.CONFIRM_THRESHOLD / 2
print("\nThe real catalog: every priceSource.price still agrees with its own row's price")

def check_every_real_price_source_matches_its_own_price():
    with open(os.path.join(ROOT, "data", "gpus.json")) as f:
        rows = json.load(f)["data"]
    bad, checked = [], 0
    for slug, row in rows.items():
        for tier, src in row.get("priceSource", {}).items():
            checked += 1
            assert "price" in src, f"{slug}/{tier}: priceSource has no 'price' field"
            price = src["price"]
            catalog = row[tier]
            delta = abs(catalog - price) / price if price else float("inf")
            if delta > PRICE_DRIFT_TOLERANCE + 1e-9:
                bad.append(f"{slug}/{tier}: catalog={catalog} priceSource.price={price} "
                           f"({delta:+.2%}, past the {PRICE_DRIFT_TOLERANCE:.1%} drift floor this "
                           f"test holds real rows to)")
    assert checked > 0, "no row in data/gpus.json carries a priceSource — nothing was actually checked"
    assert not bad, "price moved without its provenance being refreshed:\n       " + "\n       ".join(bad)

test("every sourced tier's catalog price is within the drift floor of its recorded priceSource.price",
     check_every_real_price_source_matches_its_own_price)


def check_every_real_price_source_is_backed_by_an_automated_source():
    """Cold-check finding: nothing tied priceSource's presence to SOURCE_MAP.
    An invented priceSource on rtx6000ada-48/spot (a tier SOURCE_MAP marks
    "manual": no automatable source exists) round-tripped as if a real fetch
    confirmed it, and the price-drift check above passed too, since a
    number invented to equal the catalog value trivially agrees with itself.
    A tier with a priceSource must be one price_check.py's own SOURCE_MAP
    says it can actually fetch — the only way it could have gotten one."""
    with open(os.path.join(ROOT, "data", "gpus.json")) as f:
        rows = json.load(f)["data"]
    bad, checked = [], 0
    for slug, row in rows.items():
        for tier in row.get("priceSource", {}):
            checked += 1
            cfg = pc.SOURCE_MAP.get(slug, {}).get(tier)
            if not cfg or "primary" not in cfg:
                bad.append(f"{slug}/{tier}: carries a priceSource, but SOURCE_MAP has no "
                           f"automated source for it (manual reason: "
                           f"{(cfg or {}).get('manual', '<no SOURCE_MAP entry at all>')!r})")
    assert checked > 0, "no row in data/gpus.json carries a priceSource — nothing was actually checked"
    assert not bad, "priceSource recorded for a tier with no automated source:\n       " + "\n       ".join(bad)

test("every real priceSource is on a tier SOURCE_MAP actually marks automatable",
     check_every_real_price_source_is_backed_by_an_automated_source)


# Which key in a SOURCE_MAP primary spec identifies the thing that was priced.
# A contract, so it is literal — and a kind missing from it fails below rather
# than being skipped, so a new provider cannot slip in unchecked.
SPEC_IDENTIFIER = {"azure": "sku", "aws": "instanceType", "lambda": "plan",
                   "coreweave": "name", "vast": "gpuName"}
PRICE_SOURCE_FIELDS = {"provider", "sku", "region", "date", "price"}


def check_every_real_price_source_agrees_with_the_source_it_names():
    """Round-2 cold check: the fields were required to be non-empty, and
    nothing compared them to the source they claim to come from. So
    h100-80/hyper could say provider "aws" while carrying Azure's SKU, or say
    region "westus2" when SOURCE_MAP reads eastus, or say "gcp" and render a
    raw id no fetch could produce — each of them a provenance line that names
    a source which did not supply the number, which is the whole defect this
    branch exists to remove, reintroduced inside the label meant to fix it.

    Checked against price_check.py's own SOURCE_MAP, which is what a real
    fetch would have used."""
    with open(os.path.join(ROOT, "data", "gpus.json")) as f:
        rows = json.load(f)["data"]
    bad, checked = [], 0
    for slug, row in rows.items():
        for tier, src in row.get("priceSource", {}).items():
            spec = (pc.SOURCE_MAP.get(slug, {}).get(tier) or {}).get("primary")
            if not spec:
                continue                      # the check above already owns this
            checked += 1
            kind = spec["kind"]
            if kind not in SPEC_IDENTIFIER:
                bad.append(f"{slug}/{tier}: SOURCE_MAP kind {kind!r} is not in SPEC_IDENTIFIER, "
                           f"so nothing here can check what it recorded")
                continue
            if src.get("provider") != kind:
                bad.append(f"{slug}/{tier}: provenance says provider {src.get('provider')!r}, "
                           f"but the only source that could have supplied it is {kind!r}")
            ident = spec.get(SPEC_IDENTIFIER[kind])
            if ident and ident not in str(src.get("sku", "")):
                bad.append(f"{slug}/{tier}: recorded sku {src.get('sku')!r} does not name "
                           f"{ident!r}, which is what {kind} was asked for")
            if "region" in spec and src.get("region") != spec["region"]:
                bad.append(f"{slug}/{tier}: recorded region {src.get('region')!r} is not the "
                           f"region read, {spec['region']!r}")
    assert checked > 0, "no row was checked against SOURCE_MAP — the loop matched nothing"
    assert not bad, ("provenance disagrees with the source it names:\n       "
                     + "\n       ".join(bad))

test("every real priceSource names the source SOURCE_MAP says supplied it",
     check_every_real_price_source_agrees_with_the_source_it_names)


def check_every_real_price_source_date_is_a_date_already_past():
    """Round-2 cold check: "the date it was read" was only required to be a
    non-blank string, so 2026-09-32, a 2031 date and "22 Sep 2026" all passed
    and all rendered. A date that is not a date, or has not happened, cannot
    be when something was read."""
    import datetime
    with open(os.path.join(ROOT, "data", "gpus.json")) as f:
        rows = json.load(f)["data"]
    today = datetime.date.today()
    bad, checked = [], 0
    for slug, row in rows.items():
        for tier, src in row.get("priceSource", {}).items():
            checked += 1
            raw = src.get("date")
            try:
                when = datetime.datetime.strptime(str(raw), "%Y-%m-%d").date()
            except (TypeError, ValueError):
                bad.append(f"{slug}/{tier}: date {raw!r} is not YYYY-MM-DD")
                continue
            if when > today:
                bad.append(f"{slug}/{tier}: date {raw} has not happened yet")
    assert checked > 0, "no priceSource date was checked"
    assert not bad, "priceSource carries a date that is not one:\n       " + "\n       ".join(bad)

test("every real priceSource date is a real date that has already happened",
     check_every_real_price_source_date_is_a_date_already_past)


def check_a_price_source_carries_these_fields_and_no_others():
    """Round-2 cold check: an extra key was accepted everywhere. `"estimated":
    true` round-tripped through sync, parity and both renderers, and the tier
    still displayed as a confirmed reading — a claim about the price that
    nothing renders, nothing validates and nothing can act on. The set is
    closed so a field has to be added deliberately, here, with whatever
    renders it."""
    with open(os.path.join(ROOT, "data", "gpus.json")) as f:
        rows = json.load(f)["data"]
    bad, checked = [], 0
    for slug, row in rows.items():
        for tier, src in row.get("priceSource", {}).items():
            checked += 1
            extra = set(src) - PRICE_SOURCE_FIELDS
            missing = PRICE_SOURCE_FIELDS - set(src)
            if extra or missing:
                bad.append(f"{slug}/{tier}: unexpected {sorted(extra)}, missing {sorted(missing)}")
    assert checked > 0, "no priceSource entry was checked"
    assert not bad, ("a priceSource entry is not exactly "
                     f"{sorted(PRICE_SOURCE_FIELDS)}:\n       " + "\n       ".join(bad))

test("a priceSource carries exactly provider, sku, region, date and price",
     check_a_price_source_carries_these_fields_and_no_others)


def check_every_real_price_source_field_is_non_empty():
    """Cold-check finding: a sourced tier with an empty date rendered as
    "... · read " on the page and crashed the PDF generator with a
    KeyError — neither engine's crash (or silent blank) is a real check,
    and a missing key (vs. an empty string) hits a different code path in
    each. Every field is required and required to be a genuinely non-blank
    string, checked directly against the data rather than however each
    renderer happens to fail when it is not."""
    with open(os.path.join(ROOT, "data", "gpus.json")) as f:
        rows = json.load(f)["data"]
    bad, checked = [], 0
    for slug, row in rows.items():
        for tier, src in row.get("priceSource", {}).items():
            checked += 1
            for field in ("provider", "sku", "region", "date"):
                value = src.get(field)
                if not isinstance(value, str) or not value.strip():
                    bad.append(f"{slug}/{tier}.{field}: {value!r} — must be a non-empty string")
    assert checked > 0, "no row in data/gpus.json carries a priceSource — nothing was actually checked"
    assert not bad, "priceSource field missing or blank:\n       " + "\n       ".join(bad)

test("every real priceSource's provider/sku/region/date is a non-empty string",
     check_every_real_price_source_field_is_non_empty)


# ===========================================================================
# priceRecord: a price read by hand, for a tier no automated source reads
# ===========================================================================
print("\npriceRecord: a hand-recorded price names its page, its day, and nothing it cannot back")

PRICE_RECORD_FIELDS = {"provider", "sku", "region", "date", "price", "url"}


def price_record_problems(rows, source_map, today):
    """Every rule a hand-recorded price must satisfy, as one list of problems.

    A hand record is the weaker provenance — a person read the page, nothing
    re-reads it — so it is allowed only where the stronger kind is impossible:
    on a tier whose SOURCE_MAP entry is manual. It never sits beside an
    automated reading, never on a tier with no price, and it carries the URL of
    the provider's own page so a reader can check it, and the price it read so
    a hand-edit of the catalog value cannot hide behind it."""
    import datetime as _dt
    bad = []
    for slug, row in rows.items():
        for tier, rec in (row.get("priceRecord") or {}).items():
            where = f"{slug}/{tier}"
            if tier not in ("hyper", "spec", "spot"):
                bad.append(f"{where}: not a price tier")
                continue
            if not isinstance(rec, dict) or set(rec) != PRICE_RECORD_FIELDS:
                bad.append(f"{where}: fields {sorted(rec) if isinstance(rec, dict) else rec!r}, "
                           f"expected exactly {sorted(PRICE_RECORD_FIELDS)}")
                continue
            for field in ("provider", "sku", "region", "date", "url"):
                if not isinstance(rec[field], str) or not rec[field].strip():
                    bad.append(f"{where}.{field}: {rec[field]!r} — must be a non-empty string")
            if isinstance(rec["url"], str) and not rec["url"].startswith("https://"):
                bad.append(f"{where}.url: {rec['url']!r} — must be the provider's page, over https")
            elif isinstance(rec["url"], str) and urllib.parse.urlparse(rec["url"]).path in ("", "/"):
                # A site's front page is not where a reader can check the price
                # (cold check, round 1: nothing rejected one).
                bad.append(f"{where}.url: {rec['url']!r} names a site, not the page the price was read on")
            try:
                read = _dt.datetime.strptime(str(rec["date"]), "%Y-%m-%d").date()
                if read > today:
                    bad.append(f"{where}.date: {rec['date']} is after today ({today})")
            except ValueError:
                bad.append(f"{where}.date: {rec['date']!r} is not YYYY-MM-DD")
            if row.get(tier) is None:
                bad.append(f"{where}: the tier is null — a hand record cannot price a tier the catalog "
                           "says has no confirmed price")
            elif not isinstance(rec["price"], (int, float)) or isinstance(rec["price"], bool):
                bad.append(f"{where}.price: {rec['price']!r} is not a number")
            elif abs(row[tier] - rec["price"]) > PRICE_DRIFT_TOLERANCE * rec["price"]:
                bad.append(f"{where}: catalog {row[tier]} differs from the recorded {rec['price']} by more "
                           f"than {PRICE_DRIFT_TOLERANCE:.1%}")
            if tier in (row.get("priceSource") or {}):
                bad.append(f"{where}: carries both an automated reading and a hand record")
            cfg = source_map.get(slug, {}).get(tier) or {}
            if "manual" not in cfg:
                bad.append(f"{where}: SOURCE_MAP gives this tier an automated source, so a hand record "
                           "would stand in for a reading the weekly job can make")
    return bad


def check_every_price_record_rule_is_exercised():
    """Each rule, broken on its own by one fixture, must be the one reported —
    and a record that breaks none reports nothing, so no rule passes by being
    unreachable."""
    import datetime as _dt
    today = _dt.date(2026, 9, 23)
    good = {"provider": "RunPod", "sku": "MI300X (Secure Cloud)", "region": "global",
            "date": "2026-09-23", "price": 2.39, "url": "https://www.runpod.io/gpu-models/mi300x"}
    manual = {"x": {"spec": {"manual": "no RunPod reader"}, "hyper": {"primary": {"kind": "azure"}}}}

    def rows_with(rec=good, tier="spec", **row):
        base = {"hyper": 6.0, "spec": 2.39, "spot": 1.11, "priceRecord": {tier: rec}}
        base.update(row)
        return {"x": base}

    assert price_record_problems(rows_with(), manual, today) == [], "a valid record was refused"
    # Each fixture, and the words only its own rule's report carries.
    breaks = {
        "a missing field": (rows_with({k: v for k, v in good.items() if k != "url"}), "expected exactly"),
        "an extra field": (rows_with(dict(good, note="x")), "expected exactly"),
        "a blank provider": (rows_with(dict(good, provider=" ")), ".provider:"),
        "a plain-http url": (rows_with(dict(good, url="http://www.runpod.io/")), "over https"),
        "a front-page url": (rows_with(dict(good, url="https://www.runpod.io/")), "names a site"),
        "a date after today": (rows_with(dict(good, date="2026-09-24")), "after today"),
        "a malformed date": (rows_with(dict(good, date="23/09/2026")), "not YYYY-MM-DD"),
        "a null tier": (rows_with(spec=None), "the tier is null"),
        "a non-numeric price": (rows_with(dict(good, price="2.39")), "is not a number"),
        "a catalog value the record does not back": (rows_with(spec=2.49), "differs from the recorded"),
        "an automated reading beside it": (rows_with(priceSource={"spec": {"provider": "lambda"}}),
                                           "both an automated reading and a hand record"),
        "an automated tier": (rows_with(tier="hyper"), "gives this tier an automated source"),
    }
    for what, (rows, words) in breaks.items():
        found = price_record_problems(rows, manual, today)
        assert any(words in f for f in found), f"{what} was not reported by its own rule: {found}"


test("every priceRecord rule is reached by a fixture that breaks it, and a valid record passes",
     check_every_price_record_rule_is_exercised)


def check_every_real_price_record_follows_the_rules():
    """The real catalog and the real SOURCE_MAP, held to the same rules."""
    import datetime as _dt
    with open(os.path.join(ROOT, "data", "gpus.json")) as f:
        rows = json.load(f)["data"]
    bad = price_record_problems(rows, pc.SOURCE_MAP, _dt.datetime.now(_dt.timezone.utc).date())
    assert not bad, "priceRecord breaks a rule:\n       " + "\n       ".join(bad)
    # A floor, so the rules above cannot pass by having nothing to read.
    held = sum(len(r.get("priceRecord") or {}) for r in rows.values())
    assert held >= 2, f"only {held} real priceRecord(s) — the rules above checked almost nothing"


test("every real priceRecord follows the rules", check_every_real_price_record_follows_the_rules)


PRICE_NOTE_FIELDS = {"reason", "checked"}
# Words the suites' leak checks read as a value that escaped into the text: the
# PDF refuses "none" for a Python None, the page "null", "undefined" and "NaN".
# A note is prose, and a note using one of them would read to those checks as
# a leak (one did, "none in the Azure API", and it was reworded).
NOTE_LEAK_WORDS = re.compile(r"\b(none|null|undefined|nan)\b", re.I)


def price_note_problems(rows, today):
    """Every rule a price note must satisfy, and the one rule over every tier.

    The note is the third provenance kind: why a tier has neither an automated
    reading nor a hand record. That means where its figure came from and why
    no weekly reader covers it, or, on a null tier, why no hourly price
    qualified. Every tier of every row carries exactly one of the three, so no
    figure, and no missing figure, reaches a reader without saying how it got
    there. A note carries no dollar figure, because one beside the tier's own
    would read as a price. It ends as a sentence ends, because both engines
    print "Checked <date>." straight after it."""
    import datetime as _dt
    bad = []
    for slug, row in rows.items():
        notes = row.get("priceNote") or {}
        for tier in notes:
            if tier not in ("hyper", "spec", "spot"):
                bad.append(f"{slug}/{tier}: not a price tier")
        for tier in ("hyper", "spec", "spot"):
            where = f"{slug}/{tier}"
            kinds = [k for k in ("priceSource", "priceRecord", "priceNote") if tier in (row.get(k) or {})]
            if len(kinds) != 1:
                bad.append(f"{where}: carries {kinds or 'no provenance at all'} — every tier carries exactly one "
                           "of priceSource, priceRecord and priceNote")
            note = notes.get(tier)
            if note is None:
                continue
            if not isinstance(note, dict) or set(note) != PRICE_NOTE_FIELDS:
                bad.append(f"{where}: note fields {sorted(note) if isinstance(note, dict) else note!r}, "
                           f"expected exactly {sorted(PRICE_NOTE_FIELDS)}")
                continue
            reason = note["reason"]
            if not isinstance(reason, str) or not reason.strip():
                bad.append(f"{where}.reason: {reason!r} — must be a non-empty sentence")
            else:
                if "$" in reason:
                    bad.append(f"{where}.reason: carries a dollar figure, which beside the tier's own would "
                               "read as a price")
                if not reason.rstrip().endswith("."):
                    bad.append(f"{where}.reason: must end as a sentence ends — both engines print "
                               "'Checked <date>.' straight after it")
                if NOTE_LEAK_WORDS.search(reason):
                    bad.append(f"{where}.reason: uses {NOTE_LEAK_WORDS.search(reason).group(0)!r}, one of the "
                               "words the suites' leak checks read as an escaped value")
            try:
                checked = _dt.datetime.strptime(str(note["checked"]), "%Y-%m-%d").date()
                if checked > today:
                    bad.append(f"{where}.checked: {note['checked']} is after today ({today})")
            except ValueError:
                bad.append(f"{where}.checked: {note['checked']!r} is not YYYY-MM-DD")
    return bad


def check_every_price_note_rule_is_exercised():
    """Each rule, broken on its own by one fixture, must be the one reported,
    and a row that breaks none reports nothing, so no rule passes by being
    unreachable."""
    import datetime as _dt
    today = _dt.date(2026, 9, 23)
    good = {"reason": "No hyperscaler rents this card.", "checked": "2026-09-23"}
    source = {"provider": "aws", "sku": "g6e.xlarge", "region": "US East (N. Virginia)",
              "date": "2026-09-22", "price": 1.86}
    record = {"provider": "RunPod", "sku": "X", "region": "global", "date": "2026-09-23",
              "price": 2.39, "url": "https://www.runpod.io/gpu-models/x"}

    def rows_with(note=good, **row):
        # One of each kind, so the one-per-tier rule holds unless a fixture breaks it.
        base = {"hyper": None, "spec": 2.39, "spot": 1.86, "priceNote": {"hyper": note},
                "priceRecord": {"spec": record}, "priceSource": {"spot": source}}
        base.update(row)
        return {"x": base}

    assert price_note_problems(rows_with(), today) == [], (
        f"a valid note was refused: {price_note_problems(rows_with(), today)}")
    breaks = {
        "a missing field": (rows_with({"reason": good["reason"]}), "expected exactly"),
        "an extra field": (rows_with(dict(good, price=2.0)), "expected exactly"),
        "a blank reason": (rows_with(dict(good, reason="  ")), "non-empty sentence"),
        "a dollar figure": (rows_with(dict(good, reason="Runcrate advertises $0.82/hr.")), "dollar figure"),
        "no closing period": (rows_with(dict(good, reason="No hyperscaler rents this card")), "sentence ends"),
        "a leak word": (rows_with(dict(good, reason="There are none in the Azure API.")), "leak checks"),
        "a date after today": (rows_with(dict(good, checked="2026-09-24")), "after today"),
        "a malformed date": (rows_with(dict(good, checked="23/09/2026")), "not YYYY-MM-DD"),
        "a note beside a reading": (rows_with(priceNote={"hyper": good, "spot": good}), "exactly one"),
        "a tier with nothing": (rows_with(priceSource={}), "no provenance at all"),
        "a note on no tier": (rows_with(priceNote={"hyper": good, "total": good}), "not a price tier"),
    }
    for what, (rows, words) in breaks.items():
        found = price_note_problems(rows, today)
        assert any(words in f for f in found), f"{what} was not reported by its own rule: {found}"

test("every priceNote rule is reached by a fixture that breaks it, and a valid note passes",
     check_every_price_note_rule_is_exercised)


def check_every_real_tier_says_how_its_figure_was_reached():
    """The real catalog, held to the same rules: every tier of every row carries
    exactly one of priceSource, priceRecord and priceNote, and every note
    follows the note's rules."""
    import datetime as _dt
    with open(os.path.join(ROOT, "data", "gpus.json")) as f:
        rows = json.load(f)["data"]
    bad = price_note_problems(rows, _dt.datetime.now(_dt.timezone.utc).date())
    assert not bad, "a tier does not say how its figure was reached:\n       " + "\n       ".join(bad)
    # A floor, so the rules above cannot pass by having nothing to read.
    held = sum(len(r.get("priceNote") or {}) for r in rows.values())
    assert held >= 20, f"only {held} real priceNote(s) — the rules above checked almost nothing"

test("every real tier carries exactly one of priceSource, priceRecord and priceNote, and every note follows the rules",
     check_every_real_tier_says_how_its_figure_was_reached)


PRICE_LEAD_FIELDS = {"provider", "price", "url", "date", "why"}
PRICE_LEAD_OPTIONAL = {"about"}


def price_lead_problems(rows, today):
    """Every rule a lead must satisfy. A lead is an hourly price someone lists for a
    tier that has no confirmed price, and that could not be confirmed: shown, with
    why and where, and used in no figure. Only on a null tier, because a priced
    tier's answer is its price. It carries the page it was read on (https, the
    page itself, not a site's front page), the day it was read, and why it could
    not be confirmed. The why is a sentence with no dollar figure; the lead's own
    price is its price field, printed once."""
    import datetime as _dt
    bad = []
    for slug, row in rows.items():
        for tier, leads in (row.get("priceLead") or {}).items():
            where = f"{slug}/{tier}"
            if tier not in ("hyper", "spec", "spot"):
                bad.append(f"{where}: not a price tier")
                continue
            if row.get(tier) is not None:
                bad.append(f"{where}: the tier has a price — a lead goes only where there is none")
            if not isinstance(leads, list) or not leads:
                bad.append(f"{where}: must be a non-empty list of leads")
                continue
            for i, lead in enumerate(leads):
                at = f"{where}[{i}]"
                if not isinstance(lead, dict) or not (PRICE_LEAD_FIELDS <= set(lead) <= PRICE_LEAD_FIELDS | PRICE_LEAD_OPTIONAL):
                    bad.append(f"{at}: fields {sorted(lead) if isinstance(lead, dict) else lead!r}, expected "
                               f"{sorted(PRICE_LEAD_FIELDS)} and optionally {sorted(PRICE_LEAD_OPTIONAL)}")
                    continue
                if not isinstance(lead["provider"], str) or not lead["provider"].strip():
                    bad.append(f"{at}.provider: {lead['provider']!r} — must be a non-empty name")
                if not isinstance(lead["price"], (int, float)) or isinstance(lead["price"], bool) or lead["price"] <= 0:
                    bad.append(f"{at}.price: {lead['price']!r} — must be a positive number")
                for field in ("url", "about"):
                    if field not in lead:
                        continue
                    url = lead[field]
                    if not isinstance(url, str) or not url.startswith("https://"):
                        bad.append(f"{at}.{field}: {url!r} — must be a page, over https")
                    elif urllib.parse.urlparse(url).path in ("", "/"):
                        bad.append(f"{at}.{field}: {url!r} names a site, not the page")
                try:
                    read = _dt.datetime.strptime(str(lead["date"]), "%Y-%m-%d").date()
                    if read > today:
                        bad.append(f"{at}.date: {lead['date']} is after today ({today})")
                except ValueError:
                    bad.append(f"{at}.date: {lead['date']!r} is not YYYY-MM-DD")
                why = lead["why"]
                if not isinstance(why, str) or not why.strip():
                    bad.append(f"{at}.why: {why!r} — must say why the lead is not used")
                else:
                    if "$" in why:
                        bad.append(f"{at}.why: carries a dollar figure — the lead's price is its price field")
                    if not why.rstrip().endswith("."):
                        bad.append(f"{at}.why: must end as a sentence ends")
                    if NOTE_LEAK_WORDS.search(why):
                        bad.append(f"{at}.why: uses {NOTE_LEAK_WORDS.search(why).group(0)!r}, a word the leak checks read as an escaped value")
    return bad


def check_every_price_lead_rule_is_exercised():
    """Each rule, broken on its own by one fixture, must be the one reported, and a
    valid lead reports nothing."""
    import datetime as _dt
    today = _dt.date(2026, 9, 23)
    good = {"provider": "Runcrate", "price": 0.82, "url": "https://www.runcrate.ai/pricing/gpu/mi210",
            "date": "2026-09-23", "why": "Its own pricing page lists no AMD GPU.",
            "about": "https://github.com/x/y/blob/HEAD/docs/research/runcrate-due-diligence.md"}

    def rows_with(lead=good, tier="spec", **row):
        base = {"hyper": None, "spec": None, "spot": 1.0, "priceLead": {tier: [lead]}}
        base.update(row)
        return {"x": base}

    assert price_lead_problems(rows_with(), today) == [], f"a valid lead was refused: {price_lead_problems(rows_with(), today)}"
    no_about = {k: v for k, v in good.items() if k != "about"}
    assert price_lead_problems(rows_with(no_about), today) == [], "a lead without an about link was refused"
    breaks = {
        "a priced tier": (rows_with(tier="spot"), "the tier has a price"),
        "not a list": (rows_with(priceLead={"spec": good}), "non-empty list"),
        "an empty list": (rows_with(priceLead={"spec": []}), "non-empty list"),
        "a missing field": (rows_with({k: v for k, v in good.items() if k != "url"}), "expected"),
        "an extra field": (rows_with(dict(good, region="global")), "expected"),
        "a blank provider": (rows_with(dict(good, provider=" ")), ".provider:"),
        "a zero price": (rows_with(dict(good, price=0)), "positive number"),
        "a string price": (rows_with(dict(good, price="0.82")), "positive number"),
        "a plain-http url": (rows_with(dict(good, url="http://www.runcrate.ai/pricing/gpu/mi210")), "over https"),
        "a front-page url": (rows_with(dict(good, url="https://www.runcrate.ai/")), "names a site"),
        "a front-page about": (rows_with(dict(good, about="https://github.com/")), "names a site"),
        "a date after today": (rows_with(dict(good, date="2026-09-24")), "after today"),
        "a malformed date": (rows_with(dict(good, date="23/09/2026")), "not YYYY-MM-DD"),
        "a blank why": (rows_with(dict(good, why="")), "why the lead is not used"),
        "a figure in the why": (rows_with(dict(good, why="It says $0.82 only.")), "dollar figure"),
        "no closing period": (rows_with(dict(good, why="Unconfirmed")), "sentence ends"),
        "a leak word": (rows_with(dict(good, why="There is none elsewhere.")), "leak checks"),
        "a lead on no tier": (rows_with(priceLead={"total": [good]}), "not a price tier"),
    }
    for what, (rows, words) in breaks.items():
        found = price_lead_problems(rows, today)
        assert any(words in f for f in found), f"{what} was not reported by its own rule: {found}"

test("every priceLead rule is reached by a fixture that breaks it, and a valid lead passes",
     check_every_price_lead_rule_is_exercised)


def check_every_real_price_lead_follows_the_rules():
    import datetime as _dt
    with open(os.path.join(ROOT, "data", "gpus.json")) as f:
        rows = json.load(f)["data"]
    bad = price_lead_problems(rows, _dt.datetime.now(_dt.timezone.utc).date())
    assert not bad, "priceLead breaks a rule:\n       " + "\n       ".join(bad)
    held = sum(len(ls) for r in rows.values() for ls in (r.get("priceLead") or {}).values())
    assert held >= 3, f"only {held} real lead(s) — the rules above checked almost nothing"

test("every real priceLead follows the rules", check_every_real_price_lead_follows_the_rules)


def check_a_tier_nobody_offers_has_no_price():
    """Tier honesty's sharpest case: a price shown for an offer that doesn't exist.
    Until 2026-09-23 the price map said "no hyperscaler rents this card" for
    rtx4090-24, rtx5090-32 and rtx6000ada-48 while the catalog priced their
    hyperscaler tiers. A re-check against AWS, Azure, Google Cloud and Oracle
    (docs/research/unsourced-prices.md) settled all four. Two rules:
    - a tier whose own note says no hyperscaler rents the card has no price;
    - the four the re-check settled stay as it settled them until a new check
      says otherwise: three null, and rtxpro-96's read from AWS g7e.2xlarge.
    Only the goldens saw a price put back on one of the three, and a golden
    regenerated in the same pull request would have let it through."""
    with open(os.path.join(ROOT, "data", "gpus.json")) as f:
        rows = json.load(f)["data"]
    said = 0
    for slug, row in rows.items():
        for tier, note in (row.get("priceNote") or {}).items():
            if note["reason"].startswith("No hyperscaler rents this card"):
                said += 1
                assert row[tier] is None, (
                    f"{slug}/{tier}: its note says no hyperscaler rents the card, and the tier has a price")
    assert said >= 6, f"only {said} notes say no hyperscaler rents the card; the rule above checked almost nothing"
    for slug in ("rtx4090-24", "rtx5090-32", "rtx6000ada-48"):
        assert rows[slug]["hyper"] is None, f"{slug}: the re-check found no hyperscaler renting it, and it has a price"
    src = (rows["rtxpro-96"].get("priceSource") or {}).get("hyper") or {}
    assert (src.get("provider"), src.get("sku")) == ("aws", "g7e.2xlarge"), (
        f"rtxpro-96's hyperscaler tier is no longer read from AWS g7e.2xlarge: {src}")

test("a tier nobody offers has no price, and the four hyperscaler tiers the re-check settled stay settled",
     check_a_tier_nobody_offers_has_no_price)


# Every tier the catalog holds with no confirmed hourly price. Each one tells a
# reader "no confirmed hourly price" and its note says why. The rule above
# covers the notes that say no hyperscaler rents the card; a price put on any
# other empty tier (the RX 7900 XTX's specialized tier, a spot tier nobody
# offers) passed everything but the goldens (engine_r6_amd_rows C4). A tier
# leaves this set only when a reading or a hand record lands for it, which
# changes what the page tells people, so it has to be changed here as well.
NO_CONFIRMED_PRICE = {
    "rtx4090-24": ["hyper"], "rtx5090-32": ["hyper"], "rtx6000ada-48": ["hyper"],
    "rx7900xtx-24": ["hyper", "spec", "spot"], "mi210-64": ["hyper", "spec", "spot"],
    "mi250x-128": ["hyper", "spec", "spot"], "mi325x-256": ["hyper", "spot"],
}


def check_the_tiers_with_no_confirmed_price_are_the_pinned_ones():
    with open(os.path.join(ROOT, "data", "gpus.json")) as f:
        rows = json.load(f)["data"]
    empty = {(slug, t) for slug, row in rows.items() for t in ("hyper", "spec", "spot") if row[t] is None}
    pinned = {(slug, t) for slug, tiers in NO_CONFIRMED_PRICE.items() for t in tiers}
    priced = sorted(f"{slug}/{t}" for slug, t in pinned - empty)
    unpinned = sorted(f"{slug}/{t}" for slug, t in empty - pinned)
    assert not priced and not unpinned, (
        f"priced, though pinned as having no confirmed price: {priced or 'none'}; "
        f"empty, though not pinned: {unpinned or 'none'}")

test("the tiers with no confirmed price are exactly the ones pinned here",
     check_the_tiers_with_no_confirmed_price_are_the_pinned_ones)


def check_no_two_tiers_of_a_card_name_the_same_source():
    """Two tiers of one card read off the same provider and SKU would print one
    source name beside two prices, and a reader could not tell them apart. The
    MI300X's Azure spot tier is the same VM as its hyperscaler tier, told apart
    only by the "(Spot)" the Azure reader adds; with the marker taken out of the
    catalog, only the goldens noticed (engine_r6_amd_rows C12). Every row, and
    readings and hand records alike."""
    with open(os.path.join(ROOT, "data", "gpus.json")) as f:
        rows = json.load(f)["data"]
    clashes, shared = [], 0
    for slug, row in rows.items():
        seen, providers = {}, set()
        for field in ("priceSource", "priceRecord"):
            for tier, src in (row.get(field) or {}).items():
                key = (src["provider"].strip().lower(), src["sku"].strip().lower())
                if key in seen:
                    clashes.append(f"{slug}: {seen[key]} and {tier} both name {src['provider']} {src['sku']!r}")
                seen[key] = tier
                shared += key[0] in providers
                providers.add(key[0])
    assert not clashes, "; ".join(clashes)
    assert shared, "no card has two tiers from one provider, so this checked nothing"

test("no two tiers of a card name the same provider and SKU", check_no_two_tiers_of_a_card_name_the_same_source)


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
