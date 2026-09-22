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
import sys

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


print(f"\n{pass_ct} passed, {fail_ct} failed\n")
sys.exit(1 if fail_ct else 0)
