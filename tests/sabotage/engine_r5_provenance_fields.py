#!/usr/bin/env python3
"""Round 2 against what a recorded provenance is allowed to say.

The fields were required to be present and non-blank, and nothing compared
them to the source they name. So a row could say provider "aws" while carrying
Azure's SKU, name a region it was not read in, say "gcp" and render a raw id no
fetch could produce, carry 2026-09-32 or a date in 2031, or hold an extra key
that nothing renders and nothing validates. Every one of them rendered, and
every one of them survived the round-1 tests.

These are all catalog edits, and until harness.run_driver re-generated the
blocks after one, they could not honestly go in a corpus at all: editing
data/gpus.json without re-running tools/sync_data.py turns sync.test.py red
because the generated blocks no longer match the file, which would have
recorded every one of these as caught whether or not anything looked at it.

C4 — a price edited by less than the drift floor — is deliberately absent. It
is a documented residual, not a defect; see tests/price_check.test.py.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import GPUS_JSON

S = {
    "C1 data: h100-80/hyper provider 'azure' -> 'aws' while the SKU stays Azure's (wrong attribution)":
      [
        (GPUS_JSON, '"provider": "azure", "sku": "Standard_ND96isr_H100_v5"', '"provider": "aws", "sku": "Standard_ND96isr_H100_v5"', 1),
      ],
    'C10 data (control): h100-80/hyper priceSource.price is a string':
      [
        (GPUS_JSON, '"region": "eastus", "date": "2026-09-22", "price": 12.29 }', '"region": "eastus", "date": "2026-09-22", "price": "12.29" }', 1),
      ],
    "C2 data: h100-80/hyper region 'eastus' -> 'westus2' (SOURCE_MAP's primary says eastus)":
      [
        (GPUS_JSON, '"sku": "Standard_ND96isr_H100_v5", "region": "eastus"', '"sku": "Standard_ND96isr_H100_v5", "region": "westus2"', 1),
      ],
    "C3a data: h100-80/hyper date is not a date ('2026-09-32')":
      [
        (GPUS_JSON, '"sku": "Standard_ND96isr_H100_v5", "region": "eastus", "date": "2026-09-22"', '"sku": "Standard_ND96isr_H100_v5", "region": "eastus", "date": "2026-09-32"', 1),
      ],
    "C3b data: h100-80/hyper date is in the future ('2031-01-01')":
      [
        (GPUS_JSON, '"sku": "Standard_ND96isr_H100_v5", "region": "eastus", "date": "2026-09-22"', '"sku": "Standard_ND96isr_H100_v5", "region": "eastus", "date": "2031-01-01"', 1),
      ],
    "C3c data: h100-80/hyper date is not ISO ('22 Sep 2026')":
      [
        (GPUS_JSON, '"sku": "Standard_ND96isr_H100_v5", "region": "eastus", "date": "2026-09-22"', '"sku": "Standard_ND96isr_H100_v5", "region": "eastus", "date": "22 Sep 2026"', 1),
      ],
    "C5 data: an extra field on a sourced tier ('estimated': true) that no renderer honours":
      [
        (GPUS_JSON, '"region": "eastus", "date": "2026-09-22", "price": 12.29 }', '"region": "eastus", "date": "2026-09-22", "price": 12.29, "estimated": true }', 1),
      ],
    "C6 data: h100-80/hyper provider is 'gcp', a kind SOURCE_MAP never fetches (label prints the raw id)":
      [
        (GPUS_JSON, '"provider": "azure", "sku": "Standard_ND96isr_H100_v5"', '"provider": "gcp", "sku": "Standard_ND96isr_H100_v5"', 1),
      ],
    'C7 data: an invented spot priceSource for h100-80 (a tier SOURCE_MAP marks automatable, never confirmed)':
      [
        # Since feat/rocm-guidance the tier carries a note saying why it has no source, so the
        # invented source replaces the note, as a reading landed without being confirmed would.
        (GPUS_JSON, '"spec": { "provider": "lambda", "sku": "NVIDIA H100 SXM (80 GB, 208 vCPU tier)", "region": "global", "date": "2026-09-22", "price": 3.99 } }, "priceNote": { "spot": { "reason": "In the catalog since 2026-05-28 as a mid-2026 estimate, with no provider recorded. The weekly Vast.ai read on 2026-09-22 came back 42.5% higher, past the 40% a person has to approve, and is held for a second read.", "checked": "2026-09-23" } } },', '"spec": { "provider": "lambda", "sku": "NVIDIA H100 SXM (80 GB, 208 vCPU tier)", "region": "global", "date": "2026-09-22", "price": 3.99 }, "spot": { "provider": "vast", "sku": "H100 SXM (median of 9 verified offers)", "region": "global", "date": "2026-09-22", "price": 2.25 } } },', 1),
      ],
    'C8 data (control): h100-80/hyper sku is whitespace':
      [
        (GPUS_JSON, '"provider": "azure", "sku": "Standard_ND96isr_H100_v5"', '"provider": "azure", "sku": "   "', 1),
      ],
    'C9 data (control): h100-80/hyper date is an integer':
      [
        (GPUS_JSON, '"sku": "Standard_ND96isr_H100_v5", "region": "eastus", "date": "2026-09-22"', '"sku": "Standard_ND96isr_H100_v5", "region": "eastus", "date": 20260922', 1),
      ],
}


if __name__ == "__main__":
    run_driver(S)
