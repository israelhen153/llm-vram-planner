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

The weekly price job rewrites a reading's date and price whenever it re-reads the
tier, and drops a tier's note when it confirms a reading there. A sabotage that
quotes one of them stops applying at the next refresh, so none is quoted here:
they are read from data/gpus.json as this driver loads. What the job never
rewrites, a reading's provider, SKU and region, is quoted.
"""
import json, os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver, Missing, ROOT
from anchors import GPUS_JSON

with open(os.path.join(ROOT, GPUS_JSON), encoding="utf-8") as f:
    H100 = json.load(f)["data"].get("h100-80") or {}
# h100-80's hyperscaler reading, which every sabotage here but C7 attacks, and its
# date and price as the catalog writes them. Without a reading they are never used:
# reading_edit() refuses first. Every edit to the reading goes through it: four that
# quoted it directly were refused by count, not by name, once it was taken out
# (tests/corpus.test.py, the gone refresh).
READING = (H100.get("priceSource") or {}).get("hyper")
DATE, PRICE = (json.dumps(READING["date"]), json.dumps(READING["price"])) if READING else ("", "")
# The reading up to its date, and up to its price.
AT_DATE = '"sku": "Standard_ND96isr_H100_v5", "region": "eastus", "date": '
AT_PRICE = '"region": "eastus", "date": ' + DATE + ', "price": '


def reading_edit(old, new):
    """An edit to h100-80/hyper's reading. With the reading gone there is nothing to
    edit, and the sabotage refuses by name when it is applied, rather than vanish."""
    if READING is None:
        return (GPUS_JSON, Missing("h100-80 carries no hyperscaler reading (priceSource.hyper) to edit"), "", 1)
    return (GPUS_JSON, old, new, 1)


def invented_spot_reading():
    """C7's edit: an invented Vast.ai reading on h100-80's spot tier, in place of the note
    saying why the tier has none, as a reading landed without being confirmed would. Kept
    beside the note, the tier would break the exactly-one rule instead, which is another
    sabotage. Its price is the tier's own, so the invention is all that is wrong.

    The note is read, not quoted, so rewording it while the tier is still held leaves C7
    applying. Once a reading is confirmed there the note is gone, the tier is no longer the
    unconfirmed one C7 attacks, and C7 refuses by name when it is applied."""
    note = (H100.get("priceNote") or {}).get("spot")
    if not note:
        return (GPUS_JSON, Missing("h100-80/spot has no note saying it is unconfirmed (priceNote.spot), "
                                   "so there is no unconfirmed tier to invent a reading for",
                                   note=("h100-80", "spot")), "", 1)
    held = ('{ "spot": { "reason": ' + json.dumps(note["reason"]) + ', "checked": '
            + json.dumps(note["checked"]) + " } }")
    invented = ('{ "provider": "vast", "sku": "H100 SXM (median of 9 verified offers)", "region": "global", '
                '"date": "2026-09-22", "price": ' + json.dumps(H100["spot"]) + " }")
    # The brace that closes the readings, the note, and the brace that closes the row
    # become the invented reading, the readings' brace and the row's.
    return (GPUS_JSON, ' }, "priceNote": ' + held + " }", ', "spot": ' + invented + " } }", 1)


S = {
    "C1 data: h100-80/hyper provider 'azure' -> 'aws' while the SKU stays Azure's (wrong attribution)":
      [
        reading_edit('"provider": "azure", "sku": "Standard_ND96isr_H100_v5"', '"provider": "aws", "sku": "Standard_ND96isr_H100_v5"'),
      ],
    'C10 data (control): h100-80/hyper priceSource.price is a string':
      [
        reading_edit(AT_PRICE + PRICE + " }", AT_PRICE + json.dumps(PRICE) + " }"),
      ],
    "C2 data: h100-80/hyper region 'eastus' -> 'westus2' (SOURCE_MAP's primary says eastus)":
      [
        reading_edit('"sku": "Standard_ND96isr_H100_v5", "region": "eastus"', '"sku": "Standard_ND96isr_H100_v5", "region": "westus2"'),
      ],
    "C3a data: h100-80/hyper date is not a date ('2026-09-32')":
      [
        reading_edit(AT_DATE + DATE, AT_DATE + '"2026-09-32"'),
      ],
    "C3b data: h100-80/hyper date is in the future ('2031-01-01')":
      [
        reading_edit(AT_DATE + DATE, AT_DATE + '"2031-01-01"'),
      ],
    "C3c data: h100-80/hyper date is not ISO ('22 Sep 2026')":
      [
        reading_edit(AT_DATE + DATE, AT_DATE + '"22 Sep 2026"'),
      ],
    "C5 data: an extra field on a sourced tier ('estimated': true) that no renderer honours":
      [
        reading_edit(AT_PRICE + PRICE + " }", AT_PRICE + PRICE + ', "estimated": true }'),
      ],
    "C6 data: h100-80/hyper provider is 'gcp', a kind SOURCE_MAP never fetches (label prints the raw id)":
      [
        reading_edit('"provider": "azure", "sku": "Standard_ND96isr_H100_v5"', '"provider": "gcp", "sku": "Standard_ND96isr_H100_v5"'),
      ],
    'C7 data: an invented spot priceSource for h100-80 (a tier SOURCE_MAP marks automatable, never confirmed)':
      [
        invented_spot_reading(),
      ],
    'C8 data (control): h100-80/hyper sku is whitespace':
      [
        reading_edit('"provider": "azure", "sku": "Standard_ND96isr_H100_v5"', '"provider": "azure", "sku": "   "'),
      ],
    'C9 data (control): h100-80/hyper date is an integer':
      [
        reading_edit(AT_DATE + DATE, AT_DATE + "20260922"),
      ],
}


if __name__ == "__main__":
    run_driver(S)
