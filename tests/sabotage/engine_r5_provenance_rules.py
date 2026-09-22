#!/usr/bin/env python3
"""Round 2 against fix/cost-provenance's own cost-surface tests.

Every one of these passed those tests once both goldens were regenerated —
which is what the price workflow tells a reviewer to do whenever a price
moves, so the golden was no protection on exactly the runs that change prices.

They are one defect in fourteen costumes. The sweep matched a provider NAME as
its marker and asked whether the expected label was somewhere in the text. So
deleting a label left no marker and the tier was skipped entirely; swapping two
tiers' labels left both strings present; and a composite appended after a
correct label was still "somewhere in the text". Its vocabulary was five
provider names matched case-sensitively, so "(aws/azure)", "& AWS", " and GCP",
a newline-joined list, and three providers nobody has heard of all walked past
— and the notes, which carry no price, were never looked at.

What replaced it anchors on each tier's own name and reads the segment after
it, needing no vocabulary at all; and the one sentence the notes may say about
price is a literal.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import INDEX_HTML, REPORT_PY

S = {
    "A1 js: renderCost's spot sub-label falls back to a provider the sweep does not know ('RunPod marketplace')":
      [
        (INDEX_HTML, '<span style="font-size:11px;color:var(--text-muted)">${priceSourceLabel(state, \'spot\')}</span>', '<span style="font-size:11px;color:var(--text-muted)">${state.priceSource && state.priceSource.spot ? priceSourceLabel(state, \'spot\') : \'RunPod marketplace\'}</span>', 1),
      ],
    "A2 js: composite joined by ' and ' appended after the correct hyper label (comparison card)":
      [
        (INDEX_HTML, "Hyper: ${priceSourceLabel(s, 'hyper')}<br>Spec: ${priceSourceLabel(s, 'spec')}<br>Spot: ${priceSourceLabel(s, 'spot')}", "Hyper: ${priceSourceLabel(s, 'hyper')}${s.priceSource && s.priceSource.hyper ? ' (AWS and GCP and Azure)' : ''}<br>Spec: ${priceSourceLabel(s, 'spec')}<br>Spot: ${priceSourceLabel(s, 'spot')}", 1),
      ],
    "A3 js: renderCost's hyper sub-label carries a composite of providers the regex does not know ('Google Cloud, Oracle, Nebius on-demand')":
      [
        (INDEX_HTML, '<span style="font-size:11px;color:var(--text-muted)">${priceSourceLabel(state, \'hyper\')}</span>', '<span style="font-size:11px;color:var(--text-muted)">${priceSourceLabel(state, \'hyper\')} — Google Cloud, Oracle, Nebius on-demand</span>', 1),
      ],
    'A4 js: renderCost drops the spot sub-label entirely (an empty field where the source should be)':
      [
        (INDEX_HTML, '<span style="font-size:11px;color:var(--text-muted)">${priceSourceLabel(state, \'spot\')}</span>', '<span style="font-size:11px;color:var(--text-muted)"></span>', 1),
      ],
    "A5 js: renderNotes claims 'Spot prices are from Vast.ai' for every card (a surface with no $ figure)":
      [
        (INDEX_HTML, "Reserved typically 30-60% off on-demand. ';\n", "Reserved typically 30-60% off on-demand. ';\n  notes += 'Spot prices are from Vast.ai. ';\n", 1),
      ],
    "A6 py: the PDF notes claim 'Spot prices are from Vast.ai' (not a cost-bearing string)":
      [
        (REPORT_PY, '        for n in notes:\n            story.append(Paragraph(f"• {n}", self.styles["Small"]))\n', '        notes.append("Spot prices are from Vast.ai.")\n        for n in notes:\n            story.append(Paragraph(f"• {n}", self.styles["Small"]))\n', 1),
      ],
    "A7 py: the PDF notes' composite comes back joined by ' and ' ('hyperscaler (AWS and GCP and Azure)')":
      [
        (REPORT_PY, '        notes.append("GPU prices are mid-2026 per-board/hr figures across 3 tiers: hyperscaler, "\n', '        notes.append("GPU prices are mid-2026 per-board/hr figures across 3 tiers: hyperscaler (AWS and GCP and Azure), "\n', 1),
      ],
    "A8 js: exec summary appends '& AWS' to a sourced hyper label":
      [
        (INDEX_HTML, "Hyper: ${priceSourceLabel(state, 'hyper')} · Spec: ${priceSourceLabel(state, 'spec')} · Spot: ${priceSourceLabel(state, 'spot')}", "Hyper: ${priceSourceLabel(state, 'hyper')}${state.priceSource && state.priceSource.hyper ? ' & AWS' : ''} · Spec: ${priceSourceLabel(state, 'spec')} · Spot: ${priceSourceLabel(state, 'spot')}", 1),
      ],
    'A9 js: composite joined by <br> (newline) under a sourced spec label on the comparison card':
      [
        (INDEX_HTML, "Hyper: ${priceSourceLabel(s, 'hyper')}<br>Spec: ${priceSourceLabel(s, 'spec')}<br>Spot: ${priceSourceLabel(s, 'spot')}", "Hyper: ${priceSourceLabel(s, 'hyper')}<br>Spec: ${priceSourceLabel(s, 'spec')}${s.priceSource && s.priceSource.spec ? '<br>also: Lambda<br>CoreWeave<br>RunPod' : ''}<br>Spot: ${priceSourceLabel(s, 'spot')}", 1),
      ],
    "A10 js: composite joined by the label's own ' · ' separator ('... read 2026-09-22 · AWS · GCP')":
      [
        (INDEX_HTML, '<span style="font-size:11px;color:var(--text-muted)">${priceSourceLabel(state, \'hyper\')}</span>', '<span style="font-size:11px;color:var(--text-muted)">${priceSourceLabel(state, \'hyper\')}${state.priceSource && state.priceSource.hyper ? \' · AWS · GCP\' : \'\'}</span>', 1),
      ],
    "A11 js: renderCost shows the hyper and spec labels on each other's rows (a Lambda SKU under the Azure price)":
      [
        (INDEX_HTML, '<span style="font-size:11px;color:var(--text-muted)">${priceSourceLabel(state, \'hyper\')}</span>', '<span style="font-size:11px;color:var(--text-muted)">${priceSourceLabel(state, \'__TMP__\')}</span>', 1),
        (INDEX_HTML, '<span style="font-size:11px;color:var(--text-muted)">${priceSourceLabel(state, \'spec\')}</span>', '<span style="font-size:11px;color:var(--text-muted)">${priceSourceLabel(state, \'hyper\')}</span>', 1),
        (INDEX_HTML, "priceSourceLabel(state, '__TMP__')", "priceSourceLabel(state, 'spec')", 1),
      ],
    'A12 py: the PDF Source line swaps the hyper and spec labels':
      [
        (REPORT_PY, '            f"Source — Hyperscaler: {price_source_label(gpu, \'hyper\')}. "\n', '            f"Source — Hyperscaler: {price_source_label(gpu, \'__TMP__\')}. "\n', 1),
        (REPORT_PY, '            f"Specialized: {price_source_label(gpu, \'spec\')}. "\n', '            f"Specialized: {price_source_label(gpu, \'hyper\')}. "\n', 1),
        (REPORT_PY, "price_source_label(gpu, '__TMP__')", "price_source_label(gpu, 'spec')", 1),
      ],
    "A13 js: a lowercase composite '(aws/azure)' after the hyper label — the joined-provider regex is case-sensitive":
      [
        (INDEX_HTML, '<span style="font-size:11px;color:var(--text-muted)">${priceSourceLabel(state, \'hyper\')}</span>', '<span style="font-size:11px;color:var(--text-muted)">${priceSourceLabel(state, \'hyper\')}${state.priceSource && state.priceSource.hyper ? \' (aws/azure)\' : \'\'}</span>', 1),
      ],
    "A14 py: the PDF notes' composite comes back lowercase ('hyperscaler (aws/gcp/azure)')":
      [
        (REPORT_PY, '        notes.append("GPU prices are mid-2026 per-board/hr figures across 3 tiers: hyperscaler, "\n', '        notes.append("GPU prices are mid-2026 per-board/hr figures across 3 tiers: hyperscaler (aws/gcp/azure), "\n', 1),
      ],
}


if __name__ == "__main__":
    run_driver(S)
