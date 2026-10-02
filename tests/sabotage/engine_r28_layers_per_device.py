#!/usr/bin/env python3
"""Round 28: the per-device layer count (fix/remove-layers-per-device).

The page printed a "Layers per device" tile, ~ceil(layers / devices), from five
devices up, and an L<start>–<end> range on each card from two to four. Dividing
layers across devices is pipeline parallelism, and neither engine emits
--pipeline-parallel-size: under TP x DP every device holds a slice of every layer.
Both were removed. The tests that hold every surface to it sweep every catalog row
and the synthetic dual-GCD board, 2 to 9 devices and each power of two up to the
slider's maximum, a dense and an MoE preset, and NVLink asked for and not.

Each sabotage puts the figure back: as it was; gated on one axis the sweep varies
(a dual-GCD row, PCIe, more than one NVLink domain); under another name; or on a
surface it never had (the copied report, a comparison card, a strategy badge, the
PDF). Each must be caught by those tests, not only by a golden.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import REPORT_PY

PAGE = "index.html"

# The condensed panel's tile the removed one sat in front of.
PANEL_NEXT = "    /* The per-field heuristic label. The slider runs to 128 rather than\n"
# Each individual card's detail line, which ended in the range.
CARD_DETAIL = ("      html += `<p class=\"gpu-detail\">${formatGB(perGPU.total)} / "
               "${capacityLabel(computed.deviceGB)} (${Math.min(usagePct, 999)}%)</p></div>`;\n")
CARD_RANGE = ("${computed.deviceCount > 1 ? ' · L' + Math.floor(state.layers / computed.deviceCount * i)"
              " + '–' + (Math.floor(state.layers / computed.deviceCount * (i + 1)) - 1) : ''}")
REPORT_LAYERS = ("  report += `- Layers: ${state.layers}, KV heads: ${state.kvHeads}, "
                 "head dim: ${state.headDim}\\n`;\n")
COMPARE_MAX_CTX = ('    html += `<div class="row"><span class="label">Max ctx (1u)</span>'
                   '<span class="val">${formatContext(c.maxContextSingleUser)}</span></div>`;\n')
BADGE_FP8_KV = ("  if (state.kvBytesPerValue < 2) html += '<span class=\"badge\" style=\"background:"
                "var(--success-bg);color:var(--success-text)\">FP8 KV cache</span>';\n")
PDF_LAYERS = '            ["Layers", str(cfg["layers"])],\n'


def tile(label="Layers per device", gate=""):
    """The removed tile, verbatim but for its label, behind an optional condition."""
    return ("    " + (f"if ({gate}) " if gate else "") +
            '''html += `<div style="font-size:12px"><span style="color:var(--text-muted)">''' + label +
            '''</span><br><b>~${Math.ceil(state.layers / computed.deviceCount)}</b> of ${state.layers}'''
            '''${dp > 1 ? '<br><span style="font-size:10px;color:var(--text-muted)">assumes sharding '''
            '''across all ' + computed.deviceCount + '</span>' : ''}</div>`;\n''')


S = {
    # ---- as it was ----
    "T1 the condensed panel's \"Layers per device\" tile is back": [
        (PAGE, PANEL_NEXT, tile() + PANEL_NEXT, 1)],
    "T2 each individual card's L<start>–<end> range is back": [
        (PAGE, CARD_DETAIL, CARD_DETAIL.replace("%)</p></div>", "%)" + CARD_RANGE + "</p></div>"), 1)],
    # ---- gated on one axis the sweep varies ----
    "T3 the tile is back on a dual-GCD row only (devices: 2)": [
        (PAGE, PANEL_NEXT, tile(gate="state.gpuDevices === 2") + PANEL_NEXT, 1)],
    "T4 the tile is back on PCIe only": [
        (PAGE, PANEL_NEXT, tile(gate="fabric === 'PCIe'") + PANEL_NEXT, 1)],
    "T5 the tile is back above one NVLink domain only (DP > 1)": [
        (PAGE, PANEL_NEXT, tile(gate="dp > 1") + PANEL_NEXT, 1)],
    # ---- under another name ----
    "T6 the tile is back, renamed \"Layer split\"": [
        (PAGE, PANEL_NEXT, tile(label="Layer split") + PANEL_NEXT, 1)],
    # ---- on a surface it never had ----
    "M1 the figure moves into the copied report": [
        (PAGE, REPORT_LAYERS, REPORT_LAYERS.replace(
            "- Layers: ${state.layers},",
            "- Layers: ${state.layers} (~${Math.ceil(state.layers / computed.deviceCount)} per device),"), 1)],
    "M2 the figure moves into a comparison card": [
        (PAGE, COMPARE_MAX_CTX,
         '    html += `<div class="row"><span class="label">Layers/device</span>'
         '<span class="val">~${Math.ceil(s.layers / c.deviceCount)}</span></div>`;\n' + COMPARE_MAX_CTX, 1)],
    "M3 the figure moves into a strategy badge": [
        (PAGE, BADGE_FP8_KV,
         "  if (computed.deviceCount > 1) html += `<span class=\"badge\">~"
         "${Math.ceil(state.layers / computed.deviceCount)} layers each</span>`;\n" + BADGE_FP8_KV, 1)],
    "M4 the figure moves into the PDF's model table": [
        (REPORT_PY, PDF_LAYERS,
         '            ["Layers", f\'{cfg["layers"]} (~{-(-cfg["layers"] // '
         '(cfg["n_gpu"] * cfg["gpu"].get("devices", 1)))} per device)\'],\n', 1)],
}

if __name__ == "__main__":
    run_driver(S)
