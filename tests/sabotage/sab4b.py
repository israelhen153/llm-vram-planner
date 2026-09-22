#!/usr/bin/env python3
"""Leaks conditioned on state axes the no-constants probes never vary, and a
figure in a reportlab attribute the story spy never harvests."""
import os, sys
sys.dont_write_bytecode = True   # see the note in sab.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sab
from harness import run_driver
from sab import IDX, GR, JS_TP_TILE, JS_BADGE, PY_EXPLAIN, PY_NOTMOD_ROW

EST = "Math.round(computed.deviceBandwidth * 0.7 / (state.params * state.bytesPerParam))"
FP8KV_BADGE = "  if (state.kvBytesPerValue < 2) html += '<span class=\"badge\" style=\"background:var(--success-bg);color:var(--success-text)\">FP8 KV cache</span>';\n"

S = {}
S["R1 js: throughput tile prints '~<borrowed> tok/s per user' only for single-device boards (every catalog card)"] = [
    (IDX, JS_TP_TILE, JS_TP_TILE + f"${{state.gpuDevices === 1 ? '<p class=\"sub\">~' + {EST} + ' tok/s per user (rough)</p>' : ''}}", 1)]
S["R2 js: the same leak only when the card has an FP8 KV cache selected"] = [
    (IDX, JS_TP_TILE, JS_TP_TILE + f"${{state.kvBytesPerValue < 2 ? '<p class=\"sub\">~' + {EST} + ' tok/s per user (rough)</p>' : ''}}", 1)]
S["R3 js: the same leak only for MLA or SWA attention"] = [
    (IDX, JS_TP_TILE, JS_TP_TILE + f"${{state.attnMode !== 'standard' ? '<p class=\"sub\">~' + {EST} + ' tok/s per user (rough)</p>' : ''}}", 1)]
S["R4 js: the same leak only when the state carries a catalog slug (gpuKey)"] = [
    (IDX, JS_TP_TILE, JS_TP_TILE + f"${{state.gpuKey ? '<p class=\"sub\">~' + {EST} + ' tok/s per user (rough)</p>' : ''}}", 1)]
S["R5 js: the 'FP8 KV cache' badge withheld for a card without constants"] = [
    (IDX, FP8KV_BADGE, FP8KV_BADGE.replace("if (state.kvBytesPerValue < 2)", "if (state.kvBytesPerValue < 2 && computed.throughputModelled)"), 1)]
S["R6 js: the comparison card withholds the 'KV dtype' row for a single-device card without constants"] = [
    (IDX, "    html += `<div class=\"row\"><span class=\"label\">KV dtype</span><span class=\"val\">${s.kvBytesPerValue < 2 ? 'FP8' : 'BF16'}</span></div>`;\n",
     "    if (c.throughputModelled || (s.gpuDevices || 1) > 1) html += `<div class=\"row\"><span class=\"label\">KV dtype</span><span class=\"val\">${s.kvBytesPerValue < 2 ? 'FP8' : 'BF16'}</span></div>`;\n", 1)]
S["Q11 py: PDF prints '~<n> tokens/sec' only for single-device boards"] = [
    (GR, PY_EXPLAIN, PY_EXPLAIN + "            if (gpu.get(\"devices\", 1) or 1) == 1:\n                story.append(Paragraph(f\"Rough decode estimate: ~{c['single_tok'] or 0} tokens/sec per user\", self.styles[\"Small\"]))\n", 1)]
S["Q12 py: the figure rides as the paragraph's bulletText (rendered, never harvested)"] = [
    (GR, "                \"depend on them.\",\n                self.styles[\"Small\"]\n            ))\n", "                \"depend on them.\",\n                self.styles[\"Small\"], bulletText=f\"~{c['single_tok'] or 0} tok/s\"\n            ))\n", 1)]
S["Q13 py: PDF prints the figure only when an FP8 KV cache is selected"] = [
    (GR, PY_EXPLAIN, PY_EXPLAIN + "            if cfg.get(\"kv_bpp\", 2) < 2:\n                story.append(Paragraph(f\"Rough decode estimate: ~{c['single_tok'] or 0} tokens/sec per user\", self.styles[\"Small\"]))\n", 1)]
S["Q14 py: PDF prints the figure only for MLA or SWA attention"] = [
    (GR, PY_EXPLAIN, PY_EXPLAIN + "            if cfg.get(\"attn\", \"standard\") != \"standard\":\n                story.append(Paragraph(f\"Rough decode estimate: ~{c['single_tok'] or 0} tokens/sec per user\", self.styles[\"Small\"]))\n", 1)]
S["Q15 py: the PDF's max-context row withheld for a single-device card without constants"] = [
    (GR, "            [\"Max context (1 user, 90% util)\", fmt_k(c[\"max_ctx_1\"]) + \" tokens\"],\n",
     "            [\"Max context (1 user, 90% util)\", fmt_k(c[\"max_ctx_1\"]) + \" tokens\" if c[\"throughput_modelled\"] or (gpu.get(\"devices\", 1) or 1) > 1 else \"n/a\"],\n", 1)]

if __name__ == "__main__":
    run_driver(S)
