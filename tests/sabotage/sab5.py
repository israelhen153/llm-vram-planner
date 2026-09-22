#!/usr/bin/env python3
"""Round 3: attacks on the assumptions behind the 5f8e581 rules."""
import os, sys
sys.dont_write_bytecode = True   # see the note in sab.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sab
from harness import run_driver
from sab import (INDEX_HTML, REPORT_PY, JS_TP_TILE, JS_TP_RETURN, JS_EXEC_UNKNOWN, JS_CMP_UNKNOWN, JS_NOTES_UNKNOWN,
                 JS_BADGE, JS_NOTES_PCIE, JS_EXEC_DP, JS_QUEUE, PY_EXPLAIN, PY_NOTES_PCIE, PY_DP_NOTE)

JS_DECODE_ESTIMATE = "Math.round(computed.deviceBandwidth * 0.7 / (state.params * state.bytesPerParam))"
PY_DECODE_ESTIMATE = "round(c['device_bw'] * 0.7 / (cfg['params'] * cfg['bpp']))"
SPEED_HINT_MARKUP = '<div id="throughput-output"></div>'
SLIDER = "  document.getElementById('gpu-count-display').textContent = state.gpuCount;\n"
SNAP = "  savedSnapshots.push({ name, state, computed, timestamp: Date.now() });\n"
COST_SET = "  setHTML('cost-output', html);\n"
AGG_LABEL = "<p class=\"label\">Aggregate ceiling @ ${computed.effectiveBatch} concurrent</p>"
KNOWN_TAIL = "${fabricNote}${fp8Note}</p></div>${maxBatchCard}</div>`;\n"
PY_NOTE_PARAMS = "        notes.append(\"Parameter estimates from presets are approximate. Verify against the model's config.json.\")\n"
PY_BUILD = "        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)\n"
PY_IMPORT = "        HRFlowable, KeepTogether, PageBreak\n"
PY_KVTABLE = "    def _make_kv_table(self, data, col_widths=None):\n"
PY_DOC = "        doc = SimpleDocTemplate(\n            self.output_path,\n"
PY_SAVESTATE = "    def _header_footer(self, canvas_obj, doc):\n        canvas_obj.saveState()\n"

def leak_into_throughput_tile(cond, text=None):
    text = text or f"'<p class=\"sub\">~' + {JS_DECODE_ESTIMATE} + ' tok/s per user (rough)</p>'"
    return [(INDEX_HTML, JS_TP_TILE, JS_TP_TILE + "${" + cond + " ? " + text + " : ''}", 1)]

def leak_into_pdf_explanation(cond):
    return [(REPORT_PY, PY_EXPLAIN, PY_EXPLAIN + f"            if {cond}:\n                story.append(Paragraph(f\"Rough decode estimate: ~{{{PY_DECODE_ESTIMATE}}} tokens/sec per user\", self.styles[\"Small\"]))\n", 1)]

S = {}
# ---- probe axes still fixed ----
S["V1 js: vendor-specific note prints '~<borrowed> tok/s' when state.vendor === 'amd'"] = [
    (INDEX_HTML, JS_NOTES_UNKNOWN, "    : 'Throughput is not modelled for this hardware: no measured utilisation is published for it. ' + (state.vendor === 'amd' ? 'ROCm decode on this class typically lands near ~' + " + JS_DECODE_ESTIMATE + " + ' tok/s per user. ' : '');\n", 1)]
S["V2 py: vendor-specific note prints '~<borrowed> tokens/sec' when cfg['vendor'] == 'amd'"] = [
    (REPORT_PY, PY_NOTE_PARAMS, PY_NOTE_PARAMS + f"        if cfg.get(\"vendor\") == \"amd\" and not c[\"throughput_modelled\"]:\n            notes.append(f\"ROCm decode on this class typically lands near ~{{{PY_DECODE_ESTIMATE}}} tokens/sec per user.\")\n", 1)]
S["V3 js: leak only for perfKey 'cdna3' (a specific unknown key)"] = leak_into_throughput_tile("state.perfKey === 'cdna3'")
S["V4 py: leak only for perfKey 'cdna3'"] = leak_into_pdf_explanation("cfg.get(\"perfKey\") == \"cdna3\"")
S["V5 js: leak only when prefix caching is switched off"] = leak_into_throughput_tile("state.prefixCaching === false")
S["V6 py: leak only when prefix caching is switched off"] = leak_into_pdf_explanation("cfg.get(\"prefix_caching\", True) is False")
S["V7 js: leak only for GGUF weights"] = leak_into_throughput_tile("state.quantMethod === 'gguf'")
S["V8 py: leak only for GGUF weights"] = leak_into_pdf_explanation("cfg.get(\"quant\") == \"gguf\"")
S["V9 js: leak only at exactly 8 boards"] = leak_into_throughput_tile("state.gpuCount === 8")
S["V10 py: leak only at exactly 8 boards"] = leak_into_pdf_explanation("cfg[\"n_gpu\"] == 8")
S["V11 js: leak only when a model was imported from Hugging Face (hfModelId set)"] = leak_into_throughput_tile("state.hfModelId")
S["V12 js: leak only when the card's name mentions MI300"] = leak_into_throughput_tile("/MI300/.test(state.gpuName)")
# ---- what the stub DOM does not record ----
S["W1 js: figure written with innerText (not textContent) to a side element"] = [
    (INDEX_HTML, SPEED_HINT_MARKUP, "<div id=\"throughput-output\"></div><div id=\"speed-hint\" style=\"font-size:11px\"></div>", 1),
    (INDEX_HTML, JS_TP_RETURN, JS_TP_RETURN.replace("    return;\n", f"    document.getElementById('speed-hint').innerText = 'Rough decode ~' + {JS_DECODE_ESTIMATE} + ' tok/s per user';\n    return;\n"), 1)]
S["W2 js: the cost panel is hidden (el.hidden = true) for a card without constants"] = [
    (INDEX_HTML, COST_SET, COST_SET + "  document.getElementById('cost-output').hidden = !computed.throughputModelled;\n", 1)]
S["W3 js: the cost panel is hidden via style.display for a card without constants"] = [
    (INDEX_HTML, COST_SET, COST_SET + "  document.getElementById('cost-output').style.display = computed.throughputModelled ? '' : 'none';\n", 1)]
S["W4 js: the browser tab title carries '~<borrowed> tok/s'"] = [
    (INDEX_HTML, JS_TP_RETURN, JS_TP_RETURN.replace("    return;\n", f"    document.title = 'Rough decode ~' + {JS_DECODE_ESTIMATE} + ' tok/s';\n    return;\n"), 1)]
S["W5 js: a throughput bar (no text) sized by the borrowed estimate"] = [
    (INDEX_HTML, JS_TP_TILE, JS_TP_TILE + f"<div class=\"throughput-bar\"><div style=\"width:${{Math.min({JS_DECODE_ESTIMATE} / 150 * 100, 100).toFixed(1)}}%;background:#1D9E75\"></div></div>", 1)]
S["W6 js: a <meter value=\"<borrowed>\"> gauge"] = [
    (INDEX_HTML, JS_TP_TILE, JS_TP_TILE + f"<meter value=\"${{{JS_DECODE_ESTIMATE}}}\" min=\"0\" max=\"300\"></meter>", 1)]
S["W7 js: a progressbar with aria-valuenow=\"<borrowed>\""] = [
    (INDEX_HTML, JS_TP_TILE, JS_TP_TILE + f"<div role=\"progressbar\" aria-valuenow=\"${{{JS_DECODE_ESTIMATE}}}\" aria-valuemin=\"0\" aria-valuemax=\"300\"></div>", 1)]
S["W8 js: a single-quoted title attribute: title='about <borrowed> per user'"] = [
    (INDEX_HTML, JS_TP_TILE, JS_TP_TILE.replace("<p class=\"value\">Not modelled</p>", f"<p class=\"value\" title='about ${{{JS_DECODE_ESTIMATE}}} per user'>Not modelled</p>"), 1)]
S["W9 js: renderSliderLabels (not exposed to the sweep) appends '(~<borrowed> tok/s)' to the GPU-count label"] = [
    (INDEX_HTML, SLIDER, "  document.getElementById('gpu-count-display').textContent = state.gpuCount + ((typeof state.perfKey === 'string' && Object.hasOwn(PERF, state.perfKey)) ? '' : ' (~' + Math.round(state.gpuBandwidth * 0.7 / (state.params * state.bytesPerParam)) + ' tok/s)');\n", 1)]
S["W10 js: saveSnapshot() (never called by the harness) names the snapshot with '~<borrowed> t/s'"] = [
    (INDEX_HTML, SNAP, "  savedSnapshots.push({ name: name + (computed.throughputModelled ? '' : ' ~' + Math.round(computed.deviceBandwidth * 0.7 / (state.params * state.bytesPerParam)) + ' t/s'), state, computed, timestamp: Date.now() });\n", 1)]
# ---- the moved-figures exemption: strand a KV figure inside a figure piece ----
S["X1 js: max-batch figure folded into the aggregate tile's label; standalone card dropped from both branches"] = [
    (INDEX_HTML, AGG_LABEL, "<p class=\"label\">Aggregate ceiling @ ${computed.effectiveBatch} concurrent · max batch ${computed.maxBatchByKV} seq</p>", 1),
    (INDEX_HTML, KNOWN_TAIL, "${fabricNote}${fp8Note}</p></div></div>`;\n", 1),
    (INDEX_HTML, JS_TP_RETURN, JS_TP_RETURN.replace("</div>${maxBatchCard}</div>${queueWarning}", "</div></div>${queueWarning}"), 1)]
# ---- the word-subsequence shortening exemption ----
S["Y1 js: notes say 'PCIe 64-128 GB/s — 45-60% loss, growing with device count.' without constants"] = [
    (INDEX_HTML, JS_NOTES_PCIE, "computed.throughputModelled ? 'PCIe 64-128 GB/s — 45-60% decode loss, growing with device count. ' : 'PCIe 64-128 GB/s — 45-60% loss, growing with device count. '", 1)]
S["Y2 js: notes add 'Throughput is a decode estimate.' without constants"] = [
    (INDEX_HTML, JS_NOTES_UNKNOWN, "    : 'Throughput is not modelled for this hardware: no measured utilisation is published for it. Throughput is a decode estimate. ';\n", 1)]
S["Y3 js: a second badge 'PCIe — perf loss, worse with more devices' beside the reason badge"] = [
    (INDEX_HTML, JS_BADGE, JS_BADGE + "</span>${computed.throughputModelled ? '' : '<span class=\"badge\" style=\"background:var(--warning-bg);color:var(--warning-text)\">PCIe — perf loss, worse with more devices'}", 1)]
S["Y4 js: exec dp>1 note keeps ', so treat the figures as a ceiling to test, not a quote'"] = [
    (INDEX_HTML, JS_EXEC_DP, "${computed.throughputModelled ? ', so treat the throughput figures as a ceiling to test, not a quote' : ', so treat the figures as a ceiling to test, not a quote'}", 1)]
S["Y5 js: queue warning keeps ' — aggregate above reflects the batch that actually runs'"] = [
    (INDEX_HTML, JS_QUEUE, "${computed.throughputModelled ? ' — aggregate throughput above reflects the batch that actually runs' : ' — aggregate above reflects the batch that actually runs'}", 1)]
S["Y6 py: notes say 'PCIe (64-128 GB/s) loses 30-50% vs NVLink.' without constants"] = [
    (REPORT_PY, PY_NOTES_PCIE, "'PCIe (64-128 GB/s) loses 30-50% decode throughput vs NVLink.' if c['throughput_modelled'] else 'PCIe (64-128 GB/s) loses 30-50% vs NVLink.'", 1)]
S["Y7 py: 'Throughput is a theoretical estimate. Real numbers depend on batching strategy, …' printed under 'Not modelled'"] = [
    (REPORT_PY, PY_EXPLAIN, PY_EXPLAIN + "            story.append(Paragraph(\"Throughput is a theoretical estimate. Real numbers depend on batching strategy, attention implementation, quantization kernels, and workload mix.\", self.styles[\"Small\"]))\n", 1)]
S["Y8 py: dp>1 note keeps ', so the figures inherit that uncertainty.'"] = [
    (REPORT_PY, PY_DP_NOTE, "                         + (\", so the throughput and TTFT figures inherit that uncertainty.\"\n                            if c[\"throughput_modelled\"] else \", so the figures inherit that uncertainty.\"))\n", 1)]
# ---- what the PDF spy does not see ----
S["Z1 py: a second page callback (onLaterPages) draws the figure"] = [
    (REPORT_PY, PY_BUILD, "        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._later)\n", 1),
    (REPORT_PY, PY_KVTABLE, "    def _later(self, canvas_obj, doc):\n        self._header_footer(canvas_obj, doc)\n        if not self.comp[\"throughput_modelled\"]:\n            canvas_obj.drawString(self.margin, 16*mm, f\"rough decode ~{self.comp['single_tok'] or 0} tok/s per user\")\n\n" + PY_KVTABLE, 1)]
S["Z2 py: the figure printed to stderr while the document is built"] = [
    (REPORT_PY, PY_EXPLAIN, PY_EXPLAIN + f"            print(f\"Rough decode: ~{{{PY_DECODE_ESTIMATE}}} tok/s per user\", file=sys.stderr)\n", 1)]
S["Z3 py: the PDF's document title metadata carries the figure"] = [
    (REPORT_PY, PY_DOC, PY_DOC + f"            title=(f\"rough decode ~{{{PY_DECODE_ESTIMATE}}} tok/s per user\" if not c[\"throughput_modelled\"] else \"LLM VRAM Planning Report\"),\n", 1)]
S["Z4 py: canvas.setTitle() in the header/footer carries the figure"] = [
    (REPORT_PY, PY_SAVESTATE, PY_SAVESTATE + "        if not self.comp[\"throughput_modelled\"]:\n            canvas_obj.setTitle(f\"rough decode ~{self.comp['single_tok'] or 0} tok/s per user\")\n", 1)]
S["Z5 py: the figure inside a ListFlowable (children in _flowables)"] = [
    (REPORT_PY, PY_IMPORT, "        HRFlowable, KeepTogether, PageBreak, ListFlowable, ListItem\n", 1),
    (REPORT_PY, PY_EXPLAIN, PY_EXPLAIN + f"            story.append(ListFlowable([ListItem(Paragraph(f\"Rough decode estimate: ~{{{PY_DECODE_ESTIMATE}}} tokens/sec per user\", self.styles[\"Small\"]))], bulletType=\"bullet\"))\n", 1)]

if __name__ == "__main__":
    run_driver(S)
