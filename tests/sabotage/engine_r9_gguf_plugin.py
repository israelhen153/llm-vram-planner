#!/usr/bin/env python3
"""Round 9: fix/gguf-plugin. Sabotages against the branch that made the GGUF
guidance say what vLLM needs since v0.24.0 (its plugin), carry a source for
each sentence on the page, in the copied report and in the PDF, and that made
the copied report quote the command box rather than the panel's first <code>.

Each edit reintroduces one way the guidance could go wrong again: dropped from
one surface, shown for a precision that is not GGUF, gated on one GGUF level,
printed without its sources or under a command that is not printed, the retired
single-file claim restored, or the report quoting a banner span again. The ones
marked "both" edit the two engines' tables alike, so the parity check that holds
them equal cannot be what catches them.

Usage: python3 tests/sabotage/engine_r9_gguf_plugin.py [name-substring ...]
       python3 tests/sabotage/engine_r9_gguf_plugin.py --from G5
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import (INDEX_HTML, REPORT_PY, JS_EXPORT_CODEBOX, JS_GGUF_BANNER_IF,
                     JS_GGUF_BANNER_MAP, JS_GGUF_BANNER_LINK, JS_GGUF_EXPORT_IF,
                     JS_GGUF_EXPORT_SOURCE, JS_GGUF_FIRST, JS_GGUF_ADVICE, JS_GGUF_RELEASE_SRC,
                     PY_GGUF_PDF_IF, PY_GGUF_PDF_SOURCE, PY_GGUF_FIRST, PY_GGUF_ADVICE,
                     PY_GGUF_RELEASE_SRC)

S = {}

# ---- R: the copied report's command block ----
S["R1 js: the copied report quotes the panel's first <code> again, a banner span for GGUF"] = [
    (INDEX_HTML, JS_EXPORT_CODEBOX,
     "  const emitted = document.querySelector('#command-output code');\n", 1)]

# ---- G: the guidance, one surface at a time ----
S["G1 js: the command panel drops the plugin sentence and keeps the rest"] = [
    (INDEX_HTML, JS_GGUF_BANNER_MAP, JS_GGUF_BANNER_MAP.replace("GGUF_GUIDANCE.map(", "GGUF_GUIDANCE.slice(1).map("), 1)]
S["G2 js: the command panel shows the guidance for every precision but AWQ"] = [
    (INDEX_HTML, JS_GGUF_BANNER_IF, "    if (state.quantMethod !== 'awq') {\n", 1)]
S["G3 js: the command panel prints the sources as text, with no link"] = [
    (INDEX_HTML, JS_GGUF_BANNER_LINK, "        + (source ? ' (source)' : ''));\n", 1)]
S["G4 js: the copied report loses its GGUF section"] = [
    (INDEX_HTML, JS_GGUF_EXPORT_IF, "  if (false && computed.fits && state.quantMethod === 'gguf') {\n", 1)]
S["G5 js: the copied report's GGUF section only for Q4_K_M"] = [
    (INDEX_HTML, JS_GGUF_EXPORT_IF,
     "  if (computed.fits && state.quantMethod === 'gguf' && state.bytesPerParam === 0.63) {\n", 1)]
S["G6 js: the copied report prints the guidance with no sources"] = [
    (INDEX_HTML, JS_GGUF_EXPORT_SOURCE, "- ${text}`", 1)]
S["G7 js: the copied report prints the guidance under a command it does not print"] = [
    (INDEX_HTML, JS_GGUF_EXPORT_IF, "  if (state.quantMethod === 'gguf') {\n", 1)]
S["G8 py: the PDF loses the guidance"] = [
    (REPORT_PY, PY_GGUF_PDF_IF, "        if False:\n", 1)]
S["G9 py: the PDF's guidance only for --prec q4km"] = [
    (REPORT_PY, PY_GGUF_PDF_IF,
     "        if c[\"fits\"] and cfg.get(\"quant\") == \"gguf\" and cfg[\"bpp\"] == 0.63:\n", 1)]
S["G10 py: the PDF prints the guidance with no sources"] = [
    (REPORT_PY, PY_GGUF_PDF_SOURCE, "                                       + \"\",\n", 1)]
S["G11 py: the PDF prints the guidance under a command it does not print"] = [
    (REPORT_PY, PY_GGUF_PDF_IF, "        if cfg.get(\"quant\") == \"gguf\":\n", 1)]

# ---- B: both tables edited alike, so parity holds and only the literals can catch it ----
S["B1 both: the plugin sentence deleted from both tables"] = [
    (INDEX_HTML, JS_GGUF_FIRST + JS_GGUF_RELEASE_SRC, "", 1),
    (REPORT_PY, PY_GGUF_FIRST + PY_GGUF_RELEASE_SRC, "", 1)]
S["B2 both: the retired single-file claim put back into the advice"] = [
    (INDEX_HTML, JS_GGUF_ADVICE,
     "  ['Point vllm serve at a single .gguf file, not a repo. For GGUF specifically, llama.cpp or "
     "Ollama is the better-supported path; AWQ or GPTQ is the usual choice on vLLM.',\n", 1),
    (REPORT_PY, PY_GGUF_ADVICE,
     "    (\"Point vllm serve at a single .gguf file, not a repo. For GGUF specifically, llama.cpp or \"\n"
     "     \"Ollama is the better-supported path; AWQ or GPTQ \"\n", 1)]
S["B3 both: the release-notes source replaced by nothing"] = [
    (INDEX_HTML, JS_GGUF_RELEASE_SRC, "   null],\n", 1),
    (REPORT_PY, PY_GGUF_RELEASE_SRC, "     None),\n", 1)]

run_driver(S)
