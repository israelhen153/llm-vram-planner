#!/usr/bin/env python3
"""The identity test exempts a piece that changes when the engine's throughput
figures are moved. Give the max-batch card an invisible dependency on one
(a trailing space keyed on computeBound), then withhold the card for the
unknown card."""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver
from anchors import INDEX_HTML, REPORT_PY, JS_TP_RETURN

MAXBATCH = "  const maxBatchCard = `<div class=\"reverse-card\"><p class=\"label\">Max batch at this context</p><p class=\"value\">${computed.maxBatchByKV} seq</p><p class=\"sub\">${computed.batchLimitedByKV ? `KV cache caps you below ${state.concurrency} requested` : 'KV cache has room for your concurrency'}</p></div>`;\n"

S = {}
S["T1 js: max-batch card withheld for the unknown card, exempted by an invisible computeBound dependency"] = [
    (INDEX_HTML, MAXBATCH, MAXBATCH.replace("</p></div>`;\n", "</p>${computed.computeBound ? ' ' : ''}</div>`;\n"), 1),
    (INDEX_HTML, JS_TP_RETURN, JS_TP_RETURN.replace("</div>${maxBatchCard}</div>${queueWarning}", "</div></div>${queueWarning}"), 1)]
S["T1b js (control): max-batch card withheld without the exemption trick"] = [
    (INDEX_HTML, JS_TP_RETURN, JS_TP_RETURN.replace("</div>${maxBatchCard}</div>${queueWarning}", "</div></div>${queueWarning}"), 1)]

if __name__ == "__main__":
    run_driver(S)
