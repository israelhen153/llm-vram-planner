#!/usr/bin/env python3
"""Second batch: leaks the suites' regexes may not see, and the benchmark panel
drawn for a card without constants. Same machinery as sab.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sab
from sab import IDX, GR, JS_EXEC_UNKNOWN, JS_TP_TILE, JS_CMP_UNKNOWN, JS_MD_UNKNOWN, PY_NOTMOD_ROW, PY_EXPLAIN

S = {}
# A zero with a unit the FIGURE/BAD regexes do not know, beside the reason.
S["H16 js: exec summary also prints 'Speed per user ~0 tokens per second'"] = [
    (IDX, JS_EXEC_UNKNOWN, JS_EXEC_UNKNOWN + "    html += `<div class=\"exec-row\"><span class=\"exec-label\">Speed per user</span><span class=\"exec-value\">~${formatTokens(computed.singleStreamTokS)} tokens per second</span></div>`;\n", 1)]
S["H17 js: throughput tile also prints '~0 tokens per second'"] = [
    (IDX, JS_TP_TILE, JS_TP_TILE.replace("<p class=\"sub\">${unmodelledNote}</p>", "<p class=\"sub\">${unmodelledNote} · ~${formatTokens(computed.singleStreamTokS)} tokens per second per user</p>"), 1)]
S["H18 js: comparison card also prints '0 tokens per second'"] = [
    (IDX, JS_CMP_UNKNOWN, JS_CMP_UNKNOWN.replace("not modelled<br>", "not modelled (${formatTokens(c.singleStreamTokS)} tokens per second)<br>"), 1)]
S["H19 js: markdown export also prints '- Per-user speed: ~0 tokens per second'"] = [
    (IDX, JS_MD_UNKNOWN, JS_MD_UNKNOWN.replace("do not transfer.\\n`;", "do not transfer.\\n- Per-user speed: ~${formatTokens(computed.singleStreamTokS)} tokens per second\\n`;"), 1)]
S["H20 py: PDF table also prints a 'Per user under that load' row of '~0 tokens per second'"] = [
    (GR, PY_NOTMOD_ROW, PY_NOTMOD_ROW + "                [\"Per user under that load\", f\"~{c['per_user_load'] or 0} tokens per second\"],\n", 1)]
S["H21 py: PDF explanation ends with 'expect roughly 0 tokens per second per user'"] = [
    (GR, "                \"depend on them.\",\n                self.styles[\"Small\"]\n", "                f\"depend on them. Expect roughly {c['single_tok'] or 0} tokens per second per user.\",\n                self.styles[\"Small\"]\n", 1)]
# A stray TTFT with a unit spelled out
S["H22 js: throughput tile also prints 'first token in ~0 milliseconds'"] = [
    (IDX, JS_TP_TILE, JS_TP_TILE.replace("<p class=\"sub\">${unmodelledNote}</p>", "<p class=\"sub\">${unmodelledNote} · first token in ~${computed.ttftMs || 0} milliseconds</p>"), 1)]
S["H23 py: PDF table also prints 'Est. time to first token' as '~0 milliseconds'"] = [
    (GR, PY_NOTMOD_ROW, PY_NOTMOD_ROW + "                [\"Est. time to first token\", f\"~{c['ttft_ms'] or 0} milliseconds\"],\n", 1)]
# The benchmark panel drawn for the unknown card, figures suppressed.
S["M1 js: benchmark panel still drawn for the unknown card"] = [
    (IDX, "    setHTML('throughput-output', `<div class=\"reverse-grid\"><div class=\"reverse-card\"><p class=\"label\">Throughput and TTFT</p><p class=\"value\">Not modelled</p><p class=\"sub\">${unmodelledNote}</p></div>${maxBatchCard}</div>${queueWarning}`);\n    return;\n  }\n",
     "    setHTML('throughput-output', `<div class=\"reverse-grid\"><div class=\"reverse-card\"><p class=\"label\">Throughput and TTFT</p><p class=\"value\">Not modelled</p><p class=\"sub\">${unmodelledNote}</p></div>${maxBatchCard}</div>${queueWarning}`);\n  }\n", 1),
    (IDX, "  let html = `<div class=\"reverse-grid\"><div class=\"reverse-card\"><p class=\"label\">Single-stream decode (1 user)</p>",
     "  let html = ''; if (computed.throughputModelled) { html = `<div class=\"reverse-grid\"><div class=\"reverse-card\"><p class=\"label\">Single-stream decode (1 user)</p>", 1),
    (IDX, "  html += queueWarning;\n\n  /* Benchmark panel.", "  html += queueWarning; } else { html = document.getElementById('throughput-output').innerHTML; }\n\n  /* Benchmark panel.", 1),
    # and the exact-match branch must not print our (null) estimate, or the figure regex catches it first
    (IDX, "    if (benchmark.exact) {\n      const ours", "    if (benchmark.exact && computed.throughputModelled) {\n      const ours", 1),
    (IDX, "    } else {\n      const direction = benchmark.slower", "    } else if (computed.throughputModelled) {\n      const direction = benchmark.slower", 1),
    (IDX, "      html += `<p style=\"font-size:11px;color:var(--text-muted);margin:4px 0 0\">Measured ${benchmark.requestedParams}B numbers on this card would replace this — see <a href=\"https://github.com/israelhen153/llm-vram-planner/blob/HEAD/CONTRIBUTING.md\" target=\"_blank\" style=\"color:var(--accent-text)\">CONTRIBUTING.md</a>.</p>`;\n    }\n",
     "      html += `<p style=\"font-size:11px;color:var(--text-muted);margin:4px 0 0\">Measured ${benchmark.requestedParams}B numbers on this card would replace this — see <a href=\"https://github.com/israelhen153/llm-vram-planner/blob/HEAD/CONTRIBUTING.md\" target=\"_blank\" style=\"color:var(--accent-text)\">CONTRIBUTING.md</a>.</p>`;\n    } else {\n      html += `<p style=\"font-size:12px;font-weight:500;margin:0 0 4px\">Published benchmark on this card: ${provenance}</p>`;\n    }\n", 1)]
# Reason present but the queue warning still points at figures, only for the batch-limited card
S["M2 js: queue warning keeps its 'aggregate throughput above' clause"] = sab.S["H8 js: queue warning always points at 'aggregate throughput above'"]

if __name__ == "__main__":
    pats = sys.argv[1:]
    names = [n for n in S if not pats or any(p.lower() in n.lower() for p in pats)]
    print(f"{len(names)} sabotage(s)")
    sab.require_green_baseline()
    survived, unapplied = [], []
    for name in names:
        try:
            touched = sab.apply(S[name])
        except Exception as e:
            print(f"  !! {name}: could not apply: {e}")
            unapplied.append(name)
            sab.restore([IDX, GR])
            continue
        try:
            res = sab.run_suites()
        finally:
            sab.restore(touched)
        red = {k: v for k, v in res.items() if v[0] != 0}
        if not red:
            survived.append(name)
            print(f"  GREEN  {name}   <-- SURVIVED")
        else:
            bits = []
            for k, (rc, fails, errs, tally) in red.items():
                first = (fails or errs or ["(no FAIL line)"])[0]
                bits.append(f"{k}[{tally[1] if tally else '?'} failed: {first[:110]}]")
            print(f"  red    {name}\n         " + "\n         ".join(bits))
    print(f"\n{len(names) - len(survived) - len(unapplied)} caught, "
          f"{len(survived)} survived"
          + (f", {len(unapplied)} COULD NOT BE APPLIED" if unapplied else ""))
    for s in survived:
        print("  SURVIVED: " + s)
    if unapplied:
        # A sabotage that never reached the tree judged nothing. Counting it as
        # caught is how a drifted driver reports a clean run forever.
        sys.exit(2)
