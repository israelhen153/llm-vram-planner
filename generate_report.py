#!/usr/bin/env python3
"""
LLM VRAM Planning Report Card — PDF Generator
Generates a polished PDF report from model and GPU configuration.

Usage:
    python generate_report.py                          # Interactive mode
    python generate_report.py --json config.json       # From JSON config
    python generate_report.py --preset gemma4-26b --gpu a100-40 --ngpu 1 --prec int4
"""

import argparse
import json
import math
import os
import shlex
import sys
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.colors import HexColor, Color, white, black
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether, PageBreak
    )
    from reportlab.pdfgen import canvas
    from reportlab.graphics.shapes import Drawing, Rect, String, Line
    from reportlab.graphics import renderPDF
except ImportError:
    print("reportlab is required: pip install reportlab")
    sys.exit(1)

# =============================================================================
# Colors
# =============================================================================
C_BG = HexColor("#FAFAF8")
C_BG2 = HexColor("#F1EFE8")
C_BD = HexColor("#D3D1C7")
C_FG = HexColor("#1A1A18")
C_FG2 = HexColor("#5F5E5A")
C_FG3 = HexColor("#888780")
C_ACC = HexColor("#534AB7")
C_PURPLE = HexColor("#7F77DD")
C_AMBER = HexColor("#EF9F27")
C_TEAL = HexColor("#1D9E75")
C_GRAY = HexColor("#888780")
C_OK = HexColor("#0F6E56")
C_OK_BG = HexColor("#E1F5EE")
C_WARN = HexColor("#BA7517")
C_WARN_BG = HexColor("#FAEEDA")
C_ERR = HexColor("#A32D2D")
C_ERR_BG = HexColor("#FCEBEB")

# =============================================================================
# Presets
# =============================================================================
PRESETS = {
    "llama31-8b":        {"p":8, "l":32, "kv":8, "hd":128, "a":100, "moe":False, "se":0, "attn":"standard", "max_ctx":131072, "hf":"meta-llama/Llama-3.1-8B-Instruct", "name":"Llama 3.1 8B"},
    "llama31-70b":       {"p":70, "l":80, "kv":8, "hd":128, "a":100, "moe":False, "se":0, "attn":"standard", "max_ctx":131072, "hf":"meta-llama/Llama-3.1-70B-Instruct", "name":"Llama 3.1 70B"},
    "llama33-70b":       {"p":70, "l":80, "kv":8, "hd":128, "a":100, "moe":False, "se":0, "attn":"standard", "max_ctx":131072, "hf":"meta-llama/Llama-3.3-70B-Instruct", "name":"Llama 3.3 70B"},
    "llama4-109b":       {"p":109, "l":48, "kv":8, "hd":128, "a":16, "moe":True, "se":0, "attn":"swa", "swa_win":8192, "swa_local":36, "max_ctx":10485760, "hf":"meta-llama/Llama-4-Scout-17B-16E-Instruct", "name":"Llama 4 Scout 109B MoE"},
    "qwen3-8b":          {"p":8, "l":36, "kv":8, "hd":128, "a":100, "moe":False, "se":0, "attn":"standard", "max_ctx":40960, "hf":"Qwen/Qwen3-8B", "name":"Qwen 3 8B"},
    "qwen3-30b":         {"p":30, "l":48, "kv":4, "hd":128, "a":10, "moe":True, "se":0, "attn":"standard", "max_ctx":40960, "hf":"Qwen/Qwen3-30B-A3B", "name":"Qwen 3 30B-A3B MoE"},
    "qwen25-coder-32b":  {"p":32, "l":64, "kv":8, "hd":128, "a":100, "moe":False, "se":0, "attn":"standard", "max_ctx":32768, "hf":"Qwen/Qwen2.5-Coder-32B-Instruct", "name":"Qwen 2.5 Coder 32B"},
    "qwen35-397b":       {"p":397, "l":60, "kv":2, "hd":256, "a":4, "moe":True, "se":0, "attn":"standard", "max_ctx":262144, "hf":"Qwen/Qwen3.5-397B-A17B", "name":"Qwen 3.5 397B MoE"},
    "qwen3-coder-480b":  {"p":480, "l":62, "kv":8, "hd":128, "a":7, "moe":True, "se":0, "attn":"standard", "max_ctx":262144, "hf":"Qwen/Qwen3-Coder-480B-A35B-Instruct", "name":"Qwen 3 Coder 480B MoE"},
    "gemma4-e4b":        {"p":4, "l":42, "kv":2, "hd":256, "a":100, "moe":False, "se":0, "attn":"swa", "swa_win":512, "swa_local":35, "max_ctx":131072, "hf":"google/gemma-4-E4B-it", "name":"Gemma 4 E4B"},
    "gemma4-26b":        {"p":26, "l":30, "kv":8, "hd":256, "a":15, "moe":True, "se":0, "attn":"swa", "swa_win":1024, "swa_local":25, "max_ctx":262144, "hf":"google/gemma-4-26B-A4B-it", "name":"Gemma 4 26B MoE"},
    "gemma4-31b":        {"p":31, "l":60, "kv":16, "hd":256, "a":100, "moe":False, "se":0, "attn":"swa", "swa_win":1024, "swa_local":50, "max_ctx":262144, "hf":"google/gemma-4-31B-it", "name":"Gemma 4 31B Dense"},
    "dsv3-671b":         {"p":671, "l":61, "kv":128, "hd":56, "a":5, "moe":True, "se":1, "attn":"mla", "mla_dim":576, "max_ctx":163840, "hf":"deepseek-ai/DeepSeek-V3", "name":"DeepSeek-V3 671B MoE"},
    "dsr1-671b":         {"p":671, "l":61, "kv":128, "hd":56, "a":5, "moe":True, "se":1, "attn":"mla", "mla_dim":576, "max_ctx":163840, "hf":"deepseek-ai/DeepSeek-R1", "name":"DeepSeek-R1 671B MoE"},
    "mistral-sm4-24b":   {"p":24, "l":40, "kv":8, "hd":128, "a":100, "moe":False, "se":0, "attn":"standard", "max_ctx":131072, "hf":"mistralai/Mistral-Small-4-24B-Instruct-2503", "name":"Mistral Small 4 24B"},
    "mistral-lg-123b":   {"p":123, "l":88, "kv":8, "hd":128, "a":100, "moe":False, "se":0, "attn":"standard", "max_ctx":131072, "hf":"mistralai/Mistral-Large-Instruct-2411", "name":"Mistral Large 123B"},
}

# Memory is reported in GiB (2^30 B): nvidia-smi shows an "80 GB" H100 as 79.6 GiB,
# so a card's marketing number behaves as GiB. Mixing it with decimal GB on the model
# side made every fit check ~7% pessimistic.
GIB = 1024 ** 3

# GPU_TABLE:BEGIN — generated by tools/sync_data.py from data/gpus.json. Do not hand-edit; edit the JSON and re-run the script.
GPUS = {
    "t4-16": {"gb":16,"bw":320,"hyper":0.76,"spec":0.35,"spot":0.15,"tflops":65,"name":"T4 16 GB","vendor":"nvidia","devices":1,"form":"pcie","caps":{"fp8": False}},
    "l4-24": {"gb":24,"bw":300,"hyper":0.8,"spec":0.43,"spot":0.22,"tflops":121,"name":"L4 24 GB","vendor":"nvidia","devices":1,"form":"pcie","caps":{"fp8": True}},
    "rtx4090-24": {"gb":24,"bw":1008,"hyper":0.75,"spec":0.45,"spot":0.29,"tflops":165,"name":"RTX 4090 24 GB","vendor":"nvidia","devices":1,"form":"consumer","caps":{"fp8": True}},
    "rtx5090-32": {"gb":32,"bw":1792,"hyper":0.89,"spec":0.65,"spot":0.45,"tflops":419,"name":"RTX 5090 32 GB","vendor":"nvidia","devices":1,"form":"consumer","caps":{"fp8": True}},
    "a100-40": {"gb":40,"bw":1555,"hyper":3.67,"spec":1.29,"spot":0.63,"tflops":312,"name":"A100 40 GB","vendor":"nvidia","devices":1,"form":"sxm","caps":{"fp8": False},"default":True},
    "rtx6000ada-48": {"gb":48,"bw":960,"hyper":1.5,"spec":0.9,"spot":0.6,"tflops":364,"name":"RTX 6000 Ada 48 GB","vendor":"nvidia","devices":1,"form":"pcie","caps":{"fp8": True}},
    "l40s-48": {"gb":48,"bw":864,"hyper":2.8,"spec":1.3,"spot":0.9,"tflops":362,"name":"L40S 48 GB","vendor":"nvidia","devices":1,"form":"pcie","caps":{"fp8": True}},
    "a100-80": {"gb":80,"bw":2039,"hyper":4.5,"spec":1.79,"spot":0.99,"tflops":312,"name":"A100 80 GB","vendor":"nvidia","devices":1,"form":"sxm","caps":{"fp8": False}},
    "h100-80": {"gb":80,"bw":3352,"hyper":12.3,"spec":3.99,"spot":2.25,"tflops":990,"name":"H100 80 GB","vendor":"nvidia","devices":1,"form":"sxm","caps":{"fp8": True}},
    "rtxpro-96": {"gb":96,"bw":1792,"hyper":5.0,"spec":3.5,"spot":2.0,"tflops":504,"name":"RTX PRO 6000 96 GB","vendor":"nvidia","devices":1,"form":"pcie","caps":{"fp8": True}},
    "h200-141": {"gb":141,"bw":4800,"hyper":10.6,"spec":3.99,"spot":2.5,"tflops":990,"name":"H200 141 GB","vendor":"nvidia","devices":1,"form":"sxm","caps":{"fp8": True}},
    "b200-192": {"gb":192,"bw":8000,"hyper":14.24,"spec":5.5,"spot":2.12,"tflops":2250,"name":"B200 192 GB","vendor":"nvidia","devices":1,"form":"sxm","caps":{"fp8": True}},
}
# GPU_TABLE:END


# The card the tool falls back to when nothing else is specified. Derived from
# the catalog's own `default` flag: this slug used to be written out in three
# places here and one in index.html, which is four chances to disagree about
# what "no GPU specified" means.
DEFAULT_GPU_KEY = next((k for k, g in GPUS.items() if g.get("default")), next(iter(GPUS)))


def supports_nvlink(gpu):
    """Whether this board carries NVLink at all.

    Seven of the twelve catalogued cards do not — T4, L4, RTX 4090, RTX 5090,
    RTX 6000 Ada, L40S, RTX PRO 6000 — yet nvlink defaulted to True at every
    construction site below, so a multi-GPU report on any of them claimed an
    interconnect the machine does not have and scaled throughput by 0.85 for
    it. Reads the catalog's `form` rather than a name list, so a new vendor's
    rows answer without editing this function. Mirrored by supportsNVLink() in
    index.html; the parity suite compares the two answers card by card.
    """
    return gpu.get("form") == "sxm"


def nvlink_for(gpu, requested):
    """The interconnect this card will actually have, saying so when that is not
    what was asked for. The interactive path prints a line when it skips the
    question; a JSON config or a CLI flag asking for NVLink on a board without
    it was silently downgraded instead, and the only trace was one row of the
    finished report.
    """
    if requested and not supports_nvlink(gpu):
        print(f"Note: {gpu['name']} has no NVLink — using PCIe interconnect instead.")
        return False
    return bool(requested)


PREC_LABELS = {
    2.0:  "BF16",
    1.0:  "FP8",
    0.5:  "INT4 / AWQ / GPTQ",
    0.63: "GGUF Q4_K_M",
    0.71: "GGUF Q5_K_M",
    0.82: "GGUF Q6_K",
    1.1:  "GGUF Q8_0",
    0.35: "GGUF Q2_K",
}


def capacity_label(gb):
    """Mirrors capacityLabel() in index.html: an integer catalog capacity keeps
    rendering as an integer, so an unchanged card's report is unchanged."""
    return f"{int(gb)} GiB" if float(gb).is_integer() else fmt_gb(gb)


def fmt_gb(gb):
    if gb < 0.01: return "0 GiB"
    if gb >= 100: return f"{round(gb)} GiB"
    if gb >= 10: return f"{gb:.1f} GiB"
    return f"{gb:.2f} GiB"

def fmt_k(v):
    return f"{round(v/1024)}K" if v >= 1024 else str(v)

def fmt_tok(t):
    return f"{t/1000:.1f}K" if t >= 1000 else str(round(t))

def attn_label(cfg):
    """Describe the attention regime a cfg carries, with the detail that
    actually changes the KV cache math (window/local layers for SWA, latent
    dim for MLA) — this is the number that was silently wrong for a year
    because nothing in the report said which regime produced it."""
    mode = cfg.get("attn", "standard")
    if mode == "swa":
        local = min(cfg.get("swa_local", 0), cfg["layers"])
        return f"Sliding window (SWA) — {fmt_k(cfg.get('swa_win', 0))} token window, {local}/{cfg['layers']} local layers"
    if mode == "mla":
        return f"Multi-head latent (MLA) — {cfg.get('mla_dim', 0)}-dim latent"
    return "Standard (MHA/GQA)"


# =============================================================================
# VRAM Computation
# =============================================================================
# Vendor-keyed performance constants (memory-bandwidth utilisation, decode/prefill
# MFU, and the observed-efficiency band). Only "nvidia" is populated today — see
# the matching comment inside compute() for what each value means and how it was
# chosen. Mirrors index.html's PERF table exactly; the parity suite enforces it,
# and both engines must move together when a value changes.
PERF = {
    "nvidia": {
        "mbu": 0.70,          # achieved / peak HBM bandwidth during decode
        "mfuDecode": 0.35,    # achieved / peak dense FLOPS at large batch
        "mfuPrefill": 0.45,   # prefill GEMMs are dense; published utilisation ~40-55%
        "obsLo": 0.43,        # measured/ceiling ratio floor across batch benchmarks
        "obsHi": 0.91,        # measured/ceiling ratio ceiling across batch benchmarks
    },
}


def compute(cfg):
    params = cfg["params"]
    active_pct = cfg["active"]
    bpp = cfg["bpp"]
    layers = cfg["layers"]
    kv_heads = cfg["kv_heads"]
    h_dim = cfg["h_dim"]
    shared_exp = cfg.get("shared_exp", 0)
    ctx = cfg["ctx"]
    conc = cfg["conc"]
    n_gpu = cfg["n_gpu"]
    gpu = cfg["gpu"]
    # Catalog values are per *board* — the unit you buy, price and rack. The
    # runtime sees devices: an MI250X is one OAM presenting two GCDs, each with
    # its own memory and its own share of the fabric. Cost uses n_gpu;
    # everything else uses the device-scoped values below. Mirrors the same
    # derivation in index.html, and the parity suite compares the results.
    devices_per_board = gpu.get("devices", 1) or 1
    device_count = n_gpu * devices_per_board
    device_gb = gpu["gb"] / devices_per_board
    device_bw = gpu["bw"] / devices_per_board
    device_tflops = gpu["tflops"] / devices_per_board
    nvlink = cfg.get("nvlink", True)
    kv_bpp = cfg.get("kv_bpp", 2)
    is_moe = active_pct < 100

    weights_gb = (params * 1e9 * bpp) / GIB

    # KV cache. Three regimes — see the matching comment in index.html.
    #   standard  2 * L * kv_heads * head_dim * bytes * ctx
    #   swa       local layers cap at the window; only global layers keep growing
    #   mla       one compressed latent per layer, no K/V pair, no head multiplier
    attn = cfg.get("attn", "standard")
    swa_win = cfg.get("swa_win", 0)
    swa_local = min(cfg.get("swa_local", 0), layers)
    swa_global = max(layers - swa_local, 0)
    mla_dim = cfg.get("mla_dim", 0)

    def kv_bytes_for_seq(c):
        if attn == "mla":
            return layers * mla_dim * kv_bpp * c
        if attn == "swa":
            per_layer_token = 2 * kv_heads * h_dim * kv_bpp
            return per_layer_token * (swa_local * min(c, swa_win) + swa_global * c)
        return 2 * layers * kv_heads * h_dim * kv_bpp * c

    # Bytes eligible for sharing between sequences. Under SWA only the global
    # layers qualify — local layers keep a rolling window, so a prefix at
    # position 0 has already slid out of them.
    def kv_bytes_shareable(c):
        if attn == "mla":
            return layers * mla_dim * kv_bpp * c
        if attn == "swa":
            return 2 * kv_heads * h_dim * kv_bpp * swa_global * c
        return 2 * layers * kv_heads * h_dim * kv_bpp * c

    shared_prefix = cfg.get("shared_prefix", 0)
    prefix_caching = cfg.get("prefix_caching", True)
    kv_seq_bytes = kv_bytes_for_seq(ctx)
    eff_prefix = min(shared_prefix, ctx)
    shared_bytes = kv_bytes_shareable(eff_prefix) if prefix_caching else 0
    per_seq_bytes = max(kv_seq_bytes - shared_bytes, 0)
    kv_total_bytes = shared_bytes + per_seq_bytes * conc
    kv_saved_by_prefix_gb = (kv_seq_bytes * conc - kv_total_bytes) / GIB
    kv_bytes_per_tok = kv_seq_bytes / ctx if ctx > 0 else 0
    total_tokens = ctx * conc
    kv_gb = kv_total_bytes / GIB
    active_p = params * (active_pct / 100)
    shared_p = params * 0.02 * shared_exp if is_moe and shared_exp > 0 else 0
    total_active_p = active_p + shared_p
    act_gb = max((total_active_p * 1e9 * 2 * 0.01) / GIB, 0.1)
    oh_per_gpu = 1.5
    nccl_oh = (0.3 if nvlink else 0.2) * (device_count - 1) if device_count > 1 else 0
    total_oh = oh_per_gpu * device_count + nccl_oh
    total_gb = weights_gb + kv_gb + act_gb + total_oh

    per_w = weights_gb / device_count
    per_kv = kv_gb / device_count
    per_a = act_gb / device_count
    per_oh = oh_per_gpu + ((0.3 if nvlink else 0.2) if device_count > 1 else 0)
    per_total = per_w + per_kv + per_a + per_oh
    # gpu["gb"] * n_gpu, not device_gb * device_count: the two are equal by
    # construction, and this one keeps an integer catalog value an integer
    # rather than rendering "160.0 GB" where every previous report said 160.
    total_vram = gpu["gb"] * n_gpu

    fixed_pg = per_w + per_a + per_oh
    # Both operands move together, or free KV silently halves or doubles.
    free_kv = max((device_gb * 0.9 - fixed_pg) * device_count, 0)
    kv_per_tok_gb = kv_bytes_per_tok / GIB
    free_kv_bytes = free_kv * GIB

    # Binary search rather than a divide: under SWA the KV curve bends at the
    # window, so dividing by an effective per-token rate understates reach.
    # Never report a context the model cannot address.
    CONTEXT_CEILING = min(cfg.get("max_ctx", 131072), 1048576)
    max_ctx_1 = 0
    if kv_bytes_for_seq(1) > 0 and free_kv_bytes > 0:
        lo, hi = 0, CONTEXT_CEILING
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if kv_bytes_for_seq(mid) <= free_kv_bytes:
                lo = mid
            else:
                hi = mid - 1
        max_ctx_1 = lo

    def concurrent_at(c):
        per = kv_bytes_for_seq(c)
        return int(free_kv_bytes / per) if per > 0 else 0

    max_conc_8k = concurrent_at(8192)
    max_conc_4k = concurrent_at(4096)

    # Throughput — see the matching comment in index.html. Decode is bandwidth-bound:
    #   tok/s = (B * achieved_bw) / (active_weight_bytes + B * kv_bytes_per_seq)
    # At B=1 that is per-user speed; at B=batch it is aggregate server throughput,
    # capped by the compute roofline and by how many sequences fit in KV cache.
    # These are different quantities and must never be compared to each other.
    # An unknown or absent vendor falls back to nvidia rather than raising; this
    # is called from report rendering, where a KeyError loses the whole document.
    P = PERF.get(cfg.get("vendor") or "", PERF["nvidia"])
    # NVLink is roughly flat within a domain (full bisection); PCIe worsens with GPU
    # count — decode all-reduces are small, so hop latency dominates. Heuristic step,
    # same class as MBU/MFU. Mirrors index.html exactly; the parity suite enforces it.
    if device_count <= 1:
        nv_penalty = 1.0
    elif nvlink:
        nv_penalty = 0.85
    else:
        nv_penalty = max(0.55 - 0.05 * math.log2(device_count / 2), 0.40)
    achieved_bw = device_bw * 1e9 * device_count * P["mbu"] * nv_penalty
    active_weight_bytes = total_active_p * 1e9 * bpp
    kv_bytes_per_seq = kv_seq_bytes

    def decode_at(batch):
        denom = active_weight_bytes + batch * kv_bytes_per_seq
        return (batch * achieved_bw) / denom if denom > 0 else 0

    marginal_seq_bytes = max(kv_bytes_per_seq - shared_bytes, 1)
    max_batch_kv = (max(int((free_kv * GIB - shared_bytes) / marginal_seq_bytes), 0)
                    if kv_bytes_per_seq > 0 else conc)
    eff_batch = max(min(conc, max_batch_kv), 1)
    compute_ceiling = (
        (P["mfuDecode"] * device_tflops * 1e12 * device_count * nv_penalty) / (2 * total_active_p * 1e9)
        if total_active_p > 0 else 0
    )
    single_tok = round(decode_at(1))
    agg_tok = round(min(decode_at(eff_batch), compute_ceiling))
    # Saturated batch: the only basis on which a published batch benchmark can be
    # compared like with like, since those are run with the GPU fully loaded.
    sat_batch = max(max_batch_kv, 1)
    sat_tok = round(min(decode_at(sat_batch), compute_ceiling))
    per_user_load = round(agg_tok / eff_batch) if eff_batch else 0
    # The ceiling's observed discount: measured batch benchmarks land at 43-91% of
    # the roofline (the 1.1-2.3x optimism inverted). Mirrors index.html; the JS test
    # suite re-derives the band from benchmarks/data.json to keep it honest.
    agg_obs_lo = round(agg_tok * P["obsLo"])
    agg_obs_hi = round(agg_tok * P["obsHi"])
    batch_limited = conc > max_batch_kv

    # Prefill is compute-bound, not bandwidth-bound; deriving it from decode speed
    # understated TTFT by roughly 10x. MFU_PREFILL, not MFU_DECODE: dense GEMMs.
    achieved_prefill_flops = P["mfuPrefill"] * device_tflops * 1e12 * device_count * nv_penalty

    def ttft_for(tokens):
        flops = 2 * total_active_p * 1e9 * max(tokens, 0)
        return round((flops / achieved_prefill_flops) * 1000) if achieved_prefill_flops > 0 else 0

    # APC skips the shared prefix during prefill, so TTFT differs cold vs warm.
    # It never changes decode speed.
    ttft_cold_ms = ttft_for(ctx)
    ttft_warm_ms = ttft_for(ctx - eff_prefix) if prefix_caching else ttft_cold_ms
    ttft_ms = ttft_warm_ms
    # Boards, not devices: a dual-GCD module is one line item on the invoice.
    hourly_hyper = gpu["hyper"] * n_gpu
    hourly_spec = gpu["spec"] * n_gpu
    hourly_spot = gpu["spot"] * n_gpu

    fits = per_total <= device_gb
    comfortable = per_total <= device_gb * 0.9

    return {
        "weights_gb": weights_gb, "kv_gb": kv_gb, "act_gb": act_gb,
        "total_oh": total_oh, "total_gb": total_gb,
        "per_w": per_w, "per_kv": per_kv, "per_a": per_a, "per_oh": per_oh,
        "per_total": per_total, "total_vram": total_vram,
        "free_kv": free_kv, "kv_per_tok_gb": kv_per_tok_gb,
        "kv_bytes_per_tok": kv_bytes_per_tok,
        "max_ctx_1": max_ctx_1, "max_conc_8k": max_conc_8k, "max_conc_4k": max_conc_4k,
        "single_tok": single_tok, "agg_tok": agg_tok, "per_user_load": per_user_load,
        "agg_obs_lo": agg_obs_lo, "agg_obs_hi": agg_obs_hi,
        # The constants this result was actually computed with, so callers can
        # label it without re-reading PERF and drifting. The parity suite compares
        # these, which pins the two engines to the same values as executed.
        "perf_mbu": P["mbu"], "perf_mfu_decode": P["mfuDecode"],
        "perf_mfu_prefill": P["mfuPrefill"],
        "perf_obs_lo": P["obsLo"], "perf_obs_hi": P["obsHi"],
        "eff_batch": eff_batch, "max_batch_kv": max_batch_kv, "batch_limited": batch_limited,
        "ttft_ms": ttft_ms, "ttft_cold_ms": ttft_cold_ms, "ttft_warm_ms": ttft_warm_ms,
        "kv_saved_by_prefix_gb": kv_saved_by_prefix_gb, "eff_prefix": eff_prefix, "sat_batch": sat_batch, "sat_tok": sat_tok, "hourly_hyper": hourly_hyper, "hourly_spec": hourly_spec, "hourly_spot": hourly_spot,
        "is_moe": is_moe, "total_tokens": total_tokens,
        # The device view, so a renderer never has to re-derive it and get
        # it wrong: per-device figures must be read against per-device
        # capacity, not against the board they share.
        "device_count": device_count, "device_gb": device_gb, "device_bw": device_bw,
        "fits": fits, "comfortable": comfortable,
    }


def split_parallelism(gpu_count):
    """Below 8 GPUs: tp = gpu_count, dp = 1 - exactly master's behaviour, even
    when gpu_count isn't a valid head-count split (TP=3, TP=6). That looks
    wrong, but vLLM itself rejects an invalid TP loudly at startup; splitting
    into DP replicas down here too would make the report silently claim
    N-way sharding while the command runs 1-GPU replicas that each need the
    *full* model - wrong in a way nothing catches. Fixing the <=8 regime's
    own TP-validity problem is separate work.

    Above 8 GPUs: tp = the largest power of two dividing gpu_count, capped at
    8 (today's hardcoded single-node size; becomes a per-GPU `domain` field
    once multi-node topology is modeled instead of assumed - not this
    commit). dp absorbs whatever's left. Always exact (tp * dp == gpu_count,
    so a GPU is never dropped) and tp is always a valid power-of-two split
    for real attention head counts - unlike "largest divisor <= 8", which
    picked TP=5 for 100 GPUs and TP=6 for 12. This is the regime that
    actually diverged between the two engines.

    Mirrors index.html's splitParallelism(); the parity suite enforces it."""
    # n_gpu is declared (int, float) in ARCH_TYPES, so a fractional GPU count
    # can arrive from JSON; truncate before it can leak a float into tp*dp
    # (e.g. --data-parallel-size 3.0, or disagreeing with JS on a non-integer).
    gpu_count = int(gpu_count)
    if gpu_count <= 8:
        return gpu_count, 1
    tp = 1
    while tp < 8 and tp * 2 <= gpu_count and gpu_count % (tp * 2) == 0:
        tp *= 2
    return tp, gpu_count // tp


def device_count_for(cfg):
    """Accelerator devices, not boards. A parallelism split is sized in devices:
    a single dual-GCD module still needs --tensor-parallel-size 2, or half the
    silicon sits idle. Mirrors parallelismFor() in index.html."""
    return cfg["n_gpu"] * ((cfg.get("gpu") or {}).get("devices", 1) or 1)


def build_vllm_cmd(cfg, comp):
    if not comp["fits"]:
        return "# Does not fit — increase GPUs, lower precision, or reduce context"
    # hf_model and quant can both originate from a user's own JSON now (the
    # whole point of default-allow is that cfg carries whatever they wrote),
    # and this command is meant to be copied straight into a terminal.
    # shlex.quote() is a no-op on an ordinary value and neutralizes anything
    # that isn't one, instead of executing it.
    hf = shlex.quote(cfg.get("hf_model", "/opt/models/YourModel"))
    tp, dp = split_parallelism(device_count_for(cfg))
    parts = [f"vllm serve {hf} \\"]
    parts.append("    --host 0.0.0.0 --port 8000 \\")
    if tp > 1:
        parts.append(f"    --tensor-parallel-size {tp} \\")
    if dp > 1:
        parts.append(f"    --data-parallel-size {dp} \\")
    # --dtype auto reads torch_dtype from config.json. Forcing float16 breaks the
    # bf16-trained families (Llama 3, Gemma, Qwen), whose weights can overflow fp16.
    parts.append("    --dtype auto \\")
    if cfg.get("quant"):
        parts.append(f"    --quantization {shlex.quote(cfg['quant'])} \\")
    if cfg.get("kv_bpp", 2) < 2:
        parts.append("    --kv-cache-dtype fp8 \\")
    # APC is on by default in vLLM V1, so only the opt-out is worth emitting —
    # index.html's buildVllmCommand() has always done this. cfg already
    # carries prefix_caching (validated in ARCH_TYPES, threaded through
    # from_json/from_cli_args/interactive_mode); this function just never
    # read it, so the flag silently never appeared here regardless of the
    # setting, independent of anything to do with TP/DP.
    if not cfg.get("prefix_caching", True):
        parts.append("    --no-enable-prefix-caching \\")
    # tp * dp is the device count by construction — same expression as
    # index.html, so the two cannot drift on this flag.
    if comp["is_moe"] and tp * dp > 1:
        parts.append("    --enable-expert-parallel \\")
    parts.append("    --gpu-memory-utilization 0.90 \\")
    parts.append(f"    --max-model-len {min(comp['max_ctx_1'], cfg['ctx'])}")
    return "\n".join(parts)


# =============================================================================
# PDF Generation
# =============================================================================
class ReportCard:
    def __init__(self, cfg, output_path="llm-vram-report.pdf"):
        self.cfg = cfg
        self.comp = compute(cfg)
        self.output_path = output_path
        self.width, self.height = A4
        self.margin = 18 * mm
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        self.styles.add(ParagraphStyle(
            "ReportTitle", fontName="Helvetica-Bold", fontSize=18,
            textColor=C_FG, spaceAfter=2*mm, leading=22
        ))
        self.styles.add(ParagraphStyle(
            "ReportSub", fontName="Helvetica", fontSize=10,
            textColor=C_FG2, spaceAfter=6*mm
        ))
        self.styles.add(ParagraphStyle(
            "SectionHead", fontName="Helvetica-Bold", fontSize=12,
            textColor=C_ACC, spaceBefore=5*mm, spaceAfter=3*mm,
            borderWidth=0, leading=15
        ))
        self.styles.add(ParagraphStyle(
            "Body", fontName="Helvetica", fontSize=9.5,
            textColor=C_FG, leading=13
        ))
        self.styles.add(ParagraphStyle(
            "Small", fontName="Helvetica", fontSize=8,
            textColor=C_FG3, leading=11
        ))
        self.styles.add(ParagraphStyle(
            "CmdCode", fontName="Courier", fontSize=8,
            textColor=C_FG, leading=11, backColor=C_BG2,
            borderPadding=(4, 6, 4, 6)
        ))
        self.styles.add(ParagraphStyle(
            "VerdictOK", fontName="Helvetica-Bold", fontSize=11,
            textColor=C_OK, backColor=C_OK_BG, borderPadding=(6, 8, 6, 8),
            leading=14
        ))
        self.styles.add(ParagraphStyle(
            "VerdictWarn", fontName="Helvetica-Bold", fontSize=11,
            textColor=C_WARN, backColor=C_WARN_BG, borderPadding=(6, 8, 6, 8),
            leading=14
        ))
        self.styles.add(ParagraphStyle(
            "VerdictErr", fontName="Helvetica-Bold", fontSize=11,
            textColor=C_ERR, backColor=C_ERR_BG, borderPadding=(6, 8, 6, 8),
            leading=14
        ))

    def _header_footer(self, canvas_obj, doc):
        canvas_obj.saveState()
        # Header line
        canvas_obj.setStrokeColor(C_ACC)
        canvas_obj.setLineWidth(2)
        canvas_obj.line(self.margin, self.height - 14*mm, self.width - self.margin, self.height - 14*mm)
        canvas_obj.setFont("Helvetica-Bold", 8)
        canvas_obj.setFillColor(C_ACC)
        canvas_obj.drawString(self.margin, self.height - 12*mm, "LLM VRAM PLANNING REPORT")
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillColor(C_FG3)
        canvas_obj.drawRightString(self.width - self.margin, self.height - 12*mm, datetime.now().strftime("%Y-%m-%d"))
        # Footer
        canvas_obj.setFont("Helvetica", 7)
        canvas_obj.setFillColor(C_FG3)
        canvas_obj.drawString(self.margin, 10*mm, "Generated by LLM VRAM Planner")
        canvas_obj.drawRightString(self.width - self.margin, 10*mm, f"Page {doc.page}")
        canvas_obj.setStrokeColor(C_BD)
        canvas_obj.setLineWidth(0.5)
        canvas_obj.line(self.margin, 13*mm, self.width - self.margin, 13*mm)
        canvas_obj.restoreState()

    def _make_kv_table(self, data, col_widths=None):
        """Create a styled key-value table."""
        style = TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), C_FG2),
            ("TEXTCOLOR", (1, 0), (1, -1), C_FG),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("LINEBELOW", (0, 0), (-1, -2), 0.3, C_BD),
        ])
        if col_widths is None:
            col_widths = [50*mm, 80*mm]
        t = Table(data, colWidths=col_widths)
        t.setStyle(style)
        return t

    def _make_metric_row(self, metrics):
        """Create a row of metric boxes."""
        data = [
            [Paragraph(f'<font size="8" color="#{C_FG2.hexval()[2:]}">{m[0]}</font><br/>'
                        f'<font size="14"><b>{m[1]}</b></font>', self.styles["Body"])
             for m in metrics]
        ]
        col_w = (self.width - 2 * self.margin) / len(metrics)
        t = Table(data, colWidths=[col_w] * len(metrics))
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_BG2),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
            ("BOX", (0, 0), (-1, -1), 0.3, C_BD),
            ("LINEBEFORE", (1, 0), (-1, -1), 0.3, C_BD),
        ]))
        return t

    def _vram_bar(self):
        """Create a VRAM usage bar as a Drawing."""
        bar_w = self.width - 2 * self.margin
        bar_h = 20
        d = Drawing(bar_w, bar_h + 18)
        c = self.comp
        # The bar draws per-device components, so its scale and its free
        # space are the device's. Drawn against the board it printed
        # "59.1 GiB free" directly beneath a verdict saying the same
        # configuration was over by 4.89 GiB.
        gpu_gb = c["device_gb"]
        total_max = max(c["per_total"], gpu_gb) * 1.05

        segments = [
            (c["per_w"], C_PURPLE, "Weights"),
            (c["per_kv"], C_AMBER, "KV cache"),
            (c["per_a"], C_TEAL, "Activations"),
            (c["per_oh"], C_GRAY, "Overhead"),
        ]
        x = 0
        for gb, color, label in segments:
            w = max((gb / total_max) * bar_w, 1)
            d.add(Rect(x, 0, w, bar_h, fillColor=color, strokeColor=None))
            if w > 40:
                d.add(String(x + w/2, 5, fmt_gb(gb), fontSize=7, fontName="Helvetica-Bold",
                             fillColor=white, textAnchor="middle"))
            x += w
        free = gpu_gb - c["per_total"]
        if free > 0:
            w = (free / total_max) * bar_w
            d.add(Rect(x, 0, w, bar_h, fillColor=C_BG2, strokeColor=C_BD, strokeWidth=0.3))
            if w > 35:
                d.add(String(x + w/2, 5, fmt_gb(free) + " free", fontSize=7,
                             fontName="Helvetica", fillColor=C_FG3, textAnchor="middle"))
        d.add(Rect(0, 0, bar_w, bar_h, fillColor=None, strokeColor=C_BD, strokeWidth=0.5))

        # Legend below bar
        lx = 0
        for _, color, label in segments:
            d.add(Rect(lx, bar_h + 6, 8, 8, fillColor=color, strokeColor=None))
            d.add(String(lx + 11, bar_h + 7, label, fontSize=7, fontName="Helvetica", fillColor=C_FG3))
            lx += 70

        return d

    def generate(self):
        cfg = self.cfg
        c = self.comp
        gpu = cfg["gpu"]
        story = []

        # ---- Title ----
        model_name = cfg.get("model_name", f"{cfg['params']}B model")
        story.append(Paragraph(f"GPU Deployment Plan: {model_name}", self.styles["ReportTitle"]))
        story.append(Paragraph(
            f"Generated {datetime.now().strftime('%B %d, %Y at %H:%M')} — "
            f"{cfg['n_gpu']}x {gpu['name']}"
            f"{' (NVLink)' if cfg.get('nvlink') and device_count_for(cfg) > 1 else ''}"
            f"{' (PCIe)' if not cfg.get('nvlink') and device_count_for(cfg) > 1 else ''}",
            self.styles["ReportSub"]
        ))

        # ---- Verdict ----
        if c["comfortable"]:
            style_name = "VerdictOK"
            icon = "FITS"
            msg = (f"{fmt_gb(c['per_total'])} per device of {capacity_label(c['device_gb'])} "
                   f"({round(c['per_total'] / c['device_gb'] * 100)}%). Headroom available.")
        elif c["fits"]:
            style_name = "VerdictWarn"
            icon = "TIGHT"
            msg = (f"{fmt_gb(c['per_total'])} per device of {capacity_label(c['device_gb'])} "
                   f"({round(c['per_total'] / c['device_gb'] * 100)}%). Risk of OOM under load.")
        else:
            style_name = "VerdictErr"
            icon = "DOES NOT FIT"
            # Against device capacity, and against a board count the reader
            # can act on. Measured per board while dividing per device, this
            # printed "Over by 0 GiB" on a configuration that does not fit.
            need = math.ceil(c["total_gb"] / (c["device_gb"] * 0.9))
            per_board = c["device_count"] // cfg["n_gpu"] if cfg["n_gpu"] else 1
            msg = (f"Over by {fmt_gb(c['per_total'] - c['device_gb'])} per device. "
                   f"Need {math.ceil(need / max(per_board, 1))}+ boards or lower precision.")
        story.append(Paragraph(f"[{icon}] {msg}", self.styles[style_name]))
        story.append(Spacer(1, 4*mm))

        # ---- Model Specification ----
        story.append(Paragraph("Model specification", self.styles["SectionHead"]))
        prec_label = PREC_LABELS.get(cfg["bpp"], f"{cfg['bpp']} B/param")
        kv_label = "FP8 (1 byte)" if cfg.get("kv_bpp", 2) < 2 else "BF16 (2 bytes)"
        arch_type = "MoE" if c["is_moe"] else "Dense"
        model_data = [
            ["Model", model_name],
            ["HuggingFace ID", cfg.get("hf_model", "N/A")],
            ["Parameters", f"{cfg['params']}B total" + (f" ({cfg['active']}% active per token)" if c["is_moe"] else "")],
            ["Architecture", arch_type + (f" — {cfg.get('shared_exp', 0)} shared expert(s)" if cfg.get("shared_exp", 0) else "")],
            ["Attention", attn_label(cfg)],
            ["Weight precision", prec_label],
            ["KV cache precision", kv_label],
            ["Layers", str(cfg["layers"])],
            ["KV heads / head dim", f"{cfg['kv_heads']} / {cfg['h_dim']}"],
        ]
        story.append(self._make_kv_table(model_data))
        story.append(Spacer(1, 3*mm))

        # ---- GPU Configuration ----
        story.append(Paragraph("GPU configuration", self.styles["SectionHead"]))
        # tp/dp, not cfg['n_gpu'] directly: above 8 GPUs this row used to say
        # "TP=12" while build_vllm_cmd(), printed a few sections later in the
        # same PDF, emitted --tensor-parallel-size 4 --data-parallel-size 3 —
        # one document contradicting itself.
        tp, dp = split_parallelism(device_count_for(cfg))
        parallelism_label = "Single device"
        if device_count_for(cfg) > 1:
            parallelism_label = f"Tensor parallel (TP={tp})" + (f" + data parallel (DP={dp})" if dp > 1 else "")
        gpu_data = [
            ["GPU model", gpu["name"]],
            ["GPU count", str(cfg["n_gpu"])],
            ["Total VRAM", f"{c['total_vram']} GB"],
            ["Interconnect", "NVLink" if cfg.get("nvlink") else "PCIe"],
            ["Memory bandwidth", f"{c['device_bw']:g} GB/s per device"],
            ["Parallelism", parallelism_label],
        ]
        story.append(self._make_kv_table(gpu_data))
        story.append(Spacer(1, 3*mm))

        # ---- VRAM Breakdown ----
        story.append(Paragraph("VRAM breakdown (per GPU)", self.styles["SectionHead"]))
        story.append(self._vram_bar())
        story.append(Spacer(1, 2*mm))
        story.append(self._make_metric_row([
            ("Weights", fmt_gb(c["weights_gb"])),
            ("KV cache", fmt_gb(c["kv_gb"])),
            ("Act + OH", fmt_gb(c["act_gb"] + c["total_oh"])),
            ("Total", fmt_gb(c["total_gb"])),
            ("Per GPU", fmt_gb(c["per_total"])),
        ]))
        story.append(Spacer(1, 3*mm))

        # ---- Capacity Limits ----
        story.append(Paragraph("Capacity limits", self.styles["SectionHead"]))
        cap_data = [
            ["Max context (1 user, 90% util)", fmt_k(c["max_ctx_1"]) + " tokens"],
            ["Max concurrent users @ 8K ctx", str(max(c["max_conc_8k"], 0))],
            ["Max concurrent users @ 4K ctx", str(max(c["max_conc_4k"], 0))],
            ["Free VRAM for KV cache", fmt_gb(c["free_kv"])],
            ["KV cache per token", f"{round(c['kv_bytes_per_tok'])} bytes"],
        ]
        story.append(self._make_kv_table(cap_data))
        story.append(Spacer(1, 3*mm))

        # ---- Throughput ----
        story.append(Paragraph("Throughput estimate", self.styles["SectionHead"]))
        tp_data = [
            ["Single-stream decode (1 user)", f"~{fmt_tok(c['single_tok'])} tokens/sec"],
            [f"Aggregate ceiling @ {c['eff_batch']} concurrent", f"~{fmt_tok(c['agg_tok'])} tokens/sec (upper bound)"],
            ["Observed range in practice", f"~{fmt_tok(c['agg_obs_lo'])}–{fmt_tok(c['agg_obs_hi'])} tokens/sec ({round(c['perf_obs_lo'] * 100)}–{round(c['perf_obs_hi'] * 100)}% of ceiling across measured benchmarks)"],
            ["Per user under that load", f"~{fmt_tok(c['per_user_load'])} tokens/sec"],
            ["Max batch at this context", f"{c['max_batch_kv']} sequences" + (" (below requested concurrency)" if c["batch_limited"] else "")],
            ["Est. time to first token", f"~{c['ttft_ms']} ms (at {fmt_k(cfg['ctx'])} context)"],
            ["Basis", f"Memory-bandwidth bound — {c['device_bw']:g} GB/s x {c['device_count']} device(s)"],
        ]
        story.append(self._make_kv_table(tp_data))
        story.append(Paragraph(
            "Throughput is a theoretical memory-bandwidth-bound estimate. Real numbers depend on "
            "batching strategy, attention implementation, quantization kernels, and workload mix.",
            self.styles["Small"]
        ))
        story.append(Spacer(1, 3*mm))

        # ---- Cost ----
        story.append(Paragraph("Cost estimate", self.styles["SectionHead"]))
        cost_data = [
            ["Provider tier", "Per GPU/hr", f"Total/hr ({cfg['n_gpu']}×)", "Monthly (730h)"],
            ["Hyperscaler (AWS/GCP/Azure)",
             f"${gpu['hyper']:.2f}",
             f"${c['hourly_hyper']:.2f}",
             f"${round(c['hourly_hyper']*730):,}"],
            ["Specialized (Lambda/CoreWeave)",
             f"${gpu['spec']:.2f}",
             f"${c['hourly_spec']:.2f}",
             f"${round(c['hourly_spec']*730):,}"],
            ["Spot / marketplace (Vast.ai)",
             f"${gpu['spot']:.2f}",
             f"${c['hourly_spot']:.2f}",
             f"${round(c['hourly_spot']*730):,}"],
        ]
        cost_table = Table(cost_data, colWidths=[55*mm, 30*mm, 30*mm, 35*mm])
        cost_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_FG2),
            ("TEXTCOLOR", (0, 1), (0, -1), C_FG),
            ("TEXTCOLOR", (1, 1), (-1, -1), C_FG),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -2), 0.3, C_BD),
            ("BACKGROUND", (3, 3), (3, 3), C_OK_BG),
        ]))
        story.append(cost_table)
        story.append(Paragraph(
            "Prices per GPU/hr as of mid-2026. Vary by region, commitment, and availability. "
            "Reserved instances typically 30-60% off hyperscaler on-demand. Spot can be interrupted.",
            self.styles["Small"]
        ))
        story.append(Spacer(1, 3*mm))

        # ---- vLLM Command ----
        story.append(Paragraph("vLLM deployment command", self.styles["SectionHead"]))
        cmd = build_vllm_cmd(cfg, c)
        for line in cmd.split("\n"):
            story.append(Paragraph(line, self.styles["CmdCode"]))
        story.append(Spacer(1, 3*mm))

        # ---- Notes ----
        story.append(Paragraph("Notes and assumptions", self.styles["SectionHead"]))
        notes = []
        notes.append(f"KV cache uses {'FP8 (1 byte/value)' if cfg.get('kv_bpp', 2) < 2 else 'BF16 (2 bytes/value)'} "
                      f"{'— enabled via --kv-cache-dtype fp8' if cfg.get('kv_bpp', 2) < 2 else '— default vLLM behavior'}.")
        notes.append("VRAM estimates include ~1.5 GB CUDA context overhead per device.")
        if device_count_for(cfg) > 1:
            notes.append(f"{'NVLink' if cfg.get('nvlink') else 'PCIe'} interconnect assumed. "
                         f"{'NVLink provides 600-900 GB/s bidirectional.' if cfg.get('nvlink') else 'PCIe (64-128 GB/s) loses 30-50% decode throughput vs NVLink.'}")
            notes.append(f"NCCL buffers add ~0.3 GB per GPU peer connection.")
        if cfg.get("shared_exp", 0):
            notes.append(f"{cfg['shared_exp']} shared expert(s) are always active and included in activation memory.")
        if dp > 1:
            # Same caveat index.html shows on screen (renderGPUCards' warning
            # banner) — repeated here because this PDF, not the screen, is
            # the artifact README.md calls procurement-ready and that gets
            # forwarded. A reader who only sees the PDF must not be able to
            # read "Per GPU: X GiB" next to --data-parallel-size and conclude
            # that's what actually fits.
            notes.append(f"Per-GPU VRAM above assumes weights sharded across all {device_count_for(cfg)} devices; the "
                         f"vLLM command below instead shards {tp}-way and replicates a full copy of the model "
                         f"across each of {dp} data-parallel groups, so the fit verdict on page 1 is optimistic. "
                         f"Reconciling this VRAM math with the real split is planned but not done yet.")
        notes.append("Parameter estimates from presets are approximate. Verify against the model's config.json.")
        notes.append("GPU prices are mid-2026 per-GPU/hr estimates across 3 tiers: hyperscaler (AWS/GCP/Azure), specialized (Lambda/CoreWeave/RunPod), spot/marketplace (Vast.ai). Reserved instances typically 30-60% off.")
        for n in notes:
            story.append(Paragraph(f"• {n}", self.styles["Small"]))
            story.append(Spacer(1, 1*mm))

        # ---- Build PDF ----
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            leftMargin=self.margin,
            rightMargin=self.margin,
            topMargin=20*mm,
            bottomMargin=18*mm,
        )
        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        return self.output_path


# =============================================================================
# CLI
# =============================================================================
# PRESETS keys whose short-hand differs from what compute() (and a raw JSON
# config) call them. Everything else in a preset passes through unchanged;
# name/hf are excluded because they're presentation metadata handled
# separately below (hf_model/model_name), and moe because compute() derives
# is_moe from active < 100 and never reads it — carrying it forward would
# just be a stray unused key in every cfg.
PRESET_RENAME = {"p": "params", "l": "layers", "kv": "kv_heads", "hd": "h_dim",
                  "a": "active", "se": "shared_exp"}
PRESET_EXCLUDE = {"hf", "name", "moe"}

def arch_fields(pd):
    """Translate a PRESETS entry into the cfg keys compute() reads.

    Default-allow, not default-deny: a key this function has never heard of —
    a new attention mode's parameter, a new architecture field, anything added
    to PRESETS in the future — passes through unchanged instead of needing
    this function (and every builder that calls it) edited first. That
    inversion is the actual fix for how attn/swa_win/swa_local/mla_dim, and
    separately max_ctx, went missing: naming four keys to forward is a
    whitelist by another name, and drops the fifth one exactly the same way.
    """
    return {PRESET_RENAME.get(k, k): v for k, v in pd.items() if k not in PRESET_EXCLUDE}


def interactive_mode():
    print("\n=== LLM VRAM Planning Report Card ===\n")
    print("Available presets:")
    for i, (k, v) in enumerate(PRESETS.items()):
        print(f"  {i+1:2d}. {k:24s} — {v['name']}")
    choice = input("\nSelect preset number (or 'custom'): ").strip()

    if choice.lower() == "custom":
        arch = {
            "params": float(input("  Parameters (B): ")),
            "active": float(input("  MoE active % (100 for dense): ")),
            "layers": int(input("  Layers: ")),
            "kv_heads": int(input("  KV heads: ")),
            "h_dim": int(input("  Head dimension: ")),
            "shared_exp": int(input("  Shared experts (0 if none): ") or "0"),
        }
        hf_model = input("  HuggingFace model ID: ").strip() or "/opt/models/YourModel"
        model_name = input("  Display name: ").strip() or f"{arch['params']}B model"
    else:
        # Same construction path as from_cli_args/from_json: this is the
        # no-args default a first-time user hits, so it gets the preset's
        # attention fields (and anything future PRESETS entries add) the same
        # way they do, not a fourth hand-copied field list.
        idx = int(choice) - 1
        key = list(PRESETS.keys())[idx]
        preset_data = PRESETS[key]
        arch = arch_fields(preset_data)
        hf_model = preset_data["hf"]
        model_name = preset_data["name"]

    print("\nAvailable GPUs:")
    for i, (k, v) in enumerate(GPUS.items()):
        print(f"  {i+1:2d}. {k:14s} — {v['name']} ({v['bw']} GB/s, ${v['spot']}-${v['hyper']}/hr)")
    gpu_choice = int(input("\nSelect GPU number: ").strip()) - 1
    gpu_key = list(GPUS.keys())[gpu_choice]
    gpu = GPUS[gpu_key]

    n_gpu = int(input("GPU count [1]: ").strip() or "1")
    if n_gpu * (gpu.get("devices", 1) or 1) > 1 and supports_nvlink(gpu):
        nvlink = input("NVLink? [y/n, default y]: ").strip().lower() != "n"
    else:
        nvlink = False
        if n_gpu > 1:
            print(f"{gpu['name']} has no NVLink — assuming PCIe.")

    print("\nPrecision options:")
    prec_opts = [(2.0, "BF16"), (1.0, "FP8"), (0.5, "INT4/AWQ"), (0.63, "Q4_K_M"), (0.82, "Q6_K")]
    for i, (v, l) in enumerate(prec_opts):
        print(f"  {i+1}. {l} ({v} B/param)")
    prec_choice = int(input("Select [3]: ").strip() or "3") - 1
    bpp = prec_opts[prec_choice][0]

    kv_bpp = 1 if input("FP8 KV cache? [y/n, default n]: ").strip().lower() == "y" else 2
    ctx = int(input("Context length [8192]: ").strip() or "8192")
    conc = int(input("Concurrent requests [1]: ").strip() or "1")

    return {
        **arch, "bpp": bpp, "ctx": ctx, "conc": conc,
        "n_gpu": n_gpu, "gpu": gpu, "nvlink": nvlink, "vendor": gpu["vendor"],
        "kv_bpp": kv_bpp, "hf_model": hf_model, "model_name": model_name,
    }


# Keys that describe the deployment request, not the model's architecture.
# Both from_json branches resolve these explicitly, with their own defaults,
# regardless of what the JSON says — raw's "gpu" is a lookup string, not the
# resolved dict compute() needs, for instance — so they're excluded from the
# architecture overlay below rather than merged from raw. Everything else in
# the JSON is a deliberate architecture/runtime override and applies
# unconditionally, including keys the selected preset never defines.
# Restricting the overlay to "keys the preset already defines" was the
# previous, broken version of this — it let attn flip to mla while dropping
# mla_dim entirely. validate_arch() below now refuses an attn override that
# arrives without the parameter that gives it meaning, rather than letting it
# quietly compute a confident zero.
REQUEST_KEYS = {"preset", "gpu", "bpp", "ctx", "conc", "n_gpu", "nvlink", "kv_bpp"}

# Expected type for each value that can now reach cfg from a raw JSON config
# without passing through a typed CLI flag or a Python literal in PRESETS.
# Checked once, right before cfg leaves this function, because a bad type
# reaches compute() unfiltered on purpose — that is what default-allow means —
# and a bare TypeError several frames deep names no key and helps nobody.
# bool is deliberately not in here as its own case: it is a subclass of int in
# Python, so isinstance(True, int) is True, and JSON authors routinely write
# 1/0 for true/false — both are handled explicitly in validate_arch() instead
# of by isinstance() alone, which gets both wrong in opposite directions.
ARCH_TYPES = {
    "params": (int, float), "active": (int, float), "layers": int,
    "kv_heads": int, "h_dim": int, "shared_exp": (int, float),
    "attn": str, "swa_win": (int, float), "swa_local": (int, float),
    "mla_dim": (int, float), "max_ctx": (int, float),
    "shared_prefix": (int, float), "prefix_caching": bool,
    "ctx": (int, float), "conc": (int, float), "n_gpu": (int, float),
}

def validate_arch(cfg):
    """Raise a clear, key-named TypeError for a wrong-typed or incomplete cfg
    value instead of letting it reach compute() and fail anonymously several
    frames deep — or, worse, not fail at all and silently report a confident
    zero."""
    for key, types_ in ARCH_TYPES.items():
        if key not in cfg:
            continue
        val = cfg[key]
        if types_ is bool:
            # Accept an actual bool or the 0/1 a JSON author reaches for
            # first; reject anything else, including other ints.
            ok = isinstance(val, bool) or val in (0, 1)
            want = "bool (or 0/1)"
        else:
            # bool is an int subclass — exclude it explicitly, or a stray
            # true/false silently passes as a valid layer count.
            ok = isinstance(val, types_) and not isinstance(val, bool)
            want = " or ".join(t.__name__ for t in types_) if isinstance(types_, tuple) else types_.__name__
        if not ok:
            raise TypeError(f"cfg[{key!r}] must be {want}, got {type(val).__name__}: {val!r}")

    # An attention mode's own parameter must actually be given, not just
    # implied by attn's value. This is the N1 bug's exact shape: attn flips
    # to mla/swa correctly, but mla_dim/swa_win/swa_local stay at their zero
    # default, and compute() silently reports an empty KV cache and fits=True
    # instead of complaining that the config is incomplete.
    attn = cfg.get("attn", "standard")
    if attn == "mla" and not cfg.get("mla_dim", 0) > 0:
        raise TypeError(f"cfg['attn']='mla' requires cfg['mla_dim'] > 0, got {cfg.get('mla_dim', 0)!r}")
    if attn == "swa":
        if not cfg.get("swa_win", 0) > 0:
            raise TypeError(f"cfg['attn']='swa' requires cfg['swa_win'] > 0, got {cfg.get('swa_win', 0)!r}")
        if not cfg.get("swa_local", 0) > 0:
            raise TypeError(f"cfg['attn']='swa' requires cfg['swa_local'] > 0, got {cfg.get('swa_local', 0)!r}")

    # n_gpu passes the generic (int, float) check above, but a fractional GPU
    # count is physically meaningless, and vLLM rejects it loudly at startup
    # (--tensor-parallel-size 3.7 fails to parse) — which is the correct
    # failure mode. Truncating it instead (as an earlier version of this
    # commit did, inside split_parallelism()) turns that loud rejection into
    # a silently under-sharded config: the VRAM math keeps dividing by the
    # untruncated float while the emitted command shards by the truncated
    # integer, so the report's own per-GPU figure understates what the
    # command it prints actually needs — 23% low at n_gpu=3.7. Rejecting it
    # here means split_parallelism()'s own truncation is a defensive guard
    # that should never actually be reached from a JSON config.
    n_gpu = cfg.get("n_gpu")
    if isinstance(n_gpu, (int, float)) and not isinstance(n_gpu, bool) and n_gpu != int(n_gpu):
        raise TypeError(f"cfg['n_gpu'] must be a whole number, got {n_gpu!r}")

    return cfg


def from_json(path):
    with open(path) as f:
        raw = json.load(f)
    gpu_key = raw.get("gpu", DEFAULT_GPU_KEY)
    gpu = GPUS.get(gpu_key, GPUS[DEFAULT_GPU_KEY])
    preset = raw.get("preset")
    if preset and preset in PRESETS:
        pd = PRESETS[preset]
        cfg = arch_fields(pd)
        # An explicit value in the JSON overrides the preset's — the same rule
        # ctx/conc/kv_bpp/etc. already follow below. Selecting a preset sets
        # defaults, it doesn't lock them.
        cfg.update({k: v for k, v in raw.items() if k not in REQUEST_KEYS})
        cfg.update({
            "bpp": raw.get("bpp", 0.5), "ctx": raw.get("ctx", 8192),
            "conc": raw.get("conc", 1), "n_gpu": raw.get("n_gpu", 1),
            "gpu": gpu, "nvlink": nvlink_for(gpu, raw.get("nvlink", True)),
            # Not a request field: the vendor is whatever the chosen card is,
            # and it selects the PERF constants the throughput math runs on.
            # Letting a JSON override it would mean NVIDIA's MBU on an AMD card.
            "vendor": gpu["vendor"],
            "kv_bpp": raw.get("kv_bpp", 2),
            # Same "raw wins" rule as everything else in this block — this
            # tool's origin story is an air-gapped deployment, and "pick a
            # preset, point it at my local weights" is the obvious thing to
            # write. Defaulting to the preset unconditionally silently served
            # a HuggingFace id that needs network access to resolve instead.
            "hf_model": raw.get("hf_model", pd["hf"]),
            "model_name": raw.get("model_name", pd["name"]),
        })
        return validate_arch(cfg)

    # No preset — cfg is built straight from the user's own JSON. Its keys
    # already use compute()'s names (params, layers, attn, ...), so unlike a
    # PRESETS entry there is nothing to rename: copy the config through and
    # layer the well-known defaults on top. A field the user sets that this
    # function has never heard of reaches compute() unchanged, same as above.
    cfg = dict(raw)
    for required in ("params", "layers", "kv_heads"):
        if required not in cfg:
            raise KeyError(required)
    cfg["gpu"] = gpu
    cfg["vendor"] = gpu["vendor"]
    cfg.setdefault("active", 100)
    cfg.setdefault("h_dim", 128)
    cfg.setdefault("shared_exp", 0)
    cfg.setdefault("bpp", 0.5)
    cfg.setdefault("ctx", 8192)
    cfg.setdefault("conc", 1)
    cfg.setdefault("n_gpu", 1)
    cfg["nvlink"] = nvlink_for(gpu, cfg.get("nvlink", True))
    cfg.setdefault("kv_bpp", 2)
    cfg.setdefault("hf_model", "/opt/models/YourModel")
    cfg.setdefault("model_name", f"{cfg['params']}B model")
    return validate_arch(cfg)


def from_cli_args(args):
    preset = PRESETS.get(args.preset)
    gpu = GPUS.get(args.gpu, GPUS[DEFAULT_GPU_KEY])
    if preset:
        cfg = arch_fields(preset)
        cfg.update({
            "bpp": {"bf16":2,"fp8":1,"int4":0.5,"awq":0.5,"gptq":0.5,"q4km":0.63,"q6k":0.82,"q8":1.1}.get(args.prec, 0.5),
            "quant": {"bf16":"","fp8":"fp8","int4":"awq","awq":"awq","gptq":"gptq",
                      "q4km":"gguf","q6k":"gguf","q8":"gguf"}.get(args.prec, "awq"),
            "ctx": args.ctx, "conc": args.conc,
            "n_gpu": args.ngpu, "gpu": gpu,
            "nvlink": nvlink_for(gpu, not args.no_nvlink),
            "vendor": gpu["vendor"],
            "kv_bpp": 1 if args.fp8_kv else 2,
            "hf_model": preset["hf"], "model_name": preset["name"],
        })
        return cfg
    else:
        raise ValueError(f"Unknown preset: {args.preset}. Available: {', '.join(PRESETS.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM VRAM Planning Report Card (PDF)")
    parser.add_argument("--json", help="Path to JSON config file")
    parser.add_argument("--preset", help=f"Model preset: {', '.join(PRESETS.keys())}")
    parser.add_argument("--gpu", default=DEFAULT_GPU_KEY, help=f"GPU: {', '.join(GPUS.keys())}")
    parser.add_argument("--ngpu", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--prec", default="awq", help="Precision: bf16, fp8, awq, gptq, q4km, q6k, q8")
    parser.add_argument("--fp8-kv", action="store_true", help="Use FP8 KV cache")
    parser.add_argument("--no-nvlink", action="store_true", help="PCIe only (no NVLink)")
    parser.add_argument("--ctx", type=int, default=8192, help="Context length")
    parser.add_argument("--conc", type=int, default=1, help="Concurrent requests")
    parser.add_argument("-o", "--output", default="llm-vram-report.pdf", help="Output PDF path")
    args = parser.parse_args()

    if args.json:
        cfg = from_json(args.json)
    elif args.preset:
        cfg = from_cli_args(args)
    else:
        cfg = interactive_mode()

    output = args.output
    report = ReportCard(cfg, output)
    path = report.generate()
    print(f"\nReport generated: {path}")
