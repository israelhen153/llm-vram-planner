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
    "llama31-8b":       {"p":8,   "l":32, "kv":8,  "hd":128, "a":100, "moe":False, "se":0, "hf":"meta-llama/Llama-3.1-8B-Instruct",            "name":"Llama 3.1 8B"},
    "llama31-70b":      {"p":70,  "l":80, "kv":8,  "hd":128, "a":100, "moe":False, "se":0, "hf":"meta-llama/Llama-3.1-70B-Instruct",           "name":"Llama 3.1 70B"},
    "llama33-70b":      {"p":70,  "l":80, "kv":8,  "hd":128, "a":100, "moe":False, "se":0, "hf":"meta-llama/Llama-3.3-70B-Instruct",           "name":"Llama 3.3 70B"},
    "llama4-109b":      {"p":109, "l":48, "kv":8,  "hd":128, "a":16,  "moe":True,  "se":0, "hf":"meta-llama/Llama-4-Scout-109B-A17B",          "name":"Llama 4 Scout 109B MoE"},
    "qwen3-8b":         {"p":8,   "l":36, "kv":8,  "hd":128, "a":100, "moe":False, "se":0, "hf":"Qwen/Qwen3-8B",                               "name":"Qwen 3 8B"},
    "qwen3-30b":        {"p":30,  "l":48, "kv":8,  "hd":128, "a":10,  "moe":True,  "se":0, "hf":"Qwen/Qwen3-30B-A3B",                          "name":"Qwen 3 30B-A3B MoE"},
    "qwen25-coder-32b": {"p":32,  "l":64, "kv":8,  "hd":128, "a":100, "moe":False, "se":0, "hf":"Qwen/Qwen2.5-Coder-32B-Instruct",             "name":"Qwen 2.5 Coder 32B"},
    "qwen35-397b":      {"p":397, "l":94, "kv":8,  "hd":128, "a":4,   "moe":True,  "se":0, "hf":"Qwen/Qwen3.5-397B-A17B",                      "name":"Qwen 3.5 397B MoE"},
    "qwen3-coder-480b": {"p":480, "l":94, "kv":8,  "hd":128, "a":7,   "moe":True,  "se":0, "hf":"Qwen/Qwen3-Coder-480B-A35B-Instruct",         "name":"Qwen 3 Coder 480B MoE"},
    "gemma4-e4b":       {"p":4,   "l":26, "kv":4,  "hd":256, "a":100, "moe":False, "se":0, "hf":"google/gemma-4-E4B-it",                        "name":"Gemma 4 E4B"},
    "gemma4-26b":       {"p":26,  "l":26, "kv":16, "hd":128, "a":15,  "moe":True,  "se":0, "hf":"google/gemma-4-26B-A4B-it",                    "name":"Gemma 4 26B MoE"},
    "gemma4-31b":       {"p":31,  "l":34, "kv":16, "hd":128, "a":100, "moe":False, "se":0, "hf":"google/gemma-4-31B-it",                        "name":"Gemma 4 31B Dense"},
    "dsv3-671b":        {"p":671, "l":61, "kv":1,  "hd":128, "a":5,   "moe":True,  "se":1, "hf":"deepseek-ai/DeepSeek-V3",                      "name":"DeepSeek-V3 671B MoE"},
    "dsr1-671b":        {"p":671, "l":61, "kv":1,  "hd":128, "a":5,   "moe":True,  "se":1, "hf":"deepseek-ai/DeepSeek-R1",                      "name":"DeepSeek-R1 671B MoE"},
    "mistral-sm4-24b":  {"p":24,  "l":40, "kv":8,  "hd":128, "a":100, "moe":False, "se":0, "hf":"mistralai/Mistral-Small-4-24B-Instruct-2503",  "name":"Mistral Small 4 24B"},
    "mistral-lg-123b":  {"p":123, "l":88, "kv":8,  "hd":128, "a":100, "moe":False, "se":0, "hf":"mistralai/Mistral-Large-Instruct-2411",        "name":"Mistral Large 123B"},
}

GPUS = {
    "t4-16":       {"gb":16,  "bw":320,  "hyper":0.76, "spec":0.35, "spot":0.15, "name":"T4 16 GB"},
    "rtx4090-24":  {"gb":24,  "bw":936,  "hyper":0.75, "spec":0.45, "spot":0.29, "name":"RTX 4090 24 GB"},
    "a100-40":     {"gb":40,  "bw":1555, "hyper":3.67, "spec":1.29, "spot":0.63, "name":"A100 40 GB"},
    "l40s-48":     {"gb":48,  "bw":864,  "hyper":2.80, "spec":1.30, "spot":0.90, "name":"L40S 48 GB"},
    "a100-80":     {"gb":80,  "bw":2039, "hyper":4.50, "spec":1.79, "spot":0.99, "name":"A100 80 GB"},
    "h100-80":     {"gb":80,  "bw":3352, "hyper":12.30,"spec":3.99, "spot":2.25, "name":"H100 80 GB"},
    "rtxpro-96":   {"gb":96,  "bw":1800, "hyper":5.00, "spec":3.50, "spot":2.00, "name":"RTX PRO 6000 96 GB"},
    "b200-192":    {"gb":192, "bw":8000, "hyper":14.24,"spec":5.50, "spot":2.12, "name":"B200 192 GB"},
}

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


def fmt_gb(gb):
    if gb < 0.01: return "0 GB"
    if gb >= 100: return f"{round(gb)} GB"
    if gb >= 10: return f"{gb:.1f} GB"
    return f"{gb:.2f} GB"

def fmt_k(v):
    return f"{round(v/1024)}K" if v >= 1024 else str(v)

def fmt_tok(t):
    return f"{t/1000:.1f}K" if t >= 1000 else str(round(t))


# =============================================================================
# VRAM Computation
# =============================================================================
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
    nvlink = cfg.get("nvlink", True)
    kv_bpp = cfg.get("kv_bpp", 2)
    is_moe = active_pct < 100

    weights_gb = (params * 1e9 * bpp) / 1e9
    kv_bytes_per_tok = 2 * layers * kv_heads * h_dim * kv_bpp
    total_tokens = ctx * conc
    kv_gb = (kv_bytes_per_tok * total_tokens) / 1e9
    active_p = params * (active_pct / 100)
    shared_p = params * 0.02 * shared_exp if is_moe and shared_exp > 0 else 0
    total_active_p = active_p + shared_p
    act_gb = max((total_active_p * 1e9 * 2 * 0.01) / 1e9, 0.1)
    oh_per_gpu = 1.5
    nccl_oh = (0.3 if nvlink else 0.2) * (n_gpu - 1) if n_gpu > 1 else 0
    total_oh = oh_per_gpu * n_gpu + nccl_oh
    total_gb = weights_gb + kv_gb + act_gb + total_oh

    per_w = weights_gb / n_gpu
    per_kv = kv_gb / n_gpu
    per_a = act_gb / n_gpu
    per_oh = oh_per_gpu + ((0.3 if nvlink else 0.2) if n_gpu > 1 else 0)
    per_total = per_w + per_kv + per_a + per_oh
    total_vram = gpu["gb"] * n_gpu

    fixed_pg = per_w + per_a + per_oh
    free_kv = max((gpu["gb"] * 0.9 - fixed_pg) * n_gpu, 0)
    kv_per_tok_gb = kv_bytes_per_tok / 1e9
    max_ctx_1 = min(int(free_kv / kv_per_tok_gb), 131072) if kv_per_tok_gb > 0 else 0
    max_conc_8k = int(free_kv / (kv_per_tok_gb * 8192)) if kv_per_tok_gb > 0 else 0
    max_conc_4k = int(free_kv / (kv_per_tok_gb * 4096)) if kv_per_tok_gb > 0 else 0

    mem_bound = total_active_p * 1e9 * bpp
    raw_tok = (gpu["bw"] * 1e9 * n_gpu) / mem_bound if mem_bound > 0 else 0
    nv_penalty = 0.55 if (n_gpu > 1 and not nvlink) else (0.85 if n_gpu > 1 else 1.0)
    batch_boost = min(1 + math.log2(max(conc, 1)) * 0.15, 2.0)
    est_tok = round(raw_tok * nv_penalty * batch_boost)
    hourly_hyper = gpu["hyper"] * n_gpu
    hourly_spec = gpu["spec"] * n_gpu
    hourly_spot = gpu["spot"] * n_gpu

    fits = per_total <= gpu["gb"]
    comfortable = per_total <= gpu["gb"] * 0.9

    return {
        "weights_gb": weights_gb, "kv_gb": kv_gb, "act_gb": act_gb,
        "total_oh": total_oh, "total_gb": total_gb,
        "per_w": per_w, "per_kv": per_kv, "per_a": per_a, "per_oh": per_oh,
        "per_total": per_total, "total_vram": total_vram,
        "free_kv": free_kv, "kv_per_tok_gb": kv_per_tok_gb,
        "kv_bytes_per_tok": kv_bytes_per_tok,
        "max_ctx_1": max_ctx_1, "max_conc_8k": max_conc_8k, "max_conc_4k": max_conc_4k,
        "est_tok": est_tok, "hourly_hyper": hourly_hyper, "hourly_spec": hourly_spec, "hourly_spot": hourly_spot,
        "is_moe": is_moe, "total_tokens": total_tokens,
        "fits": fits, "comfortable": comfortable,
    }


def build_vllm_cmd(cfg, comp):
    if not comp["fits"]:
        return "# Does not fit — increase GPUs, lower precision, or reduce context"
    hf = cfg.get("hf_model", "/opt/models/YourModel")
    parts = [f"vllm serve {hf} \\"]
    parts.append("    --host 0.0.0.0 --port 8000 \\")
    if cfg["n_gpu"] > 1:
        parts.append(f"    --tensor-parallel-size {cfg['n_gpu']} \\")
    if cfg["bpp"] <= 0.5:
        parts.append("    --quantization awq --dtype float16 \\")
    elif cfg["bpp"] <= 1:
        parts.append("    --dtype float16 \\")
    else:
        parts.append("    --dtype auto \\")
    if cfg.get("kv_bpp", 2) < 2:
        parts.append("    --kv-cache-dtype fp8 \\")
    if comp["is_moe"] and cfg["n_gpu"] > 1:
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
        gpu_gb = self.cfg["gpu"]["gb"]
        c = self.comp
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
            f"{' (NVLink)' if cfg.get('nvlink') and cfg['n_gpu'] > 1 else ''}"
            f"{' (PCIe)' if not cfg.get('nvlink') and cfg['n_gpu'] > 1 else ''}",
            self.styles["ReportSub"]
        ))

        # ---- Verdict ----
        if c["comfortable"]:
            style_name = "VerdictOK"
            icon = "FITS"
            msg = f"{fmt_gb(c['per_total'])} per GPU of {gpu['gb']} GB ({round(c['per_total']/gpu['gb']*100)}%). Headroom available."
        elif c["fits"]:
            style_name = "VerdictWarn"
            icon = "TIGHT"
            msg = f"{fmt_gb(c['per_total'])} per GPU of {gpu['gb']} GB ({round(c['per_total']/gpu['gb']*100)}%). Risk of OOM under load."
        else:
            style_name = "VerdictErr"
            icon = "DOES NOT FIT"
            need = math.ceil(c["total_gb"] / (gpu["gb"] * 0.9))
            msg = f"Over by {fmt_gb(c['per_total'] - gpu['gb'])} per GPU. Need {need}+ GPUs or lower precision."
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
            ["Weight precision", prec_label],
            ["KV cache precision", kv_label],
            ["Layers", str(cfg["layers"])],
            ["KV heads / head dim", f"{cfg['kv_heads']} / {cfg['h_dim']}"],
        ]
        story.append(self._make_kv_table(model_data))
        story.append(Spacer(1, 3*mm))

        # ---- GPU Configuration ----
        story.append(Paragraph("GPU configuration", self.styles["SectionHead"]))
        gpu_data = [
            ["GPU model", gpu["name"]],
            ["GPU count", str(cfg["n_gpu"])],
            ["Total VRAM", f"{c['total_vram']} GB"],
            ["Interconnect", "NVLink" if cfg.get("nvlink") else "PCIe"],
            ["Memory bandwidth", f"{gpu['bw']} GB/s per GPU"],
            ["Parallelism", f"Tensor parallel (TP={cfg['n_gpu']})" if cfg["n_gpu"] > 1 else "Single GPU"],
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
        ttft = round(1000 / c["est_tok"] * cfg["ctx"] / 1000) if c["est_tok"] > 0 else "N/A"
        tp_data = [
            ["Est. decode throughput", f"~{fmt_tok(c['est_tok'])} tokens/sec"],
            ["Est. time to first token", f"~{ttft} ms (at {fmt_k(cfg['ctx'])} context)"],
            ["Basis", f"Memory-bandwidth bound — {gpu['bw']} GB/s x {cfg['n_gpu']} GPU(s)"],
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
        notes.append("VRAM estimates include ~1.5 GB CUDA context overhead per GPU.")
        if cfg["n_gpu"] > 1:
            notes.append(f"{'NVLink' if cfg.get('nvlink') else 'PCIe'} interconnect assumed. "
                         f"{'NVLink provides 600-900 GB/s bidirectional.' if cfg.get('nvlink') else 'PCIe (64-128 GB/s) loses 30-50% decode throughput vs NVLink.'}")
            notes.append(f"NCCL buffers add ~0.3 GB per GPU peer connection.")
        if cfg.get("shared_exp", 0):
            notes.append(f"{cfg['shared_exp']} shared expert(s) are always active and included in activation memory.")
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
def interactive_mode():
    print("\n=== LLM VRAM Planning Report Card ===\n")
    print("Available presets:")
    for i, (k, v) in enumerate(PRESETS.items()):
        print(f"  {i+1:2d}. {k:24s} — {v['name']}")
    choice = input("\nSelect preset number (or 'custom'): ").strip()

    if choice.lower() == "custom":
        params = float(input("  Parameters (B): "))
        active = float(input("  MoE active % (100 for dense): "))
        layers = int(input("  Layers: "))
        kv_heads = int(input("  KV heads: "))
        h_dim = int(input("  Head dimension: "))
        shared_exp = int(input("  Shared experts (0 if none): ") or "0")
        hf_model = input("  HuggingFace model ID: ").strip() or "/opt/models/YourModel"
        model_name = input("  Display name: ").strip() or f"{params}B model"
        preset_data = None
    else:
        idx = int(choice) - 1
        key = list(PRESETS.keys())[idx]
        preset_data = PRESETS[key]
        params = preset_data["p"]
        active = preset_data["a"]
        layers = preset_data["l"]
        kv_heads = preset_data["kv"]
        h_dim = preset_data["hd"]
        shared_exp = preset_data["se"]
        hf_model = preset_data["hf"]
        model_name = preset_data["name"]

    print("\nAvailable GPUs:")
    for i, (k, v) in enumerate(GPUS.items()):
        print(f"  {i+1:2d}. {k:14s} — {v['name']} ({v['bw']} GB/s, ${v['spot']}-${v['hyper']}/hr)")
    gpu_choice = int(input("\nSelect GPU number: ").strip()) - 1
    gpu_key = list(GPUS.keys())[gpu_choice]
    gpu = GPUS[gpu_key]

    n_gpu = int(input("GPU count [1]: ").strip() or "1")
    nvlink = input("NVLink? [y/n, default y]: ").strip().lower() != "n" if n_gpu > 1 else False

    print("\nPrecision options:")
    prec_opts = [(2.0, "BF16"), (1.0, "FP8"), (0.5, "INT4/AWQ"), (0.63, "Q4_K_M"), (0.82, "Q6_K")]
    for i, (v, l) in enumerate(prec_opts):
        print(f"  {i+1}. {l} ({v} B/param)")
    prec_choice = int(input("Select [3]: ").strip() or "3") - 1
    bpp = prec_opts[prec_choice][0]

    kv_bpp = 1 if input("FP8 KV cache? [y/n, default n]: ").strip().lower() == "y" else 2
    ctx = int(input("Context length [8192]: ").strip() or "8192")
    conc = int(input("Concurrent requests [1]: ").strip() or "1")

    cfg = {
        "params": params, "active": active, "bpp": bpp,
        "layers": layers, "kv_heads": kv_heads, "h_dim": h_dim,
        "shared_exp": shared_exp, "ctx": ctx, "conc": conc,
        "n_gpu": n_gpu, "gpu": gpu, "nvlink": nvlink,
        "kv_bpp": kv_bpp, "hf_model": hf_model, "model_name": model_name,
    }
    return cfg


def from_json(path):
    with open(path) as f:
        raw = json.load(f)
    gpu_key = raw.get("gpu", "a100-40")
    gpu = GPUS.get(gpu_key, GPUS["a100-40"])
    preset = raw.get("preset")
    if preset and preset in PRESETS:
        pd = PRESETS[preset]
        return {
            "params": pd["p"], "active": pd["a"], "bpp": raw.get("bpp", 0.5),
            "layers": pd["l"], "kv_heads": pd["kv"], "h_dim": pd["hd"],
            "shared_exp": pd["se"], "ctx": raw.get("ctx", 8192),
            "conc": raw.get("conc", 1), "n_gpu": raw.get("n_gpu", 1),
            "gpu": gpu, "nvlink": raw.get("nvlink", True),
            "kv_bpp": raw.get("kv_bpp", 2), "hf_model": pd["hf"],
            "model_name": pd["name"],
        }
    return {
        "params": raw["params"], "active": raw.get("active", 100),
        "bpp": raw.get("bpp", 0.5), "layers": raw["layers"],
        "kv_heads": raw["kv_heads"], "h_dim": raw.get("h_dim", 128),
        "shared_exp": raw.get("shared_exp", 0), "ctx": raw.get("ctx", 8192),
        "conc": raw.get("conc", 1), "n_gpu": raw.get("n_gpu", 1),
        "gpu": gpu, "nvlink": raw.get("nvlink", True),
        "kv_bpp": raw.get("kv_bpp", 2),
        "hf_model": raw.get("hf_model", "/opt/models/YourModel"),
        "model_name": raw.get("model_name", f"{raw['params']}B model"),
    }


def from_cli_args(args):
    preset = PRESETS.get(args.preset)
    gpu = GPUS.get(args.gpu, GPUS["a100-40"])
    if preset:
        return {
            "params": preset["p"], "active": preset["a"],
            "bpp": {"bf16":2,"fp8":1,"int4":0.5,"q4km":0.63,"q6k":0.82,"q8":1.1}.get(args.prec, 0.5),
            "layers": preset["l"], "kv_heads": preset["kv"], "h_dim": preset["hd"],
            "shared_exp": preset["se"], "ctx": args.ctx, "conc": args.conc,
            "n_gpu": args.ngpu, "gpu": gpu, "nvlink": not args.no_nvlink,
            "kv_bpp": 1 if args.fp8_kv else 2,
            "hf_model": preset["hf"], "model_name": preset["name"],
        }
    else:
        raise ValueError(f"Unknown preset: {args.preset}. Available: {', '.join(PRESETS.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM VRAM Planning Report Card (PDF)")
    parser.add_argument("--json", help="Path to JSON config file")
    parser.add_argument("--preset", help=f"Model preset: {', '.join(PRESETS.keys())}")
    parser.add_argument("--gpu", default="a100-40", help=f"GPU: {', '.join(GPUS.keys())}")
    parser.add_argument("--ngpu", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--prec", default="int4", help="Precision: bf16, fp8, int4, q4km, q6k, q8")
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
