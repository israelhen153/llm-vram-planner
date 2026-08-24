# AMD GPU Specifications for VLLM Planner

**Date:** 2026-07-31

**Purpose:** Source data for v1.1.0 AMD support — see ROADMAP.md

**Critical Note on TFLOPS Reporting:**
This table reports **dense (not sparse) TFLOPS** for BF16/FP16 matrix operations. AMD datasheets often headline sparsity-enhanced numbers; we use the dense figures for conservative planning. Where a source does not explicitly distinguish dense vs. sparse, the entry is flagged as unverified.

## Specifications Table

| GPU | VRAM (GB) | Memory Type | Bandwidth (GB/s) | Dense BF16/FP16 TFLOPS | TDP (W) | Sources |
|-----|-----------|-------------|------------------|------------------------|---------|---------|
| MI210 | 64 | HBM2e | 1638.4 | 181 | 300 | [AMD MI210 Brochure](https://www.amd.com/en/products/accelerators/instinct/mi200/mi210.html), [TechPowerUp MI210 News](https://www.techpowerup.com/293166/amd-introduces-instinct-mi210-data-center-accelerator-for-exascale-class-hpc-and-ai-in-a-pcie-form-factor) |
| MI250X | 128 | HBM2e | 3276.8 | 383* | 500–560 | [AMD MI250X Product Page](https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html), [AMD MI200 Datasheet (PDF)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instinct-mi200-datasheet.pdf), [TechPowerUp MI250X Specs](https://www.techpowerup.com/gpu-specs/radeon-instinct-mi250x.c3837) |
| MI300X | 192 | HBM3 | 5325 | 1307.4 | 750 | [AMD MI300X Datasheet (PDF)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf), [AMD MI300X Product Page](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) |
| MI325X | 256 | HBM3e | 6000 | 1307.4 | 1000 | [AMD MI325X Datasheet (PDF)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf), [AMD MI325X Product Page](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html), [TechPowerUp MI325X Launch](https://www.techpowerup.com/327553/amd-launches-instinct-mi325x-accelerator-for-ai-workloads-256-gb-hbm3e-memory-and-2-6-petaflops-fp8-compute) |
| Radeon RX 7900 XTX | 24 | GDDR6 | 960 | 123 | 355 | [AMD Radeon RX 7900 XTX Product Page](https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html), [TechPowerUp RX 7900 XTX Specs](https://www.techpowerup.com/gpu-specs/xfx-mercury-magair-rx-7900-xtx.b11842), [TechPowerUp Review](https://www.techpowerup.com/review/amd-radeon-rx-7900-xtx/) |

*: MI250X TFLOPS (383) is per OAM (Accelerator Module). MI250X is a dual-OAM module, so aggregate system TFLOPS = 2 × 383 = 766 TFLOPS dense FP16/BF16.

## Caveats and Unverified Numbers

1. **MI210 — All verified from primary sources**
   - VRAM (64 GB), memory type (HBM2e), and bandwidth (1638.4 GB/s) confirmed from AMD official brochure and product page.
   - FP16 TFLOPS (181) confirmed from AMD specifications and TechPowerUp news.
   - TDP (300 W) confirmed from multiple sources.

2. **MI250X — Most verified; TFLOPS per-OAM caveat**
   - VRAM (128 GB), memory type (HBM2e), and bandwidth (3276.8 GB/s) confirmed from AMD datasheets and TechPowerUp.
   - FP16/BF16 TFLOPS (383) is reported per OAM from AMD datasheet; MI250X is a dual-OAM design, so total aggregate = 766 TFLOPS.
   - TDP (500–560 W) range confirmed from AMD datasheet.

3. **MI300X — All verified from primary sources**
   - VRAM (192 GB), memory type (HBM3), and bandwidth (5325 GB/s) confirmed from AMD MI300X datasheet.
   - Dense FP16/BF16 TFLOPS (1307.4) explicitly confirmed as dense (not sparse) in AMD official datasheet and announcements.
   - TDP (750 W) confirmed from AMD datasheet and Lenovo press references.

4. **MI325X — All verified from primary sources**
   - VRAM (256 GB), memory type (HBM3e), and bandwidth (6000 GB/s) confirmed from AMD MI325X datasheet and product page.
   - Dense BF16 TFLOPS (1307.4) explicitly stated in AMD announcements; matches MI300X per-GPU (same chiplet design, same core count).
   - TDP (1000 W) confirmed from AMD announcements and datasheets.

5. **Radeon RX 7900 XTX — Partial secondary source**
   - VRAM (24 GB) and memory type (GDDR6) confirmed from AMD official product page.
   - Memory bandwidth (960 GB/s) confirmed from multiple sources (AMD + TechPowerUp).
   - Dense FP16 matrix TFLOPS (123) sourced from TechPowerUp GPU specs database. This is labeled as "FP16 Matrix" (vs. FP16 vector at 61.4 TFLOPS), confirming matrix operation performance. **Secondary source; not independently verified against AMD datasheet.**
   - TDP (355 W) confirmed from TechPowerUp and AMD product page.

## Notes for Planning

- **MI210** and **MI250X** are older CDNA2 generation; primarily for HPC reference or legacy deployments.
- **MI300X** and **MI325X** are current-generation CDNA3, with significant TFLOPS increase (~7x vs. MI250X per GPU).
- **Radeon RX 7900 XTX** is a consumer/prosumer GPU with 8x lower FP16 TFLOPS than MI325X; included for reference but not recommended for LLM inference at scale.
- All Instinct GPUs (MI2xx, MI3xx series) support 8 GPU interconnect via AMD Infinity Fabric for distributed inference.
