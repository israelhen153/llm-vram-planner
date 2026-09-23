# AMD GPU Specifications for VLLM Planner

**Read:** 2026-09-23, from AMD's own datasheets, product pages, architecture whitepapers
and ROCm documentation. Supersedes the 2026-07-31 table this file used to hold (see
"What changed" below).

**Purpose:** the source for the five AMD rows in `data/gpus.json`. Every figure below is
AMD's own; secondary sites were used only to find pages, never as a source.

**Convention:** the catalog stores **board** figures, the unit you buy, price and rack, and
the engine divides by `devices` for anything per device. Bandwidth is the most precise
figure AMD itself prints: where a claim footnote gives the exact value (e.g. "1.6384 TB/s"),
that value; where AMD prints only a rounded one (MI325X's "6 TB/s"), that one. TFLOPS are
**dense** (no structured sparsity), matching the catalog's NVIDIA rows.

## Specifications

| GPU | Arch / LLVM target | VRAM | Bandwidth (GB/s) | Dense BF16 TFLOPS | FP8 in hardware | Devices per board | Form |
|---|---|---|---|---|---|---|---|
| MI210 | CDNA2 / gfx90a | 64 GB HBM2e | 1638.4 | 181.0 | No | 1 | PCIe card |
| MI250X | CDNA2 / gfx90a | 128 GB HBM2e (64 GB per GCD) | 3276.8 | 383.0 for the board (≈191.5 per GCD) | No | 2 | OAM module |
| MI300X | CDNA3 / gfx942 | 192 GB HBM3 | 5325 | 1307.4 | Yes (2614.9 dense FP8) | 1 | OAM module |
| MI325X | CDNA3 / gfx942 | 256 GB HBM3E | 6000 | 1307.4 | Yes (2614.9 dense FP8) | 1 | OAM module |
| Radeon RX 7900 XTX | RDNA3 / gfx1100 | 24 GB GDDR6 | 960 | 123 | No | 1 | Consumer card |

### Where each figure comes from

**MI210.** Product page: "Dedicated Memory Size 64 GB … HBM2e", "Peak Memory Bandwidth 1.6 TB/s",
"Peak Half Precision (FP16) Performance 181 TFLOPs … Peak bfloat16 181 TFLOPs", "GPU Form Factor
PCIe® Add-in Card". CDNA2 whitepaper footnote MI200-42: "MI210 memory bus interface is 4,096 bits
and memory data rate is 3.20 Gbps for total memory bandwidth of 1.6384 TB/s"; its product table
gives one GCD and "Peak FP16/BF16 181.0 TF". ROCm GPU specs: "MI210 | CDNA2 | gfx90a".

**MI250X.** Product page: "Dedicated Memory Size 128 GB … HBM2e", "Peak Memory Bandwidth 3.2 TB/s",
"Peak Half Precision (FP16) Performance 383 TFLOPs … Peak bfloat16 383 TFLOPs", "GPU Form Factor
OAM Module". MI200 datasheet footnote MI200-07: "memory bus interface is 4,096 bits times 2 die and
memory data rate is 3.20 Gbps for total memory bandwidth of 3.2768 TB/s". ROCm's MI250 page: the
OAM package "consists of two GCDs, each of which constitutes one GPU device in the system". See the
next section for the TFLOPS.

**MI300X.** Product page: "Dedicated Memory Size 192 GB … HBM3", "Peak Memory Bandwidth 5.3 TB/s"
with footnote MI300-05A ("8,192 bits memory bus interface * 5.2 Gbps memory data rate/8 … 5.325
TB/s"), and footnote MI300-18: "1307.4 TFLOPS peak theoretical half precision (FP16), 1307.4 TFLOPS
peak theoretical Bfloat16" and "2614.9 TFLOPS peak theoretical 8-bit precision (FP8)". The product
page also prints the sparsity figures ("…with Structured Sparsity 2.61 PFLOPs"), which are not
used. ROCm GPU specs: "MI300X | CDNA3 | gfx942". By default (SPX mode) its eight XCDs are one
device with 192 GB.

**MI325X.** Product page: "Dedicated Memory Size 256 GB … HBM3E", "Peak Memory Bandwidth 6 TB/s"
(footnote MI325-001A: "6 TB/s GPU peak theoretical memory bandwidth"; AMD prints no finer figure),
and footnote MI325-002: "1307.4 TFLOPS peak theoretical half precision (FP16), 1307.4 TFLOPS … (BF16)"
and "2614.9 TFLOPS peak theoretical 8-bit precision (FP8)". ROCm GPU specs: "MI325X | CDNA3 | gfx942".

**Radeon RX 7900 XTX.** Product page: "Max Memory Size 24 GB … Memory Type GDDR6", "Memory
Bandwidth Up to 960 GB/s" (not the "Effective Memory Bandwidth Up to 3500 GB/s" line, which counts
Infinity Cache hits), and "Peak Half Precision (FP16 Matrix) Performance 123 TFLOPs" beside "Peak
Half Precision (FP16 Vector) Performance 61.4 TFLOPs". The matrix figure is the WMMA path: AMD's
GPUOpen article on WMMA for RDNA3 gives 512 FLOPS/clock/CU for both FP16 and BF16, and 96 CU ×
2.5 GHz × 512 = 122.9. It does not depend on dual-issue; the 61.4 vector figures do. That BF16 runs
at the same rate is read from the GPUOpen table, not printed as a TFLOPS figure. ROCm GPU specs:
"Radeon RX 7900 XTX | RDNA3 | gfx1100".

## MI250X: 383 TFLOPS is the whole board

The 2026-07-31 version of this file read 383 as a *per-OAM* figure and doubled it to 766 "for
the module", while the catalog planned to read it as the whole two-GCD board. The two readings
differ by 2×. Resolved from AMD's own documents on 2026-09-23. **383.0 is the board, about 191.5
per GCD, and 766 was wrong.**

- **MI200 datasheet, footnote MI200-01:** "for the AMD Instinct™ MI250X (128GB HBM2e OAM module)
  accelerator at 1,700 MHz … 383.0 TFLOPS peak theoretical half precision (FP16)". The figure is
  stated for the OAM module.
- **CDNA2 whitepaper product table:** "Graphics Compute Die (GCD) | 1 | 2 | 2 … Compute Units | 104
  CU | 208CU | 220CU … Peak FP16/BF16 | 181.0 TF | 362.1 TF | 383.0 TF" for MI210, MI250 and MI250X.
- **ROCm's MI250 page:** "The third column lists the theoretical peak performance of the OAM
  module", giving 362.1 for the MI250's matrix FP16. The same convention gives 383.0 for the MI250X.
- **Arithmetic:** 220 CU × 1.7 GHz × 1024 FLOP/clock/CU = 383.0 TFLOPS. Per GCD it is
  110 × 1.7 × 1024 = 191.5. The single-GCD MI210 is 104 × 1.7 × 1024 = 181.0, its published
  figure. The three agree only if 383 is the two-GCD total.

The catalog row is therefore one row with `devices: 2` and the board's 128 GB, 3276.8 GB/s and
383 TFLOPS, and each device the runtime sees gets 64 GB, 1638.4 GB/s and 191.5 TFLOPS.

## FP8

ROCm's precision-support table lists float8 (E4M3 and E5M2) matrix-core support as ✅ on CDNA3
and RDNA4 and ❌ on CDNA2 and RDNA3. The MI300X datasheet says so directly for the older parts:
"The MI200 Series does not support TF32, FP8, or sparsity". So `caps.fp8` is true for MI300X
and MI325X, and false for MI210, MI250X and the RX 7900 XTX. Whether vLLM will *run* an FP8
model on each card is a separate question, recorded for the ROCm guidance unit.

## Interconnect

AMD calls its GPU-to-GPU link Infinity Fabric.

- **MI250X.** Infinity Fabric is native to the OAM, not a bridge. The datasheet (MI200-13) gives
  "up to eight links providing up to 800GB/s peak aggregate theoretical GPU (P2P) transport rate
  bandwidth performance per GPU OAM card". ROCm's MI250 page gives "200 GB/sec peak transfer
  bandwidth between the two GCDs of an OAM".
- **MI300X and MI325X.** Both are sold as an 8-GPU platform on a Universal Base Board (UBB 2.0),
  with "Scale-up Infinity Fabric™ Links 7x 128 GB/s" to the other seven GPUs.
- **MI210.** Infinity Fabric only through an optional bridge card joining two or four GPUs (up to
  300 GB/s peer to peer); otherwise PCIe 4.0. The catalog's `form` is `pcie`.
- **RX 7900 XTX.** None.

## What changed since 2026-07-31

- **MI250X TFLOPS:** 383 for the board, not 766. The DISPUTED marker this file carried is gone.
- **MI300X bandwidth:** 5325 GB/s is AMD's own footnote figure (MI300-05A), not a derivation.
- **RX 7900 XTX TFLOPS:** 123 is AMD's own "FP16 Matrix" figure, where the old table cited only
  TechPowerUp.
- **New columns:** LLVM target, FP8 support, devices per board and interconnect.

## Newer parts, out of scope

MI350X and MI355X (CDNA4) are now rentable: Oracle announced MI355X general availability on
2025-10-14, and Vultr offers MI355X. They are candidates for v1.2 and are not in the catalog.

## Sources (read 2026-09-23)

- MI210 product page: https://www.amd.com/en/products/accelerators/instinct/mi200/mi210.html
- MI250X product page: https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html
- MI200 datasheet (PDF): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instinct-mi200-datasheet.pdf
- CDNA2 architecture whitepaper (PDF): https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf
- MI300X product page: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
- MI300X data sheet (PDF): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
- CDNA3 architecture whitepaper (PDF): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
- MI325X product page: https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
- MI325X datasheet (PDF): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf
- RX 7900 XTX product page: https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html
- ROCm GPU architecture specs: https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
- ROCm MI250 architecture page: https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi250.html
- ROCm precision support: https://rocm.docs.amd.com/en/latest/reference/precision-support.html
- MI300X partitioning (SPX default): https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/mi300x/overview.html
- GPUOpen, WMMA on RDNA3: https://gpuopen.com/learn/wmma_on_rdna3/

**Who read what.** A research pass read every figure above from these pages. Before any of it
entered the catalog, a second reader re-read the two that decide the disputed questions: ROCm's
MI250 page (its peak column is the OAM module's, and each GCD is one device) and ROCm's precision
table (FP8 is absent on CDNA2 and RDNA3). The release's cold check re-verifies every figure
against these sources again.
