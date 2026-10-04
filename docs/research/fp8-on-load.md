Research pass by Fable, read 2026-10-02 against vLLM v0.30.0 (tmp/research/R2-fp8-on-load.md), kept below as written.
T4 follow-up checked by Opus against the v0.30.0 tag on 2026-10-04 (tmp/research/R2-t4-fp8-verified-2026-10-04.md), kept as written after it.

# R2 — Does `--quantization fp8` on a BF16 repo load, and where? (vLLM v0.30.0)

**Read:** 2026-10-02. **Against:** vLLM tag `v0.30.0` = commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (tag date 2026-09-21), read from a sparse clone. Model files: `google/gemma-4-26B-A4B-it` at revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. Plugin: `vllm-project/vllm-gguf-plugin` at `e2b8ad532b8b5ea175100202c30430c1d2b5e6a8` (committed 2026-09-24).

**How it was read.** Code paths were traced at the tag; nothing was run on a GPU. "Traced" below means read, not executed. Every number in this file is pasted from a command output in this session; arithmetic on those numbers is labelled *derivation*.

**Why it matters.** The planner's default plan is preset `gemma4-26b` (`index.html:261`, `selected`) on the catalog's `default: true` card `a100-40` (`data/gpus.json`, `DEFAULT_GPU_KEY` at `index.html:690`) with precision AWQ (`index.html:273`, `<option value="0.5" selected data-q="awq">`). AWQ on a base BF16 repo does not load. This file answers whether FP8 can take its place.

---

## 1. Summary

| # | Question | Answer | Source | Confidence |
|---|---|---|---|---|
| 1 | Does `--quantization fp8` quantize BF16/FP16 weights at load? | **Yes.** With no `quantization_config` in the repo, `Fp8Config()` is built with `is_checkpoint_fp8_serialized=False`; Linear layers get `Fp8PerTensorOnlineLinearMethod`, MoE layers get `Fp8PerTensorOnlineMoEMethod`. Both quantize per-tensor to FP8 E4M3 during loading. MoE included. | `weight_utils.py:381-388`, `fp8.py:99-109,147-153,179-222`, `online/fp8.py:177-226,570-606`, docs `llm_compressor/fp8.md:144-148` | High |
| 2a | NVIDIA | Global gate = capability ≥ 75. **sm_75/80/86: loads as W8A16** (Marlin for Linear *and* MoE). **sm_89/90: W8A8.** **sm_100/120: W8A8** (torch `_scaled_mm` path certain; CUTLASS build coverage for sm_120 not read). | `config/vllm.py:831-841`, `marlin_utils_fp8.py:31-32`, `cuda.py:617-618`, `scaled_mm_entry.cu:145-159`, `oracle/fp8.py:79-93,429-450`, `marlin_moe.py:559-583`, `triton_moe.py:150-156` | High (≤ sm_90), Medium-High (sm_100/120) |
| 2b | AMD | **gfx942: loads** (W8A8, FNUZ dtype). **gfx90a: does not load** in the default configuration (no Linear kernel accepts it; `supports_fp8()` is misleadingly true). **gfx1100: does not load** (`supports_fp8()` false, no Linear kernel, no MoE backend). | `rocm.py:207-222,352-361,1009-1022`, `scaled_mm/rocm.py:76-90`, `scaled_mm/pytorch.py:31-49`, `aiter.py:39-51,136-151`, `triton_moe.py:150-156` | High (default env); AITER-on-gfx90a UNCONFIRMED |
| 3 | Peak memory while loading | **Per layer, not whole-model.** Online FP8 allocates weights on the `meta` device and materializes, loads and quantizes one layer at a time. Transient ≈ one layer's BF16 + its FP8 copy. | `online/fp8.py:115-156`, `online/moe_base.py:20-97`, `reload/layerwise.py:122-227,332-378`, `base_loader.py:74-91` | High (traced, not run) |
| 4 | Does Gemma 4 26B A4B load on one A100 40 GB with `--quantization fp8`? | **Yes, by trace.** Weights ≈ 25.3 GiB FP8-resident (24.7 GiB if the vision tower is quantized too) vs 48.07 GiB BF16; load-time transient ≈ +2.1 GiB; budget at the default `gpu_memory_utilization=0.92` leaves ≈ 9–11.5 GiB for KV cache and activations. Linear → Marlin W8A16, MoE → Marlin W8A16 (`gelu_tanh` is supported). | Safetensors headers (both shards), `cache.py:103`, `gemma4.py:363-372`, `activation.py:122-136` | Medium-High (traced; arithmetic is derivation) |
| 5 | GGUF at v0.30.0 | **Plugin only.** Docs: "GGUF support has migrated to OOT vllm-gguf-plugin"; no `gguf` quantization module, no `gguf` entry in `QUANTIZATION_METHODS`, no gguf loader in `model_loader/`. `repo_id:quant_type` is the documented form. `--tokenizer <base repo>` is **recommended**, not required (conversion "time-consuming and unstable"). | docs `gguf.md:6-19,34-45`; plugin README L3-5, L60-62, L93, L99-103; `quantization/__init__.py`; `ls model_loader/` | High; location of the `:quant_type` parser UNCONFIRMED |

---

## 2. Per-card verdicts (`data/gpus.json`, 17 rows)

Architecture receipts: NVIDIA compute capabilities from https://developer.nvidia.com/cuda-gpus (read 2026-10-02; the page is a flat list after tag-stripping, so a card's section header is the nearest preceding number). AMD `gfx` from the catalog row's `gfx` field, matching vLLM's own list: "MI200s (gfx90a), MI300 (gfx942), ... Radeon RX 7900 series (gfx1100/1101)" ([`gpu.rocm.inc.md` L16](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/getting_started/installation/gpu.rocm.inc.md#L16)).

| Row | Name | Arch (receipt) | `caps.fp8` | FP8 loads? | Linear path | MoE path | Note |
|---|---|---|---|---|---|---|---|
| `t4-16` | T4 16 GB | sm_75 ("7.5 NVIDIA T4") | false | **Conditional** | Marlin W8A16 | Marlin W8A16 | Passes the ≥75 gate. BF16 is not a Turing dtype; vLLM's dtype choice by capability (`cuda.py:256-259`) was not read in full → model dtype on T4 UNCONFIRMED |
| `l4-24` | L4 24 GB | sm_89 ("8.9 NVIDIA L4") | true | **Loads** | CUTLASS/torch W8A8 (CUDA ≥ 12.4 for CUTLASS) | Triton FP8 | |
| `rtx4090-24` | RTX 4090 24 GB | sm_89 (in the 8.9 block: "...RTX 2000 Ada GeForce RTX 4090...") | true | **Loads** | W8A8 | Triton FP8 | |
| `rtx5090-32` | RTX 5090 32 GB | sm_120 (in the 12.0 block: "...RTX PRO 2000 Blackwell GeForce RTX 5090...") | true | **Loads** | torch W8A8 certain; CUTLASS gate passes (≥90 branch), sm_120 build coverage not read | Triton FP8 | FlashInfer preferred if installed (≥100) |
| `a100-40` | A100 40 GB (default card) | sm_80 ("8.0 NVIDIA A100") | false | **Loads** | Marlin W8A16 | Marlin W8A16 | Triton rejects FP8 below cap 89; Marlin MoE accepts cap ≥ 7.5 |
| `rtx6000ada-48` | RTX 6000 Ada 48 GB | sm_89 ("8.9 ... NVIDIA RTX 6000 Ada") | true | **Loads** | W8A8 | Triton FP8 | |
| `l40s-48` | L40S 48 GB | sm_89 ("8.9 NVIDIA L4 NVIDIA L40 NVIDIA L40S") | true | **Loads** | W8A8 | Triton FP8 | |
| `a100-80` | A100 80 GB | sm_80 | false | **Loads** | Marlin W8A16 | Marlin W8A16 | as `a100-40` |
| `h100-80` | H100 80 GB | sm_90 ("9.0 NVIDIA GH200 NVIDIA H200 NVIDIA H100") | true | **Loads** | CUTLASS W8A8 (CUDA ≥ 12.0) | Triton FP8 | |
| `rtxpro-96` | RTX PRO 6000 96 GB | sm_120 ("12.0 NVIDIA RTX PRO 6000 Blackwell Server Edition") | true | **Loads** | as `rtx5090-32` | Triton FP8 | |
| `h200-141` | H200 141 GB | sm_90 | true | **Loads** | CUTLASS W8A8 | Triton FP8 | |
| `b200-192` | B200 192 GB | sm_100 ("10.0 NVIDIA GB200 NVIDIA B200") | true | **Loads** | FlashInfer (if installed) / CUTLASS / torch W8A8 | Triton FP8 (FlashInfer TRTLLM moved to front only for DeepEP block-FP8) | |
| `rx7900xtx-24` | RX 7900 XTX 24 GB | gfx1100 (catalog `gfx`) | false | **Does not load** | none: ROCm kernel "requires CDNA3+ (gfx942/gfx950) or RDNA4 (gfx12x)", torch path needs gfx942/950/12x/1250, Marlin "requires CUDA", AITER needs `VLLM_ROCM_USE_AITER=1` + aiter | none: `supports_fp8()` false → Triton rejects; Marlin requires CUDA → `NotImplementedError("No FP8 MoE backend ...")` | matches `docs/research/vllm-rocm.md` |
| `mi210-64` | MI210 64 GB | gfx90a | false | **Does not load** (default env) | none (CDNA version 2 refused; torch path refuses; AITER off by default) | Triton gate *would* accept (`supports_fp8()` is true for any gfx9) — moot, Linear fails first at `create_weights` | AITER with `VLLM_ROCM_USE_AITER=1` on gfx90a: UNCONFIRMED |
| `mi250x-128` | MI250X 128 GB | gfx90a | false | **Does not load** (default env) | as `mi210-64` | as `mi210-64` | |
| `mi300x-192` | MI300X 192 GB | gfx942 | true | **Loads** | ROCm FP8 kernel (CDNA 3) / torch; AITER if enabled | Triton FP8 (AITER first if enabled) | FP8 dtype is `float8_e4m3fnuz` (`is_fp8_fnuz`: "gfx94") |
| `mi325x-256` | MI325X 256 GB | gfx942 | true | **Loads** | as `mi300x-192` | as `mi300x-192` | |

`caps.fp8` in the catalog means FP8 tensor cores. It disagrees with "FP8 loads" on exactly the five rows where that is expected: A100 ×2 and T4 (no tensor cores, loads via Marlin W8A16), and nothing else; MI210/MI250X/RX 7900 XTX are false on both.

---

## 3. Evidence

### Q1 — Online quantization

**How `--quantization fp8` becomes an online config on a BF16 repo.**
- `Fp8Config.get_config_filenames()` returns `[]` — [`fp8.py` L151-153](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/fp8.py#L151-L153).
- `get_quant_config` ([`weight_utils.py` L381-388](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/model_loader/weight_utils.py#L381-L388)):
  > `possible_config_filenames = quant_cls.get_config_filenames()` / `# If the quantization config is not found, use the default config.` / `if not possible_config_filenames:` … `return maybe_compose_online_quantization(quant_cls())`
- That reaches the fallback only when the HF config carries no `quantization_config` (L268-309 return `quant_cls.from_config(hf_quant_config)` otherwise). Gemma 4 26B's `config.json` has no `quantization_config` (`top-level quantization_config: None`, read from the file at revision `4d7ae49`).
- `Fp8Config.__init__(self, is_checkpoint_fp8_serialized: bool = False, …)` — [`fp8.py` L99-109](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/fp8.py#L99-L109).

**Dispatch** ([`fp8.py` L179-222](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/fp8.py#L179-L222)):
> `if isinstance(layer, LinearBase):` … `if not self.is_checkpoint_fp8_serialized:` / `from vllm.model_executor.layers.quantization.online.fp8 import (Fp8PerTensorOnlineLinearMethod,)` / `online_method = Fp8PerTensorOnlineLinearMethod()` … `return online_method` / `else:` / `offline_method = Fp8LinearMethod(self)`
> `elif isinstance(layer, RoutedExperts):` … `if self.is_checkpoint_fp8_serialized:` / `return Fp8MoEMethod(self, layer)` / `else:` / `from vllm.model_executor.layers.quantization.online.fp8 import (Fp8PerTensorOnlineMoEMethod,)` / `return Fp8PerTensorOnlineMoEMethod(moe=layer.moe_config)`

The legacy `Fp8MoEMethod` no longer accepts BF16 checkpoints: `assert self.quant_config.is_checkpoint_fp8_serialized` in its `create_weights` ([`fp8.py` L571](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/fp8.py#L571)). The docstring above it still says "Also supports loading quantized FP16/BF16 model checkpoints" (L487-489); the assert is what runs.

**What the online methods do** ([`online/fp8.py`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/online/fp8.py)):
- L177-179: `class Fp8PerTensorOnlineLinearMethod(OnlineLinearBase):` `"""Online tensorwise FP8 linear quantization. Loads fp16/bf16 weights and quantizes them per-tensor during loading."""`
- L207-218 `process_weights_after_loading`: `amax = weight_amax(layer.weight).reshape(1)` … `weight_scale = _fp8_scale(amax)` / `qweight, _ = ops.scaled_fp8_quant(layer.weight, scale=weight_scale)` / `replace_parameter(layer, "weight", qweight.t().data)`
- L570-572: `class Fp8PerTensorOnlineMoEMethod(_Fp8OnlineMoEBase):` `"""Online tensorwise FP8 MoE quantization. Loads fp16/bf16 weights and quantizes them per-tensor during loading."""`
- L589-592, L604-610: `# If checkpoint is fp16, quantize in place.` / `fp8_dtype = current_platform.fp8_dtype()` / `w13 = torch.empty_like(layer.w13_weight, dtype=fp8_dtype)` / `w2 = torch.empty_like(layer.w2_weight, dtype=fp8_dtype)` … `for expert in range(layer.local_num_experts):` / `w13[expert, :, :], _ = ops.scaled_fp8_quant(layer.w13_weight[expert, :, :], scale=w13_scale[expert])`

**Docs at the tag** ([`llm_compressor/fp8.md` L144-148](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/quantization/llm_compressor/fp8.md#L144-L148)):
> "## Online Dynamic Quantization — Dynamic quantization of an original precision BF16/FP16 model to FP8 can be achieved with vLLM without any calibration data required. You can enable the feature by specifying `--quantization="fp8"` in the command line or setting `quantization="fp8"` in the LLM constructor." / "In this mode, all Linear modules (except for the final `lm_head`) have their weights quantized down to FP8_E4M3 precision with a per-tensor scale. Activations have their minimum and maximum values calculated during each forward pass to provide a dynamic per-tensor scale for high accuracy. As a result, latency improvements are limited in this mode."

The newer scheme names are documented separately ([`online.md` L1-6, L34-37](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/quantization/online.md#L1-L6)): "Online quantization lets you take a BF16/FP16 model and quantize its Linear and MoE weights to lower precision (such as FP8) at load time" … `vllm serve meta-llama/Llama-3.1-8B --quantization fp8_per_tensor`. *Derivation:* `--quantization fp8` on a BF16 repo and `--quantization fp8_per_tensor` reach the same two method classes (`fp8.py` L190-196/L216-222 vs `online/base.py` L88-104 dispatch tables); `fp8` is the legacy spelling and is still accepted.

**MoE receipts for Gemma 4 specifically.** `Gemma4MoE` builds its experts with `FusedMoEFactory(… quant_config=quant_config, prefix=f"{prefix}.experts", custom_routing_function=routing_function, activation="gelu_tanh",` ([`gemma4.py` L363-372](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/gemma4.py#L363-L372)); the factory creates a `RoutedExperts` (`# Create RoutedExperts instance BEFORE create_weights()` / `routed_experts_cls = RoutedExperts`, [`fused_moe/layer.py` L368-371](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/fused_moe/layer.py#L368-L371)), so the `isinstance(layer, RoutedExperts)` branch above applies.

### Q2 — Which hardware

**Gate 1, global minimum capability** ([`config/vllm.py` L831-841](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/vllm.py#L831-L841)):
> `capability_tuple = current_platform.get_device_capability()` / `if capability_tuple is not None:` / `capability = capability_tuple.to_int()` / `if capability < quant_config.get_min_capability():` / `raise ValueError(f"The quantization method {model_config.quantization} is not supported for the current GPU. Minimum capability: {quant_config.get_min_capability()}. Current capability: {capability}.")`

`Fp8Config.get_min_capability()` → `return 75` ([`fp8.py` L147-149](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/fp8.py#L147-L149)). On ROCm, `get_device_capability` returns torch's `(major, minor)` ([`rocm.py` L792-802](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/platforms/rocm.py#L792-L802); mapping examples L235-236: "gfx90a -> (9, 0)    gfx942 -> (9, 4) … gfx1100 -> (11, 0)"), all ≥ 75, so this gate never blocks an AMD card. L842-847 also require the model dtype to be in `get_supported_act_dtypes()` = `[torch.bfloat16, torch.half]` (`fp8.py` L143-145).

**Gate 2, platform list.** `current_platform.verify_quantization(self.quantization)` ([`config/model.py` L1340-1345](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/model.py#L1340-L1345)) raises only `if cls.supported_quantization and quant not in cls.supported_quantization` ([`interface.py` L984-991](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/platforms/interface.py#L984-L991)). ROCm's list contains `"fp8"` ([`rocm.py` L503-528](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/platforms/rocm.py#L503-L528)). A grep of `cuda.py` for `supported_quantization` returned nothing; the base-class default was not read (*assumption:* no CUDA restriction).

**Gate 3, Linear kernel selection.** Candidate order ([`kernels/linear/__init__.py` L421-438](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/kernels/linear/__init__.py#L421-L438)):
> CUDA: `FlashInferFP8ScaledMMLinearKernel, CutlassFP8ScaledMMLinearKernel, B12xTensorFP8ScaledMMLinearKernel, PerTensorTorchFP8ScaledMMLinearKernel, ChannelWiseTorchFP8ScaledMMLinearKernel, MarlinFP8ScaledMMLinearKernel, HummingFP8ScaledMMLinearKernel`
> ROCM: `AiterHipbMMPerTokenFp8ScaledMMLinearKernel, AiterPreshuffledPerTokenFp8ScaledMMLinearKernel, AiterPerTokenFp8ScaledMMLinearKernel, ROCmFP8ScaledMMLinearKernel, PerTensorTorchFP8ScaledMMLinearKernel, RowWiseTorchFP8ScaledMMLinearKernel, ChannelWiseTorchFP8ScaledMMLinearKernel`

`choose_scaled_mm_linear_kernel` walks the list and documents `Raises: ValueError: If no kernel can implement the given config.` (same file, docstring of the function). It is called from `create_weights` (`online/fp8.py` L215-223), i.e. at model construction.

Per-kernel gates:
- FlashInfer: `if not current_platform.is_cuda(): return False, "requires CUDA."` / `if not has_flashinfer(): return False, "requires FlashInfer to be installed."` / `if compute_capability is not None and compute_capability < 100: return False, "requires compute capability 100 and above."` — [`scaled_mm/flashinfer.py` L41-53](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py#L41-L53).
- CUTLASS: Python side `cutlass_fp8_supported()` → `ops.cutlass_scaled_mm_supports_fp8(capability)` ([`w8a8_utils.py` L11-18](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/utils/w8a8_utils.py#L11-L18)); C++ ([`scaled_mm_entry.cu` L145-159](https://github.com/vllm-project/vllm/blob/v0.30.0/csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu#L145-L159)):
  > `// CUTLASS FP8 kernels need at least` / `//   CUDA 12.0 on SM90 systems (Hopper)` / `//   CUDA 12.4 on SM89 systems (Lovelace)` / `if (cuda_device_capability >= 90) { return CUDA_VERSION >= 12000; } else if (cuda_device_capability >= 89) { return CUDA_VERSION >= 12040; }` / `return false;`
  The body of `CutlassFP8ScaledMMLinearKernel.can_implement` (`cutlass.py` L172) was not read; the online Linear method uses `cutlass_fp8_supported()` directly to pick per-token activation scaling (`online/fp8.py` L189-193).
- torch `_scaled_mm`: `_supports_torch_fp8_scaled_mm()` → on ROCm `_rocm_torch_fp8_scaled_mm_supported()`, else `return current_platform.supports_fp8()` ([`scaled_mm/pytorch.py` L37-49](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/kernels/linear/scaled_mm/pytorch.py#L37-L49)); `_rocm_torch_fp8_scaled_mm_supported` = `on_gfx942() or on_gfx950() or on_gfx12x() or on_gfx1250()` (L31-34). CUDA `supports_fp8()` = `return cls.has_device_capability(89)` ([`cuda.py` L617-618](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/platforms/cuda.py#L617-L618)).
- Marlin FP8: `"""FP8 Marlin kernel for GPUs that lack FP8 hardware support. Leverages the Marlin kernel for fast weight-only FP8 quantization."""` / `if not current_platform.is_cuda(): return False, "requires CUDA."` / `if not is_fp8_marlin_supported(): return False, "FP8 Marlin requires compute capability 7.5 or higher"` ([`scaled_mm/marlin.py` L29-46](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/kernels/linear/scaled_mm/marlin.py#L29-L46)); `def is_fp8_marlin_supported(): return current_platform.has_device_capability(75)` ([`marlin_utils_fp8.py` L31-32](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py#L31-L32)).
- ROCm FP8 kernel: `if get_cdna_version() <= 2 and not on_gfx12x(): return False, "requires CDNA3+ (gfx942/gfx950) or RDNA4 (gfx12x)"` ([`scaled_mm/rocm.py` L88-90](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/kernels/linear/scaled_mm/rocm.py#L88-L90)); `get_cdna_version`: gfx90a → 2, gfx942 → 3, gfx950 → 4, gfx1250 → 5, else 0 ([`rocm.py` L352-361](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/platforms/rocm.py#L352-L361)).
- AITER (ROCm): `if compute_capability is not None and compute_capability < 90: return False, "requires compute capability 90 and above."`, then `import aiter` must succeed ([`scaled_mm/aiter.py` L39-51](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/kernels/linear/scaled_mm/aiter.py#L39-L51)); the per-token variant additionally: `"requires setting VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_LINEAR=1. VLLM_ROCM_USE_AITER_LINEAR default is True."` (L136-151). gfx90a is (9,0) = 90 and passes the numeric check, so AITER on MI210/MI250X with aiter installed and `VLLM_ROCM_USE_AITER=1` is **UNCONFIRMED** either way (aiter's own FP8 GEMM support was not read).

ROCm `supports_fp8()` = `return on_cdna() or on_rdna4()` ([`rocm.py` L1009-1010](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/platforms/rocm.py#L1009-L1010)) with `_ON_CDNA = any(arch in _GCN_ARCH for arch in ["gfx9", "gfx1250"])` (L219) — true for gfx90a, which has no FP8 hardware; only the kernel gates above refuse it. `is_fp8_fnuz`: `return "gfx94" in _GCN_ARCH`; `fp8_dtype`: `torch.float8_e4m3fnuz` when FNUZ (L1013-1022).

**Gate 4, MoE backend selection** ([`fused_moe/oracle/fp8.py`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/fused_moe/oracle/fp8.py)):
- Priority L79-93: `AITER, FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, DEEPGEMM, VLLM_CUTLASS, TRITON, MARLIN, HUMMING, BATCHED_DEEPGEMM, BATCHED_VLLM_CUTLASS, BATCHED_TRITON, XPU, CPU, HPC`. The online per-tensor MoE method passes `allow_vllm_cutlass=False` (`online/fp8.py` L449-475), so `AVAILABLE_BACKENDS.remove(Fp8MoeBackend.VLLM_CUTLASS)` (L423-425).
- Loop L429-450: `for backend in AVAILABLE_BACKENDS: for k_cls in backend_to_kernel_cls(backend): supported, reason = k_cls.is_supported_config(...)` … `if current_platform.is_cuda() or current_platform.is_rocm(): raise NotImplementedError("No FP8 MoE backend supports the deployment configuration.")`
- `# MARLIN and CPU are mixed precision W8A16 config.` (L630-631).
- TritonExperts gates ([`experts/triton_moe.py`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/fused_moe/experts/triton_moe.py)): `_supports_current_device`: `return current_platform.is_cuda_alike() or current_platform.is_xpu()` (L118-119); `_supports_quant_scheme` adds the FP8 pairs only under `if current_platform.supports_fp8():` (L150-156, includes `(kFp8StaticTensorSym, kFp8DynamicTensorSym)`); `_supports_activation` → `apply_moe_activation_supported(activation)` (L173-174).
- MarlinExpertsBase gates ([`experts/marlin_moe.py` L559-583](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py#L559-L583)): `_supports_current_device`: `return p.is_cuda() and p.has_device_capability((7, 5))`; `SUPPORTED_W = [kFp8Static128BlockSym, kFp8StaticChannelSym, kFp8StaticTensorSym, …]`; `_supports_activation` → `apply_moe_activation_supported(activation)`.
- `_APPLY_MOE_ACTIVATIONS` contains `MoEActivation.GELU_TANH` ([`fused_moe/activation.py` L122-136](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/fused_moe/activation.py#L122-L136)); Gemma 4 passes `activation="gelu_tanh"` (`gemma4.py` L372; HF `hidden_activation` = `gelu_pytorch_tanh`).

*Derivation per architecture:* sm_75/80/86 → `supports_fp8()` false → Triton rejects FP8, DeepGEMM/FlashInfer need ≥ 90/100 → **Marlin MoE** (cap ≥ 7.5, FP8 per-tensor weight accepted, gelu_tanh accepted). sm_89+ → Triton FP8 (or AITER/FlashInfer/DeepGEMM when their own conditions hold). gfx942 → `supports_fp8()` true → Triton accepted. gfx1100 → false, Marlin needs CUDA → no backend. gfx90a → Triton gate accepts (misleading `supports_fp8()`), but the Linear `create_weights` raises first.

**Docs at the tag agree** ([`llm_compressor/fp8.md` L3-5](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/quantization/llm_compressor/fp8.md#L3-L5)): "Ada Lovelace, Hopper, and Blackwell GPUs are supported for W8A8. Turing/Ampere GPUs are supported for W8A16 (weight-only FP8) utilizing Marlin kernels." The hardware table ([`quantization/README.md` L67-74](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/quantization/README.md#L67-L74)) marks "Marlin (GPTQ/AWQ/FP8/FP4)" ✅ on Turing/Ampere/Ada/Hopper and ❌ on AMD, and "llm-compressor FP8 (W8A8)" ✅ on "AMD GPU" with no architecture qualifier — the code above is narrower (gfx942/950/12x/1250 only).

### Q3 — Peak memory while loading

- `OnlineLinearBase`: `uses_meta_device: bool = True` and `weight = ModelWeightParameter(data=torch.empty(output_size_per_partition, input_size_per_partition, device="meta",  # materialized and processed during loading` … `initialize_online_processing(layer)` ([`online/fp8.py` L115-156](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/online/fp8.py#L115-L156)). Class docstring: "Loads fp16/bf16 checkpoint weights onto meta device and materializes them just-in-time."
- `OnlineMoEMethodBase`: `uses_meta_device: bool = True`; `w13`/`w2` created with `device="meta"`; `initialize_online_processing(layer)` ([`online/moe_base.py` L20-97](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/online/moe_base.py#L20-L97)).
- `initialize_online_processing`: "Wrap a layer's weight loaders with online processing loaders" / `info.load_numel = 0` / `info.load_numel_total = get_layer_size(layer)` ([`reload/layerwise.py` L122-136](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/model_loader/reload/layerwise.py#L122-L136)).
- The wrapped loader (L149-227): `# Buffer loaded weights, track loading progress` / `info.loaded_weights.append((param_name, bound_args))` / `info.load_numel += num_loaded` … `# Process and copy when all weights are loaded` / `if info.load_numel >= info.load_numel_total:` / `_layerwise_process(layer, info)`. It also warns when more than one layer is mid-load: `"Allocating %.1f MB of device memory to buffers to load %s layers. This extra memory usage can be avoided by ordering weights by their parent layer when reloading."` (L199-213).
- `_layerwise_process` (L332-378): `1. Materializes the layer onto the target device 2. Loads all buffered weights 3. Runs quantization processing if applicable` → `materialize_layer(layer, info)` … `param.weight_loader(*args.args, **args.kwargs)` … `quant_method.process_weights_after_loading(layer)`.
- Loader entry point ([`base_loader.py` L74-91](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/model_loader/base_loader.py#L74-L91)): `self.load_weights(model, model_config)` / `# Log peak GPU memory after loading weights. This is needed to have test coverage on peak memory for online quantization.` / `# Process weights into kernel format. Note that when using online quantization, weights are (typically) quantized as they are loaded.` / `if _has_online_quant(model): finalize_layerwise_processing(model, model_config)` / `process_weights_after_loading(model, model_config, target_device)`.

*Derivation:* the BF16 copy of a quantized layer exists only between `materialize_layer` and `replace_parameter`; inside `process_weights_after_loading` the FP8 output is allocated while the BF16 input is live (`w13 = torch.empty_like(layer.w13_weight, dtype=fp8_dtype)`), so the per-layer transient is ≈ 1.5× that layer's BF16 bytes. Non-quantized parameters (embeddings, norms, router, `layer_scalar`) are ordinary device tensors from model construction. Peak ≈ final FP8-resident footprint + the largest layer's transient (+ a second layer's buffers if the checkpoint interleaves layers, per the warning above). Whether the buffered `bound_args` tensors are CPU-side for the default safetensors loader was not verified.

### Q4 — The default plan: Gemma 4 26B A4B on one A100 40 GB with `--quantization fp8`

**Model facts (safetensors headers of both shards at revision `4d7ae49`, read 2026-10-02; HF API `safetensors.total` agrees):**

| Group (my grouping by tensor name) | Params | BF16 bytes | BF16 GiB | FP8 under `fp8`? |
|---|---:|---:|---:|---|
| MoE experts (`*.experts.gate_up_proj` / `down_proj`, 60 tensors) | 22,837,985,280 | 45,675,970,560 | 42.539 | yes (`RoutedExperts`) |
| Dense Linear (attention q/k/v/o, dense `mlp.*`, `router.proj`; 295 tensors) | 1,656,346,880 | 3,312,693,760 | 3.085 | yes (`LinearBase`) |
| `model.language_model.embed_tokens.weight` `[262144, 2816]` | 738,197,504 | 1,476,395,008 | 1.375 | no (not `LinearBase`; `tie_word_embeddings: true`, no `lm_head` tensor in the files) |
| Vision tower + `embed_vision` (356 tensors) | 572,794,416 | 1,145,588,832 | 1.067 | unresolved — `gemma4_mm.py:1009` passes `quant_config=quant_config`; which module that is was not read |
| Norms (271) + `layer_scalar` (30) | 612,126 | 1,224,252 | 0.001 | no |
| **Total** | **25,805,936,206** | **51,611,872,412** | **48.067** | |

`model.safetensors.index.json` says `total_parameters: 26544131376`, `total_size: 51611872412`. The byte total matches the header sum exactly; the parameter total is 738,195,170 higher, within 2,334 of one `embed_tokens` matrix, consistent with a tied `lm_head` counted twice. Not resolved; bytes are what matter here.

Per layer (layer 0 shapes): `experts.gate_up_proj [128, 1408, 2816]` = 0.9453125 GiB BF16 (largest non-embedding tensor), `experts.down_proj [128, 2816, 704]`. Config: `num_hidden_layers 30`, `num_experts 128`, `top_k_experts 8`, `moe_intermediate_size 704`, `hidden_size 2816`, `hidden_size_per_layer_input 0`, `dtype bfloat16`, 25 `sliding_attention` + 5 `full_attention` layers.

**Derivation (mine):**
1. FP8-resident weights = experts 22,837,985,280 B (21.270 GiB) + dense Linear 1,656,346,880 B (1.543 GiB) + embeddings BF16 1.375 GiB + norms 0.001 GiB + vision tower 1.067 GiB (BF16, upper bound) = **25.26 GiB**; 24.72 GiB if the vision Linear layers are quantized. Per-tensor scales are negligible (one fp32 per tensor / per expert).
2. Load-time transient, largest layer = one layer's experts: gate_up 507,510,784 params + down 253,755,392 params = 761,266,176 params → 1.418 GiB BF16 + 0.709 GiB FP8 copy = **≈ 2.13 GiB**. Peak ≈ 25.3 + 2.1 ≈ **27.4 GiB**, versus 48.07 GiB if the whole BF16 model were resident — the per-layer path is what makes this card viable at all.
3. Budget: catalog `gb: 40`; vLLM default `gpu_memory_utilization: float = Field(default=0.92, gt=0, le=1)` ([`config/cache.py` L103](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/cache.py#L103)). 40 × 10⁹ B = 37.25 GiB → 34.27 GiB usable; if the card exposes 40 GiB → 36.8 GiB usable. Whether the A100 40 GB reports 40 GB or 40 GiB was not checked against an NVIDIA spec (UNCONFIRMED). Headroom after weights ≈ **9.0–11.5 GiB** for KV cache, activations and CUDA graphs.
4. Kernels on sm_80 (from Q2): Linear → Marlin W8A16; MoE → Marlin W8A16 (Triton rejects FP8 below cap 89; Marlin accepts cap ≥ 7.5, `kFp8StaticTensorSym`, `gelu_tanh`). Both store 1 byte/weight.

**Verdict:** loads, by trace. Not run on hardware. The KV-cache budget that the headroom buys is the planner's existing arithmetic, not this file's.

### Q5 — GGUF at v0.30.0

**vLLM docs at the tag** ([`gguf.md`](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/quantization/gguf.md)):
- L6-7: "!!! note — GGUF support has migrated to OOT [vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin). Make sure you have GGUF plugin installed before serving a GGUF model."
- L9-13: "Before serving a GGUF model, make sure to install the vllm-gguf-plugin: `uv pip install vllm-gguf-plugin`"
- L15-19: "To run a GGUF model with vLLM, you can use the `repo_id:quant_type` format to load directly from HuggingFace." / `# We recommend using the tokenizer from base model to avoid long-time and buggy tokenizer conversion.` / `vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M --tokenizer Qwen/Qwen3-0.6B`
- L34-35: "!!! warning — We recommend using the tokenizer from base model instead of GGUF model. Because the tokenizer conversion from GGUF is time-consuming and unstable, especially for some models with large vocab size."
- L37-45: a local `.gguf` path also works (`vllm serve ./Qwen3-0.6B-Q4_K_M.gguf --tokenizer Qwen/Qwen3-0.6B`); `--hf-config-path <base repo>` when "HuggingFace doesn't support your model".
- L3-4 warning: "GGUF support in vLLM is highly experimental and under-optimized at the moment, it might be incompatible with other features."

**In-tree at the tag:** `ls vllm/model_executor/layers/quantization/` has no gguf module; `QUANTIZATION_METHODS` in `quantization/__init__.py` has no `"gguf"` entry (grep returned nothing; `"fp8"`, `"online"`, `"fp8_per_tensor"` are present at L18/39/43); `ls vllm/model_executor/model_loader/` lists no gguf loader; a case-insensitive grep for `gguf` over the checked-out subset (quantization, platforms, model_loader, config, engine, fused_moe, models, kernels) hits only `models/qwen2_moe.py` and `models/exaone_moe.py`. `vllm/transformers_utils` was not checked out, so **where the `repo_id:quant_type` string is parsed (core or plugin) is UNCONFIRMED**.

**Plugin README** at [`e2b8ad5`](https://github.com/vllm-project/vllm-gguf-plugin/blob/e2b8ad532b8b5ea175100202c30430c1d2b5e6a8/README.md) (committed 2026-09-24T03:44:44Z, read 2026-10-02):
- L3-5: "This plugin provides out-of-tree GGUF quantization support for vLLM after in-tree support deprecation ([vllm-project/vllm#39583](https://github.com/vllm-project/vllm/issues/39583))."
- L60-62 Usage: `vllm serve Qwen/Qwen3-0.6B-GGUF:Q8_0 --tokenizer Qwen/Qwen3-0.6B`
- L35-43 install from source: `uv pip install -e . --no-build-isolation` ("ensures that the CUDA extension is compiled against the same PyTorch installation used by vLLM"); prerequisites "CUDA toolkit or ROCm toolkit" (L11). The README does not mention the PyPI name the vLLM docs give; whether `uv pip install vllm-gguf-plugin` resolves on PyPI was not checked.
- L93: tested coverage includes "Vision-language | Gemma 4 | Q4_K_M backbone with BF16 projector".
- L99-103: "Other vLLM-supported architectures may work when their GGUF tensor names map to the corresponding Hugging Face model. A model appearing in vLLM's general supported-model list does not by itself guarantee GGUF compatibility."

*Reading:* `--tokenizer <base repo>` is recommended by both sources and used in every example; neither says it is mandatory. The `repo_id:quant_type` form is the documented primary form.

---

## 4. UNCONFIRMED (what was tried)

- **CUTLASS kernels compiled for sm_120** — the C++ capability gate passes (`>= 90` branch) but the CMake architecture list was not read. Loading does not depend on it: the torch `_scaled_mm` kernel is in the candidate list and its gate (`has_device_capability(89)`) passes.
- **`CutlassFP8ScaledMMLinearKernel.can_implement` body** (`cutlass.py` L172) — not read; the C++ gate it must consult is quoted above.
- **AITER FP8 on gfx90a** with aiter installed and `VLLM_ROCM_USE_AITER=1` — the numeric gate (`< 90`) passes for (9,0); aiter's own support was not read. Default configuration: not reached.
- **T4 model dtype** — BF16 is not a Turing dtype; vLLM's capability-based dtype logic (`cuda.py` L256-259) was seen but not read in full.
- **Vision tower under `fp8`** — `gemma4_mm.py` L1009 passes `quant_config`; the enclosing module was not read. Affects 0.53 GiB of the Q4 estimate.
- **A100 40 GB: 40 GB vs 40 GiB** — not checked against an NVIDIA spec; both cases are in the arithmetic.
- **Where `repo_id:quant_type` is parsed** — `vllm/transformers_utils` not checked out.
- **CUDA platform `supported_quantization`** — grep of `cuda.py` returned nothing; base-class default not read (assumed unrestricted).
- Nothing in this file was run on a GPU.

## 5. What this means for the planner (analysis, not evidence)

- FP8 is a loadable default on the default card: `vllm serve google/gemma-4-26B-A4B-it --quantization fp8` on `a100-40` is a per-layer online quantization that peaks ≈ 27 GiB and settles ≈ 25 GiB, as W8A16 Marlin on both Linear and MoE layers.
- The rule "FP8 weights load" is by architecture, not by `caps.fp8`: true for every NVIDIA row (cap ≥ 75, W8A16 below 89) and for gfx942; false for gfx90a and gfx1100. That is the table `docs/research/vllm-rocm.md` already proposes; this file confirms its three AMD verdicts at the tag and adds the NVIDIA side and the MoE path.
- Memory per weight for the FP8 option is 1 B/param on every card where it loads; embeddings (and possibly the vision tower) stay at 2 B/param. For Gemma 4 26B that is 0.74 B (+0.57 B) of 25.8 B parameters.
- GGUF options need the plugin and should print `--tokenizer <base repo>` with the `repo_id:quant_type` form; the planner cannot promise a given GGUF repo loads.

---

# R2 follow-up: FP8 on the T4 (sm_75) at vLLM v0.30.0

Checked by Opus (the main loop) on 2026-10-04 against the tag `v0.30.0` (commit `ced6857afa0e`), raw files from
`https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/<path>`. This closes R2's open item "the Marlin kernel path on sm_75 (T4)".

**Finding: `--quantization fp8` loads on a T4 as W8A16 Marlin with FP16 activations, for every planner preset.**

| Question | Answer at the tag | Receipt |
|---|---|---|
| Does the FP8 config admit sm_75? | Yes: `Fp8Config.get_min_capability()` returns 75 (R2) | `vllm/model_executor/layers/quantization/fp8.py` |
| Does the Linear path pick Marlin on sm_75? | Yes: `is_fp8_marlin_supported()` is `current_platform.has_device_capability(75)`; the kernel refuses only with "FP8 Marlin requires compute capability 7.5 or higher" | `.../quantization/utils/marlin_utils_fp8.py:31-32`; `vllm/model_executor/kernels/linear/scaled_mm/marlin.py:42-43` |
| Does the MoE path accept sm_75? | Yes: `_supports_current_device` returns `p.is_cuda() and p.has_device_capability((7, 5))` | `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:559-561` |
| Are sm_75 kernels compiled? | Yes, dedicated ones: `# marlin has limited support for turing` / `MARLIN_SM75_ARCHS "7.5"`, and the same for MoE (`MARLIN_MOE_SM75_ARCHS "7.5"`) | `CMakeLists.txt:617-618`, `:1333-1334`, sources `sm75_kernel_*.cu` (`:701`) |
| Do they include FP8 weights? | Yes, FP16 only: the generator adds `(a, b, c)` to its sm_75 set only when `a_type in ["kFloat16", "kS8"] and c_type == "kFloat16"`; the FP8 entry (`b_type: kFE4M3fn`) takes the default `a_type`/`c_type` lists, so `(kFloat16, kFE4M3fn, kFloat16)` is built for sm_75 (2 stages) and no BF16 pairing is | `csrc/libtorch_stable/quantization/marlin/generate_kernels.py:87-93, 186-229`; same logic in `csrc/libtorch_stable/moe/marlin_moe_wna16/generate_kernels.py:178-228` |
| Does the published build include 7.5? | Yes for x86: `CUDA_ARCH_X86: "7.5 8.0 8.6 8.9 9.0 10.0 12.0"` (also the CUDA 12.9 x86 build); the Dockerfile default is `'7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0'`. The aarch64 builds start at 8.0 (no T4 there) | `.buildkite/release-pipeline.yaml:10,17`; `docker/Dockerfile:293` |
| What dtype does a BF16 checkpoint run in? | FP16: below compute capability 8.0 `supported_dtypes` is `[float16, float32]`, and `--dtype auto` falls back to `supported_dtypes[0]` with the warning "Your device … doesn't support torch.bfloat16. Falling back to torch.float16 for compatibility." FP16 is exactly the activation type the sm_75 FP8 kernels take | `vllm/platforms/cuda.py:255-261`; `vllm/config/model.py:2277-2318` |

**The exception.** `_FLOAT16_NOT_SUPPORTED_MODELS` = `gemma2`, `gemma3`, `gemma3_text`, `glm4` (`vllm/config/model.py:2252-2257`). For those model types `--dtype auto` on a T4 resolves to FP32, and Marlin has no FP32-activation kernel (MoE asserts `hidden_states.dtype in [torch.float16, torch.bfloat16]`, `marlin_moe.py:300,454`), so FP8 would not run. **No planner preset is one of them** (the 16 presets are Llama 3.1/3.3/4, Qwen 3/2.5/3.5/3-Coder, Gemma 4, DeepSeek V3/R1, Mistral Small 4/Large). A user-supplied model of those families on a T4 would hit it at any precision: a BF16 plan would also run as FP32, at twice the planned weight memory. That is outside fix 1 and not handled today.

**Not verified:** a run on real T4 hardware; the sm_75 kernels' numerical behaviour beyond being built and selected ("limited support" in the CMake comment is, per the generator, FP16 only and 2 pipeline stages).

**For fix 1:** the T4 defaults to FP8, as the A100s do (W8A16). The cards that default to BF16 stay the ones R2 found cannot load FP8: gfx90a (MI210, MI250X) and gfx1100 (RX 7900 XTX).
