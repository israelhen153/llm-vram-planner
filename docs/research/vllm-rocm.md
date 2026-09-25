# vLLM on ROCm, as of v0.30.0

**Read:** 2026-09-23. **Against:** vLLM `v0.30.0`, released 2026-09-22. Source links are pinned to that tag; AMD's pages are dated by when they were read.

**How it was read.** A Fable research agent gathered it first (`tmp/research/vllm-rocm-2026-09-23.md`, not in the repo). Then every claim below was re-read at its source by hand the same day. "Traced" means the code path was read but not run on hardware; nothing here was run on an AMD GPU.

Each section ends with **what the planner does** about it. The command panel, the copied report and the PDF link to this page.

## What vLLM supports

- **GPUs:** "MI200s (gfx90a), MI300 (gfx942), MI350 (gfx950), Radeon RX 7900 series (gfx1100/1101), Radeon RX 9000 series (gfx1200/1201), Ryzen AI MAX / AI 300 Series (gfx1151/1150)"; "ROCm 6.3 or above". ([`gpu.rocm.inc.md` L16-17](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/getting_started/installation/gpu.rocm.inc.md#L16-L17))
- All five catalog cards are covered:
  - MI210 and MI250X are gfx90a;
  - MI300X and MI325X are gfx942;
  - the RX 7900 XTX is gfx1100.

## How to run it: vLLM's own image

- **The image.** vLLM publishes `vllm/vllm-openai-rocm`, with `:latest` for the stable release. "AMD's Docker images (`rocm/vllm` and `rocm/vllm-dev`) are deprecated in favor of the official vLLM Docker images." ([L353-356, L391-394](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/getting_started/installation/gpu.rocm.inc.md#L353-L394))
- **The tag.** `v0.30.0` exists on Docker Hub (updated 2026-09-22, linux/amd64).
- **The `docker run` flags**, verbatim from the docs ([L359-371](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/getting_started/installation/gpu.rocm.inc.md#L359-L371)):
  - `--group-add=video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined`
  - `--device /dev/kfd --device /dev/dri`
  - `-v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=$HF_TOKEN"`
  - `-p 8000:8000 --ipc=host`
- **The entrypoint** is `vllm serve` ([`docker/Dockerfile.rocm` L1006](https://github.com/vllm-project/vllm/blob/v0.30.0/docker/Dockerfile.rocm#L1006)), so the serve arguments follow the image name.
- **Why the image rather than pip.** "ROCm pre-built wheels are only available for **Python 3.12**." On other versions the installer "**will silently fall back** to the CUDA wheel", which then fails on AMD GPUs. ([L30-32](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/getting_started/installation/gpu.rocm.inc.md#L30-L32))
- **Docker's `--device` is not vLLM's.** vLLM's own `--device` flag was removed in v0.10.0 ([#21349](https://github.com/vllm-project/vllm/pull/21349), "Remove deprecated args in v0.10", merged 2025-07-22). The two `--device` flags above are Docker's, and they map the GPU device files into the container.

**What the planner does.** On an AMD card, the command is a `docker run` of `vllm/vllm-openai-rocm:v0.30.0` with exactly those flags. It adds a volume for a local model path, then the image, then the same `vllm serve` arguments an NVIDIA card gets. The tag is pinned to the version this page was read against. A newer tag may behave differently.

## Choosing GPUs: `HIP_VISIBLE_DEVICES`

- v0.30.0 removed "the `CUDA_VISIBLE_DEVICES` fallback on ROCm (use `HIP_VISIBLE_DEVICES`)" ([release notes](https://github.com/vllm-project/vllm/releases/tag/v0.30.0), under "Items deprecated for 0.29 removed").
- If both are set and differ, vLLM stops with "Inconsistent GPU visibility env vars". ([`rocm.py` L127-137](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/platforms/rocm.py#L127-L137))
- **AMD's guide says the opposite.** Its [vLLM optimization guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html) uses `CUDA_VISIBLE_DEVICES` in its examples and says "Keep HIP_VISIBLE_DEVICES unset to avoid conflicts." vLLM's v0.30.0 release notes are the later statement, and the image here is vLLM's, so this page follows vLLM.
- **MI250X:** "Each MI250 OAM hosts two Graphics Compute Dies (GCDs), each enumerated as an independent GPU by ROCm tools." ([AMD Instinct MI250 system acceptance](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi250.html)) One MI250X board is therefore two device IDs, and the catalog row carries `devices: 2`.

**What the planner does.** It prints `--env HIP_VISIBLE_DEVICES=…` as guidance under the command, not inside it. The container sees every GPU the host maps, and the planner can't know the host.

## FP8 weights: not on CDNA2 or RDNA3

- **The platform check alone would let them through.** `supports_fp8()` returns `on_cdna() or on_rdna4()` ([`rocm.py` L1009-1011](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/platforms/rocm.py#L1009-L1011)), and `_ON_CDNA` matches any `gfx9` target, gfx90a included ([L219](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/platforms/rocm.py#L219)).
- **The FP8 matrix kernels refuse.** The ROCm scaled-mm kernel returns "requires CDNA3+ (gfx942/gfx950) or RDNA4 (gfx12x)" ([`scaled_mm/rocm.py` L89-90](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/kernels/linear/scaled_mm/rocm.py#L89-L90)). The PyTorch path is limited to gfx942, gfx950, gfx12x and gfx1250 ([`scaled_mm/pytorch.py` L31-34](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/kernels/linear/scaled_mm/pytorch.py#L31-L34)). The AITER kernels need CDNA3 or newer (below).
- So on gfx90a (MI210, MI250X) and gfx1100 (RX 7900 XTX), no per-tensor or per-channel FP8 kernel accepts the layer. Traced, not run.
- **The hardware agrees.** CDNA3 introduced FP8, and RDNA3 has none. The catalog's `caps.fp8` records that separately (see `amd-gpu-specs.md`).
- **The docs disagree.** vLLM's quantization table marks "llm-compressor FP8 (W8A8)" ✅ on "AMD GPU" with no architecture qualifier ([`quantization/README.md` L74](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/quantization/README.md#L74)). The code is what runs.

**What the planner does.** On MI210, MI250X and RX 7900 XTX, FP8 weights can't be chosen:
- on the page, the option is disabled and a selected FP8 falls back to BF16, with the reason and this page linked;
- the CLI refuses `--prec fp8` for these cards.

The rule lives in a table keyed by architecture, separate from `caps.fp8`. `caps.fp8` means FP8 tensor cores; this table means what vLLM can load. On NVIDIA Ampere the two differ: vLLM runs FP8 weights there without FP8 tensor cores.

## FP8 KV cache: allowed

- "`kv_cache_dtype="fp8_e4m3"`: Supported on CUDA 11.8+ and ROCm (AMD GPUs)." ([`quantized_kvcache.md` L40](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/quantization/quantized_kvcache.md#L40))
- The ROCm attention backend lists `fp8`, `fp8_e4m3` and `fp8_e5m2` among its KV-cache types ([`rocm_attn.py` L169-176](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/attention/backends/rocm_attn.py#L169-L176)).
- **On RDNA3,** the custom paged-attention kernel requires `kv_cache_dtype == "auto"` ([`rocm.py` L401-410](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/platforms/rocm.py#L401-L410)). An FP8 cache therefore runs on a different kernel path there. Whether it is numerically right on gfx1100 was not verified.

**What the planner does.** It allows an FP8 KV cache on every AMD card, and on the RX 7900 XTX it says the path is unverified.

## AWQ and GPTQ: the docs and the code disagree

- **The docs:** vLLM's table marks AWQ ❌ and GPTQ ❌ on "AMD GPU" ([`quantization/README.md` L69-70](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/quantization/README.md#L69-L70)).
- **The code:** the ROCm platform's `supported_quantization` includes `awq`, `auto_awq`, `awq_marlin`, `gptq` and `auto_gptq` ([`rocm.py` L503-527](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/platforms/rocm.py#L503-L527)). For AWQ it turns on the Triton kernels itself: "Using AWQ quantization with ROCm, but VLLM_USE_TRITON_AWQ is not set, enabling VLLM_USE_TRITON_AWQ." ([L977-982](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/platforms/rocm.py#L977-L982))

**What the planner does.** It offers both on AMD, and names the disagreement with both links. It doesn't claim either works, because neither was run.

## GGUF: the plugin, on every vendor

- GGUF support left vLLM's core in v0.24.0 ([#39612](https://github.com/vllm-project/vllm/pull/39612)). It needs `vllm-gguf-plugin`.
- That is why `gguf` is missing from the ROCm allow-list above. Registering a plugin's quantization adds it to the current platform's list: "Automatically assume the custom quantization config is supported" ([`quantization/__init__.py` L96-99](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/__init__.py#L96-L99)). So the plugin passes ROCm's check too.
- The plugin lists "CUDA toolkit or ROCm toolkit" as prerequisites ([README](https://github.com/vllm-project/vllm-gguf-plugin)).
- It is not in the ROCm image. Whether its PyPI wheels carry ROCm kernels was not verified.

**What the planner does.** It shows the same GGUF guidance as on NVIDIA. On AMD it adds that the plugin isn't in the image and that its ROCm support was not verified.

## Attention kernels and AITER

This is what the roadmap's "Flash Attention support varies by GPU arch" comes to at v0.30.0.
- **AITER, AMD's kernel library, is built and enabled only on CDNA3 and newer.**
  - `is_aiter_found_and_supported()` checks "device arch is CDNA 3 or better" ([`_aiter_ops.py` L141, L157](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/_aiter_ops.py#L138-L160)).
  - The image builds it for `gfx942;gfx950` only ([`Dockerfile.rocm_base` L37](https://github.com/vllm-project/vllm/blob/v0.30.0/docker/Dockerfile.rocm_base#L37)).
- **vLLM leaves it off by default** ([`envs.py` L1231-1232](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/envs.py#L1231-L1232)).
- **AMD says to turn it on.** Its vLLM guide: "Always set VLLM_ROCM_USE_AITER=1 even when using --attention-backend explicitly", because it "is still required to enable AITER for GEMM, RMSNorm, and MoE kernels" ([AMD vLLM optimization guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html)).
- **Without AITER**, MI210, MI250X and the RX 7900 XTX use vLLM's ROCm and Triton attention backends.

**What the planner does.** It adds `--env VLLM_ROCM_USE_AITER=1` to the command on MI300X and MI325X only. On the others it says AITER isn't built for that architecture.

## Could not verify

- The runtime error `--quantization fp8` produces on gfx90a and gfx1100. The kernel gates were traced, not run. The MoE FP8 path was not traced.
- Whether a block-scaled FP8 checkpoint loads on gfx90a through the Triton block-scaled kernel. The planner blocks FP8 weights on these cards regardless.
- Whether an FP8 KV cache is numerically right on gfx1100, and whether `fp8_e5m2` works on ROCm. The backend lists it; the docs say CUDA only.
- Whether `vllm-gguf-plugin` works on ROCm.
- The ROCm version inside `vllm/vllm-openai-rocm:v0.30.0`. The Dockerfile's default base is `rocm/dev-ubuntu-22.04:7.2.3-complete` ([`Dockerfile.rocm_base` L1](https://github.com/vllm-project/vllm/blob/v0.30.0/docker/Dockerfile.rocm_base#L1)), but the release build's arguments were not read.
- Whether AMD's AITER advice, written for its Instinct MI300-series guide, applies unchanged to the MI325X. It shares the MI300X's gfx942 target.
