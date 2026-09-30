# Changelog

## v1.1.0 — 2026-09-30

AMD GPUs, correct figures above 8 GPUs, and prices that say where they came from. Everything
below is in the page and in the PDF report (`generate_report.py`), unless it says otherwise.

### Added

- **Five AMD GPUs:** MI210 64 GB, MI250X 128 GB (one row for the two-GCD board), MI300X 192 GB,
  MI325X 256 GB and RX 7900 XTX 24 GB. Their VRAM, fit, command and cost figures are complete.
  Their throughput reads as not modelled: AMD cards never borrow NVIDIA's performance
  constants, and the published ROCm measurements are not yet enough to derive their own.
- **ROCm guidance:** on an AMD card, the command runs vLLM's own ROCm image,
  `vllm/vllm-openai-rocm`, pinned to v0.30.0. It picks GPUs with `HIP_VISIBLE_DEVICES`, keeps FP8
  weights off CDNA2 and RDNA3 (the FP8 KV cache stays available), and states the ROCm caveats with
  their sources ([docs/research/vllm-rocm.md](docs/research/vllm-rocm.md)).
- **Every price says where it came from:** the provider, SKU, region and date it was read, or a
  hand record with its page and day, or a note saying why it has neither. A price found but not
  confirmed is shown as a lead, with why, and used in no figure.
- **A weekly price check** (in the repository, not in the page): `tools/price_check.py` reads
  published prices from Azure, AWS, Lambda, CoreWeave and Vast.ai. A weekly job opens a pull
  request when a price is confirmed or has moved. It never pushes to master, and it leaves a move
  over 40% for a person. A watchdog checks the next day that the run happened and delivered what
  it found.

### Fixed

- **Per-device VRAM above 8 GPUs** was divided by every device in the cluster, which is not how
  the emitted command shards. It was 2× optimistic at 16 GPUs and 16× at 128. Weights and
  activations now divide by the tensor-parallel group. Mixture-of-experts models keep the old
  divisor, and say so.
- **The compute ceiling and time to first token** were charged at 16-bit cost at every precision.
  That understated the FP8 ceiling by half and doubled its time to first token. Both are now
  charged at the precision the matrix multiplies run in.
- **A card without performance constants of its own** would have borrowed NVIDIA's. Each catalog
  row now names the constants its silicon was measured with, and a row with none reads as not
  modelled.
- **The PDF report's JSON configs and menu** now refuse, with the reason, input they used to plan
  wrongly or crash on:
  - a model path that starts with `~`, names a `$` variable, or is relative;
  - a precision width its quantization contradicts, or one of zero or less;
  - a quantization vLLM does not list for the card's vendor;
  - a Hugging Face model id that is empty, blank or null, or starts with `-`;
  - a `NaN` or `Infinity` in any field.
- **The documents and images** were corrected against the code, and tests now fail when they drift
  from it.

### Known issues

- **Hyperscaler prices for more than one GPU read low.** A hyperscaler tier is priced from the
  card's smallest one-GPU instance, and larger instances can cost more per GPU. AWS's 2-, 4- and
  8-GPU G7e sizes cost $4.14 a GPU (read 2026-09-25), so an RTX PRO 6000 96 GB plan on two or more
  GPUs reads about 19% low. The L40S 48 GB hyperscaler tier is priced the same way. v1.2 prices by
  GPU count.
- The rest are stated where they apply, in the [README](README.md) and
  [docs/MODEL.md](docs/MODEL.md). They cover per-user speed and time to first token under data
  parallelism, the mixture-of-experts divisor, and throughput above 8 GPUs, which is unmeasured.

## v1.0.0

NVIDIA GPUs and vLLM. It was released before this changelog was kept, and never tagged.
