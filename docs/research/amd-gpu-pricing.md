# AMD GPU Cloud Pricing Research

**Date:** 2026-07-31  
**Purpose:** 3-tier cost data for v1.1.0 AMD support  
**Status:** On-demand pricing snapshot across hyperscaler, specialized, and marketplace tiers

---

## Pricing Summary Table

| GPU | Tier | Provider | USD/GPU/hr | Instance Name | Source URL | Date Seen |
|-----|------|----------|-----------|---------------|-----------|-----------|
| MI300X | Marketplace | Spot (Various) | $0.95 | Spot/Interruptible | https://www.spheron.network/blog/amd-mi300x-mi355x-pricing-2026/ | 2026-07-31 |
| MI300X | Specialized | TensorWave | $1.71 | Standard on-demand | https://www.thundercompute.com/blog/amd-mi300x-pricing | 2026-07-31 |
| MI300X | Specialized | Vultr | $1.85 | 8x MI300X cluster (÷8 per GPU) | https://www.thundercompute.com/blog/amd-mi300x-pricing | 2026-07-31 |
| MI300X | Specialized | DigitalOcean | $1.99 | gpu-mi300x1-192gb | https://getdeploying.com/gpus/amd-mi300x | 2026-07-31 |
| MI300X | Specialized | Hot Aisle | $2.99 | 1x MI300X | https://getdeploying.com/gpus/amd-mi300x | 2026-07-31 |
| MI300X | Specialized | RunPod | $2.19–$2.39 | Secure Cloud MI300X | https://getdeploying.com/gpus/amd-mi300x | 2026-07-31 |
| MI300X | Specialized | Crusoe Cloud | $3.45 | mi300x-192gb-ib.8x (÷8) | https://getdeploying.com/gpus/amd-mi300x | 2026-07-31 |
| MI300X | Specialized | Cirrascale | $3.85 | MI300X cluster | https://www.thundercompute.com/blog/amd-mi300x-pricing | 2026-07-31 |
| MI300X | Hyperscaler | Oracle Cloud | $6.00 | BM.GPU.MI300X.8 bare-metal (÷8) | https://getdeploying.com/gpus/amd-mi300x | 2026-07-31 |
| MI300X | Hyperscaler | Azure | $6.00 | Standard_ND96isr_MI300X_v5 (÷8) | https://getdeploying.com/gpus/amd-mi300x | 2026-07-31 |
| MI300X | Hyperscaler | Azure | $7.86 | Standard_ND96isr_MI300X_v5 (East US, ÷8) | https://getdeploying.com/gpus/amd-mi300x | 2026-07-31 |
| MI300X | Hyperscaler | CoreWeave | $6.31 | 8x MI300X node (÷8, SXM variant) | https://www.thundercompute.com/blog/amd-mi300x-pricing | 2026-07-31 |
| MI325X | Specialized | TensorWave | $2.25 | Standard on-demand (lowest tracked) | https://getdeploying.com/gpus/amd-mi325x | 2026-07-31 |
| MI325X | Specialized | Bentaus | $2.25 | 8x MI325X on-demand (÷8) | https://getdeploying.com/gpus/amd-mi325x | 2026-07-31 |
| MI325X | Marketplace | Vultr | $2.00 | 8x MI325X spot (÷8, out of stock) | https://getdeploying.com/gpus/amd-mi325x | 2026-07-31 |
| MI325X | Specialized | Cyfuture AI | $3.21–$3.31 | 1x MI325X on-demand | https://getdeploying.com/gpus/amd-mi325x | 2026-07-31 |
| MI325X | Specialized | DigitalOcean | $2.88 | 1x MI325X (12-month reserved) | https://getdeploying.com/gpus/amd-mi325x | 2026-07-31 |
| MI325X | Specialized | Cyfuture AI | $1.57–$1.67 | 8x MI325X (12-month reserved, ÷8) | https://getdeploying.com/gpus/amd-mi325x | 2026-07-31 |
| MI250X | Specialized | Runcrate | $1.35 | Standard on-demand | https://www.runcrate.ai/pricing/gpu/mi250x | 2026-07-31 |
| MI250X | Specialized | Runcrate | $1.20–$1.50 | Regional variance | https://www.runcrate.ai/pricing/gpu/mi250x | 2026-07-31 |
| MI250X | Specialized | Cirrascale | $1.28–$1.60 | Cloud rental | https://www.thundercompute.com/blog/amd-mi300x-pricing | 2026-07-31 |
| MI250X | Specialized | Runcrate | $0.94 | Reserved (30% discount) | https://www.runcrate.ai/pricing/gpu/mi250x | 2026-07-31 |
| MI210 | Specialized | Runcrate | $0.70–$0.82 | Standard rental | https://www.runcrate.ai/pricing/gpu/mi210 | 2026-07-31 |
| RX 7900 XTX | Marketplace | Vast.ai | Dynamic | Marketplace (consumer GPU) | https://vast.ai/pricing/gpu/RX-7900-XTX | 2026-07-31 |

---

## Notes

### Availability Summary

| GPU | Availability | Rarity Notes | Cloud Coverage |
|-----|--------------|--------------|-----------------|
| MI300X | Widely available | Common; available at 10+ providers | Hyperscalers + neoclouds + marketplace |
| MI325X | Limited | Newer; offered by 5 providers; no AWS/GCP/Azure hyperscaler SKUs | Specialized + neocloud only |
| MI250X | Rare | Offered by Cirrascale, Runcrate; older generation; declining availability | Specialist providers only |
| MI210 | Rare | Minimal cloud presence; only found on Runcrate; effectively obsolete for cloud rental | Single provider only |
| RX 7900 XTX | Marketplace only | Consumer GPU; available on peer-to-peer platforms; pricing highly volatile | Vast.ai listed; pricing dynamic |

### Key Findings

1. **Pricing Per-GPU vs. Per-Instance:** Most sources quote per-instance rates for multi-GPU clusters (e.g., 8×MI300X at $48/hr = $6/GPU/hr). The table normalizes to per-GPU-per-hour for comparability. Verify provider quotes whether you're billed per-instance or per-GPU.

2. **Tier Structure:**
   - **Hyperscaler (AWS/GCP/Azure/Oracle):** $6.00–$7.86/GPU/hr for MI300X. AWS and GCP do not yet offer dedicated AMD GPU instances. Azure offers ND MI300X v5 series; Oracle offers BM.GPU bare-metal.
   - **Specialized (CoreWeave/TensorWave/DigitalOcean/Lambda/Crusoe):** $1.71–$3.85/GPU/hr for MI300X, with TensorWave and Vultr at the low end ($1.71–$1.85).
   - **Marketplace/Spot (Vast.ai/RunPod spot):** $0.95–$2.39/GPU/hr for MI300X; Vast.ai offers consumer GPUs (RX 7900 XTX) with dynamic pricing.

3. **Reserved vs. On-Demand:**
   - Reserved instances (12-month commitments) typically offer 25–35% savings.
   - Spot pricing on Azure and Vultr reaches $1.45–$1.85/GPU/hr (81–82% discounts off on-demand).
   - Vast.ai and RunPod marketplace pricing is lowest but interruptible.

4. **MI210 and MI250X Rarity:**
   - **MI210:** Essentially not rentable on mainstream cloud; only Runcrate lists it at $0.70–$0.82/hr. These are end-of-life GPUs from AMD's perspective; cloud providers have moved to MI300X/MI325X.
   - **MI250X:** Found on Cirrascale (monthly commitment) and Runcrate. Pricing ranges $1.28–$1.60/hr. Availability declining as MI300X matures. Still 30–40% cheaper than H100.

5. **MI325X vs. MI300X:**
   - MI325X (256GB VRAM, newer) ranges from $1.57/hr (reserved) to $3.31/hr (on-demand), averaging $2.30/hr.
   - MI300X (192GB VRAM, established) ranges from $0.95/hr (spot) to $7.86/hr (hyperscaler), averaging $2.79/hr.
   - MI325X demand is high but supply-constrained; no major hyperscaler (AWS/GCP/Azure) has launched dedicated MI325X SKUs yet.

6. **RX 7900 XTX:**
   - Consumer-grade GPU; listed on Vast.ai only. Pricing is dynamic (marketplace-based); no fixed hourly rate published. Expect $0.10–$0.50/hr on spot markets but highly volatile.

7. **Hidden Costs & Caveats:**
   - Egress/bandwidth charges not listed here; can be significant for large inference workloads.
   - Cirrascale requires multi-month commitments ($22,499 minimum); no hourly billing.
   - CoreWeave negotiates custom pricing per customer; published rates unavailable for AMD GPUs.
   - Spot/interruptible instances (Azure low-priority, Vast.ai, RunPod) can be reclaimed with no notice.
   - TensorWave and Hot Aisle are neocloud specialists; SLA/uptime terms differ from hyperscalers.

8. **Data Currency:**
   - All prices current as of 2026-07-31.
   - GPU market is volatile; hyperscaler discounts (3-year reserved) and spot pricing fluctuate weekly.
   - Specialist/neocloud pricing more stable but tied to individual provider contract terms.

---

## Provider Coverage by GPU Model

```
Provider         | MI210 | MI250X | MI300X | MI325X | RX 7900 XTX
-----------------|-------|--------|--------|--------|------------
Azure            | ✗     | ✗      | ✓      | ✗      | ✗
AWS              | ✗     | ✗      | ✗      | ✗      | ✗
GCP              | ✗     | ✗      | ✗      | ✗      | ✗
Oracle Cloud     | ✗     | ✗      | ✓      | ✗      | ✗
CoreWeave        | ✗     | ?      | ✓      | ✗      | ✗
Lambda Labs      | ✗     | ✗      | ✓*     | ✗      | ✗
TensorWave       | ✗     | ✗      | ✓      | ✓      | ✗
DigitalOcean     | ✗     | ✗      | ✓      | ✓      | ✗
Vultr            | ✗     | ✗      | ✓      | ✓      | ✗
RunPod           | ✗     | ✗      | ✓      | ✗      | ✗
Crusoe Cloud     | ✗     | ✗      | ✓      | ✗      | ✗
Cirrascale       | ✗     | ✓      | ✓      | ✗      | ✗
Hot Aisle        | ✗     | ✗      | ✓      | ✗      | ✗
Runcrate         | ✓     | ✓      | ✗      | ✗      | ✗
Vast.ai          | ✗     | ✗      | ✗      | ✗      | ✓
Seeweb           | ✗     | ✗      | ✓      | ✗      | ✗
Bentaus          | ✗     | ✗      | ✗      | ✓      | ✗
Cyfuture AI      | ✗     | ✗      | ✗      | ✓      | ✗

Legend: ✓ = Available | ✗ = Not offered | ? = Unclear/rumored | * = SXM variant only
```

---

## Recommendations for v1.1.0 AMD Support

1. **For Prod Workloads:** Quote Azure/Oracle on-demand ($6/GPU/hr) as ceiling; target DigitalOcean/TensorWave ($1.99–$1.71) for cost-sensitive deployments.
2. **For Reserved/Committed:** Cyfuture AI MI325X at $1.57/hr (12mo) or Azure spot at $1.45/hr (risk of preemption).
3. **For Development/Testing:** RunPod spot ($1.49/hr) or Vast.ai marketplace (highly variable).
4. **Avoid:** MI210 and MI250X for new deployments; recommend MI300X or MI325X instead due to better availability and pricing.
5. **Reserved Capacity:** Clarify in tool whether user wants on-demand only or is willing to commit; 30% savings available via annual/monthly contracts.

---

## Sources Cited

- [AMD MI300X Pricing (July 2026) — Thunder Compute](https://www.thundercompute.com/blog/amd-mi300x-pricing)
- [MI300X Cloud Pricing: Compare 10+ Providers (2026) — GetDeploying](https://getdeploying.com/gpus/amd-mi300x)
- [MI325X Cloud Pricing: Compare 5+ Providers (2026) — GetDeploying](https://getdeploying.com/gpus/amd-mi325x)
- [AMD MI300X and MI355X Pricing 2026 — Spheron Blog](https://www.spheron.network/blog/amd-mi300x-mi355x-pricing-2026/)
- [AMD MI250X Pricing — Runcrate](https://www.runcrate.ai/pricing/gpu/mi250x)
- [AMD MI210 Pricing — Runcrate](https://www.runcrate.ai/pricing/gpu/mi210)
- [AMD GPUs Cloud Pricing — Compute Prices](https://computeprices.com/gpus/category/amd)
- [Vast.ai RX 7900 XTX Pricing](https://vast.ai/pricing/gpu/RX-7900-XTX)
- [RunPod vs Vast.ai GPU Pricing Comparison (2026) — GPUs.io](https://gpus.io/en/providers/compare/runpod-vs-vast-ai)
- [DigitalOcean GPU Cloud Pricing Guide — DeployBase](https://deploybase.ai/articles/digitalocean-gpu-cloud-pricing-complete-guide-vs-hr-for)
