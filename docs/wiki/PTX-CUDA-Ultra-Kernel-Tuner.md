# Jesse PTX-CUDA Ultra Kernel Tuner (`1,000D x 100W`)

The **Jesse PTX-CUDA Ultra Kernel Tuner** (`v3/crates/rvllm-cutlass/src/ptx_tuner`) synthesizes hardware-optimal PTX 8.7 and CUDA C++ kernels across all NVIDIA GPU generations without relying on vLLM's Python runtime or static heuristic tables.

## 1. Search Topology (`1,000D x 100W`)

For every `(HardwareDescriptor, WorkloadShape)` pair, `TunerCampaign` executes:
- **Depth ($D = 1,000$)**: `1,000` sequential evolutionary generations.
- **Width ($W = 100$)**: `100` concurrent schedule genomes per generation (`100,000` total schedule evaluations per workload shape).
- **Laplace-Smoothed Survival Ranking**:
  Each worker maintains good/bad mutation counters $(g, r)$ and is ranked by its Laplace-smoothed survival score combined with predicted latency:
  $$p_{\text{laplace}} = \frac{g + 1}{g + r + 2}$$
- **16-Rule Evolutionary Mutation Engine (`MutationOp`)**:
  Explores CTA tile dimensions (`tile_m`, `tile_n`, `tile_k`), warp specialization (`warps_per_cta`, `producer_warps`, `consumer_warps`), software pipeline stages (`pipeline_stages`), CTA cluster shapes (`cluster_m`, `cluster_n`), split-$K$ reductions (`split_k`), TMA multicast, shared-memory swizzles (`32B`/`64B`/`128B`), and cache hints (`L2::128B`, `evict_first`, `evict_last`).

## 2. Static Correctness & Zero-Spill Invariants

Every candidate genome must pass `validate_schedule` before roofline scoring or code emission:
1. **`MatchedPair` Producer-Consumer Synchronization**:
   Async TMA (`cp.async.bulk.tensor`), Hopper `wgmma.mma_async`, and Blackwell `tcgen05.mma` schedules must pair producer transaction arrivals with consumer phase parity waits (`mbarrier.init` $\leftrightarrow$ `mbarrier.arrive.expect_tx` $\leftrightarrow$ `mbarrier.try_wait.parity`).
2. **Zero Register Spills**:
   Enforces `regs_per_thread * threads_per_cta <= regs_per_sm` and `regs_per_thread <= max_regs_per_thread`, preventing local-memory spill stores/loads.
3. **Shared Memory & Tensor Memory Capacity**:
   Multi-stage shared-memory tiles and Blackwell TMEM columns are bounded by `HardwareDescriptor::shared_mem_per_sm_bytes` and `tmem_columns_per_sm`.

## 3. Authenticated `rvllm_cutlass::Policy` Integration

Winning schedules are compiled into an authenticated `rvllm_cutlass::Policy` (`schema = "rvllm.cutlass.policy.v1"`) via `build_authenticated_policy`:
- Populates `rules` and `tuned_kernels` with exact `m`, `n`, `k`, `arch`, `variant`, and `tuned_latency_us`.
- Computes and embeds the canonical SHA-256 `policy_sha256` digest verified at load time by `Policy::verify_integrity()`.

## 4. CLI Usage

```bash
cargo run --release --locked --manifest-path v3/Cargo.toml \
  -p rvllm-cutlass --bin rvllm_ptx_tuner -- \
  --depth 1000 --width 100 \
  --emit-bundle ./artifacts/ptx-tuner
```
