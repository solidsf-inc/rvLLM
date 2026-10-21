# AGENTS.md — rvLLM Agent & Contributor Guide

This file defines the architecture, verification gates, kernel-tuning workflows, and release-surface rules for agents and contributors working in `solidsf-inc/rvLLM`.

## 1. Authoritative Workspace & Layout

`v3/Cargo.toml` is the authoritative Rust workspace manifest.

| Path | Responsibility |
| --- | --- |
| `v3/crates/rvllm-core` | Shared model types, tensor descriptors, and error definitions (`RvllmError`). |
| `v3/crates/rvllm-mem` | Device HBM arena, pinned host buffers, streams, and CUDA context lifecycle. |
| `v3/crates/rvllm-kernels` | PTX module loader, kernel manifest verification, and launch abstractions. |
| `v3/crates/rvllm-cutlass` | CUTLASS/cuBLASLt FP8 dispatch, SHA-256 authenticated `Policy` table, and the **Jesse PTX-CUDA `1,000D x 100W` Ultra Kernel Tuner** (`ptx_tuner` + `rvllm_ptx_tuner` binary). |
| `v3/crates/rvllm-attention` | Paged decode attention and FlashAttention-3 prefill dispatch. |
| `v3/crates/rvllm-fused` | Fused RMSNorm, RoPE, GEGLU/SwiGLU, quantization, and host reference golden functions. |
| `v3/crates/rvllm-metadata` | Device metadata tables and layout hashes. |
| `v3/crates/rvllm-graph` | CUDA graph capture pools and replay invariants. |
| `v3/crates/rvllm-loader` | SafeTensors weight loading and validation. |
| `v3/crates/rvllm-sampling` | Token sampling and logit processors. |
| `v3/crates/rvllm-runtime` | End-to-end model execution plans and batch scheduler. |
| `v3/crates/rvllm-serve` | Loopback HTTP server (`/health`, `/status`, `/metrics`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`). |
| `v3/crates/rvllm-bench` | Fixed-state decode and prefill benchmarking harness. |
| `v3/crates/rvllm-invariants` | CI enforcement of crate DAG dependencies and per-crate/per-file line budgets (`tests/dag.rs`). |
| `scripts/vast_ptx_tuner.py` | Vast.ai GPU fleet dispatcher for live hardware compilation (`nvcc`/`ptxas -v`), FP32 parity (`cosine >= 0.9999`), and CUDA-event timing. |
| `scripts/remote_gpu_eval.sh` | Remote GPU evaluation harness executed on rented Vast.ai nodes. |
| `docs/wiki/` | Structured wiki pages covering architecture, the `1,000D x 100W` PTX-CUDA tuner, Vast.ai fleet dispatch, and hardware support matrix. |

## 2. Jesse PTX-CUDA Ultra Kernel Tuner (`1,000D x 100W`)

`rvllm-cutlass::ptx_tuner` replaces static vLLM/PyTorch dispatch heuristics with an evolutionary schedule synthesis engine operating over a **`1,000D x 100W`** topology (`1,000` evolutionary generations $\times$ `100` parallel beam workers = `100,000` schedule evaluations per workload shape):

- **Uniform Hardware Descriptor (`v3/crates/rvllm-cutlass/src/ptx_tuner/hardware.rs`)**:
  Models all NVIDIA SM generations uniformly via `HardwareDescriptor` and `IsaCaps`:
  - `sm_70` (Volta V100 — `mma.sync`)
  - `sm_80` (Ampere A100 — `cp.async`, `mma.sync.m16n8k16`)
  - `sm_86` (Ampere RTX 3090)
  - `sm_89` (Ada L40S / RTX 4090 — FP8 `e4m3`/`e5m2`)
  - `sm_90a` (Hopper H100 / H200 — TMA `cp.async.bulk.tensor`, `wgmma.mma_async`, DSMEM clusters)
  - `sm_100` (Blackwell B200 — 5th-gen `tcgen05.mma`, TMEM, 2-CTA clusters, FP4/FP8 block scaling)
  - `sm_103a` (Blackwell Ultra GB300 — enhanced TMEM + 288 KiB SMEM)
  - `sm_120` (Blackwell GeForce RTX 5090 — `mma.sync` block-scaled FP4/FP8)
  - `sm_121a` (Blackwell DGX Spark GB10 — coherent LPDDR5x unified memory + block-scaled MMA)
  - Runtime hardware discovery via `HardwareDescriptor::from_device_probe(...)`.
- **8 LLM Operator Families (`v3/crates/rvllm-cutlass/src/ptx_tuner/genome.rs`)**:
  `ChannelScaleGemv`, `BlockScaledGemm`, `PagedAttentionDecode`, `FlashPrefill`, `FusedRmsNormQkvRope`, `FusedSwigluDownProj`, `MoETopKExpertDispatch`, and `SpeculativeDraftVerify`.
- **Static Correctness & Zero-Spill Gates**:
  Every candidate genome is checked before scoring:
  1. `MatchedPair` producer-consumer barrier invariant (`mbarrier.init` / `mbarrier.arrive.expect_tx` / `mbarrier.try_wait.parity` for TMA/WGMMA/TCGEN05 pipelines; `cp.async.commit_group` / `cp.async.wait_group` for Ampere/Ada).
  2. Strict register file budget (`regs_per_thread * threads_per_cta <= regs_per_sm` and `regs_per_thread <= max_regs_per_thread`) guaranteeing zero local-memory register spills (`0 bytes spill stores, 0 bytes spill loads`).
  3. Shared memory and TMEM allocation bounds within `HardwareDescriptor` limits.
- **Authenticated `rvllm_cutlass::Policy` Output**:
  Winning schedules are exported via `build_authenticated_policy(...)` into a `rvllm.cutlass.policy.v1` JSON document sealed with its canonical SHA-256 digest (`policy.verify_integrity()`).

### Running the Tuner Locally

```bash
cargo run --release --locked --manifest-path v3/Cargo.toml \
  -p rvllm-cutlass --bin rvllm_ptx_tuner -- \
  --depth 1000 --width 100 \
  --emit-bundle ./artifacts/ptx-tuner
```

### Dispatching Live GPU Verification on Vast.ai

```bash
# Dry-run ($0 spend, verifies manifest & offer query formatting):
python3 scripts/vast_ptx_tuner.py --mode dry-run --depth 1000 --width 100 \
  --emit-bundle ./artifacts/vast-ptx-tuner

# Live Vast.ai dispatch with automatic instance teardown on exit/SIGINT/SIGTERM:
python3 scripts/vast_ptx_tuner.py dispatch \
  --manifest ./artifacts/vast-ptx-tuner/vast_work_units.json \
  --out ./artifacts/vast-ptx-tuner/results --pilot
```

## 3. Mandatory Pre-PR Validation Checklist

Before opening or merging any pull request, run:

```bash
./scripts/validate-local.sh
python3 tests/test_vast_ptx_tuner.py
```

`./scripts/validate-local.sh` enforces:
1. **Release-surface hygiene**: Rejects local user paths, SSH paths, disabled SSH host-key checking (`accept-new` is required), private keys, and Hugging Face tokens.
2. **Attribution records**: Verifies `LICENSES/` commit pins.
3. **Python & Shell syntax**: Runs `py_compile` across all Python tools and `bash -n` on shell scripts.
4. **Rust formatting, compilation & unit tests**:
   - `cargo fmt --manifest-path v3/Cargo.toml --all -- --check`
   - `cargo check --manifest-path v3/Cargo.toml --workspace --all-targets --locked`
   - `cargo test --manifest-path v3/Cargo.toml --workspace --locked` (including `rvllm-invariants` DAG and source-size budget checks).
   - `cargo check --manifest-path chat-client/Cargo.toml --locked`
