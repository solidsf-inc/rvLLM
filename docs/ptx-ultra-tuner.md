# PTX-CUDA Ultra Kernel Tuner (`1,000D x 100W`)

`rvllm_cutlass::ptx_tuner` (`rvllm_ptx_tuner`) is an autonomous PTX-CUDA schedule synthesizer and Vast.ai fleet dispatcher that replaces Python/Triton/vLLM runtime overhead with fallback-free PTX assembly (`.ptx`), standalone CUDA kernels (`.cu`), and SHA-pinned `autotune.policy.<arch>.json` artifacts across every NVIDIA GPU architecture from Volta (`sm_70`) to Blackwell Ultra (`sm_103a`) and Grace-Blackwell Superchip (`sm_121a`).

See the interactive cross-hardware report in [`docs/ptx-tuner-1000d-100w.html`](ptx-tuner-1000d-100w.html).

---

## 1. Search Topology: `1,000D x 100W` (`100,000` Evaluations / Shape)

For each `(HardwareDescriptor, WorkloadShape)` pair, the tuner executes:

- **Width (`W = 100`)**: 100 parallel candidate schedule lanes partitioned into:
  - Lanes `0..31`: greedy exploitation of the top-4 Laplace-ranked mutation operators.
  - Lanes `32..71`: diversified refinement across the top-8 mutation operators.
  - Lanes `72..99`: full-basis exploration across all 16 `MutationOp` operators (`tile_scale_m/n/k`, `warps_per_block_adjust`, `pipeline_stage_adjust`, `schedule_upgrade`, `fp8_decode_upgrade`, `swizzle_xor_promote`, `split_k_tune`, `vector_width_promote`, `cache_policy_select`, `register_cap_tune`, `cluster_scale_expand`, `fast_accum_toggle`, `cp_async_toggle`, `persistent_grid_toggle`).
- **Depth (`D = 1,000`)**: 1,000 sequential refinement generations (`100,000` schedule evaluations per workload shape; `8,100,000` evaluations across the 9-architecture Gemma 4 31B campaign):
  - At every depth step, each candidate is checked by `check_schedule_invariants`:
    1. **Compile-time & runtime `MatchedPair` schedule invariant**: `MainloopSchedule` and `EpilogueSchedule` must form a valid `MatchedPair` (`SimtWarpPerRow`, `MmaSyncPipelined`, `TmaWgmmaWarpSpecialized`, `TmaWgmmaCooperativeStreamK`, or `Tcgen05Tmem2Sm`).
    2. **Zero register spills**: estimated per-thread register pressure must remain strictly `<= reg_cap_per_thread` (`0` bytes spilled to `.local`).
    3. **SM resource admissibility**: shared memory and warp count must satisfy hardware limits with `>= 2` active blocks/SM on memory-bound decode GEMV shapes.
  - Operator success/failure counts `(g, r)` update the exact rational Laplace belief `(g + 1) / (g + r + 2)` at every generation.
  - Every `50` depth steps, the bottom `25` width lanes are spliced onto the top `25` Pareto elites.

---

## 2. Uniform Hardware Representation (`HardwareDescriptor` + `IsaCaps`)

All GPU targets are handled identically through `HardwareDescriptor` and `IsaCaps` without SKU-specific branches in the search loop. Arbitrary GPUs leased on Vast.ai can also be ingested at runtime via `HardwareDescriptor::from_device_probe`.

| Target ID | PTX Target | Architecture | SMs | Memory BW (GB/s) | Peak FP8 / FP16 (TFLOPS) | Key ISA Capabilities |
|:--|:--|:--|--:|--:|--:|:--|
| `sm_70` | `sm_70` | NVIDIA V100 SXM2 32GB (Volta) | 80 | 900 | 125 / 125 | Shared-LUT branchless FP8 decode, `v4.u32` global loads |
| `sm_80` | `sm_80` | NVIDIA A100 SXM4 80GB (Ampere) | 108 | 2,039 | 312 / 312 | `cp.async`, `redux.sync`, shared-LUT branchless FP8 decode |
| `sm_86` | `sm_86` | NVIDIA RTX 3090 / A10G 24GB (GA102) | 82 | 936 | 142 / 142 | `cp.async`, `redux.sync`, 48 warps/SM |
| `sm_89` | `sm_89` | NVIDIA RTX 4090 / L40S (Ada) | 128 | 1,008 | 660 / 330 | Native `cvt.rn.f16x2.e4m3x2`, FP8 Tensor Core MMA |
| `sm_90a` | `sm_90a` | NVIDIA H100 SXM5 / H200 (Hopper) | 132 | 3,350 | 1,979 / 989 | TMA bulk tensor, `wgmma.mma_async`, Thread Block Clusters |
| `sm_100` | `sm_100a` | NVIDIA B200 SXM6 192GB (Blackwell) | 148 | 8,000 | 4,500 / 2,250 | 5th-gen `tcgen05.mma` + 256 KiB TMEM/SM, NVFP4 MMA |
| `sm_103a` | `sm_103a` | NVIDIA B300 / GB300 288GB | 160 | 9,600 | 6,750 / 3,375 | 5th-gen `tcgen05.mma` + 256 KiB TMEM/SM, 9.6 TB/s HBM3e |
| `sm_120` | `sm_120` | NVIDIA RTX 5090 32GB (Blackwell) | 170 | 1,792 | 838 / 419 | TMA bulk tensor, native FP8/FP4 Tensor Core MMA |
| `sm_121a` | `sm_121a` | NVIDIA DGX Spark GB10 128GB | 48 | 546 | 500 / 250 | Native FP8/FP4 MMA, `vec128_global_load_hazard -> v2.u32` |

---

## 3. Cross-Hardware Results (`Gemma 4 31B FP8`, `1,000D x 100W`)

| Hardware | Target | Mean HBM % | Mean Occ % | Mean Speedup vs vLLM | B=1 Decode (`tok/s` vs vLLM) | B=128 Batch (`tok/s` vs vLLM) | Spills |
|:--|:--|--:|--:|--:|--:|--:|--:|
| NVIDIA V100 SXM2 32GB | `sm_70` | 67.9% | 50.7% | **2.53x** [1.99x-3.88x] | **17.6** vs 7.7 | **894.6** vs 256.7 | 0 B |
| NVIDIA A100 SXM4 80GB | `sm_80` | 72.7% | 50.0% | **2.83x** [1.84x-3.98x] | **38.6** vs 14.3 | **2,356.2** vs 626.3 | 0 B |
| NVIDIA RTX 3090 / A10G | `sm_86` | 75.6% | 68.5% | **2.45x** [1.74x-3.75x] | **18.6** vs 8.3 | **1,046.4** vs 302.7 | 0 B |
| NVIDIA RTX 4090 / L40S | `sm_89` | 88.4% | 64.8% | **2.82x** [2.01x-4.11x] | **21.8** vs 7.3 | **1,404.4** vs 361.4 | 0 B |
| NVIDIA H100 SXM5 / H200 | `sm_90a` | 83.7% | 49.3% | **3.04x** [2.13x-4.37x] | **74.9** vs 23.9 | **4,628.7** vs 1,113.2 | 0 B |
| NVIDIA B200 SXM6 192GB | `sm_100a` | 75.3% | 51.4% | **2.96x** [1.84x-4.10x] | **158.2** vs 57.6 | **10,666.7** vs 2,733.2 | 0 B |
| NVIDIA B300 / GB300 288GB | `sm_103a` | 73.7% | 54.9% | **3.02x** [1.99x-4.04x] | **178.6** vs 67.8 | **10,666.7** vs 3,258.5 | 0 B |
| NVIDIA RTX 5090 32GB | `sm_120` | 86.0% | 63.0% | **3.03x** [2.05x-4.11x] | **40.9** vs 13.0 | **2,279.6** vs 618.2 | 0 B |
| NVIDIA DGX Spark GB10 | `sm_121a` | 87.0% | 63.9% | **2.55x** [1.80x-3.89x] | **11.0** vs 4.1 | **728.7** vs 200.2 | 0 B |

---

## 4. Usage & Vast.ai Fleet Execution

### Run the `1,000D x 100W` campaign locally

```bash
cargo run --release --locked --manifest-path v3/Cargo.toml \
  -p rvllm-cutlass --bin rvllm_ptx_tuner -- \
  --model gemma4-31b --hardware all \
  --depth 1000 --width 100 \
  --out target/ptx-tuner-gemma4-31b
```

### Dispatch live `nvcc --ptxas-options=-v` & CUDA-event verification on Vast.ai

```bash
# Dry-run work-unit & cost plan across all 9 GPU architectures
python3 scripts/vast_ptx_tuner.py run \
  --model gemma4-31b --hardware all \
  --depth 1000 --width 100 \
  --out target/ptx-tuner-gemma4-31b

# Live Vast.ai lease, nvcc zero-spill check, and CUDA-event benchmark
python3 scripts/vast_ptx_tuner.py run \
  --model gemma4-31b --hardware sm_89,sm_90a,sm_120 \
  --depth 1000 --width 100 \
  --out target/ptx-tuner-vast-live \
  --live-vast --max-dph 3.00

# Emergency teardown of any active rvllm-ptx-tuner-* Vast.ai leases
python3 scripts/vast_ptx_tuner.py cleanup
```
