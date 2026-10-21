# Hardware Support Matrix (`sm_70` to `sm_121a`)

`rvllm-cutlass::ptx_tuner::hardware` models every NVIDIA GPU architecture through a uniform `HardwareDescriptor` and `IsaCaps` capability vector, plus runtime hardware discovery (`HardwareDescriptor::from_device_probe`) for unseen SKU variants.

## Cataloged Architectures

| Target ID | Architecture | Representative GPU | SMs | HBM / Memory BW | Shared Mem / SM | Async Copy / Tensor Core ISA |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `sm_70` | Volta | NVIDIA V100 SXM2 32GB | 80 | 900 GB/s | 96 KiB | Global-to-reg + `mma.sync.aligned.m8n8k4` |
| `sm_80` | Ampere | NVIDIA A100 SXM4 80GB | 108 | 2,039 GB/s | 164 KiB | `cp.async.cg.shared.global` + `mma.sync.m16n8k16` |
| `sm_86` | Ampere | NVIDIA RTX 3090 24GB | 82 | 936 GB/s | 100 KiB | `cp.async.cg.shared.global` + `mma.sync.m16n8k16` |
| `sm_89` | Ada Lovelace | NVIDIA L40S / RTX 4090 | 142 | 1,008 GB/s | 100 KiB | `cp.async` + FP8 (`e4m3`/`e5m2`) `mma.sync.m16n8k32` |
| `sm_90a` | Hopper | NVIDIA H100 SXM5 80GB | 132 | 3,350 GB/s | 228 KiB | TMA (`cp.async.bulk.tensor`) + `wgmma.mma_async` + DSMEM clusters |
| `sm_100` | Blackwell | NVIDIA B200 SXM6 192GB | 148 | 8,000 GB/s | 256 KiB | TMA + 5th-gen `tcgen05.mma` + 256 KiB TMEM + 2-CTA clusters |
| `sm_103a` | Blackwell Ultra | NVIDIA GB300 SXM6 288GB | 160 | 12,000 GB/s | 288 KiB | TMA + `tcgen05.mma` + enhanced TMEM + 2-CTA clusters |
| `sm_120` | Blackwell GeForce | NVIDIA RTX 5090 32GB | 170 | 1,792 GB/s | 128 KiB | TMA + block-scaled FP4/FP8 `mma.sync` |
| `sm_121a` | Blackwell Spark | NVIDIA DGX Spark GB10 | 80 | 546 GB/s | 160 KiB | Coherent LPDDR5x unified memory + block-scaled `mma.sync` |

## Runtime Hardware Probe (`from_device_probe`)

For any SKU not in the static catalog, `HardwareDescriptor::from_device_probe` constructs a valid `HardwareDescriptor` directly from runtime CUDA driver attributes (`cc_major`, `cc_minor`, `sm_count`, `shared_mem_per_sm_bytes`, `regs_per_sm`, `hbm_bw_gbps`, `fp16_tflops`, `fp8_tflops`), automatically inferring `IsaCaps` from the compute capability.
