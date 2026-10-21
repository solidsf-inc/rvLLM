# rvLLM Wiki

Welcome to the **rvLLM** documentation wiki. `rvLLM` is a native Rust inference engine and PTX-CUDA kernel synthesis platform for Gemma 4 and modern transformer/MoE architectures.

## Navigation

- **[Architecture Overview](../arch.md)** — Crate DAG (`v3/Cargo.toml`), memory hierarchy, loopback server surface, and runtime execution pipeline.
- **[PTX-CUDA Ultra Kernel Tuner (`1,000D x 100W`)](PTX-CUDA-Ultra-Kernel-Tuner.md)** — Evolutionary schedule synthesis (`100,000` evaluations per workload shape), `MatchedPair` producer-consumer barrier verification, zero-spill register invariants, and SHA-256 `Policy` generation.
- **[Vast.ai GPU Fleet Dispatch](Vast-AI-Fleet-Dispatch.md)** — Ephemeral multi-architecture GPU benchmarking (`scripts/vast_ptx_tuner.py` and `scripts/remote_gpu_eval.sh`), `ptxas -v` zero-spill validation, FP32 cosine parity (`>= 0.9999`), and teardown safety.
- **[Hardware Support Matrix (`sm_70` to `sm_121a`)](Hardware-Support-Matrix.md)** — Uniform `HardwareDescriptor` and `IsaCaps` across Volta, Ampere, Ada, Hopper, and Blackwell (`B200`, `GB300`, `RTX 5090`, `DGX Spark GB10`), plus runtime `from_device_probe`.
- **[Interactive Reports & Benchmarks](../bench.html)**:
  - [`docs/bench.html`](../bench.html) — Measured H100 fixed-state decode sweep and TPU v6e-4 snapshot.
  - [`docs/ptx-tuner-1000d-100w.html`](../ptx-tuner-1000d-100w.html) — Interactive `1,000D x 100W` PTX-CUDA schedule convergence and roofline report.
  - [`docs/ptx-ultra-tuner.md`](../ptx-ultra-tuner.md) — Full technical specification of the PTX-CUDA Ultra Kernel Tuner.
- **[Agent & Contributor Guide (`AGENTS.md`)](../../AGENTS.md)** — Workspace rules, DAG invariants, and local validation commands.
