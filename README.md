# rvLLM

Gemma 4 inference in Rust.

rvLLM keeps model loading, scheduling, KV-cache management, sampling, and serving in native code. This release contains a CUDA runtime and server, plus Metal, vision, and XLA/TPU work whose target-specific validation status is stated below.

## Release boundary

| Backend | Status |
| --- | --- |
| NVIDIA CUDA | Primary runtime source. Build and validate a complete kernel bundle on the target GPU before serving. |
| Apple Metal | Backend source behind `metal`; end-to-end generation and parity are not asserted by this release. |
| Vision | Integration source is present; end-to-end Gemma 4 vision is not asserted by this release. |
| XLA/TPU | Experimental operator and conformance work only; no TPU server is shipped in this release. |

Required accelerator artifacts are loaded explicitly. Missing or incompatible kernels are errors, not silent fallbacks.

## Build

```bash
git submodule update --init --recursive
cargo build --release --locked --manifest-path v3/Cargo.toml -p rvllm-serve
```

CUDA:

```bash
# Build the target-specific artifacts described in kernels/README.md first.
cargo build --release --locked --manifest-path v3/Cargo.toml \
  -p rvllm-serve --features cuda,cublaslt
```

Apple Silicon:

```bash
cargo build --release --locked --manifest-path v3/Cargo.toml \
  -p rvllm-serve --features metal
```

Run `rvllm-server --help` for the model, kernel, authentication, and backend settings:

```bash
./v3/target/release/rvllm-server --help
```

The server binds to loopback and exposes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Generation is non-streaming; requests with `stream=true` or `stop` are rejected.

Use a trusted reverse proxy for remote access. Set `RVLLM_API_KEY` to require bearer authentication.

## Validate

```bash
cargo fmt --manifest-path v3/Cargo.toml --all -- --check
cargo test --locked --manifest-path v3/Cargo.toml --workspace --no-default-features
```

Accelerator claims require runtime loading, numerical parity, and edge-case tests on the advertised hardware. Compilation alone is not a passing result.

On a CUDA host, verify the context and HBM-copy lifecycle directly:

```bash
RVLLM_RUN_CUDA_CONTEXT_SMOKE=1 cargo test --locked --manifest-path v3/Cargo.toml \
  -p rvllm-mem --features cuda --test cuda_context_smoke -- --ignored
```

## Layout

- `v3/crates/` — Rust runtime, server, loader, scheduler, sampling, vision, and backends
- `v3/kernels/` — runtime CUDA kernels
- `kernels/` — accelerator build and artifact tooling
- `tpu/` — experimental XLA/TPU operator work
- `tests/` — API and parity checks
- `docs/` — architecture and measured-result methodology

## Benchmarks

[`docs/bench.html`](docs/bench.html) preserves the project-reported TPU snapshot and states its provenance limits. The prior H100 rows were removed after harness revalidation; replacements require the exact rvLLM commit, model revision, hardware, precision, inputs, warmup, synchronization, token accounting, parity result, and raw samples.

## License

Apache-2.0. Third-party notices are in [`LICENSES/`](LICENSES/).
