# Profiling rvLLM

Build the current CUDA benchmark from `v3`:

```bash
cargo build --manifest-path v3/Cargo.toml --release --locked \
  --features cuda -p rvllm-bench
```

Set the model and compiled artifact variables documented by
`rvllm-bench --help`, plus public receipt metadata:

```bash
export RVLLM_MODEL_DIR=/path/to/model
export RVLLM_KERNELS_DIR=/path/to/kernels
export RVLLM_MODEL_ID=publisher/model@revision
export RVLLM_MODEL_SHA256='<64-character model artifact digest>'
export RVLLM_SOURCE_SHA="$(git rev-parse HEAD)"
export RVLLM_HARDWARE='exact GPU and memory configuration'
export RVLLM_DRIVER='exact driver version'
export RVLLM_TOOLCHAIN="$(rustc --version)"
scripts/profile_compare.sh --batches 1,8,32 --nsys
```

For an Nsight Compute capture, run `scripts/full_stack_profile.sh`. Both tools
write hashes for their raw outputs. Treat profiler captures as untrusted until
the receipt and source tree match; do not publish local model paths or host
identifiers.
