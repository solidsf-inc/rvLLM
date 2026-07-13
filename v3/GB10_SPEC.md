# GB10 backend status

GB10 support is selected by the `gb10` Cargo feature and the runtime's
compute-capability checks. The code contains sm_121 kernel loading, FP8 KV
attention, and GB10-specific dispatch. Unsupported artifacts,
head dimensions, or launch shapes must return errors rather than select an
unrelated backend.

Build check:

```bash
cargo check --manifest-path v3/Cargo.toml --locked \
  -p rvllm-runtime -p rvllm-attention --features gb10
```

Release support additionally requires real-device eager/replay parity, prefill
and decode parity for head dimensions 128/256/512, malformed-metadata tests,
and a receipt naming the GPU, driver, CUDA toolkit, source SHA, kernel manifest
digest, model revision, and command. No throughput or roofline result is
asserted here.
