# rvLLM XLA/TPU operator work

This directory contains experimental JAX reference operators and StableHLO
lowering tools. It is not an end-to-end rvLLM TPU server, and the checked-in
ports do not yet carry CUDA/TPU parity evidence.

`manifest.toml` is an explicit inventory of selected operators. A `todo`
status means the operator is unverified and must not be treated as a supported
runtime path.

## Host-side inspection

```bash
cd tpu
python3 -m pip install -e .
make status
make test
make emit
```

Emission lowers the listed JAX functions to StableHLO on the host. `make
verify` fails closed unless the required comparison environment and evidence
are available. TPU timing scripts synchronize device work before reporting
diagnostic measurements.

## Layout

- `manifest.toml` — selected operator inventory and verification status
- `ports/` — JAX reference implementations
- `harness/` — lowering, inspection, and comparison tools
- `out/` — generated StableHLO output; not committed

Two comparison helpers in `harness/` are not TPU implementations:
`bench_driver.py` measures an external HTTP server, and
`bench_rvllm_multi.sh` is an NVIDIA/CUDA diagnostic. Neither is TPU evidence.
