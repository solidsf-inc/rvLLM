# rvLLM CUDA kernels

This directory builds the PTX and shared libraries loaded by the rvLLM CUDA
runtime. Generated artifacts are not committed; each PTX bundle includes a
manifest with its source revision, ABI, byte length, and SHA-256 digest.

The validated H100 / sm_90 bundle is attached to the
[v0.3.0 release](https://github.com/solidsf-inc/rvLLM/releases/tag/v0.3.0).
Its archive SHA-256 is
`97a8a8de4218733514e493193c4f3182ff5b309bf9225fd135e9b64231e6d250`;
runtime and benchmark receipts are in
[`docs/receipts/h100-2026-07-13/`](../docs/receipts/h100-2026-07-13/).

## PTX bundle

Install a CUDA toolkit that supports the target architecture, then build from
a committed checkout:

```bash
./kernels/build.sh sm_90
```

`build.sh` compiles its explicit source allowlist from `kernels/` and
`v3/kernels/` into `kernels/<arch>/`. It fails if a required source, revision,
or tool is missing. The generated manifest prints the trust values required by
the runtime. Hash `policy.json` separately and set `RVLLM_POLICY_SHA256` to that
digest.

## CUTLASS libraries

CUTLASS is pinned as a submodule. Initialize it before running the relevant
`build_cutlass*.sh` or `build_w4a8.sh` script:

```bash
git submodule update --init --recursive
./kernels/build_cutlass_so.sh
```

`build_cutlass_so.sh` is Hopper SM90-only. SM121 uses
`build_cutlass_sm120_so.sh`; other targets use the runtime's cuBLASLt/PTX
fallbacks. Each script validates the expected CUTLASS revision before
compiling.

## FlashAttention 3 for SM90

The optional FA3 build requires a separate checkout at the revision enforced
by `build_fa3.sh`:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git flash-attention
git -C flash-attention checkout 1233b73b6c95340c65c9edfe929611838354fc6e
./kernels/build_fa3.sh
```

The script validates both FlashAttention and CUTLASS revisions, compiles the
required head-dimension instantiations, and checks the exported ABI symbols.
See `LICENSES/` for their BSD-3-Clause notices.

CI runs `ci_compile_release_cuda.sh` in a digest-pinned CUDA 13 container for
both SM90 and SM121. It compiles and links every release PTX/shared-library
artifact without claiming that the CPU-only runner executes GPU kernels.
