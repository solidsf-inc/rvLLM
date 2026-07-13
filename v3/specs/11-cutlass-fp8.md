# CUTLASS FP8 dispatch

rvLLM uses compiled CUTLASS/cuBLASLt libraries for selected FP8 GEMMs and keeps
custom kernels for other shapes. A source file or design entry does not imply
that a variant is shipped or selected.

Each loaded library needs an immutable source revision, compiler/toolchain,
target architecture, exported ABI version, symbol list, build command, and
digest. Policy files are untrusted: validate schema, dimensions, dtypes,
alignment, workspace, architecture, and that the selected symbol belongs to the
verified library. Prebuilt artifacts require a public signature channel.

Launch wrappers must check every extent and leading dimension, keep descriptor
lifetimes through completion, and fail closed on unavailable epilogues or
unsupported schedules. Reference parity and eager/graph tests are required for
every policy entry before it becomes a default.
