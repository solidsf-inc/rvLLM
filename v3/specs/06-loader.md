# Loader contract

Model directories are untrusted. Resolve every index entry beneath the
configured root, reject absolute/traversal/symlink escapes, cap index size,
shard count, tensor count, rank, name length, and total mapped bytes, and use
checked arithmetic for offsets, dimensions, alignment, and allocations.

For each tensor, validate dtype, exact shape, byte extent, non-overlap where
required, and scale layout before upload. Unknown quantization metadata is an
error. Do not scan arbitrary shards for undeclared tensors or fetch missing
files from the network.

FP8 conversion must match reviewed E4M3FN vectors, including signed zero,
subnormals, ties, saturation, infinity, and NaN. `scripts/test_fp8_encode.py`
is an external oracle; Rust unit tests must carry the same vectors. Fuzzed and
truncated safetensors/index fixtures are release gates.
