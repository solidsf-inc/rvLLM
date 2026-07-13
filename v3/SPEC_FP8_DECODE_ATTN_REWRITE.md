# FP8 decode-attention rewrite

Goal: decode from paged E4M3 KV without materializing a full dequantized cache.
Each query must apply the configured attention scale, address only committed KV
slots, honor sliding/global attention, use the correct per-tensor or per-token
scales, accumulate in a reviewed higher-precision type, and write the declared
output dtype.

Required gates:

1. CPU/reference parity over random and adversarial page tables.
2. Context boundaries at 0, 1, block edges, maximum blocks, and ring wrap.
3. Head dimensions 128/256/512 and all supported query/KV head ratios.
4. Invalid pointer, extent, scale, and workspace rejection before launch.
5. Eager/graph parity and sanitizer-clean runs.
6. A receipt-bound benchmark before changing dispatch priority.

No host, job-sharing, or unpublished measurement procedure is part of this
public design.
