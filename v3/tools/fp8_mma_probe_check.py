#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/fp8_mma_probe_check.py [sm_xxx]
#
# Validates `fp8_e4m3_mma_probe_kernel`: the standalone one-warp FP8
# E4M3 tensor-core MMA. Pass criteria:
#
#   * PTX loads + launches on the target arch (assembly + link check).
#   * D fragment read back from the kernel matches an fp64 reference
#     A @ B^T with scale_rel (abs_err / mean|ref|) <= 5e-2 — i.e.
#     within FP8-quant noise.
#
# Fragment layout under test (per `kernels/fp8_e4m3_mma_probe.cu`,
# which defines the probe's fragment contract):
#
#   A (m16, k32, row-major):   lane i  rows {i/4, i/4+8}
#                              a[0..3] as 4 × u32 spanning k=(i%4)*8..+8
#   B (n8,  k32, col-major):   lane i  col (i/4)
#                              b[0..1] as 2 × u32 spanning k=(i%4)*8..+8
#   D (m16, n8,  f32):         lane i  rows {i/4, i/4+8}
#                              d[0..3] at cols (i%4)*2 + [0, 1]

import argparse, atexit, hashlib, json, math, pathlib, re, sys
import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("arch", nargs="?", default="sm_121")
args = parser.parse_args()
ARCH = args.arch
if not re.fullmatch(r"sm_[0-9]{2,3}", ARCH):
    parser.error("arch must look like sm_90 or sm_121")
PTX = REPO / "kernels" / ARCH / "fp8_e4m3_mma_probe.ptx"
if not PTX.is_file() or PTX.is_symlink():
    sys.exit(f"missing or unsafe PTX: {PTX}")


def CHECK(res, what):
    if isinstance(res, tuple):
        err, *rest = res
    else:
        err, rest = res, ()
    if err != drv.CUresult.CUDA_SUCCESS:
        _, name = drv.cuGetErrorName(err)
        sys.exit(f"{what} failed: {err} ({name.decode() if name else '?'})")
    return rest[0] if len(rest) == 1 else tuple(rest) if rest else None


CHECK(drv.cuInit(0), "cuInit")
dev = CHECK(drv.cuDeviceGet(0), "cuDeviceGet")
ctx = CHECK(drv.cuDevicePrimaryCtxRetain(dev), "cuDevicePrimaryCtxRetain")
CHECK(drv.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
allocations = []
modules = []


def cleanup():
    for ptr in reversed(allocations):
        drv.cuMemFree(ptr)
    for module in reversed(modules):
        drv.cuModuleUnload(module)
    drv.cuDevicePrimaryCtxRelease(dev)


atexit.register(cleanup)
cc_major = CHECK(drv.cuDeviceGetAttribute(
    drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "cc")
cc_minor = CHECK(drv.cuDeviceGetAttribute(
    drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "cc")
driver_version = CHECK(drv.cuDriverGetVersion(), "cuDriverGetVersion")
print(f"device: cc {cc_major}.{cc_minor}, PTX: {PTX.name}")
if ARCH != f"sm_{cc_major}{cc_minor}":
    sys.exit(f"PTX arch {ARCH} does not match device sm_{cc_major}{cc_minor}")

ptx_bytes = PTX.read_bytes() + b"\0"
mod = CHECK(drv.cuModuleLoadData(ptx_bytes), "cuModuleLoadData")
modules.append(mod)
print(json.dumps({"ptx_sha256": hashlib.sha256(ptx_bytes[:-1]).hexdigest(), "seed": 13,
                  "device_cc": f"{cc_major}.{cc_minor}",
                  "cuda_driver_version": driver_version}, sort_keys=True))
fn = CHECK(drv.cuModuleGetFunction(mod, b"fp8_e4m3_mma_probe_kernel"),
           "cuModuleGetFunction")

# --- FP8 E4M3 round-trip identical to the kernel's `fp8kv_decode_byte` --
def e4m3_to_f32(b):
    if b == 0 or b == 0x80:
        return 0.0
    sign = -1.0 if (b & 0x80) else 1.0
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    if exp == 0:
        return sign * mant * (2.0 ** -9)
    if exp == 0xF and mant == 0x7:
        return math.nan
    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))


FP8_MAX = 448.0


def f32_to_e4m3(v):
    sign = 1 if math.copysign(1.0, v) < 0.0 else 0
    if math.isnan(v):
        return (sign << 7) | 0x7F
    a = abs(v)
    if a == 0.0:
        return (sign << 7)
    if not math.isfinite(a) or a > FP8_MAX:
        return (sign << 7) | 0x7E
    if a < 2.0 ** -6:
        mant_bits = round(a * 512.0)
        if mant_bits == 0:
            return sign << 7
        if mant_bits >= 8:
            return (sign << 7) | 0x08
        return (sign << 7) | mant_bits
    e = math.floor(math.log2(a))
    exp_bits = e + 7
    mant_bits = round((a / (2.0 ** e) - 1.0) * 8.0)
    if mant_bits == 8:
        mant_bits = 0
        exp_bits += 1
    if exp_bits > 15 or (exp_bits == 15 and mant_bits > 6):
        exp_bits, mant_bits = 15, 6
    return (sign << 7) | ((exp_bits & 0xF) << 3) | (mant_bits & 0x7)


# --- Test inputs: small-magnitude random values that round cleanly ---
rng = np.random.default_rng(13)
A_f32 = (rng.normal(0, 0.25, (16, 32))).astype(np.float32)
B_f32 = (rng.normal(0, 0.25, (8, 32))).astype(np.float32)

# Quantise to FP8 bytes
A_b = np.zeros((16, 32), dtype=np.uint8)
B_b = np.zeros((8, 32), dtype=np.uint8)
for m in range(16):
    for k in range(32):
        A_b[m, k] = f32_to_e4m3(float(A_f32[m, k]))
for n in range(8):
    for k in range(32):
        B_b[n, k] = f32_to_e4m3(float(B_f32[n, k]))

# Round-trip the bytes so the reference uses what the kernel will see
A_rt = np.vectorize(e4m3_to_f32, otypes=[np.float64])(A_b)
B_rt = np.vectorize(e4m3_to_f32, otypes=[np.float64])(B_b)

# Reference: D = A @ B^T in fp64
D_ref = A_rt @ B_rt.T  # shape (16, 8)

# --- Pack A into per-lane fragments ---------------------------------------
# Lane i ∈ [0, 31]:
#   rows     = {i/4, i/4+8}
#   k_group  = i % 4
#   a[0]: row (i/4),    k = k_group*8 + [0..3]
#   a[1]: row (i/4+8),  k = k_group*8 + [0..3]
#   a[2]: row (i/4),    k = k_group*8 + [4..7]
#   a[3]: row (i/4+8),  k = k_group*8 + [4..7]
a_frag = np.zeros((32, 4), dtype=np.uint32)
for lane in range(32):
    r_lo = lane // 4
    r_hi = r_lo + 8
    k_base = (lane % 4) * 8
    def pack4(row, k0):
        u = 0
        for j in range(4):
            u |= int(A_b[row, k0 + j]) << (j * 8)
        return u
    a_frag[lane, 0] = pack4(r_lo, k_base + 0)
    a_frag[lane, 1] = pack4(r_hi, k_base + 0)
    a_frag[lane, 2] = pack4(r_lo, k_base + 4)
    a_frag[lane, 3] = pack4(r_hi, k_base + 4)

# --- Pack B into per-lane fragments ---------------------------------------
# Lane i: col = i/4, k_group = i%4
#   b[0]: k = k_group*8 + [0..3]
#   b[1]: k = k_group*8 + [4..7]
b_frag = np.zeros((32, 2), dtype=np.uint32)
for lane in range(32):
    n = lane // 4
    k_base = (lane % 4) * 8
    def pack4b(col, k0):
        u = 0
        for j in range(4):
            u |= int(B_b[col, k0 + j]) << (j * 8)
        return u
    b_frag[lane, 0] = pack4b(n, k_base + 0)
    b_frag[lane, 1] = pack4b(n, k_base + 4)

# --- GPU launch -----------------------------------------------------------


def alloc(n):
    if n <= 0 or n > 1024**3:
        sys.exit(f"invalid allocation size: {n}")
    ptr = CHECK(drv.cuMemAlloc(n), "cuMemAlloc")
    allocations.append(ptr)
    return ptr


def h2d(dst, arr):
    arr = np.ascontiguousarray(arr)
    CHECK(drv.cuMemcpyHtoD(dst, arr.ctypes.data, arr.nbytes), "HtoD")


d_a = alloc(a_frag.nbytes); h2d(d_a, a_frag)
d_b = alloc(b_frag.nbytes); h2d(d_b, b_frag)
d_d = alloc(32 * 4 * 4)  # 32 lanes × 4 f32

params = [
    np.array([int(d_a)], dtype=np.uint64),
    np.array([int(d_b)], dtype=np.uint64),
    np.array([int(d_d)], dtype=np.uint64),
]
param_ptrs = np.array([p.ctypes.data for p in params], dtype=np.uint64)

CHECK(drv.cuLaunchKernel(
    fn,
    1, 1, 1,
    32, 1, 1,   # one warp
    0, 0,
    param_ptrs.ctypes.data, 0,
), "cuLaunchKernel")
CHECK(drv.cuCtxSynchronize(), "cuCtxSynchronize")

d_frag = np.empty((32, 4), dtype=np.float32)
CHECK(drv.cuMemcpyDtoH(d_frag.ctypes.data, d_d, d_frag.nbytes), "DtoH")

# --- Unpack D fragment into [16, 8] f32 -----------------------------------
# Lane i: rows {i/4, i/4+8}, cols (i%4)*2 + [0, 1]
#   d[0]: row (i/4),    col (i%4)*2 + 0
#   d[1]: row (i/4),    col (i%4)*2 + 1
#   d[2]: row (i/4+8),  col (i%4)*2 + 0
#   d[3]: row (i/4+8),  col (i%4)*2 + 1
D = np.full((16, 8), np.nan, dtype=np.float32)
for lane in range(32):
    r_lo = lane // 4
    r_hi = r_lo + 8
    c = (lane % 4) * 2
    D[r_lo, c + 0] = d_frag[lane, 0]
    D[r_lo, c + 1] = d_frag[lane, 1]
    D[r_hi, c + 0] = d_frag[lane, 2]
    D[r_hi, c + 1] = d_frag[lane, 3]

if not np.isfinite(D).all():
    sys.exit(f"FAIL: output contains NaN or infinity.\n{D}")

# --- Compare --------------------------------------------------------------
D_ref_f32 = D_ref.astype(np.float32)
abs_err = np.abs(D - D_ref_f32)
ref_mean_abs = float(np.abs(D_ref_f32).mean())
scale_rel = abs_err / max(ref_mean_abs, 1e-30)

print(f"A, B: random N(0, 0.25), shape A={A_f32.shape} B={B_f32.shape}")
print(f"ref (fp64 A@B.T):  range [{D_ref_f32.min():+.3e}, {D_ref_f32.max():+.3e}], "
      f"|ref| mean {ref_mean_abs:.3e}")
print(f"kernel D:          range [{D.min():+.3e}, {D.max():+.3e}]")
print(f"abs_err:  max {abs_err.max():.3e}  mean {abs_err.mean():.3e}")
print(f"scale_rel: max {scale_rel.max():.3e}  mean {scale_rel.mean():.3e}")

THRESHOLD = 5e-2
if scale_rel.max() > THRESHOLD:
    print(f"\nFAIL: scale_rel.max {scale_rel.max():.3e} > {THRESHOLD:.0e}")
    print("mma output:")
    print(D)
    print("ref:")
    print(D_ref_f32)
    sys.exit(1)
print(f"\nOK: scale_rel.max {scale_rel.max():.3e} <= {THRESHOLD:.0e}")
