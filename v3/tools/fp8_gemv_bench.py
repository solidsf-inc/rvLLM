#!/usr/bin/env python3
# Usage:
#   ~/.venv/bin/python3 v3/tools/fp8_gemv_bench.py \
#       [sm_xxx] [duration_s] [N] [K]
#
# Microbench for the three warp-per-row fp8_gemv variants:
#   * fp8_gemv_blockwise_wpr_kernel        — baseline WPR, scalar decode
#   * fp8_gemv_blockwise_wpr_lut_kernel    — WPR + shared-mem LUT (throttle-friendly)
#   * fp8_gemv_blockwise_wpr_native_kernel — WPR + native cvt.rn.f16x2.e4m3x2 (sm_100+)
#
# Default shape: N=32768, K=8192 — weight bytes = 256 MB, forces
# LPDDR5X reads every iteration (L2 on GB10 is ~100 MB). Override to
# N=2048 K=5120 (~10.5 MB) to enter the L2-hot regime where the
# kernel's decode path (not LPDDR5X bandwidth) dominates.
#
# Output: text table + one JSONL record per iter to bench_<variant>.jsonl.

import argparse, atexit, hashlib, json, math, pathlib, re, subprocess, sys, time
import numpy as np
from cuda.bindings import driver as drv

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
N_DEFAULT, K_DEFAULT = 32768, 8192   # 256 MB weight, LPDDR5X-bound
parser = argparse.ArgumentParser()
parser.add_argument("arch", nargs="?", default="sm_121")
parser.add_argument("duration", nargs="?", type=float, default=5.0)
parser.add_argument("n", nargs="?", type=int, default=N_DEFAULT)
parser.add_argument("k", nargs="?", type=int, default=K_DEFAULT)
args = parser.parse_args()
ARCH, DURATION, N, K = args.arch, args.duration, args.n, args.k
if not re.fullmatch(r"sm_[0-9]{2,3}", ARCH):
    parser.error("arch must look like sm_90 or sm_121")
if not math.isfinite(DURATION) or not 0.1 <= DURATION <= 60.0:
    parser.error("duration must be finite and in 0.1..60 seconds")
if N <= 0 or K <= 0 or N > 131072 or K > 65536 or N % 128 or K % 128 or N % 8:
    parser.error("N/K exceed limits or violate the 128-element tile geometry")
if N * K > 1024**3:
    parser.error("N * K must not exceed 1,073,741,824 bytes")
M = 1
BN, BK = N // 128, K // 128
PTX = REPO / "kernels" / ARCH / "fp8_gemv.ptx"
if not PTX.is_file() or PTX.is_symlink():
    sys.exit(f"missing or unsafe PTX: {PTX}")
OUT_DIR = REPO / "v3" / "tools" / "bench_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
if OUT_DIR.is_symlink() or not OUT_DIR.is_dir():
    sys.exit(f"unsafe output directory: {OUT_DIR}")


def output_path(name):
    path = OUT_DIR / name
    if path.is_symlink() or (path.exists() and not path.is_file()):
        sys.exit(f"unsafe output path: {path}")
    return path

VARIANTS = [
    "fp8_gemv_blockwise_wpr_kernel",
    "fp8_gemv_blockwise_wpr_lut_kernel",
    "fp8_gemv_blockwise_wpr_native_kernel",
]

# -------- CUDA init ---------------------------------------------------------

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
print(f"device: cc {cc_major}.{cc_minor}, PTX: {PTX.name} ({ARCH})")
if ARCH != f"sm_{cc_major}{cc_minor}":
    sys.exit(f"PTX arch {ARCH} does not match device sm_{cc_major}{cc_minor}")
print(f"shape: M={M} N={N} K={K}, scales [{BN},{BK}], duration {DURATION}s/variant")

# -------- PTX load + alloc --------------------------------------------------

ptx_bytes = PTX.read_bytes() + b"\0"
mod = CHECK(drv.cuModuleLoadData(ptx_bytes), "cuModuleLoadData")
modules.append(mod)
metadata = {"arch": ARCH, "device_cc": f"{cc_major}.{cc_minor}",
            "cuda_driver_version": driver_version,
            "duration_s": DURATION, "n": N, "k": K,
            "ptx_sha256": hashlib.sha256(ptx_bytes[:-1]).hexdigest(), "seed": 42}
output_path("manifest.json").write_text(json.dumps(metadata, sort_keys=True) + "\n")

def alloc(bytes_):
    if bytes_ <= 0 or bytes_ > 2 * 1024**3:
        sys.exit(f"invalid allocation size: {bytes_}")
    ptr = CHECK(drv.cuMemAlloc(bytes_), "cuMemAlloc")
    allocations.append(ptr)
    return ptr

d_output = alloc(M * N * 4)
d_weight = alloc(N * K)
d_scale  = alloc(BN * BK * 4)
d_input  = alloc(M * K * 4)

def h2d(dst, arr):
    arr = np.ascontiguousarray(arr)
    CHECK(drv.cuMemcpyHtoD(dst, arr.ctypes.data, arr.nbytes), "HtoD")

rng = np.random.default_rng(42)
weight = rng.integers(0, 256, size=(N, K), dtype=np.uint8)
weight[weight == 0x7f] = 0x7e
weight[weight == 0xff] = 0xfe
scales = rng.normal(0.0, 0.1, size=(BN, BK)).astype(np.float32)
inp    = rng.normal(size=(M, K)).astype(np.float32)
h2d(d_weight, weight)
h2d(d_scale, scales)
h2d(d_input, inp)

# Kernel params (same for all variants — identical signature)
params = [
    np.array([int(d_output)], dtype=np.uint64),
    np.array([int(d_weight)], dtype=np.uint64),
    np.array([int(d_scale)],  dtype=np.uint64),
    np.array([int(d_input)],  dtype=np.uint64),
    np.array([M], dtype=np.int32),
    np.array([N], dtype=np.int32),
    np.array([K], dtype=np.int32),
    np.array([BK], dtype=np.int32),
]
param_ptrs = np.array([p.ctypes.data for p in params], dtype=np.uint64)

# Grid/block — matches the precision-check setup (8 rows per block.x)
GRID = ((N + 7) // 8, M, 1)
BLOCK = (256, 1, 1)

# -------- nvidia-smi sampler (lightweight, called once per iter) ------------

def smi_sample():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,power.draw",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=0.5)
        if out.returncode != 0:
            return (0, 0.0)
        line = out.stdout.strip().split(",")
        return (int(line[0]), float(line[1]))
    except Exception:
        return (0, 0.0)

# -------- Bench loop --------------------------------------------------------

def bench_variant(entry_name):
    fn = CHECK(drv.cuModuleGetFunction(mod, entry_name.encode()), "cuModuleGetFunction")
    # warmup
    for _ in range(100):
        CHECK(drv.cuLaunchKernel(fn, *GRID, *BLOCK, 0, 0,
                                 param_ptrs.ctypes.data, 0), "launch")
    CHECK(drv.cuCtxSynchronize(), "sync")

    ev_start = CHECK(drv.cuEventCreate(0), "eventCreate")
    ev_end   = CHECK(drv.cuEventCreate(0), "eventCreate")

    records = []
    t0 = time.perf_counter()
    last_smi_t = 0.0
    clock_mhz, power_w = smi_sample()
    while True:
        t_elapsed = time.perf_counter() - t0
        if t_elapsed >= DURATION:
            break
        CHECK(drv.cuEventRecord(ev_start, 0), "eventRecord")
        CHECK(drv.cuLaunchKernel(fn, *GRID, *BLOCK, 0, 0,
                                 param_ptrs.ctypes.data, 0), "launch")
        CHECK(drv.cuEventRecord(ev_end, 0), "eventRecord")
        CHECK(drv.cuEventSynchronize(ev_end), "eventSync")
        kern_ms = CHECK(drv.cuEventElapsedTime(ev_start, ev_end), "elapsedTime")
        if not math.isfinite(kern_ms) or kern_ms <= 0.0:
            sys.exit(f"invalid kernel timing: {kern_ms}")
        # nvidia-smi is expensive (~10 ms) — sample once per 100 ms of
        # wall-clock to avoid dominating the bench
        if t_elapsed - last_smi_t >= 0.1:
            clock_mhz, power_w = smi_sample()
            last_smi_t = t_elapsed
        records.append({
            "t_ms": t_elapsed * 1000.0,
            "kern_us": kern_ms * 1000.0,
            "clocks_sm_mhz": clock_mhz,
            "power_draw_w": power_w,
        })

    CHECK(drv.cuEventDestroy(ev_start), "eventDestroy")
    CHECK(drv.cuEventDestroy(ev_end),   "eventDestroy")
    return records

def summary(name, recs):
    if not recs:
        return
    lats = np.array([r["kern_us"] for r in recs])
    clocks = np.array([r["clocks_sm_mhz"] for r in recs])
    powers = np.array([r["power_draw_w"] for r in recs])
    t_ms = np.array([r["t_ms"] for r in recs])

    # First second vs last second
    first_mask = t_ms < 1000.0
    last_mask = t_ms >= (DURATION - 1.0) * 1000.0
    def stats(mask, label):
        if not mask.any():
            return f"  {label}: (empty)"
        l = lats[mask]; c = clocks[mask]; p = powers[mask]
        return (f"  {label:>12}  iters={int(mask.sum()):5d}  "
                f"lat {l.mean():6.1f}±{l.std():5.1f} µs  "
                f"p50 {np.median(l):6.1f}  p99 {np.percentile(l,99):6.1f}   "
                f"clock {c.mean():6.0f} MHz   power {p.mean():5.1f} W")

    # Effective bandwidth (weight-only read): N*K bytes / latency
    weight_gb = N * K / 1e9
    gbps_p50 = weight_gb / (np.median(lats) / 1e6)

    print(f"\n== {name} == ({len(recs)} iters, {t_ms[-1]/1000.0:.2f}s)")
    print(stats(first_mask, "first 1s"))
    print(stats(last_mask, "last 1s"))
    print(f"  overall p50 {np.median(lats):6.1f} µs  →  eff BW {gbps_p50:5.1f} GB/s  "
          f"(weight = {weight_gb*1000:.1f} MB)")

# -------- Run ---------------------------------------------------------------

all_records = {}
for v in VARIANTS:
    print(f"\n...benching {v}", flush=True)
    recs = bench_variant(v)
    all_records[v] = recs
    # Dump JSONL for offline analysis
    out_file = output_path(f"{v}.jsonl")
    with open(out_file, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {out_file}")

print("\n" + "=" * 78)
for v, recs in all_records.items():
    summary(v, recs)
print("\n" + "=" * 78)

# Throttle-onset heuristic: find the largest jump in 200ms-window mean
# latency after the first second.
print("\nclock-regime detection (200ms windows, mean latency):")
for v, recs in all_records.items():
    if len(recs) < 50:
        continue
    t_ms = np.array([r["t_ms"] for r in recs])
    lats = np.array([r["kern_us"] for r in recs])
    window_ms = 200.0
    n_windows = int(t_ms[-1] / window_ms)
    means = []
    for w in range(n_windows):
        mask = (t_ms >= w * window_ms) & (t_ms < (w + 1) * window_ms)
        if mask.any():
            means.append((w * window_ms / 1000.0, float(lats[mask].mean())))
    if len(means) >= 6:
        print(f"  {v}:")
        for t_s, m in means[::max(1, len(means)//8)]:
            print(f"    t={t_s:5.2f}s  mean lat {m:6.1f} µs")
