"""Execute every impl port on the active JAX device (TPU if available).

Unlike emit_all.py (which only lowers to StableHLO text), this actually
compiles and runs each port on real accelerator silicon with random inputs
drawn from the manifest's shape template. Reports per-kernel:
  - device executed on
  - wall time (ms)
  - output shape(s) + dtype(s)

No numerical parity vs CUDA here; that's verify_all.py's job. This is a
"does it run on TPU at all" proof.
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import sys
import time

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

import jax
import jax.numpy as jnp
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "manifest.toml"


def load_manifest():
    with open(MANIFEST, "rb") as f:
        return tomllib.load(f)["kernel"]


def resolve(port_ref: str):
    mod_name, sym = port_ref.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, sym), getattr(mod, f"{sym}_trace_spec", None)


def materialize(spec_arg, rng):
    # spec_arg is jax.ShapeDtypeStruct — build a random array of that shape/dtype.
    dt = spec_arg.dtype
    shape = spec_arg.shape
    if jnp.issubdtype(dt, jnp.integer):
        # integer args are typically indices; keep values in a safe range
        hi = 2 if shape == () else 64
        return jnp.asarray(rng.integers(0, hi, size=shape, dtype=np.int32), dtype=dt)
    return jnp.asarray(rng.standard_normal(size=shape).astype(np.float32), dtype=dt)


def describe(out):
    def one(x):
        return f"{tuple(x.shape)}:{x.dtype}"
    if isinstance(out, (tuple, list)):
        return ", ".join(one(x) for x in out)
    return one(out)


def run_one(entry, rng):
    name = entry["name"]
    status = entry["status"]
    if status == "todo":
        return (name, "SKIP-todo", "")

    fn, spec = resolve(entry["port"])
    if spec is None:
        return (name, "FAIL-no-spec", "")

    dtypes = entry.get("dtypes", ["bf16"])
    primary = next((d for d in dtypes if d in {"bf16", "f16", "f32"}), "bf16")

    try:
        spec_args, kwargs = spec(entry["shapes"], dtype=primary)
    except Exception as exc:
        return (name, f"FAIL-spec: {exc}", "")

    try:
        args = [materialize(a, rng) for a in spec_args]
    except Exception as exc:
        return (name, f"FAIL-mat: {exc}", "")

    try:
        jitted = jax.jit(fn, static_argnames=tuple(kwargs.keys()))
        # warmup: 3 runs to stabilise JIT + HBM placement
        for _ in range(3):
            out = jitted(*args, **kwargs)
            jax.block_until_ready(out)
        # measured: 20 iters, report median (µs)
        iters = 20
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = jitted(*args, **kwargs)
            jax.block_until_ready(out)
            samples.append((time.perf_counter() - t0) * 1e6)
        samples.sort()
        us = samples[iters // 2]
    except NotImplementedError as exc:
        return (name, f"SKIP-notimpl: {exc}", "")
    except Exception as exc:
        return (name, f"FAIL-run: {exc}", "")

    return (name, f"OK {us:8.1f}us", describe(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"jax={jax.__version__}  devices={jax.devices()}  backend={jax.default_backend()}")
    rng = np.random.default_rng(args.seed)

    entries = load_manifest()
    ok = fail = skip = 0
    results = []
    for e in entries:
        if args.only and e["name"] != args.only:
            continue
        name, status, desc = run_one(e, rng)
        tag = status.split()[0]
        print(f"[{tag:14s}] {name:40s} {status:40s} {desc}")
        us = None
        if status.startswith("OK"):
            ok += 1
            us = float(status.split()[1].rstrip("us"))
        elif status.startswith("SKIP"):
            skip += 1
        else:
            fail += 1
        results.append({
            "name": name,
            "family": e.get("family", ""),
            "status": tag.strip("[]"),
            "us": us,
            "output": desc,
        })

    print(f"\nsummary: ok={ok}  skip={skip}  fail={fail}  total={ok+skip+fail}")

    import json
    out = {
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "results": results,
    }
    with open("/tmp/tpu_bench.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote /tmp/tpu_bench.json")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
