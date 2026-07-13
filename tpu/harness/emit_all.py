"""Batch StableHLO emitter.

Walks manifest.toml, resolves each kernel's port callable + trace spec, and
lowers it to StableHLO text. Writes one .mlir file per kernel under tpu/out.

`--status` prints the manifest summary without emitting. `--only NAME` limits
emission to a single kernel.

Do not execute on an accelerator. Emission is host-only; it traces with
jax.ShapeDtypeStruct and calls .compiler_ir(dialect='stablehlo').
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import sys
import traceback

try:
    import tomllib  # py311+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

import jax


ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "manifest.toml"
OUT = ROOT / "out"


def load_manifest():
    with open(MANIFEST, "rb") as f:
        return tomllib.load(f)["kernel"]


def resolve(port_ref: str):
    # "ports.rms_norm:rms_norm" -> (module, fn)
    mod_name, sym = port_ref.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, sym)
    # trace-spec symbol convention: <sym>_trace_spec
    spec = getattr(mod, f"{sym}_trace_spec", None)
    return fn, spec


def emit_one(entry: dict, out_dir: pathlib.Path) -> tuple[str, str]:
    name = entry["name"]
    status = entry["status"]
    if status == "todo":
        return (name, "SKIP-todo")

    fn, spec = resolve(entry["port"])
    if spec is None:
        return (name, "FAIL-no-trace-spec")

    dtypes = entry.get("dtypes", ["bf16"])
    primary = next((d for d in dtypes if d in {"bf16", "f16", "f32"}), "bf16")

    try:
        args, kwargs = spec(entry["shapes"], dtype=primary)
    except Exception as exc:
        return (name, f"FAIL-trace-spec: {exc}")

    try:
        lowered = jax.jit(fn, static_argnames=tuple(kwargs.keys())).lower(*args, **kwargs)
        mlir = lowered.compiler_ir(dialect="stablehlo")
    except NotImplementedError as exc:
        return (name, f"SKIP-notimpl: {exc}")
    except Exception as exc:
        tb = traceback.format_exc(limit=4)
        return (name, f"FAIL-lower: {exc}\n{tb}")

    out_path = out_dir / f"{name}.mlir"
    out_path.write_text(str(mlir))
    return (name, f"OK {out_path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="print manifest summary")
    ap.add_argument("--only", default=None, help="emit only this kernel name")
    args = ap.parse_args()

    entries = load_manifest()

    if args.status:
        counts: dict[str, int] = {}
        for e in entries:
            counts[e["status"]] = counts.get(e["status"], 0) + 1
        print(f"manifest: {len(entries)} kernels")
        for k, v in sorted(counts.items()):
            print(f"  {k:6s} {v}")
        for e in entries:
            print(f"  {e['status']:6s}  {e['name']:40s}  {e['port']}")
        return 0

    OUT.mkdir(exist_ok=True)
    failed = 0
    for e in entries:
        if args.only and e["name"] != args.only:
            continue
        name, msg = emit_one(e, OUT)
        print(f"[{msg.split()[0]:12s}] {name:40s} {msg}")
        if msg.startswith("FAIL"):
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
