"""Emit StableHLO bytecode (.mlirbc) for each impl kernel.

Same as emit_all.py but writes binary bytecode instead of text MLIR.
PJRT_Client_Compile parses bytecode more reliably than text (avoids
StableHLO version mismatch in text syntax).
"""

from __future__ import annotations
import importlib, io, pathlib, sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import jax

ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "manifest.toml"
OUT = ROOT / "out"


def load_manifest():
    with open(MANIFEST, "rb") as f:
        return tomllib.load(f)["kernel"]


def resolve(port_ref):
    mod_name, sym = port_ref.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, sym)
    spec = getattr(mod, f"{sym}_trace_spec", None)
    return fn, spec


def main() -> int:
    OUT.mkdir(exist_ok=True)
    entries = load_manifest()
    ok = fail = skip = 0
    for e in entries:
        name = e["name"]
        if e["status"] == "todo":
            skip += 1
            continue
        fn, spec = resolve(e["port"])
        if spec is None:
            fail += 1
            continue
        dtypes = e.get("dtypes", ["bf16"])
        primary = next((d for d in dtypes if d in {"bf16", "f16", "f32"}), "bf16")
        try:
            args, kwargs = spec(e["shapes"], dtype=primary)
            lowered = jax.jit(fn, static_argnames=tuple(kwargs.keys())).lower(*args, **kwargs)
            mlir = lowered.compiler_ir(dialect="stablehlo")
            buf = io.BytesIO()
            mlir.operation.write_bytecode(buf)
            bc = buf.getvalue()
            path = OUT / f"{name}.mlirbc"
            path.write_bytes(bc)
            print(f"OK  {name:40s} {len(bc):6d} bytes")
            ok += 1
        except NotImplementedError:
            skip += 1
        except Exception as ex:
            print(f"FAIL {name:40s} {ex}")
            fail += 1
    print(f"bytecode: ok={ok} skip={skip} fail={fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
