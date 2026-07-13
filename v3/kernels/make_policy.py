#!/usr/bin/env python3
"""Generate an explicit, validated rvLLM decode dispatch policy."""

import argparse
import hashlib
import json
import os
import pathlib
import re
import tempfile


REVISION = re.compile(r"^[0-9a-f]{40}$")
ARCH = re.compile(r"^sm_(?:80|89|90|100|121)$")
MAX_DIMENSION = (1 << 31) - 1
MAX_WORKSPACE_BYTES = 8 << 30
MAX_ENTRIES = 65_536

CANONICAL_VARIANTS = {
    0: {
        "id": 0,
        "tile": {"m": 128, "n": 128, "k": 128},
        "cluster": {"m": 1, "n": 1, "k": 1},
        "mainloop": "Coop",
        "epilogue": "Coop",
    },
    1: {
        "id": 1,
        "tile": {"m": 128, "n": 256, "k": 128},
        "cluster": {"m": 1, "n": 1, "k": 1},
        "mainloop": "Coop",
        "epilogue": "Coop",
    },
    2: {
        "id": 2,
        "tile": {"m": 64, "n": 128, "k": 128},
        "cluster": {"m": 1, "n": 1, "k": 1},
        "mainloop": "WS",
        "epilogue": "WS",
    },
    3: {
        "id": 3,
        "tile": {"m": 128, "n": 128, "k": 128},
        "cluster": {"m": 1, "n": 1, "k": 1},
        "mainloop": "Fp8Coop",
        "epilogue": "Fp8Coop",
    },
    4: {
        "id": 4,
        "tile": {"m": 64, "n": 128, "k": 128},
        "cluster": {"m": 1, "n": 1, "k": 1},
        "mainloop": "Fp8WS",
        "epilogue": "Fp8WS",
    },
    100: {
        "id": 100,
        "tile": {"m": 128, "n": 128, "k": 128},
        "cluster": {"m": 1, "n": 1, "k": 1},
        "mainloop": "Coop",
        "epilogue": "Coop",
    },
}


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_buckets(value: str):
    try:
        buckets = [positive(item) for item in value.split(",")]
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError("buckets must be comma-separated positive integers") from exc
    if buckets != sorted(set(buckets)):
        raise argparse.ArgumentTypeError("buckets must be unique and sorted")
    return buckets


def canonical_json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path")
    parser.add_argument("revision")
    parser.add_argument("--arch", required=True)
    parser.add_argument("--hidden", type=positive, required=True)
    parser.add_argument("--q-heads", type=positive, required=True)
    parser.add_argument("--kv-heads", type=positive, required=True)
    parser.add_argument("--head-dim", type=positive, required=True)
    parser.add_argument("--intermediate", type=positive, required=True)
    parser.add_argument("--vocab", type=positive, required=True)
    parser.add_argument("--buckets", type=parse_buckets, required=True)
    parser.add_argument("--nonres-variant", type=int, required=True)
    parser.add_argument("--residual-variant", type=int, required=True)
    parser.add_argument("--workspace-bytes", type=positive, required=True)
    parser.add_argument("--generator-revision", required=True)
    args = parser.parse_args()

    if not REVISION.fullmatch(args.revision) or not REVISION.fullmatch(args.generator_revision):
        parser.error("revisions must be full lowercase 40-hex commits")
    if not ARCH.fullmatch(args.arch):
        parser.error("unsupported architecture")
    if args.q_heads % args.kv_heads:
        parser.error("--q-heads must be divisible by --kv-heads")
    if args.nonres_variant not in {0, 1, 2, 3, 4}:
        parser.error("--nonres-variant must be one of 0,1,2,3,4")
    if args.residual_variant != 100:
        parser.error("--residual-variant must be 100")
    if args.workspace_bytes > MAX_WORKSPACE_BYTES:
        parser.error("--workspace-bytes exceeds the runtime 8 GiB limit")
    dimensions = (
        args.hidden,
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        args.intermediate,
        args.vocab,
        *args.buckets,
    )
    if any(value > MAX_DIMENSION for value in dimensions):
        parser.error("dimensions and buckets must fit in the runtime i32 shape limit")

    q_dim = args.q_heads * args.head_dim
    kv_dim = args.kv_heads * args.head_dim
    qkv_rows = q_dim + 2 * kv_dim
    if any(value > MAX_DIMENSION for value in (q_dim, kv_dim, qkv_rows, 2 * args.intermediate)):
        parser.error("derived projection dimensions must fit in the runtime i32 shape limit")
    modes = {
        "qkv": {"n": qkv_rows, "k": args.hidden, "mode": "plain", "variant": args.nonres_variant},
        "gate_up": {"n": 2 * args.intermediate, "k": args.hidden, "mode": "plain", "variant": args.nonres_variant},
        "lm_head": {"n": args.vocab, "k": args.hidden, "mode": "plain", "variant": args.nonres_variant},
        "o_proj": {"n": args.hidden, "k": q_dim, "mode": "residual", "variant": args.residual_variant},
        "down_proj": {"n": args.hidden, "k": args.intermediate, "mode": "residual", "variant": args.residual_variant},
    }
    entries = {}
    for bucket in args.buckets:
        for config in modes.values():
            key = f"{bucket}_{config['n']}_{config['k']}_Fp8E4M3_{config['mode']}"
            entry = {
                "variant": config["variant"],
                "workspace_bytes": args.workspace_bytes,
            }
            previous = entries.setdefault(key, entry)
            if previous != entry:
                parser.error(f"conflicting dispatch entries for {key}")
    if len(entries) > MAX_ENTRIES:
        parser.error(f"policy exceeds the runtime {MAX_ENTRIES}-entry limit")

    inputs = {
        "hidden": args.hidden,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "intermediate": args.intermediate,
        "vocab": args.vocab,
        "buckets": args.buckets,
        "workspace_bytes": args.workspace_bytes,
    }
    input_hash = hashlib.sha256(canonical_json(inputs)).hexdigest()
    policy = {
        "revision": args.revision,
        "arch": args.arch,
        "variants": [
            CANONICAL_VARIANTS[args.nonres_variant],
            CANONICAL_VARIANTS[args.residual_variant],
        ],
        "entries": entries,
    }

    output = pathlib.Path(args.out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        parser.error("output path cannot be a symlink")
    data = canonical_json(policy)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(data)
    os.replace(temporary, output)
    print(
        f"wrote {output} with {len(entries)} entries "
        f"generator={args.generator_revision} input={input_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
