#!/usr/bin/env python3
"""Measure per-row FP8 rescaling error without loading a full checkpoint."""

import argparse
import json
from pathlib import Path

import torch
import safetensors
from safetensors import safe_open

TORCH_VERSION = "2.10.0"
SAFETENSORS_VERSION = "0.7.0"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sample-rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load(index, root, name):
    base = name.removesuffix("_scale")
    shard = index["weight_map"].get(name) or index["weight_map"].get(base)
    if shard is None:
        raise KeyError(name)
    with safe_open(root / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def main():
    args = parse_args()
    if torch.__version__.split("+")[0] != TORCH_VERSION:
        raise SystemExit(f"requires torch {TORCH_VERSION}, found {torch.__version__}")
    if safetensors.__version__ != SAFETENSORS_VERSION:
        raise SystemExit(
            f"requires safetensors {SAFETENSORS_VERSION}, found {safetensors.__version__}"
        )
    if args.sample_rows < 1:
        raise SystemExit("--sample-rows must be positive")
    index_path = args.model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    prefix = f"model.language_model.layers.{args.layer}.self_attn"
    pairs = []
    for projection in ("q_proj", "k_proj", "v_proj"):
        weight = load(index, args.model_dir, f"{prefix}.{projection}.weight")
        scale = load(index, args.model_dir, f"{prefix}.{projection}.weight_scale").float().flatten()
        if weight.ndim != 2 or scale.numel() not in (1, weight.shape[0]):
            raise SystemExit(f"unsupported scale shape for {projection}: {list(scale.shape)}")
        pairs.append((projection, weight, scale.expand(weight.shape[0])))

    global_scale = max(scales.max().item() for _, _, scales in pairs)
    if not torch.isfinite(torch.tensor(global_scale)) or global_scale <= 0:
        raise SystemExit("weight scales must have a finite positive maximum")
    generator = torch.Generator().manual_seed(args.seed)
    summary = {
        "torch": torch.__version__,
        "safetensors": safetensors.__version__,
        "layer": args.layer,
        "seed": args.seed,
        "global_scale": global_scale,
        "projections": {},
    }
    for name, weight, scales in pairs:
        count = min(args.sample_rows, weight.shape[0])
        rows = torch.randperm(weight.shape[0], generator=generator)[:count]
        errors = []
        flushed = 0
        elements = 0
        for row in rows.tolist():
            source = weight[row].float()
            correct = source * scales[row]
            encoded = (source * (scales[row] / global_scale)).to(torch.float8_e4m3fn)
            reconstructed = encoded.float() * global_scale
            mask = correct.abs() > 1e-10
            errors.append(((reconstructed[mask] - correct[mask]).abs() / correct[mask].abs()).cpu())
            flushed += int(((encoded == 0) & (source != 0)).sum())
            elements += source.numel()
        nonempty = [error for error in errors if error.numel()]
        if not nonempty:
            raise SystemExit(f"{name} sample contained no nonzero reference values")
        relative = torch.cat(nonempty).sort().values
        summary["projections"][name] = {
            "sample_rows": count,
            "elements": elements,
            "flushed_to_zero": flushed,
            "mean_relative_error": relative.mean().item(),
            "p95_relative_error": relative[int(0.95 * (relative.numel() - 1))].item(),
            "p99_relative_error": relative[int(0.99 * (relative.numel() - 1))].item(),
            "max_relative_error": relative[-1].item(),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
