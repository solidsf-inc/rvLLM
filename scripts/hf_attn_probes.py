#!/usr/bin/env python3
"""Capture bounded Gemma attention tensors for rvLLM parity checks."""

import argparse
import re
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

TRANSFORMERS_VERSION = "5.2.0"
TRANSFORMERS_REVISION = "7d9754a05193eb79b1d86aa744b622b8068008cd"
TORCH_VERSION = "2.10.0"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local model directory or pinned Hub id")
    parser.add_argument("--revision", help="Required commit SHA when --model is a Hub id")
    parser.add_argument("--token-id", type=int, default=2)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    local = Path(args.model).is_dir()
    if not local and not re.fullmatch(r"[0-9a-f]{40}", args.revision or ""):
        raise SystemExit("a Hub model requires --revision with an immutable commit SHA")
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise SystemExit(
            f"requires transformers {TRANSFORMERS_VERSION} "
            f"({TRANSFORMERS_REVISION}), found {transformers.__version__}"
        )
    if torch.__version__.split("+")[0] != TORCH_VERSION:
        raise SystemExit(f"requires torch {TORCH_VERSION}, found {torch.__version__}")
    print(
        f"reference=transformers@{TRANSFORMERS_REVISION} "
        f"transformers={transformers.__version__} torch={torch.__version__}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        local_files_only=not args.allow_download,
        trust_remote_code=False,
        attn_implementation="eager",
    ).eval()
    text = model.model.language_model if hasattr(model.model, "language_model") else model.model
    layer = text.layers[0]
    captures = {}

    def output_hook(name):
        def capture(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captures[name] = tensor.detach().float().cpu()
        return capture

    def input_hook(name):
        def capture(_module, inputs):
            captures[name] = inputs[0].detach().float().cpu()
        return capture

    handles = [
        layer.input_layernorm.register_forward_hook(output_hook("input_norm")),
        layer.self_attn.q_proj.register_forward_hook(output_hook("q_proj")),
        layer.self_attn.k_proj.register_forward_hook(output_hook("k_proj")),
        layer.self_attn.v_proj.register_forward_hook(output_hook("v_proj")),
        layer.self_attn.o_proj.register_forward_pre_hook(input_hook("pre_o_proj")),
        layer.self_attn.o_proj.register_forward_hook(output_hook("o_proj")),
    ]
    for name in ("q_norm", "k_norm"):
        module = getattr(layer.self_attn, name, None)
        if module is not None:
            handles.append(module.register_forward_hook(output_hook(name)))

    device = next(model.parameters()).device
    ids = torch.tensor([[args.token_id]], dtype=torch.long, device=device)
    with torch.inference_mode():
        model(input_ids=ids, use_cache=False)
    for handle in handles:
        handle.remove()

    for name, tensor in captures.items():
        flat = tensor.flatten()
        print(
            f"{name}: shape={list(tensor.shape)} "
            f"amax={flat.abs().max().item():.8g} first8={flat[:8].tolist()}"
        )


if __name__ == "__main__":
    main()
