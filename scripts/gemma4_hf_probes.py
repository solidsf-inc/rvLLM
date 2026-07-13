#!/usr/bin/env python3
"""Print bounded, non-mutating Hugging Face Gemma parity probes."""

import argparse
import re
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

TRANSFORMERS_VERSION = "5.2.0"
TRANSFORMERS_REVISION = "7d9754a05193eb79b1d86aa744b622b8068008cd"
TORCH_VERSION = "2.10.0"


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", help="Required commit SHA for a Hub model")
    parser.add_argument("--token-id", type=int, default=2)
    parser.add_argument("--layers", default="0,1")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def stats(label, value):
    flat = value.detach().float().flatten().cpu()
    print(
        f"{label}: shape={list(value.shape)} min={flat.min().item():.8g} "
        f"max={flat.max().item():.8g} mean={flat.mean().item():.8g} "
        f"first4={flat[:4].tolist()}"
    )


def main():
    options = args()
    local = Path(options.model).is_dir()
    if not local and not re.fullmatch(r"[0-9a-f]{40}", options.revision or ""):
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
    indices = sorted({int(value) for value in options.layers.split(",") if value.strip()})

    model = AutoModelForCausalLM.from_pretrained(
        options.model,
        revision=options.revision,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=not options.allow_download,
        trust_remote_code=False,
        attn_implementation="eager",
    ).eval()
    text = model.model.language_model if hasattr(model.model, "language_model") else model.model
    if any(index < 0 or index >= len(text.layers) for index in indices):
        raise SystemExit(f"--layers must be within 0..{len(text.layers) - 1}")

    captures = {}
    handles = []
    for index in indices:
        layer = text.layers[index]

        def hook(_module, inputs, output, index=index):
            tensor = output[0] if isinstance(output, tuple) else output
            captures[index] = (
                inputs[0].detach().float().cpu(),
                tensor.detach().float().cpu(),
            )

        handles.append(layer.register_forward_hook(hook))
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            module = getattr(layer, name, None)
            if module is not None:
                stats(f"layer.{index}.{name}.weight", module.weight)

    device = next(model.parameters()).device
    input_ids = torch.tensor([[options.token_id]], dtype=torch.long, device=device)
    stats("embedding", text.embed_tokens(input_ids))
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=False)
    for handle in handles:
        handle.remove()
    for index in indices:
        before, after = captures[index]
        stats(f"layer.{index}.input", before)
        stats(f"layer.{index}.output", after)
    stats("logits", output.logits[:, -1, :])


if __name__ == "__main__":
    main()
