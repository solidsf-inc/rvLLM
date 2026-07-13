#!/usr/bin/env python3
"""Save bounded Hugging Face Gemma intermediates with a provenance manifest."""

import argparse
import json
import re
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

TRANSFORMERS_VERSION = "5.2.0"
TRANSFORMERS_REVISION = "7d9754a05193eb79b1d86aa744b622b8068008cd"
TORCH_VERSION = "2.10.0"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", help="Required commit SHA for a Hub model")
    parser.add_argument("--text", default="The quick brown")
    parser.add_argument("--layers", type=int, default=2, choices=range(1, 9))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is not empty; pass --overwrite to reuse it")
    args.output.mkdir(parents=True, exist_ok=True)

    common = {
        "revision": args.revision,
        "local_files_only": not args.allow_download,
        "trust_remote_code": False,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        **common,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    ids = tokenizer.encode(args.text, add_special_tokens=True)[:16]
    input_ids = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    captures = {}
    handles = []

    def hook(index):
        def capture(_module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captures[f"layer_{index}_input"] = inputs[0][0, 0].detach().float().cpu()
            captures[f"layer_{index}_output"] = tensor[0, 0].detach().float().cpu()
        return capture

    for index, layer in enumerate(text_model.layers[: args.layers]):
        handles.append(layer.register_forward_hook(hook(index)))
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=False)
    for handle in handles:
        handle.remove()
    captures["logits_first_token"] = output.logits[0, 0].detach().float().cpu()

    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in captures.values())
    if total_bytes > 256 * 1024 * 1024:
        raise SystemExit("capture exceeds the 256 MiB safety limit")
    for name, tensor in captures.items():
        torch.save(tensor, args.output / f"{name}.pt")
    manifest = {
        "model": Path(args.model).name if local else args.model,
        "revision": args.revision,
        "transformers": transformers.__version__,
        "transformers_revision": TRANSFORMERS_REVISION,
        "torch": torch.__version__,
        "token_ids": ids,
        "layers": args.layers,
        "tensor_count": len(captures),
        "bytes": total_bytes,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
