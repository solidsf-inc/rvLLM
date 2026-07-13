"""External Transformers/torch-xla TPU diagnostic benchmark.

This does not exercise rvLLM: it measures naive batched `model.generate()`
with no paged KV cache or continuous batching.
"""

from __future__ import annotations
import argparse, json, os, sys, time

os.environ.setdefault("PJRT_DEVICE", "TPU")

import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer


def sync_xla():
    xm.mark_step()
    xm.wait_device_ops()


def fixed_length_inputs(tokenizer, device, n: int, seq_in: int):
    seed = tokenizer("The meaning of life is", add_special_tokens=True)["input_ids"]
    if not seed:
        raise ValueError("tokenizer produced an empty seed")
    row = (seed * ((seq_in + len(seed) - 1) // len(seed)))[:seq_in]
    input_ids = torch.tensor([row] * n, dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def bench_batch(model, tokenizer, device, n: int, seq_in: int, seq_out: int):
    inputs = fixed_length_inputs(tokenizer, device, n, seq_in)
    input_len = inputs["input_ids"].shape[1]

    # Compile both timed shapes and wait for device completion.
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        _ = model.generate(**inputs, max_new_tokens=seq_out, do_sample=False)
    sync_xla()

    # timed run
    sync_xla()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=seq_out, do_sample=False)
    sync_xla()
    wall = time.perf_counter() - t0

    gen_tokens = out.shape[1] - input_len
    total_out = gen_tokens * n
    toks = total_out / wall

    # TTFT approximation: time to first token = time for 1-token generate
    sync_xla()
    t1 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    sync_xla()
    ttft = (time.perf_counter() - t1) * 1000

    return toks, ttft, wall, total_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--seq-in", type=int, default=16)
    ap.add_argument("--seq-out", type=int, default=512)
    ap.add_argument("--batches", default="1,8,16,64,128")
    ap.add_argument("--out", default="/tmp/tpu_e2e_bench.json")
    args = ap.parse_args()
    if args.seq_in <= 0 or args.seq_out <= 0:
        ap.error("--seq-in and --seq-out must be positive")
    try:
        batches = [int(value) for value in args.batches.split(",")]
    except ValueError:
        ap.error("--batches must be comma-separated integers")
    if not batches or any(value <= 0 or value > 1024 for value in batches):
        ap.error("every batch size must be in 1..1024")

    device = xm.xla_device()
    print(f"device: {device}", flush=True)

    print(f"loading {args.model} bf16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and args.seq_in + args.seq_out > max_positions:
        ap.error("requested input and output exceed model context length")
    print("model loaded", flush=True)

    results = {
        "kind": "external-transformers-torch-xla-diagnostic",
        "model": args.model,
        "revision": args.revision,
        "device": str(device),
        "runs": [],
    }
    failed = False
    for n in batches:
        try:
            toks, ttft, wall, total_out = bench_batch(
                model, tokenizer, device, n, args.seq_in, args.seq_out
            )
            print(f"N={n:4d}  toks={toks:10.2f}/s  ttft={ttft:8.2f}ms  "
                  f"wall={wall:6.2f}s  out={total_out}", flush=True)
            results["runs"].append({
                "n": n, "toks": round(toks, 2), "ttft_ms": round(ttft, 2),
                "wall_s": round(wall, 2), "out_tokens": total_out,
            })
        except Exception as exc:
            failed = True
            print(f"N={n}: FAILED {exc}", flush=True)
            results["runs"].append({"n": n, "error": str(exc)})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}", flush=True)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
