"""External Transformers/Flax TPU diagnostic benchmark.

This does not exercise the rvLLM runtime. It loads one pinned model revision
through the Transformers Flax backend and synchronizes every timed device run.
"""

from __future__ import annotations
import argparse, json, os, sys, time

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import jax.numpy as jnp
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--seq-out", type=int, default=64)
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--out", default="/tmp/tpu_e2e_bench.json")
    args = ap.parse_args()
    if args.seq_out <= 0:
        ap.error("--seq-out must be positive")
    try:
        batches = [int(value) for value in args.batches.split(",")]
    except ValueError:
        ap.error("--batches must be comma-separated integers")
    if not batches or any(value <= 0 or value > 1024 for value in batches):
        ap.error("every batch size must be in 1..1024")

    print(f"jax {jax.__version__} backend={jax.default_backend()} "
          f"devices={jax.devices()}", flush=True)

    from transformers import AutoTokenizer

    try:
        from transformers import FlaxAutoModelForCausalLM
    except ImportError:
        print("FlaxAutoModelForCausalLM not available", flush=True)
        sys.exit(1)

    print(f"loading {args.model} (from_pt=True)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FlaxAutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        from_pt=True,
        dtype=jnp.bfloat16,
    )

    print("model loaded", flush=True)

    prompt = "The meaning of life is"
    results = {
        "kind": "external-transformers-flax-diagnostic",
        "model": args.model,
        "revision": args.revision,
        "device": str(jax.devices()[0]),
        "runs": [],
    }

    for n in batches:
        print(f"bench N={n}...", flush=True)
        inputs = tokenizer([prompt] * n, return_tensors="np", padding=True)
        input_ids = jnp.array(inputs["input_ids"])
        attn_mask = jnp.array(inputs["attention_mask"])

        # Compile both timed shapes and wait for device completion.
        try:
            first = model.generate(
                input_ids, attention_mask=attn_mask, max_new_tokens=1, do_sample=False
            )
            jax.block_until_ready(first.sequences)
            full = model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=args.seq_out,
                do_sample=False,
            )
            jax.block_until_ready(full.sequences)
        except Exception as exc:
            print(f"N={n}: warmup failed: {exc}", flush=True)
            results["runs"].append({"n": n, "error": str(exc)})
            continue

        # TTFT: time to generate 1 token
        t0 = time.perf_counter()
        first = model.generate(
            input_ids, attention_mask=attn_mask, max_new_tokens=1, do_sample=False
        )
        jax.block_until_ready(first.sequences)
        ttft = (time.perf_counter() - t0) * 1000

        # throughput: generate seq_out tokens
        t0 = time.perf_counter()
        out = model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=args.seq_out,
            do_sample=False,
        )
        jax.block_until_ready(out.sequences)
        wall = time.perf_counter() - t0
        gen_tokens = out.sequences.shape[1] - input_ids.shape[1]
        total_out = gen_tokens * n
        toks = total_out / wall

        print(f"N={n:4d}  toks={toks:10.2f}/s  ttft={ttft:8.2f}ms  "
              f"wall={wall:6.2f}s  out={total_out}", flush=True)
        results["runs"].append({
            "n": n, "toks": round(toks, 2), "ttft_ms": round(ttft, 2),
            "wall_s": round(wall, 2), "out_tokens": total_out,
        })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
