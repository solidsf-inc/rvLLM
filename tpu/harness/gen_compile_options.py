#!/usr/bin/env python3
"""Generate compile_options protobuf for PJRT multi-device SPMD compilation.

Produces a serialized xla::CompileOptionsProto that the Rust PJRT client
passes directly to PJRT_Client_Compile. Intended for TPU v6e-4 (4 chips)
with tensor parallelism across all 4 devices.

Usage:
    python3 gen_compile_options.py --num-partitions 4 --output compile_options_tp4.pb
    python3 gen_compile_options.py --num-partitions 1 --output compile_options_tp1.pb
"""
import argparse
import importlib.metadata
import sys


SUPPORTED_JAX_VERSION = "0.4.38"


def build_compile_options_proto(num_replicas: int, num_partitions: int) -> bytes:
    """Build a serialized xla::CompileOptionsProto using JAX's xla_client.

    The proto contains:
      - ExecutableBuildOptionsProto with num_replicas, num_partitions,
        use_spmd_partitioning=True, and a DeviceAssignment.
      - DeviceAssignment maps (replica, partition) -> device ordinal.

    For TP=4 on v6e-4: 1 replica x 4 partitions, device assignment
    [[0, 1, 2, 3]].
    """
    actual_version = importlib.metadata.version("jax")
    if actual_version != SUPPORTED_JAX_VERSION:
        raise RuntimeError(
            f"JAX {actual_version} is unsupported; install the pinned {SUPPORTED_JAX_VERSION} build"
        )
    from jax._src.lib import xla_client

    opts = xla_client.CompileOptions()
    opts.num_replicas = num_replicas
    opts.num_partitions = num_partitions

    total_devices = num_replicas * num_partitions

    # Build device assignment: shape [num_replicas, num_partitions]
    # Each entry is the device ordinal. For a single-replica TP setup,
    # replica 0 gets devices 0..num_partitions-1.
    import numpy as np
    device_assignment = np.arange(total_devices, dtype=np.int64).reshape(
        num_replicas, num_partitions
    )
    opts.device_assignment = xla_client.DeviceAssignment.create(device_assignment)

    # Enable SPMD partitioning (required for sharded HLO)
    opts.executable_build_options.use_spmd_partitioning = True

    # Serialize to protobuf bytes -- this is the format PJRT_Client_Compile expects
    # xla_client.CompileOptions has a SerializeAsString method that returns
    # the serialized xla::CompileOptionsProto
    serialize = getattr(opts, "SerializeAsString", None)
    if serialize is None:
        raise RuntimeError("pinned JAX build does not expose CompileOptions serialization")
    return serialize()


def main():
    parser = argparse.ArgumentParser(
        description="Generate compile_options.pb for PJRT SPMD compilation"
    )
    parser.add_argument(
        "--num-replicas", type=int, default=1,
        help="Number of replicas (default: 1)"
    )
    parser.add_argument(
        "--num-partitions", type=int, default=4,
        help="Number of partitions / TP degree (default: 4)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="compile_options_tp4.pb",
        help="Output file path (default: compile_options_tp4.pb)"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print human-readable summary of the generated proto"
    )
    args = parser.parse_args()

    if args.num_partitions < 1:
        print("ERROR: --num-partitions must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.num_replicas < 1:
        print("ERROR: --num-replicas must be >= 1", file=sys.stderr)
        sys.exit(1)

    total = args.num_replicas * args.num_partitions
    print(f"generating CompileOptionsProto: {args.num_replicas} replica(s) x "
          f"{args.num_partitions} partition(s) = {total} device(s)", file=sys.stderr)

    data = build_compile_options_proto(args.num_replicas, args.num_partitions)

    with open(args.output, "wb") as f:
        f.write(data)
    print(f"wrote {args.output} ({len(data)} bytes)", file=sys.stderr)

    if args.summary:
        print(file=sys.stderr)
        print("--- compile options summary ---", file=sys.stderr)
        print(f"num_replicas:          {args.num_replicas}", file=sys.stderr)
        print(f"num_partitions:        {args.num_partitions}", file=sys.stderr)
        print(f"use_spmd_partitioning: true", file=sys.stderr)
        print(f"device_assignment:     [{args.num_replicas}x{args.num_partitions}] "
              f"ordinals 0..{total-1}", file=sys.stderr)
        print(f"total devices:         {total}", file=sys.stderr)
        print(f"proto hex:             {data.hex()}", file=sys.stderr)
        print(f"proto bytes:           {list(data)}", file=sys.stderr)


if __name__ == "__main__":
    main()
