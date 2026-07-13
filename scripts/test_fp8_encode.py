#!/usr/bin/env python3
"""Check canonical E4M3FN bytes with PyTorch as an independent oracle."""

import math

import torch

TORCH_VERSION = "2.10.0"

VECTORS = [
    (0.0, 0x00),
    (-0.0, 0x80),
    (1.0, 0x38),
    (-1.0, 0xB8),
    (0.001953125, 0x01),
    (0.00390625, 0x02),
    (10.0, 0x52),
    (448.0, 0x7E),
    (449.0, 0x7E),
    (math.inf, 0x7F),
    (-math.inf, 0xFF),
    (math.nan, 0x7F),
    (-10.071, 0xD2),
    (-80.569, 0xEA),
    (9.352, 0x51),
    (-74.814, 0xE9),
    (-63.304, 0xE8),
    (-25.897, 0xDD),
    (-4.316, 0xC9),
    (-20.142, 0xDA),
]


def main():
    if torch.__version__.split("+")[0] != TORCH_VERSION:
        raise SystemExit(f"requires torch {TORCH_VERSION}, found {torch.__version__}")
    values = torch.tensor([value for value, _ in VECTORS], dtype=torch.float32)
    actual = values.to(torch.float8_e4m3fn).view(torch.uint8).tolist()
    expected = [byte for _, byte in VECTORS]
    for (value, byte), got in zip(VECTORS, actual):
        print(f"{value!r}: expected=0x{byte:02x} actual=0x{got:02x}")
    if actual != expected:
        raise SystemExit("E4M3FN oracle mismatch")


if __name__ == "__main__":
    main()
