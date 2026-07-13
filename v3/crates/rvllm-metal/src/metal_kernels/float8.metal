// Adapted from mistral.rs revision 31c13eb4587d3e4a5204870c98b70c05a1e5c943:
// mistralrs-quant/src/metal_kernels/float8.metal
// License: MIT (mistralrs) + original Apache-2.0 (Apple/MLX where applicable)
// Modifications: correct subnormal decoding and round-to-nearest-even encoding.

#include <metal_stdlib>

using namespace metal;

// ————————————————————————————————————————————————————————————————
// F8E4M3 (Sign=1, Exponent=4, Mantissa=3; bias=2^(4−1)−1 = 7)
// ————————————————————————————————————————————————————————————————

inline float fp8_e4m3_to_float(uchar v) {
  const uint sign = (v >> 7) & 0x1;
  const uint exp_bits = (v >> 3) & 0xF;
  const uint man_bits = v & 0x7;

  // handle zero / subnormals
  if (exp_bits == 0) {
    if (man_bits == 0) {
      return sign ? -0.0f : 0.0f;
    }
    // man_bits * 2^(1 - bias - mantissa_bits)
    float val = ldexp(float(man_bits), 1 - 7 - 3);
    return sign ? -val : val;
  }
  // handle NaN: E4M3 has no infinity, only NaN when exp=15 and mantissa=7
  if (exp_bits == 0xF && man_bits == 0x7) {
    return NAN;
  }
  // normalised (including exp_bits == 0xF with mantissa 0-6)
  float mant = 1.0f + float(man_bits) / float(1 << 3);
  int expn = int(exp_bits) - 7;
  float val = ldexp(mant, expn);
  return sign ? -val : val;
}

inline uchar float_to_fp8_e4m3(float f) {
  const uint sign = as_type<uint>(f) >> 31;
  const uchar sign_bits = uchar(sign << 7);
  if (isnan(f)) {
    return uchar(sign_bits | 0x7F);
  }
  const float magnitude = fabs(f);
  if (magnitude == 0.0f) {
    return sign_bits;
  }
  if (isinf(magnitude) || magnitude > 448.0f) {
    return uchar(sign_bits | 0x7E);
  }

  // Subnormal spacing is 2^-9. rint is round-to-nearest-even.
  if (magnitude < 0x1p-6f) {
    uint mantissa = uint(rint(ldexp(magnitude, 9)));
    if (mantissa == 0) {
      return sign_bits;
    }
    if (mantissa >= 8) {
      return uchar(sign_bits | 0x08); // rounded to the minimum normal
    }
    return uchar(sign_bits | mantissa);
  }

  int exponent;
  (void)frexp(magnitude, exponent);
  int unbiased = exponent - 1;
  uint exponent_bits = uint(unbiased + 7);
  float significand = ldexp(magnitude, -unbiased);
  uint mantissa = uint(rint((significand - 1.0f) * 8.0f));
  if (mantissa == 8) {
    mantissa = 0;
    exponent_bits += 1;
  }
  if (exponent_bits > 15 || (exponent_bits == 15 && mantissa > 6)) {
    return uchar(sign_bits | 0x7E);
  }
  return uchar(sign_bits | (exponent_bits << 3) | mantissa);
}

// ————————————————————————————————————————————————————————————————
// F8E5M2 (Sign=1, Exponent=5, Mantissa=2; bias=2^(5−1)−1 = 15)
// ————————————————————————————————————————————————————————————————

inline float fp8_e5m2_to_float(uchar v) {
  const uint sign = (v >> 7) & 0x1;
  const uint exp_bits = (v >> 2) & 0x1F;
  const uint man_bits = v & 0x3;

  if (exp_bits == 0) {
    if (man_bits == 0) {
      return sign ? -0.0f : 0.0f;
    }
    float val = ldexp(float(man_bits), 1 - 15 - 2);
    return sign ? -val : val;
  }
  if (exp_bits == 0x1F) {
    if (man_bits != 0) {
      return NAN;
    }
    return sign ? -INFINITY : INFINITY;
  }
  float mant = 1.0f + float(man_bits) / float(1 << 2);
  int expn = int(exp_bits) - 15;
  float val = ldexp(mant, expn);
  return sign ? -val : val;
}

inline uchar float_to_fp8_e5m2(float f) {
  const uint sign = as_type<uint>(f) >> 31;
  const uchar sign_bits = uchar(sign << 7);
  if (isnan(f)) {
    return uchar(sign_bits | 0x7D);
  }
  const float magnitude = fabs(f);
  if (magnitude == 0.0f) {
    return sign_bits;
  }
  if (isinf(magnitude) || magnitude > 57344.0f) {
    return uchar(sign_bits | 0x7C);
  }
  if (magnitude < 0x1p-14f) {
    uint mantissa = uint(rint(ldexp(magnitude, 16)));
    if (mantissa == 0) {
      return sign_bits;
    }
    if (mantissa >= 4) {
      return uchar(sign_bits | 0x04);
    }
    return uchar(sign_bits | mantissa);
  }

  int exponent;
  (void)frexp(magnitude, exponent);
  int unbiased = exponent - 1;
  uint exponent_bits = uint(unbiased + 15);
  float significand = ldexp(magnitude, -unbiased);
  uint mantissa = uint(rint((significand - 1.0f) * 4.0f));
  if (mantissa == 4) {
    mantissa = 0;
    exponent_bits += 1;
  }
  if (exponent_bits >= 31) {
    return uchar(sign_bits | 0x7C);
  }
  return uchar(sign_bits | (exponent_bits << 2) | mantissa);
}
