//! Canonical f32 to FP8 E4M3FN encoder shared by host reference paths.
//!
//! The finite encode path mirrors NVIDIA `__NV_SATFINITE` rounding and
//! saturation, except that all zero results use one deterministic `+0` byte.

pub const FP8_E4M3_MAX: f32 = 448.0;

/// Encode an f32 as FP8 E4M3FN with round-half-to-even and finite saturation.
///
/// NaN becomes canonical NaN (`0x7f`), values outside the finite range
/// saturate to +/-448, and signed zero or complete underflow becomes `0x00`.
pub fn f32_to_fp8_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.to_bits() >> 31 != 0 { 0x80 } else { 0 };
    let magnitude = x.abs();
    if magnitude == 0.0 {
        return 0;
    }
    if magnitude > FP8_E4M3_MAX {
        return sign | 0x7e;
    }

    let bits = magnitude.to_bits();
    let exp32 = ((bits >> 23) & 0xff) as i32 - 127;
    let mant32 = bits & 0x7f_ffff;
    let mut exp8 = exp32 + 7;

    if exp8 <= 0 {
        let shift = 1 - exp8;
        let full = mant32 | (1 << 23);
        let rshift = (20 + shift) as u32;
        if rshift >= 32 {
            return 0;
        }
        let mut mantissa = full >> rshift;
        let round_bit = (full >> (rshift - 1)) & 1;
        let sticky = (full & ((1 << (rshift - 1)) - 1) != 0) as u32;
        mantissa += round_bit & (sticky | (mantissa & 1));
        if mantissa >= 8 {
            return sign | 0x08;
        }
        if mantissa == 0 {
            return 0;
        }
        return sign | (mantissa as u8 & 0x07);
    }

    let truncated = mant32 >> 20;
    let round_bit = (mant32 >> 19) & 1;
    let sticky = (mant32 & 0x7_ffff) != 0;
    let mantissa = truncated + (round_bit & (sticky as u32 | (truncated & 1)));
    if mantissa >= 8 {
        exp8 += 1;
        if exp8 > 15 {
            return sign | 0x7e;
        }
        return sign | ((exp8 as u8 & 0x0f) << 3);
    }
    if exp8 > 15 {
        return sign | 0x7e;
    }
    sign | ((exp8 as u8 & 0x0f) << 3) | (mantissa as u8 & 0x07)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_complete_underflow_are_canonical() {
        assert_eq!(f32_to_fp8_e4m3(-0.0), 0x00);
        assert_eq!(f32_to_fp8_e4m3(0.0), 0x00);
        assert_eq!(f32_to_fp8_e4m3(f32::from_bits(1)), 0x00);
        assert_eq!(f32_to_fp8_e4m3(-f32::from_bits(1)), 0x00);
    }

    #[test]
    fn saturation_boundary_is_finite() {
        assert_eq!(f32_to_fp8_e4m3(256.0), 0x78);
        assert_eq!(f32_to_fp8_e4m3(288.0), 0x79);
        assert_eq!(f32_to_fp8_e4m3(448.0), 0x7e);
        assert_eq!(f32_to_fp8_e4m3(-448.0), 0xfe);
        assert_eq!(f32_to_fp8_e4m3(449.0), 0x7e);
        assert_eq!(f32_to_fp8_e4m3(-10_000.0), 0xfe);
    }

    #[test]
    fn round_half_to_even_at_a_tie() {
        assert_eq!(f32_to_fp8_e4m3(1.0625), 0x38);
        assert_eq!(f32_to_fp8_e4m3(1.1875), 0x3a);
        assert_eq!(f32_to_fp8_e4m3(1.1), 0x39);
    }

    #[test]
    fn denormal_underflow_rounds_to_nearest() {
        let min_subnormal = 2f32.powi(-9);
        assert_eq!(f32_to_fp8_e4m3(min_subnormal), 0x01);
        assert_eq!(f32_to_fp8_e4m3(min_subnormal / 2.0), 0x00);
        assert_eq!(f32_to_fp8_e4m3(-min_subnormal / 4.0), 0x00);
        assert_eq!(f32_to_fp8_e4m3(min_subnormal * 0.75), 0x01);
    }
}
