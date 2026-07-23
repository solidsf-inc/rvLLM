//! Validated CPU references for fused-kernel parity tests.

pub use crate::fp8::FP8_E4M3_MAX;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceError(pub &'static str);

impl std::fmt::Display for ReferenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for ReferenceError {}

pub type ReferenceResult<T> = std::result::Result<T, ReferenceError>;

fn checked_product(values: &[usize]) -> ReferenceResult<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(ReferenceError("reference shape overflow"))
    })
}

fn validate_eps(eps: f32) -> ReferenceResult<()> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(ReferenceError("epsilon must be finite and non-negative"));
    }
    Ok(())
}

fn validate_finite(values: &[f32]) -> ReferenceResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(ReferenceError("reference input must be finite"));
    }
    Ok(())
}

pub fn rmsnorm_ref(
    x: &[f32],
    gamma: &[f32],
    eps: f32,
    hidden: usize,
    out: &mut [f32],
) -> ReferenceResult<()> {
    validate_eps(eps)?;
    if hidden == 0
        || x.is_empty()
        || !x.len().is_multiple_of(hidden)
        || gamma.len() != hidden
        || out.len() != x.len()
    {
        return Err(ReferenceError("invalid RMSNorm shapes"));
    }
    validate_finite(x)?;
    validate_finite(gamma)?;
    for (row_in, row_out) in x.chunks_exact(hidden).zip(out.chunks_exact_mut(hidden)) {
        let ms = row_in.iter().try_fold(0.0f32, |sum, value| {
            let next = sum + value * value;
            next.is_finite()
                .then_some(next)
                .ok_or(ReferenceError("RMSNorm accumulation overflow"))
        })? / hidden as f32;
        let denominator = ms + eps;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(ReferenceError("RMSNorm denominator is not positive"));
        }
        let inv = denominator.sqrt().recip();
        for (output, (value, weight)) in row_out.iter_mut().zip(row_in.iter().zip(gamma)) {
            let next = value * inv * weight;
            if !next.is_finite() {
                return Err(ReferenceError("RMSNorm output overflow"));
            }
            *output = next;
        }
    }
    Ok(())
}

pub fn quantize_fp8_per_token_ref(
    x: &[f32],
    hidden: usize,
    out_fp8: &mut [u8],
    scales: &mut [f32],
) -> ReferenceResult<()> {
    if hidden == 0 || x.is_empty() || !x.len().is_multiple_of(hidden) {
        return Err(ReferenceError("invalid FP8 quantization shape"));
    }
    let rows = x.len() / hidden;
    if out_fp8.len() != x.len() || scales.len() != rows {
        return Err(ReferenceError("invalid FP8 quantization output shape"));
    }
    validate_finite(x)?;
    for (row_index, row) in x.chunks_exact(hidden).enumerate() {
        let amax = row
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max)
            .max(1e-12);
        let scale = amax / FP8_E4M3_MAX;
        scales[row_index] = scale;
        let output = &mut out_fp8[row_index * hidden..(row_index + 1) * hidden];
        for (encoded, value) in output.iter_mut().zip(row) {
            *encoded = f32_to_fp8_e4m3(*value / scale);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn fused_add_rmsnorm_fp8_quant_ref(
    x: &[f32],
    residual_in: &[f32],
    gamma: &[f32],
    eps: f32,
    hidden: usize,
    residual_out: &mut [f32],
    fp8_out: &mut [u8],
    scales: &mut [f32],
) -> ReferenceResult<()> {
    if x.len() != residual_in.len() || residual_out.len() != x.len() {
        return Err(ReferenceError("invalid fused add/RMSNorm shapes"));
    }
    for ((output, left), right) in residual_out.iter_mut().zip(x).zip(residual_in) {
        let next = left + right;
        if !next.is_finite() {
            return Err(ReferenceError("fused residual addition overflow"));
        }
        *output = next;
    }
    let mut normed = vec![0.0f32; x.len()];
    rmsnorm_ref(residual_out, gamma, eps, hidden, &mut normed)?;
    quantize_fp8_per_token_ref(&normed, hidden, fp8_out, scales)
}

pub fn fused_silu_mul_fp8_quant_ref(
    gate_up: &[f32],
    num_tokens: usize,
    intermediate: usize,
    fp8_out: &mut [u8],
    scales: &mut [f32],
) -> ReferenceResult<()> {
    let input_len = checked_product(&[num_tokens, 2, intermediate])?;
    let output_len = checked_product(&[num_tokens, intermediate])?;
    if num_tokens == 0
        || intermediate == 0
        || gate_up.len() != input_len
        || fp8_out.len() != output_len
        || scales.len() != num_tokens
    {
        return Err(ReferenceError("invalid SiLU fused-reference shapes"));
    }
    validate_finite(gate_up)?;
    let mut output = vec![0.0f32; output_len];
    for token in 0..num_tokens {
        let base = token * 2 * intermediate;
        for index in 0..intermediate {
            let gate = gate_up[base + index];
            output[token * intermediate + index] =
                gate / (1.0 + (-gate).exp()) * gate_up[base + intermediate + index];
        }
    }
    validate_finite(&output)?;
    quantize_fp8_per_token_ref(&output, intermediate, fp8_out, scales)
}

pub fn fused_gelu_mul_fp8_quant_ref(
    gate_up: &[f32],
    num_tokens: usize,
    intermediate: usize,
    fp8_out: &mut [u8],
    scales: &mut [f32],
) -> ReferenceResult<()> {
    let input_len = checked_product(&[num_tokens, 2, intermediate])?;
    let output_len = checked_product(&[num_tokens, intermediate])?;
    if num_tokens == 0
        || intermediate == 0
        || gate_up.len() != input_len
        || fp8_out.len() != output_len
        || scales.len() != num_tokens
    {
        return Err(ReferenceError("invalid GELU fused-reference shapes"));
    }
    validate_finite(gate_up)?;
    let mut output = vec![0.0f32; output_len];
    for token in 0..num_tokens {
        let base = token * 2 * intermediate;
        for index in 0..intermediate {
            let gate = gate_up[base + index];
            let inner = 0.797_884_6 * (gate + 0.044_715 * gate * gate * gate);
            let gelu = 0.5 * gate * (1.0 + inner.tanh());
            output[token * intermediate + index] = gelu * gate_up[base + intermediate + index];
        }
    }
    validate_finite(&output)?;
    quantize_fp8_per_token_ref(&output, intermediate, fp8_out, scales)
}

pub fn argmax_ref(
    logits: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [i32],
) -> ReferenceResult<()> {
    if rows == 0
        || cols == 0
        || logits.len() != checked_product(&[rows, cols])?
        || out.len() != rows
        || cols > i32::MAX as usize
    {
        return Err(ReferenceError("invalid argmax shapes"));
    }
    validate_finite(logits)?;
    for (row_index, row) in logits.chunks_exact(cols).enumerate() {
        let mut best = 0usize;
        for index in 1..cols {
            if row[index] > row[best] {
                best = index;
            }
        }
        out[row_index] = best as i32;
    }
    Ok(())
}

pub fn residual_add_ref(x: &mut [f32], y: &[f32]) -> ReferenceResult<()> {
    if x.is_empty() || x.len() != y.len() {
        return Err(ReferenceError("invalid residual-add shapes"));
    }
    validate_finite(x)?;
    validate_finite(y)?;
    for (left, right) in x.iter_mut().zip(y) {
        let next = *left + right;
        if !next.is_finite() {
            return Err(ReferenceError("residual addition overflow"));
        }
        *left = next;
    }
    Ok(())
}

pub fn embedding_gather_ref(
    token_ids: &[u32],
    weight: &[f32],
    hidden: usize,
    vocab: usize,
    out: &mut [f32],
) -> ReferenceResult<()> {
    if token_ids.is_empty()
        || hidden == 0
        || vocab == 0
        || weight.len() != checked_product(&[hidden, vocab])?
        || out.len() != checked_product(&[hidden, token_ids.len()])?
    {
        return Err(ReferenceError("invalid embedding-gather shapes"));
    }
    validate_finite(weight)?;
    for (token_index, token_id) in token_ids.iter().copied().enumerate() {
        let row = token_id as usize;
        if row >= vocab {
            return Err(ReferenceError("token id is outside the vocabulary"));
        }
        out[token_index * hidden..(token_index + 1) * hidden]
            .copy_from_slice(&weight[row * hidden..(row + 1) * hidden]);
    }
    Ok(())
}

pub fn rope_ref(
    q: &mut [f32],
    positions: &[u32],
    cos: &[f32],
    sin: &[f32],
    num_heads: usize,
    head_dim: usize,
) -> ReferenceResult<()> {
    if positions.is_empty()
        || num_heads == 0
        || head_dim == 0
        || !head_dim.is_multiple_of(2)
        || q.len() != checked_product(&[positions.len(), num_heads, head_dim])?
        || cos.len() != sin.len()
    {
        return Err(ReferenceError("invalid RoPE shapes"));
    }
    validate_finite(q)?;
    validate_finite(cos)?;
    validate_finite(sin)?;
    let pairs = head_dim / 2;
    let positions_supported = cos.len() / pairs;
    if !cos.len().is_multiple_of(pairs)
        || positions
            .iter()
            .any(|position| *position as usize >= positions_supported)
    {
        return Err(ReferenceError("RoPE position is outside the table"));
    }
    for (token, position) in positions.iter().copied().enumerate() {
        let table = position as usize * pairs;
        for head in 0..num_heads {
            let base = (token * num_heads + head) * head_dim;
            for pair in 0..pairs {
                let first = q[base + pair];
                let second = q[base + pair + pairs];
                q[base + pair] = first * cos[table + pair] - second * sin[table + pair];
                q[base + pair + pairs] = first * sin[table + pair] + second * cos[table + pair];
            }
        }
    }
    validate_finite(q)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// f32 → FP8 E4M3 round-half-to-even with saturation to ±448. The canonical
// encoder lives in `crate::fp8` so this and the Gemma 4 reference share one
// implementation that matches the CUDA kernel's __NV_SATFINITE rounding.
// ---------------------------------------------------------------------------

pub(crate) use crate::fp8::f32_to_fp8_e4m3;

/// Decode a positive FP8 E4M3 code to f32 (test support: round-trip identity
/// against the canonical `crate::fp8` encoder).
#[cfg(test)]
fn fp8_e4m3_positive_to_f32(code: u8) -> f32 {
    let exponent = (code >> 3) & 0x0f;
    let mantissa = code & 0x07;
    if exponent == 0 {
        mantissa as f32 * 2.0f32.powi(-9)
    } else {
        (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_matches_hand_calculation() {
        let x = [1.0f32, 2.0, 3.0];
        let gamma = [1.0f32; 3];
        let mut output = [0.0f32; 3];
        rmsnorm_ref(&x, &gamma, 0.0, 3, &mut output).unwrap();
        let rms = (14f32 / 3.0).sqrt();
        for index in 0..3 {
            assert!((output[index] - x[index] / rms).abs() < 1e-6);
        }
    }

    #[test]
    fn public_helpers_reject_zero_dimensions() {
        assert!(rmsnorm_ref(&[], &[], 1e-6, 0, &mut []).is_err());
        assert!(argmax_ref(&[], 0, 0, &mut []).is_err());
        assert!(quantize_fp8_per_token_ref(&[], 0, &mut [], &mut []).is_err());
    }

    #[test]
    fn epsilon_must_be_finite() {
        let mut output = [0.0];
        assert!(rmsnorm_ref(&[1.0], &[1.0], f32::NAN, 1, &mut output).is_err());
    }

    #[test]
    fn fp8_rounding_is_ties_to_even() {
        assert_eq!(f32_to_fp8_e4m3(1.0625), 0x38);
        assert_eq!(f32_to_fp8_e4m3(1.1875), 0x3a);
        assert_eq!(f32_to_fp8_e4m3(2.0f32.powi(-9)), 0x01);
    }

    #[test]
    fn fp8_canonicalizes_zero_and_nan() {
        // Canonical `crate::fp8` contract: every zero result is the one
        // deterministic +0 byte, and every NaN is canonical 0x7f — matching
        // the shared encoder used by the loader and Gemma 4 references.
        assert_eq!(f32_to_fp8_e4m3(-0.0), 0x00);
        assert_eq!(f32_to_fp8_e4m3(f32::NAN), 0x7f);
        assert_eq!(f32_to_fp8_e4m3(f32::from_bits(0xffc0_0000)), 0x7f);
    }

    #[test]
    fn fp8_roundtrips_every_finite_positive_code() {
        for code in 0u8..=0x7e {
            assert_eq!(f32_to_fp8_e4m3(fp8_e4m3_positive_to_f32(code)), code);
        }
    }

    #[test]
    fn argmax_and_embedding_are_fallible() {
        let mut indices = [0i32; 2];
        argmax_ref(&[0.1, 0.9, 0.3, 0.5, -1.0, 0.8], 2, 3, &mut indices).unwrap();
        assert_eq!(indices, [1, 2]);

        let mut output = [0.0f32; 4];
        embedding_gather_ref(
            &[2, 0],
            &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
            2,
            3,
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [30.0, 31.0, 10.0, 11.0]);
        assert!(embedding_gather_ref(&[3], &[0.0; 6], 2, 3, &mut [0.0; 2]).is_err());
    }

    #[test]
    fn fused_quant_and_rope_validate_and_run() {
        let gate_up = [0.0, 1.0, -1.0, 2.0, 1.0, 1.0, 1.0, 1.0];
        let mut fp8 = [0u8; 4];
        let mut scale = [0.0f32; 1];
        fused_silu_mul_fp8_quant_ref(&gate_up, 1, 4, &mut fp8, &mut scale).unwrap();
        assert!(fp8.iter().any(|value| *value != 0));

        let mut q = [1.0, 0.0, 0.0, 1.0];
        rope_ref(&mut q, &[0], &[1.0, 1.0], &[0.0, 0.0], 1, 4).unwrap();
        assert_eq!(q, [1.0, 0.0, 0.0, 1.0]);
    }
}
