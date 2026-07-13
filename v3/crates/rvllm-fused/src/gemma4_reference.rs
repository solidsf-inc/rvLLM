//! Validated Gemma 4 CPU references used by kernel parity tests.

use crate::reference::{f32_to_fp8_e4m3, ReferenceError, ReferenceResult, FP8_E4M3_MAX};

fn checked_product(values: &[usize]) -> ReferenceResult<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(ReferenceError("Gemma reference shape overflow"))
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
        return Err(ReferenceError("Gemma reference input must be finite"));
    }
    Ok(())
}

pub fn gelu_tanh(x: f32) -> f32 {
    let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

pub fn fused_gelu_mul_fp8_quant_ref(
    gate_up: &[f32],
    intermediate: usize,
    out_fp8: &mut [u8],
    scales: &mut [f32],
) -> ReferenceResult<()> {
    let row_width = checked_product(&[2, intermediate])?;
    if intermediate == 0 || gate_up.is_empty() || !gate_up.len().is_multiple_of(row_width) {
        return Err(ReferenceError("invalid Gemma GELU reference shape"));
    }
    validate_finite(gate_up)?;
    let rows = gate_up.len() / row_width;
    if out_fp8.len() != checked_product(&[rows, intermediate])? || scales.len() != rows {
        return Err(ReferenceError("invalid Gemma GELU output shape"));
    }
    for row in 0..rows {
        let base = row * 2 * intermediate;
        let gate = &gate_up[base..base + intermediate];
        let up = &gate_up[base + intermediate..base + 2 * intermediate];
        let values: Vec<_> = gate
            .iter()
            .zip(up)
            .map(|(gate, up)| gelu_tanh(*gate) * up)
            .collect();
        validate_finite(&values)?;
        let amax = values
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max)
            .max(1e-12);
        let scale = amax / FP8_E4M3_MAX;
        scales[row] = scale;
        for (index, value) in values.iter().enumerate() {
            out_fp8[row * intermediate + index] = f32_to_fp8_e4m3(*value / scale);
        }
    }
    Ok(())
}

pub fn qk_rmsnorm_ref(
    input: &[f32],
    gamma: &[f32],
    eps: f32,
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    output: &mut [f32],
) -> ReferenceResult<()> {
    validate_eps(eps)?;
    let expected = checked_product(&[num_tokens, num_heads, head_dim])?;
    if num_tokens == 0
        || num_heads == 0
        || head_dim == 0
        || input.len() != expected
        || output.len() != expected
        || gamma.len() != head_dim
    {
        return Err(ReferenceError("invalid QK RMSNorm shapes"));
    }
    validate_finite(input)?;
    validate_finite(gamma)?;
    for (head, out) in input
        .chunks_exact(head_dim)
        .zip(output.chunks_exact_mut(head_dim))
    {
        let ms = head.iter().try_fold(0.0f32, |sum, value| {
            let next = sum + value * value;
            next.is_finite()
                .then_some(next)
                .ok_or(ReferenceError("QK RMSNorm accumulation overflow"))
        })? / head_dim as f32;
        let denominator = ms + eps;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(ReferenceError("QK RMSNorm denominator is not positive"));
        }
        let inv = denominator.sqrt().recip();
        for (output, (value, weight)) in out.iter_mut().zip(head.iter().zip(gamma)) {
            let next = value * inv * weight;
            if !next.is_finite() {
                return Err(ReferenceError("QK RMSNorm output overflow"));
            }
            *output = next;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn partial_rope_ref(
    x: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    positions: &[i32],
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> ReferenceResult<()> {
    if num_tokens == 0
        || num_heads == 0
        || head_dim == 0
        || !head_dim.is_multiple_of(2)
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || positions.len() != num_tokens
        || x.len() != checked_product(&[num_tokens, num_heads, head_dim])?
        || cos.len() != sin.len()
    {
        return Err(ReferenceError("invalid partial-RoPE shapes"));
    }
    validate_finite(x)?;
    validate_finite(cos)?;
    validate_finite(sin)?;
    let pairs = rotary_dim / 2;
    let half_head = head_dim / 2;
    if !cos.len().is_multiple_of(pairs)
        || positions
            .iter()
            .any(|position| *position < 0 || *position as usize >= cos.len() / pairs)
    {
        return Err(ReferenceError("partial-RoPE position is outside the table"));
    }
    for (token, position) in positions.iter().copied().enumerate() {
        let table = position as usize * pairs;
        for head in 0..num_heads {
            let base = (token * num_heads + head) * head_dim;
            for pair in 0..pairs {
                let first = x[base + pair];
                let second = x[base + half_head + pair];
                x[base + pair] = first * cos[table + pair] - second * sin[table + pair];
                x[base + half_head + pair] = first * sin[table + pair] + second * cos[table + pair];
            }
        }
    }
    validate_finite(x)?;
    Ok(())
}

pub fn rmsnorm_gamma_ref(
    x: &[f32],
    gamma: &[f32],
    eps: f32,
    out: &mut [f32],
) -> ReferenceResult<()> {
    validate_eps(eps)?;
    if x.is_empty() || gamma.len() != x.len() || out.len() != x.len() {
        return Err(ReferenceError("invalid gamma RMSNorm shapes"));
    }
    validate_finite(x)?;
    validate_finite(gamma)?;
    let ms = x.iter().try_fold(0.0f32, |sum, value| {
        let next = sum + value * value;
        next.is_finite()
            .then_some(next)
            .ok_or(ReferenceError("gamma RMSNorm accumulation overflow"))
    })? / x.len() as f32;
    let denominator = ms + eps;
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(ReferenceError("gamma RMSNorm denominator is not positive"));
    }
    let inv = denominator.sqrt().recip();
    for (output, (value, weight)) in out.iter_mut().zip(x.iter().zip(gamma)) {
        let next = value * inv * weight;
        if !next.is_finite() {
            return Err(ReferenceError("gamma RMSNorm output overflow"));
        }
        *output = next;
    }
    Ok(())
}

pub fn ple_gather_scale_ref(
    embed_row: &[f32],
    num_layers: usize,
    h_ple: usize,
    embed_scale_folded: bool,
) -> ReferenceResult<Vec<f32>> {
    if num_layers == 0 || h_ple == 0 || embed_row.len() != checked_product(&[num_layers, h_ple])? {
        return Err(ReferenceError("invalid PLE gather shape"));
    }
    validate_finite(embed_row)?;
    let scale = if embed_scale_folded {
        1.0
    } else {
        (h_ple as f32).sqrt()
    };
    let output: Vec<_> = embed_row.iter().map(|value| value * scale).collect();
    validate_finite(&output)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn ple_model_projection_combine_ref(
    proj_in: &[f32],
    per_layer_inputs: &[f32],
    proj_norm_gamma: &[f32],
    num_layers: usize,
    h_ple: usize,
    hidden: usize,
    eps: f32,
) -> ReferenceResult<Vec<f32>> {
    validate_eps(eps)?;
    let expected = checked_product(&[num_layers, h_ple])?;
    if num_layers == 0
        || h_ple == 0
        || hidden == 0
        || proj_in.len() != expected
        || per_layer_inputs.len() != expected
        || proj_norm_gamma.len() != h_ple
    {
        return Err(ReferenceError("invalid PLE projection-combine shapes"));
    }
    validate_finite(proj_in)?;
    validate_finite(per_layer_inputs)?;
    validate_finite(proj_norm_gamma)?;
    let projection_scale = (hidden as f32).sqrt().recip();
    let input_scale = 2.0f32.sqrt().recip();
    let mut output = vec![0.0f32; expected];
    let mut normed = vec![0.0f32; h_ple];
    let mut scaled = vec![0.0f32; h_ple];
    for layer in 0..num_layers {
        let base = layer * h_ple;
        for index in 0..h_ple {
            scaled[index] = proj_in[base + index] * projection_scale;
        }
        rmsnorm_gamma_ref(&scaled, proj_norm_gamma, eps, &mut normed)?;
        for index in 0..h_ple {
            output[base + index] = (normed[index] + per_layer_inputs[base + index]) * input_scale;
        }
    }
    validate_finite(&output)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn ple_gate_ref(
    residual: &mut [f32],
    hidden_states: &[f32],
    per_layer_input: &[f32],
    gate_w: &[f32],
    proj_w: &[f32],
    post_norm_gamma: &[f32],
    hidden: usize,
    h_ple: usize,
    eps: f32,
) -> ReferenceResult<()> {
    validate_eps(eps)?;
    if hidden == 0
        || h_ple == 0
        || residual.len() != hidden
        || hidden_states.len() != hidden
        || per_layer_input.len() != h_ple
        || gate_w.len() != checked_product(&[h_ple, hidden])?
        || proj_w.len() != checked_product(&[hidden, h_ple])?
        || post_norm_gamma.len() != hidden
    {
        return Err(ReferenceError("invalid PLE gate shapes"));
    }
    validate_finite(residual)?;
    validate_finite(hidden_states)?;
    validate_finite(per_layer_input)?;
    validate_finite(gate_w)?;
    validate_finite(proj_w)?;
    validate_finite(post_norm_gamma)?;

    let mut gated = vec![0.0f32; h_ple];
    for output in 0..h_ple {
        let row = &gate_w[output * hidden..(output + 1) * hidden];
        let value = row
            .iter()
            .zip(hidden_states)
            .map(|(weight, input)| weight * input)
            .sum::<f32>();
        gated[output] = gelu_tanh(value) * per_layer_input[output];
    }
    let mut contribution = vec![0.0f32; hidden];
    for output in 0..hidden {
        let row = &proj_w[output * h_ple..(output + 1) * h_ple];
        contribution[output] = row
            .iter()
            .zip(&gated)
            .map(|(weight, input)| weight * input)
            .sum();
    }
    validate_finite(&contribution)?;
    let mut normed = vec![0.0f32; hidden];
    rmsnorm_gamma_ref(&contribution, post_norm_gamma, eps, &mut normed)?;
    for (residual, contribution) in residual.iter_mut().zip(normed) {
        *residual += contribution;
        if !residual.is_finite() {
            return Err(ReferenceError("PLE residual addition overflow"));
        }
    }
    Ok(())
}

pub fn logit_softcap_ref(logits: &mut [f32], cap: f32) -> ReferenceResult<()> {
    if logits.is_empty() || !cap.is_finite() || cap <= 0.0 {
        return Err(ReferenceError("invalid logit-softcap inputs"));
    }
    validate_finite(logits)?;
    for value in logits {
        *value = cap * (*value / cap).tanh();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_slice_apis_reject_bad_shapes() {
        assert!(qk_rmsnorm_ref(&[], &[], 1e-6, 0, 0, 0, &mut []).is_err());
        assert!(partial_rope_ref(&mut [], &[], &[], &[], 0, 0, 0, 0).is_err());
        assert!(rmsnorm_gamma_ref(&[], &[], f32::NAN, &mut []).is_err());
        assert!(ple_gather_scale_ref(&[], 0, 0, false).is_err());
    }

    #[test]
    fn gemma4_global_partial_rope_golden_vector() {
        const HEADS: usize = 32;
        const HEAD_DIM: usize = 512;
        const ROTARY_DIM: usize = 128;
        let mut input = vec![0.0f32; HEADS * HEAD_DIM];
        input[0] = 1.0;
        input[HEAD_DIM / 2] = 2.0;
        input[1] = 3.0;
        input[HEAD_DIM / 2 + 1] = 4.0;
        input[ROTARY_DIM / 2] = 7.0;
        let mut cos = vec![1.0f32; 8 * (ROTARY_DIM / 2)];
        let mut sin = vec![0.0f32; 8 * (ROTARY_DIM / 2)];
        let base = 7 * (ROTARY_DIM / 2);
        cos[base] = 0.5;
        sin[base] = 0.866_025_4;
        cos[base + 1] = -0.25;
        sin[base + 1] = 0.968_245_86;
        partial_rope_ref(&mut input, &cos, &sin, &[7], 1, HEADS, HEAD_DIM, ROTARY_DIM).unwrap();
        let expected = [-1.232_050_8, 1.866_025_4, -4.622_983_5, 1.904_737_5];
        let actual = [
            input[0],
            input[HEAD_DIM / 2],
            input[1],
            input[HEAD_DIM / 2 + 1],
        ];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }
        assert_eq!(input[ROTARY_DIM / 2], 7.0);
    }

    #[test]
    fn ple_projection_and_gate_are_fallible() {
        let output =
            ple_model_projection_combine_ref(&[4.0, 8.0], &[1.0, 2.0], &[1.0, 1.0], 1, 2, 4, 1e-6)
                .unwrap();
        assert_eq!(output.len(), 2);

        let mut residual = vec![1.0, 2.0, 3.0, 4.0];
        let before = residual.clone();
        ple_gate_ref(
            &mut residual,
            &before,
            &[0.0, 0.0],
            &[0.5; 8],
            &[0.5; 8],
            &[1.0; 4],
            4,
            2,
            1e-6,
        )
        .unwrap();
        assert_eq!(residual, before);
    }

    #[test]
    fn fp8_reference_preserves_rounding_contract() {
        let mut output = [0u8; 2];
        let mut scales = [0.0f32; 1];
        fused_gelu_mul_fp8_quant_ref(&[1.0, -1.0, 1.0, 1.0], 2, &mut output, &mut scales).unwrap();
        assert!(scales[0] > 0.0);
        assert_ne!(output, [0, 0]);
    }

    #[test]
    fn softcap_validates_cap() {
        let mut logits = [1000.0, -1000.0];
        logit_softcap_ref(&mut logits, 30.0).unwrap();
        assert!((logits[0] - 30.0).abs() < 0.01);
        assert!(logit_softcap_ref(&mut logits, 0.0).is_err());
    }
}
