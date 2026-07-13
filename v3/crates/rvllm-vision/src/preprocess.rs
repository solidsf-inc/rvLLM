// Ported from huggingface/transformers revision
// 10555512868d663ee1ff627e4f5c5c260114235b:
// src/transformers/models/gemma4/image_processing_gemma4.py
// Apache-2.0 License, Copyright (c) HuggingFace and Google.
// Rust port of aspect-ratio resize, channel-last patch conversion, padding,
// and position IDs. CatmullRom is not byte-identical to torchvision bicubic.

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use image::{imageops, DynamicImage, GenericImageView, RgbImage};

use crate::config::{MAX_VISION_PATCHES, MAX_VISION_SOFT_TOKENS};

/// Released Gemma 4 image-processor defaults.
pub const DEFAULT_POOLING_KERNEL_SIZE: u32 = 3;
pub const DEFAULT_MAX_SOFT_TOKENS: u32 = 280;
pub const DEFAULT_PADDED_PATCHES: usize = 2520; // 280 * 3^2
const MAX_SOURCE_PIXELS: u64 = 100_000_000;
const MAX_TARGET_PIXELS: u64 = 100_000_000;
const MAX_PREPARED_FLOATS: usize = 64_000_000;

pub struct PreparedImage {
    /// Channel-last patches `[1,max_patches,3*patch_size^2]` in `[0,1]`.
    pub pixel_values: Tensor,
    /// `[1, max_patches, 2]` i64, each row `(x, y)` patch coordinates. Padding
    /// rows (beyond the valid patch count) are `(-1, -1)`.
    pub pixel_position_ids: Tensor,
    /// Valid (non-padded) patch grid dimensions.
    pub patch_grid_w: u32,
    pub patch_grid_h: u32,
}

/// Largest area-preserving size within the patch budget and pooling stride.
fn aspect_ratio_preserving_size(
    height: u32,
    width: u32,
    patch_size: u32,
    max_patches: u32,
    pooling_kernel_size: u32,
) -> Result<(u32, u32)> {
    if height == 0 || width == 0 || patch_size == 0 || pooling_kernel_size == 0 {
        return Err(anyhow!(
            "image, patch, and pooling dimensions must be nonzero"
        ));
    }
    if max_patches == 0 || max_patches as usize > MAX_VISION_PATCHES {
        return Err(anyhow!("max_patches must be in 1..={MAX_VISION_PATCHES}"));
    }
    let total_px = (height as f64) * (width as f64);
    let target_px = (max_patches as f64) * (patch_size as f64).powi(2);
    let factor = (target_px / total_px).sqrt();
    let ideal_height = factor * (height as f64);
    let ideal_width = factor * (width as f64);
    let side_mult = pooling_kernel_size
        .checked_mul(patch_size)
        .ok_or_else(|| anyhow!("pooling_kernel_size * patch_size overflow"))?;
    let mut target_height = ((ideal_height / side_mult as f64).floor() as u32) * side_mult;
    let mut target_width = ((ideal_width / side_mult as f64).floor() as u32) * side_mult;
    if target_height == 0 && target_width == 0 {
        return Err(anyhow!(
            "resize target is 0x0; resized dims must be divisible by pooling_kernel_size*patch_size={side_mult}"
        ));
    }
    let pooling_square = pooling_kernel_size
        .checked_mul(pooling_kernel_size)
        .ok_or_else(|| anyhow!("pooling_kernel_size^2 overflow"))?;
    let max_side_length = (max_patches / pooling_square)
        .checked_mul(side_mult)
        .ok_or_else(|| anyhow!("maximum resize side overflow"))?;
    if target_height == 0 {
        target_height = side_mult;
        target_width =
            (((width as f64 / height as f64).floor() as u32) * side_mult).min(max_side_length);
    } else if target_width == 0 {
        target_width = side_mult;
        target_height =
            (((height as f64 / width as f64).floor() as u32) * side_mult).min(max_side_length);
    }
    if (target_height as f64) * (target_width as f64) > target_px {
        return Err(anyhow!(
            "resize target {target_height}x{target_width} exceeds the {max_patches}-patch budget at patch_size {patch_size}"
        ));
    }
    let target_pixels = u64::from(target_height)
        .checked_mul(u64::from(target_width))
        .ok_or_else(|| anyhow!("resize target pixel count overflow"))?;
    if target_pixels == 0 || target_pixels > MAX_TARGET_PIXELS {
        return Err(anyhow!(
            "resize target has {target_pixels} pixels; limit is {MAX_TARGET_PIXELS}"
        ));
    }
    Ok((target_height, target_width))
}

pub fn prepare_image(
    img: &DynamicImage,
    patch_size: u32,
    pooling_kernel_size: u32,
    max_soft_tokens: u32,
) -> Result<PreparedImage> {
    if patch_size == 0 {
        return Err(anyhow!("patch_size must be > 0"));
    }
    if pooling_kernel_size == 0 {
        return Err(anyhow!("pooling_kernel_size must be > 0"));
    }
    if max_soft_tokens == 0 || max_soft_tokens > MAX_VISION_SOFT_TOKENS {
        return Err(anyhow!(
            "max_soft_tokens must be in 1..={MAX_VISION_SOFT_TOKENS}"
        ));
    }
    let pooling_square = pooling_kernel_size
        .checked_mul(pooling_kernel_size)
        .ok_or_else(|| anyhow!("pooling_kernel_size^2 overflow"))?;
    let max_patches = max_soft_tokens
        .checked_mul(pooling_square)
        .ok_or_else(|| anyhow!("max_soft_tokens * pooling_kernel_size^2 overflow"))?;
    if max_patches as usize > MAX_VISION_PATCHES {
        return Err(anyhow!(
            "patch budget {max_patches} exceeds {MAX_VISION_PATCHES}"
        ));
    }
    let (w0, h0) = img.dimensions();
    if w0 == 0 || h0 == 0 {
        return Err(anyhow!("empty image: {}x{}", w0, h0));
    }
    let source_pixels = u64::from(w0)
        .checked_mul(u64::from(h0))
        .ok_or_else(|| anyhow!("source image pixel count overflow"))?;
    if source_pixels > MAX_SOURCE_PIXELS {
        return Err(anyhow!(
            "source image has {source_pixels} pixels; limit is {MAX_SOURCE_PIXELS}"
        ));
    }
    let (target_h, target_w) =
        aspect_ratio_preserving_size(h0, w0, patch_size, max_patches, pooling_kernel_size)?;
    // Resize (HF: torchvision BICUBIC + antialias). `image`'s CatmullRom filter
    // is the closest bicubic-family kernel available in this crate.
    let rgb_src = img.to_rgb8();
    let resized: RgbImage = if (target_w, target_h) == (w0, h0) {
        rgb_src
    } else {
        imageops::resize(
            &rgb_src,
            target_w,
            target_h,
            imageops::FilterType::CatmullRom,
        )
    };
    let patch_grid_w = target_w / patch_size;
    let patch_grid_h = target_h / patch_size;
    let num_patches = (patch_grid_w as usize)
        .checked_mul(patch_grid_h as usize)
        .ok_or_else(|| anyhow!("patch count overflow"))?;
    if num_patches == 0 {
        return Err(anyhow!(
            "no patches after preprocessing (resized to {}x{}, patch {})",
            target_w,
            target_h,
            patch_size
        ));
    }
    let max_patches = max_patches as usize;
    if num_patches > max_patches {
        return Err(anyhow!(
            "resized image has {num_patches} patches, exceeds the {max_patches}-patch budget"
        ));
    }
    let ps = patch_size as usize;
    let row_len = 3usize
        .checked_mul(ps)
        .and_then(|value| value.checked_mul(ps))
        .ok_or_else(|| anyhow!("patch feature count overflow"))?;
    let prepared_floats = max_patches
        .checked_mul(row_len)
        .ok_or_else(|| anyhow!("prepared image allocation overflow"))?;
    if prepared_floats > MAX_PREPARED_FLOATS {
        return Err(anyhow!(
            "prepared image requires {prepared_floats} floats; limit is {MAX_PREPARED_FLOATS}"
        ));
    }
    let position_values = max_patches
        .checked_mul(2)
        .ok_or_else(|| anyhow!("position-id allocation overflow"))?;
    let mut pixel_values: Vec<f32> = Vec::with_capacity(prepared_floats);
    let mut position_ids: Vec<i64> = Vec::with_capacity(position_values);
    // HF row-major patch extraction with channel-last feature order.
    let raw = resized.as_raw(); // length = target_w * target_h * 3, layout (y, x, c)
    let stride_y = (target_w as usize) * 3;
    let inv255 = 1.0f32 / 255.0f32;
    for py in 0..patch_grid_h as usize {
        for px in 0..patch_grid_w as usize {
            let y0 = py * ps;
            let x0 = px * ps;
            for dy in 0..ps {
                let row_base = (y0 + dy) * stride_y + x0 * 3;
                for dx in 0..ps {
                    let px_base = row_base + dx * 3;
                    for c in 0..3 {
                        let v = raw[px_base + c];
                        pixel_values.push((v as f32) * inv255);
                    }
                }
            }
            position_ids.push(px as i64);
            position_ids.push(py as i64);
        }
    }
    // Pad patches with zeros and positions with the -1 sentinel.
    let pad_patches = max_patches - num_patches;
    pixel_values.extend(std::iter::repeat(0.0f32).take(pad_patches * row_len));
    for _ in 0..pad_patches {
        position_ids.push(-1);
        position_ids.push(-1);
    }
    let device = Device::Cpu;
    let pixel_values = Tensor::from_vec(pixel_values, (1, max_patches, row_len), &device)?;
    let pixel_position_ids = Tensor::from_vec(position_ids, (1, max_patches, 2), &device)?;
    Ok(PreparedImage {
        pixel_values,
        pixel_position_ids,
        patch_grid_w,
        patch_grid_h,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, Rgb, RgbImage};
    #[test]
    fn square_336_matches_hf_padded_patch_budget() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(336, 336, Rgb([128u8, 64u8, 32u8])));
        let prepared = prepare_image(
            &img,
            16,
            DEFAULT_POOLING_KERNEL_SIZE,
            DEFAULT_MAX_SOFT_TOKENS,
        )
        .expect("prepare_image");
        assert_eq!(prepared.patch_grid_w, 48, "pW");
        assert_eq!(prepared.patch_grid_h, 48, "pH");
        assert_eq!(
            prepared.pixel_values.dims(),
            &[1, DEFAULT_PADDED_PATCHES, 3 * 16 * 16],
            "pixel_values shape"
        );
        assert_eq!(
            prepared.pixel_position_ids.dims(),
            &[1, DEFAULT_PADDED_PATCHES, 2],
            "pixel_position_ids shape"
        );
        let pids = prepared
            .pixel_position_ids
            .flatten_all()
            .unwrap()
            .to_vec1::<i64>()
            .unwrap();
        assert_eq!(&pids[0..2], &[0, 0]);
        let last_valid = (48 * 48 - 1) * 2;
        assert_eq!(&pids[last_valid..last_valid + 2], &[47, 47]);
        let first_pad = 48 * 48 * 2;
        assert_eq!(&pids[first_pad..first_pad + 2], &[-1, -1]);
        assert_eq!(&pids[pids.len() - 2..], &[-1, -1]);
    }
    #[test]
    fn nonsquare_240x320_uses_area_preserving_scale() {
        let w: u32 = 240;
        let h: u32 = 320;
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, Rgb([255u8, 0, 0])));
        let prepared = prepare_image(
            &img,
            16,
            DEFAULT_POOLING_KERNEL_SIZE,
            DEFAULT_MAX_SOFT_TOKENS,
        )
        .expect("prepare_image");
        assert_eq!(prepared.patch_grid_w, 42, "pW");
        assert_eq!(prepared.patch_grid_h, 57, "pH");
        // Divisible by pooling_kernel_size in both dims (no pooling remainder).
        assert_eq!(prepared.patch_grid_w % 3, 0);
        assert_eq!(prepared.patch_grid_h % 3, 0);
        let dims = prepared.pixel_values.dims();
        assert_eq!(dims, &[1, DEFAULT_PADDED_PATCHES, 3 * 16 * 16]);
        let pv = prepared
            .pixel_values
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for &x in &pv {
            assert!(x.is_finite(), "non-finite pixel value");
            assert!((0.0..=1.0).contains(&x), "pixel out of [0,1]: {}", x);
        }
        // Solid red, channel-LAST layout: every pixel is (R=1,G=0,B=0), so the
        // first patch's 256 pixels are [1,0,0, 1,0,0, ...].
        for i in 0..(16 * 16) {
            let base = i * 3;
            assert!((pv[base] - 1.0).abs() < 1e-6, "R chan should be 1.0");
            assert!(pv[base + 1].abs() < 1e-6, "G chan should be 0.0");
            assert!(pv[base + 2].abs() < 1e-6, "B chan should be 0.0");
        }
        let pids = prepared
            .pixel_position_ids
            .flatten_all()
            .unwrap()
            .to_vec1::<i64>()
            .unwrap();
        for py in 0..57i64 {
            for px in 0..42i64 {
                let idx = ((py * 42 + px) * 2) as usize;
                assert_eq!(pids[idx], px, "px at patch ({},{})", px, py);
                assert_eq!(pids[idx + 1], py, "py at patch ({},{})", px, py);
            }
        }
    }
    #[test]
    fn nonsquare_681x336_matches_reference_geometry() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(681, 336, Rgb([10, 20, 30])));
        let prepared = prepare_image(
            &img,
            16,
            DEFAULT_POOLING_KERNEL_SIZE,
            DEFAULT_MAX_SOFT_TOKENS,
        )
        .expect("prepare_image");
        assert_eq!(prepared.patch_grid_w, 69, "pW");
        assert_eq!(prepared.patch_grid_h, 33, "pH");
        let valid_patches = (prepared.patch_grid_w * prepared.patch_grid_h) as usize;
        assert_eq!(valid_patches, 2277);
        assert_eq!(
            valid_patches / (DEFAULT_POOLING_KERNEL_SIZE as usize).pow(2),
            253
        );
    }
    #[test]
    fn rejects_unbounded_token_budget() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 32, Rgb([0, 0, 0])));
        assert!(prepare_image(
            &img,
            16,
            DEFAULT_POOLING_KERNEL_SIZE,
            MAX_VISION_SOFT_TOKENS + 1
        )
        .is_err());
    }
}
