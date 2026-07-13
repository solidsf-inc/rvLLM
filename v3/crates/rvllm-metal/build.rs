// Copyright 2026 m0at
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const HEADER_ONLY: &[&str] = &["common.metal", "float8.metal", "pagedattn_utils.metal"];
const REQUIRED_SYMBOLS: &[&str] = &[
    "fp8_perchannel_dequant_bfloat16_t",
    "dequant_fp8_blockwise_bfloat16_t",
    "paged_attention_bfloat16_t_cache_bfloat16_t_hs128_bs16_nt256_nsl32_ps0",
    "reshape_and_cache_kv_bfloat16_t_cache_bfloat16_t",
    "gather_kv_cache_cache_bfloat16_t_out_bfloat16_t",
    "softmax_with_sinks_bfloat",
    "sdpa_vector_with_sinks_bfloat16_t_128",
    "flash_attn_sinks_bfloat16_t_hd128_br8_bc32",
    "rvllm_fp8_gemv_bf16scale_f32",
];

fn main() -> Result<(), String> {
    println!("cargo:rerun-if-changed=build.rs");
    for name in [
        "CARGO_CFG_TARGET_OS",
        "CARGO_CFG_TARGET_ARCH",
        "CARGO_FEATURE_METAL",
        "RVLLM_EXPECTED_METAL_VERSION",
        "RVLLM_EXPECTED_MACOSX_SDK",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let metal_enabled = env::var_os("CARGO_FEATURE_METAL").is_some();
    if !metal_enabled || target_os != "macos" || target_arch != "aarch64" {
        return Ok(());
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").map_err(|_| "OUT_DIR not set".to_string())?);
    let metallib_path = out_dir.join("rvllm_kernels.metallib");
    println!(
        "cargo:rustc-env=RVLLM_METALLIB_PATH={}",
        metallib_path.display()
    );

    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").map_err(|_| "CARGO_MANIFEST_DIR not set".to_string())?,
    );
    let sources_dir = manifest_dir.join("src/metal_kernels");
    let all_sources = collect_metal_files(&sources_dir)?;
    for source in &all_sources {
        println!("cargo:rerun-if-changed={}", source.display());
    }
    let translation_units: Vec<_> = all_sources
        .into_iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| !HEADER_ONLY.contains(&name))
        })
        .collect();
    if translation_units.is_empty() {
        return Err("no Metal translation units found".into());
    }

    let metal_version = command_text(
        Command::new("xcrun").args(["-sdk", "macosx", "metal", "--version"]),
        "query Metal compiler version",
    )?;
    let sdk_version = command_text(
        Command::new("xcrun").args(["--sdk", "macosx", "--show-sdk-version"]),
        "query macOS SDK version",
    )?;
    check_expected("RVLLM_EXPECTED_METAL_VERSION", &metal_version)?;
    check_expected("RVLLM_EXPECTED_MACOSX_SDK", &sdk_version)?;
    println!("cargo:warning=rvLLM Metal compiler: {metal_version}");
    println!("cargo:warning=rvLLM macOS SDK: {sdk_version}");

    let mut air_files = Vec::with_capacity(translation_units.len());
    for source in &translation_units {
        let stem = source
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| format!("invalid Metal source name: {}", source.display()))?;
        let air_path = out_dir.join(format!("{stem}.air"));
        let mut command = Command::new("xcrun");
        command
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-std=metal3.1",
                "-Wall",
                "-Wextra",
                "-O3",
            ])
            .arg("-I")
            .arg(&sources_dir)
            .arg("-c")
            .arg(source)
            .arg("-o")
            .arg(&air_path);
        run(command, &format!("compile {}", source.display()))?;
        air_files.push(air_path);
    }

    let mut link = Command::new("xcrun");
    link.args(["-sdk", "macosx", "metallib", "-o"])
        .arg(&metallib_path);
    for air in &air_files {
        link.arg(air);
    }
    run(link, "link rvllm_kernels.metallib")?;

    let metadata = fs::metadata(&metallib_path)
        .map_err(|error| format!("inspect {}: {error}", metallib_path.display()))?;
    if metadata.len() == 0 {
        return Err(format!("empty metallib at {}", metallib_path.display()));
    }

    let mut nm = Command::new("xcrun");
    nm.args(["-sdk", "macosx", "metal-nm"]).arg(&metallib_path);
    let symbols = command_text(&mut nm, "inspect metallib symbols")?;
    for symbol in REQUIRED_SYMBOLS {
        if !symbols
            .lines()
            .any(|line| line.split_whitespace().last() == Some(symbol))
        {
            return Err(format!("metallib is missing required symbol `{symbol}`"));
        }
    }
    Ok(())
}

fn check_expected(variable: &str, actual: &str) -> Result<(), String> {
    if let Ok(expected) = env::var(variable) {
        if expected.trim() != actual.trim() {
            return Err(format!(
                "{variable} expected {:?}, found {:?}",
                expected.trim(),
                actual.trim()
            ));
        }
    }
    Ok(())
}

fn run(mut command: Command, description: &str) -> Result<Output, String> {
    let output = command
        .output()
        .map_err(|error| format!("{description}: failed to start: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{description}: {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(output)
}

fn command_text(command: &mut Command, description: &str) -> Result<String, String> {
    let output = command
        .output()
        .map_err(|error| format!("{description}: failed to start: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{description}: {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("{description}: non-UTF-8 output: {error}"))?;
    Ok(stdout.trim().to_string())
}

fn collect_metal_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).map_err(|error| format!("read {}: {error}", dir.display()))? {
        let path = entry
            .map_err(|error| format!("read entry in {}: {error}", dir.display()))?
            .path();
        if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("metal") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}
