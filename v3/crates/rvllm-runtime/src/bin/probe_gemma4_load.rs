//! Explicit GB10 hardware probe with a machine-readable receipt.
//!
//! Required inputs:
//! `RVLLM_MODEL_DIR`, `RVLLM_KERNELS_DIR`, `RVLLM_CUTLASS_SO`,
//! `RVLLM_FA3_SO`, `RVLLM_POLICY`, `RVLLM_ARENA_GB`, and
//! `RVLLM_PROBE_OUTPUT`. Optional generation additionally requires
//! `RVLLM_PROMPT_IDS`, `RVLLM_EOS_IDS`, and `RVLLM_MAX_NEW`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rvllm_core::{CompileTarget, ModelArch as HfModelArch, ModelConfig};
use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};
use serde_json::{json, Value};

const MAX_ARENA_GB: u64 = 256;
const MAX_PROMPT_TOKENS: usize = 4096;
const MAX_NEW_TOKENS: usize = 4096;

fn env_path(name: &str) -> Result<PathBuf, String> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing env var: {name}"))
}

fn existing_dir(name: &str) -> Result<PathBuf, String> {
    let path = env_path(name)?;
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("{name} {}: {error}", path.display()))?;
    if !canonical.is_dir() {
        return Err(format!(
            "{name} is not a directory: {}",
            canonical.display()
        ));
    }
    Ok(canonical)
}

fn existing_file(name: &str) -> Result<PathBuf, String> {
    let path = env_path(name)?;
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("{name} {}: {error}", path.display()))?;
    if !canonical.is_file() {
        return Err(format!("{name} is not a file: {}", canonical.display()));
    }
    Ok(canonical)
}

fn parse_bounded_usize(name: &str, maximum: usize) -> Result<usize, String> {
    let value = std::env::var(name).map_err(|_| format!("missing env var: {name}"))?;
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("{name}={value:?}: {error}"))?;
    if parsed == 0 || parsed > maximum {
        return Err(format!("{name} must be in 1..={maximum}"));
    }
    Ok(parsed)
}

fn parse_token_ids(name: &str, maximum_len: usize, vocab: usize) -> Result<Vec<u32>, String> {
    let value = std::env::var(name).map_err(|_| format!("missing env var: {name}"))?;
    let ids: Vec<u32> = value
        .split(',')
        .map(str::trim)
        .map(|part| {
            if part.is_empty() {
                return Err(format!("{name} contains an empty token id"));
            }
            part.parse::<u32>()
                .map_err(|error| format!("{name} token {part:?}: {error}"))
        })
        .collect::<Result<_, _>>()?;
    if ids.is_empty() || ids.len() > maximum_len {
        return Err(format!("{name} length must be in 1..={maximum_len}"));
    }
    if ids.iter().any(|id| *id as usize >= vocab) {
        return Err(format!(
            "{name} contains a token outside vocab size {vocab}"
        ));
    }
    Ok(ids)
}

fn write_receipt(path: &Path, receipt: &Value) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(format!(
            "receipt parent does not exist: {}",
            parent.display()
        ));
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create receipt {}: {error}", path.display()))?;
    serde_json::to_writer_pretty(&mut file, receipt)
        .map_err(|error| format!("write receipt {}: {error}", path.display()))?;
    use std::io::Write;
    file.write_all(b"\n")
        .map_err(|error| format!("finish receipt {}: {error}", path.display()))
}

fn run() -> Result<(PathBuf, Value), String> {
    let model_dir = existing_dir("RVLLM_MODEL_DIR")?;
    let kernels_dir = existing_dir("RVLLM_KERNELS_DIR")?;
    let cutlass_so = existing_file("RVLLM_CUTLASS_SO")?;
    let fa3_so = existing_file("RVLLM_FA3_SO")?;
    let policy_json = existing_file("RVLLM_POLICY")?;
    let receipt_path = env_path("RVLLM_PROBE_OUTPUT")?;
    if receipt_path.exists() {
        return Err(format!(
            "refusing to overwrite receipt {}",
            receipt_path.display()
        ));
    }
    let arena_gb = parse_bounded_usize("RVLLM_ARENA_GB", MAX_ARENA_GB as usize)? as u64;
    let arena_bytes = usize::try_from(arena_gb)
        .ok()
        .and_then(|value| value.checked_mul(1024 * 1024 * 1024))
        .ok_or_else(|| "RVLLM_ARENA_GB byte conversion overflow".to_string())?;

    let config =
        ModelConfig::load_hf(&model_dir).map_err(|error| format!("load model config: {error}"))?;
    if config.architecture != HfModelArch::Gemma4 {
        return Err(format!(
            "expected Gemma4 architecture, got {:?}",
            config.architecture
        ));
    }

    let paths = Gemma4EnginePaths {
        model_dir: model_dir.clone(),
        kernels_dir: kernels_dir.clone(),
        cutlass_so: cutlass_so.clone(),
        fa3_so: fa3_so.clone(),
        policy_json: policy_json.clone(),
        w4a8_so: std::env::var_os("RVLLM_W4A8_SO").map(PathBuf::from),
    };
    let started = std::time::Instant::now();
    let bringup = Gemma4Bringup::load(paths, arena_bytes)
        .map_err(|error| format!("Gemma4Bringup::load: {error}"))?;
    let bringup_seconds = started.elapsed().as_secs_f64();
    let (cc_major, cc_minor) = bringup.ctx.compute_capability();
    let target = CompileTarget::from_compute_capability(cc_major, cc_minor)
        .ok_or_else(|| format!("unsupported compute capability {cc_major}.{cc_minor}"))?;
    if target != CompileTarget::Sm121 {
        return Err(format!(
            "probe-gemma4-load requires Sm121, live target is {}",
            target.as_sm_str()
        ));
    }
    if bringup.arena.capacity() != arena_bytes {
        return Err(format!(
            "arena capacity {} does not match requested {arena_bytes}",
            bringup.arena.capacity()
        ));
    }

    let resolved = rvllm_kernels::Fp8GemvVariant::WprNativeF16In
        .load_verified(&bringup.kernels, target)
        .map_err(|error| format!("resolve WprNativeF16In: {error}"))?;
    let installed = bringup
        .fused
        .fn_fp8_gemv_wpr_native_f16in
        .as_ref()
        .ok_or_else(|| "bring-up did not retain WprNativeF16In".to_string())?;
    if resolved.raw() == 0 || installed.raw() == 0 {
        return Err("WprNativeF16In resolved to a null handle".into());
    }

    let mut generation = Value::Null;
    if std::env::var_os("RVLLM_PROMPT_IDS").is_some() {
        let prompt_ids = parse_token_ids("RVLLM_PROMPT_IDS", MAX_PROMPT_TOKENS, config.vocab_size)?;
        let eos_ids = parse_token_ids("RVLLM_EOS_IDS", 64, config.vocab_size)?;
        let max_new = parse_bounded_usize("RVLLM_MAX_NEW", MAX_NEW_TOKENS)?;
        let embedding_module = bringup
            .kernels
            .load_ptx("embedding_gather_f16")
            .map_err(|error| format!("load embedding_gather_f16: {error}"))?;
        let embedding = embedding_module
            .get_function("embedding_gather_f16_kernel")
            .map_err(|error| format!("resolve embedding_gather_f16_kernel: {error}"))?;
        let generation_started = std::time::Instant::now();
        let output_ids = unsafe {
            bringup
                .run_generate(
                    embedding,
                    bringup.fused.fn_argmax.clone(),
                    &prompt_ids,
                    max_new,
                    &eos_ids,
                    &[],
                )
                .map_err(|error| format!("run_generate: {error}"))?
        };
        let elapsed = generation_started.elapsed().as_secs_f64();
        let new_tokens = output_ids.len().saturating_sub(prompt_ids.len());
        generation = json!({
            "classification": "diagnostic_only",
            "prompt_ids": prompt_ids,
            "eos_ids": eos_ids,
            "max_new_tokens": max_new,
            "output_ids": output_ids,
            "new_tokens": new_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": if elapsed > 0.0 { Some(new_tokens as f64 / elapsed) } else { None },
        });
    }

    let receipt = json!({
        "schema": "rvllm.hardware_probe.v1",
        "result": "pass",
        "target": target.as_sm_str(),
        "compute_capability": [cc_major, cc_minor],
        "inputs": {
            "model_dir": model_dir,
            "kernels_dir": kernels_dir,
            "cutlass_so": cutlass_so,
            "fa3_so": fa3_so,
            "policy": policy_json,
            "arena_bytes": arena_bytes,
        },
        "bringup": {
            "elapsed_seconds": bringup_seconds,
            "arena_capacity": bringup.arena.capacity(),
            "arena_used": bringup.arena.used(),
            "fp8_gemv_module": bringup.fused.fp8_gemv_mod.path(),
            "wpr_native_f16in_symbol": rvllm_kernels::Fp8GemvVariant::WprNativeF16In.entry_point(),
        },
        "generation": generation,
    });
    Ok((receipt_path, receipt))
}

fn main() -> ExitCode {
    match run() {
        Ok((path, receipt)) => match write_receipt(&path, &receipt) {
            Ok(()) => {
                println!(
                    "{}",
                    serde_json::to_string(&receipt).expect("receipt serializes")
                );
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("probe failed: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("probe failed: {error}");
            ExitCode::FAILURE
        }
    }
}
