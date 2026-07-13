//! rvllm-bench: loads a model + kernels + cutlass .so + fa3 .so, runs
//! `iters` decode-step forwards on a fixed-batch bucket, and reports
//! tokens/sec.
//! Prints one JSON line per run: {batch, iters, tok_per_sec, ms_per_step}.
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use rvllm_core::{ModelArch as HfModelArch, ModelConfig};
use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};
use rvllm_runtime::{Bringup, EnginePaths};
use sha2::{Digest, Sha256};

const MAX_PROMPT_FILE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_METADATA_TEXT: usize = 256;

fn env_path(k: &str) -> Result<PathBuf, String> {
    let value = std::env::var_os(k).ok_or_else(|| format!("missing env var: {k}"))?;
    if value.is_empty() {
        return Err(format!("{k} must not be empty"));
    }
    Ok(PathBuf::from(value))
}

fn env_path_or_placeholder(k: &str) -> PathBuf {
    std::env::var_os(k)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("<unset:{k}>")))
}

fn env_u32(k: &str, default: u32) -> Result<u32, String> {
    match std::env::var(k) {
        Ok(value) => value
            .parse::<u32>()
            .map_err(|_| format!("{k} must be a non-negative integer")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("{k}: {error}")),
    }
}

fn env_f64(k: &str, default: f64) -> Result<f64, String> {
    let value = match std::env::var(k) {
        Ok(value) => value
            .parse::<f64>()
            .map_err(|_| format!("{k} must be a number"))?,
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => return Err(format!("{k}: {error}")),
    };
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{k} must be finite and >= 0"));
    }
    Ok(value)
}

fn receipt_metadata(with_model: bool) -> Result<serde_json::Value, String> {
    let mut metadata = serde_json::json!({
        "schema": "rvllm.local-benchmark.v1",
        "source_sha": required_hex("RVLLM_SOURCE_SHA", 40)?,
        "hardware": required_text("RVLLM_HARDWARE")?,
        "driver": required_text("RVLLM_DRIVER")?,
        "toolchain": required_text("RVLLM_TOOLCHAIN")?,
        "backend": "cuda",
        "command": command(),
        "unix_time_seconds": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| format!("system clock: {error}"))?
            .as_secs(),
    });
    let object = metadata.as_object_mut().unwrap();
    if with_model {
        object.insert("model".into(), required_text("RVLLM_MODEL_ID")?.into());
        object.insert(
            "model_sha256".into(),
            required_hex("RVLLM_MODEL_SHA256", 64)?.into(),
        );
    } else {
        object.insert("model".into(), "synthetic".into());
    }
    Ok(metadata)
}

fn artifact_digests(
    paths: &EnginePaths,
    w4a8_so: Option<&std::path::Path>,
) -> Result<serde_json::Value, String> {
    let manifest_path = paths
        .kernels_dir
        .join(required_text("RVLLM_KERNEL_ARCH")?)
        .join("manifest.json");
    let mut artifacts = serde_json::json!({
        "kernel_manifest_sha256": sha256_file(&manifest_path)?,
    });
    let object = artifacts.as_object_mut().unwrap();
    for (name, path) in [
        ("cutlass_sha256", paths.cutlass_so.as_path()),
        ("flash_attention_sha256", paths.fa3_so.as_path()),
        ("policy_sha256", paths.policy_json.as_path()),
    ] {
        if path.is_file() {
            object.insert(name.into(), sha256_file(path)?.into());
        }
    }
    for (name, variable) in [
        ("attention_fallback_sha256", "RVLLM_FA_FALLBACK_SO"),
        ("cutlass_sm120_sha256", "RVLLM_CUTLASS_SM120_SO"),
    ] {
        if let Some(path) = std::env::var_os(variable).map(PathBuf::from) {
            if path.is_file() {
                object.insert(name.into(), sha256_file(&path)?.into());
            }
        }
    }
    if let Some(path) = w4a8_so {
        object.insert(
            "w4a8_sha256".into(),
            serde_json::Value::String(sha256_file(path)?),
        );
    }
    Ok(artifacts)
}

fn finalize_receipt(
    mut result: serde_json::Value,
    metadata: &serde_json::Value,
    artifacts: &serde_json::Value,
) -> Result<String, String> {
    {
        let object = result
            .as_object_mut()
            .ok_or("benchmark result must be a JSON object")?;
        for (key, value) in metadata
            .as_object()
            .ok_or("receipt metadata must be a JSON object")?
        {
            object.insert(key.clone(), value.clone());
        }
        object.insert("artifacts".into(), artifacts.clone());
    }
    let canonical =
        serde_json::to_vec(&result).map_err(|error| format!("receipt JSON: {error}"))?;
    result.as_object_mut().unwrap().insert(
        "receipt_sha256".into(),
        serde_json::Value::String(sha256_bytes(&canonical)),
    );
    serde_json::to_string(&result).map_err(|error| format!("receipt JSON: {error}"))
}

fn required_text(name: &str) -> Result<String, String> {
    let value = std::env::var(name).map_err(|_| format!("missing env var: {name}"))?;
    if value.is_empty()
        || value.len() > MAX_METADATA_TEXT
        || value.trim() != value
        || value.chars().any(|character| character.is_control())
    {
        return Err(format!(
            "{name} must be 1..={MAX_METADATA_TEXT} characters without controls or outer whitespace"
        ));
    }
    Ok(value)
}

fn required_hex(name: &str, length: usize) -> Result<String, String> {
    let value = required_text(name)?;
    if value.len() != length
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "{name} must be {length} lowercase hexadecimal characters"
        ));
    }
    Ok(value)
}

fn command() -> Vec<String> {
    let mut arguments = std::env::args_os();
    let _ = arguments.next();
    let binary = std::env::current_exe()
        .ok()
        .and_then(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "rvllm-bench".into());
    std::iter::once(binary)
        .chain(arguments.map(|argument| argument.to_string_lossy().into_owned()))
        .collect()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn sha256_file(path: &std::path::Path) -> Result<String, String> {
    let mut file =
        std::fs::File::open(path).map_err(|error| format!("hash {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("hash {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Mode {
    Bench,
    Generate,
    Fp8Gemv,
    Help,
}

fn requested_mode() -> Result<Mode, String> {
    let mut selected = None;
    for argument in std::env::args_os().skip(1) {
        let argument = argument
            .to_str()
            .ok_or("command-line arguments must be valid UTF-8")?;
        let mode = match argument {
            "--generate" => Mode::Generate,
            "--fp8-gemv" => Mode::Fp8Gemv,
            "--help" | "-h" => Mode::Help,
            _ => return Err(format!("unknown argument: {argument}")),
        };
        if selected.replace(mode).is_some() {
            return Err("select at most one benchmark mode".into());
        }
    }

    let cli = selected.unwrap_or(Mode::Bench);
    let env_generate = std::env::var("RVLLM_BENCH_GENERATE").ok().as_deref() == Some("1");
    let env_fp8 = std::env::var("RVLLM_FP8_GEMV").ok().as_deref() == Some("1");
    if env_generate && env_fp8 {
        return Err("RVLLM_BENCH_GENERATE and RVLLM_FP8_GEMV are mutually exclusive".into());
    }
    let from_env = if env_generate {
        Some(Mode::Generate)
    } else if env_fp8 {
        Some(Mode::Fp8Gemv)
    } else {
        None
    };
    match (cli, from_env) {
        (Mode::Bench, Some(mode)) => Ok(mode),
        (Mode::Bench, None) | (Mode::Help, None) => Ok(cli),
        (mode, None) => Ok(mode),
        (mode, Some(env_mode)) if mode == env_mode => Ok(mode),
        _ => Err("command-line and environment benchmark modes conflict".into()),
    }
}

fn is_gemma4_model_dir(model_dir: &std::path::Path) -> Result<bool, String> {
    Ok(matches!(
        ModelConfig::load_hf(model_dir)
            .map_err(|e| format!("config parse {}: {e}", model_dir.display()))?
            .architecture,
        HfModelArch::Gemma4
    ))
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let result = run();
    if let Err(e) = result {
        eprintln!("rvllm-bench: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mode = requested_mode()?;
    if mode == Mode::Help {
        println!("usage: rvllm-bench [--generate | --fp8-gemv]");
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    if mode == Mode::Fp8Gemv {
        return run_fp8_gemv_bench();
    }

    let paths = EnginePaths {
        model_dir: env_path("RVLLM_MODEL_DIR")?,
        kernels_dir: env_path("RVLLM_KERNELS_DIR")?,
        cutlass_so: env_path_or_placeholder("RVLLM_CUTLASS_SO"),
        fa3_so: env_path_or_placeholder("RVLLM_FA3_SO"),
        policy_json: env_path_or_placeholder("RVLLM_POLICY"),
    };
    let w4a8_so = std::env::var_os("RVLLM_W4A8_SO").map(std::path::PathBuf::from);
    let metadata = receipt_metadata(true)?;
    let artifacts = artifact_digests(&paths, w4a8_so.as_deref())?;
    let batch = env_u32("RVLLM_BATCH", 128)?;
    let iters = env_u32("RVLLM_ITERS", 100)?;
    let warmup = env_u32("RVLLM_WARMUP", 10)?;
    if !(1..=65_536).contains(&batch) {
        return Err("RVLLM_BATCH must be in 1..=65536".into());
    }
    if !(1..=1_000_000).contains(&iters) || warmup > 1_000_000 {
        return Err("RVLLM_ITERS must be in 1..=1000000 and RVLLM_WARMUP <= 1000000".into());
    }
    let generate_mode = mode == Mode::Generate;

    let arena_gb = env_u32("RVLLM_ARENA_GB", 32)? as usize;
    if !(1..=256).contains(&arena_gb) {
        return Err("RVLLM_ARENA_GB must be in 1..=256".into());
    }
    let arena_bytes = arena_gb
        .checked_mul(1usize << 30)
        .ok_or("RVLLM_ARENA_GB byte count overflow")?;

    eprintln!("== rvllm-bench v3 ==");
    eprintln!("batch       = {batch}");
    eprintln!("iters       = {iters} (warmup {warmup})");

    let is_gemma4 = is_gemma4_model_dir(&paths.model_dir)?;

    if is_gemma4 {
        eprintln!("== Gemma 4 detected, using Gemma4Bringup ==");
        let g4_paths = Gemma4EnginePaths {
            model_dir: paths.model_dir,
            kernels_dir: paths.kernels_dir,
            cutlass_so: paths.cutlass_so,
            fa3_so: paths.fa3_so,
            policy_json: paths.policy_json,
            w4a8_so,
        };
        let t0 = Instant::now();
        let g4 = Gemma4Bringup::load(g4_paths, arena_bytes)
            .map_err(|e| format!("gemma4 bringup: {e}"))?;
        if g4.is_e4b_engine() {
            return Err("rvllm-bench does not yet produce verified E4B receipts".into());
        }
        eprintln!(
            "bringup: {:.2}s | arch layers={} hidden={} heads={} sliding_kv={} global_kv={}",
            t0.elapsed().as_secs_f64(),
            g4.arch.num_hidden_layers,
            g4.arch.hidden_size,
            g4.arch.num_attention_heads,
            g4.arch.num_kv_heads_sliding,
            g4.arch.num_kv_heads_global,
        );
        eprintln!("arena used = {} MiB", g4.arena.used() / (1024 * 1024));
        if generate_mode {
            return run_generate_bench(&g4, &metadata, &artifacts, arena_bytes);
        }
        let result = unsafe { g4.run_bench(batch, iters, warmup) }
            .map_err(|error| format!("run_bench: {error}"))?;
        print_result(
            result,
            warmup,
            arena_bytes,
            Some(g4.arch.sliding_window_size as u32),
            &metadata,
            &artifacts,
        )?;
        return Ok(());
    }

    let t0 = Instant::now();
    let br = Bringup::load(paths, arena_bytes).map_err(|e| format!("bringup: {e}"))?;
    eprintln!(
        "bringup: {:.2}s | arch layers={} hidden={} heads={} kv_heads={}",
        t0.elapsed().as_secs_f64(),
        br.arch.num_hidden_layers,
        br.arch.hidden_size,
        br.arch.num_attention_heads,
        br.arch.num_key_value_heads,
    );
    eprintln!("arena used = {} MiB", br.arena.used() / (1024 * 1024));

    if std::env::var("RVLLM_SWEEP").ok().as_deref() == Some("1") {
        return run_sweep(
            &br,
            batch,
            iters,
            warmup,
            arena_bytes,
            &metadata,
            &artifacts,
        );
    }

    let result =
        unsafe { br.run_bench(batch, iters, warmup) }.map_err(|e| format!("run_bench: {e}"))?;
    print_result(result, warmup, arena_bytes, None, &metadata, &artifacts)?;
    Ok(())
}

// FP8 E4M3FN -> f32 host decoder, including subnormals and signed zero.
#[cfg(feature = "cuda")]
fn fp8_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 == 0 { 1.0 } else { -1.0 };
    let e = ((b >> 3) & 0xF) as u32;
    let m = (b & 0x7) as u32;
    match (e, m) {
        (0, 0) => f32::from_bits((b as u32 & 0x80) << 24),
        (0, _) => sign * m as f32 * 2.0_f32.powi(-9),
        (0xF, 0x7) => f32::NAN,
        _ => sign * (1.0 + m as f32 / 8.0) * 2.0_f32.powi(e as i32 - 7),
    }
}

// Deterministic finite FP8 E4M3FN bytes spanning zeros, subnormals, and normals.
#[cfg(feature = "cuda")]
fn gen_fp8(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut byte = (s >> 32) as u8;
        if byte & 0x7f == 0x7f {
            byte ^= 1;
        }
        v.push(byte);
    }
    v
}

#[cfg(feature = "cuda")]
fn f32_as_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|value| value.to_le_bytes()).collect()
}

#[cfg(feature = "cuda")]
fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

#[cfg(feature = "cuda")]
fn max_relative_error(actual: &[f32], expected: &[f32]) -> Result<f64, String> {
    if actual.len() != expected.len() {
        return Err("parity vectors have different lengths".into());
    }
    actual
        .iter()
        .zip(expected)
        .try_fold(0.0_f64, |maximum, (&actual, &expected)| {
            if !actual.is_finite() || !expected.is_finite() {
                return Err("parity vectors contain non-finite values".into());
            }
            let difference = (actual as f64 - expected as f64).abs();
            Ok(maximum.max(difference / (expected as f64).abs().max(1e-3)))
        })
}

#[cfg(feature = "cuda")]
fn parity_rows(n: usize, count: usize, seed: u64) -> Vec<usize> {
    if n == 0 || count == 0 {
        return Vec::new();
    }
    let target = count.min(n);
    let mut rows = Vec::with_capacity(target);
    rows.push(0);
    if target > 1 && n > 1 {
        rows.push(n - 1);
    }
    let mut state = seed | 1;
    while rows.len() < target {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let row = state.wrapping_mul(0x2545_F491_4F6C_DD1D) as usize % n;
        if !rows.contains(&row) {
            rows.push(row);
        }
    }
    rows.sort_unstable();
    rows
}

// Standalone FP8 E4M3FN M=1 GEMV comparison with full-output parity checks.
// Needs the kernel bundle root and RVLLM_KERNEL_ARCH; it does not load a model.
#[cfg(feature = "cuda")]
fn run_fp8_gemv_bench() -> Result<(), String> {
    use rvllm_cutlass::CublasLt;
    use rvllm_fused::launch_raw;
    use rvllm_kernels::{manifest::KernelManifest, KernelLoader};
    use rvllm_mem::context::CudaContextHandle;
    use rvllm_mem::stream::Stream;
    use rvllm_mem::HbmArena;

    let kernels_dir = env_path("RVLLM_KERNELS_DIR")?.join(required_text("RVLLM_KERNEL_ARCH")?);
    let metadata = receipt_metadata(false)?;
    let artifacts = serde_json::json!({
        "kernel_manifest_sha256": sha256_file(&kernels_dir.join("manifest.json"))?,
    });
    let iters = env_u32("RVLLM_ITERS", 300)? as usize;
    let warmup = env_u32("RVLLM_WARMUP", 50)? as usize;
    if !(1..=1_000_000).contains(&iters) || warmup > 1_000_000 {
        return Err("RVLLM_ITERS must be in 1..=1000000 and RVLLM_WARMUP <= 1000000".into());
    }

    let ctx = CudaContextHandle::init(0).map_err(|e| e.to_string())?;
    let (cc_major, cc_minor) = ctx.compute_capability();
    if cc_major < 8 || (cc_major == 8 && cc_minor < 9) {
        return Err(format!(
            "FP8 E4M3 GEMV kernels require compute capability 8.9 or newer; got {cc_major}.{cc_minor}"
        ));
    }
    let arena = HbmArena::new(&ctx, 12_usize << 30).map_err(|e| e.to_string())?;
    let stream = Stream::new(&ctx).map_err(|e| e.to_string())?;
    let sraw = stream.raw();

    let manifest = KernelManifest::load_and_verify(&kernels_dir.join("manifest.json"))
        .map_err(|e| e.to_string())?;
    let loader = KernelLoader::new(manifest, &ctx);
    let module = loader
        .load_ptx("fp8_e4m3_gemv")
        .map_err(|e| e.to_string())?;
    let gemv_fn = module
        .get_function("fp8_e4m3_gemv_kernel")
        .map_err(|e| e.to_string())?;
    // K-tiled channel-scale GEMV bounds shared memory to one
    // K-tile so occupancy holds at large K (down-proj K=21504). TILE_K is runtime.
    let module_kt = loader
        .load_ptx("fp8_channelscale_gemv_ktiled")
        .map_err(|e| e.to_string())?;
    let ktiled_fn = module_kt
        .get_function("fp8_channelscale_gemv_ktiled_kernel")
        .map_err(|e| e.to_string())?;
    let tile_k = env_u32("RVLLM_TILE_K", 8192)? as i32;
    if !(1..=65_536).contains(&tile_k) {
        return Err("RVLLM_TILE_K must be in 1..=65536".into());
    }
    // Split-K channelscale GEMV: SPLIT blocks cooperate per output row (atomicAdd
    // partials) -> fills SMs on the small-N large-K shapes (down, o_global).
    let module_sk = loader
        .load_ptx("fp8_channelscale_gemv_splitk")
        .map_err(|e| e.to_string())?;
    let splitk_fn = module_sk
        .get_function("fp8_channelscale_gemv_splitk_kernel")
        .map_err(|e| e.to_string())?;
    let split_k = env_u32("RVLLM_SPLITK", 8)? as i32;
    if !(1..=1024).contains(&split_k) {
        return Err("RVLLM_SPLITK must be in 1..=1024".into());
    }
    let max_relative_error_limit = env_f64("RVLLM_MAX_REL_ERROR", 0.05)?;

    let ws_bytes = 32_usize << 20;
    let ws = arena
        .region("cublaslt_ws", ws_bytes, 256)
        .map_err(|e| e.to_string())?;
    let cublaslt = CublasLt::new(ws.device_ptr(), ws_bytes).map_err(|e| e.to_string())?;

    // Representative Gemma 4 31B FP8 projection dimensions.
    // qkv/gate_up are large-N small-K; o(global)/down are small-N LARGE-K (the
    // shapes where the full-K-smem kernel collapses and K-tiling must recover).
    let shapes: &[(&str, usize, usize)] = &[
        ("qkv_global", 20480, 5376),
        ("o_proj_global", 5376, 16384),
        ("gate_up", 43008, 5376),
        ("down", 5376, 21504),
    ];
    eprintln!("== synthetic FP8 GEMV (iters={iters} warmup={warmup}) ==");

    for &(name, n, k) in shapes {
        let w_bytes = n.checked_mul(k).ok_or("weight byte count overflow")?;
        let out_bytes = n.checked_mul(4).ok_or("output byte count overflow")?;
        let w = arena
            .region("w", w_bytes.next_multiple_of(256), 256)
            .map_err(|e| e.to_string())?;
        let a = arena
            .region("a", k.next_multiple_of(256), 256)
            .map_err(|e| e.to_string())?;
        let wscale = arena
            .region("wscale", (n * 4).next_multiple_of(256), 256)
            .map_err(|e| e.to_string())?;
        let ascale = arena
            .region("ascale", 256, 256)
            .map_err(|e| e.to_string())?;
        let bscalar = arena
            .region("bscalar", 256, 256)
            .map_err(|e| e.to_string())?;
        let out_baseline = arena
            .region("out_baseline", out_bytes.next_multiple_of(256), 256)
            .map_err(|e| e.to_string())?;
        let out_kt = arena
            .region("out_kt", out_bytes.next_multiple_of(256), 256)
            .map_err(|e| e.to_string())?;
        let out_sk = arena
            .region("out_sk", out_bytes.next_multiple_of(256), 256)
            .map_err(|e| e.to_string())?;
        let out_ref = arena
            .region("out_ref", out_bytes.next_multiple_of(256), 256)
            .map_err(|e| e.to_string())?;
        let out_ref_f32 = arena
            .region("out_ref_f32", out_bytes.next_multiple_of(256), 256)
            .map_err(|e| e.to_string())?;

        let w_host = gen_fp8(w_bytes, 0x1234_5678 ^ (n as u64).wrapping_mul(2654435761));
        let a_host = gen_fp8(k, 0x9E37_79B9 ^ (k as u64).wrapping_mul(40503));
        let wscale_host = vec![0.02_f32; n];
        let ascale_host = [0.05_f32];
        unsafe {
            w.copy_from_host(&w_host).map_err(|e| e.to_string())?;
            a.copy_from_host(&a_host).map_err(|e| e.to_string())?;
            wscale
                .copy_from_host(&f32_as_bytes(&wscale_host))
                .map_err(|e| e.to_string())?;
            ascale
                .copy_from_host(&f32_as_bytes(&ascale_host))
                .map_err(|e| e.to_string())?;
            bscalar
                .copy_from_host(&f32_as_bytes(&[0.02_f32]))
                .map_err(|e| e.to_string())?;
        }

        let warps_per_block = 8_u32;
        let block = (warps_per_block * 32, 1_u32, 1_u32);
        let grid = ((n as u32).div_ceil(warps_per_block), 1_u32, 1_u32);

        let launch_baseline = || -> Result<(), String> {
            let mut p0 = out_baseline.device_ptr();
            let mut p1 = w.device_ptr();
            let mut p2 = wscale.device_ptr();
            let mut p3 = a.device_ptr();
            let mut p4 = ascale.device_ptr();
            let mut ni = n as i32;
            let mut ki = k as i32;
            let args = [
                (&mut p0) as *mut u64 as *mut core::ffi::c_void,
                (&mut p1) as *mut u64 as *mut core::ffi::c_void,
                (&mut p2) as *mut u64 as *mut core::ffi::c_void,
                (&mut p3) as *mut u64 as *mut core::ffi::c_void,
                (&mut p4) as *mut u64 as *mut core::ffi::c_void,
                (&mut ni) as *mut i32 as *mut core::ffi::c_void,
                (&mut ki) as *mut i32 as *mut core::ffi::c_void,
            ];
            unsafe {
                launch_raw(&gemv_fn, grid, block, (k as u32) * 2, sraw, &args)
                    .map_err(|e| e.to_string())
            }
        };

        // K-tiled channelscale GEMV: same ABI + a trailing TILE_K arg; smem bounded
        // to one tile (TILE_K halfs) instead of the full K -> occupancy holds at K=21504.
        let launch_ktiled = || -> Result<(), String> {
            let mut p0 = out_kt.device_ptr();
            let mut p1 = w.device_ptr();
            let mut p2 = wscale.device_ptr();
            let mut p3 = a.device_ptr();
            let mut p4 = ascale.device_ptr();
            let mut ni = n as i32;
            let mut ki = k as i32;
            let mut tki = tile_k;
            let args = [
                (&mut p0) as *mut u64 as *mut core::ffi::c_void,
                (&mut p1) as *mut u64 as *mut core::ffi::c_void,
                (&mut p2) as *mut u64 as *mut core::ffi::c_void,
                (&mut p3) as *mut u64 as *mut core::ffi::c_void,
                (&mut p4) as *mut u64 as *mut core::ffi::c_void,
                (&mut ni) as *mut i32 as *mut core::ffi::c_void,
                (&mut ki) as *mut i32 as *mut core::ffi::c_void,
                (&mut tki) as *mut i32 as *mut core::ffi::c_void,
            ];
            let smem = (tile_k.min(k as i32) as u32) * 2;
            unsafe {
                launch_raw(&ktiled_fn, grid, block, smem, sraw, &args).map_err(|e| e.to_string())
            }
        };

        // Split-K: grid.y = SPLIT cooperating blocks per row; out must be zeroed
        // (atomicAdd accumulation), then each block adds its scaled K-slice partial.
        let grid_sk = ((n as u32).div_ceil(warps_per_block), split_k as u32, 1_u32);
        let launch_splitk = || -> Result<(), String> {
            unsafe {
                let status = cudarc::driver::sys::cuMemsetD8Async(
                    out_sk.device_ptr(),
                    0u8,
                    out_bytes,
                    sraw as cudarc::driver::sys::CUstream,
                );
                if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("cuMemsetD8Async: {status:?}"));
                }
            }
            let mut p0 = out_sk.device_ptr();
            let mut p1 = w.device_ptr();
            let mut p2 = wscale.device_ptr();
            let mut p3 = a.device_ptr();
            let mut p4 = ascale.device_ptr();
            let mut ni = n as i32;
            let mut ki = k as i32;
            let mut spi = split_k;
            let args = [
                (&mut p0) as *mut u64 as *mut core::ffi::c_void,
                (&mut p1) as *mut u64 as *mut core::ffi::c_void,
                (&mut p2) as *mut u64 as *mut core::ffi::c_void,
                (&mut p3) as *mut u64 as *mut core::ffi::c_void,
                (&mut p4) as *mut u64 as *mut core::ffi::c_void,
                (&mut ni) as *mut i32 as *mut core::ffi::c_void,
                (&mut ki) as *mut i32 as *mut core::ffi::c_void,
                (&mut spi) as *mut i32 as *mut core::ffi::c_void,
            ];
            unsafe {
                launch_raw(&splitk_fn, grid_sk, block, 0, sraw, &args).map_err(|e| e.to_string())
            }
        };

        // Correct M=1 cuBLASLt FP8: scalar (tensorwise) scales + bf16 out. The
        // OUTER_VEC channelscale path + f32 out fails the FP8 (M*Csize)%16 rule
        // at M=1 (per cuBLAS docs); the real model uses scalar+bf16 too.
        let launch_ref = || -> Result<(), String> {
            unsafe {
                cublaslt
                    .fp8_gemm_bf16(
                        a.device_ptr(),
                        w.device_ptr(),
                        out_ref.device_ptr(),
                        1,
                        n as i32,
                        k as i32,
                        ascale.device_ptr(),
                        bscalar.device_ptr(),
                        sraw,
                    )
                    .map_err(|e| e.to_string())
            }
        };

        // Reference cuBLASLt route with f32 output.
        let launch_ref_f32 = || -> Result<(), String> {
            unsafe {
                cublaslt
                    .fp8_gemm_f32(
                        a.device_ptr(),
                        w.device_ptr(),
                        out_ref_f32.device_ptr(),
                        1,
                        n as i32,
                        k as i32,
                        ascale.device_ptr(),
                        bscalar.device_ptr(),
                        sraw,
                    )
                    .map_err(|e| e.to_string())
            }
        };

        let time_it = |f: &dyn Fn() -> Result<(), String>| -> Result<f64, String> {
            for _ in 0..warmup {
                f()?;
            }
            stream.fence().map_err(|e| e.to_string())?;
            let t0 = Instant::now();
            for _ in 0..iters {
                f()?;
            }
            stream.fence().map_err(|e| e.to_string())?;
            let seconds = t0.elapsed().as_secs_f64();
            if !seconds.is_finite() || seconds <= 0.0 {
                return Err("benchmark timer returned a non-positive duration".into());
            }
            Ok(seconds)
        };

        let t_baseline = time_it(&launch_baseline)?;
        let t_kt = time_it(&launch_ktiled)?;
        let t_sk = time_it(&launch_splitk)?;
        let t_ref = time_it(&launch_ref)?;
        let t_ref_f32 = time_it(&launch_ref_f32)?;

        let bytes = w_bytes as f64 * iters as f64;
        let bw_baseline = bytes / t_baseline / 1e12;
        let bw_kt = bytes / t_kt / 1e12;
        let bw_sk = bytes / t_sk / 1e12;
        let bw_ref = bytes / t_ref / 1e12;

        let mut h_baseline = vec![0_f32; n];
        let mut h_kt = vec![0_f32; n];
        let mut h_sk = vec![0_f32; n];
        let mut h_ref_bf16 = vec![0_u16; n];
        let mut h_ref_f32 = vec![0_f32; n];
        unsafe {
            let statuses = [
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    h_baseline.as_mut_ptr() as *mut _,
                    out_baseline.device_ptr(),
                    out_bytes,
                ),
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    h_kt.as_mut_ptr() as *mut _,
                    out_kt.device_ptr(),
                    out_bytes,
                ),
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    h_sk.as_mut_ptr() as *mut _,
                    out_sk.device_ptr(),
                    out_bytes,
                ),
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    h_ref_bf16.as_mut_ptr() as *mut _,
                    out_ref.device_ptr(),
                    n * core::mem::size_of::<u16>(),
                ),
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    h_ref_f32.as_mut_ptr() as *mut _,
                    out_ref_f32.device_ptr(),
                    out_bytes,
                ),
            ];
            if let Some(status) = statuses
                .into_iter()
                .find(|status| *status != cudarc::driver::sys::CUresult::CUDA_SUCCESS)
            {
                return Err(format!("cuMemcpyDtoH_v2: {status:?}"));
            }
        }

        // Compare every custom output row with cuBLASLt f32, validate the BF16
        // cuBLASLt route, then independently recompute deterministic rows on CPU.
        let max_rel = max_relative_error(&h_baseline, &h_ref_f32)?;
        let max_rel_kt = max_relative_error(&h_kt, &h_ref_f32)?;
        let max_rel_sk = max_relative_error(&h_sk, &h_ref_f32)?;
        let h_ref_bf16_f32: Vec<f32> = h_ref_bf16.into_iter().map(bf16_to_f32).collect();
        let max_rel_ref_bf16 = max_relative_error(&h_ref_bf16_f32, &h_ref_f32)?;

        let a_f: Vec<f32> = a_host.iter().map(|&b| fp8_to_f32(b)).collect();
        let mut max_rel_host_ref = 0_f64;
        for row in parity_rows(n, 64, (n as u64) ^ ((k as u64) << 32)) {
            let base = row * k;
            let mut acc = 0_f64;
            for kk in 0..k {
                acc += fp8_to_f32(w_host[base + kk]) as f64 * a_f[kk] as f64;
            }
            let refv = acc * wscale_host[row] as f64 * ascale_host[0] as f64;
            let difference = (h_ref_f32[row] as f64 - refv).abs();
            max_rel_host_ref = max_rel_host_ref.max(difference / refv.abs().max(1e-3));
        }

        let bw_ref_f32 = bytes / t_ref_f32 / 1e12;

        for (variant, error) in [
            ("baseline", max_rel),
            ("ktiled", max_rel_kt),
            ("splitk", max_rel_sk),
            ("cublaslt_bf16", max_rel_ref_bf16),
            ("cublaslt_host_reference", max_rel_host_ref),
        ] {
            if !error.is_finite() || error > max_relative_error_limit {
                return Err(format!(
                    "{name} {variant} relative error {error:.3e} exceeds {max_relative_error_limit:.3e}"
                ));
            }
        }

        let result = serde_json::json!({
            "mode": "fp8_gemv",
            "input": "synthetic",
            "shape": name,
            "n": n,
            "k": k,
            "weight_bytes": w_bytes,
            "tile_k": tile_k,
            "split_k": split_k,
            "baseline_weight_gb_per_s": bw_baseline * 1000.0,
            "ktiled_weight_gb_per_s": bw_kt * 1000.0,
            "splitk_weight_gb_per_s": bw_sk * 1000.0,
            "cublaslt_bf16_weight_gb_per_s": bw_ref * 1000.0,
            "cublaslt_f32_weight_gb_per_s": bw_ref_f32 * 1000.0,
            "baseline_max_relative_error": max_rel,
            "cublaslt_bf16_max_relative_error": max_rel_ref_bf16,
            "cublaslt_host_reference_max_relative_error": max_rel_host_ref,
            "ktiled_us": t_kt / iters as f64 * 1e6,
            "splitk_us": t_sk / iters as f64 * 1e6,
            "cublaslt_f32_us": t_ref_f32 / iters as f64 * 1e6,
            "ktiled_max_relative_error": max_rel_kt,
            "splitk_max_relative_error": max_rel_sk,
                "max_relative_error_limit": max_relative_error_limit,
            "iters": iters,
            "warmup": warmup,
        });
        println!("{}", finalize_receipt(result, &metadata, &artifacts)?);
    }
    Ok(())
}

/// SHA-256 over the little-endian token-id stream.
fn sha256_tokens(ids: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for &id in ids {
        hasher.update(id.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// `--generate` / RVLLM_BENCH_GENERATE=1: drive `run_generate` on a fixed
/// synthetic prompt for N decode tokens (greedy, empty stop set) and report
/// decode tok/s + the output-token SHA-256 for eager-vs-graph parity.
#[cfg(feature = "cuda")]
fn run_generate_bench(
    g4: &rvllm_runtime::gemma4_bring_up::Gemma4Bringup,
    metadata: &serde_json::Value,
    artifacts: &serde_json::Value,
    arena_bytes: usize,
) -> Result<(), String> {
    let gen_tokens = env_u32("RVLLM_GEN_TOKENS", 200)? as usize;
    let prompt_len = env_u32("RVLLM_GEN_PROMPT", 32)? as usize;
    if gen_tokens == 0 || prompt_len == 0 {
        return Err("RVLLM_GEN_TOKENS and RVLLM_GEN_PROMPT must be greater than zero".into());
    }
    let max_context = g4.arch.max_position_embeddings;
    let max_prompt_tokens = max_context
        .checked_sub(gen_tokens)
        .ok_or("RVLLM_GEN_TOKENS exceeds the model context")?;
    if max_prompt_tokens == 0 {
        return Err("generation length leaves no room for a prompt".into());
    }
    let vocab = g4.arch.vocab_size as u32;
    if vocab <= 100 {
        return Err("model vocabulary is too small for the synthetic prompt".into());
    }

    // Resolve embed/argmax handles exactly like rvllm-serve's worker.
    let embedding_mod = g4
        .kernels
        .load_ptx("embedding_gather_f16")
        .map_err(|e| format!("load embedding_gather_f16: {e}"))?;
    let fn_embed = embedding_mod
        .get_function("embedding_gather_f16_kernel")
        .map_err(|e| format!("get embedding_gather_f16_kernel: {e}"))?;
    let fn_argmax = &g4.fused.fn_argmax;

    let prompt: Vec<u32> = if let Ok(pf) = std::env::var("RVLLM_GEN_PROMPT_FILE") {
        let path = std::path::Path::new(&pf);
        let bytes = read_bounded(path, MAX_PROMPT_FILE_BYTES, "RVLLM_GEN_PROMPT_FILE")?;
        let txt = String::from_utf8(bytes)
            .map_err(|e| format!("RVLLM_GEN_PROMPT_FILE {pf} is not UTF-8: {e}"))?;
        let mut ids = Vec::new();
        for token in txt.split_whitespace() {
            if ids.len() >= max_prompt_tokens {
                return Err(format!(
                    "RVLLM_GEN_PROMPT_FILE {pf}: exceeds {max_prompt_tokens} tokens"
                ));
            }
            let id = token
                .parse::<u32>()
                .map_err(|_| format!("RVLLM_GEN_PROMPT_FILE {pf}: invalid token {token:?}"))?;
            if id >= vocab {
                return Err(format!(
                    "RVLLM_GEN_PROMPT_FILE {pf}: token {id} exceeds vocabulary {vocab}"
                ));
            }
            ids.push(id);
        }
        if ids.is_empty() {
            return Err(format!("RVLLM_GEN_PROMPT_FILE {pf}: no token ids"));
        }
        ids
    } else {
        if prompt_len > max_prompt_tokens {
            return Err(format!(
                "RVLLM_GEN_PROMPT must be in 1..={max_prompt_tokens} for this generation length"
            ));
        }
        let mut prompt: Vec<u32> = Vec::with_capacity(prompt_len);
        prompt.push(2u32.min(vocab.saturating_sub(1)));
        let mut s: u32 = 0x1234_5678;
        while prompt.len() < prompt_len {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            prompt.push(100 + (s >> 16) % (vocab.saturating_sub(100)).max(1));
        }
        prompt
    };
    let requested_total = prompt
        .len()
        .checked_add(gen_tokens)
        .ok_or("prompt plus generation length overflow")?;
    if requested_total > max_context {
        return Err(format!(
            "prompt plus generation ({requested_total}) exceeds model context {}",
            max_context
        ));
    }
    let eos: [u32; 0] = [];
    let graph_on = std::env::var("RVLLM_DECODE_GRAPH").ok().as_deref() != Some("0");

    eprintln!(
        "== generate: prompt_len={} gen_tokens={} decode_graph={} vocab={} ==",
        prompt.len(),
        gen_tokens,
        graph_on,
        vocab
    );

    let t0 = Instant::now();
    let out = unsafe { g4.run_generate(&fn_embed, fn_argmax, &prompt, gen_tokens, &eos, &[]) }
        .map_err(|e| format!("run_generate: {e}"))?;
    let wall_s = t0.elapsed().as_secs_f64();

    let token_sha256 = sha256_tokens(&out);
    let n = out.len();
    let kv_cache_dtype = generation_kv_cache_dtype(g4, prompt.len());
    let e2e_tok_s = if wall_s > 0.0 { n as f64 / wall_s } else { 0.0 };
    eprintln!(
        "generate: {} tokens, wall={:.3}s, e2e={:.1} tok/s, decode_graph={}",
        n, wall_s, e2e_tok_s, graph_on
    );
    eprintln!("generate: first8_ids={:?}", &out[..8.min(n)]);
    if let Ok(dump) = std::env::var("RVLLM_GEN_DUMP") {
        let line = out
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        std::fs::write(&dump, line + "\n")
            .map_err(|e| format!("write RVLLM_GEN_DUMP {dump}: {e}"))?;
    }
    let result = serde_json::json!({
        "mode": "generate",
        "input": if std::env::var_os("RVLLM_GEN_PROMPT_FILE").is_some() { "token_file" } else { "synthetic" },
        "decode_graph": graph_on,
        "prompt_tokens": prompt.len(),
        "requested_output_tokens": gen_tokens,
        "output_tokens": n,
        "weight_dtype": "fp8_e4m3",
        "kv_cache_dtype": kv_cache_dtype,
        "elapsed_s": wall_s,
        "tokens_per_second": e2e_tok_s,
        "token_sha256": token_sha256,
        "arena_bytes": arena_bytes,
    });
    println!("{}", finalize_receipt(result, metadata, artifacts)?);
    Ok(())
}

#[cfg(feature = "cuda")]
fn generation_kv_cache_dtype(g4: &Gemma4Bringup, prompt_len: usize) -> &'static str {
    let within_window = prompt_len <= g4.arch.sliding_window_size;
    let skip_decode = std::env::var_os("RVLLM_DIAG_SKIP_DECODE").is_some() && within_window;
    let fast_prefill = std::env::var_os("RVLLM_BATCH_PREFILL").is_some()
        && (within_window || std::env::var_os("RVLLM_CHUNKED_PREFILL").is_some());
    let compare = std::env::var_os("RVLLM_DIAG_COMPARE").is_some() && !skip_decode;
    let spec_decode = std::env::var("RVLLM_SPEC_DECODE").ok().as_deref() == Some("1")
        && !skip_decode
        && !fast_prefill
        && !compare;
    if std::env::var("RVLLM_F16_KV").map_or(true, |value| value != "0")
        && !fast_prefill
        && !spec_decode
    {
        "f16"
    } else {
        "fp8_e4m3"
    }
}

#[cfg(not(feature = "cuda"))]
fn run_generate_bench(
    _g4: &rvllm_runtime::gemma4_bring_up::Gemma4Bringup,
    _metadata: &serde_json::Value,
    _artifacts: &serde_json::Value,
    _arena_bytes: usize,
) -> Result<(), String> {
    Err("rvllm-bench --generate requires --features cuda".into())
}

fn print_result(
    r: rvllm_runtime::bring_up::BenchResult,
    warmup: u32,
    arena_bytes: usize,
    sliding_window_size: Option<u32>,
    metadata: &serde_json::Value,
    artifacts: &serde_json::Value,
) -> Result<(), String> {
    if r.total_ns == 0 || r.iters == 0 || r.num_seqs == 0 {
        return Err("benchmark returned zero work or zero elapsed time".into());
    }
    let tok_per_sec = (r.iters as f64 * r.num_seqs as f64) * 1.0e9 / r.total_ns as f64;
    let ms_per_step = r.ns_per_step as f64 / 1.0e6;
    let block_size = env_u32("RVLLM_BLOCK_SIZE", 32)?;
    let num_blocks_total = env_u32("RVLLM_NUM_BLOCKS", 1024)?;
    let max_blocks_per_sequence = num_blocks_total / r.num_seqs;
    let effective_sliding_window_tokens = sliding_window_size
        .map(|window| window.min(max_blocks_per_sequence.saturating_mul(block_size)));
    let kv_cache_dtype = if std::env::var("RVLLM_F16_KV").ok().as_deref() == Some("0") {
        "fp8_e4m3"
    } else {
        "f16"
    };
    let ttft_str = match (r.ttft_ns, r.ttft_hot_ns) {
        (Some(cold), Some(hot)) => format!(
            " ttft_cold={:.2}ms ttft_hot={:.2}ms",
            cold as f64 / 1.0e6,
            hot as f64 / 1.0e6
        ),
        (Some(cold), None) => format!(" ttft={:.2}ms", cold as f64 / 1.0e6),
        _ => String::new(),
    };
    eprintln!(
        "bench: batch={} context=1 iters={} -> {:.0} tok/s ({:.3} ms/step){}",
        r.num_seqs, r.iters, tok_per_sec, ms_per_step, ttft_str
    );
    let mut output = serde_json::json!({
        "mode": "decode",
        "input": "synthetic_fixed_state",
        "batch": r.num_seqs,
        "context_tokens_per_sequence": 1,
        "block_size": block_size,
        "num_blocks_total": num_blocks_total,
        "max_blocks_per_sequence": max_blocks_per_sequence,
        "effective_sliding_window_tokens": effective_sliding_window_tokens,
        "weight_dtype": "fp8_e4m3",
        "kv_cache_dtype": kv_cache_dtype,
        "graph_capture": std::env::var("RVLLM_NO_GRAPH").ok().as_deref() != Some("1"),
        "fp8_gemm_lt_m1": std::env::var("RVLLM_FP8_GEMM_LT_M1").ok(),
        "fp8_gemm_lt_max_m": std::env::var("RVLLM_FP8_GEMM_LT_MAX_M").ok(),
        "iters": r.iters,
        "warmup": warmup,
        "arena_bytes": arena_bytes,
        "tokens_per_second": tok_per_sec,
        "milliseconds_per_step": ms_per_step,
    });
    if let Some(object) = output.as_object_mut() {
        match (r.ttft_ns, r.ttft_hot_ns) {
            (Some(cold), Some(hot)) => {
                object.insert(
                    "ttft_cold_ms".into(),
                    serde_json::json!(cold as f64 / 1.0e6),
                );
                object.insert("ttft_hot_ms".into(), serde_json::json!(hot as f64 / 1.0e6));
            }
            (Some(cold), None) => {
                object.insert("ttft_ms".into(), serde_json::json!(cold as f64 / 1.0e6));
            }
            _ => {}
        }
    }
    println!("{}", finalize_receipt(output, metadata, artifacts)?);
    Ok(())
}

fn run_sweep(
    br: &Bringup,
    batch: u32,
    iters: u32,
    warmup: u32,
    arena_bytes: usize,
    metadata: &serde_json::Value,
    artifacts: &serde_json::Value,
) -> Result<(), String> {
    // Variant grid. Policy knows 40 non-residual + 10 residual (per the
    // autotune .so). Sample a promising subset.
    let nonres: &[u32] = &[0, 2, 5, 8, 10, 12, 14];
    let residuals: &[u32] = &[100, 102, 105, 108];

    let mut best: Option<(u128, u32, u32)> = None;
    eprintln!("== sweep @ N={batch} ==");
    for &nr in nonres {
        for &r in residuals {
            let ck = br.arena.checkpoint();
            let res =
                unsafe { br.run_bench_with_variants(batch, iters, warmup, Some(nr), Some(r)) };
            unsafe { br.arena.restore(ck) }.map_err(|e| e.to_string())?;
            match res {
                Ok(r_) => {
                    let tok_per_sec = if r_.total_ns > 0 {
                        (r_.iters as f64 * r_.num_seqs as f64) * 1.0e9 / r_.total_ns as f64
                    } else {
                        0.0
                    };
                    eprintln!(
                        "nonres={nr} res={r} -> {:.0} tok/s ({:.3} ms/step)",
                        tok_per_sec,
                        r_.ns_per_step as f64 / 1.0e6
                    );
                    let output = serde_json::json!({
                        "mode": "variant_sweep",
                        "nonres": nr,
                        "res": r,
                        "batch": batch,
                        "iters": iters,
                        "warmup": warmup,
                        "arena_bytes": arena_bytes,
                        "tokens_per_second": tok_per_sec,
                        "milliseconds_per_step": r_.ns_per_step as f64 / 1.0e6,
                    });
                    println!("{}", finalize_receipt(output, metadata, artifacts)?);
                    if best.map_or(true, |current| r_.ns_per_step < current.0) {
                        best = Some((r_.ns_per_step, nr, r));
                    }
                }
                Err(e) => {
                    eprintln!("nonres={nr} res={r} -> ERROR: {e}");
                }
            }
        }
    }
    let best = best.ok_or("every sweep variant failed")?;
    eprintln!(
        "BEST: nonres={} res={} ({:.3} ms/step)",
        best.1,
        best.2,
        best.0 as f64 / 1.0e6
    );
    Ok(())
}

fn read_bounded(path: &std::path::Path, limit: u64, label: &str) -> Result<Vec<u8>, String> {
    let file = std::fs::File::open(path)
        .map_err(|error| format!("{label} {}: {error}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.len() as u64 > limit {
        return Err(format!("{label} {} exceeds {limit} bytes", path.display()));
    }
    Ok(bytes)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn e4m3fn_decoder_covers_edges() {
        assert_eq!(fp8_to_f32(0x00).to_bits(), 0.0_f32.to_bits());
        assert_eq!(fp8_to_f32(0x80).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(fp8_to_f32(0x01), 1.0 / 512.0);
        assert_eq!(fp8_to_f32(0x07), 7.0 / 512.0);
        assert_eq!(fp8_to_f32(0x08), 1.0 / 64.0);
        assert_eq!(fp8_to_f32(0x7e), 448.0);
        assert_eq!(fp8_to_f32(0xfe), -448.0);
        assert!(fp8_to_f32(0x7f).is_nan());
        assert!(fp8_to_f32(0xff).is_nan());
    }

    #[test]
    fn generated_fp8_is_finite_and_exercises_subnormals() {
        let values = gen_fp8(4096, 7);
        assert!(values.iter().all(|value| fp8_to_f32(*value).is_finite()));
        assert!(values
            .iter()
            .any(|value| value & 0x78 == 0 && value & 0x07 != 0));
    }

    #[test]
    fn parity_row_sample_includes_both_edges() {
        let rows = parity_rows(10_000, 64, 42);
        assert_eq!(rows.len(), 64);
        assert_eq!(rows[0], 0);
        assert_eq!(*rows.last().unwrap(), 9_999);
        assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn receipt_hash_covers_every_other_field() {
        let metadata = serde_json::json!({"schema": "test", "source_sha": "abc"});
        let artifacts = serde_json::json!({"kernel": "def"});
        let encoded =
            finalize_receipt(serde_json::json!({"result": 1}), &metadata, &artifacts).unwrap();
        let mut receipt: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        let expected = receipt
            .as_object_mut()
            .unwrap()
            .remove("receipt_sha256")
            .unwrap();
        assert_eq!(
            expected,
            sha256_bytes(&serde_json::to_vec(&receipt).unwrap())
        );
    }
}
