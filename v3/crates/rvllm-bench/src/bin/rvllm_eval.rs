//! Single-sequence Gemma 4 generation for quality evaluation.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rvllm_core::{ModelArch, ModelConfig};
use rvllm_runtime::gemma4_bring_up::{Gemma4Bringup, Gemma4EnginePaths};
use sha2::{Digest, Sha256};

const MAX_PROMPT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TOKENIZER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_TOKENIZER_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_ARENA_GIB: usize = 256;
const MAX_METADATA_TEXT: usize = 256;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    if let Err(error) = run() {
        eprintln!("rvllm-eval: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let model_dir = env_path("RVLLM_MODEL_DIR")?;
    let config = ModelConfig::load_hf(&model_dir)
        .map_err(|error| format!("model config {}: {error}", model_dir.display()))?;
    if config.architecture != ModelArch::Gemma4 {
        return Err("rvllm-eval supports Gemma 4 models only".into());
    }

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer_bytes = read_bounded(&tokenizer_path, MAX_TOKENIZER_BYTES, "tokenizer")?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|error| format!("tokenizer {}: {error}", tokenizer_path.display()))?;
    let (bos_id, eos_ids) = special_token_ids(&model_dir, &tokenizer)?;
    validate_token_id(bos_id, config.vocab_size, "BOS")?;
    for token in &eos_ids {
        validate_token_id(*token, config.vocab_size, "EOS")?;
    }

    let prompt = read_prompt()?;
    let prompt_sha256 = sha256_bytes(prompt.as_bytes());
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|error| format!("tokenize: {error}"))?;
    let mut prompt_ids = Vec::with_capacity(encoding.len().saturating_add(1));
    prompt_ids.push(bos_id);
    prompt_ids.extend_from_slice(encoding.get_ids());
    if prompt_ids
        .iter()
        .any(|token| *token as usize >= config.vocab_size)
    {
        return Err("tokenizer emitted an ID outside the model vocabulary".into());
    }

    let max_new = env_usize("RVLLM_MAX_TOKENS", 256)?;
    if max_new == 0 {
        return Err("RVLLM_MAX_TOKENS must be greater than zero".into());
    }
    let total = prompt_ids
        .len()
        .checked_add(max_new)
        .ok_or("prompt plus output token count overflow")?;
    if total > config.max_position_embeddings {
        return Err(format!(
            "prompt ({}) plus requested output ({max_new}) exceeds model context {}",
            prompt_ids.len(),
            config.max_position_embeddings
        ));
    }

    let paths = Gemma4EnginePaths {
        model_dir: model_dir.clone(),
        kernels_dir: env_path("RVLLM_KERNELS_DIR")?,
        cutlass_so: env_path("RVLLM_CUTLASS_SO")?,
        fa3_so: env_path("RVLLM_FA3_SO")?,
        policy_json: env_path("RVLLM_POLICY")?,
        w4a8_so: None,
    };
    let metadata = receipt_metadata()?;
    let artifacts = artifact_digests(&paths)?;
    let arena_bytes = arena_bytes()?;
    let load_started = Instant::now();
    let engine = Gemma4Bringup::load(paths, arena_bytes)
        .map_err(|error| format!("Gemma 4 bringup: {error}"))?;
    if engine.arch.is_e4b() {
        return Err("this Gemma 4 variant is not supported by the public evaluator".into());
    }
    eprintln!("bringup_s={:.3}", load_started.elapsed().as_secs_f64());

    let embedding_module = engine
        .kernels
        .load_ptx("embedding_gather_f16")
        .map_err(|error| format!("load embedding kernel: {error}"))?;
    let embedding = embedding_module
        .get_function("embedding_gather_f16_kernel")
        .map_err(|error| format!("resolve embedding kernel: {error}"))?;

    let started = Instant::now();
    let output_ids = unsafe {
        engine.run_generate(
            &embedding,
            &engine.fused.fn_argmax,
            &prompt_ids,
            max_new,
            &eos_ids,
            &[],
        )
    }
    .map_err(|error| format!("generate: {error}"))?;
    if output_ids
        .iter()
        .any(|token| *token as usize >= config.vocab_size)
    {
        return Err("model returned an ID outside the vocabulary".into());
    }
    let elapsed = started.elapsed().as_secs_f64();
    let text = tokenizer
        .decode(&output_ids, true)
        .map_err(|error| format!("detokenize: {error}"))?;
    let tokens_per_second = if elapsed > 0.0 {
        output_ids.len() as f64 / elapsed
    } else {
        0.0
    };
    let result = serde_json::json!({
        "mode": "generation",
        "text": text,
        "prompt_tokens": prompt_ids.len(),
        "output_tokens": output_ids.len(),
        "elapsed_s": elapsed,
        "tokens_per_second": tokens_per_second,
        "input_sha256": prompt_sha256,
        "max_new_tokens": max_new,
        "arena_bytes": arena_bytes,
    });
    println!("{}", finalize_receipt(result, metadata, artifacts)?);
    Ok(())
}

fn read_prompt() -> Result<String, String> {
    if let Ok(prompt) = std::env::var("RVLLM_PROMPT") {
        if prompt.len() as u64 > MAX_PROMPT_BYTES {
            return Err(format!("RVLLM_PROMPT exceeds {MAX_PROMPT_BYTES} bytes"));
        }
        if prompt.is_empty() {
            return Err("RVLLM_PROMPT is empty".into());
        }
        return Ok(prompt);
    }
    let mut bytes = std::io::stdin().take(MAX_PROMPT_BYTES + 1);
    let mut prompt = String::new();
    bytes
        .read_to_string(&mut prompt)
        .map_err(|error| format!("stdin: {error}"))?;
    if prompt.len() as u64 > MAX_PROMPT_BYTES {
        return Err(format!("stdin exceeds {MAX_PROMPT_BYTES} bytes"));
    }
    if prompt.is_empty() {
        return Err("empty prompt (set RVLLM_PROMPT or pipe stdin)".into());
    }
    Ok(prompt)
}

fn special_token_ids(
    model_dir: &Path,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<(u32, Vec<u32>), String> {
    let path = model_dir.join("tokenizer_config.json");
    let bytes = read_bounded(&path, MAX_TOKENIZER_CONFIG_BYTES, "tokenizer metadata")?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let bos = token_strings(value.get("bos_token"))
        .into_iter()
        .next()
        .and_then(|token| tokenizer.token_to_id(&token))
        .ok_or("tokenizer metadata does not define a resolvable BOS token")?;
    let mut eos: Vec<u32> = token_strings(value.get("eos_token"))
        .into_iter()
        .map(|token| {
            tokenizer
                .token_to_id(&token)
                .ok_or_else(|| format!("EOS token {token:?} is absent from tokenizer.json"))
        })
        .collect::<Result<_, _>>()?;
    eos.sort_unstable();
    eos.dedup();
    if eos.is_empty() {
        return Err("tokenizer metadata does not define an EOS token".into());
    }
    Ok((bos, eos))
}

fn token_strings(value: Option<&serde_json::Value>) -> Vec<String> {
    match value {
        Some(serde_json::Value::String(value)) => vec![value.clone()],
        Some(serde_json::Value::Object(value)) => value
            .get("content")
            .and_then(serde_json::Value::as_str)
            .map(|value| vec![value.to_owned()])
            .unwrap_or_default(),
        Some(serde_json::Value::Array(values)) => values
            .iter()
            .flat_map(|value| token_strings(Some(value)))
            .collect(),
        _ => Vec::new(),
    }
}

fn arena_bytes() -> Result<usize, String> {
    if let Ok(value) = std::env::var("RVLLM_ARENA_GB") {
        let gib = value
            .parse::<usize>()
            .map_err(|_| "RVLLM_ARENA_GB must be an integer")?;
        if !(1..=MAX_ARENA_GIB).contains(&gib) {
            return Err(format!("RVLLM_ARENA_GB must be in 1..={MAX_ARENA_GIB}"));
        }
        return gib
            .checked_mul(1usize << 30)
            .ok_or("RVLLM_ARENA_GB byte count overflow".into());
    }

    let context = rvllm_mem::context::CudaContextHandle::init(0)
        .map_err(|error| format!("CUDA context: {error}"))?;
    let mut free = 0usize;
    let mut total = 0usize;
    let status = unsafe { cudarc::driver::sys::cuMemGetInfo_v2(&mut free, &mut total) };
    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemGetInfo_v2 failed: {status:?}"));
    }
    drop(context);
    let reserve = 1024usize << 20;
    free.checked_sub(reserve)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| "less than 1 GiB of free CUDA memory".into())
}

fn env_path(name: &str) -> Result<PathBuf, String> {
    let path = std::env::var_os(name).ok_or_else(|| format!("missing env var: {name}"))?;
    if path.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    Ok(PathBuf::from(path))
}

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| format!("{name} must be a non-negative integer")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("{name}: {error}")),
    }
}

fn validate_token_id(token: u32, vocab: usize, label: &str) -> Result<(), String> {
    if token as usize >= vocab {
        return Err(format!(
            "{label} token {token} is outside vocabulary {vocab}"
        ));
    }
    Ok(())
}

fn read_bounded(path: &Path, limit: u64, label: &str) -> Result<Vec<u8>, String> {
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

fn receipt_metadata() -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "schema": "rvllm.local-benchmark.v1",
        "source_sha": required_hex("RVLLM_SOURCE_SHA", 40)?,
        "model": required_text("RVLLM_MODEL_ID")?,
        "model_sha256": required_hex("RVLLM_MODEL_SHA256", 64)?,
        "hardware": required_text("RVLLM_HARDWARE")?,
        "driver": required_text("RVLLM_DRIVER")?,
        "toolchain": required_text("RVLLM_TOOLCHAIN")?,
        "backend": "cuda",
        "command": command(),
        "unix_time_seconds": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock: {error}"))?
            .as_secs(),
    }))
}

fn artifact_digests(paths: &Gemma4EnginePaths) -> Result<serde_json::Value, String> {
    let manifest = paths
        .kernels_dir
        .join(required_text("RVLLM_KERNEL_ARCH")?)
        .join("manifest.json");
    Ok(serde_json::json!({
        "kernel_manifest_sha256": sha256_file(&manifest)?,
        "cutlass_sha256": sha256_file(&paths.cutlass_so)?,
        "flash_attention_sha256": sha256_file(&paths.fa3_so)?,
        "policy_sha256": sha256_file(&paths.policy_json)?,
    }))
}

fn finalize_receipt(
    mut result: serde_json::Value,
    metadata: serde_json::Value,
    artifacts: serde_json::Value,
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
        object.insert("artifacts".into(), artifacts);
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
        .unwrap_or_else(|| "rvllm-eval".into());
    std::iter::once(binary)
        .chain(arguments.map(|argument| argument.to_string_lossy().into_owned()))
        .collect()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn sha256_file(path: &Path) -> Result<String, String> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receipt_hash_covers_every_other_field() {
        let encoded = finalize_receipt(
            serde_json::json!({"result": 1}),
            serde_json::json!({"schema": "test"}),
            serde_json::json!({"artifact": "abc"}),
        )
        .unwrap();
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
