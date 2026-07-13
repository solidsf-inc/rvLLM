pub mod http;
pub mod openai;
pub mod worker;

use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct ServeConfig {
    pub backend: Backend,
    pub host: String,
    pub port: u16,
    pub served_model_name: String,
    pub default_system_prompt: Option<String>,
    pub max_model_len: usize,
    pub max_num_seqs: usize,
    pub max_inflight_requests: usize,
    pub max_num_batched_tokens: usize,
    pub max_prefill_chunk: usize,
    pub dry_run: bool,
    /// Optional bearer token required by every generation and metadata API.
    /// The health endpoint remains unauthenticated for local supervision.
    pub api_key: Option<String>,
    /// Path to the vision weights directory. Only used when the binary is
    /// built with `--features vision`. When the flag is set on a build
    /// without the feature, `from_env_and_args` returns an error so users
    /// don't silently run text-only on a vision request.
    pub vision_weights_dir: Option<PathBuf>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Metal,
}

impl Backend {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "cuda" | "gpu" => Ok(Self::Cuda),
            "metal" | "apple" | "mac" => Ok(Self::Metal),
            other => Err(format!("invalid backend {other:?}; expected cuda or metal")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
}

impl ServeConfig {
    pub fn from_env_and_args<I>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = String>,
    {
        if let Some(name) = rvllm_core::env::first_unknown_rvllm_env() {
            return Err(format!("unknown environment variable: {name}"));
        }
        let mut cfg = Self {
            backend: match non_empty_env("RVLLM_BACKEND") {
                Some(v) => Backend::parse(&v)?,
                None => Backend::Cuda,
            },
            host: "127.0.0.1".into(),
            port: 8080,
            served_model_name: std::env::var("RVLLM_SERVED_MODEL_NAME")
                .unwrap_or_else(|_| "gemma-4".into()),
            default_system_prompt: None,
            max_model_len: 8192,
            max_num_seqs: 1,
            max_inflight_requests: env_usize("RVLLM_MAX_INFLIGHT_REQUESTS").unwrap_or(4),
            max_num_batched_tokens: 2048,
            max_prefill_chunk: 128,
            dry_run: env_bool("RVLLM_DRY_RUN"),
            api_key: non_empty_env("RVLLM_API_KEY"),
            vision_weights_dir: non_empty_env("RVLLM_VISION_WEIGHTS_DIR").map(PathBuf::from),
        };
        let mut system_prompt = non_empty_env("RVLLM_SYSTEM_PROMPT");
        let mut system_prompt_file = non_empty_env("RVLLM_SYSTEM_PROMPT_FILE");

        let mut it = args.into_iter().peekable();
        while let Some(arg) = it.next() {
            let (key, value) = if let Some((k, v)) = arg.split_once('=') {
                (k.to_string(), Some(v.to_string()))
            } else {
                (arg, None)
            };
            match key.as_str() {
                "--host" => cfg.host = next_value("--host", value, &mut it)?,
                "--port" => cfg.port = parse_value("--port", value, &mut it)?,
                "--backend" => {
                    cfg.backend = Backend::parse(&next_value("--backend", value, &mut it)?)?
                }
                "--model" | "--served-model-name" => {
                    cfg.served_model_name = next_value(&key, value, &mut it)?
                }
                "--system-prompt" => {
                    system_prompt = Some(next_value("--system-prompt", value, &mut it)?)
                }
                "--system-prompt-file" => {
                    system_prompt_file = Some(next_value("--system-prompt-file", value, &mut it)?)
                }
                "--max-model-len" => {
                    cfg.max_model_len = parse_value("--max-model-len", value, &mut it)?
                }
                "--max-num-seqs" => {
                    cfg.max_num_seqs = parse_value("--max-num-seqs", value, &mut it)?
                }
                "--max-inflight-requests" => {
                    cfg.max_inflight_requests =
                        parse_value("--max-inflight-requests", value, &mut it)?
                }
                "--max-num-batched-tokens" => {
                    cfg.max_num_batched_tokens =
                        parse_value("--max-num-batched-tokens", value, &mut it)?
                }
                "--max-prefill-chunk" => {
                    cfg.max_prefill_chunk = parse_value("--max-prefill-chunk", value, &mut it)?
                }
                "--dry-run" => cfg.dry_run = true,
                "--vision-weights-dir" => {
                    cfg.vision_weights_dir = Some(PathBuf::from(next_value(
                        "--vision-weights-dir",
                        value,
                        &mut it,
                    )?))
                }
                other => return Err(format!("unknown argument: {other}")),
            }
        }

        if cfg.max_num_seqs != 1 {
            return Err("rvllm-server currently supports --max-num-seqs 1".into());
        }
        if cfg.max_inflight_requests == 0 {
            return Err("max_inflight_requests must be >= 1".into());
        }
        if cfg.max_model_len == 0 || cfg.max_num_batched_tokens == 0 || cfg.max_prefill_chunk == 0 {
            return Err("model, batching, and prefill limits must be >= 1".into());
        }
        if cfg.max_prefill_chunk > cfg.max_num_batched_tokens {
            return Err("max_prefill_chunk must not exceed max_num_batched_tokens".into());
        }
        if cfg.served_model_name.trim().is_empty() {
            return Err("served model name must not be empty".into());
        }
        if let Some(api_key) = cfg.api_key.as_deref() {
            if api_key.len() > 4096
                || api_key.trim() != api_key
                || api_key.bytes().any(|byte| byte.is_ascii_control())
            {
                return Err(
                    "RVLLM_API_KEY must be at most 4096 bytes with no surrounding whitespace or control characters"
                        .into(),
                );
            }
        }
        if !is_loopback_host(&cfg.host) {
            return Err(
                "rvllm-server currently supports loopback binds only; terminate TLS and authentication in an audited local proxy"
                    .into(),
            );
        }
        #[cfg(not(feature = "vision"))]
        if cfg.vision_weights_dir.is_some() {
            return Err(
                "--vision-weights-dir / RVLLM_VISION_WEIGHTS_DIR was set but rvllm-server was built without --features vision"
                    .into(),
            );
        }
        cfg.default_system_prompt = load_system_prompt(system_prompt, system_prompt_file)?;
        Ok(cfg)
    }

    pub fn addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

fn env_bool(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn is_loopback_host(host: &str) -> bool {
    matches!(host.trim(), "127.0.0.1" | "::1" | "[::1]" | "localhost")
}

fn load_system_prompt(
    inline: Option<String>,
    file: Option<String>,
) -> Result<Option<String>, String> {
    if let Some(path) = file {
        let metadata = std::fs::metadata(&path)
            .map_err(|e| format!("stat RVLLM_SYSTEM_PROMPT_FILE {path}: {e}"))?;
        if metadata.len() > 1024 * 1024 {
            return Err("RVLLM_SYSTEM_PROMPT_FILE exceeds 1 MiB".into());
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("read RVLLM_SYSTEM_PROMPT_FILE {path}: {e}"))?;
        let text = text.trim_end().to_string();
        return Ok((!text.is_empty()).then_some(text));
    }
    Ok(inline
        .map(|s| s.trim_end().to_string())
        .filter(|s| !s.is_empty()))
}

fn next_value<I>(
    flag: &str,
    value: Option<String>,
    it: &mut std::iter::Peekable<I>,
) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    match value {
        Some(v) => Ok(v),
        None => it.next().ok_or_else(|| format!("missing value for {flag}")),
    }
}

fn parse_value<T, I>(
    flag: &str,
    value: Option<String>,
    it: &mut std::iter::Peekable<I>,
) -> Result<T, String>
where
    T: std::str::FromStr,
    I: Iterator<Item = String>,
{
    let raw = next_value(flag, value, it)?;
    raw.parse()
        .map_err(|_| format!("invalid value for {flag}: {raw}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_backend_flag() {
        let cfg = ServeConfig::from_env_and_args(["--backend=metal".to_string()]).unwrap();
        assert_eq!(cfg.backend, Backend::Metal);
    }

    #[test]
    fn rejects_invalid_backend() {
        let err = ServeConfig::from_env_and_args(["--backend=hypura".to_string()]).unwrap_err();
        assert!(err.contains("expected cuda or metal"));
    }
}
