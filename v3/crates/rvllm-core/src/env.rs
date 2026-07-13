//! Environment-variable whitelist per `v3/specs/02-config.md`.
//!
//! Every `RVLLM_*` variable accepted by the server process is listed here.
//! `rvllm-serve` scans once while parsing configuration and rejects variables
//! outside this list. This catches typos such as `RVLLM_NOGRAPH=1` that would
//! otherwise be silently ignored.

/// The whitelist. Alphabetical.
pub const ENV_WHITELIST: &[&str] = &[
    "RVLLM_API_KEY",
    "RVLLM_ARENA_GB",
    "RVLLM_ARGMAX_GRID",
    "RVLLM_AUTOTUNE_CACHE",
    "RVLLM_BACKEND",
    "RVLLM_BASE_URL",
    "RVLLM_BATCH",
    "RVLLM_BATCH_PREFILL",
    "RVLLM_BENCH_GENERATE",
    "RVLLM_BLOCK_SIZE",
    "RVLLM_CHUNKED_PREFILL",
    "RVLLM_CUTLASS_SM120_SO",
    "RVLLM_CUTLASS_SO",
    "RVLLM_DBG_LAYER",
    "RVLLM_DBG_LAYER0",
    "RVLLM_DBG_PLE",
    "RVLLM_DBG_PRED",
    "RVLLM_DBG_RES",
    "RVLLM_DECODE_GRAPH",
    "RVLLM_DEFAULT_MAX_TOKENS",
    "RVLLM_DEVICE",
    "RVLLM_DIAG_COMPARE",
    "RVLLM_DIAG_SKIP_DECODE",
    "RVLLM_DIAG_VERBOSE",
    "RVLLM_DISABLE_PLE",
    "RVLLM_DRY_RUN",
    "RVLLM_DRY_RUN_DELAY_MS",
    "RVLLM_DUMP_IDS",
    "RVLLM_DUMP_LOGITS",
    "RVLLM_DUMP_TOKENS",
    "RVLLM_E4B",
    "RVLLM_E4B_REQUIRE",
    "RVLLM_EOS",
    "RVLLM_EOS_IDS",
    "RVLLM_F16_KV",
    "RVLLM_F16_LAYERS",
    "RVLLM_F16_ONLY",
    "RVLLM_FA3_SO",
    "RVLLM_FA_FALLBACK_SO",
    "RVLLM_FORCE_FA2_PTX",
    "RVLLM_FP8_GEMM_CUTLASS_SM120",
    "RVLLM_FP8_GEMM_LT_F16OUT",
    "RVLLM_FP8_GEMM_LT_M1",
    "RVLLM_FP8_GEMM_LT_MAX_M",
    "RVLLM_FP8_GEMV_M1",
    "RVLLM_GEMM_DIAG",
    "RVLLM_GENERATE",
    "RVLLM_GEN_DUMP",
    "RVLLM_GEN_PROMPT",
    "RVLLM_GEN_PROMPT_FILE",
    "RVLLM_GEN_TOKENS",
    "RVLLM_INT4",
    "RVLLM_INT4_REQUIRE",
    "RVLLM_ITERS",
    "RVLLM_KERNELS_DIR",
    "RVLLM_KERNEL_ARCH",
    "RVLLM_KERNEL_DIR",
    "RVLLM_KERNEL_MANIFEST_SHA256",
    "RVLLM_KV_SCALE",
    "RVLLM_KV_SCALE_ABSMAX",
    "RVLLM_LMHEAD_FULLVOCAB",
    "RVLLM_LMHEAD_PRUNE",
    "RVLLM_LMHEAD_PRUNE_REQUIRE",
    "RVLLM_LOG",
    "RVLLM_M1_GEMV",
    "RVLLM_MAX_INFLIGHT_REQUESTS",
    "RVLLM_MAX_LAYERS",
    "RVLLM_MAX_NEW",
    "RVLLM_MAX_TOKENS",
    "RVLLM_METALLIB_PATH",
    "RVLLM_METAL_ATTENTION",
    "RVLLM_METAL_ATTENTION_MIN",
    "RVLLM_METAL_BATCH_GEMV",
    "RVLLM_METAL_FUSED_MLP",
    "RVLLM_METAL_LMHEAD_ARGMAX",
    "RVLLM_METAL_LMHEAD_FULL_LOGITS",
    "RVLLM_METAL_LMHEAD_NLL",
    "RVLLM_METAL_MAX_CACHED_LAYERS",
    "RVLLM_METAL_PAR_ATTEND_MIN",
    "RVLLM_METAL_PRECOMPILE",
    "RVLLM_METAL_PREFETCH_LAYERS",
    "RVLLM_MODEL",
    "RVLLM_MODEL_DIR",
    "RVLLM_NAN_CHECK",
    "RVLLM_NO_GRAPH",
    "RVLLM_NO_PLE_GATE",
    "RVLLM_NO_SOFTCAP",
    "RVLLM_NUM_BLOCKS",
    "RVLLM_PER_TOKEN_Q_SCALE",
    "RVLLM_PLE_FOLD",
    "RVLLM_POLICY",
    "RVLLM_POLICY_SHA256",
    "RVLLM_PPL_CHUNK",
    "RVLLM_PPL_CHUNKS",
    "RVLLM_PPL_FULLHEAD",
    "RVLLM_PPL_MAX_MODEL_LEN",
    "RVLLM_PPL_STRIDE",
    "RVLLM_PPL_TEXT",
    "RVLLM_PREFILL",
    "RVLLM_PREFILL_CHUNK",
    "RVLLM_PREFILL_LEN",
    "RVLLM_PROBE_OUTPUT",
    "RVLLM_PROMPT",
    "RVLLM_PROMPT_IDS",
    "RVLLM_PROMPT_LEN",
    "RVLLM_Q_SCALE",
    "RVLLM_REAL_PREFILL",
    "RVLLM_RELEASE_REVISION",
    "RVLLM_REQUEST_TIMEOUT_SECS",
    "RVLLM_SAMPLE_DUMP_LOGITS",
    "RVLLM_SAMPLE_SEED",
    "RVLLM_SAMPLE_STAT",
    "RVLLM_SAMPLE_T",
    "RVLLM_SAMPLE_TOPK",
    "RVLLM_SAMPLE_TOPP",
    "RVLLM_SERVED_MODEL_NAME",
    "RVLLM_SERVE_SESSION",
    "RVLLM_SKIP_LM_HEAD",
    "RVLLM_SLIDING_WINDOW",
    "RVLLM_SPEC_DECODE",
    "RVLLM_SPEC_GRAPH",
    "RVLLM_SPEC_K",
    "RVLLM_SPEC_NGRAM",
    "RVLLM_SPLITK",
    "RVLLM_SPLIT_QKV",
    "RVLLM_SWEEP",
    "RVLLM_SYNC_LAYERS",
    "RVLLM_SYSTEM_PROMPT",
    "RVLLM_SYSTEM_PROMPT_FILE",
    "RVLLM_TILE_K",
    "RVLLM_TTFT",
    "RVLLM_VISION",
    "RVLLM_VISION_WEIGHTS_DIR",
    "RVLLM_W4A8_SO",
    "RVLLM_WARMUP",
];

/// Returns the first unknown `RVLLM_*` var found in the process env,
/// or `None` if every such var is in the whitelist.
pub fn first_unknown_rvllm_env() -> Option<String> {
    let mut unknown: Vec<String> = std::env::vars()
        .map(|(key, _)| key)
        .filter(|key| key.starts_with("RVLLM_"))
        .filter(|key| !ENV_WHITELIST.contains(&key.as_str()))
        .collect();
    unknown.sort();
    unknown.into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitelist_is_sorted_and_unique() {
        let mut sorted = ENV_WHITELIST.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.as_slice(), ENV_WHITELIST);
    }

    #[test]
    fn detects_unknown_env_in_process() {
        // Set one bad, one good; detect the bad one.
        std::env::set_var("RVLLM_DEFINITELY_NOT_REAL", "1");
        let bad = first_unknown_rvllm_env();
        std::env::remove_var("RVLLM_DEFINITELY_NOT_REAL");
        assert_eq!(bad.as_deref(), Some("RVLLM_DEFINITELY_NOT_REAL"));
    }
}
