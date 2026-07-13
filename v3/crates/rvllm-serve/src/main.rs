use tracing_subscriber::EnvFilter;

fn main() {
    if let Err(e) = run() {
        eprintln!("rvllm-server: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        return Ok(());
    }

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let config = rvllm_serve::ServeConfig::from_env_and_args(args)?;
    let worker = rvllm_serve::worker::WorkerHandle::start(config.clone())?;
    rvllm_serve::http::serve(config, worker)
}

fn print_help() {
    println!(
        "rvllm-server\n\
\n\
Usage:\n\
  rvllm-server [--backend cuda|metal] [--host 127.0.0.1] [--port 8080]\n\
               [--max-model-len 8192]\n\
               [--max-inflight-requests 4] [--vision-weights-dir <PATH>]\n\
\n\
Environment:\n\
  RVLLM_BACKEND              cuda (default) or metal\n\
  RVLLM_MODEL_DIR            HF model directory with tokenizer.json\n\
  RVLLM_KERNELS_DIR          kernel bundle root containing architecture dirs\n\
  RVLLM_CUTLASS_SO           SM90 CUTLASS shared object\n\
  RVLLM_FA3_SO               FA3 shared object\n\
  RVLLM_POLICY               CUTLASS policy.json\n\
  RVLLM_SERVED_MODEL_NAME    public model id, default gemma-4\n\
  RVLLM_API_KEY              optional bearer token for HTTP endpoints\n\
  RVLLM_SYSTEM_PROMPT        default system prompt prepended to chat requests\n\
  RVLLM_SYSTEM_PROMPT_FILE   path to default system prompt file\n\
  RVLLM_MAX_INFLIGHT_REQUESTS requests admitted before 429 busy, default 4\n\
  RVLLM_DRY_RUN=1            bind HTTP without loading CUDA\n\
  RVLLM_VISION_WEIGHTS_DIR   vision-tower weights dir (requires --features vision)\n\
  RVLLM_SPEC_DECODE=1        speculative decode for greedy requests (RVLLM_SPEC_K, default 4)\n\
  RVLLM_SERVE_SESSION=0      legacy per-request engine path (re-captures graphs every request)\n\
\n\
The server accepts loopback host addresses only. Use a trusted local reverse\n\
proxy for remote access."
    );
}
