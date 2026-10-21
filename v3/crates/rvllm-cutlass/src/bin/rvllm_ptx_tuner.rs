//! CLI entry point for the `rvLLM` `1,000D x 100W` PTX-CUDA Ultra Kernel Tuner.

use std::path::PathBuf;

use rvllm_cutlass::ptx_tuner::hardware::all_hardware_catalog;
use rvllm_cutlass::ptx_tuner::{
    run_tuning_campaign, write_campaign_bundle, DEFAULT_SEARCH_DEPTH, DEFAULT_SEARCH_WIDTH,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return;
    }
    if args.get(1).is_some_and(|s| s == "hardware-matrix") {
        println!(
            "{}",
            serde_json::to_string_pretty(&all_hardware_catalog()).expect("serialize catalog")
        );
        return;
    }

    let model = flag_value(&args, "--model").unwrap_or_else(|| "gemma4-31b".into());
    let hardware = flag_value(&args, "--hardware").unwrap_or_else(|| "all".into());
    let revision = flag_value(&args, "--revision").unwrap_or_else(|| "0.3.0".into());
    let depth = flag_value(&args, "--depth")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(DEFAULT_SEARCH_DEPTH);
    let width = flag_value(&args, "--width")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(DEFAULT_SEARCH_WIDTH);
    let out_dir = flag_value(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/ptx-tuner-bundle"));

    let report = run_tuning_campaign(&model, &hardware, depth, width, &revision);
    if let Err(err) = write_campaign_bundle(&report, &out_dir) {
        eprintln!(
            "failed to write PTX tuner bundle to {}: {err}",
            out_dir.display()
        );
        std::process::exit(1);
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "overall": report.overall,
            "model_id": report.model_id,
            "search_topology": format!("{}D x {}W", report.search_depth, report.search_width),
            "evaluations_per_workload": report.evaluations_per_workload,
            "total_evaluations_run": report.total_evaluations_run,
            "out_dir": out_dir,
            "hardware_summaries": report.hardware_summaries,
        }))
        .expect("serialize summary")
    );
    if report.overall != "pass" {
        std::process::exit(2);
    }
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    let mut i = 1;
    while i < args.len() {
        if args[i] == flag {
            return args.get(i + 1).cloned();
        }
        if let Some(rest) = args[i].strip_prefix(&format!("{flag}=")) {
            return Some(rest.to_string());
        }
        i += 1;
    }
    None
}

fn print_usage() {
    println!(
        "rvllm_ptx_tuner — 1,000D x 100W PTX-CUDA Ultra Kernel Tuner\n\n\
         USAGE:\n\
         \x20 rvllm_ptx_tuner [OPTIONS]\n\
         \x20 rvllm_ptx_tuner hardware-matrix\n\n\
         OPTIONS:\n\
         \x20 --model <ID>         Model profile (gemma4-31b, gemma4-e4b, qwen25-7b, llama3-70b, deepseek-v41)\n\
         \x20 --hardware <FILTER>  Target hardware ('all' or comma-separated sm_70..sm_121a)\n\
         \x20 --depth <N>          Sequential refinement generations (default: 1000)\n\
         \x20 --width <W>          Parallel schedule beam width (default: 100)\n\
         \x20 --revision <REV>     Policy revision string (default: 0.3.0)\n\
         \x20 --out <DIR>          Output bundle directory"
    );
}
