#![cfg(feature = "cuda")]

use cudarc::driver::sys::{cuCtxGetCurrent, cuMemcpyDtoH_v2, CUcontext, CUresult};
use rvllm_mem::{CudaContextHandle, HbmArena};

#[test]
#[ignore = "requires a real CUDA device; run with `--ignored`"]
fn context_is_current_and_hbm_copy_round_trips() {
    if std::env::var("RVLLM_RUN_CUDA_CONTEXT_SMOKE")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skipped: set RVLLM_RUN_CUDA_CONTEXT_SMOKE=1");
        return;
    }

    let context = CudaContextHandle::init(0).expect("CudaContextHandle::init");
    let mut current: CUcontext = std::ptr::null_mut();
    assert_eq!(
        unsafe { cuCtxGetCurrent(&mut current) },
        CUresult::CUDA_SUCCESS
    );
    assert!(!current.is_null(), "init must leave its context current");

    let arena = HbmArena::new(&context, 4096).expect("HbmArena::new");
    let region = arena.region("context_smoke", 16, 16).expect("region");
    let expected = *b"rvLLM context OK";
    unsafe { region.copy_from_host(&expected).expect("H2D") };

    let mut actual = [0u8; 16];
    assert_eq!(
        unsafe {
            cuMemcpyDtoH_v2(
                actual.as_mut_ptr().cast(),
                region.device_ptr(),
                actual.len(),
            )
        },
        CUresult::CUDA_SUCCESS
    );
    assert_eq!(actual, expected);
}
