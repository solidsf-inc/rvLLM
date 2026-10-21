# Vast.ai GPU Fleet Dispatch

`scripts/vast_ptx_tuner.py` and `scripts/remote_gpu_eval.sh` connect the `1,000D x 100W` PTX-CUDA Ultra Kernel Tuner to ephemeral Vast.ai GPU instances so every synthesized kernel is compiled and benchmarked on physical silicon across `sm_70` through `sm_121a`.

## 1. End-to-End Workflow

1. **Synthesize Bundle**:
   `rvllm_ptx_tuner` (or `vast_ptx_tuner.py --mode dry-run`) runs the `1,000D x 100W` search and writes `vast_work_units.json` along with `kernels/<arch>/<workload>.ptx` and `kernels/<arch>/<workload>.cu`.
2. **Provision Ephemeral GPU Node**:
   `scripts/vast_ptx_tuner.py dispatch` queries Vast.ai offers matching the target architecture (`vast_offer_query`), rents a single-GPU container with label `rvllm-ptx-tuner-<tag>-<unit_id>`, and records the lease before SSH handshake.
3. **Compile & Verify on Silicon (`scripts/remote_gpu_eval.sh`)**:
   - Compiles `kernel.cu` with `nvcc -O3 -arch=<ptx_target> -Xptxas -v` and inspects `ptxas.log` to assert **`0 bytes spill stores, 0 bytes spill loads`**.
   - Executes the generated binary against an FP32 CPU reference to verify **`cosine >= 0.9999`** and **`max_rel_err <= 1e-3`**.
   - Records hardware `cudaEventElapsedTime` median/min/max latency across timed repeats.
4. **Automatic Instance Destruction**:
   Every rented instance is destroyed immediately in a `finally:` block, backed by `atexit` and `SIGINT`/`SIGTERM`/`SIGHUP` signal traps.

## 2. Commands

```bash
# 1. Generate a 1,000D x 100W bundle and preview Vast.ai offers ($0 spend):
python3 scripts/vast_ptx_tuner.py --mode dry-run --depth 1000 --width 100 \
  --emit-bundle ./artifacts/vast-ptx-tuner

# 2. Dry-run the dispatcher against an existing manifest ($0 spend):
python3 scripts/vast_ptx_tuner.py dispatch \
  --manifest ./artifacts/vast-ptx-tuner/vast_work_units.json \
  --out ./artifacts/vast-ptx-tuner/results --dry-run

# 3. Run a live single-unit pilot on Vast.ai:
python3 scripts/vast_ptx_tuner.py dispatch \
  --manifest ./artifacts/vast-ptx-tuner/vast_work_units.json \
  --out ./artifacts/vast-ptx-tuner/results --pilot

# 4. Emergency scoped teardown of any active rvllm-ptx-tuner leases:
python3 scripts/vast_ptx_tuner.py teardown
```
