#!/usr/bin/env python3
"""Unit tests for scripts/vast_ptx_tuner.py."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import vast_ptx_tuner  # noqa: E402


class VastPtxTunerTests(unittest.TestCase):
    def test_dry_run_dispatch_and_manifest_parsing(self) -> None:
        scratch_dir = os.environ.get("SOLIDSF_SCRATCH")
        if scratch_dir:
            pathlib.Path(scratch_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="rvllm-vast-test-", dir=scratch_dir or None
        ) as td:
            root = pathlib.Path(td)
            manifest = root / "vast_work_units.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "rvllm.ptx_tuner.vast_work_units.v1",
                        "model_id": "gemma4-31b",
                        "search_topology": "1000D x 100W",
                        "units": [
                            {
                                "unit_id": "sm_90a__qkv_gemv",
                                "hardware_id": "sm_90a",
                                "ptx_target": "sm_90a",
                                "vast_offer_query": "gpu_name=H100_SXM num_gpus=1",
                                "workload_name": "qkv_gemv",
                                "entry_symbol": "rvllm_qkv_gemv_sm_90a",
                                "variant_name": "wpr_1x8x1024_s3_sk1_v16_e4m3_f16in",
                                "cuda_rel_path": "kernels/sm_90a/qkv_gemv.cu",
                                "ptx_rel_path": "kernels/sm_90a/qkv_gemv.ptx",
                                "nvcc_command": "nvcc -O3 -arch=sm_90a kernels/sm_90a/qkv_gemv.cu",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            out_dir = root / "out"
            rc = vast_ptx_tuner.main(
                [
                    "dispatch",
                    "--manifest",
                    str(manifest),
                    "--out",
                    str(out_dir),
                    "--dry-run",
                ]
            )
            self.assertEqual(rc, 0)
            summaries = list(out_dir.glob("vast-tuner-*.json"))
            self.assertEqual(len(summaries), 1)
            payload = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["results"][0]["status"], "DRY_RUN")

    def test_lease_tracker_destroys_registered_instances(self) -> None:
        tracker = vast_ptx_tuner.LeaseTracker()
        unit = vast_ptx_tuner.VastWorkUnit(
            unit_id="sm_120__gemm",
            hardware_id="sm_120",
            ptx_target="sm_120",
            vast_offer_query="gpu_name=RTX_5090 num_gpus=1",
            workload_name="gemm",
            entry_symbol="rvllm_gemm",
            variant_name="wmma_128x128x64",
            cuda_rel_path="kernels/sm_120/gemm.cu",
            ptx_rel_path="kernels/sm_120/gemm.ptx",
            nvcc_command="nvcc",
        )
        lease = vast_ptx_tuner.ActiveLease(
            instance_id=424242,
            unit=unit,
            offer_id=999,
            dph_total=0.85,
            label="rvllm-ptx-tuner-test",
        )
        tracker.register(lease)
        with mock.patch("subprocess.run") as m_run:
            m_run.return_value.returncode = 0
            m_run.return_value.stderr = ""
            tracker.destroy_all()
            self.assertTrue(lease.destroyed)
            m_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
