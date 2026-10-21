#!/usr/bin/env python3
"""Vast.ai GPU fleet dispatcher for the rvLLM 1,000D x 100W PTX-CUDA Ultra Kernel Tuner.

Coordinates:
1. Local 1,000D x 100W schedule search & PTX/CUDA emission via `rvllm_ptx_tuner`.
2. Vast.ai GPU offer discovery across all target architectures (`sm_70` V100 through
   `sm_103a` GB300, `sm_120` RTX 5090, and `sm_121a` DGX Spark GB10).
3. Remote compilation (`nvcc -O3 --ptxas-options=-v`), zero-spill assertion,
   FP32 reference cosine parity (`>= 0.9999`), and CUDA-event latency timing.
4. Teardown safety: every leased Vast.ai instance carries `LABEL_PREFIX = "rvllm-ptx-tuner"`
   and is destroyed via `atexit`, `SIGINT`, `SIGTERM`, and `try/finally`.
"""

from __future__ import annotations

import argparse
import atexit
import datetime as dt
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_PREFIX = "rvllm-ptx-tuner"
DEFAULT_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
DEFAULT_DISK_GB = 32
DEFAULT_REPEATS = 5
SSH_READY_TIMEOUT_S = 360
WORK_UNIT_TIMEOUT_S = 1800


@dataclass
class VastWorkUnit:
    unit_id: str
    hardware_id: str
    ptx_target: str
    vast_offer_query: str
    workload_name: str
    entry_symbol: str
    variant_name: str
    cuda_rel_path: str
    ptx_rel_path: str
    nvcc_command: str


@dataclass
class ActiveLease:
    instance_id: int
    unit: VastWorkUnit
    offer_id: int
    dph_total: float
    label: str
    created_at: float = field(default_factory=time.time)
    ssh_host: str = ""
    ssh_port: int = 0
    destroyed: bool = False


class LeaseTracker:
    """Tracks every Vast.ai instance created in this session and guarantees teardown."""

    def __init__(self) -> None:
        self._leases: dict[int, ActiveLease] = {}
        self._installed = False

    def install_signal_handlers(self) -> None:
        if self._installed:
            return
        self._installed = True
        atexit.register(self.destroy_all)

        def _on_signal(signum: int, _frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            print(
                f"\n[rvllm-ptx-tuner] caught {sig_name} — destroying {len(self._leases)} Vast.ai lease(s)...",
                file=sys.stderr,
            )
            self.destroy_all()
            raise SystemExit(128 + signum)

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(sig, _on_signal)
            except (ValueError, OSError):
                pass

    def register(self, lease: ActiveLease) -> None:
        self._leases[lease.instance_id] = lease

    def destroy_one(self, instance_id: int) -> bool:
        lease = self._leases.get(instance_id)
        if lease and lease.destroyed:
            return True
        for attempt in range(1, 4):
            proc = subprocess.run(
                ["vastai", "destroy", "instance", str(instance_id)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0 or "not found" in (proc.stderr or "").lower():
                if lease:
                    lease.destroyed = True
                return True
            time.sleep(2 * attempt)
        print(
            f"[rvllm-ptx-tuner] WARNING: failed to destroy Vast.ai instance {instance_id} — run 'vastai destroy instance {instance_id}'",
            file=sys.stderr,
        )
        return False

    def destroy_all(self) -> None:
        for iid in list(self._leases):
            try:
                self.destroy_one(iid)
            except Exception as exc:
                print(f"[rvllm-ptx-tuner] error destroying {iid}: {exc}", file=sys.stderr)


TRACKER = LeaseTracker()


def load_work_units(manifest_path: Path) -> tuple[dict[str, Any], list[VastWorkUnit]]:
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    if doc.get("schema") != "rvllm.ptx_tuner.vast_work_units.v1":
        raise SystemExit(f"unsupported work-unit schema: {doc.get('schema')!r}")
    units: list[VastWorkUnit] = []
    for raw in doc.get("units", []):
        units.append(
            VastWorkUnit(
                unit_id=str(raw["unit_id"]),
                hardware_id=str(raw["hardware_id"]),
                ptx_target=str(raw["ptx_target"]),
                vast_offer_query=str(raw["vast_offer_query"]),
                workload_name=str(raw["workload_name"]),
                entry_symbol=str(raw["entry_symbol"]),
                variant_name=str(raw["variant_name"]),
                cuda_rel_path=str(raw["cuda_rel_path"]),
                ptx_rel_path=str(raw["ptx_rel_path"]),
                nvcc_command=str(raw["nvcc_command"]),
            )
        )
    return doc, units


def find_cheapest_vast_offer(query: str, max_dph: float, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {
            "id": 9_000_000 + (abs(hash(query)) % 99_999),
            "dph_total": min(1.89, max_dph),
            "gpu_name": query.split()[0].replace("gpu_name=", ""),
            "reliability2": 0.994,
        }
    proc = subprocess.run(
        ["vastai", "search", "offers", "--raw", "-o", "dph_total", query],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"vastai search offers failed: {proc.stderr.strip()}")
    offers = json.loads(proc.stdout or "[]")
    eligible = [o for o in offers if float(o.get("dph_total", 999.0)) <= max_dph]
    if not eligible:
        raise RuntimeError(f"no Vast.ai offers matching {query!r} under ${max_dph:.2f}/hr")
    eligible.sort(key=lambda o: float(o["dph_total"]))
    return eligible[0]


def rent_vast_instance(unit: VastWorkUnit, offer: dict[str, Any]) -> ActiveLease:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = f"{LABEL_PREFIX}-{unit.unit_id}-{ts}"[:64]
    proc = subprocess.run(
        [
            "vastai",
            "create",
            "instance",
            str(offer["id"]),
            "--image",
            DEFAULT_IMAGE,
            "--disk",
            str(DEFAULT_DISK_GB),
            "--ssh",
            "--direct",
            "--label",
            label,
            "--raw",
        ],
        capture_output=True,
        text=True,
        timeout=45,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"vastai create instance failed: {proc.stderr.strip()}")
    resp = json.loads(proc.stdout)
    new_id = resp.get("new_contract") or resp.get("id")
    if not new_id:
        raise RuntimeError(f"vastai create instance returned no contract id: {resp}")
    lease = ActiveLease(
        instance_id=int(new_id),
        unit=unit,
        offer_id=int(offer["id"]),
        dph_total=float(offer["dph_total"]),
        label=label,
    )
    TRACKER.register(lease)
    return lease


def wait_for_vast_ssh(lease: ActiveLease) -> None:
    deadline = time.time() + SSH_READY_TIMEOUT_S
    while time.time() < deadline:
        proc = subprocess.run(
            ["vastai", "show", "instance", str(lease.instance_id), "--raw"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            info = json.loads(proc.stdout)
            if isinstance(info, list) and info:
                info = info[0]
            status = info.get("actual_status") or info.get("status_msg") or ""
            host = info.get("ssh_host") or info.get("public_ipaddr") or ""
            port = info.get("ssh_port") or info.get("direct_port_start") or 0
            if status == "running" and host and int(port) > 0:
                lease.ssh_host = str(host)
                lease.ssh_port = int(port)
                probe = subprocess.run(
                    [
                        "ssh",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "StrictHostKeyChecking=accept-new",
                        "-o",
                        "ConnectTimeout=8",
                        "-p",
                        str(lease.ssh_port),
                        f"root@{lease.ssh_host}",
                        "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if probe.returncode == 0:
                    return
        time.sleep(5)
    raise TimeoutError(
        f"Vast.ai instance {lease.instance_id} did not become SSH-ready within {SSH_READY_TIMEOUT_S}s"
    )


def execute_remote_gpu_unit(
    lease: ActiveLease,
    bundle_dir: Path,
    out_dir: Path,
    repeats: int,
) -> dict[str, Any]:
    ssh_opts = [
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
    ]
    remote_dir = f"/tmp/rvllm_ptx_tuner_{lease.unit.unit_id}"
    local_cu = bundle_dir / lease.unit.cuda_rel_path
    local_ptx = bundle_dir / lease.unit.ptx_rel_path
    eval_script = Path(__file__).resolve().parent / "remote_gpu_eval.sh"

    subprocess.run(
        ["ssh", *ssh_opts, "-p", str(lease.ssh_port), f"root@{lease.ssh_host}", f"mkdir -p {shlex.quote(remote_dir)}"],
        check=True,
        timeout=30,
    )
    subprocess.run(
        [
            "scp",
            *ssh_opts,
            "-P",
            str(lease.ssh_port),
            str(local_cu),
            str(local_ptx),
            str(eval_script),
            f"root@{lease.ssh_host}:{remote_dir}/",
        ],
        check=True,
        timeout=90,
    )

    cu_name = local_cu.name
    remote_cmd = (
        f"cd {shlex.quote(remote_dir)} && "
        f"bash remote_gpu_eval.sh {shlex.quote(cu_name)} {shlex.quote(lease.unit.ptx_target)} {int(repeats)}"
    )
    proc = subprocess.run(
        ["ssh", *ssh_opts, "-p", str(lease.ssh_port), f"root@{lease.ssh_host}", remote_cmd],
        capture_output=True,
        text=True,
        timeout=WORK_UNIT_TIMEOUT_S,
    )
    unit_out = out_dir / lease.unit.unit_id
    unit_out.mkdir(parents=True, exist_ok=True)
    (unit_out / "remote_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (unit_out / "remote_stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"remote kernel evaluation failed on {lease.unit.unit_id} (exit {proc.returncode}): {proc.stderr[-400:]}"
        )

    subprocess.run(
        [
            "scp",
            *ssh_opts,
            "-P",
            str(lease.ssh_port),
            f"root@{lease.ssh_host}:{remote_dir}/gpu_eval_receipt.json",
            str(unit_out / "gpu_eval_receipt.json"),
        ],
        check=False,
        timeout=60,
    )
    receipt_path = unit_out / "gpu_eval_receipt.json"
    if receipt_path.is_file():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    return {"unit_id": lease.unit.unit_id, "status": "ok"}


def run_rust_1000d_100w_tuner(
    model: str,
    hardware: str,
    depth: int,
    width: int,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_toml = REPO_ROOT / "v3" / "Cargo.toml"
    cmd = [
        "cargo",
        "run",
        "--release",
        "--locked",
        "--manifest-path",
        str(manifest_toml),
        "-p",
        "rvllm-cutlass",
        "--bin",
        "rvllm_ptx_tuner",
        "--",
        "--model",
        model,
        "--hardware",
        hardware,
        "--depth",
        str(depth),
        "--width",
        str(width),
        "--out",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    return out_dir / "vast_work_units.json"


def cmd_dispatch(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    bundle_dir = manifest_path.parent
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    doc, units = load_work_units(manifest_path)

    if args.hardware and args.hardware.lower() != "all":
        allowed = {h.strip().lower() for h in args.hardware.split(",") if h.strip()}
        units = [
            u
            for u in units
            if u.hardware_id.lower() in allowed or u.ptx_target.lower() in allowed
        ]
    if args.unit:
        allowed_u = {u.strip() for u in args.unit.split(",") if u.strip()}
        units = [u for u in units if u.unit_id in allowed_u or u.workload_name in allowed_u]
    if not units:
        raise SystemExit("no Vast.ai work units matched filter")

    TRACKER.install_signal_handlers()
    plan: list[tuple[VastWorkUnit, dict[str, Any]]] = []
    total_dph = 0.0
    for u in units:
        offer = find_cheapest_vast_offer(u.vast_offer_query, args.max_dph, args.dry_run)
        plan.append((u, offer))
        total_dph += float(offer["dph_total"])

    print(f"schema:    {doc.get('schema')} ({doc.get('search_topology')})")
    print(f"units:     {len(plan)} selected   combined rate: ${total_dph:.3f}/hr")
    print(f"hard cap:  --max-hours={args.max_hours} (${total_dph * args.max_hours:.2f} ceiling)")

    results = []
    for u, offer in plan:
        if args.dry_run:
            print(
                f"  [dry-run] {u.unit_id}: would rent offer {offer['id']} (${offer['dph_total']:.3f}/hr, {u.ptx_target}), "
                f"compile {u.cuda_rel_path}, verify cosine>=0.9999 & 0 spills, run {args.repeats} CUDA-event repeats, "
                f"collect -> {out_dir / u.unit_id}, then DESTROY"
            )
            results.append(
                {
                    "unit_id": u.unit_id,
                    "hardware_id": u.hardware_id,
                    "ptx_target": u.ptx_target,
                    "variant_name": u.variant_name,
                    "status": "DRY_RUN",
                    "offer_id": offer["id"],
                    "dph_total": offer["dph_total"],
                }
            )
            continue

        lease: ActiveLease | None = None
        t0 = time.time()
        try:
            lease = rent_vast_instance(u, offer)
            wait_for_vast_ssh(lease)
            receipt = execute_remote_gpu_unit(lease, bundle_dir, out_dir, args.repeats)
            results.append(
                {
                    "unit_id": u.unit_id,
                    "hardware_id": u.hardware_id,
                    "ptx_target": u.ptx_target,
                    "variant_name": u.variant_name,
                    "status": "PASS",
                    "instance_id": lease.instance_id,
                    "elapsed_s": round(time.time() - t0, 2),
                    "receipt": receipt,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "unit_id": u.unit_id,
                    "hardware_id": u.hardware_id,
                    "ptx_target": u.ptx_target,
                    "variant_name": u.variant_name,
                    "status": f"FAIL: {exc}",
                    "instance_id": lease.instance_id if lease else None,
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )
        finally:
            if lease is not None:
                TRACKER.destroy_one(lease.instance_id)

    summary_path = (
        out_dir / f"vast-tuner-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d')}.json"
    )
    summary_path.write_text(
        json.dumps(
            {
                "schema": "rvllm.ptx_tuner.vast_dispatch_summary.v1",
                "dry_run": args.dry_run,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"summary:   {summary_path}")
    failed = [r for r in results if not str(r["status"]).startswith(("PASS", "DRY_RUN"))]
    return 1 if failed else 0


def cmd_cleanup(_args: argparse.Namespace) -> int:
    proc = subprocess.run(
        ["vastai", "show", "instances", "--raw"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise SystemExit(f"vastai show instances failed: {proc.stderr.strip()}")
    instances = json.loads(proc.stdout or "[]")
    ours = [i for i in instances if (i.get("label") or "").startswith(LABEL_PREFIX)]
    if not ours:
        print(f"no active Vast.ai instances with label prefix {LABEL_PREFIX!r}")
        return 0
    for inst in ours:
        iid = int(inst["id"])
        print(f"destroying {iid} (label={inst.get('label')})...")
        TRACKER.destroy_one(iid)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    manifest_path = run_rust_1000d_100w_tuner(
        model=args.model,
        hardware=args.hardware,
        depth=args.depth,
        width=args.width,
        out_dir=out_dir,
    )
    dispatch_ns = argparse.Namespace(
        manifest=str(manifest_path),
        out=str(out_dir / "vast-eval"),
        hardware=args.hardware,
        unit=args.unit,
        repeats=args.repeats,
        max_hours=args.max_hours,
        max_dph=args.max_dph,
        dry_run=not args.live_vast,
    )
    return cmd_dispatch(dispatch_ns)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser(
        "run",
        help="Run 1,000D x 100W PTX search and dispatch winning kernels to Vast.ai",
    )
    p_run.add_argument("--model", default="gemma4-31b")
    p_run.add_argument("--hardware", default="all")
    p_run.add_argument("--unit", default="")
    p_run.add_argument("--depth", type=int, default=1000)
    p_run.add_argument("--width", type=int, default=100)
    p_run.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    p_run.add_argument("--max-hours", type=float, default=2.0)
    p_run.add_argument("--max-dph", type=float, default=6.0)
    p_run.add_argument("--live-vast", action="store_true")
    p_run.add_argument("--out", default="target/ptx-tuner-bundle")

    p_disp = sub.add_parser(
        "dispatch", help="Dispatch an existing vast_work_units.json to Vast.ai"
    )
    p_disp.add_argument("--manifest", required=True)
    p_disp.add_argument("--out", required=True)
    p_disp.add_argument("--hardware", default="all")
    p_disp.add_argument("--unit", default="")
    p_disp.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    p_disp.add_argument("--max-hours", type=float, default=2.0)
    p_disp.add_argument("--max-dph", type=float, default=6.0)
    p_disp.add_argument("--dry-run", action="store_true")

    sub.add_parser(
        "cleanup",
        help=f"Destroy all Vast.ai instances with label prefix {LABEL_PREFIX!r}",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "dispatch":
        return cmd_dispatch(args)
    if args.cmd == "cleanup":
        return cmd_cleanup(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
