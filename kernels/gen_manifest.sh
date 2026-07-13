#!/bin/bash
# Generate manifest.json for a kernel output directory.
#
# Usage: ./gen_manifest.sh <arch_dir> <revision> <ptx_abi> <new_artifact>...
#   arch_dir: e.g. kernels/sm_121/ — must contain *.ptx and/or *.so files
#   revision: 7-64 character hexadecimal git commit ID
#   new_artifact: basename of an artifact produced by the current build
#
# Emits <arch_dir>/manifest.json with:
#   { revision, arch, entries: { <name>: { path, sha256, bytes, kind, abi } } }
#
# The logical name is the PTX file stem (e.g. fp8_gemv.ptx → "fp8_gemv").
# This matches how `KernelLoader::load_ptx(name)` looks up artifacts.

set -euo pipefail

ARCH_DIR="${1:?usage: $0 <arch_dir> <revision> <ptx_abi> <new_artifact>...}"
REVISION="${2:?usage: $0 <arch_dir> <revision> <ptx_abi> <new_artifact>...}"
ABI="${3:?usage: $0 <arch_dir> <revision> <ptx_abi> <new_artifact>...}"
shift 3
[ "$#" -gt 0 ] || {
    echo "gen_manifest: declare at least one artifact produced by this build" >&2
    exit 1
}

if [ ! -d "$ARCH_DIR" ]; then
    echo "gen_manifest: $ARCH_DIR is not a directory" >&2
    exit 1
fi

ARCH_NAME="$(basename "$ARCH_DIR")"
OUT="$ARCH_DIR/manifest.json"

python3 - "$ARCH_DIR" "$ARCH_NAME" "$REVISION" "$ABI" "$OUT" "$@" <<'PY'
import hashlib, json, os, re, stat, sys, tempfile
arch_dir, arch_name, revision, abi, out = sys.argv[1:6]
new_artifacts = sys.argv[6:]
if arch_name not in {"sm_80", "sm_89", "sm_90", "sm_100", "sm_121"}:
    raise SystemExit(f"gen_manifest: unsupported architecture {arch_name!r}")
if not 7 <= len(revision) <= 64 or any(c not in "0123456789abcdef" for c in revision):
    raise SystemExit("gen_manifest: revision must be a lowercase 7-64 character hexadecimal commit ID")
if abi != "cuda-ptx-v1":
    raise SystemExit("gen_manifest: ABI must be cuda-ptx-v1")
shared_libraries = {
    "libcutlass_kernels.so",
    "libcutlass_sm120.so",
    "libfa3_kernels.so",
    "libfa_sm89_kernels.so",
    "libw4a8_gemm.so",
}
safe_file = re.compile(r"^[A-Za-z0-9_.-]+$")
if len(new_artifacts) != len(set(new_artifacts)):
    raise SystemExit("gen_manifest: duplicate new artifact argument")

def contract(fn):
    if not isinstance(fn, str) or not safe_file.fullmatch(fn) or fn in {".", ".."}:
        raise SystemExit(f"gen_manifest: artifact must be a safe basename: {fn!r}")
    if fn.endswith(".ptx"):
        name = fn[:-len(".ptx")]
        kind = "ptx"
        artifact_abi = abi
    elif fn.endswith(".so"):
        if fn not in shared_libraries:
            raise SystemExit(f"gen_manifest: unexpected shared library {fn!r}")
        name = fn
        kind = "shared_object"
        artifact_abi = (
            "rvllm-cuda-so-v2"
            if fn in {"libfa3_kernels.so", "libfa_sm89_kernels.so"}
            else "rvllm-cuda-so-v1"
        )
    else:
        raise SystemExit(f"gen_manifest: unsupported artifact {fn!r}")
    if not 1 <= len(name) <= 128:
        raise SystemExit(f"gen_manifest: artifact name exceeds the runtime limit: {name!r}")
    return name, kind, artifact_abi

def hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def inspect(fn):
    name, kind, artifact_abi = contract(fn)
    path = os.path.join(arch_dir, fn)
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(f"gen_manifest: artifact must be a regular file: {fn}")
    if not 0 < metadata.st_size <= 512 * 1024 * 1024:
        raise SystemExit(f"gen_manifest: invalid artifact size for {fn}")
    return name, {
        "path": fn,
        "sha256": hash_file(path),
        "bytes": metadata.st_size,
        "kind": kind,
        "abi": artifact_abi,
    }

recognized = sorted(
    fn for fn in os.listdir(arch_dir)
    if fn.endswith(".ptx") or fn.endswith(".so")
)
for fn in recognized:
    contract(fn)
for fn in new_artifacts:
    if fn not in recognized:
        raise SystemExit(f"gen_manifest: declared artifact is missing: {fn}")

entries = {}
old_paths = set()
digest_path = os.path.join(arch_dir, "manifest.sha256")
if os.path.exists(out):
    if not os.path.isfile(out) or os.path.islink(out):
        raise SystemExit("gen_manifest: existing manifest must be a regular file")
    old_body = open(out, "rb").read(1024 * 1024 + 1)
    if len(old_body) > 1024 * 1024:
        raise SystemExit("gen_manifest: existing manifest exceeds the runtime limit")
    if not os.path.isfile(digest_path) or os.path.islink(digest_path):
        raise SystemExit("gen_manifest: existing manifest is missing its digest sidecar")
    expected_sidecar = f"{hashlib.sha256(old_body).hexdigest()}  manifest.json\n"
    with open(digest_path, encoding="ascii") as handle:
        if handle.read() != expected_sidecar:
            raise SystemExit("gen_manifest: existing manifest digest sidecar does not match")
    try:
        old = json.loads(old_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"gen_manifest: existing manifest is invalid: {error}") from error
    if set(old) != {"revision", "arch", "entries"}:
        raise SystemExit("gen_manifest: existing manifest has an incompatible schema")
    if old["revision"] != revision or old["arch"] != arch_name:
        raise SystemExit("gen_manifest: existing manifest has a different revision or architecture; use a clean staging directory")
    if not isinstance(old["entries"], dict):
        raise SystemExit("gen_manifest: existing manifest entries are invalid")
    for name, entry in old["entries"].items():
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "bytes", "kind", "abi"}:
            raise SystemExit(f"gen_manifest: existing entry has an incompatible schema: {name!r}")
        fn = entry["path"]
        if fn in old_paths:
            raise SystemExit(f"gen_manifest: duplicate existing artifact path: {fn!r}")
        old_paths.add(fn)
        expected_name, expected_kind, expected_abi = contract(fn)
        if name != expected_name or entry["kind"] != expected_kind or entry["abi"] != expected_abi:
            raise SystemExit(f"gen_manifest: existing entry contract mismatch: {name!r}")
        if fn not in new_artifacts:
            actual_name, actual = inspect(fn)
            if actual_name != name or actual != entry:
                raise SystemExit(f"gen_manifest: existing artifact no longer matches its manifest: {fn}")
            entries[name] = entry
elif os.path.exists(digest_path):
    raise SystemExit("gen_manifest: digest sidecar exists without manifest.json; use a clean staging directory")

permitted = old_paths | set(new_artifacts)
unexpected = sorted(set(recognized) - permitted)
if unexpected:
    raise SystemExit(
        "gen_manifest: undeclared artifacts in staging directory: " + ", ".join(unexpected)
    )
for fn in new_artifacts:
    name, entry = inspect(fn)
    entries[name] = entry

if not entries or len(entries) > 4096:
    raise SystemExit("gen_manifest: artifact count must be in 1..4096")
if sum(entry["bytes"] for entry in entries.values()) > 4 * 512 * 1024 * 1024:
    raise SystemExit("gen_manifest: total artifact bytes exceed the runtime limit")
doc = {"revision": revision, "arch": arch_name, "entries": entries}
body = (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode()
if len(body) > 1024 * 1024:
    raise SystemExit("gen_manifest: manifest exceeds the runtime size limit")
with tempfile.NamedTemporaryFile(dir=arch_dir, prefix=".manifest.", delete=False) as handle:
    manifest_tmp = handle.name
    handle.write(body)
os.replace(manifest_tmp, out)
digest = hashlib.sha256(body).hexdigest()
with tempfile.NamedTemporaryFile(
    mode="w", dir=arch_dir, prefix=".manifest-sha256.", delete=False
) as handle:
    digest_tmp = handle.name
    handle.write(f"{digest}  {os.path.basename(out)}\n")
os.replace(digest_tmp, digest_path)
print(f"  manifest: {out} ({len(entries)} entries)")
print(f"  RVLLM_KERNEL_MANIFEST_SHA256={digest}")
print(f"  RVLLM_RELEASE_REVISION={revision}")
print(f"  RVLLM_KERNEL_ARCH={arch_name}")
PY
