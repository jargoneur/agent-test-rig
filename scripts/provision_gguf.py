#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
LLAMA_CPP = ROOT / ".upstreams" / "llama.cpp"
DEFAULT_REGISTRY = ROOT / "model_artifacts" / "registry.yml"
GIB = 1024 ** 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "models": {}}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Registry must be a YAML mapping")
    value.setdefault("schema_version", 1)
    value.setdefault("models", {})
    if not isinstance(value["models"], dict):
        raise ValueError("Registry models must be a mapping")
    return value


def run(command, cwd=None):
    print("+", " ".join(str(value) for value in command), flush=True)
    subprocess.run([str(value) for value in command], cwd=cwd, check=True)


def repository_size_bytes(info: Any) -> Optional[int]:
    sizes = []
    for sibling in getattr(info, "siblings", None) or []:
        size = getattr(sibling, "size", None)
        if isinstance(size, int) and size >= 0:
            sizes.append(size)
    if not sizes:
        return None
    return sum(sizes)


def source_weight_manifest(info: Any) -> Dict[str, Any]:
    files = []
    for sibling in getattr(info, "siblings", None) or []:
        name = str(getattr(sibling, "rfilename", "") or "")
        if not name.lower().endswith((".safetensors", ".bin", ".pt")):
            continue
        lfs = getattr(sibling, "lfs", None)
        digest = (
            getattr(lfs, "sha256", None)
            if lfs is not None and not isinstance(lfs, dict)
            else (lfs or {}).get("sha256")
        )
        if not digest:
            raise RuntimeError("Source weight has no immutable LFS hash: %s" % name)
        files.append(
            {
                "path": name,
                "size": int(getattr(sibling, "size", 0) or 0),
                "sha256": str(digest),
            }
        )
    if not files:
        raise RuntimeError("Model repository reports no source weight files")
    files.sort(key=lambda item: item["path"])
    payload = json.dumps(
        files, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "files": files,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def check_storage(
    path: Path,
    estimated_required_bytes: Optional[int],
    reserve_gib: float,
    estimate_rule: str,
) -> Dict[str, Any]:
    usage = shutil.disk_usage(path)
    reserve_bytes = max(0, int(reserve_gib * GIB))
    report = {
        "filesystem_free_gib": round(usage.free / GIB, 3),
        "storage_reserve_gib": reserve_gib,
        "estimated_required_gib": (
            round(estimated_required_bytes / GIB, 3)
            if estimated_required_bytes is not None
            else None
        ),
        "path": str(path),
        "estimate_rule": estimate_rule,
    }
    if estimated_required_bytes is not None:
        required_with_reserve = estimated_required_bytes + reserve_bytes
        if usage.free < required_with_reserve:
            raise RuntimeError(
                "Insufficient free storage: %.3f GiB available, approximately %.3f GiB "
                "required plus %.3f GiB reserve"
                % (
                    usage.free / GIB,
                    estimated_required_bytes / GIB,
                    reserve_gib,
                )
            )
    elif usage.free < reserve_bytes:
        raise RuntimeError(
            "Filesystem free space %.3f GiB is below the %.3f GiB reserve"
            % (usage.free / GIB, reserve_gib)
        )
    print("Storage preflight:", json.dumps(report, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision one pinned Hugging Face checkpoint as a Q8_0 GGUF."
    )
    parser.add_argument("--profile-key", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--revision",
        required=True,
        help="Exact Hugging Face commit SHA; symbolic branches are rejected",
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output-root", default=str(ROOT / "model_artifacts"))
    parser.add_argument(
        "--work-root",
        help=(
            "Scratch root for the downloaded snapshot, F16 conversion and Q8_0 "
            "quantizer output. Defaults to --output-root."
        ),
    )
    parser.add_argument("--context-size", type=int, default=32768)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--fit-target-mib", type=int, default=1536)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--existing-gguf",
        help="Register an already-created Q8_0 GGUF instead of downloading/converting",
    )
    parser.add_argument(
        "--storage-reserve-gib",
        type=float,
        default=5.0,
        help="Free storage that must remain beyond the estimated conversion peak.",
    )
    parser.add_argument(
        "--skip-storage-check",
        action="store_true",
        help="Skip the conservative free-space estimate.",
    )
    parser.add_argument(
        "--keep-source-snapshot",
        action="store_true",
        help="Retain the downloaded Hugging Face snapshot after successful conversion.",
    )
    parser.add_argument(
        "--keep-failed-intermediates",
        action="store_true",
        help="Retain downloaded/F16 intermediates after a failed conversion.",
    )
    args = parser.parse_args()

    revision = args.revision.strip()
    if len(revision) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in revision):
        raise ValueError("--revision must be an exact 40-character commit SHA")

    api = HfApi(token=args.token)
    info = api.model_info(
        repo_id=args.model_id,
        revision=revision,
        token=args.token,
        files_metadata=True,
    )
    resolved_revision = str(info.sha)
    if resolved_revision.lower() != revision.lower():
        raise RuntimeError(
            "Revision mismatch: requested %s, resolved %s"
            % (revision, resolved_revision)
        )
    source_weights = source_weight_manifest(info)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir = output_root / args.profile_key
    model_dir.mkdir(parents=True, exist_ok=True)
    final_path = model_dir / (args.profile_key + "-Q8_0.gguf")
    partial_final = final_path.with_suffix(final_path.suffix + ".partial")
    partial_final.unlink(missing_ok=True)

    work_root = (
        Path(args.work_root).expanduser().resolve()
        if args.work_root
        else output_root
    )
    work_model_dir = work_root / args.profile_key
    snapshot = work_model_dir / "hf_snapshot"
    f16_path = work_model_dir / (args.profile_key + "-F16.gguf")
    quantized_work_path = work_model_dir / (args.profile_key + "-Q8_0.gguf.partial")
    storage_report: Dict[str, Any] = {}
    conversion_succeeded = False

    if args.existing_gguf:
        source = Path(args.existing_gguf).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if not args.skip_storage_check:
            estimated = source.stat().st_size if source != final_path else 0
            storage_report = check_storage(
                output_root,
                estimated,
                args.storage_reserve_gib,
                "existing_gguf_copy_plus_reserve",
            )
        if source != final_path:
            shutil.copy2(source, partial_final)
            os.replace(str(partial_final), str(final_path))
        conversion_succeeded = True
    else:
        converter = LLAMA_CPP / "convert_hf_to_gguf.py"
        quantizer = LLAMA_CPP / "build" / "bin" / "llama-quantize"
        if not converter.is_file():
            raise FileNotFoundError(
                "Pinned llama.cpp checkout is missing; run scripts/bootstrap.sh"
            )
        if not quantizer.is_file():
            raise FileNotFoundError(
                "llama-quantize is missing; run scripts/bootstrap.sh"
            )

        snapshot_size = repository_size_bytes(info)
        estimated_peak = (
            int(snapshot_size * 2.75) if snapshot_size is not None else None
        )
        work_model_dir.mkdir(parents=True, exist_ok=True)
        quantized_work_path.unlink(missing_ok=True)
        if not args.skip_storage_check:
            work_report = check_storage(
                work_root,
                estimated_peak,
                args.storage_reserve_gib,
                "snapshot_metadata_times_2.75_plus_reserve",
            )
            if work_root == output_root:
                output_report = work_report
            else:
                output_report = check_storage(
                    output_root,
                    snapshot_size,
                    args.storage_reserve_gib,
                    "snapshot_metadata_upper_bound_for_q8_0_plus_reserve",
                )
            storage_report = {
                "work": work_report,
                "durable_output": output_report,
            }

        try:
            snapshot_download(
                repo_id=args.model_id,
                revision=revision,
                local_dir=str(snapshot),
                token=args.token,
            )
            run(
                [
                    str(ROOT / ".venv" / "bin" / "python"),
                    str(converter),
                    str(snapshot),
                    "--outfile",
                    str(f16_path),
                    "--outtype",
                    "f16",
                ]
            )
            run(
                [
                    str(quantizer),
                    str(f16_path),
                    str(quantized_work_path),
                    "Q8_0",
                ]
            )
            if quantized_work_path != partial_final:
                shutil.copy2(quantized_work_path, partial_final)
            os.replace(str(partial_final), str(final_path))
            conversion_succeeded = True
        finally:
            partial_final.unlink(missing_ok=True)
            quantized_work_path.unlink(missing_ok=True)
            if conversion_succeeded or not args.keep_failed_intermediates:
                f16_path.unlink(missing_ok=True)
                if snapshot.exists() and not args.keep_source_snapshot:
                    shutil.rmtree(snapshot)
                if work_model_dir != model_dir:
                    try:
                        work_model_dir.rmdir()
                    except OSError:
                        pass

    if not final_path.is_file():
        raise RuntimeError("Provisioning did not produce the final GGUF: %s" % final_path)

    digest = sha256_file(final_path)
    registry_path = Path(args.registry).expanduser().resolve()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = load_registry(registry_path)
    try:
        artifact_value = str(final_path.relative_to(ROOT))
    except ValueError:
        artifact_value = str(final_path)
    registry["models"][args.profile_key] = {
        "artifact": artifact_value,
        "sha256": digest,
        "alias": args.profile_key,
        "upstream_model_id": args.model_id,
        "upstream_revision": resolved_revision,
        "quantization": "Q8_0",
        "context_size": args.context_size,
        "gpu_layers": "all",
        "parallel": args.parallel,
        "cache_type_k": args.cache_type_k,
        "cache_type_v": args.cache_type_v,
        "flash_attention": "auto",
        "fit": True,
        "fit_target_mib": args.fit_target_mib,
    }
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False),
        encoding="utf-8",
    )

    provenance = {
        "schema_version": 2,
        "profile_key": args.profile_key,
        "upstream_model_id": args.model_id,
        "upstream_revision": resolved_revision,
        "artifact": str(final_path),
        "artifact_sha256": digest,
        "quantization": "Q8_0",
        "source_weight_manifest_sha256": source_weights["sha256"],
        "source_weight_files": source_weights["files"],
        "source_snapshot_retained": bool(args.keep_source_snapshot and snapshot.exists()),
        "storage_preflight": storage_report,
        "llama_cpp_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=LLAMA_CPP, text=True
        ).strip(),
    }
    metadata_dir = registry_path.parent / args.profile_key
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Provisioned:", args.profile_key)
    print("Artifact:", final_path)
    print("SHA-256:", digest)
    print("Registry:", registry_path)
    if not args.keep_source_snapshot:
        print("Source snapshot and F16 intermediate removed after success.")


if __name__ == "__main__":
    main()
