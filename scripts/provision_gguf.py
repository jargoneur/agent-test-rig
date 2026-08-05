#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import yaml
from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
LLAMA_CPP = ROOT / ".upstreams" / "llama.cpp"
DEFAULT_REGISTRY = ROOT / "model_artifacts" / "registry.yml"


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
    args = parser.parse_args()

    revision = args.revision.strip()
    if len(revision) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in revision):
        raise ValueError("--revision must be an exact 40-character commit SHA")

    api = HfApi(token=args.token)
    info = api.model_info(
        repo_id=args.model_id,
        revision=revision,
        token=args.token,
        files_metadata=False,
    )
    resolved_revision = str(info.sha)
    if resolved_revision.lower() != revision.lower():
        raise RuntimeError(
            "Revision mismatch: requested %s, resolved %s"
            % (revision, resolved_revision)
        )

    output_root = Path(args.output_root).expanduser().resolve()
    model_dir = output_root / args.profile_key
    model_dir.mkdir(parents=True, exist_ok=True)
    final_path = model_dir / (args.profile_key + "-Q8_0.gguf")

    if args.existing_gguf:
        source = Path(args.existing_gguf).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if source != final_path:
            shutil.copy2(source, final_path)
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

        snapshot = model_dir / "hf_snapshot"
        snapshot_download(
            repo_id=args.model_id,
            revision=revision,
            local_dir=str(snapshot),
            token=args.token,
        )
        f16_path = model_dir / (args.profile_key + "-F16.gguf")
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
        run([str(quantizer), str(f16_path), str(final_path), "Q8_0"])
        f16_path.unlink(missing_ok=True)

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
        "schema_version": 1,
        "profile_key": args.profile_key,
        "upstream_model_id": args.model_id,
        "upstream_revision": resolved_revision,
        "artifact": str(final_path),
        "artifact_sha256": digest,
        "quantization": "Q8_0",
        "llama_cpp_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=LLAMA_CPP, text=True
        ).strip(),
    }
    (model_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Provisioned:", args.profile_key)
    print("Artifact:", final_path)
    print("SHA-256:", digest)
    print("Registry:", registry_path)


if __name__ == "__main__":
    main()
