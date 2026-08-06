#!/usr/bin/env python3
"""Fetch and hash the small metadata files for every locked model checkpoint.

This intentionally does not download model weights. Weight and GGUF hashes remain
blocked until the artifact acquisition/conversion stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError as error:  # pragma: no cover - operator guidance
    raise SystemExit(
        "huggingface_hub is required: python -m pip install huggingface_hub"
    ) from error


ALLOW_PATTERNS = [
    "README.md",
    "LICENSE*",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template*",
    "*.jinja",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            yield path


def file_manifest(root: Path) -> List[Dict[str, Any]]:
    records = []
    for path in iter_files(root):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def load_lock(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Model lock must be a YAML mapping")
    models = value.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Model lock contains no models")
    return value


def fetch_one(
    api: HfApi,
    key: str,
    profile: Dict[str, Any],
    output_root: Path,
    token: str | None,
) -> Dict[str, Any]:
    repo_id = str(profile["model_id"])
    requested_revision = str(profile["model_revision"])
    info = api.model_info(
        repo_id=repo_id,
        revision=requested_revision,
        token=token,
        files_metadata=False,
    )
    resolved_revision = str(info.sha)
    if resolved_revision != requested_revision:
        raise RuntimeError(
            "%s resolved to %s, expected %s"
            % (repo_id, resolved_revision, requested_revision)
        )

    snapshot_dir = output_root / key / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=requested_revision,
        allow_patterns=ALLOW_PATTERNS,
        local_dir=str(snapshot_dir),
        token=token,
    )

    files = file_manifest(snapshot_dir)
    if not files:
        raise RuntimeError("No metadata files downloaded for %s" % repo_id)

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "profile_key": key,
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "allow_patterns": ALLOW_PATTERNS,
        "files": files,
    }
    manifest["metadata_manifest_sha256"] = hashlib.sha256(
        canonical_json(manifest)
    ).hexdigest()

    manifest_path = output_root / key / "metadata_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and hash metadata for locked Hugging Face models."
    )
    parser.add_argument(
        "--lock",
        default="experiments/model_profiles.lock.yml",
    )
    parser.add_argument("--output", default="model_metadata")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--token")
    args = parser.parse_args()

    lock = load_lock(Path(args.lock))
    profiles: Dict[str, Dict[str, Any]] = lock["models"]
    selected = set(args.models or profiles.keys())
    unknown = sorted(selected.difference(profiles))
    if unknown:
        raise ValueError("Unknown model profile(s): " + ", ".join(unknown))

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=args.token)

    failures = []
    for key, profile in profiles.items():
        if key not in selected:
            continue
        try:
            manifest = fetch_one(
                api,
                key,
                profile,
                output_root,
                args.token,
            )
            print(
                "%s: %d files, %s"
                % (
                    key,
                    len(manifest["files"]),
                    manifest["metadata_manifest_sha256"],
                )
            )
        except Exception as error:  # continue to report all invalid pins
            failures.append((key, str(error)))
            print("%s: FAILED: %s" % (key, error), file=sys.stderr)

    if failures:
        print("\nMetadata validation failures:", file=sys.stderr)
        for key, error in failures:
            print("- %s: %s" % (key, error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
