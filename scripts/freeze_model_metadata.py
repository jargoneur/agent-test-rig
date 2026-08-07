#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[1]
QWEN_PRECISE_CODING = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": True},
}
GEMMA_STANDARD = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "chat_template_kwargs": {"enable_thinking": True},
}
RESOURCE_CLASSES = {
    "qwen3_5_0_8b": "small",
    "qwen3_5_2b": "small",
    "qwen3_5_4b": "small",
    "qwen3_5_9b": "medium",
    "qwen3_5_27b": "large",
    "qwen3_5_35b_a3b": "large",
    "gemma4_e2b_it": "small",
    "gemma4_e4b_it": "small",
    "gemma4_12b_it": "medium",
    "gemma4_26b_a4b_it": "large",
    "gemma4_31b_it": "large",
}


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON mapping: %s" % path)
    return value


def file_hashes(manifest: Dict[str, Any]) -> Dict[str, str]:
    return {
        str(item["path"]): str(item["sha256"])
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    }


def eos_ids(config: Dict[str, Any], generation: Dict[str, Any]) -> list[int]:
    value = generation.get("eos_token_id")
    if value is None:
        text_config = config.get("text_config") or config
        value = text_config.get("eos_token_id")
    values = value if isinstance(value, list) else [value]
    return [int(item) for item in values if item is not None]


def update_profile(
    key: str,
    profile: Dict[str, Any],
    metadata_root: Path,
) -> None:
    model_root = metadata_root / key
    snapshot = model_root / "snapshot"
    manifest = load_json(model_root / "metadata_manifest.json")
    if manifest.get("repo_id") != profile.get("model_id"):
        raise RuntimeError("%s metadata repository mismatch" % key)
    if manifest.get("resolved_revision") != profile.get("model_revision"):
        raise RuntimeError("%s metadata revision mismatch" % key)

    hashes = file_hashes(manifest)
    config = load_json(snapshot / "config.json")
    generation_path = snapshot / "generation_config.json"
    generation = load_json(generation_path) if generation_path.is_file() else {}
    card = (snapshot / "README.md").read_text(encoding="utf-8")
    text_config = config.get("text_config") or config
    context = int(text_config["max_position_embeddings"])
    experts = text_config.get("num_experts")
    is_moe = bool(text_config.get("enable_moe_block")) or (
        isinstance(experts, int) and experts > 1
    )

    if str(profile.get("family")) == "qwen3.5":
        normalized_card = card.lower()
        if "thinking mode" not in normalized_card or "precise coding" not in normalized_card:
            raise RuntimeError("%s card lacks the precise-coding profile" % key)
        options = dict(QWEN_PRECISE_CODING)
        thinking_mode = "enabled_precise_coding"
        profile_source = "README.md precise-coding thinking recommendation"
    elif str(profile.get("family")) == "gemma4":
        if "standardized sampling configuration across all use cases" not in card:
            raise RuntimeError("%s card lacks standardized sampling guidance" % key)
        options = dict(GEMMA_STANDARD)
        thinking_mode = "enabled"
        profile_source = "README.md best-practices standardized sampling"
    else:
        raise RuntimeError("Unsupported family for %s" % key)

    profile.update(
        {
            "generation_profile_status": "resolved_from_exact_official_model_card",
            "generation_profile_source": profile_source,
            "resolved_generation_options": options,
            "thinking_mode": thinking_mode,
            "input_context_tokens": context,
            "max_new_tokens_per_action": None,
            "metadata_manifest_sha256": manifest["metadata_manifest_sha256"],
            "config_sha256": hashes["config.json"],
            "generation_config_sha256": hashes.get("generation_config.json"),
            "tokenizer_sha256": hashes["tokenizer.json"],
            "chat_template_sha256": hashes["chat_template.jinja"],
            "stop_token_ids": eos_ids(config, generation),
            "architecture": "mixture_of_experts" if is_moe else "dense",
            "resource_class": RESOURCE_CLASSES[key],
            "deployment_profile": {
                "context_size": context,
                "gpu_layers": "all",
                "parallel": 1,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "flash_attention": "auto",
                "fit": True,
                "fit_target_mib": 1536,
                "validation_status": "pending_artifact_measurement",
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze official metadata and generation profiles into the model lock."
    )
    parser.add_argument(
        "--lock",
        default=str(ROOT / "experiments" / "model_profiles.lock.yml"),
    )
    parser.add_argument(
        "--metadata-root",
        default=str(ROOT / "model_metadata"),
    )
    args = parser.parse_args()

    lock_path = Path(args.lock).expanduser().resolve()
    metadata_root = Path(args.metadata_root).expanduser().resolve()
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or not isinstance(lock.get("models"), dict):
        raise ValueError("Model profile lock contains no models")

    for key, profile in lock["models"].items():
        update_profile(str(key), profile, metadata_root)

    lock["status"] = "official_metadata_frozen_artifacts_pending"
    lock["frozen_at"] = "2026-08-07"
    limits = lock.setdefault("common_protocol_limits", {})
    limits["input_context_tokens"] = None
    limits["max_new_tokens_per_action"] = None

    temporary = lock_path.with_suffix(lock_path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(lock, sort_keys=False, width=100),
        encoding="utf-8",
    )
    temporary.replace(lock_path)
    print("Frozen official metadata profiles:", len(lock["models"]))
    print("Lock:", lock_path)


if __name__ == "__main__":
    main()
