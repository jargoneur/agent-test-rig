#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import threading
from typing import Any, Dict, Iterable, List, Optional

import requests
import yaml

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from harness.llama_cpp_server import LlamaCppServerManager
from harness.reproducibility import utc_now_iso


ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_FIELDS = (
    "context_size",
    "gpu_layers",
    "parallel",
    "cache_type_k",
    "cache_type_v",
    "flash_attention",
    "fit",
    "fit_target_mib",
    "gpu_count",
    "split_mode",
    "tensor_split",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(value: str, registry_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    registry_relative = (registry_path.parent / path).resolve()
    if registry_relative.exists():
        return registry_relative
    return (ROOT / path).resolve()


def build_token_prompt(seed_tokens: Iterable[int], target_tokens: int) -> List[int]:
    seed = [int(value) for value in seed_tokens]
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if not seed:
        raise ValueError("tokenizer returned no seed tokens")
    repeats = (target_tokens + len(seed) - 1) // len(seed)
    return (seed * repeats)[:target_tokens]


def assess_prefill_completion(
    completion: Dict[str, Any],
    target_tokens: int,
    context_size: int,
) -> Dict[str, Any]:
    evaluated = int(completion.get("tokens_evaluated") or 0)
    server_truncated = bool(completion.get("truncated", False))
    prompt_complete = evaluated == target_tokens
    capacity_boundary_stop = bool(
        server_truncated
        and prompt_complete
        and target_tokens + 1 >= context_size
    )
    return {
        "tokens_evaluated": evaluated,
        "server_reported_truncated": server_truncated,
        "prompt_tokens_truncated": evaluated < target_tokens,
        "prompt_prefill_complete": prompt_complete,
        "capacity_boundary_stop": capacity_boundary_stop,
    }


def capacity_has_margin(
    peak_mib: float,
    total_mib: float,
    fit_target_mib: float,
) -> bool:
    return peak_mib > 0 and total_mib - peak_mib >= fit_target_mib


def capacity_group_has_margin(
    peaks_mib: Dict[str, float],
    totals_mib: Dict[str, float],
    fit_target_mib: float,
) -> bool:
    return bool(peaks_mib) and all(
        capacity_has_margin(peaks_mib.get(gpu_uuid, 0.0), total, fit_target_mib)
        for gpu_uuid, total in totals_mib.items()
    )


def _csv_rows(command: List[str]) -> List[List[str]]:
    output = subprocess.check_output(command, text=True).strip()
    if not output:
        return []
    return [
        [column.strip() for column in line.split(",")]
        for line in output.splitlines()
        if line.strip()
    ]


def gpu_memory(gpu_uuid: str) -> Dict[str, float]:
    rows = _csv_rows(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    for uuid, total, used in rows:
        if uuid == gpu_uuid:
            return {
                "total_mib": float(total),
                "used_mib": float(used),
            }
    raise RuntimeError("GPU UUID is not present: %s" % gpu_uuid)


def gpu_compute_processes(gpu_uuid: str) -> List[Dict[str, Any]]:
    rows = _csv_rows(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    return [
        {
            "gpu_uuid": uuid,
            "pid": int(pid),
            "process_name": process_name,
        }
        for uuid, pid, process_name in rows
        if uuid == gpu_uuid
    ]


class GpuSampler(threading.Thread):
    def __init__(self, gpu_uuid: str, interval_seconds: float = 2.0):
        super().__init__(daemon=True)
        self.gpu_uuid = gpu_uuid
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.stop_event = threading.Event()
        self.samples: List[Dict[str, Any]] = []
        self.errors: List[str] = []

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                sample = gpu_memory(self.gpu_uuid)
                sample["timestamp"] = utc_now_iso()
                self.samples.append(sample)
            except Exception as error:
                self.errors.append(str(error))
            self.stop_event.wait(self.interval_seconds)

    def finish(self) -> None:
        self.stop_event.set()
        self.join(timeout=5)

    def maximum_used_mib(self) -> float:
        return max(
            (float(sample["used_mib"]) for sample in self.samples),
            default=0.0,
        )


def exact_deployment(
    entry: Dict[str, Any],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    deployment = profile.get("deployment_profile")
    if not isinstance(deployment, dict):
        raise RuntimeError("frozen profile has no deployment_profile")
    expected = {field: deployment.get(field) for field in DEPLOYMENT_FIELDS}
    actual = {field: entry.get(field) for field in DEPLOYMENT_FIELDS}
    if actual != expected:
        raise RuntimeError("registry deployment differs from the frozen profile")
    return expected


def request_json(
    method: str,
    url: str,
    *,
    timeout: int,
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = requests.request(method, url, json=body, timeout=timeout)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError("server returned a non-object JSON response")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate one locked model at native context and record VRAM evidence."
    )
    parser.add_argument("profile_key")
    parser.add_argument(
        "--gpu-uuid",
        action="append",
        required=True,
        help="Exact GPU UUID. Repeat in CUDA device order for multi-GPU profiles.",
    )
    parser.add_argument(
        "--lock",
        default=str(ROOT / "experiments" / "model_profiles.lock.yml"),
    )
    parser.add_argument(
        "--registry",
        default=str(ROOT / "model_artifacts" / "registry.yml"),
    )
    parser.add_argument(
        "--binary",
        default=str(ROOT / ".upstreams" / "llama.cpp" / "build" / "bin" / "llama-server"),
    )
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--startup-timeout-seconds", type=int, default=1800)
    parser.add_argument("--request-timeout-seconds", type=int, default=14400)
    parser.add_argument("--evidence")
    args = parser.parse_args()

    lock_path = Path(args.lock).expanduser().resolve()
    registry_path = Path(args.registry).expanduser().resolve()
    evidence_path = (
        Path(args.evidence).expanduser().resolve()
        if args.evidence
        else ROOT
        / "model_artifacts"
        / args.profile_key
        / "validation-evidence.json"
    )
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    profile = lock["models"][args.profile_key]
    entry = dict(registry["models"][args.profile_key])
    deployment = exact_deployment(entry, profile)

    artifact = resolve_artifact(str(entry["artifact"]), registry_path)
    actual_hash = sha256_file(artifact)
    if actual_hash != str(entry.get("sha256") or ""):
        raise RuntimeError("registry artifact hash does not match the GGUF")
    revision = str(
        entry.get("model_revision")
        or entry.get("upstream_revision")
        or ""
    )
    if revision != str(profile["model_revision"]):
        raise RuntimeError("registry revision differs from the frozen profile")

    llama_cpp = ROOT / ".upstreams" / "llama.cpp"
    llama_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=llama_cpp,
        text=True,
    ).strip()
    expected_llama_commit = str(lock["runtime"]["backend_revision"])
    if llama_commit != expected_llama_commit:
        raise RuntimeError("llama.cpp checkout differs from the frozen runtime")

    gpu_uuids = list(args.gpu_uuid)
    if len(gpu_uuids) != len(set(gpu_uuids)):
        raise RuntimeError("selected GPU UUIDs must be unique")
    expected_gpu_count = int(deployment.get("gpu_count") or 1)
    if len(gpu_uuids) != expected_gpu_count:
        raise RuntimeError(
            "profile requires %d GPUs but %d were selected"
            % (expected_gpu_count, len(gpu_uuids))
        )
    baselines = {gpu_uuid: gpu_memory(gpu_uuid) for gpu_uuid in gpu_uuids}
    active = [
        process
        for gpu_uuid in gpu_uuids
        for process in gpu_compute_processes(gpu_uuid)
    ]
    if active:
        raise RuntimeError(
            "selected GPU already has compute processes: %s"
            % json.dumps(active, sort_keys=True)
        )

    evidence: Dict[str, Any] = {
        "schema_version": 3,
        "status": "running",
        "profile_key": args.profile_key,
        "model_id": profile["model_id"],
        "model_revision": profile["model_revision"],
        "artifact": str(artifact),
        "artifact_sha256": actual_hash,
        "llama_cpp_commit": llama_commit,
        "gpu_uuid": gpu_uuids[0],
        "gpu_uuids": gpu_uuids,
        "gpu_total_vram_gib": sum(
            baseline["total_mib"] for baseline in baselines.values()
        ) / 1024.0,
        "gpu_total_vram_by_uuid_mib": {
            gpu_uuid: baseline["total_mib"]
            for gpu_uuid, baseline in baselines.items()
        },
        "gpu_baseline_used_by_uuid_mib": {
            gpu_uuid: baseline["used_mib"]
            for gpu_uuid, baseline in baselines.items()
        },
        "context_size": int(deployment["context_size"]),
        "cache_type_k": deployment["cache_type_k"],
        "cache_type_v": deployment["cache_type_v"],
        "deployment_profile": deployment,
        "started_at": utc_now_iso(),
        "chat_template_validated": False,
        "full_context_prefill_completed": False,
        "truncated": None,
        "server_reported_truncated": None,
        "prompt_tokens_truncated": None,
        "prompt_prefill_complete": False,
        "capacity_boundary_stop": False,
        "context_shift_disabled": True,
        "tokens_requested": None,
        "tokens_evaluated": None,
    }
    samplers = {
        gpu_uuid: GpuSampler(gpu_uuid) for gpu_uuid in gpu_uuids
    }
    manager: Optional[LlamaCppServerManager] = None
    failure: Optional[Exception] = None

    try:
        for sampler in samplers.values():
            sampler.start()
        manager = LlamaCppServerManager(
            {
                "registry": str(registry_path),
                "binary": str(Path(args.binary).expanduser().resolve()),
                "host": "127.0.0.1",
                "port": int(args.port),
                "gpus": gpu_uuids,
                "startup_timeout_seconds": int(args.startup_timeout_seconds),
                "shutdown_timeout_seconds": 30,
                "log_path": str(
                    evidence_path.parent / "validation-llama-server.log"
                ),
            },
            worker_id="validate-%s" % args.profile_key,
        )
        runtime = manager.ensure_model(args.profile_key)
        server_root = str(runtime["base_url"]).removesuffix("/v1")
        alias = str(runtime["runtime_name"])

        generation = dict(profile.get("resolved_generation_options") or {})
        chat_body: Dict[str, Any] = {
            "model": alias,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 1,
            "stream": False,
        }
        for key in ("temperature", "top_p", "top_k", "min_p"):
            if key in generation:
                chat_body[key] = generation[key]
        if generation.get("chat_template_kwargs"):
            chat_body["chat_template_kwargs"] = generation[
                "chat_template_kwargs"
            ]
        chat = request_json(
            "POST",
            server_root + "/v1/chat/completions",
            timeout=min(600, int(args.request_timeout_seconds)),
            body=chat_body,
        )
        if not isinstance(chat.get("choices"), list):
            raise RuntimeError("chat-template validation returned no choices")
        evidence["chat_template_validated"] = True

        tokenized = request_json(
            "POST",
            server_root + "/tokenize",
            timeout=120,
            body={
                "content": "context validation token\n",
                "add_special": False,
                "parse_special": False,
            },
        )
        target_tokens = int(deployment["context_size"]) - 1
        prompt_tokens = build_token_prompt(
            tokenized.get("tokens") or [],
            target_tokens,
        )
        completion = request_json(
            "POST",
            server_root + "/completion",
            timeout=int(args.request_timeout_seconds),
            body={
                "prompt": prompt_tokens,
                "n_predict": 0,
                "n_keep": -1,
                "cache_prompt": False,
            },
        )
        assessment = assess_prefill_completion(
            completion,
            target_tokens,
            int(deployment["context_size"]),
        )
        evidence["tokens_requested"] = target_tokens
        evidence.update(assessment)
        evidence["truncated"] = assessment["server_reported_truncated"]
        if not assessment["prompt_prefill_complete"]:
            raise RuntimeError(
                "native-context prompt prefill was incomplete: "
                "requested=%d evaluated=%d server_truncated=%s"
                % (
                    target_tokens,
                    assessment["tokens_evaluated"],
                    assessment["server_reported_truncated"],
                )
            )
        if (
            assessment["server_reported_truncated"]
            and not assessment["capacity_boundary_stop"]
        ):
            raise RuntimeError("llama.cpp reported an unexpected truncation")
        evidence["full_context_prefill_completed"] = True
        evidence["status"] = "validated"
    except Exception as error:
        failure = error
        evidence["status"] = "failed"
        evidence["error"] = {
            "type": error.__class__.__name__,
            "message": str(error),
        }
    finally:
        if manager is not None:
            manager.stop()
        for sampler in samplers.values():
            sampler.finish()
        peaks_mib = {
            gpu_uuid: sampler.maximum_used_mib()
            for gpu_uuid, sampler in samplers.items()
        }
        totals_mib = {
            gpu_uuid: baseline["total_mib"]
            for gpu_uuid, baseline in baselines.items()
        }
        sample_errors = [
            "%s: %s" % (gpu_uuid, error)
            for gpu_uuid, sampler in samplers.items()
            for error in sampler.errors
        ]
        margins_mib = {
            gpu_uuid: totals_mib[gpu_uuid] - peaks_mib[gpu_uuid]
            for gpu_uuid in gpu_uuids
        }
        evidence["gpu_samples"] = sum(
            len(sampler.samples) for sampler in samplers.values()
        )
        evidence["gpu_samples_by_uuid"] = {
            gpu_uuid: len(sampler.samples)
            for gpu_uuid, sampler in samplers.items()
        }
        evidence["gpu_sample_errors"] = sample_errors
        evidence["measured_peak_vram_by_uuid_mib"] = peaks_mib
        evidence["measured_peak_vram_mib"] = sum(peaks_mib.values())
        evidence["measured_peak_vram_gib"] = sum(peaks_mib.values()) / 1024.0
        evidence["runtime_safety_margin_by_uuid_mib"] = margins_mib
        evidence["runtime_safety_margin_mib"] = min(margins_mib.values())
        evidence["required_safety_margin_mib"] = float(
            deployment["fit_target_mib"]
        )
        evidence["safety_margin_validated"] = capacity_group_has_margin(
            peaks_mib,
            totals_mib,
            float(deployment["fit_target_mib"]),
        )
        if (
            evidence["status"] == "validated"
            and (
                sample_errors
                or not evidence["safety_margin_validated"]
                or any(not sampler.samples for sampler in samplers.values())
            )
        ):
            evidence["status"] = "failed"
            evidence["error"] = {
                "type": "GpuMeasurementError",
                "message": "VRAM sampling failed or required margin was not met",
            }
            failure = RuntimeError(evidence["error"]["message"])
        evidence["finished_at"] = utc_now_iso()
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = evidence_path.with_suffix(evidence_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(evidence_path)

    print(json.dumps(evidence, indent=2, sort_keys=True))
    print("Validation evidence:", evidence_path)
    if failure is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
