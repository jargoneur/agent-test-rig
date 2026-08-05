from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yaml

from harness.experiment_jobs import atomic_write_json, result_file
from harness.llama_cpp_server import LlamaCppServerManager
from harness.resource_policy import ResourcePolicy, ResourceYieldRequested
from run_worker import execute_job


class SchedulerClient:
    def __init__(self, base_url: str, token: Optional[str], timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = "Bearer %s" % token

    def post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            self.base_url + path,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise RuntimeError("Scheduler returned a non-object response")
        return value


class HeartbeatThread(threading.Thread):
    def __init__(
        self,
        client: SchedulerClient,
        payload: Dict[str, Any],
        interval: int,
    ):
        super().__init__(daemon=True)
        self.client = client
        self.payload = payload
        self.interval = max(10, interval)
        self.stop_event = threading.Event()
        self.invalid = False

    def run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                response = self.client.post("/heartbeat", self.payload)
                if not response.get("valid"):
                    self.invalid = True
                    return
            except Exception as error:
                print("Heartbeat warning:", error)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=5)


def load_config(path: str) -> Dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Worker config must be a YAML mapping")
    return value


def release_scheduler_lease_for_yield(
    client: SchedulerClient,
    worker_id: str,
    block_id: str,
    lease_token: str,
    reason: str,
) -> None:
    client.post(
        "/fail",
        {
            "worker_id": worker_id,
            "block_id": block_id,
            "lease_token": lease_token,
            "error": "resource_yield: %s" % reason,
            "retry": True,
        },
    )


def _build_model_manager(config: Dict[str, Any], worker_id: str):
    runtime = config.get("model_runtime")
    if not runtime:
        return None
    if not isinstance(runtime, dict):
        raise ValueError("model_runtime must be a mapping")
    runtime_type = str(runtime.get("type") or "llama_cpp")
    if runtime_type != "llama_cpp":
        raise ValueError("Unsupported model_runtime.type: %s" % runtime_type)
    return LlamaCppServerManager(runtime, worker_id=worker_id)


def _resolve_capabilities(
    config: Dict[str, Any],
    model_manager: Optional[LlamaCppServerManager],
) -> Dict[str, Any]:
    capabilities = dict(config.get("capabilities") or {})
    if model_manager is not None:
        provisioned = set(model_manager.model_ids())
        configured = set(str(value) for value in capabilities.get("model_ids", []))
        if configured:
            missing = configured.difference(provisioned)
            if missing:
                raise ValueError(
                    "Worker advertises models missing from registry: %s"
                    % ", ".join(sorted(missing))
                )
            capabilities["model_ids"] = sorted(configured)
        else:
            capabilities["model_ids"] = sorted(provisioned)
    return capabilities


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull complete paired blocks from the central scheduler."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument("--pause-file", default="PAUSE")
    args = parser.parse_args()

    config = load_config(args.config)
    resource_policy = ResourcePolicy.from_worker_config(
        config,
        pause_file=args.pause_file,
    )
    worker_id = str(config.get("worker_id") or socket.gethostname())
    scheduler_url = str(config["scheduler_url"])
    token = config.get("scheduler_token") or os.environ.get("SCHEDULER_TOKEN")
    model_manager = _build_model_manager(config, worker_id)
    capabilities = _resolve_capabilities(config, model_manager)
    results_root = Path(config.get("results_root", "distributed_results/%s" % worker_id))
    results_root.mkdir(parents=True, exist_ok=True)
    lease_seconds = int(config.get("lease_seconds", 7200))
    heartbeat_seconds = int(config.get("heartbeat_seconds", 60))
    idle_seconds = int(config.get("idle_seconds", 30))
    stop_when_idle = bool(config.get("stop_when_idle", False))

    try:
        resource_policy.checkpoint("before_worker_registration")
    except ResourceYieldRequested as error:
        print("Worker did not start:", error)
        resource_policy.release()
        return

    client = SchedulerClient(scheduler_url, token)
    active_model_id = None
    client.post(
        "/register",
        {
            "worker_id": worker_id,
            "capabilities": capabilities,
            "active_model_id": active_model_id,
        },
    )
    print("Worker registered:", worker_id)
    print("Scheduler:", scheduler_url)
    print("Capabilities:", json.dumps(capabilities, ensure_ascii=False))
    print(
        "Resource policy:",
        json.dumps(resource_policy.metadata(), ensure_ascii=False),
    )
    if model_manager is not None:
        print("Model registry:", model_manager.registry_path)
        print("llama-server:", model_manager.binary)
        print("GPU selector:", model_manager.gpu)

    completed_blocks = 0
    yielded = False
    try:
        while True:
            try:
                resource_policy.checkpoint("before_block_claim")
            except ResourceYieldRequested as error:
                print("Yielding before next block:", error)
                break

            if args.max_blocks is not None and completed_blocks >= args.max_blocks:
                break

            response = client.post(
                "/claim",
                {
                    "worker_id": worker_id,
                    "capabilities": capabilities,
                    "active_model_id": active_model_id,
                    "lease_seconds": lease_seconds,
                },
            )
            claim = response.get("claim")
            if not claim:
                print("No compatible block available.")
                if stop_when_idle:
                    break
                time.sleep(idle_seconds)
                continue

            block = claim["block"]
            block_id = block["block_id"]
            lease_token = claim["lease_token"]
            active_model_id = block["model_id"]
            heartbeat_payload = {
                "worker_id": worker_id,
                "block_id": block_id,
                "lease_token": lease_token,
                "lease_seconds": lease_seconds,
                "active_model_id": active_model_id,
            }
            heartbeat = HeartbeatThread(
                client,
                heartbeat_payload,
                heartbeat_seconds,
            )
            heartbeat.start()
            records = []
            try:
                print(
                    "Claimed %s task=%s model=%s repeat=%s (%d scaffolds)"
                    % (
                        block_id,
                        block["task_id"],
                        block["model_id"],
                        block["repeat"],
                        len(block["jobs"]),
                    )
                )
                runtime_overrides = None
                if model_manager is not None:
                    resource_policy.checkpoint("before_model_server_start")
                    runtime_overrides = model_manager.ensure_model(active_model_id)
                    print(
                        "Active model server:",
                        json.dumps(runtime_overrides, ensure_ascii=False),
                    )

                for job in block["jobs"]:
                    resource_policy.checkpoint(
                        "before_scaffold:%s" % job["scaffold"]
                    )
                    output_path = result_file(results_root, job["run_id"])
                    if output_path.exists():
                        try:
                            existing = json.loads(
                                output_path.read_text(encoding="utf-8")
                            )
                        except (OSError, ValueError, json.JSONDecodeError):
                            existing = None
                        if existing and existing.get("status") == "completed":
                            records.append(existing)
                            continue
                    record = execute_job(
                        job,
                        results_root,
                        worker_id,
                        resource_policy=resource_policy,
                        runtime_overrides=runtime_overrides,
                    )
                    atomic_write_json(output_path, record)
                    records.append(record)

                heartbeat.stop()
                if heartbeat.invalid:
                    raise RuntimeError("Scheduler lease expired during block execution")
                client.post(
                    "/complete",
                    {
                        "worker_id": worker_id,
                        "block_id": block_id,
                        "lease_token": lease_token,
                        "results": records,
                    },
                )
                completed_blocks += 1
                print("Completed block:", block_id)
            except ResourceYieldRequested as error:
                heartbeat.stop()
                print("Shared resource requested; releasing block:", error)
                try:
                    release_scheduler_lease_for_yield(
                        client,
                        worker_id,
                        block_id,
                        lease_token,
                        str(error),
                    )
                except Exception as scheduler_error:
                    print("Could not immediately release yielded block:", scheduler_error)
                    print("The lease will expire and return the block to the queue.")
                yielded = True
            except KeyboardInterrupt:
                heartbeat.stop()
                try:
                    client.post(
                        "/fail",
                        {
                            "worker_id": worker_id,
                            "block_id": block_id,
                            "lease_token": lease_token,
                            "error": "worker interrupted",
                            "retry": True,
                        },
                    )
                finally:
                    raise
            except Exception as error:
                heartbeat.stop()
                print("Block failed:", error)
                failure_path = results_root / "failed_blocks" / (block_id + ".json")
                atomic_write_json(
                    failure_path,
                    {
                        "block_id": block_id,
                        "worker_id": worker_id,
                        "resource_policy": resource_policy.metadata(),
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                try:
                    client.post(
                        "/fail",
                        {
                            "worker_id": worker_id,
                            "block_id": block_id,
                            "lease_token": lease_token,
                            "error": str(error),
                            "retry": bool(config.get("retry_failed_blocks", True)),
                        },
                    )
                except Exception as scheduler_error:
                    print("Could not release failed block:", scheduler_error)
                time.sleep(min(idle_seconds, 30))

            if yielded:
                break
    finally:
        if model_manager is not None:
            model_manager.stop()
        release_code = resource_policy.release()
        if release_code not in (None, 0):
            print("Resource release command returned:", release_code)

    print("Completed blocks in this session:", completed_blocks)


if __name__ == "__main__":
    main()
