from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from harness.distributed_scheduler_store import DistributedSchedulerStore


class SchedulerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


class BackupManager:
    def __init__(
        self,
        store: DistributedSchedulerStore,
        directory: str,
        interval_seconds: int,
        retain: int,
    ):
        self.store = store
        self.directory = Path(directory).expanduser().resolve()
        self.interval_seconds = max(60, int(interval_seconds))
        self.retain = max(1, int(retain))
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.last_backup: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None

    def create(self, label: Optional[str] = None) -> Dict[str, Any]:
        self.directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = "-%s" % label if label else ""
        path = self.directory / ("scheduler-%s%s.sqlite3" % (timestamp, suffix))
        result = self.store.backup_database(str(path))
        self.last_backup = result
        self.last_error = None
        self.prune()
        return result

    def prune(self) -> None:
        backups = sorted(
            self.directory.glob("scheduler-*.sqlite3"),
            key=lambda value: value.stat().st_mtime,
            reverse=True,
        )
        for path in backups[self.retain :]:
            path.unlink(missing_ok=True)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            try:
                self.create()
            except Exception as error:
                self.last_error = str(error)
                print("Scheduler backup failed:", error, flush=True)

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._run,
            name="scheduler-backup",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)

    def status(self) -> Dict[str, Any]:
        return {
            "directory": str(self.directory),
            "interval_seconds": self.interval_seconds,
            "retain": self.retain,
            "last_backup": self.last_backup,
            "last_error": self.last_error,
        }


class SchedulerHandler(BaseHTTPRequestHandler):
    server_version = "AgentRigScheduler/1.2"

    def _json_response(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = self.server.scheduler_token
        if not expected:
            return True
        header = self.headers.get("Authorization", "")
        return header == "Bearer %s" % expected

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length) if length else b"{}"
        value = json.loads(data.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Request body must be a JSON object")
        return value

    def do_GET(self) -> None:
        if not self._authorized():
            self._json_response(401, {"error": "unauthorized"})
            return
        path = urlparse(self.path).path
        if path == "/health":
            integrity = self.server.store.integrity_check(thorough=False)
            status = 200 if integrity.get("ok") else 503
            self._json_response(
                status,
                {
                    "status": "ok" if status == 200 else "degraded",
                    "control": self.server.store.pause_state(),
                    "integrity": integrity,
                    "backup": self.server.backup_manager.status(),
                },
            )
            return
        if path == "/status":
            value = self.server.store.status()
            value["backup"] = self.server.backup_manager.status()
            self._json_response(200, value)
            return
        if path == "/control":
            self._json_response(200, self.server.store.pause_state())
            return
        self._json_response(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._json_response(401, {"error": "unauthorized"})
            return
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/register":
                self.server.store.register_worker(
                    payload["worker_id"],
                    payload.get("capabilities") or {},
                    payload.get("active_model_id"),
                )
                result = {
                    "status": "registered",
                    "control": self.server.store.pause_state(),
                }
            elif path == "/claim":
                control = self.server.store.pause_state()
                claim = None
                if not control.get("paused"):
                    claim = self.server.store.claim_block(
                        payload["worker_id"],
                        payload.get("capabilities") or {},
                        int(payload.get("lease_seconds", 3600)),
                        payload.get("active_model_id"),
                    )
                result = {"claim": claim, "control": control}
            elif path == "/heartbeat":
                valid = self.server.store.heartbeat(
                    payload["worker_id"],
                    payload["block_id"],
                    payload["lease_token"],
                    int(payload.get("lease_seconds", 3600)),
                    payload.get("active_model_id"),
                )
                result = {
                    "valid": valid,
                    "control": self.server.store.pause_state(),
                }
            elif path == "/complete":
                self.server.store.complete_block(
                    payload["worker_id"],
                    payload["block_id"],
                    payload["lease_token"],
                    payload.get("results") or [],
                )
                result = {"status": "completed"}
            elif path == "/fail":
                self.server.store.fail_block(
                    payload["worker_id"],
                    payload["block_id"],
                    payload["lease_token"],
                    str(payload.get("error", "unknown error")),
                    bool(payload.get("retry", True)),
                )
                result = {"status": "released"}
            elif path == "/pause":
                result = self.server.store.set_paused(
                    True, str(payload.get("reason") or "operator pause")
                )
            elif path == "/resume":
                result = self.server.store.set_paused(False, None)
            elif path == "/backup":
                result = self.server.backup_manager.create(
                    str(payload.get("label")) if payload.get("label") else "manual"
                )
            else:
                self._json_response(404, {"error": "not_found"})
                return
            self._json_response(200, result)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._json_response(400, {"error": str(error)})
        except Exception as error:
            self._json_response(500, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), format % args), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the central experiment scheduler.")
    parser.add_argument("--database", default="scheduler/scheduler.sqlite3")
    parser.add_argument("--manifest", action="append")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--token", default=os.environ.get("SCHEDULER_TOKEN"))
    parser.add_argument("--export-results")
    parser.add_argument("--backup-directory", default="scheduler/backups")
    parser.add_argument("--backup-interval-seconds", type=int, default=900)
    parser.add_argument("--backup-retain", type=int, default=96)
    args = parser.parse_args()

    store = DistributedSchedulerStore(args.database)
    integrity = store.integrity_check(thorough=True)
    if not integrity.get("ok"):
        raise RuntimeError("Scheduler database failed integrity check: %s" % integrity)

    for manifest in args.manifest or []:
        print("Imported:", manifest, store.import_manifest(manifest))
    if args.export_results:
        print("Exported results:", store.export_results(args.export_results))
        return

    backup_manager = BackupManager(
        store,
        directory=args.backup_directory,
        interval_seconds=args.backup_interval_seconds,
        retain=args.backup_retain,
    )
    startup_backup = backup_manager.create("startup")
    print("Startup backup:", startup_backup)
    backup_manager.start()

    server = SchedulerHTTPServer((args.host, args.port), SchedulerHandler)
    server.store = store
    server.scheduler_token = args.token
    server.backup_manager = backup_manager
    print("Scheduler listening on http://%s:%d" % (args.host, args.port))
    print("Database:", args.database)
    print("Backup directory:", args.backup_directory)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Scheduler shutdown requested.")
    finally:
        server.shutdown()
        server.server_close()
        backup_manager.stop()
        try:
            final_backup = backup_manager.create("shutdown")
            print("Shutdown backup:", final_backup)
        except Exception as error:
            print("Shutdown backup failed:", error)


if __name__ == "__main__":
    main()
