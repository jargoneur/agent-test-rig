from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

from harness.scheduler_store import SchedulerStore


class SchedulerHandler(BaseHTTPRequestHandler):
    server_version = "AgentRigScheduler/1.0"

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
            self._json_response(200, {"status": "ok"})
            return
        if path == "/status":
            self._json_response(200, self.server.store.status())
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
                result = {"status": "registered"}
            elif path == "/claim":
                claim = self.server.store.claim_block(
                    payload["worker_id"],
                    payload.get("capabilities") or {},
                    int(payload.get("lease_seconds", 3600)),
                    payload.get("active_model_id"),
                )
                result = {"claim": claim}
            elif path == "/heartbeat":
                valid = self.server.store.heartbeat(
                    payload["worker_id"],
                    payload["block_id"],
                    payload["lease_token"],
                    int(payload.get("lease_seconds", 3600)),
                    payload.get("active_model_id"),
                )
                result = {"valid": valid}
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
            else:
                self._json_response(404, {"error": "not_found"})
                return
            self._json_response(200, result)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._json_response(400, {"error": str(error)})
        except Exception as error:
            self._json_response(500, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), format % args))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the central experiment scheduler.")
    parser.add_argument("--database", default="scheduler/scheduler.sqlite3")
    parser.add_argument("--manifest", action="append")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--token", default=os.environ.get("SCHEDULER_TOKEN"))
    parser.add_argument("--export-results")
    args = parser.parse_args()

    store = SchedulerStore(args.database)
    for manifest in args.manifest or []:
        print("Imported:", manifest, store.import_manifest(manifest))
    if args.export_results:
        print("Exported results:", store.export_results(args.export_results))
        return

    server = ThreadingHTTPServer((args.host, args.port), SchedulerHandler)
    server.store = store
    server.scheduler_token = args.token
    print("Scheduler listening on http://%s:%d" % (args.host, args.port))
    print("Database:", args.database)
    server.serve_forever()


if __name__ == "__main__":
    main()
