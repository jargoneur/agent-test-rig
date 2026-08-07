from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests


def request(
    method: str,
    url: str,
    token: Optional[str],
    payload: Optional[Dict[str, Any]] = None,
):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer %s" % token
    response = requests.request(
        method,
        url,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect, pause, resume or back up the scheduler."
    )
    parser.add_argument("action", choices=["status", "pause", "resume", "backup"])
    parser.add_argument(
        "--scheduler-url",
        default=os.environ.get("SCHEDULER_URL", "http://127.0.0.1:8787"),
    )
    parser.add_argument("--token", default=os.environ.get("SCHEDULER_TOKEN"))
    parser.add_argument("--token-file")
    parser.add_argument("--reason", default="manual operator pause")
    parser.add_argument("--label", default="manual")
    args = parser.parse_args()

    token = args.token
    if args.token_file:
        if token:
            raise ValueError("Use either --token/SCHEDULER_TOKEN or --token-file")
        token = Path(args.token_file).read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("Scheduler token file is empty")

    base = args.scheduler_url.rstrip("/")
    if args.action == "status":
        value = request("GET", base + "/status", token)
    elif args.action == "pause":
        value = request(
            "POST", base + "/pause", token, {"reason": args.reason}
        )
    elif args.action == "resume":
        value = request("POST", base + "/resume", token, {})
    else:
        value = request(
            "POST", base + "/backup", token, {"label": args.label}
        )
    print(json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
