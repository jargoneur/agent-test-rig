from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import requests


def request(method: str, url: str, token: str | None, payload: Dict[str, Any] | None = None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer %s" % token
    response = requests.request(
        method,
        url,
        headers=headers,
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or change scheduler pause state.")
    parser.add_argument("action", choices=["status", "pause", "resume"])
    parser.add_argument(
        "--scheduler-url",
        default=os.environ.get("SCHEDULER_URL", "http://127.0.0.1:8787"),
    )
    parser.add_argument("--token", default=os.environ.get("SCHEDULER_TOKEN"))
    parser.add_argument("--reason", default="manual operator pause")
    args = parser.parse_args()

    base = args.scheduler_url.rstrip("/")
    if args.action == "status":
        value = request("GET", base + "/status", args.token)
    elif args.action == "pause":
        value = request(
            "POST", base + "/pause", args.token, {"reason": args.reason}
        )
    else:
        value = request("POST", base + "/resume", args.token, {})
    print(json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
