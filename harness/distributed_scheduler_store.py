from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.reproducibility import utc_now_iso
from harness.scheduler_store import SchedulerStore


class DistributedSchedulerStore(SchedulerStore):
    """Scheduler store with persistent pause state and strict capabilities.

    Model IDs identify locally verified artifacts. Resource classes are an
    additional hardware constraint, never an alternative to model presence.
    """

    def _initialize(self) -> None:
        super()._initialize()
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduler_control (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO scheduler_control (key, value_json, updated_at)
                VALUES ('pause', '{"paused": false, "reason": null}', ?)
                """,
                (utc_now_iso(),),
            )

    def set_paused(self, paused: bool, reason: Optional[str] = None) -> Dict[str, Any]:
        state = {
            "paused": bool(paused),
            "reason": str(reason) if reason else None,
            "updated_at": utc_now_iso(),
        }
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO scheduler_control (key, value_json, updated_at)
                VALUES ('pause', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (json.dumps(state, sort_keys=True), state["updated_at"]),
            )
        return state

    def pause_state(self) -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM scheduler_control WHERE key = 'pause'"
            ).fetchone()
        if row is None:
            return {"paused": False, "reason": None, "updated_at": None}
        value = json.loads(row["value_json"])
        return value if isinstance(value, dict) else {"paused": False}

    def claim_block(
        self,
        worker_id: str,
        capabilities: Dict[str, Any],
        lease_seconds: int,
        active_model_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.pause_state().get("paused"):
            return None

        model_ids = [str(value) for value in capabilities.get("model_ids", [])]
        resource_classes = [
            str(value) for value in capabilities.get("resource_classes", [])
        ]
        waves = [int(value) for value in capabilities.get("waves", [])]
        if not model_ids:
            raise ValueError(
                "Worker must advertise at least one locally verified model_id"
            )

        token = "%s:%d" % (worker_id, time.time_ns())
        now_text = utc_now_iso()
        now_epoch = time.time()
        lease_expires_at = now_epoch + max(30, int(lease_seconds))

        model_placeholders = ",".join("?" for _ in model_ids)
        clauses = ["state = 'pending'", "model_id IN (%s)" % model_placeholders]
        parameters: List[Any] = list(model_ids)

        if resource_classes:
            class_placeholders = ",".join("?" for _ in resource_classes)
            clauses.append(
                "(resource_class IS NULL OR resource_class IN (%s))"
                % class_placeholders
            )
            parameters.extend(resource_classes)
        else:
            clauses.append("resource_class IS NULL")

        if waves:
            wave_placeholders = ",".join("?" for _ in waves)
            clauses.append("wave IN (%s)" % wave_placeholders)
            parameters.extend(waves)

        order_parameters: List[Any] = []
        if active_model_id:
            order_sql = (
                "CASE WHEN model_id = ? THEN 0 ELSE 1 END, "
                "priority DESC, wave ASC, attempts ASC, block_id ASC"
            )
            order_parameters.append(active_model_id)
        else:
            order_sql = (
                "priority DESC, wave ASC, attempts ASC, model_id ASC, block_id ASC"
            )

        query = (
            "SELECT * FROM blocks WHERE "
            + " AND ".join(clauses)
            + " ORDER BY "
            + order_sql
            + " LIMIT 1"
        )

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._release_expired(connection)
                if self.pause_state().get("paused"):
                    connection.execute("COMMIT")
                    return None
                row = connection.execute(
                    query, tuple(parameters + order_parameters)
                ).fetchone()
                if row is None:
                    connection.execute("COMMIT")
                    return None
                updated = connection.execute(
                    """
                    UPDATE blocks
                    SET state = 'leased', lease_owner = ?, lease_token = ?,
                        lease_expires_at = ?, attempts = attempts + 1,
                        updated_at = ?
                    WHERE block_id = ? AND state = 'pending'
                    """,
                    (
                        worker_id,
                        token,
                        lease_expires_at,
                        now_text,
                        row["block_id"],
                    ),
                )
                if updated.rowcount != 1:
                    connection.execute("ROLLBACK")
                    return None
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

        block = json.loads(row["payload_json"])
        return {
            "lease_token": token,
            "lease_expires_at": lease_expires_at,
            "block": block,
        }

    def status(self) -> Dict[str, Any]:
        value = super().status()
        value["control"] = self.pause_state()
        return value
