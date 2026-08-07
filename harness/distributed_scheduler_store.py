from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from harness.experiment_jobs import canonical_json
from harness.reproducibility import utc_now_iso
from harness.scheduler_store import SchedulerStore


class DistributedSchedulerStore(SchedulerStore):
    """Persistent scheduler store for distributed experiment workers.

    Model IDs identify locally verified artifacts. Resource classes are an
    additional hardware constraint, never an alternative to model presence.
    SQLite remains the single authoritative queue, with online backups and
    integrity checks for restart recovery.
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

    @staticmethod
    def _pause_state_from_connection(connection: sqlite3.Connection) -> Dict[str, Any]:
        row = connection.execute(
            "SELECT value_json FROM scheduler_control WHERE key = 'pause'"
        ).fetchone()
        if row is None:
            return {"paused": False, "reason": None, "updated_at": None}
        value = json.loads(row["value_json"])
        return value if isinstance(value, dict) else {"paused": False}

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
            return self._pause_state_from_connection(connection)

    def integrity_check(self, thorough: bool = False) -> Dict[str, Any]:
        pragma = "integrity_check" if thorough else "quick_check"
        with self._lock, self._connect() as connection:
            rows = [str(row[0]) for row in connection.execute("PRAGMA %s" % pragma)]
        ok = rows == ["ok"]
        return {
            "ok": ok,
            "check": pragma,
            "details": rows,
            "checked_at": utc_now_iso(),
        }

    def checkpoint_wal(self, mode: str = "PASSIVE") -> Dict[str, int]:
        normalized = str(mode).upper()
        if normalized not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            raise ValueError("Unsupported WAL checkpoint mode: %s" % mode)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "PRAGMA wal_checkpoint(%s)" % normalized
            ).fetchone()
        return {
            "busy": int(row[0]),
            "log_frames": int(row[1]),
            "checkpointed_frames": int(row[2]),
        }

    def backup_database(self, output_path: str) -> Dict[str, Any]:
        """Create an online, transactionally consistent SQLite backup."""
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.unlink(missing_ok=True)

        with self._lock, self._connect() as source:
            target = sqlite3.connect(str(temporary))
            try:
                source.backup(target)
                target.execute("PRAGMA journal_mode=DELETE")
                target.commit()
                rows = [str(row[0]) for row in target.execute("PRAGMA quick_check")]
                if rows != ["ok"]:
                    raise RuntimeError(
                        "Scheduler backup failed integrity check: %s" % rows
                    )
            finally:
                target.close()

        os.replace(str(temporary), str(destination))
        return {
            "path": str(destination),
            "size_bytes": destination.stat().st_size,
            "created_at": utc_now_iso(),
            "integrity": "ok",
        }

    @staticmethod
    def restore_database(backup_path: str, database_path: str) -> Dict[str, Any]:
        """Restore a validated backup while the scheduler is stopped."""
        source_path = Path(backup_path).expanduser().resolve()
        destination = Path(database_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError("Scheduler backup is missing: %s" % source_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".restore.tmp")
        temporary.unlink(missing_ok=True)

        source = sqlite3.connect(str(source_path))
        try:
            rows = [str(row[0]) for row in source.execute("PRAGMA quick_check")]
            if rows != ["ok"]:
                raise RuntimeError("Backup integrity check failed: %s" % rows)
            target = sqlite3.connect(str(temporary))
            try:
                source.backup(target)
                target.commit()
            finally:
                target.close()
        finally:
            source.close()

        for suffix in ("-wal", "-shm"):
            Path(str(destination) + suffix).unlink(missing_ok=True)
        os.replace(str(temporary), str(destination))
        return {
            "backup": str(source_path),
            "database": str(destination),
            "restored_at": utc_now_iso(),
        }

    def claim_block(
        self,
        worker_id: str,
        capabilities: Dict[str, Any],
        lease_seconds: int,
        active_model_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        model_ids = [str(value) for value in capabilities.get("model_ids", [])]
        resource_classes = [
            str(value) for value in capabilities.get("resource_classes", [])
        ]
        waves = [int(value) for value in capabilities.get("waves", [])]
        model_profiles = dict(capabilities.get("model_profiles") or {})
        harness_commit = str(capabilities.get("harness_commit") or "")
        if not model_ids:
            raise ValueError(
                "Worker must advertise at least one locally verified model_id"
            )
        if not harness_commit:
            raise ValueError("Worker must advertise its exact harness_commit")

        profile_clauses = []
        profile_parameters: List[Any] = []
        for model_id in model_ids:
            profile = dict(model_profiles.get(model_id) or {})
            profile_sha256 = str(profile.get("profile_sha256") or "")
            artifact_sha256 = str(
                profile.get("quantized_artifact_sha256")
                or profile.get("artifact_sha256")
                or ""
            )
            if not profile_sha256 or not artifact_sha256:
                raise ValueError(
                    "Worker model %s lacks exact profile/artifact provenance"
                    % model_id
                )
            profile_clauses.append(
                "(model_id = ? AND profile_sha256 = ? AND artifact_sha256 = ?)"
            )
            profile_parameters.extend(
                [model_id, profile_sha256, artifact_sha256]
            )

        token = "%s:%d" % (worker_id, time.time_ns())
        now_text = utc_now_iso()
        now_epoch = time.time()
        lease_expires_at = now_epoch + max(30, int(lease_seconds))

        clauses = [
            "state = 'pending'",
            "harness_commit = ?",
            "(" + " OR ".join(profile_clauses) + ")",
        ]
        parameters: List[Any] = [harness_commit] + profile_parameters

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
                if self._pause_state_from_connection(connection).get("paused"):
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
            "attempt": int(row["attempts"]) + 1,
            "infrastructure_failures": int(row["infrastructure_failures"]),
            "block": block,
        }

    def complete_block(
        self,
        worker_id: str,
        block_id: str,
        lease_token: str,
        results: Sequence[Dict[str, Any]],
    ) -> None:
        """Complete a block idempotently.

        If the scheduler committed the results but the HTTP response was lost,
        the worker may safely submit the exact same completion again.
        """
        now_text = utc_now_iso()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT payload_json, state, lease_owner, lease_token
                    FROM blocks WHERE block_id = ?
                    """,
                    (block_id,),
                ).fetchone()
                if row is None:
                    raise ValueError("Unknown block: %s" % block_id)

                block = json.loads(row["payload_json"])
                self._validate_completed_results(block, results)

                if row["state"] == "completed":
                    existing_rows = connection.execute(
                        "SELECT run_id, result_json FROM results WHERE block_id = ?",
                        (block_id,),
                    ).fetchall()
                    existing = {
                        existing_row["run_id"]: existing_row["result_json"]
                        for existing_row in existing_rows
                    }
                    submitted = {
                        record["run_id"]: canonical_json(record) for record in results
                    }
                    if existing != submitted:
                        raise ValueError(
                            "Block is already completed with different result payloads"
                        )
                    connection.execute("COMMIT")
                    return

                if (
                    row["state"] != "leased"
                    or row["lease_owner"] != worker_id
                    or row["lease_token"] != lease_token
                ):
                    raise ValueError("Lease is no longer valid")

                for record in results:
                    connection.execute(
                        """
                        INSERT INTO results (
                            run_id, block_id, result_json, received_at
                        ) VALUES (?, ?, ?, ?)
                        ON CONFLICT(run_id) DO UPDATE SET
                            result_json = excluded.result_json,
                            received_at = excluded.received_at
                        """,
                        (
                            record["run_id"],
                            block_id,
                            canonical_json(record),
                            now_text,
                        ),
                    )
                connection.execute(
                    """
                    UPDATE blocks
                    SET state = 'completed', lease_owner = NULL,
                        lease_token = NULL, lease_expires_at = NULL,
                        last_error = NULL, updated_at = ?
                    WHERE block_id = ?
                    """,
                    (now_text, block_id),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def status(self) -> Dict[str, Any]:
        value = super().status()
        value["control"] = self.pause_state()
        value["integrity"] = self.integrity_check(thorough=False)
        return value
