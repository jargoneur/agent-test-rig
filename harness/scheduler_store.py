from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from harness.experiment_jobs import canonical_json, group_jobs_by_block, load_jobs
from harness.reproducibility import utc_now_iso


class SchedulerStore:
    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS blocks (
                    block_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    wave INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    profile_sha256 TEXT,
                    artifact_sha256 TEXT,
                    harness_commit TEXT NOT NULL DEFAULT '',
                    resource_class TEXT,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending',
                    priority INTEGER NOT NULL DEFAULT 0,
                    lease_owner TEXT,
                    lease_token TEXT,
                    lease_expires_at REAL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    infrastructure_failures INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    last_failure_kind TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_blocks_state_model
                ON blocks(state, model_id, wave, priority);

                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    capabilities_json TEXT NOT NULL,
                    active_model_id TEXT,
                    last_seen_at REAL NOT NULL,
                    registered_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS results (
                    run_id TEXT PRIMARY KEY,
                    block_id TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    FOREIGN KEY(block_id) REFERENCES blocks(block_id)
                );
                """
            )

            self._ensure_column(connection, "blocks", "profile_sha256", "TEXT")
            self._ensure_column(connection, "blocks", "artifact_sha256", "TEXT")
            self._ensure_column(
                connection,
                "blocks",
                "harness_commit",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                connection,
                "blocks",
                "infrastructure_failures",
                "INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(connection, "blocks", "last_failure_kind", "TEXT")

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(%s)" % table)
        }
        if column not in columns:
            connection.execute(
                "ALTER TABLE %s ADD COLUMN %s %s"
                % (table, column, declaration)
            )

    def import_manifest(self, manifest_path: str) -> Dict[str, int]:
        jobs = load_jobs(manifest_path)
        blocks = group_jobs_by_block(jobs)
        inserted = 0
        unchanged = 0
        now = utc_now_iso()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for block in blocks:
                    model = dict(block.get("model") or {})
                    profile_sha256 = model.get("profile_sha256")
                    artifact_sha256 = model.get("quantized_artifact_sha256")
                    harness_commit = str(block.get("harness_commit") or "")
                    payload = canonical_json(block)
                    existing = connection.execute(
                        "SELECT payload_json FROM blocks WHERE block_id = ?",
                        (block["block_id"],),
                    ).fetchone()
                    if existing is None:
                        connection.execute(
                            """
                            INSERT INTO blocks (
                                block_id, experiment_id, wave, model_id,
                                profile_sha256, artifact_sha256, harness_commit,
                                resource_class, payload_json, state,
                                created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                            """,
                            (
                                block["block_id"],
                                block.get("experiment_id"),
                                int(block.get("wave", 0)),
                                block["model_id"],
                                profile_sha256,
                                artifact_sha256,
                                harness_commit,
                                block.get("resource_class"),
                                payload,
                                now,
                                now,
                            ),
                        )
                        inserted += 1
                    elif existing["payload_json"] != payload:
                        raise ValueError(
                            "Conflicting block definition for %s"
                            % block["block_id"]
                        )
                    else:
                        unchanged += 1
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return {"inserted": inserted, "unchanged": unchanged}

    def register_worker(
        self,
        worker_id: str,
        capabilities: Dict[str, Any],
        active_model_id: Optional[str] = None,
    ) -> None:
        now_text = utc_now_iso()
        now_epoch = time.time()
        payload = canonical_json(capabilities)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workers (
                    worker_id, capabilities_json, active_model_id,
                    last_seen_at, registered_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    capabilities_json = excluded.capabilities_json,
                    active_model_id = excluded.active_model_id,
                    last_seen_at = excluded.last_seen_at,
                    updated_at = excluded.updated_at
                """,
                (
                    worker_id,
                    payload,
                    active_model_id,
                    now_epoch,
                    now_text,
                    now_text,
                ),
            )

    @staticmethod
    def _max_infrastructure_retries(block: Dict[str, Any]) -> int:
        recovery = dict(block.get("recovery") or {})
        return max(0, int(recovery.get("max_infrastructure_retries", 2)))

    @staticmethod
    def _validate_completed_results(
        block: Dict[str, Any],
        results: Sequence[Dict[str, Any]],
    ) -> None:
        expected_jobs = {
            str(job["run_id"]): job for job in block.get("jobs", [])
        }
        received_ids = [str(record.get("run_id") or "") for record in results]
        if (
            len(received_ids) != len(expected_jobs)
            or len(set(received_ids)) != len(received_ids)
            or set(received_ids) != set(expected_jobs)
        ):
            raise ValueError(
                "Completed block must contain exactly one result for every expected run ID"
            )

        for record in results:
            run_id = str(record["run_id"])
            if record.get("status") != "completed":
                raise ValueError(
                    "Result %s is not a completed terminal result" % run_id
                )
            outcome = record.get("outcome")
            if (
                not isinstance(outcome, dict)
                or outcome.get("usable_result") is not True
            ):
                raise ValueError(
                    "Result %s lacks a usable terminal outcome" % run_id
                )
            submitted_job = record.get("job")
            if not isinstance(submitted_job, dict) or canonical_json(
                submitted_job
            ) != canonical_json(expected_jobs[run_id]):
                raise ValueError(
                    "Result %s does not match its scheduled job" % run_id
                )

    def _release_expired(self, connection: sqlite3.Connection) -> int:
        now_epoch = time.time()
        rows = connection.execute(
            """
            SELECT block_id, payload_json, infrastructure_failures
            FROM blocks
            WHERE state = 'leased' AND lease_expires_at < ?
            """,
            (now_epoch,),
        ).fetchall()
        now_text = utc_now_iso()
        for row in rows:
            block = json.loads(row["payload_json"])
            failures = int(row["infrastructure_failures"]) + 1
            maximum = self._max_infrastructure_retries(block)
            target_state = "pending" if failures <= maximum else "failed"
            connection.execute(
                """
                UPDATE blocks
                SET state = ?, lease_owner = NULL, lease_token = NULL,
                    lease_expires_at = NULL, infrastructure_failures = ?,
                    last_error = ?, last_failure_kind = ?, updated_at = ?
                WHERE block_id = ? AND state = 'leased'
                """,
                (
                    target_state,
                    failures,
                    "scheduler lease expired",
                    "lease_expired",
                    now_text,
                    row["block_id"],
                ),
            )
        return len(rows)

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
        if not model_ids and not resource_classes:
            raise ValueError(
                "Worker must advertise model_ids or resource_classes"
            )

        token = "%s:%d" % (worker_id, time.time_ns())
        now_text = utc_now_iso()
        now_epoch = time.time()
        lease_expires_at = now_epoch + max(30, int(lease_seconds))

        clauses = ["state = 'pending'"]
        parameters: List[Any] = []
        eligibility = []
        if model_ids:
            placeholders = ",".join("?" for _ in model_ids)
            eligibility.append("model_id IN (%s)" % placeholders)
            parameters.extend(model_ids)
        if resource_classes:
            placeholders = ",".join("?" for _ in resource_classes)
            eligibility.append("resource_class IN (%s)" % placeholders)
            parameters.extend(resource_classes)
        clauses.append("(" + " OR ".join(eligibility) + ")")
        if waves:
            placeholders = ",".join("?" for _ in waves)
            clauses.append("wave IN (%s)" % placeholders)
            parameters.extend(waves)

        order_parameters: List[Any] = []
        if active_model_id:
            order_sql = (
                "CASE WHEN model_id = ? THEN 0 ELSE 1 END, "
                "priority DESC, wave ASC, attempts ASC, block_id ASC"
            )
            order_parameters.append(active_model_id)
        else:
            order_sql = "priority DESC, wave ASC, attempts ASC, model_id ASC, block_id ASC"

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
                row = connection.execute(
                    query,
                    tuple(parameters + order_parameters),
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

    def heartbeat(
        self,
        worker_id: str,
        block_id: str,
        lease_token: str,
        lease_seconds: int,
        active_model_id: Optional[str] = None,
    ) -> bool:
        now_epoch = time.time()
        now_text = utc_now_iso()
        new_expiry = now_epoch + max(30, int(lease_seconds))
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE blocks
                SET lease_expires_at = ?, updated_at = ?
                WHERE block_id = ? AND state = 'leased'
                  AND lease_owner = ? AND lease_token = ?
                """,
                (
                    new_expiry,
                    now_text,
                    block_id,
                    worker_id,
                    lease_token,
                ),
            )
            connection.execute(
                """
                UPDATE workers
                SET last_seen_at = ?, active_model_id = ?, updated_at = ?
                WHERE worker_id = ?
                """,
                (now_epoch, active_model_id, now_text, worker_id),
            )
            return cursor.rowcount == 1

    def complete_block(
        self,
        worker_id: str,
        block_id: str,
        lease_token: str,
        results: Sequence[Dict[str, Any]],
    ) -> None:
        now_text = utc_now_iso()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT payload_json FROM blocks
                    WHERE block_id = ? AND state = 'leased'
                      AND lease_owner = ? AND lease_token = ?
                    """,
                    (block_id, worker_id, lease_token),
                ).fetchone()
                if row is None:
                    raise ValueError("Lease is no longer valid")
                block = json.loads(row["payload_json"])
                self._validate_completed_results(block, results)
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

    def fail_block(
        self,
        worker_id: str,
        block_id: str,
        lease_token: str,
        error: str,
        retry: bool = True,
        failure_kind: str = "infrastructure",
    ) -> Dict[str, Any]:
        normalized_kind = str(failure_kind or "infrastructure")
        operator_release = normalized_kind in {"operator_stop", "resource_yield"}
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT payload_json, infrastructure_failures
                    FROM blocks
                    WHERE block_id = ? AND state = 'leased'
                      AND lease_owner = ? AND lease_token = ?
                    """,
                    (block_id, worker_id, lease_token),
                ).fetchone()
                if row is None:
                    raise ValueError("Lease is no longer valid")

                block = json.loads(row["payload_json"])
                failures = int(row["infrastructure_failures"])
                if not operator_release:
                    failures += 1
                maximum = self._max_infrastructure_retries(block)
                target_state = (
                    "pending"
                    if retry and (operator_release or failures <= maximum)
                    else "failed"
                )
                connection.execute(
                    """
                    UPDATE blocks
                    SET state = ?, lease_owner = NULL, lease_token = NULL,
                        lease_expires_at = NULL, infrastructure_failures = ?,
                        last_error = ?, last_failure_kind = ?, updated_at = ?
                    WHERE block_id = ?
                    """,
                    (
                        target_state,
                        failures,
                        error[-8000:],
                        normalized_kind,
                        utc_now_iso(),
                        block_id,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return {
            "state": target_state,
            "infrastructure_failures": failures,
            "max_infrastructure_retries": maximum,
            "failure_kind": normalized_kind,
        }

    def status(self) -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            self._release_expired(connection)
            states = {
                row["state"]: row["count"]
                for row in connection.execute(
                    "SELECT state, COUNT(*) AS count FROM blocks GROUP BY state"
                )
            }
            models = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT model_id, state, COUNT(*) AS count
                    FROM blocks GROUP BY model_id, state
                    ORDER BY model_id, state
                    """
                )
            ]
            workers = [
                {
                    "worker_id": row["worker_id"],
                    "active_model_id": row["active_model_id"],
                    "last_seen_at": row["last_seen_at"],
                    "capabilities": json.loads(row["capabilities_json"]),
                }
                for row in connection.execute(
                    "SELECT * FROM workers ORDER BY worker_id"
                )
            ]
        return {
            "states": states,
            "models": models,
            "workers": workers,
            "generated_at": utc_now_iso(),
        }

    def export_results(self, output_directory: str) -> int:
        destination = Path(output_directory)
        destination.mkdir(parents=True, exist_ok=True)
        count = 0
        with self._lock, self._connect() as connection:
            for row in connection.execute(
                "SELECT run_id, result_json FROM results ORDER BY run_id"
            ):
                path = destination / (row["run_id"] + ".json")
                path.write_text(row["result_json"] + "\n", encoding="utf-8")
                count += 1
        return count
