from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from string import hexdigits

from .homogeneous_campaign_config import (
    HomogeneousCampaignConfig,
    campaign_config_matches_after_path_relocation,
)


@dataclass(frozen=True)
class CampaignReplicaTask:
    replica_index: int
    replica_name: str
    random_seed: int


def campaign_database_path(config: HomogeneousCampaignConfig) -> Path:
    return config.output_root / "campaign.sqlite3"


def _connect(config: HomogeneousCampaignConfig) -> sqlite3.Connection:
    connection = sqlite3.connect(
        campaign_database_path(config), timeout=60.0, isolation_level=None
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 60000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    return connection


def initialize_campaign_queue(
    config: HomogeneousCampaignConfig,
    *,
    retry_failed: bool,
) -> None:
    config.output_root.mkdir(parents=True, exist_ok=True)
    connection = _connect(config)
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS campaign_metadata (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS replicas (
                replica_index INTEGER PRIMARY KEY,
                replica_name TEXT NOT NULL UNIQUE,
                random_seed INTEGER NOT NULL UNIQUE,
                md_status TEXT NOT NULL CHECK (
                    md_status IN ('queued', 'running', 'complete', 'failed')
                ),
                analysis_status TEXT NOT NULL CHECK (
                    analysis_status IN ('blocked', 'pending', 'running', 'complete', 'failed')
                ),
                md_worker TEXT,
                analysis_worker TEXT,
                outcome TEXT,
                raw_directory TEXT,
                run_metadata_sha256 TEXT,
                online_threshold_event_json TEXT,
                full_analysis_sha256 TEXT,
                md_error TEXT,
                analysis_error TEXT
            );
            """
        )
        existing_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(replicas)").fetchall()
        }
        for column_name in (
            "run_metadata_sha256",
            "online_threshold_event_json",
            "full_analysis_sha256",
        ):
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE replicas ADD COLUMN {column_name} TEXT"
                )
        serialized_config = json.dumps(
            config.to_dict(), sort_keys=True, separators=(",", ":")
        )
        observed = connection.execute(
            "SELECT value_json FROM campaign_metadata WHERE key='campaign_config'"
        ).fetchone()
        if observed is None:
            connection.execute(
                "INSERT INTO campaign_metadata(key, value_json) VALUES (?, ?)",
                ("campaign_config", serialized_config),
            )
        elif observed["value_json"] != serialized_config:
            observed_config = json.loads(observed["value_json"])
            if not campaign_config_matches_after_path_relocation(
                observed_config, config.to_dict()
            ):
                raise RuntimeError(
                    f"{campaign_database_path(config)}: persisted campaign configuration "
                    "differs from the requested configuration beyond repository config "
                    "file relocation. Use the original physical configuration or select "
                    "a new output_root."
                )
            connection.execute(
                "UPDATE campaign_metadata SET value_json=? WHERE key='campaign_config'",
                (serialized_config,),
            )
        existing = connection.execute(
            "SELECT replica_index, replica_name, random_seed FROM replicas "
            "ORDER BY replica_index"
        ).fetchall()
        expected = [
            (index, f"replica_{index:03d}", seed)
            for index, seed in enumerate(config.homogeneous.random_seeds)
        ]
        if not existing:
            connection.executemany(
                "INSERT INTO replicas(replica_index, replica_name, random_seed, "
                "md_status, analysis_status) VALUES (?, ?, ?, 'queued', 'blocked')",
                expected,
            )
        else:
            observed_rows = [
                (row["replica_index"], row["replica_name"], row["random_seed"])
                for row in existing
            ]
            if observed_rows != expected:
                raise RuntimeError(
                    f"{campaign_database_path(config)}: persisted replica assignment "
                    f"{observed_rows} differs from configured assignment {expected}."
                )
        connection.execute(
            "UPDATE replicas SET md_status='queued', analysis_status='blocked', "
            "md_worker=NULL, analysis_worker=NULL, outcome=NULL, raw_directory=NULL, "
            "run_metadata_sha256=NULL, online_threshold_event_json=NULL, "
            "full_analysis_sha256=NULL WHERE md_status='running'"
        )
        connection.execute(
            "UPDATE replicas SET analysis_status='pending', analysis_worker=NULL, "
            "full_analysis_sha256=NULL "
            "WHERE analysis_status='running' AND md_status='complete'"
        )
        # A database created by the pre-anchor implementation can contain completed
        # rows without an external digest/event commit. Requeue those rows so the raw
        # artifacts are revalidated and anchored before analysis/finalize.
        connection.execute(
            "UPDATE replicas SET md_status='queued', analysis_status='blocked', "
            "md_worker=NULL, analysis_worker=NULL, outcome=NULL, raw_directory=NULL, "
            "run_metadata_sha256=NULL, online_threshold_event_json=NULL, "
            "full_analysis_sha256=NULL WHERE md_status='complete' AND "
            "(run_metadata_sha256 IS NULL OR online_threshold_event_json IS NULL)"
        )
        connection.execute(
            "UPDATE replicas SET analysis_status='pending', analysis_worker=NULL, "
            "full_analysis_sha256=NULL WHERE md_status='complete' AND "
            "analysis_status='complete' AND full_analysis_sha256 IS NULL"
        )
        if retry_failed:
            connection.execute(
                "UPDATE replicas SET md_status='queued', analysis_status='blocked', "
                "md_worker=NULL, analysis_worker=NULL, md_error=NULL, "
                "analysis_error=NULL, outcome=NULL, raw_directory=NULL, "
                "run_metadata_sha256=NULL, online_threshold_event_json=NULL, "
                "full_analysis_sha256=NULL WHERE md_status='failed'"
            )
            connection.execute(
                "UPDATE replicas SET analysis_status='pending', analysis_worker=NULL, "
                "analysis_error=NULL, full_analysis_sha256=NULL "
                "WHERE md_status='complete' "
                "AND analysis_status='failed'"
            )
    finally:
        connection.close()


def claim_md_task(
    config: HomogeneousCampaignConfig, *, worker_name: str
) -> CampaignReplicaTask | None:
    connection = _connect(config)
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT replica_index, replica_name, random_seed FROM replicas "
            "WHERE md_status='queued' ORDER BY replica_index LIMIT 1"
        ).fetchone()
        if row is None:
            connection.execute("COMMIT")
            return None
        updated = connection.execute(
            "UPDATE replicas SET md_status='running', md_worker=? "
            "WHERE replica_index=? AND md_status='queued'",
            (worker_name, row["replica_index"]),
        ).rowcount
        if updated != 1:
            connection.execute("ROLLBACK")
            raise RuntimeError(
                f"Failed to atomically claim replica index={row['replica_index']}."
            )
        connection.execute("COMMIT")
        return CampaignReplicaTask(
            replica_index=int(row["replica_index"]),
            replica_name=str(row["replica_name"]),
            random_seed=int(row["random_seed"]),
        )
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def complete_md_task(
    config: HomogeneousCampaignConfig,
    *,
    task: CampaignReplicaTask,
    outcome: str,
    raw_directory: Path,
    run_metadata_sha256: str,
    online_threshold_event: dict[str, object],
) -> None:
    if (
        len(run_metadata_sha256) != 64
        or any(character not in hexdigits for character in run_metadata_sha256)
    ):
        raise ValueError(
            f"Replica {task.replica_name} run_metadata_sha256 must be a 64-character "
            f"hexadecimal digest, got {run_metadata_sha256!r}."
        )
    online_threshold_event_json = json.dumps(
        online_threshold_event,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    connection = _connect(config)
    try:
        updated = connection.execute(
            "UPDATE replicas SET md_status='complete', analysis_status='pending', "
            "outcome=?, raw_directory=?, run_metadata_sha256=?, "
            "online_threshold_event_json=?, full_analysis_sha256=NULL, md_error=NULL "
            "WHERE replica_index=? "
            "AND md_status='running'",
            (
                outcome,
                str(raw_directory),
                run_metadata_sha256,
                online_threshold_event_json,
                task.replica_index,
            ),
        ).rowcount
        if updated != 1:
            raise RuntimeError(
                f"Replica {task.replica_name} cannot transition from running to complete."
            )
    finally:
        connection.close()


def fail_md_task(
    config: HomogeneousCampaignConfig,
    *,
    task: CampaignReplicaTask,
    error: str,
) -> None:
    connection = _connect(config)
    try:
        updated = connection.execute(
            "UPDATE replicas SET md_status='failed', analysis_status='blocked', "
            "md_error=? WHERE replica_index=? AND md_status='running'",
            (error, task.replica_index),
        ).rowcount
        if updated != 1:
            raise RuntimeError(
                f"Replica {task.replica_name} cannot transition from running to failed."
            )
    finally:
        connection.close()


def claim_analysis_task(
    config: HomogeneousCampaignConfig, *, worker_name: str
) -> CampaignReplicaTask | None:
    connection = _connect(config)
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT replica_index, replica_name, random_seed FROM replicas "
            "WHERE md_status='complete' AND analysis_status='pending' "
            "ORDER BY replica_index LIMIT 1"
        ).fetchone()
        if row is None:
            connection.execute("COMMIT")
            return None
        updated = connection.execute(
            "UPDATE replicas SET analysis_status='running', analysis_worker=? "
            "WHERE replica_index=? AND analysis_status='pending'",
            (worker_name, row["replica_index"]),
        ).rowcount
        if updated != 1:
            connection.execute("ROLLBACK")
            raise RuntimeError(
                f"Failed to atomically claim analysis for index={row['replica_index']}."
            )
        connection.execute("COMMIT")
        return CampaignReplicaTask(
            replica_index=int(row["replica_index"]),
            replica_name=str(row["replica_name"]),
            random_seed=int(row["random_seed"]),
        )
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def complete_analysis_task(
    config: HomogeneousCampaignConfig,
    *,
    task: CampaignReplicaTask,
    full_analysis_sha256: str,
) -> None:
    if (
        len(full_analysis_sha256) != 64
        or any(character not in hexdigits for character in full_analysis_sha256)
    ):
        raise ValueError(
            f"Replica {task.replica_name} full_analysis_sha256 must be a 64-character "
            f"hexadecimal digest, got {full_analysis_sha256!r}."
        )
    connection = _connect(config)
    try:
        updated = connection.execute(
            "UPDATE replicas SET analysis_status='complete', full_analysis_sha256=?, "
            "analysis_error=NULL "
            "WHERE replica_index=? AND analysis_status='running'",
            (full_analysis_sha256, task.replica_index),
        ).rowcount
        if updated != 1:
            raise RuntimeError(
                f"Replica {task.replica_name} analysis cannot transition to complete."
            )
    finally:
        connection.close()


def fail_analysis_task(
    config: HomogeneousCampaignConfig,
    *,
    task: CampaignReplicaTask,
    error: str,
) -> None:
    connection = _connect(config)
    try:
        updated = connection.execute(
            "UPDATE replicas SET analysis_status='failed', analysis_error=? "
            "WHERE replica_index=? AND analysis_status='running'",
            (error, task.replica_index),
        ).rowcount
        if updated != 1:
            raise RuntimeError(
                f"Replica {task.replica_name} analysis cannot transition to failed."
            )
    finally:
        connection.close()


def campaign_rows(config: HomogeneousCampaignConfig) -> list[dict[str, object]]:
    connection = _connect(config)
    try:
        rows = connection.execute(
            "SELECT * FROM replicas ORDER BY replica_index"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def campaign_row(
    config: HomogeneousCampaignConfig, *, replica_index: int
) -> dict[str, object]:
    connection = _connect(config)
    try:
        row = connection.execute(
            "SELECT * FROM replicas WHERE replica_index=?", (replica_index,)
        ).fetchone()
        if row is None:
            raise RuntimeError(
                f"{campaign_database_path(config)}: no replica row at index="
                f"{replica_index}."
            )
        return dict(row)
    finally:
        connection.close()
