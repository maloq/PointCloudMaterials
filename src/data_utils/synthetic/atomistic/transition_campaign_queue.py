from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from string import hexdigits

import numpy as np

from .transition_campaign_config import TransitionCampaignConfig


@dataclass(frozen=True)
class TransitionCampaignTask:
    task_index: int
    run_name: str
    branch_index: int
    branch_name: str
    replica_index: int
    configured_replica_seed: int
    simulation_seed: int
    claim_role: str
    claim_generation: int
    claim_token: str

    @property
    def __dict__(self) -> dict[str, object]:
        """Keep execution leases out of persisted scientific task identity.

        Campaign/checkpoint producers already serialize ``task.__dict__``.  A lease is
        deliberately process-local execution authority: binding it into raw artifacts or
        checkpoint identity would make a correctly reclaimed task impossible to resume.
        """

        return {
            "task_index": self.task_index,
            "run_name": self.run_name,
            "branch_index": self.branch_index,
            "branch_name": self.branch_name,
            "replica_index": self.replica_index,
            "configured_replica_seed": self.configured_replica_seed,
            "simulation_seed": self.simulation_seed,
        }


def campaign_database_path(config: TransitionCampaignConfig) -> Path:
    return config.output_root / "transition_campaign.sqlite3"


def validate_transition_queue_identity(config: TransitionCampaignConfig) -> None:
    database_path = campaign_database_path(config)
    if not database_path.is_file():
        raise FileNotFoundError(
            f"{database_path}: transition queue must be initialized before a worker starts."
        )
    connection = _connect(config)
    try:
        row = connection.execute(
            "SELECT value_json FROM campaign_metadata WHERE key='campaign_config'"
        ).fetchone()
    finally:
        connection.close()
    expected = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    if row is None or row["value_json"] != expected:
        raise RuntimeError(
            f"{database_path}: worker campaign/source identity differs from the initialized "
            "queue. Refusing to claim a task from replaced source content."
        )


def _connect(config: TransitionCampaignConfig) -> sqlite3.Connection:
    connection = sqlite3.connect(
        campaign_database_path(config), timeout=60.0, isolation_level=None
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 60000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    return connection


def _expected_tasks(config: TransitionCampaignConfig) -> list[tuple[object, ...]]:
    tasks: list[tuple[object, ...]] = []
    task_index = 0
    for branch_index, branch in enumerate(config.transition.temperature_runs):
        for replica_index, configured_seed in enumerate(config.transition.random_seeds):
            simulation_seed = int(
                np.random.SeedSequence([configured_seed, branch_index]).generate_state(1)[0]
            )
            tasks.append(
                (
                    task_index,
                    f"{branch.name}/replica_{replica_index:03d}",
                    branch_index,
                    branch.name,
                    replica_index,
                    configured_seed,
                    simulation_seed,
                )
            )
            task_index += 1
    return tasks


def initialize_transition_queue(
    config: TransitionCampaignConfig, *, retry_failed: bool
) -> None:
    database_path = campaign_database_path(config)
    if config.output_root.exists() and not database_path.exists():
        allowed_bootstrap_entries = {"transition_campaign.lock"}
        unexpected = sorted(
            path.name
            for path in config.output_root.iterdir()
            if path.name not in allowed_bootstrap_entries
        )
        if unexpected:
            raise FileExistsError(
                f"{config.output_root}: non-campaign output already exists with entries="
                f"{unexpected}. Select a new transition output root; queued execution "
                "will not adopt or overwrite ambiguous legacy artifacts."
            )
    config.output_root.mkdir(parents=True, exist_ok=True)
    connection = _connect(config)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS campaign_metadata (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_index INTEGER PRIMARY KEY,
                run_name TEXT NOT NULL UNIQUE,
                branch_index INTEGER NOT NULL,
                branch_name TEXT NOT NULL,
                replica_index INTEGER NOT NULL,
                configured_replica_seed INTEGER NOT NULL,
                simulation_seed INTEGER NOT NULL UNIQUE,
                md_status TEXT NOT NULL CHECK (
                    md_status IN ('queued', 'running', 'complete', 'failed')
                ),
                analysis_status TEXT NOT NULL CHECK (
                    analysis_status IN ('blocked', 'pending', 'running', 'complete', 'failed')
                ),
                md_worker TEXT,
                analysis_worker TEXT,
                md_claim_generation INTEGER NOT NULL DEFAULT 0,
                md_claim_token TEXT,
                analysis_claim_generation INTEGER NOT NULL DEFAULT 0,
                analysis_claim_token TEXT,
                raw_directory TEXT,
                raw_commit_sha256 TEXT,
                analysis_directory TEXT,
                analysis_commit_sha256 TEXT,
                md_error TEXT,
                analysis_error TEXT
            )
            """
        )
        existing_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
        }
        for column, declaration in (
            ("md_claim_generation", "INTEGER NOT NULL DEFAULT 0"),
            ("md_claim_token", "TEXT"),
            ("analysis_claim_generation", "INTEGER NOT NULL DEFAULT 0"),
            ("analysis_claim_token", "TEXT"),
        ):
            if column not in existing_columns:
                connection.execute(f"ALTER TABLE tasks ADD COLUMN {column} {declaration}")
        serialized = json.dumps(
            config.to_dict(), sort_keys=True, separators=(",", ":")
        )
        row = connection.execute(
            "SELECT value_json FROM campaign_metadata WHERE key='campaign_config'"
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO campaign_metadata(key, value_json) VALUES (?, ?)",
                ("campaign_config", serialized),
            )
        elif row["value_json"] != serialized:
            raise RuntimeError(
                f"{campaign_database_path(config)}: persisted campaign configuration "
                "differs from the requested configuration. Resume with the exact original "
                "campaign file or choose a new transition output root."
            )

        expected = _expected_tasks(config)
        existing = connection.execute(
            "SELECT task_index, run_name, branch_index, branch_name, replica_index, "
            "configured_replica_seed, simulation_seed FROM tasks ORDER BY task_index"
        ).fetchall()
        if not existing:
            connection.executemany(
                "INSERT INTO tasks(task_index, run_name, branch_index, branch_name, "
                "replica_index, configured_replica_seed, simulation_seed, md_status, "
                "analysis_status) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 'blocked')",
                expected,
            )
        elif [tuple(row) for row in existing] != expected:
            raise RuntimeError(
                f"{campaign_database_path(config)}: persisted task assignment differs "
                "from the configured temperature/replica Cartesian product."
            )

        connection.execute(
            "UPDATE tasks SET md_status='queued', md_worker=NULL, md_claim_token=NULL "
            "WHERE md_status='running'"
        )
        connection.execute(
            "UPDATE tasks SET analysis_status='pending', analysis_worker=NULL, "
            "analysis_claim_token=NULL "
            "WHERE md_status='complete' AND analysis_status='running'"
        )
        if retry_failed:
            connection.execute(
                "UPDATE tasks SET md_status='queued', analysis_status='blocked', "
                "md_worker=NULL, md_claim_token=NULL, md_error=NULL, raw_directory=NULL, "
                "raw_commit_sha256=NULL, analysis_directory=NULL, "
                "analysis_commit_sha256=NULL WHERE md_status='failed'"
            )
            connection.execute(
                "UPDATE tasks SET analysis_status='pending', analysis_worker=NULL, "
                "analysis_claim_token=NULL, analysis_error=NULL, analysis_directory=NULL, "
                "analysis_commit_sha256=NULL WHERE md_status='complete' "
                "AND analysis_status='failed'"
            )
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def _task(
    row: sqlite3.Row,
    *,
    claim_role: str,
    claim_generation: int,
    claim_token: str,
) -> TransitionCampaignTask:
    return TransitionCampaignTask(
        task_index=int(row["task_index"]),
        run_name=str(row["run_name"]),
        branch_index=int(row["branch_index"]),
        branch_name=str(row["branch_name"]),
        replica_index=int(row["replica_index"]),
        configured_replica_seed=int(row["configured_replica_seed"]),
        simulation_seed=int(row["simulation_seed"]),
        claim_role=claim_role,
        claim_generation=claim_generation,
        claim_token=claim_token,
    )


def _claim(
    config: TransitionCampaignConfig, *, worker_name: str, analysis: bool
) -> TransitionCampaignTask | None:
    connection = _connect(config)
    try:
        connection.execute("BEGIN IMMEDIATE")
        condition = (
            "md_status='complete' AND analysis_status='pending'"
            if analysis
            else "md_status='queued'"
        )
        row = connection.execute(
            "SELECT task_index, run_name, branch_index, branch_name, replica_index, "
            "configured_replica_seed, simulation_seed, md_claim_generation, "
            "analysis_claim_generation FROM tasks WHERE "
            f"{condition} ORDER BY task_index LIMIT 1"
        ).fetchone()
        if row is None:
            connection.execute("COMMIT")
            return None
        claim_token = uuid.uuid4().hex
        if analysis:
            generation = int(row["analysis_claim_generation"]) + 1
            updated = connection.execute(
                "UPDATE tasks SET analysis_status='running', analysis_worker=?, "
                "analysis_claim_generation=?, analysis_claim_token=? "
                "WHERE task_index=? AND analysis_status='pending'",
                (worker_name, generation, claim_token, row["task_index"]),
            ).rowcount
        else:
            generation = int(row["md_claim_generation"]) + 1
            updated = connection.execute(
                "UPDATE tasks SET md_status='running', md_worker=?, "
                "md_claim_generation=?, md_claim_token=? WHERE task_index=? "
                "AND md_status='queued'",
                (worker_name, generation, claim_token, row["task_index"]),
            ).rowcount
        if updated != 1:
            connection.execute("ROLLBACK")
            raise RuntimeError(
                f"Failed to atomically claim transition task index={row['task_index']}."
            )
        connection.execute("COMMIT")
        return _task(
            row,
            claim_role="analysis" if analysis else "md",
            claim_generation=generation,
            claim_token=claim_token,
        )
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def claim_md_task(
    config: TransitionCampaignConfig, *, worker_name: str
) -> TransitionCampaignTask | None:
    return _claim(config, worker_name=worker_name, analysis=False)


def claim_analysis_task(
    config: TransitionCampaignConfig, *, worker_name: str
) -> TransitionCampaignTask | None:
    return _claim(config, worker_name=worker_name, analysis=True)


def _digest(value: str, *, context: str) -> None:
    if len(value) != 64 or any(character not in hexdigits for character in value):
        raise ValueError(f"{context} must be a 64-character SHA-256 digest, got {value!r}.")


def _validate_claim(task: TransitionCampaignTask, *, role: str) -> None:
    if task.claim_role != role:
        raise ValueError(
            f"{task.run_name}: {role} transition requires a {role} claim, got "
            f"claim_role={task.claim_role!r}."
        )
    if task.claim_generation <= 0 or len(task.claim_token) != 32:
        raise ValueError(
            f"{task.run_name}: invalid {role} claim authority: generation="
            f"{task.claim_generation}, token={task.claim_token!r}."
        )


def complete_md_task(
    config: TransitionCampaignConfig,
    *,
    task: TransitionCampaignTask,
    raw_directory: Path,
    raw_commit_sha256: str,
) -> None:
    _validate_claim(task, role="md")
    _digest(raw_commit_sha256, context=f"{task.run_name} raw_commit_sha256")
    connection = _connect(config)
    try:
        updated = connection.execute(
            "UPDATE tasks SET md_status='complete', analysis_status='pending', "
            "raw_directory=?, raw_commit_sha256=?, md_error=NULL WHERE task_index=? "
            "AND md_status='running' AND md_claim_generation=? AND md_claim_token=?",
            (
                str(raw_directory),
                raw_commit_sha256,
                task.task_index,
                task.claim_generation,
                task.claim_token,
            ),
        ).rowcount
        if updated != 1:
            raise RuntimeError(f"{task.run_name}: cannot transition MD to complete.")
    finally:
        connection.close()


def complete_analysis_task(
    config: TransitionCampaignConfig,
    *,
    task: TransitionCampaignTask,
    analysis_directory: Path,
    analysis_commit_sha256: str,
) -> None:
    _validate_claim(task, role="analysis")
    _digest(
        analysis_commit_sha256, context=f"{task.run_name} analysis_commit_sha256"
    )
    connection = _connect(config)
    try:
        updated = connection.execute(
            "UPDATE tasks SET analysis_status='complete', analysis_directory=?, "
            "analysis_commit_sha256=?, analysis_error=NULL WHERE task_index=? "
            "AND analysis_status='running' AND analysis_claim_generation=? "
            "AND analysis_claim_token=?",
            (
                str(analysis_directory),
                analysis_commit_sha256,
                task.task_index,
                task.claim_generation,
                task.claim_token,
            ),
        ).rowcount
        if updated != 1:
            raise RuntimeError(f"{task.run_name}: cannot transition analysis to complete.")
    finally:
        connection.close()


def fail_task(
    config: TransitionCampaignConfig,
    *,
    task: TransitionCampaignTask,
    error: str,
    analysis: bool,
) -> None:
    role = "analysis" if analysis else "md"
    _validate_claim(task, role=role)
    connection = _connect(config)
    try:
        if analysis:
            updated = connection.execute(
                "UPDATE tasks SET analysis_status='failed', analysis_error=? "
                "WHERE task_index=? AND analysis_status='running' "
                "AND analysis_claim_generation=? AND analysis_claim_token=?",
                (
                    error,
                    task.task_index,
                    task.claim_generation,
                    task.claim_token,
                ),
            ).rowcount
        else:
            updated = connection.execute(
                "UPDATE tasks SET md_status='failed', analysis_status='blocked', "
                "md_error=? WHERE task_index=? AND md_status='running' "
                "AND md_claim_generation=? AND md_claim_token=?",
                (
                    error,
                    task.task_index,
                    task.claim_generation,
                    task.claim_token,
                ),
            ).rowcount
        if updated != 1:
            raise RuntimeError(
                f"{task.run_name}: cannot transition failed "
                f"{'analysis' if analysis else 'MD'} task."
            )
    finally:
        connection.close()


def campaign_rows(config: TransitionCampaignConfig) -> list[dict[str, object]]:
    connection = _connect(config)
    try:
        return [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM tasks ORDER BY task_index"
            ).fetchall()
        ]
    finally:
        connection.close()


def campaign_row(
    config: TransitionCampaignConfig, *, task_index: int
) -> dict[str, object]:
    connection = _connect(config)
    try:
        row = connection.execute(
            "SELECT * FROM tasks WHERE task_index=?", (task_index,)
        ).fetchone()
        if row is None:
            raise RuntimeError(
                f"{campaign_database_path(config)}: no task row at index={task_index}."
            )
        return dict(row)
    finally:
        connection.close()
