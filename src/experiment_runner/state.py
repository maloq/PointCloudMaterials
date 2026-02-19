"""Persistent state for experiment runs, enabling --resume."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


STATE_FILENAME = "state.json"


class RunState:
    """Tracks the progress of an experiment plan execution.

    Persisted to state.json in the output directory after every mutation,
    so that --resume can recover from interruptions.
    """

    def __init__(self, output_dir: Path, plan_name: str) -> None:
        self.output_dir = output_dir
        self.plan_name = plan_name
        self.created_at: str = datetime.now().isoformat()
        self.stages: Dict[str, StageState] = {}
        self._path = output_dir / STATE_FILENAME

    def get_or_create_stage(self, stage_name: str) -> "StageState":
        if stage_name not in self.stages:
            self.stages[stage_name] = StageState(name=stage_name)
        return self.stages[stage_name]

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "plan_name": self.plan_name,
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(),
            "stages": {
                name: stage.to_dict() for name, stage in self.stages.items()
            },
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.rename(self._path)

    @classmethod
    def load(cls, output_dir: Path) -> "RunState":
        path = output_dir / STATE_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"No state file found at {path}.")

        data = json.loads(path.read_text())
        state = cls(output_dir=output_dir, plan_name=data["plan_name"])
        state.created_at = data.get("created_at", "")
        for stage_name, stage_data in data.get("stages", {}).items():
            state.stages[stage_name] = StageState.from_dict(stage_data)
        return state

    def is_stage_complete(self, stage_name: str) -> bool:
        stage = self.stages.get(stage_name)
        if stage is None:
            return False
        return stage.status in {"completed", "completed_with_errors"}


class StageState:
    def __init__(self, name: str) -> None:
        self.name = name
        self.status: str = "pending"  # pending | running | completed | completed_with_errors | failed
        self.jobs: Dict[str, JobState] = {}
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None

    def record_job(
        self,
        experiment_name: str,
        *,
        job_id: Optional[str] = None,
        run_dir: Optional[str] = None,
        status: str = "submitted",
    ) -> "JobState":
        js = JobState(
            experiment_name=experiment_name,
            job_id=job_id,
            run_dir=run_dir,
            status=status,
        )
        self.jobs[experiment_name] = js
        return js

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "jobs": {name: j.to_dict() for name, j in self.jobs.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageState":
        stage = cls(name=data["name"])
        stage.status = data.get("status", "pending")
        stage.started_at = data.get("started_at")
        stage.finished_at = data.get("finished_at")
        for name, jdata in data.get("jobs", {}).items():
            stage.jobs[name] = JobState.from_dict(jdata)
        return stage


class JobState:
    def __init__(
        self,
        experiment_name: str,
        *,
        job_id: Optional[str] = None,
        run_dir: Optional[str] = None,
        status: str = "pending",
    ) -> None:
        self.experiment_name = experiment_name
        self.job_id = job_id
        self.run_dir = run_dir
        self.status = status
        self.checkpoint_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "job_id": self.job_id,
            "run_dir": self.run_dir,
            "status": self.status,
            "checkpoint_path": self.checkpoint_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobState":
        js = cls(
            experiment_name=data["experiment_name"],
            job_id=data.get("job_id"),
            run_dir=data.get("run_dir"),
            status=data.get("status", "pending"),
        )
        js.checkpoint_path = data.get("checkpoint_path")
        return js
