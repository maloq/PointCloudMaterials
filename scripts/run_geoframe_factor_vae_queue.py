#!/usr/bin/env python3
"""Run the paired GeoFrame V1/V2 FactorVAE sweep sequentially on one GPU."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "src/training_methods/contrastive_learning/train_contrastive.py"
V1_CHECKPOINT = (
    REPO_ROOT
    / "output/detached/vicreg_geoframe_corrected_h100_20260829_220250"
    / "VICREG_GEOFRAME_MULTISCALE_8_16_l128_CORRECTED_H100_N160_M80_"
    "GeoFrameTransformer-epoch=159.ckpt"
)


@dataclass(frozen=True)
class FactorSetting:
    label: str
    enabled: bool
    gamma: float
    noise_std: float
    discriminator_lr: float
    update_interval: int


@dataclass(frozen=True)
class RunSpec:
    index: int
    label: str
    encoder: str
    config_name: str
    experiment_name: str
    epochs: int
    factor: FactorSetting | None
    run_dir: str
    wandb_run_id: str


FACTOR_SETTINGS = (
    FactorSetting("control", False, 0.1, 0.0, 3.0e-6, 2),
    FactorSetting("midpoint_n0", True, 0.1, 0.0, 3.0e-6, 2),
    FactorSetting("noise_005", True, 0.1, 0.05, 3.0e-6, 2),
    FactorSetting("noise_010", True, 0.1, 0.10, 3.0e-6, 2),
    FactorSetting("noise_025", True, 0.1, 0.25, 3.0e-6, 2),
    FactorSetting("noise_050", True, 0.1, 0.50, 3.0e-6, 2),
    FactorSetting("gamma_003_n010", True, 0.03, 0.10, 3.0e-6, 2),
    FactorSetting("gamma_030_n010", True, 0.30, 0.10, 3.0e-6, 2),
    FactorSetting("weak_d_n010", True, 0.1, 0.10, 1.0e-6, 4),
    FactorSetting("strong_d_n010", True, 0.1, 0.10, 1.0e-5, 1),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _wandb_id(queue_name: str, run_label: str) -> str:
    return hashlib.sha256(f"{queue_name}:{run_label}".encode("utf-8")).hexdigest()[:8]


def _build_specs(output_root: Path, fine_tune_epochs: int, pretrain_epochs: int) -> list[RunSpec]:
    specs = [
        RunSpec(
            index=0,
            label="v2_pretrain",
            encoder="V2",
            config_name="vicreg_geo_frame_transformer_v2",
            experiment_name="GF_FACTOR_SWEEP_V2_PRETRAIN_S123",
            epochs=pretrain_epochs,
            factor=None,
            run_dir=str(output_root / "00_v2_pretrain"),
            wandb_run_id=_wandb_id(output_root.name, "v2_pretrain"),
        )
    ]
    index = 1
    for setting in FACTOR_SETTINGS:
        for encoder, config_name in (
            ("V1", "vicreg_geo_frame_factor_vae_sweep_v1"),
            ("V2", "vicreg_geo_frame_factor_vae_sweep_v2"),
        ):
            label = f"{encoder.lower()}_{setting.label}"
            specs.append(
                RunSpec(
                    index=index,
                    label=label,
                    encoder=encoder,
                    config_name=config_name,
                    experiment_name=f"GF_FACTOR_SWEEP_{encoder}_{setting.label.upper()}_S123",
                    epochs=fine_tune_epochs,
                    factor=setting,
                    run_dir=str(output_root / f"{index:02d}_{label}"),
                    wandb_run_id=_wandb_id(output_root.name, label),
                )
            )
            index += 1
    return specs


def _expected_periodic_checkpoint(spec: RunSpec) -> Path:
    final_epoch = spec.epochs - 1
    return Path(spec.run_dir) / (
        f"{spec.experiment_name}-periodic-epoch={final_epoch:03d}.ckpt"
    )


def _checkpoint_monitor_summary(checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_epoch = int(checkpoint_path.stem.rsplit("=", maxsplit=1)[-1])
    actual_epoch = int(payload.get("epoch", -1))
    if actual_epoch != expected_epoch:
        raise RuntimeError(
            f"Checkpoint epoch mismatch for {checkpoint_path}: "
            f"filename epoch={expected_epoch}, payload epoch={actual_epoch}."
        )

    monitored_callbacks = []
    for callback_state in payload.get("callbacks", {}).values():
        if not isinstance(callback_state, dict):
            continue
        monitor = callback_state.get("monitor")
        if monitor is None:
            continue
        score = callback_state.get("best_model_score")
        monitored_callbacks.append(
            {
                "monitor": str(monitor),
                "best_score": None if score is None else float(score),
                "best_checkpoint": str(callback_state.get("best_model_path", "")),
            }
        )
    if len(monitored_callbacks) != 1:
        raise RuntimeError(
            f"Expected exactly one monitored checkpoint callback in {checkpoint_path}, "
            f"found {len(monitored_callbacks)}: {monitored_callbacks}."
        )
    return {
        "epoch": actual_epoch,
        "global_step": int(payload.get("global_step", -1)),
        **monitored_callbacks[0],
    }


def _write_results_csv(path: Path, run_states: list[dict[str, Any]]) -> None:
    fieldnames = (
        "index",
        "label",
        "encoder",
        "state",
        "elapsed_seconds",
        "factor_enabled",
        "gamma",
        "noise_std",
        "discriminator_lr",
        "update_interval",
        "monitor",
        "best_score",
        "best_checkpoint",
        "final_checkpoint",
        "wandb_run_id",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for state in run_states:
            factor = state.get("factor") or {}
            result = state.get("result") or {}
            writer.writerow(
                {
                    "index": state["index"],
                    "label": state["label"],
                    "encoder": state["encoder"],
                    "state": state["state"],
                    "elapsed_seconds": state.get("elapsed_seconds"),
                    "factor_enabled": factor.get("enabled"),
                    "gamma": factor.get("gamma"),
                    "noise_std": factor.get("noise_std"),
                    "discriminator_lr": factor.get("discriminator_lr"),
                    "update_interval": factor.get("update_interval"),
                    "monitor": result.get("monitor"),
                    "best_score": result.get("best_score"),
                    "best_checkpoint": result.get("best_checkpoint"),
                    "final_checkpoint": state.get("final_checkpoint"),
                    "wandb_run_id": state["wandb_run_id"],
                }
            )


def _record_failed_run(
    *,
    output_root: Path,
    manifest: dict[str, Any],
    run_states: list[dict[str, Any]],
    spec: RunSpec,
    state: dict[str, Any],
    message: str,
) -> None:
    state["state"] = "failed"
    state["error"] = message
    state["finished_at"] = _utc_now()
    _write_json(
        output_root / "queue_status.json",
        {
            "state": "failed",
            "failed_run_index": spec.index,
            "failed_run_label": spec.label,
            "error": message,
            **{key: value for key, value in manifest.items() if key != "runs"},
            "runs": run_states,
        },
    )
    _write_results_csv(output_root / "results.csv", run_states)


def _base_command(spec: RunSpec, device: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        "--config-name",
        spec.config_name,
        f"hydra.run.dir={spec.run_dir}",
        "hydra.job.chdir=false",
        f"experiment_name={spec.experiment_name}",
        f"epochs={spec.epochs}",
        "devices=[0]",
        "run_post_training_analysis=false",
        "run_test_after_training=false",
        "++enable_model_summary=false",
        "++num_sanity_val_steps=0",
        "seed_everything=123",
    ]


def _run_command(spec: RunSpec, device: int, v2_checkpoint: Path | None) -> list[str]:
    command = _base_command(spec, device)
    if spec.factor is None:
        command.extend(
            [
                "factor_vae_enabled=false",
                "batch_size=8192",
                "learning_rate=1.0e-3",
                "warmup_epochs=10",
                "check_val_every_n_epoch=10",
                "checkpoint_monitor=loss/val",
                "checkpoint_mode=min",
                "checkpoint_every_n_epochs=40",
            ]
        )
        return command

    init_checkpoint = V1_CHECKPOINT if spec.encoder == "V1" else v2_checkpoint
    if init_checkpoint is None:
        raise RuntimeError(f"V2 checkpoint is unresolved before dependent run {spec.label}.")
    if not init_checkpoint.is_file():
        raise FileNotFoundError(
            f"Initialization checkpoint for {spec.label} does not exist: {init_checkpoint}"
        )

    factor = spec.factor
    command.extend(
        [
            f"init_from_checkpoint={json.dumps(str(init_checkpoint))}",
            f"factor_vae_enabled={str(factor.enabled).lower()}",
            f"factor_vae_gamma={factor.gamma}",
            f"factor_vae_latent_noise_std={factor.noise_std}",
            f"factor_vae_discriminator_learning_rate={factor.discriminator_lr}",
            f"factor_vae_discriminator_update_interval={factor.update_interval}",
        ]
    )
    return command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fine-tune-epochs", type=int, default=100)
    parser.add_argument("--v2-pretrain-epochs", type=int, default=160)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a failed or intentionally stopped queue after validating its "
            "manifest and completed checkpoint prefix."
        ),
    )
    return parser.parse_args()


def _queued_run_states(specs: list[RunSpec]) -> list[dict[str, Any]]:
    return [
        {
            **asdict(spec),
            "factor": None if spec.factor is None else asdict(spec.factor),
            "state": "queued",
        }
        for spec in specs
    ]


def _load_queue_for_resume(
    *,
    output_root: Path,
    specs: list[RunSpec],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = output_root / "manifest.json"
    status_path = output_root / "queue_status.json"
    if not manifest_path.is_file() or not status_path.is_file():
        raise FileNotFoundError(
            "--resume requires both queue state files; missing one of "
            f"{manifest_path} and {status_path}."
        )

    manifest = _read_json(manifest_path)
    status = _read_json(status_path)
    resume_state = status["state"]
    if resume_state not in {"failed", "stopped"}:
        raise RuntimeError(
            "--resume requires queue state 'failed' or 'stopped', "
            f"found {resume_state!r} in {status_path}."
        )

    expected_metadata = {
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "physical_cuda_device": args.device,
        "v1_checkpoint": str(V1_CHECKPOINT),
        "fine_tune_epochs": args.fine_tune_epochs,
        "v2_pretrain_epochs": args.v2_pretrain_epochs,
        "run_count": len(specs),
    }
    for key, expected_value in expected_metadata.items():
        if manifest[key] != expected_value:
            raise RuntimeError(
                f"Cannot resume queue because manifest field {key!r} changed: "
                f"saved={manifest[key]!r}, requested={expected_value!r}."
            )

    expected_initial_states = _queued_run_states(specs)
    if manifest["runs"] != expected_initial_states:
        raise RuntimeError(
            f"Cannot resume queue because saved run specifications in {manifest_path} "
            "do not match the requested queue."
        )

    run_states = status["runs"]
    if len(run_states) != len(expected_initial_states):
        raise RuntimeError(
            f"Cannot resume queue with {len(run_states)} status entries; "
            f"expected {len(expected_initial_states)}."
        )

    failed_indices = []
    encountered_unfinished = False
    for expected_state, state, spec in zip(
        expected_initial_states, run_states, specs, strict=True
    ):
        for key, expected_value in expected_state.items():
            if key == "state":
                continue
            if state[key] != expected_value:
                raise RuntimeError(
                    f"Cannot resume run {spec.index} because field {key!r} changed: "
                    f"saved={state[key]!r}, requested={expected_value!r}."
                )

        state_name = state["state"]
        if state_name == "completed":
            if encountered_unfinished:
                raise RuntimeError(
                    f"Cannot resume a non-prefix completion at run {spec.index} ({spec.label})."
                )
            expected_checkpoint = _expected_periodic_checkpoint(spec)
            if state.get("final_checkpoint") != str(expected_checkpoint):
                raise RuntimeError(
                    f"Completed run {spec.label} records unexpected final checkpoint "
                    f"{state.get('final_checkpoint')!r}; expected {str(expected_checkpoint)!r}."
                )
            if not expected_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Completed run {spec.label} checkpoint is missing: {expected_checkpoint}"
                )
            _checkpoint_monitor_summary(expected_checkpoint)
        elif state_name == "failed":
            encountered_unfinished = True
            failed_indices.append(spec.index)
        elif state_name == "queued":
            encountered_unfinished = True
        else:
            raise RuntimeError(
                f"Cannot resume run {spec.index} ({spec.label}) from state {state_name!r}."
            )

    if resume_state == "failed":
        if failed_indices != [status["failed_run_index"]]:
            raise RuntimeError(
                "Queue status must contain exactly its recorded failed run before resume; "
                f"recorded={status['failed_run_index']}, failed states={failed_indices}."
            )
    else:
        if failed_indices:
            raise RuntimeError(
                f"Stopped queue contains failed run states: {failed_indices}."
            )
        completed_indices = [
            state["index"] for state in run_states if state["state"] == "completed"
        ]
        if not completed_indices:
            raise RuntimeError("Stopped queue has no completed run prefix to resume.")
        if status["stopped_after_run_index"] != completed_indices[-1]:
            raise RuntimeError(
                "Stopped queue boundary does not match its completed prefix: "
                f"recorded={status['stopped_after_run_index']}, "
                f"last_completed={completed_indices[-1]}."
            )
    return manifest, run_states


def main() -> None:
    args = _parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "pointnet":
        raise RuntimeError(
            "The GeoFrame FactorVAE queue must run in conda environment 'pointnet'; "
            f"CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV')!r}."
        )
    if args.device < 0:
        raise ValueError(f"--device must be >= 0, got {args.device}.")
    if args.fine_tune_epochs < 1 or args.v2_pretrain_epochs < 1:
        raise ValueError(
            "Queue epoch counts must be positive; "
            f"fine_tune={args.fine_tune_epochs}, pretrain={args.v2_pretrain_epochs}."
        )
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"Training entrypoint does not exist: {TRAIN_SCRIPT}")
    if not V1_CHECKPOINT.is_file():
        raise FileNotFoundError(f"V1 initialization checkpoint does not exist: {V1_CHECKPOINT}")

    os.chdir(REPO_ROOT)
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    if args.resume:
        if not output_root.is_dir():
            raise FileNotFoundError(f"Queue output root does not exist: {output_root}")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"

    specs = _build_specs(output_root, args.fine_tune_epochs, args.v2_pretrain_epochs)
    if args.resume:
        manifest, run_states = _load_queue_for_resume(
            output_root=output_root,
            specs=specs,
            args=args,
        )
        print(f"[queue] RESUME validated saved queue at {output_root}", flush=True)
    else:
        if manifest_path.exists():
            raise FileExistsError(
                f"Queue manifest already exists at {manifest_path}; refusing to overwrite a prior queue."
            )
        run_states = _queued_run_states(specs)
        manifest = {
            "created_at": _utc_now(),
            "repo_root": str(REPO_ROOT),
            "python": sys.executable,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "physical_cuda_device": args.device,
            "v1_checkpoint": str(V1_CHECKPOINT),
            "fine_tune_epochs": args.fine_tune_epochs,
            "v2_pretrain_epochs": args.v2_pretrain_epochs,
            "run_count": len(specs),
            "runs": run_states,
        }
        _write_json(manifest_path, manifest)
        _write_json(output_root / "queue_status.json", {"state": "running", **manifest})
    (output_root / "queue_worker.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    _write_results_csv(output_root / "results.csv", run_states)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    environment["PYTHONUNBUFFERED"] = "1"
    v2_checkpoint: Path | None = None
    queue_start = time.monotonic()
    elapsed_before_restart = sum(
        float(state.get("elapsed_seconds", 0.0)) for state in run_states
    )

    for spec, state in zip(specs, run_states, strict=True):
        if state["state"] == "completed":
            if spec.factor is None:
                v2_checkpoint = Path(state["final_checkpoint"])
            continue

        run_dir = Path(spec.run_dir)
        if state["state"] == "failed":
            if not run_dir.is_dir():
                raise FileNotFoundError(
                    f"Failed run directory is missing before resume: {run_dir}"
                )
            unsafe_artifacts = sorted(run_dir.glob("*.ckpt"))
            if (run_dir / ".hydra").exists() or unsafe_artifacts:
                raise RuntimeError(
                    f"Automatic retry is only supported for a pre-launch failure, but {run_dir} "
                    f"contains .hydra or checkpoint artifacts: {unsafe_artifacts}."
                )
            attempt_history = state.setdefault("attempt_history", [])
            attempt_history.append(
                {
                    key: state[key]
                    for key in (
                        "state",
                        "started_at",
                        "finished_at",
                        "elapsed_seconds",
                        "return_code",
                        "process_id",
                        "command",
                        "error",
                    )
                    if key in state
                }
            )
            prior_attempt_count = int(state.get("attempt_count", 1))
            for key in (
                "started_at",
                "finished_at",
                "elapsed_seconds",
                "return_code",
                "process_id",
                "command",
                "error",
                "final_checkpoint",
                "result",
            ):
                state.pop(key, None)
            log_mode = "a"
        elif state["state"] == "queued":
            run_dir.mkdir(parents=True, exist_ok=False)
            prior_attempt_count = 0
            log_mode = "w"
        else:
            raise RuntimeError(
                f"Run {spec.index} ({spec.label}) has unsupported state {state['state']!r}."
            )

        command = _run_command(spec, args.device, v2_checkpoint)
        state.update(
            {
                "state": "running",
                "started_at": _utc_now(),
                "command": command,
                "attempt_count": prior_attempt_count + 1,
            }
        )
        status = {
            "state": "running",
            "active_run_index": spec.index,
            "active_run_label": spec.label,
            **{key: value for key, value in manifest.items() if key != "runs"},
            "runs": run_states,
        }
        _write_json(output_root / "queue_status.json", status)
        print(
            f"[queue] START {spec.index:02d}/{len(specs) - 1:02d} {spec.label} "
            f"at {state['started_at']}",
            flush=True,
        )

        run_start = time.monotonic()
        log_path = Path(spec.run_dir) / "train_contrastive.log"
        run_environment = environment.copy()
        run_environment["WANDB_RUN_ID"] = spec.wandb_run_id
        run_environment["WANDB_RESUME"] = "never"
        try:
            with log_path.open(log_mode, encoding="utf-8") as log_handle:
                if log_mode == "a":
                    log_handle.write(
                        f"\n[queue] RETRY attempt={state['attempt_count']} "
                        f"started_at={state['started_at']}\n"
                    )
                    log_handle.flush()
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=run_environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                state["process_id"] = process.pid
                _write_json(output_root / "queue_status.json", status)
                return_code = process.wait()
        except Exception as exc:
            message = f"Could not execute run {spec.label}: {exc}"
            _record_failed_run(
                output_root=output_root,
                manifest=manifest,
                run_states=run_states,
                spec=spec,
                state=state,
                message=message,
            )
            raise RuntimeError(message) from exc

        elapsed = time.monotonic() - run_start
        state["elapsed_seconds"] = elapsed
        state["finished_at"] = _utc_now()
        state["return_code"] = return_code
        if return_code != 0:
            message = (
                f"Training subprocess exited with code {return_code}; inspect {log_path}."
            )
            _record_failed_run(
                output_root=output_root,
                manifest=manifest,
                run_states=run_states,
                spec=spec,
                state=state,
                message=message,
            )
            raise RuntimeError(message)

        try:
            final_checkpoint = _expected_periodic_checkpoint(spec)
            if not final_checkpoint.is_file():
                available = sorted(path.name for path in Path(spec.run_dir).glob("*.ckpt"))
                raise FileNotFoundError(
                    f"Run {spec.label} exited successfully but its expected final checkpoint "
                    f"is missing: {final_checkpoint}. Available checkpoints: {available}"
                )
            result = _checkpoint_monitor_summary(final_checkpoint)
        except Exception as exc:
            message = f"Run {spec.label} artifact validation failed: {exc}"
            _record_failed_run(
                output_root=output_root,
                manifest=manifest,
                run_states=run_states,
                spec=spec,
                state=state,
                message=message,
            )
            raise RuntimeError(message) from exc
        state.update(
            {
                "state": "completed",
                "final_checkpoint": str(final_checkpoint),
                "result": result,
            }
        )
        if spec.factor is None:
            v2_checkpoint = final_checkpoint
        _write_results_csv(output_root / "results.csv", run_states)
        _write_json(output_root / "queue_status.json", status)
        print(
            f"[queue] DONE  {spec.index:02d}/{len(specs) - 1:02d} {spec.label} "
            f"elapsed={elapsed:.1f}s best_{result['monitor']}={result['best_score']}",
            flush=True,
        )

    total_elapsed = elapsed_before_restart + time.monotonic() - queue_start
    completion = {
        "state": "completed",
        "completed_at": _utc_now(),
        "elapsed_seconds": total_elapsed,
        **{key: value for key, value in manifest.items() if key != "runs"},
        "runs": run_states,
    }
    _write_json(output_root / "queue_status.json", completion)
    _write_results_csv(output_root / "results.csv", run_states)
    print(f"[queue] COMPLETE elapsed={total_elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
