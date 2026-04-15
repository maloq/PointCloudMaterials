import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

sys.path.append(os.getcwd())

from src.data_utils.data_module import RealPointCloudDataModule, SyntheticPointCloudDataModule
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
from src.utils.model_utils import load_model_from_checkpoint, resolve_config_path


torch.set_float32_matmul_precision("high")


def _resolve_checkpoint_path(path: str) -> str:
    checkpoint_path = os.path.expanduser(path)
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _resolve_output_dir(path: str | None, *, checkpoint_path: str) -> Path:
    if path is None:
        return Path(checkpoint_path).resolve().parent / "eval_results"
    output_dir = os.path.expanduser(path)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    return Path(output_dir).resolve()


def load_vicreg_model(
    checkpoint_path: str,
    cuda_device: int = 0,
    cfg: DictConfig | None = None,
) -> tuple[VICRegModule, DictConfig, str]:
    """Restore the contrastive module together with its Hydra config and device."""
    if cuda_device < 0:
        raise ValueError(f"cuda_device must be >= 0, got {cuda_device}")

    if cfg is None:
        config_dir, config_name = resolve_config_path(checkpoint_path)
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parents[2]
        absolute_config_dir = project_root / config_dir
        relative_config_dir = os.path.relpath(absolute_config_dir, current_dir)
        with initialize(version_base=None, config_path=relative_config_dir):
            cfg = compose(config_name=config_name)

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if cuda_device >= device_count:
            raise ValueError(
                f"Requested cuda_device={cuda_device}, but only {device_count} CUDA device(s) are available."
            )
        device = f"cuda:{cuda_device}"
    else:
        device = "cpu"
        print("[eval] CUDA is not available; running evaluation on CPU.")

    model: VICRegModule = load_model_from_checkpoint(
        checkpoint_path,
        cfg,
        device=device,
        module=VICRegModule,
    )
    model.to(device).eval()
    return model, cfg, device


def build_datamodule(
    cfg: DictConfig,
    *,
    data_files_override: list[str] | None = None,
    num_workers_override: int | None = None,
    batch_size_override: int | None = None,
):
    """Instantiate and setup the matching datamodule."""
    if getattr(cfg, "data", None) is None:
        raise ValueError("Config missing required 'data' section.")
    if data_files_override:
        if getattr(cfg.data, "kind", None) != "real":
            raise ValueError(
                "data_files_override can only be used for real datasets (cfg.data.kind == 'real')."
            )
        cfg.data.data_files = [str(v) for v in data_files_override]

    if num_workers_override is not None:
        num_workers = int(num_workers_override)
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}.")
        cfg.num_workers = num_workers

    if batch_size_override is not None:
        batch_size = int(batch_size_override)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        cfg.batch_size = batch_size

    if getattr(cfg.data, "kind", None) == "synthetic":
        dm = SyntheticPointCloudDataModule(cfg)
    else:
        dm = RealPointCloudDataModule(cfg)
    dm.setup(stage="test")
    return dm


def _resolve_eval_precision(cfg: DictConfig, *, device: str):
    configured_precision = getattr(cfg, "precision", "32-true")
    if device.startswith("cuda"):
        return configured_precision
    if str(configured_precision).lower() != "32-true":
        print(
            f"[eval] precision={configured_precision!r} is not supported for CPU eval; "
            "forcing precision='32-true'."
        )
    return "32-true"


def _build_eval_trainer(
    cfg: DictConfig,
    *,
    device: str,
    cuda_device: int,
    output_dir: Path,
    disable_progress_bar: bool,
) -> pl.Trainer:
    precision = _resolve_eval_precision(cfg, device=device)
    if device.startswith("cuda"):
        accelerator = "gpu"
        devices: int | list[int] = [int(cuda_device)]
    else:
        accelerator = "cpu"
        devices = 1

    return pl.Trainer(
        default_root_dir=str(output_dir),
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        benchmark=device.startswith("cuda"),
        precision=precision,
        log_every_n_steps=int(getattr(cfg, "log_every_n_steps", 50)),
        enable_progress_bar=not disable_progress_bar,
    )


def _to_jsonable_metric_value(value: Any, *, metric_name: str):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                f"Metric '{metric_name}' must be scalar, got tensor with shape {tuple(value.shape)}."
            )
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    raise TypeError(
        f"Metric '{metric_name}' has unsupported type {type(value)!r}; expected scalar tensor or primitive."
    )


def _normalize_metrics(raw_metrics: list[dict[str, Any]], *, stage: str) -> list[dict[str, Any]]:
    if not isinstance(raw_metrics, list):
        raise TypeError(f"{stage} metrics must be a list, got {type(raw_metrics)!r}.")
    normalized = []
    for idx, metrics in enumerate(raw_metrics):
        if not isinstance(metrics, dict):
            raise TypeError(f"{stage} metrics entry #{idx} must be a dict, got {type(metrics)!r}.")
        normalized_entry = {}
        for key, value in metrics.items():
            metric_name = f"{stage}[{idx}].{key}"
            normalized_entry[str(key)] = _to_jsonable_metric_value(value, metric_name=metric_name)
        normalized.append(normalized_entry)
    return normalized


def _print_metrics(stage: str, metrics: list[dict[str, Any]]) -> None:
    if not metrics:
        print(f"[eval] {stage}: no metrics returned.")
        return
    for idx, entry in enumerate(metrics):
        print(f"[eval] {stage} dataloader #{idx}")
        for key in sorted(entry.keys()):
            print(f"  {key}: {entry[key]}")


def run_checkpoint_evaluation(
    checkpoint_path: str,
    *,
    output_dir: Path,
    cuda_device: int = 0,
    cfg: DictConfig | None = None,
    run_validation: bool = True,
    run_test: bool = True,
    data_files_override: list[str] | None = None,
    num_workers_override: int | None = None,
    batch_size_override: int | None = None,
    disable_progress_bar: bool = False,
) -> dict[str, Any]:
    if not run_validation and not run_test:
        raise ValueError("Both run_validation and run_test are disabled. Enable at least one stage.")

    model, cfg, device = load_vicreg_model(
        checkpoint_path=checkpoint_path,
        cuda_device=cuda_device,
        cfg=cfg,
    )
    dm = build_datamodule(
        cfg,
        data_files_override=data_files_override,
        num_workers_override=num_workers_override,
        batch_size_override=batch_size_override,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = _build_eval_trainer(
        cfg,
        device=device,
        cuda_device=cuda_device,
        output_dir=output_dir,
        disable_progress_bar=disable_progress_bar,
    )

    report: dict[str, Any] = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "device": device,
        "batch_size": int(cfg.batch_size),
        "num_workers": int(cfg.num_workers),
        "stages": {},
    }

    if run_validation:
        raw_val = trainer.validate(model=model, datamodule=dm, verbose=not disable_progress_bar)
        val_metrics = _normalize_metrics(raw_val, stage="val")
        report["stages"]["val"] = val_metrics
        _print_metrics("val", val_metrics)

    if run_test:
        raw_test = trainer.test(model=model, datamodule=dm, verbose=not disable_progress_bar)
        test_metrics = _normalize_metrics(raw_test, stage="test")
        report["stages"]["test"] = test_metrics
        _print_metrics("test", test_metrics)

    report_path = output_dir / "metrics.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[eval] Saved metrics to {report_path}")

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run validation/test for a trained contrastive checkpoint.",
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to a trained checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for JSON metrics (default: <ckpt_dir>/eval_results).",
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="CUDA device index when running on GPU (default: 0).",
    )
    parser.add_argument(
        "--skip_val",
        action="store_true",
        help="Skip validation stage.",
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Skip test stage.",
    )
    parser.add_argument(
        "--data_file",
        action="append",
        default=None,
        help="Override cfg.data.data_files for real datasets. Repeat for multiple files.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Optional override for cfg.num_workers.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional override for cfg.batch_size.",
    )
    parser.add_argument(
        "--disable_progress_bar",
        action="store_true",
        help="Disable Lightning progress bar and metric tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)
    output_dir = _resolve_output_dir(args.output_dir, checkpoint_path=checkpoint_path)

    run_checkpoint_evaluation(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        cuda_device=int(args.cuda_device),
        cfg=None,
        run_validation=not bool(args.skip_val),
        run_test=not bool(args.skip_test),
        data_files_override=args.data_file,
        num_workers_override=args.num_workers,
        batch_size_override=args.batch_size,
        disable_progress_bar=bool(args.disable_progress_bar),
    )


if __name__ == "__main__":
    main()
