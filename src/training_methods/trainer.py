import os
# Hack to fix multi-GPU training on server (NCCL P2P hang)
os.environ["NCCL_P2P_DISABLE"] = "0"

import math
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from datetime import datetime
from omegaconf import DictConfig, ListConfig, open_dict
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import wandb


sys.path.append(os.getcwd())
from src.utils.logging_config import setup_logging
from src.data_utils.data_module import (
    RealPointCloudDataModule,
    SyntheticPointCloudDataModule,
    TemporalLAMMPSDataModule,
)
torch.set_float32_matmul_precision('high')

logger = setup_logging()


def _seed_training_run(cfg: DictConfig) -> None:
    """Optionally seed Lightning/PyTorch for reproducible repeated experiments."""
    seed_value = getattr(cfg, "seed_everything", None)
    if seed_value is None:
        return
    seed_int = int(seed_value)
    pl.seed_everything(seed_int, workers=True)
    logger.print(f"Global random seed set to {seed_int}.")


def _resolve_checkpoint_path(cfg: DictConfig, field_names: tuple[str, ...]) -> str | None:
    """Resolve an optional checkpoint path from one of *field_names*."""
    checkpoint_path = None
    for field_name in field_names:
        value = getattr(cfg, field_name, None)
        if value is None:
            continue
        value_str = str(value).strip()
        if not value_str:
            continue
        checkpoint_path = value_str
        break

    if checkpoint_path is None:
        return None

    checkpoint_path = os.path.expanduser(checkpoint_path)
    if not os.path.isabs(checkpoint_path):
        base_dir = os.getcwd()
        try:
            base_dir = HydraConfig.get().runtime.cwd
        except Exception as exc:
            logger.warning(
                "Hydra runtime cwd is unavailable; resolving checkpoint "
                "relative to current directory '%s'. Error: %s",
                base_dir,
                exc,
            )
        checkpoint_path = os.path.join(base_dir, checkpoint_path)
    checkpoint_path = os.path.abspath(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path


def _resolve_resume_checkpoint(cfg: DictConfig) -> str | None:
    """Resolve optional training resume checkpoint from config."""
    return _resolve_checkpoint_path(
        cfg,
        ("resume_from_checkpoint", "resume_checkpoint_path"),
    )


def _resolve_init_checkpoint(cfg: DictConfig) -> str | None:
    """Resolve optional model-initialization checkpoint from config."""
    return _resolve_checkpoint_path(
        cfg,
        (
            "init_from_checkpoint",
            "init_weights_from_checkpoint",
            "weights_checkpoint_path",
        ),
    )


def _extract_state_dict(checkpoint) -> dict:
    """Extract model weights dictionary from a Lightning/PyTorch checkpoint payload."""
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict")
        if isinstance(state_dict, dict):
            return state_dict
        model_state_dict = checkpoint.get("model_state_dict")
        if isinstance(model_state_dict, dict):
            return model_state_dict
        if all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "Checkpoint does not contain a recognized state dictionary "
        "(`state_dict` or `model_state_dict`)."
    )


def _strip_state_dict_prefixes(state_dict: dict) -> dict:
    """Strip common wrapper prefixes (e.g. model./module.) from state dict keys."""
    prefixes = ("model.", "module.")
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        normalized[new_key] = value
    return normalized


def _compatible_state_dict_for_model(state_dict: dict, model_state: dict) -> tuple[dict, list[str]]:
    """Keep only tensors whose names and shapes match the current model."""
    compatible = {}
    shape_mismatch = []
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            continue
        if tuple(target.shape) != tuple(value.shape):
            shape_mismatch.append(key)
            continue
        compatible[key] = value
    return compatible, shape_mismatch


def _load_model_weights_from_checkpoint(model: pl.LightningModule, checkpoint_path: str, *, strict: bool = False) -> None:
    """Load model weights from *checkpoint_path* without restoring trainer state."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    source_state_dict = _extract_state_dict(checkpoint)
    model_state = model.state_dict()
    ckpt_hp = checkpoint.get("hyper_parameters") if isinstance(checkpoint, dict) else None
    ckpt_enc = getattr(getattr(ckpt_hp, "encoder", None), "name", None)
    model_enc = getattr(getattr(getattr(model, "hparams", None), "encoder", None), "name", None)
    if ckpt_enc and model_enc and str(ckpt_enc) != str(model_enc):
        logger.print(
            f"Warning: init checkpoint encoder differs from current config: "
            f"checkpoint={ckpt_enc}, current={model_enc}. "
            "Only tensors with matching names and shapes will be loaded."
        )

    raw_compatible, raw_shape_mismatch = _compatible_state_dict_for_model(source_state_dict, model_state)
    stripped_state_dict = _strip_state_dict_prefixes(source_state_dict)
    stripped_compatible, stripped_shape_mismatch = _compatible_state_dict_for_model(stripped_state_dict, model_state)

    use_stripped = len(stripped_compatible) > len(raw_compatible)
    selected_state_dict = stripped_compatible if use_stripped else raw_compatible
    selected_shape_mismatch = stripped_shape_mismatch if use_stripped else raw_shape_mismatch

    if not selected_state_dict:
        raise RuntimeError(
            f"No compatible tensors found when loading checkpoint '{checkpoint_path}'. "
            "Check that the checkpoint matches the current architecture."
        )

    missing_keys, unexpected_keys = model.load_state_dict(selected_state_dict, strict=strict)

    source_encoder_keys = sum(1 for k in source_state_dict if k.startswith("encoder."))
    model_encoder_keys = sum(1 for k in model_state if k.startswith("encoder."))
    loaded_encoder_keys = sum(1 for k in selected_state_dict if k.startswith("encoder."))

    logger.print(f"Initialized model weights from checkpoint: {checkpoint_path}")
    logger.print(
        "Checkpoint load summary: "
        f"loaded={len(selected_state_dict)} "
        f"/ model_tensors={len(model_state)}, "
        f"shape_mismatch_skipped={len(selected_shape_mismatch)}, "
        f"missing_after_load={len(missing_keys)}, "
        f"unexpected_after_load={len(unexpected_keys)}, "
        f"strict={strict}"
    )

    if source_encoder_keys > 0 and model_encoder_keys > 0 and loaded_encoder_keys == 0:
        logger.print(
            "Warning: No encoder weights were loaded from init checkpoint. "
            "Current encoder will remain randomly initialized."
        )


def _resolve_accumulate_grad_batches(cfg: DictConfig) -> int:
    """Resolve gradient accumulation from config with backward-compatible aliases."""
    value = getattr(cfg, "accumulate_grad_batches", None)
    if value is None:
        value = getattr(cfg, "gradient_accumulation_steps", None)
    if value is None:
        return 1

    accum = int(value)
    if accum < 1:
        raise ValueError(
            "Gradient accumulation must be >= 1. "
            f"Got accumulate_grad_batches={value!r}."
        )
    return accum


def _set_cfg_value(cfg: DictConfig, key: str, value) -> None:
    """Update a Hydra config value safely even when struct mode is enabled."""
    with open_dict(cfg):
        setattr(cfg, key, value)


def _set_model_hparam(model: pl.LightningModule, key: str, value) -> None:
    """Best-effort update of model hyperparameters after runtime auto-tuning."""
    hparams = getattr(model, "hparams", None)
    if hparams is None:
        return
    try:
        hparams[key] = value
        return
    except Exception as exc:
        logger.debug("hparams[%r] = %r failed via __setitem__: %s", key, value, exc)
    try:
        setattr(hparams, key, value)
    except Exception as exc:
        logger.debug("setattr(hparams, %r, %r) also failed: %s", key, value, exc)
        return


def _nearest_accumulate_grad_batches(target_effective_batch: int, batch_size: int) -> int:
    """Choose accumulation steps that keep batch_size*accumulation close to target."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    target = max(1, int(target_effective_batch))
    ratio = float(target) / float(batch_size)
    lower = max(1, int(math.floor(ratio)))
    upper = max(1, int(math.ceil(ratio)))
    if abs(lower * batch_size - target) <= abs(upper * batch_size - target):
        return lower
    return upper


def _run_auto_batch_size_search(
    *,
    cfg: DictConfig,
    model: pl.LightningModule,
    dm,
    run_dir: str,
    precision,
    devices: list[int],
    ddp_strategy,
) -> int | None:
    """Run optional Lightning batch-size finder and return the selected batch size."""
    if not bool(getattr(cfg, "auto_batch_size_search", False)):
        return None

    strict = bool(getattr(cfg, "auto_batch_size_strict", False))
    mode = str(getattr(cfg, "auto_batch_size_mode", "power")).strip().lower()
    if mode not in {"power", "binsearch"}:
        raise ValueError(f"auto_batch_size_mode must be 'power' or 'binsearch', got {mode!r}")

    init_val = getattr(cfg, "auto_batch_size_init_val", None)
    if init_val is None:
        init_val = getattr(cfg, "batch_size", 1)
    init_val = max(1, int(init_val))
    max_trials = max(1, int(getattr(cfg, "auto_batch_size_max_trials", 12)))
    steps_per_trial = max(1, int(getattr(cfg, "auto_batch_size_steps_per_trial", 3)))
    use_single_device = bool(getattr(cfg, "auto_batch_size_use_single_device", True))

    search_devices: int | list[int]
    search_strategy = ddp_strategy
    if bool(getattr(cfg, "gpu", False)):
        if use_single_device and len(devices) > 1:
            search_devices = [int(devices[0])]
            search_strategy = None
            logger.print(
                "Auto batch-size search will run on a single GPU before multi-GPU training "
                f"(selected device index: {search_devices[0]})."
            )
        else:
            search_devices = devices
            if len(devices) <= 1:
                search_strategy = None
    else:
        search_devices = 1
        search_strategy = None

    trainer_kwargs = dict(
        default_root_dir=run_dir,
        max_epochs=1,
        accelerator='gpu' if cfg.gpu else 'cpu',
        devices=search_devices,
        logger=False,
        callbacks=[],
        enable_checkpointing=False,
        log_every_n_steps=max(1, int(getattr(cfg, "log_every_n_steps", 1))),
        precision=precision,
        benchmark=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    if search_strategy is not None:
        trainer_kwargs["strategy"] = search_strategy

    try:
        from pytorch_lightning.tuner import Tuner
    except Exception as exc:
        message = (
            "Auto batch-size search is enabled, but Lightning Tuner is unavailable. "
            f"Error: {exc}"
        )
        if strict:
            raise RuntimeError(message) from exc
        logger.print(f"Warning: {message}. Continuing with configured batch_size={cfg.batch_size}.")
        return None

    logger.print(
        "Running auto batch-size search with "
        f"mode={mode}, init_val={init_val}, max_trials={max_trials}, steps_per_trial={steps_per_trial}."
    )
    try:
        search_trainer = pl.Trainer(**trainer_kwargs)
        suggested = Tuner(search_trainer).scale_batch_size(
            model,
            datamodule=dm,
            mode=mode,
            init_val=init_val,
            max_trials=max_trials,
            steps_per_trial=steps_per_trial,
            batch_arg_name="batch_size",
        )
    except Exception as exc:
        message = f"Auto batch-size search failed with error: {exc}"
        if strict:
            raise RuntimeError(message) from exc
        logger.print(f"Warning: {message}. Continuing with configured batch_size={cfg.batch_size}.")
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if suggested is None:
        suggested = getattr(dm, "batch_size", getattr(cfg, "batch_size", None))
    if suggested is None:
        message = "Auto batch-size search returned no batch size."
        if strict:
            raise RuntimeError(message)
        logger.print(f"Warning: {message} Continuing with configured batch_size={cfg.batch_size}.")
        return None

    suggested = int(suggested)
    if suggested < 1:
        message = f"Auto batch-size search returned invalid batch size {suggested}."
        if strict:
            raise RuntimeError(message)
        logger.print(f"Warning: {message} Continuing with configured batch_size={cfg.batch_size}.")
        return None
    return suggested

@rank_zero_only
def get_rundir_name() -> str:
    now = datetime.now()
    return str(f"output/{now:%Y-%m-%d}/{now:%H-%M-%S}")

@rank_zero_only
def init_wandb(cfg: DictConfig, run_dir):
    os.environ['WANDB_MODE'] = cfg.wandb_mode
    os.environ['WANDB_DIR'] = 'output/wandb'
    os.environ['WANDB_CONFIG_DIR'] = 'output/wandb'
    os.environ['WANDB_CACHE_DIR'] = 'output/wandb'
    wandb.init(project='PointCloudMaterials', name=cfg.experiment_name)
    return WandbLogger(save_dir=os.path.join(os.getcwd(), run_dir),
                       project=cfg.project_name,
                       name=cfg.experiment_name,
                       log_model=False)


def _validate_train_batches_available(
    *,
    dm,
    batch_size: int,
    devices,
    cfg: DictConfig,
    capacity: tuple[int, int, int] | None = None,
) -> None:
    """Fail loudly when DDP + drop_last would produce zero train batches."""
    if capacity is None:
        capacity = _get_train_batch_capacity(dm=dm, devices=devices, cfg=cfg)
    total_train_samples, world_size, per_rank_samples = capacity

    if per_rank_samples < int(batch_size):
        raise RuntimeError(
            "Configured training batch size yields zero train batches with drop_last=True. "
            f"experiment={cfg.experiment_name!r}, train_samples_total={total_train_samples}, "
            f"devices={list(devices)}, world_size={world_size}, "
            f"estimated_samples_per_rank={per_rank_samples}, batch_size={int(batch_size)}. "
            "Lower batch_size, reduce the number of devices, or change the dataloader drop_last behavior."
        )


def _get_train_batch_capacity(
    *,
    dm,
    devices,
    cfg: DictConfig,
) -> tuple[int, int, int]:
    """Return (train_samples_total, world_size, estimated_samples_per_rank)."""
    dm.setup("fit")

    train_dataset = getattr(dm, "train_dataset", None)
    if train_dataset is None and hasattr(dm, "impl"):
        train_dataset = getattr(dm.impl, "train_dataset", None)
    if train_dataset is None:
        raise RuntimeError(
            "Training datamodule did not expose train_dataset after setup('fit'). "
            "Cannot validate train batch availability."
        )

    total_train_samples = int(len(train_dataset))
    if total_train_samples <= 0:
        raise RuntimeError(
            f"Training split is empty for experiment {cfg.experiment_name!r}."
        )

    world_size = len(devices) if bool(getattr(cfg, "gpu", False)) else 1
    world_size = max(1, int(world_size))
    per_rank_samples = int(math.ceil(total_train_samples / world_size))
    return total_train_samples, world_size, per_rank_samples


def _cap_batch_size_to_train_split(
    *,
    cfg: DictConfig,
    model: pl.LightningModule,
    dm,
    requested_batch_size: int,
    total_train_samples: int,
    world_size: int,
    per_rank_samples: int,
) -> int:
    """Cap batch size so each rank keeps at least one batch when drop_last=True."""
    batch_size = int(requested_batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if per_rank_samples < 1:
        raise RuntimeError(
            "Estimated per-rank train samples must be >= 1, "
            f"got {per_rank_samples} for experiment {cfg.experiment_name!r}."
        )
    if batch_size <= per_rank_samples:
        return batch_size

    capped_batch_size = int(per_rank_samples)
    _set_cfg_value(cfg, "batch_size", capped_batch_size)
    dm.batch_size = capped_batch_size
    _set_model_hparam(model, "batch_size", capped_batch_size)

    auto_batch_size_init_val = getattr(cfg, "auto_batch_size_init_val", None)
    if auto_batch_size_init_val is not None and int(auto_batch_size_init_val) > capped_batch_size:
        _set_cfg_value(cfg, "auto_batch_size_init_val", capped_batch_size)

    logger.print(
        "Capping batch_size to fit the available training split: "
        f"requested={batch_size}, using={capped_batch_size}, "
        f"train_samples_total={total_train_samples}, world_size={world_size}, "
        f"estimated_samples_per_rank={per_rank_samples}."
    )
    return capped_batch_size


def train_model(cfg: DictConfig, model_class, run_dir=None, checkpoint_callbacks=None,
                devices=None, run_test=True):
    """
    Generic training function that can be used with any PyTorch Lightning model.

    Args:
        cfg: Hydra configuration
        model_class: The LightningModule class to instantiate (e.g., BarlowTwinsModule, ShapePoseDisentanglement)
        run_dir: Optional custom output directory (default: auto-generated from timestamp)
        checkpoint_callbacks: Optional list of checkpoint callbacks (default: single checkpoint monitoring val/loss)
        devices: Optional device list (default: from cfg.devices or [0])
        run_test: Whether to run test phase after training (default: True)

    Returns:
        tuple: (trainer, model, datamodule, checkpoint_callbacks) for post-training processing
    """
    logger.print(f"Starting in {os.getcwd()}")

    if run_dir is None:
        try:
            run_dir = HydraConfig.get().runtime.output_dir
        except Exception as exc:
            run_dir = get_rundir_name()
            logger.warning(
                "Hydra runtime output_dir is unavailable; falling back to "
                "timestamped run directory '%s'. Error: %s",
                run_dir,
                exc,
            )

    _seed_training_run(cfg)
    wandb_logger = init_wandb(cfg, run_dir)

    if cfg.data.kind == "synthetic":
        dm = SyntheticPointCloudDataModule(cfg)
    elif cfg.data.kind == "temporal_lammps":
        dm = TemporalLAMMPSDataModule(cfg)
    else:
        dm = RealPointCloudDataModule(cfg)
    model = model_class(cfg)

    checkpoint_monitor = str(getattr(cfg, "checkpoint_monitor", "val/loss"))
    checkpoint_mode = str(getattr(cfg, "checkpoint_mode", "min")).strip().lower()
    if checkpoint_mode not in {"min", "max"}:
        raise ValueError(
            f"checkpoint_mode must be 'min' or 'max', got {checkpoint_mode!r}"
        )
    checkpoint_save_top_k = int(getattr(cfg, "checkpoint_save_top_k", 3))
    checkpoint_save_top_k = max(1, checkpoint_save_top_k)

    # Set up checkpoint callbacks
    if checkpoint_callbacks is None:
        checkpoint_callbacks = [ModelCheckpoint(
            dirpath=run_dir,
            monitor=checkpoint_monitor,
            filename=f'{cfg.experiment_name}-{{epoch:02d}}',
            save_top_k=checkpoint_save_top_k,
            mode=checkpoint_mode,
        )]

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = list(checkpoint_callbacks) + [lr_monitor]

    # Set up devices
    if devices is None:
        if isinstance(cfg.devices, ListConfig):
            devices = list(cfg.devices)
        elif isinstance(cfg.devices, (list, tuple)):
            devices = list(cfg.devices) if len(cfg.devices) > 0 else [0]
        else:
            devices = [0]

    ddp_strategy = None
    if cfg.gpu and len(devices) > 1 and hasattr(cfg, 'ddp_find_unused_parameters') and cfg.ddp_find_unused_parameters:
        ddp_strategy = DDPStrategy(find_unused_parameters=True)

    precision = getattr(cfg, "precision", "bf16-mixed")
    check_val_every_n_epoch = int(getattr(cfg, "check_val_every_n_epoch", 10))
    if check_val_every_n_epoch < 1:
        raise ValueError(
            f"check_val_every_n_epoch must be >= 1, got {check_val_every_n_epoch}"
        )
    num_sanity_val_steps = int(getattr(cfg, "num_sanity_val_steps", 2))
    if num_sanity_val_steps < 0:
        raise ValueError(
            f"num_sanity_val_steps must be >= 0, got {num_sanity_val_steps}"
        )
    configured_batch_size = int(getattr(cfg, "batch_size", 1))
    if configured_batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {configured_batch_size}")
    accumulate_grad_batches = _resolve_accumulate_grad_batches(cfg)
    target_effective_batch = getattr(cfg, "auto_batch_size_target_effective_batch", None)
    if target_effective_batch is None:
        target_effective_batch = configured_batch_size * accumulate_grad_batches
    target_effective_batch = int(target_effective_batch)
    if target_effective_batch < 1:
        raise ValueError(
            "auto_batch_size_target_effective_batch must be >= 1 when set. "
            f"Got {target_effective_batch!r}."
        )

    train_batch_capacity = _get_train_batch_capacity(
        dm=dm,
        devices=devices,
        cfg=cfg,
    )
    total_train_samples, world_size, per_rank_samples = train_batch_capacity
    configured_batch_size = _cap_batch_size_to_train_split(
        cfg=cfg,
        model=model,
        dm=dm,
        requested_batch_size=configured_batch_size,
        total_train_samples=total_train_samples,
        world_size=world_size,
        per_rank_samples=per_rank_samples,
    )

    tuned_batch_size = _run_auto_batch_size_search(
        cfg=cfg,
        model=model,
        dm=dm,
        run_dir=run_dir,
        precision=precision,
        devices=devices,
        ddp_strategy=ddp_strategy,
    )
    if tuned_batch_size is not None:
        _set_cfg_value(cfg, "batch_size", tuned_batch_size)
        dm.batch_size = tuned_batch_size
        _set_model_hparam(model, "batch_size", tuned_batch_size)
        logger.print(
            "Auto batch-size search selected "
            f"batch_size={tuned_batch_size} (previous={configured_batch_size})."
        )

        if bool(getattr(cfg, "auto_batch_size_adjust_accumulate", True)):
            previous_accum = accumulate_grad_batches
            accumulate_grad_batches = _nearest_accumulate_grad_batches(
                target_effective_batch=target_effective_batch,
                batch_size=tuned_batch_size,
            )
            _set_cfg_value(cfg, "accumulate_grad_batches", accumulate_grad_batches)
            if hasattr(cfg, "gradient_accumulation_steps"):
                _set_cfg_value(cfg, "gradient_accumulation_steps", accumulate_grad_batches)
            _set_model_hparam(model, "accumulate_grad_batches", accumulate_grad_batches)
            logger.print(
                "Adjusted gradient accumulation after auto batch-size search: "
                f"{previous_accum} -> {accumulate_grad_batches} "
                f"(target_effective_batch_per_device={target_effective_batch}, "
                f"actual={tuned_batch_size * accumulate_grad_batches})."
            )

    effective_batch_size = int(getattr(dm, "batch_size", getattr(cfg, "batch_size", configured_batch_size)))
    effective_batch_size = _cap_batch_size_to_train_split(
        cfg=cfg,
        model=model,
        dm=dm,
        requested_batch_size=effective_batch_size,
        total_train_samples=total_train_samples,
        world_size=world_size,
        per_rank_samples=per_rank_samples,
    )
    _validate_train_batches_available(
        dm=dm,
        batch_size=effective_batch_size,
        devices=devices,
        cfg=cfg,
        capacity=train_batch_capacity,
    )

    trainer_kwargs = dict(
        default_root_dir=run_dir,
        max_epochs=cfg.epochs,
        accelerator='gpu' if cfg.gpu else 'cpu',
        devices=devices,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.log_every_n_steps,
        precision=precision,
        benchmark=True,
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=num_sanity_val_steps,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    if accumulate_grad_batches > 1:
        logger.print(
            "Gradient accumulation enabled: "
            f"accumulate_grad_batches={accumulate_grad_batches}"
        )

    if hasattr(cfg, 'gradient_clip_val'):
        trainer_kwargs['gradient_clip_val'] = cfg.gradient_clip_val
        trainer_kwargs['gradient_clip_algorithm'] = 'norm'

    if ddp_strategy is not None:
        trainer_kwargs["strategy"] = ddp_strategy

    trainer = pl.Trainer(**trainer_kwargs)

    resume_ckpt_path = _resolve_resume_checkpoint(cfg)
    init_ckpt_path = _resolve_init_checkpoint(cfg)

    if resume_ckpt_path is not None and init_ckpt_path is not None:
        raise ValueError(
            "Both resume and init checkpoints are set. Use only one mode:\n"
            "- resume_from_checkpoint (full-state resume)\n"
            "- init_from_checkpoint (weights-only initialization)"
        )

    if init_ckpt_path is not None:
        init_strict = bool(getattr(cfg, "init_from_checkpoint_strict", False))
        _load_model_weights_from_checkpoint(model, init_ckpt_path, strict=init_strict)
        logger.print("Starting fresh training from loaded model weights (epoch/optimizer reset).")
        trainer.fit(model, dm)
    else:
        if resume_ckpt_path is not None:
            logger.print(f"Resuming training from checkpoint: {resume_ckpt_path}")
        trainer.fit(model, dm, ckpt_path=resume_ckpt_path)

    # Resolve the checkpoint used for final test.
    test_ckpt_path = None
    for callback in checkpoint_callbacks:
        path = getattr(callback, "best_model_path", "")
        if isinstance(path, str) and path and os.path.exists(path):
            test_ckpt_path = path
            break
    if test_ckpt_path is None:
        for callback in checkpoint_callbacks:
            path = getattr(callback, "last_model_path", "")
            if isinstance(path, str) and path and os.path.exists(path):
                test_ckpt_path = path
                break

    # Run test after training completes
    if run_test:
        logger.print("Starting test phase...")
        run_test_single_device = bool(getattr(cfg, "test_single_device", True))
        if cfg.gpu and len(devices) > 1 and run_test_single_device:
            logger.print("Running test on a single device to avoid duplicated DDP samples.")

            # Tear down the DDP process group before creating a single-device
            # Trainer, otherwise rank-1 blocks on NCCL collectives and the
            # whole run hangs.
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()

            test_trainer_kwargs = dict(
                default_root_dir=run_dir,
                accelerator='gpu',
                devices=1,
                logger=wandb_logger,
                precision=precision,
                benchmark=True,
                log_every_n_steps=cfg.log_every_n_steps,
            )
            if hasattr(cfg, 'gradient_clip_val'):
                test_trainer_kwargs['gradient_clip_val'] = cfg.gradient_clip_val
                test_trainer_kwargs['gradient_clip_algorithm'] = 'norm'
            test_trainer = pl.Trainer(**test_trainer_kwargs)
            test_trainer.test(model, dm, ckpt_path=test_ckpt_path)
        else:
            trainer.test(model, dm, ckpt_path=test_ckpt_path)

    return trainer, model, dm, checkpoint_callbacks
