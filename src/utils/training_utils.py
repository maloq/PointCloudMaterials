"""Shared optimizer, scheduler, and metric-cache utilities."""

import torch


def get_optimizers_and_scheduler(hparams, parameters):
    """Build the cosine schedule used by every current training config."""
    if hparams.scheduler_name != "Cosine":
        raise ValueError(
            "Current repository training configs use scheduler_name='Cosine'; "
            f"got {hparams.scheduler_name!r}."
        )

    learning_rate = float(hparams.learning_rate)
    minimum_lr = float(hparams.scheduler_min_lr)
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")
    if not 0.0 <= minimum_lr <= learning_rate:
        raise ValueError(
            "scheduler_min_lr must be between zero and learning_rate; "
            f"got scheduler_min_lr={minimum_lr}, learning_rate={learning_rate}."
        )

    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=hparams.decay_rate,
    )
    epochs_before_swa = int(
        hparams.swa_epoch_start + 1 if hparams.enable_swa else hparams.epochs
    )
    if epochs_before_swa < 1:
        raise ValueError(f"Cosine scheduler requires at least one epoch, got {epochs_before_swa}.")

    warmup_epochs = int(hparams.warmup_epochs) if hparams.warmup_enabled else 0
    if hparams.warmup_enabled:
        if not 0 < warmup_epochs < epochs_before_swa:
            raise ValueError(
                "warmup_epochs must leave at least one epoch for cosine decay; "
                f"got warmup_epochs={warmup_epochs}, scheduled_epochs={epochs_before_swa}."
            )
        if not 0.0 < hparams.warmup_start_factor <= 1.0:
            raise ValueError(
                "warmup_start_factor must be in (0, 1], "
                f"got {hparams.warmup_start_factor}."
            )

    cosine_epochs = epochs_before_swa - warmup_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=minimum_lr,
    )
    if hparams.warmup_enabled:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=hparams.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, scheduler],
            milestones=[warmup_epochs],
        )

    return [optimizer], [
        {
            "scheduler": scheduler,
            "name": "trainer/lr-AdamW",
            "interval": "epoch",
            "frequency": 1,
        }
    ]


def cached_sample_count(cache: dict[str, list[torch.Tensor]]) -> int:
    """Count samples in a repository supervised-metric cache."""
    return sum(latents.shape[0] for latents in cache["latents"])
