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

    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=hparams.decay_rate,
    )
    epochs_before_swa = int(
        hparams.swa_epoch_start + 1 if hparams.enable_swa else hparams.epochs
    )

    warmup_epochs = int(hparams.warmup_epochs) if hparams.warmup_enabled else 0

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
