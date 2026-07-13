"""Shared optimizer, scheduler, and metric-cache utilities."""

from bisect import bisect_right

import torch


class _SequentialLRNoEpoch(torch.optim.lr_scheduler.SequentialLR):
    """SequentialLR variant that avoids deprecated scheduler.step(epoch=...)."""

    def step(self):  # type: ignore[override]
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        scheduler.step()
        self._last_lr = scheduler.get_last_lr()


def get_optimizers_and_scheduler(hparams, parameters):
    """Build the cosine schedule used by every current training config."""
    if hparams.scheduler_name != "Cosine":
        raise ValueError(
            "Current repository training configs use scheduler_name='Cosine'; "
            f"got {hparams.scheduler_name!r}."
        )

    optimizer = torch.optim.AdamW(
        parameters,
        lr=hparams.learning_rate,
        weight_decay=hparams.decay_rate,
    )
    epochs_before_swa = hparams.swa_epoch_start + 1 if hparams.enable_swa else hparams.epochs
    if epochs_before_swa < 1:
        raise ValueError(f"Cosine scheduler requires at least one epoch, got {epochs_before_swa}.")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs_before_swa,
        eta_min=hparams.scheduler_min_lr,
    )
    if hparams.warmup_enabled:
        if not 0 < hparams.warmup_epochs <= epochs_before_swa:
            raise ValueError(
                "warmup_epochs must be between 1 and the scheduled training length; "
                f"got warmup_epochs={hparams.warmup_epochs}, epochs={epochs_before_swa}."
            )
        if not 0.0 < hparams.warmup_start_factor <= 1.0:
            raise ValueError(
                "warmup_start_factor must be in (0, 1], "
                f"got {hparams.warmup_start_factor}."
            )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=hparams.warmup_start_factor,
            end_factor=1.0,
            total_iters=hparams.warmup_epochs,
        )
        scheduler = _SequentialLRNoEpoch(
            optimizer,
            schedulers=[warmup, scheduler],
            milestones=[hparams.warmup_epochs],
        )

    return [optimizer], [{"scheduler": scheduler, "name": "trainer/lr-AdamW"}]


def cached_sample_count(cache: dict[str, list[torch.Tensor]]) -> int:
    """Count samples in a repository supervised-metric cache."""
    return sum(latents.shape[0] for latents in cache["latents"])
