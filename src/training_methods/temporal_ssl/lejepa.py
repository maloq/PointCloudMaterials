import torch
import torch.distributed as dist
import torch.nn as nn


class LeJEPALoss(nn.Module):
    """Lean temporal LeJEPA loss: predict the last-frame embedding from prior frames."""

    def __init__(
        self,
        *,
        enabled: bool,
        weight: float,
        lambd: float,
        start_epoch: int,
        context_frames: int,
        num_slices: int,
        integration_min: float,
        integration_max: float,
        integration_points: int,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.weight = float(weight)
        self.lambd = float(lambd)
        self.start_epoch = max(0, int(start_epoch))
        self.context_frames = int(context_frames)
        self.num_slices = int(num_slices)
        self.integration_min = float(integration_min)
        self.integration_max = float(integration_max)
        self.integration_points = int(integration_points)

        if not (0.0 <= self.lambd <= 1.0):
            raise ValueError(f"lejepa_lambda must be in [0, 1], got {self.lambd}.")
        if self.context_frames <= 0:
            raise ValueError(f"lejepa_context_frames must be > 0, got {self.context_frames}.")
        if self.num_slices <= 0:
            raise ValueError(f"lejepa_num_slices must be > 0, got {self.num_slices}.")
        if self.integration_points < 2:
            raise ValueError(
                "lejepa_integration_points must be >= 2 for trapezoidal integration, "
                f"got {self.integration_points}."
            )
        if self.integration_max <= self.integration_min:
            raise ValueError(
                "lejepa_integration_max must be > lejepa_integration_min, "
                f"got min={self.integration_min}, max={self.integration_max}."
            )

    @classmethod
    def from_config(cls, cfg, *, sequence_length: int):
        sequence_length = int(sequence_length)
        context_frames = getattr(cfg, "lejepa_context_frames", None)
        if context_frames is None:
            context_frames = sequence_length - 1
        context_frames = int(context_frames)
        if context_frames >= sequence_length:
            raise ValueError(
                "LeJEPA requires lejepa_context_frames < data.sequence_length so one frame remains as target. "
                f"Got lejepa_context_frames={context_frames}, data.sequence_length={sequence_length}."
            )

        return cls(
            enabled=bool(getattr(cfg, "lejepa_enabled", False)),
            weight=float(getattr(cfg, "lejepa_weight", 0.0)),
            lambd=float(getattr(cfg, "lejepa_lambda", 0.05)),
            start_epoch=int(getattr(cfg, "lejepa_start_epoch", 0)),
            context_frames=context_frames,
            num_slices=int(getattr(cfg, "lejepa_num_slices", 512)),
            integration_min=float(getattr(cfg, "lejepa_integration_min", -5.0)),
            integration_max=float(getattr(cfg, "lejepa_integration_max", 5.0)),
            integration_points=int(getattr(cfg, "lejepa_integration_points", 17)),
        )

    def should_run(self, *, current_epoch: int) -> bool:
        return bool(
            self.enabled
            and self.weight > 0.0
            and int(current_epoch) >= self.start_epoch
        )

    def compute_loss(
        self,
        *,
        frame_embeddings: torch.Tensor,
        current_epoch: int,
        global_step: int,
    ):
        if not self.should_run(current_epoch=current_epoch):
            return None, {}

        expected_views = self.context_frames + 1
        self._validate_frame_embeddings(frame_embeddings, expected_views=expected_views)

        context_embeddings = frame_embeddings[:, : self.context_frames]
        target_embeddings = frame_embeddings[:, self.context_frames]
        context_center = context_embeddings.mean(dim=1)
        sim = (context_center - target_embeddings).square().mean()

        sigreg_terms = [
            self._sigreg(frame_embeddings[:, view_idx], global_step=int(global_step))
            for view_idx in range(expected_views)
        ]
        sigreg = torch.stack(sigreg_terms, dim=0).mean()
        loss = (1.0 - self.lambd) * sim + self.lambd * sigreg

        metrics = {
            "lejepa_sim": sim,
            "lejepa_sigreg": sigreg,
        }
        if not torch.isfinite(loss).item():
            metrics["lejepa_nonfinite"] = frame_embeddings.new_tensor(1.0)
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss, metrics

    @staticmethod
    def _validate_frame_embeddings(
        frame_embeddings: torch.Tensor,
        *,
        expected_views: int,
    ) -> None:
        if not torch.is_tensor(frame_embeddings):
            raise TypeError(f"frame_embeddings must be a torch.Tensor, got {type(frame_embeddings)}.")
        if frame_embeddings.dim() != 3:
            raise ValueError(
                "LeJEPA expects frame_embeddings with shape (B, T, D), "
                f"got {tuple(frame_embeddings.shape)}."
            )
        if int(frame_embeddings.shape[1]) != int(expected_views):
            raise ValueError(
                "LeJEPA frame embedding count does not match configured context window. "
                f"Expected {expected_views} views (context + target), got {int(frame_embeddings.shape[1])}."
            )

    def _sample_projection_matrix(
        self,
        *,
        embedding_dim: int,
        device,
        global_step: int,
    ) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(global_step))
        projections = torch.randn(
            (int(embedding_dim), self.num_slices),
            generator=generator,
            dtype=torch.float32,
        )
        projections = projections.to(device=device)
        norms = projections.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
        return projections / norms

    @staticmethod
    def _distributed_sum_(
        real_sum: torch.Tensor,
        imag_sum: torch.Tensor,
        sample_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(real_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(imag_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
        return real_sum, imag_sum, sample_count

    def _sigreg(
        self,
        embeddings: torch.Tensor,
        *,
        global_step: int,
    ) -> torch.Tensor:
        if embeddings.dim() != 2:
            raise ValueError(
                "LeJEPA SIGReg expects embeddings with shape (B, D), "
                f"got {tuple(embeddings.shape)}."
            )

        x = embeddings.to(dtype=torch.float32)
        projection_matrix = self._sample_projection_matrix(
            embedding_dim=int(x.shape[1]),
            device=x.device,
            global_step=int(global_step),
        )
        t = torch.linspace(
            self.integration_min,
            self.integration_max,
            self.integration_points,
            device=x.device,
            dtype=torch.float32,
        )
        gaussian_cf = torch.exp(-0.5 * t.square()).unsqueeze(0)

        projected = x @ projection_matrix
        x_t = projected.unsqueeze(-1) * t.view(1, 1, -1)
        real_sum = torch.cos(x_t).sum(dim=0)
        imag_sum = torch.sin(x_t).sum(dim=0)
        sample_count = x.new_tensor(float(x.shape[0]), dtype=torch.float32)
        real_sum, imag_sum, sample_count = self._distributed_sum_(
            real_sum,
            imag_sum,
            sample_count,
        )
        sample_count = sample_count.clamp_min(1.0)

        empirical_real = real_sum / sample_count
        empirical_imag = imag_sum / sample_count
        err = ((empirical_real - gaussian_cf) ** 2 + empirical_imag.square()) * gaussian_cf
        slice_statistics = torch.trapz(err, t, dim=1) * sample_count
        return slice_statistics.mean()
