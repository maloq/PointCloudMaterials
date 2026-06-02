from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class SwAVLoss(nn.Module):
    """Swapped assignment prediction loss over learned prototypes."""

    def __init__(
        self,
        *,
        enabled: bool,
        weight: float,
        input_dim: int | None,
        projection_dim: int,
        hidden_dim: int,
        num_prototypes: int,
        temperature: float,
        epsilon: float,
        sinkhorn_iterations: int,
        start_epoch: int,
        freeze_prototypes_steps: int,
        view_mode: str,
        view_points: int | None,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.weight = float(weight)
        self.input_dim = int(input_dim) if input_dim is not None else None
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_prototypes = int(num_prototypes)
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.start_epoch = max(0, int(start_epoch))
        self.freeze_prototypes_steps = max(0, int(freeze_prototypes_steps))
        self.view_mode = str(view_mode).strip().lower()
        self.view_points = int(view_points) if view_points is not None else None

        self.projector = None
        self.prototypes = None
        if self.enabled and self.weight > 0.0:
            if self.input_dim is None:
                raise ValueError("SwAV requires a resolved encoder latent dimension.")
            if self.hidden_dim == 0:
                self.projector = nn.Linear(self.input_dim, self.projection_dim)
            else:
                self.projector = nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_dim, self.projection_dim),
                )
            self.prototypes = nn.Linear(self.projection_dim, self.num_prototypes, bias=False)

    @classmethod
    def from_config(cls, cfg, *, input_dim: int | None):
        data_cfg = getattr(cfg, "data", None)
        view_points = getattr(cfg, "swav_view_points", None)
        if view_points is None and data_cfg is not None:
            view_points = getattr(data_cfg, "model_points", None)
        if view_points is None and data_cfg is not None:
            view_points = getattr(data_cfg, "num_points", None)
        if view_points is None:
            view_points = getattr(cfg, "model_points", None)

        return cls(
            enabled=bool(getattr(cfg, "swav_enabled", False)),
            weight=float(getattr(cfg, "swav_weight", 0.0)),
            input_dim=input_dim,
            projection_dim=int(getattr(cfg, "swav_projection_dim", 128)),
            hidden_dim=int(getattr(cfg, "swav_hidden_dim", 512)),
            num_prototypes=int(getattr(cfg, "swav_num_prototypes", 3000)),
            temperature=float(getattr(cfg, "swav_temperature", 0.1)),
            epsilon=float(getattr(cfg, "swav_epsilon", 0.05)),
            sinkhorn_iterations=int(getattr(cfg, "swav_sinkhorn_iterations", 20)),
            start_epoch=int(getattr(cfg, "swav_start_epoch", 0)),
            freeze_prototypes_steps=int(getattr(cfg, "swav_freeze_prototypes_steps", 0)),
            view_mode=str(getattr(cfg, "swav_view_mode", "center_adjacent")),
            view_points=view_points,
        )

    def should_run(self, *, current_epoch: int) -> bool:
        return bool(
            self.enabled
            and self.weight > 0.0
            and self.projector is not None
            and self.prototypes is not None
            and int(current_epoch) >= self.start_epoch
        )

    def should_freeze_prototypes(self, *, global_step: int) -> bool:
        return bool(
            self.should_run(current_epoch=self.start_epoch)
            and int(global_step) < self.freeze_prototypes_steps
        )

    def clear_prototype_gradients(self) -> None:
        if self.prototypes is None:
            return
        for parameter in self.prototypes.parameters():
            parameter.grad = None

    @staticmethod
    def _distributed_is_initialized() -> bool:
        return dist.is_available() and dist.is_initialized()

    @classmethod
    def _distributed_sum(cls, value: torch.Tensor) -> torch.Tensor:
        if cls._distributed_is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    @classmethod
    def _distributed_max(cls, value: torch.Tensor) -> torch.Tensor:
        if cls._distributed_is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return value

    @torch.no_grad()
    def normalize_prototypes(self) -> None:
        if self.prototypes is None:
            raise RuntimeError("Cannot normalize SwAV prototypes before they are initialized.")
        weight = self.prototypes.weight.data
        self.prototypes.weight.copy_(F.normalize(weight, dim=1, p=2))

    def _prototype_logits(self, features: torch.Tensor) -> torch.Tensor:
        if self.projector is None or self.prototypes is None:
            raise RuntimeError("SwAV projector/prototypes are not initialized.")
        projector_dtype = next(self.projector.parameters()).dtype
        projected = self.projector(features.to(dtype=projector_dtype))
        projected = F.normalize(projected, dim=1, p=2)
        return self.prototypes(projected)

    @torch.no_grad()
    def _sinkhorn(self, logits: torch.Tensor, *, iterations: int | None = None) -> torch.Tensor:
        sinkhorn_iterations = self.sinkhorn_iterations if iterations is None else int(iterations)

        scores = logits.detach().to(dtype=torch.float32) / self.epsilon
        max_score = scores.max()
        max_score = self._distributed_max(max_score)
        q = torch.exp(scores - max_score).t()

        sum_q = q.sum()
        sum_q = self._distributed_sum(sum_q)
        q = q / sum_q.clamp_min(1e-12)

        local_batch = q.new_tensor(float(q.shape[1]))
        global_batch = self._distributed_sum(local_batch)
        num_prototypes = q.new_tensor(float(q.shape[0]))

        for _ in range(sinkhorn_iterations):
            row_sums = q.sum(dim=1, keepdim=True)
            row_sums = self._distributed_sum(row_sums)
            q = q / row_sums.clamp_min(1e-12)
            q = q / num_prototypes

            q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
            q = q / global_batch

        q = q * global_batch
        return q.t()

    def compute_loss(
        self,
        *,
        view_features: list[torch.Tensor],
        current_epoch: int,
    ):
        if not self.should_run(current_epoch=current_epoch):
            return None, {}

        batch_size = int(view_features[0].shape[0])

        self.normalize_prototypes()
        logits_by_view = [self._prototype_logits(features) for features in view_features]
        loss_terms = []
        assignment_max_probs = []
        assignment_entropies = []
        assignment_prototype_masses = []
        for assign_idx, assignment_logits in enumerate(logits_by_view):
            assignments = self._sinkhorn(assignment_logits)
            assignment_max_probs.append(assignments.max(dim=1).values.mean())
            assignment_entropy = -(assignments * assignments.clamp_min(1e-12).log()).sum(dim=1).mean()
            assignment_entropies.append(assignment_entropy)
            assignment_prototype_masses.append(assignments.sum(dim=0) / float(batch_size))

            subloss_terms = []
            for pred_idx, prediction_logits in enumerate(logits_by_view):
                if pred_idx == assign_idx:
                    continue
                log_probs = F.log_softmax(prediction_logits / self.temperature, dim=1)
                subloss_terms.append(-(assignments * log_probs).sum(dim=1).mean())
            loss_terms.append(torch.stack(subloss_terms, dim=0).mean())

        loss = torch.stack(loss_terms, dim=0).mean()
        metrics = {
            "swav_assignment_entropy": torch.stack(assignment_entropies, dim=0).mean(),
            "swav_assignment_max_prob": torch.stack(assignment_max_probs, dim=0).mean(),
        }
        prototype_mass = torch.stack(assignment_prototype_masses, dim=0).mean(dim=0)
        metrics["swav_prototype_mass_min"] = prototype_mass.min()
        metrics["swav_prototype_mass_max"] = prototype_mass.max()
        nonfinite = (~torch.isfinite(loss)).to(dtype=loss.dtype)
        metrics["swav_nonfinite"] = nonfinite
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss, metrics

    def forward(self, features: torch.Tensor, *, profile_logits: bool = False) -> torch.Tensor:
        return self._prototype_logits(features)


__all__ = ["SwAVLoss"]
