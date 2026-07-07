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
        prior_type: str,
        prior_exponent: float,
        prior_probs,
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
        self.prior_type = str(prior_type).strip().lower()
        self.prior_exponent = float(prior_exponent)

        if self.num_prototypes <= 0:
            raise ValueError(f"swav_num_prototypes must be > 0, got {self.num_prototypes}.")
        if self.prior_type not in {"uniform", "powerlaw", "custom"}:
            raise ValueError(
                "swav_prior_type must be one of ['uniform', 'powerlaw', 'custom'], "
                f"got {prior_type!r}."
            )
        if self.prior_type == "powerlaw" and self.prior_exponent <= 0.0:
            raise ValueError(
                "swav_prior_exponent must be > 0 for swav_prior_type='powerlaw', "
                f"got {self.prior_exponent}."
            )

        explicit_prior = self._build_explicit_prior(prior_probs)
        if explicit_prior.numel() > 0 and self.prior_type != "custom":
            raise ValueError(
                "swav_prior_probs was provided, so swav_prior_type must be 'custom' "
                f"or omitted; got {self.prior_type!r}."
            )
        if explicit_prior.numel() == 0 and self.prior_type == "custom":
            raise ValueError("swav_prior_type='custom' requires swav_prior_probs.")
        self.register_buffer("_explicit_prior", explicit_prior, persistent=True)

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
        prior_probs = getattr(cfg, "swav_prior_probs", None)
        raw_prior_type = getattr(cfg, "swav_prior_type", None)
        if raw_prior_type is None:
            prior_type = "custom" if prior_probs is not None else "uniform"
        else:
            prior_type = str(raw_prior_type)

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
            prior_type=prior_type,
            prior_exponent=float(getattr(cfg, "swav_prior_exponent", 1.0)),
            prior_probs=prior_probs,
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

    def _build_explicit_prior(self, prior_probs) -> torch.Tensor:
        if prior_probs is None:
            return torch.empty(0, dtype=torch.float32)
        prior = torch.as_tensor(list(prior_probs), dtype=torch.float32)
        if prior.numel() != self.num_prototypes:
            raise ValueError(
                "swav_prior_probs length must match swav_num_prototypes: "
                f"got {prior.numel()} values for {self.num_prototypes} prototypes."
            )
        if not torch.isfinite(prior).all():
            raise ValueError(f"swav_prior_probs must be finite, got {prior.tolist()}.")
        if (prior <= 0.0).any():
            raise ValueError(f"swav_prior_probs must be strictly positive, got {prior.tolist()}.")
        prior_sum = prior.sum()
        if prior_sum <= 0.0:
            raise ValueError(f"swav_prior_probs must have positive sum, got {prior.tolist()}.")
        return prior / prior_sum

    def _assignment_prior(self, *, device, dtype) -> torch.Tensor:
        if self._explicit_prior.numel() > 0:
            return self._explicit_prior.to(device=device, dtype=dtype)
        if self.prior_type == "uniform":
            return torch.full(
                (self.num_prototypes,),
                1.0 / float(self.num_prototypes),
                device=device,
                dtype=dtype,
            )
        if self.prior_type == "powerlaw":
            ranks = torch.arange(
                1,
                self.num_prototypes + 1,
                device=device,
                dtype=dtype,
            )
            prior = ranks.pow(-self.prior_exponent)
            return prior / prior.sum().clamp_min(1e-12)
        raise RuntimeError(f"Unsupported SwAV prior type at runtime: {self.prior_type!r}.")

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
        prototype_prior = self._assignment_prior(device=q.device, dtype=q.dtype).view(-1, 1)

        for _ in range(sinkhorn_iterations):
            row_sums = q.sum(dim=1, keepdim=True)
            row_sums = self._distributed_sum(row_sums)
            q = q / row_sums.clamp_min(1e-12)
            q = q * prototype_prior

            q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
            q = q / global_batch

        q = q * global_batch
        return q.t()

    def _global_assignment_mass(self, assignments: torch.Tensor) -> torch.Tensor:
        local_mass = assignments.sum(dim=0)
        global_mass = self._distributed_sum(local_mass)
        local_batch = assignments.new_tensor(float(assignments.shape[0]))
        global_batch = self._distributed_sum(local_batch)
        return global_mass / global_batch.clamp_min(1e-12)

    def compute_loss(
        self,
        *,
        view_features: list[torch.Tensor],
        current_epoch: int,
    ):
        if not self.should_run(current_epoch=current_epoch):
            return None, {}

        self.normalize_prototypes()
        logits_by_view = [self._prototype_logits(features) for features in view_features]
        loss_terms = []
        assignment_entropies = []
        assignment_prototype_masses = []
        for assign_idx, assignment_logits in enumerate(logits_by_view):
            assignments = self._sinkhorn(assignment_logits)
            assignment_entropy = -(assignments * assignments.clamp_min(1e-12).log()).sum(dim=1).mean()
            assignment_entropies.append(assignment_entropy)
            assignment_prototype_masses.append(self._global_assignment_mass(assignments))

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
        }
        prototype_mass = torch.stack(assignment_prototype_masses, dim=0).mean(dim=0)
        prototype_prior = self._assignment_prior(
            device=prototype_mass.device,
            dtype=prototype_mass.dtype,
        )
        prototype_prior_kl = (
            prototype_mass
            * (prototype_mass.clamp_min(1e-12).log() - prototype_prior.clamp_min(1e-12).log())
        ).sum()
        metrics["swav_prototype_mass_min"] = prototype_mass.min()
        metrics["swav_prototype_mass_max"] = prototype_mass.max()
        metrics["swav_prior_kl"] = prototype_prior_kl
        if not torch.isfinite(loss).item():
            raise RuntimeError(
                "SwAV loss became non-finite. "
                f"view_shapes={[tuple(features.shape) for features in view_features]}, "
                f"logit_shapes={[tuple(logits.shape) for logits in logits_by_view]}, "
                f"temperature={self.temperature}, epsilon={self.epsilon}, "
                f"sinkhorn_iterations={self.sinkhorn_iterations}."
            )
        return loss, metrics

    def forward(self, features: torch.Tensor, *, profile_logits: bool = False) -> torch.Tensor:
        return self._prototype_logits(features)


__all__ = ["SwAVLoss"]
