import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.distributed.nn.functional as dist_nn_functional
except ImportError as exc:
    dist_nn_functional = None
    _DIST_NN_FUNCTIONAL_IMPORT_ERROR = exc
else:
    _DIST_NN_FUNCTIONAL_IMPORT_ERROR = None


class LineJEPALoss(nn.Module):
    """Prediction loss with optional representation regularization for Line-JEPA."""

    def __init__(
        self,
        *,
        weight: float,
        prediction_coeff: float,
        sigreg_coeff: float,
        std_coeff: float,
        cov_coeff: float,
        std_eps: float,
        std_target: float,
        start_epoch: int,
        prediction_loss: str,
        num_slices: int,
        integration_min: float,
        integration_max: float,
        integration_points: int,
        context_match_coeff: float = 0.0,
        context_match_temperature: float = 0.1,
        context_match_negative_count: int = 64,
        context_match_negative_max_target_cosine: float = 0.98,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.prediction_coeff = float(prediction_coeff)
        self.sigreg_coeff = float(sigreg_coeff)
        self.std_coeff = float(std_coeff)
        self.cov_coeff = float(cov_coeff)
        self.std_eps = float(std_eps)
        self.std_target = float(std_target)
        self.start_epoch = max(0, int(start_epoch))
        self.prediction_loss = str(prediction_loss).strip().lower()
        self.num_slices = int(num_slices)
        self.integration_min = float(integration_min)
        self.integration_max = float(integration_max)
        self.integration_points = int(integration_points)
        self.context_match_coeff = float(context_match_coeff)
        self.context_match_temperature = float(context_match_temperature)
        self.context_match_negative_count = int(context_match_negative_count)
        self.context_match_negative_max_target_cosine = float(
            context_match_negative_max_target_cosine
        )

        if self.weight <= 0.0:
            raise ValueError(f"line_jepa_weight must be > 0 for Line-JEPA, got {self.weight}.")
        if self.prediction_coeff < 0.0:
            raise ValueError(f"line_jepa_prediction_coeff must be >= 0, got {self.prediction_coeff}.")
        if self.sigreg_coeff < 0.0:
            raise ValueError(f"line_jepa_sigreg_coeff must be >= 0, got {self.sigreg_coeff}.")
        if self.std_coeff < 0.0:
            raise ValueError(f"line_jepa_std_coeff must be >= 0, got {self.std_coeff}.")
        if self.cov_coeff < 0.0:
            raise ValueError(f"line_jepa_cov_coeff must be >= 0, got {self.cov_coeff}.")
        if self.std_eps <= 0.0:
            raise ValueError(f"line_jepa_std_eps must be > 0, got {self.std_eps}.")
        if self.std_target <= 0.0:
            raise ValueError(f"line_jepa_std_target must be > 0, got {self.std_target}.")
        if self.prediction_loss not in {"mse", "smooth_l1", "cosine"}:
            raise ValueError(
                "line_jepa_prediction_loss must be 'mse', 'smooth_l1', or 'cosine', "
                f"got {prediction_loss!r}."
            )
        if self.num_slices <= 0:
            raise ValueError(f"line_jepa_sigreg_num_slices must be > 0, got {self.num_slices}.")
        if self.integration_points < 2:
            raise ValueError(
                "line_jepa_sigreg_integration_points must be >= 2 for trapezoidal integration, "
                f"got {self.integration_points}."
            )
        if self.integration_max <= self.integration_min:
            raise ValueError(
                "line_jepa_sigreg_integration_max must be > line_jepa_sigreg_integration_min, "
                f"got min={self.integration_min}, max={self.integration_max}."
            )
        if self.context_match_coeff < 0.0:
            raise ValueError(
                "line_jepa_context_match_coeff must be >= 0, "
                f"got {self.context_match_coeff}."
            )
        if self.context_match_temperature <= 0.0:
            raise ValueError(
                "line_jepa_context_match_temperature must be > 0, "
                f"got {self.context_match_temperature}."
            )
        if self.context_match_negative_count <= 0:
            raise ValueError(
                "line_jepa_context_match_negative_count must be > 0, "
                f"got {self.context_match_negative_count}."
            )
        if not (-1.0 <= self.context_match_negative_max_target_cosine < 1.0):
            raise ValueError(
                "line_jepa_context_match_negative_max_target_cosine must be in [-1, 1), "
                f"got {self.context_match_negative_max_target_cosine}."
            )

    @classmethod
    def from_config(cls, cfg):
        return cls(
            weight=float(getattr(cfg, "line_jepa_weight", 1.0)),
            prediction_coeff=float(getattr(cfg, "line_jepa_prediction_coeff", 1.0)),
            sigreg_coeff=float(getattr(cfg, "line_jepa_sigreg_coeff", 0.05)),
            std_coeff=float(getattr(cfg, "line_jepa_std_coeff", 0.0)),
            cov_coeff=float(getattr(cfg, "line_jepa_cov_coeff", 0.0)),
            std_eps=float(getattr(cfg, "line_jepa_std_eps", 1e-4)),
            std_target=float(getattr(cfg, "line_jepa_std_target", 1.0)),
            start_epoch=int(getattr(cfg, "line_jepa_start_epoch", 0)),
            prediction_loss=str(getattr(cfg, "line_jepa_prediction_loss", "mse")),
            num_slices=int(getattr(cfg, "line_jepa_sigreg_num_slices", 512)),
            integration_min=float(getattr(cfg, "line_jepa_sigreg_integration_min", -5.0)),
            integration_max=float(getattr(cfg, "line_jepa_sigreg_integration_max", 5.0)),
            integration_points=int(getattr(cfg, "line_jepa_sigreg_integration_points", 17)),
            context_match_coeff=float(getattr(cfg, "line_jepa_context_match_coeff", 0.0)),
            context_match_temperature=float(
                getattr(cfg, "line_jepa_context_match_temperature", 0.1)
            ),
            context_match_negative_count=int(
                getattr(cfg, "line_jepa_context_match_negative_count", 64)
            ),
            context_match_negative_max_target_cosine=float(
                getattr(cfg, "line_jepa_context_match_negative_max_target_cosine", 0.98)
            ),
        )

    def should_run(self, *, current_epoch: int) -> bool:
        return int(current_epoch) >= self.start_epoch

    def compute_loss(
        self,
        *,
        prediction: torch.Tensor | None,
        target: torch.Tensor | None,
        regularized_embeddings: dict[str, torch.Tensor] | list[torch.Tensor],
        current_epoch: int,
        global_step: int,
        prediction_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if not self.should_run(current_epoch=current_epoch):
            return None, {}
        embedding_items = self._regularized_embedding_items(regularized_embeddings)
        device = self._loss_device(prediction=prediction, embedding_items=embedding_items)
        needs_regularized_embeddings = (
            self.sigreg_coeff > 0.0
            or self.std_coeff > 0.0
            or self.cov_coeff > 0.0
        )
        if needs_regularized_embeddings and not embedding_items:
            raise ValueError(
                "LineJEPALoss requires at least one embedding tensor when "
                "line_jepa_sigreg_coeff, line_jepa_std_coeff, or line_jepa_cov_coeff is positive."
            )

        prediction_active = self.prediction_coeff > 0.0
        if prediction_active:
            if prediction is None or target is None:
                raise ValueError(
                    "LineJEPALoss requires prediction and target tensors when "
                    f"line_jepa_prediction_coeff={self.prediction_coeff}."
                )
            self._validate_prediction_inputs(prediction=prediction, target=target)
            pred = prediction.float()
            target_detached = target.detach().float()
            prediction_loss = self._prediction_loss(
                prediction=pred,
                target=target_detached,
                prediction_weights=prediction_weights,
            )
        else:
            if prediction_weights is not None:
                raise ValueError(
                    "LineJEPALoss received prediction_weights while "
                    "line_jepa_prediction_coeff is 0. Disable hard prediction weighting."
                )
            prediction_loss = torch.zeros((), device=device, dtype=torch.float32)

        if self.context_match_coeff > 0.0:
            if not prediction_active or prediction is None or target is None:
                raise ValueError(
                    "line_jepa_context_match_coeff > 0 requires active prediction and aligned "
                    "prediction/target tensors."
                )
            context_match, context_match_metrics = self._context_match_loss(
                prediction=prediction.float(),
                target=target.detach().float(),
            )
        else:
            context_match = torch.zeros((), device=device, dtype=torch.float32)
            context_match_metrics = {}

        pooled_embeddings = (
            self._pooled_regularizer_embeddings(embedding_items)
            if needs_regularized_embeddings
            else None
        )
        sigreg = self._pooled_regularizer(
            pooled_embeddings,
            coeff=self.sigreg_coeff,
            regularizer=lambda embeddings: self._sigreg(
                embeddings,
                global_step=int(global_step),
            ),
            device=device,
        )
        std_loss = self._pooled_regularizer(
            pooled_embeddings,
            coeff=self.std_coeff,
            regularizer=self._variance_loss,
            device=device,
        )
        cov_loss = self._pooled_regularizer(
            pooled_embeddings,
            coeff=self.cov_coeff,
            regularizer=self._covariance_loss,
            device=device,
        )
        unweighted = (
            self.prediction_coeff * prediction_loss
            + self.context_match_coeff * context_match
            + self.sigreg_coeff * sigreg
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        loss = self.weight * unweighted

        metrics = {}
        if prediction_active:
            metrics["jepa/pred/loss"] = prediction_loss
        if self.context_match_coeff > 0.0:
            metrics["jepa/sim/loss"] = context_match
            metrics.update(context_match_metrics)
        if self.sigreg_coeff > 0.0:
            metrics["jepa/reg/sigreg"] = sigreg
        if self.std_coeff > 0.0:
            metrics["jepa/reg/std"] = std_loss
        if self.cov_coeff > 0.0:
            metrics["jepa/reg/cov"] = cov_loss
        if not torch.isfinite(loss).item():
            raise RuntimeError(
                "Line-JEPA loss became non-finite. "
                f"prediction_loss={float(prediction_loss.detach())}, "
                f"context_match={float(context_match.detach())}, "
                f"sigreg={float(sigreg.detach())}, std={float(std_loss.detach())}, "
                f"cov={float(cov_loss.detach())}, weight={self.weight}, "
                f"prediction_coeff={self.prediction_coeff}, "
                f"context_match_coeff={self.context_match_coeff}, "
                f"sigreg_coeff={self.sigreg_coeff}, std_coeff={self.std_coeff}, "
                f"cov_coeff={self.cov_coeff}."
            )
        return loss, metrics

    def _context_match_loss(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Match each prediction to its target using data-agnostic hard in-batch negatives."""
        if prediction.shape != target.shape or prediction.dim() != 2:
            raise ValueError(
                "Line-JEPA context matching expects aligned (B, D) tensors, "
                f"got prediction={tuple(prediction.shape)}, target={tuple(target.shape)}."
            )
        batch_size = int(prediction.shape[0])
        if batch_size <= 1:
            raise ValueError(
                "Line-JEPA context matching requires at least two prediction tasks per batch, "
                f"got {batch_size}."
            )

        target_mean = target.mean(dim=0, keepdim=True)
        centered_target = F.normalize(target - target_mean, dim=-1, eps=1.0e-6)
        centered_prediction = F.normalize(prediction - target_mean, dim=-1, eps=1.0e-6)
        target_similarity = centered_target @ centered_target.T
        eye = torch.eye(batch_size, dtype=torch.bool, device=target.device)
        valid_negative = (~eye) & (
            target_similarity <= self.context_match_negative_max_target_cosine
        )
        valid_count = valid_negative.sum(dim=1)
        if torch.any(valid_count == 0).item():
            bad_count = int((valid_count == 0).sum().item())
            raise RuntimeError(
                "Line-JEPA context matching found no structurally distinct in-batch negative "
                f"for {bad_count}/{batch_size} tasks. Increase batch diversity or increase "
                "line_jepa_context_match_negative_max_target_cosine "
                f"(currently {self.context_match_negative_max_target_cosine})."
            )

        negative_count = min(self.context_match_negative_count, batch_size - 1)
        candidate_scores = target_similarity.masked_fill(~valid_negative, -torch.inf)
        _, negative_indices = torch.topk(
            candidate_scores,
            k=negative_count,
            dim=1,
            largest=True,
            sorted=False,
        )
        selected_valid = valid_negative.gather(1, negative_indices)
        prediction_target_logits = centered_prediction @ centered_target.T
        positive_logits = prediction_target_logits.diagonal().unsqueeze(1)
        negative_logits = prediction_target_logits.gather(1, negative_indices)
        negative_logits = negative_logits.masked_fill(~selected_valid, -torch.inf)
        logits = torch.cat((positive_logits, negative_logits), dim=1)
        logits = logits / self.context_match_temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=prediction.device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            hardest_negative = negative_logits.max(dim=1).values
            accuracy = (positive_logits.squeeze(1) > hardest_negative).float().mean()
        return loss, {
            "jepa/sim/top1": accuracy,
        }

    @staticmethod
    def _loss_device(
        *,
        prediction: torch.Tensor | None,
        embedding_items: list[tuple[str, torch.Tensor]],
    ):
        if prediction is not None:
            return prediction.device
        if embedding_items:
            return embedding_items[0][1].device
        return torch.device("cpu")

    @staticmethod
    def _regularized_embedding_items(
        regularized_embeddings: dict[str, torch.Tensor] | list[torch.Tensor],
    ) -> list[tuple[str, torch.Tensor]]:
        if isinstance(regularized_embeddings, dict):
            return [(str(name), value) for name, value in regularized_embeddings.items()]
        return [
            (f"embedding_{index}", value)
            for index, value in enumerate(regularized_embeddings)
        ]

    @staticmethod
    def _pooled_regularizer_embeddings(
        embedding_items: list[tuple[str, torch.Tensor]],
    ) -> torch.Tensor:
        if not embedding_items:
            raise ValueError("Line-JEPA regularizer pool received no embedding tensors.")
        feature_dim = None
        tensors = []
        for name, embeddings in embedding_items:
            if embeddings.dim() != 2:
                raise ValueError(
                    "Line-JEPA regularizer pool expects every embedding tensor to have shape (B, D). "
                    f"Got name={name!r}, shape={tuple(embeddings.shape)}."
                )
            if int(embeddings.shape[0]) == 0:
                raise ValueError(f"Line-JEPA regularizer pool received empty tensor name={name!r}.")
            current_dim = int(embeddings.shape[1])
            if feature_dim is None:
                feature_dim = current_dim
            elif current_dim != feature_dim:
                raise ValueError(
                    "Line-JEPA regularizer pool requires a shared feature dimension. "
                    f"Got name={name!r}, dim={current_dim}, expected_dim={feature_dim}."
                )
            tensors.append(embeddings)
        return torch.cat(tensors, dim=0)

    @staticmethod
    def _pooled_regularizer(
        embeddings: torch.Tensor | None,
        *,
        coeff: float,
        regularizer,
        device,
    ) -> torch.Tensor:
        if coeff <= 0.0:
            return torch.zeros((), device=device, dtype=torch.float32)
        if embeddings is None:
            raise ValueError("Line-JEPA regularizer is active, but the pooled embeddings tensor is missing.")
        return regularizer(embeddings)

    @staticmethod
    def _validate_prediction_inputs(*, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if not torch.is_tensor(prediction):
            raise TypeError(f"prediction must be a torch.Tensor, got {type(prediction)}.")
        if not torch.is_tensor(target):
            raise TypeError(f"target must be a torch.Tensor, got {type(target)}.")
        if prediction.dim() != 2 or target.dim() != 2:
            raise ValueError(
                "Line-JEPA prediction and target embeddings must have shape (B, D). "
                f"Got prediction={tuple(prediction.shape)}, target={tuple(target.shape)}."
            )
        if prediction.shape != target.shape:
            raise ValueError(
                "Line-JEPA prediction and target embeddings must have identical shapes. "
                f"Got prediction={tuple(prediction.shape)}, target={tuple(target.shape)}."
            )

    def _prediction_loss(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        prediction_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.prediction_loss == "cosine":
            per_sample = 1.0 - F.cosine_similarity(prediction, target, dim=-1)
        elif self.prediction_loss == "mse":
            per_dim = F.mse_loss(prediction, target, reduction="none")
            per_sample = per_dim.mean(dim=1)
        elif self.prediction_loss == "smooth_l1":
            per_dim = F.smooth_l1_loss(prediction, target, reduction="none")
            per_sample = per_dim.mean(dim=1)
        else:
            raise RuntimeError(f"Unsupported prediction loss at runtime: {self.prediction_loss!r}.")
        if prediction_weights is None:
            return per_sample.mean()
        weights = prediction_weights.detach().to(device=per_sample.device, dtype=per_sample.dtype).reshape(-1)
        if int(weights.numel()) != int(per_sample.numel()):
            raise ValueError(
                "Line-JEPA prediction_weights must have one value per prediction row. "
                f"Got weights={tuple(weights.shape)}, prediction={tuple(prediction.shape)}."
            )
        if not torch.isfinite(weights).all().item():
            raise ValueError("Line-JEPA prediction_weights contain non-finite values.")
        if torch.any(weights < 0.0).item():
            raise ValueError("Line-JEPA prediction_weights must be non-negative.")
        return (per_sample * weights).mean()

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
    def _distributed_sum_tensor_(tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return tensor
        if tensor.requires_grad:
            if dist_nn_functional is None:
                raise RuntimeError(
                    "Line-JEPA distributed SIGReg requires "
                    "torch.distributed.nn.functional.all_reduce for gradient-carrying tensors, "
                    f"but it is unavailable. tensor_shape={tuple(tensor.shape)}, "
                    f"tensor_dtype={tensor.dtype}. Use a PyTorch build that provides "
                    "torch.distributed.nn.functional, or disable multi-device Line-JEPA."
                ) from _DIST_NN_FUNCTIONAL_IMPORT_ERROR
            return dist_nn_functional.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    @staticmethod
    def _distributed_sum_(
        real_sum: torch.Tensor,
        imag_sum: torch.Tensor,
        sample_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real_sum = LineJEPALoss._distributed_sum_tensor_(real_sum)
        imag_sum = LineJEPALoss._distributed_sum_tensor_(imag_sum)
        sample_count = LineJEPALoss._distributed_sum_tensor_(sample_count)
        return real_sum, imag_sum, sample_count

    @staticmethod
    def _gather_all(z: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return z
        world_size = dist.get_world_size()
        if world_size <= 1:
            return z

        local_size = torch.tensor([z.shape[0]], device=z.device, dtype=torch.long)
        sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)
        sizes = [int(size.item()) for size in sizes]
        max_size = max(sizes)
        if max_size == 0:
            return z

        if z.shape[0] != max_size:
            pad = z.new_zeros((max_size - z.shape[0], *z.shape[1:]))
            z_padded = torch.cat([z, pad], dim=0)
        else:
            z_padded = z
        z_padded = z_padded.contiguous()

        gathered = [z_padded.new_zeros((max_size, *z.shape[1:])) for _ in range(world_size)]
        dist.all_gather(gathered, z_padded)
        gathered[dist.get_rank()] = z_padded
        return torch.cat([chunk[:size] for chunk, size in zip(gathered, sizes)], dim=0)

    def _validate_regularizer_embeddings(self, embeddings: torch.Tensor, *, name: str) -> torch.Tensor:
        if embeddings.dim() != 2:
            raise ValueError(
                f"Line-JEPA {name} expects embeddings with shape (B, D), "
                f"got {tuple(embeddings.shape)}."
            )
        if embeddings.shape[0] == 0:
            raise ValueError(f"Line-JEPA {name} received an empty embedding batch.")
        return embeddings.to(dtype=torch.float32)

    def _variance_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = self._gather_all(
            self._validate_regularizer_embeddings(embeddings, name="variance regularizer")
        )
        var = x.var(dim=0, unbiased=False)
        std = torch.sqrt(var + self.std_eps)
        return torch.mean(F.relu(self.std_target - std))

    def _covariance_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = self._gather_all(
            self._validate_regularizer_embeddings(embeddings, name="covariance regularizer")
        )
        if x.shape[0] < 2:
            return x.new_tensor(0.0)
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (x.shape[0] - 1)
        return self._off_diagonal(cov).pow(2).sum() / x.shape[1]

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        if n != m:
            raise ValueError(f"Input must be a square matrix, got shape={tuple(x.shape)}.")
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _sigreg(
        self,
        embeddings: torch.Tensor,
        *,
        global_step: int,
    ) -> torch.Tensor:
        x = self._validate_regularizer_embeddings(embeddings, name="SIGReg")
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
        slice_statistics = torch.trapz(err, t, dim=1)
        return slice_statistics.mean()


__all__ = ["LineJEPALoss"]
