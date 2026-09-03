import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from src.data_utils.data_kinds import normalize_data_kind
from src.training_methods.base_ssl_module import BaseSSLModule
from src.utils.pointcloud_ops import crop_to_num_points
from src.utils.training_utils import get_optimizers_and_scheduler


class FactorVAELoss(nn.Module):
    """Adversarial total-correlation penalty from FactorVAE (arXiv:1802.05983).

    The discriminator assigns class 0 to samples from the joint latent
    distribution and class 1 to samples from the product of its marginals.
    Independent per-dimension batch permutations produce the latter samples.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        gamma: float,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        discriminator_spectral_norm: bool = False,
        discriminator_coordinate_group_size: int = 0,
        discriminator_bottleneck_dim: int = 0,
        latent_noise_std: float = 0.0,
        start_epoch: int = 0,
        discriminator_warmup_epochs: int = 0,
        gamma_warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.gamma = float(gamma)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.discriminator_spectral_norm = bool(discriminator_spectral_norm)
        configured_group_size = int(discriminator_coordinate_group_size)
        self.discriminator_coordinate_group_size = (
            self.input_dim if configured_group_size == 0 else configured_group_size
        )
        self.discriminator_bottleneck_dim = int(discriminator_bottleneck_dim)
        self.latent_noise_std = float(latent_noise_std)
        self.start_epoch = int(start_epoch)
        self.discriminator_warmup_epochs = int(discriminator_warmup_epochs)
        self.gamma_warmup_epochs = int(gamma_warmup_epochs)

        if self.enabled and self.gamma <= 0.0:
            raise ValueError(
                "factor_vae_gamma must be > 0 when factor_vae_enabled=true, "
                f"got {self.gamma}."
            )
        if self.enabled and self.input_dim <= 0:
            raise ValueError(
                "FactorVAE requires a positive projected latent dimension, "
                f"got {self.input_dim}."
            )
        if self.enabled and self.hidden_dim <= 0:
            raise ValueError(
                "factor_vae_discriminator_hidden_dim must be > 0, "
                f"got {self.hidden_dim}."
            )
        if self.enabled and self.num_hidden_layers <= 0:
            raise ValueError(
                "factor_vae_discriminator_num_hidden_layers must be > 0, "
                f"got {self.num_hidden_layers}."
            )
        if self.enabled and not (
            2 <= self.discriminator_coordinate_group_size <= self.input_dim
        ):
            raise ValueError(
                "factor_vae_discriminator_coordinate_group_size must be 0 (all "
                "coordinates) or between 2 and factor_vae input_dim inclusive; "
                f"got group_size={self.discriminator_coordinate_group_size}, "
                f"input_dim={self.input_dim}."
            )
        if (
            self.enabled
            and self.input_dim % self.discriminator_coordinate_group_size != 0
        ):
            raise ValueError(
                "factor_vae_discriminator_coordinate_group_size must divide the "
                "projected latent dimension so every coordinate is used exactly once; "
                f"got input_dim={self.input_dim}, "
                f"group_size={self.discriminator_coordinate_group_size}."
            )
        if self.enabled and self.discriminator_bottleneck_dim < 0:
            raise ValueError(
                "factor_vae_discriminator_bottleneck_dim must be >= 0, "
                f"got {self.discriminator_bottleneck_dim}."
            )
        if (
            self.enabled
            and self.discriminator_bottleneck_dim
            >= self.discriminator_coordinate_group_size
        ):
            raise ValueError(
                "factor_vae_discriminator_bottleneck_dim must be 0 (disabled) or "
                "strictly smaller than the discriminator coordinate-group size; "
                f"got bottleneck_dim={self.discriminator_bottleneck_dim}, "
                f"group_size={self.discriminator_coordinate_group_size}."
            )
        if self.enabled and self.latent_noise_std < 0.0:
            raise ValueError(
                "factor_vae_latent_noise_std must be >= 0, "
                f"got {self.latent_noise_std}."
            )
        if self.enabled and self.start_epoch < 0:
            raise ValueError(
                "factor_vae_start_epoch must be >= 0, "
                f"got {self.start_epoch}."
            )
        if self.enabled and self.discriminator_warmup_epochs < 0:
            raise ValueError(
                "factor_vae_discriminator_warmup_epochs must be >= 0, "
                f"got {self.discriminator_warmup_epochs}."
            )
        if self.enabled and self.gamma_warmup_epochs < 0:
            raise ValueError(
                "factor_vae_gamma_warmup_epochs must be >= 0, "
                f"got {self.gamma_warmup_epochs}."
            )

        self.discriminator: nn.Sequential | None = None
        if self.enabled:
            layers: list[nn.Module] = []
            layer_input_dim = self.discriminator_coordinate_group_size
            if self.discriminator_bottleneck_dim > 0:
                layers.extend(
                    [
                        self._make_discriminator_linear(
                            layer_input_dim,
                            self.discriminator_bottleneck_dim,
                        ),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ]
                )
                layer_input_dim = self.discriminator_bottleneck_dim
            for _ in range(self.num_hidden_layers):
                layers.extend(
                    [
                        self._make_discriminator_linear(
                            layer_input_dim,
                            self.hidden_dim,
                        ),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ]
                )
                layer_input_dim = self.hidden_dim
            layers.append(self._make_discriminator_linear(layer_input_dim, 2))
            self.discriminator = nn.Sequential(*layers)

    def _make_discriminator_linear(
        self,
        input_dim: int,
        output_dim: int,
    ) -> nn.Linear:
        layer = nn.Linear(input_dim, output_dim)
        if not self.discriminator_spectral_norm:
            return layer
        return nn.utils.parametrizations.spectral_norm(layer)

    @classmethod
    def from_config(cls, cfg, *, input_dim: int) -> "FactorVAELoss":
        return cls(
            enabled=bool(getattr(cfg, "factor_vae_enabled", False)),
            gamma=float(getattr(cfg, "factor_vae_gamma", 10.0)),
            input_dim=input_dim,
            hidden_dim=int(getattr(cfg, "factor_vae_discriminator_hidden_dim", 1000)),
            num_hidden_layers=int(
                getattr(cfg, "factor_vae_discriminator_num_hidden_layers", 6)
            ),
            discriminator_spectral_norm=bool(
                getattr(cfg, "factor_vae_discriminator_spectral_norm", False)
            ),
            discriminator_coordinate_group_size=int(
                getattr(
                    cfg,
                    "factor_vae_discriminator_coordinate_group_size",
                    0,
                )
            ),
            discriminator_bottleneck_dim=int(
                getattr(cfg, "factor_vae_discriminator_bottleneck_dim", 0)
            ),
            latent_noise_std=float(
                getattr(cfg, "factor_vae_latent_noise_std", 0.0)
            ),
            start_epoch=int(getattr(cfg, "factor_vae_start_epoch", 0)),
            discriminator_warmup_epochs=int(
                getattr(cfg, "factor_vae_discriminator_warmup_epochs", 0)
            ),
            gamma_warmup_epochs=int(
                getattr(cfg, "factor_vae_gamma_warmup_epochs", 0)
            ),
        )

    def should_run(self, *, current_epoch: int) -> bool:
        return self.enabled and int(current_epoch) >= self.start_epoch

    def effective_gamma(self, *, current_epoch: int) -> float:
        if not self.should_run(current_epoch=current_epoch):
            return 0.0
        penalty_start_epoch = self.start_epoch + self.discriminator_warmup_epochs
        if int(current_epoch) < penalty_start_epoch:
            return 0.0
        if self.gamma_warmup_epochs == 0:
            return self.gamma
        warmup_epoch = int(current_epoch) - penalty_start_epoch + 1
        fraction = min(1.0, warmup_epoch / float(self.gamma_warmup_epochs))
        return self.gamma * fraction

    def _require_discriminator(self) -> nn.Sequential:
        if self.discriminator is None:
            raise RuntimeError(
                "FactorVAE loss was requested while factor_vae_enabled=false; "
                "enable it before computing the total-correlation objective."
            )
        return self.discriminator

    def _validate_embeddings(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> None:
        if z_a.dim() != 2 or z_b.dim() != 2:
            raise ValueError(
                "FactorVAE expects two projected embedding matrices with shape (B, D); "
                f"got z_a={tuple(z_a.shape)}, z_b={tuple(z_b.shape)}."
            )
        if z_a.shape != z_b.shape:
            raise ValueError(
                "FactorVAE view embeddings must have identical shapes; "
                f"got z_a={tuple(z_a.shape)}, z_b={tuple(z_b.shape)}."
            )
        if z_a.shape[0] < 2:
            raise ValueError(
                "FactorVAE dimension permutation requires at least two samples per batch; "
                f"got batch size {z_a.shape[0]}."
            )
        if z_a.shape[1] != self.input_dim:
            raise ValueError(
                "FactorVAE projected embedding dimension does not match its discriminator; "
                f"got D={z_a.shape[1]}, expected {self.input_dim}."
            )

    @staticmethod
    def permute_dimensions(z: torch.Tensor) -> torch.Tensor:
        """Sample from the product of marginals by permuting each column."""
        batch_size, latent_dim = z.shape
        permutations = torch.stack(
            [torch.randperm(batch_size, device=z.device) for _ in range(latent_dim)],
            dim=1,
        )
        return z.gather(dim=0, index=permutations)

    def _sample_coordinate_groups(self, *, device: torch.device) -> torch.Tensor:
        if self.discriminator_coordinate_group_size == self.input_dim:
            return torch.arange(self.input_dim, device=device).unsqueeze(0)
        coordinate_order = torch.randperm(self.input_dim, device=device)
        return coordinate_order.reshape(
            self.input_dim // self.discriminator_coordinate_group_size,
            self.discriminator_coordinate_group_size,
        )

    @staticmethod
    def _select_coordinate_groups(
        z: torch.Tensor,
        coordinate_groups: torch.Tensor,
    ) -> torch.Tensor:
        grouped = z[:, coordinate_groups]
        return grouped.reshape(-1, coordinate_groups.shape[1])

    def _discriminator_logits(
        self,
        z: torch.Tensor,
        *,
        train_discriminator: bool,
    ) -> torch.Tensor:
        discriminator = self._require_discriminator()
        discriminator_dtype = next(discriminator.parameters()).dtype
        discriminator_input = z.to(dtype=discriminator_dtype)
        if train_discriminator:
            return discriminator(discriminator_input)

        detached_state = {
            name: value.detach()
            for name, value in discriminator.named_parameters()
        }
        detached_state.update(
            {
                # Spectral normalization updates its power-iteration buffers in
                # training mode. Clone them so the frozen encoder-side call cannot
                # mutate discriminator state.
                name: value.detach().clone()
                for name, value in discriminator.named_buffers()
            }
        )
        return functional_call(discriminator, detached_state, (discriminator_input,))

    def _add_latent_noise(self, z: torch.Tensor) -> torch.Tensor:
        if self.latent_noise_std == 0.0:
            return z
        return z + self.latent_noise_std * torch.randn_like(z)

    def total_correlation_loss(
        self,
        *,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> torch.Tensor:
        self._require_discriminator()
        self._validate_embeddings(z_a, z_b)

        coordinate_groups = self._sample_coordinate_groups(device=z_a.device)
        joint = self._select_coordinate_groups(
            self._add_latent_noise(torch.cat([z_a, z_b], dim=0)),
            coordinate_groups,
        )
        tc_logits = self._discriminator_logits(joint, train_discriminator=False)
        return (tc_logits[:, 0] - tc_logits[:, 1]).mean()

    def discriminator_loss(
        self,
        *,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._require_discriminator()
        self._validate_embeddings(z_a, z_b)

        split_index = z_a.shape[0] // 2
        if split_index < 2 or z_a.shape[0] - split_index < 2:
            raise ValueError(
                "FactorVAE discriminator training requires at least four samples so "
                "joint and factorized examples can come from independent batch halves; "
                f"got batch size {z_a.shape[0]}."
            )

        # Algorithm 2 uses a fresh sample from q(z) to construct the product of
        # marginals. Split by underlying point cloud so the discriminator never
        # sees a joint sample and its shuffled counterpart from the same example.
        joint = self._add_latent_noise(
            torch.cat(
                [z_a[:split_index].detach(), z_b[:split_index].detach()],
                dim=0,
            )
        )
        factorized_source = torch.cat(
            [z_a[split_index:].detach(), z_b[split_index:].detach()],
            dim=0,
        )
        factorized = self._add_latent_noise(
            self.permute_dimensions(factorized_source)
        )
        coordinate_groups = self._sample_coordinate_groups(device=z_a.device)
        joint_logits = self._discriminator_logits(
            self._select_coordinate_groups(joint, coordinate_groups),
            train_discriminator=True,
        )
        factorized_logits = self._discriminator_logits(
            self._select_coordinate_groups(factorized, coordinate_groups),
            train_discriminator=True,
        )
        joint_targets = torch.zeros(
            joint_logits.shape[0],
            dtype=torch.long,
            device=joint_logits.device,
        )
        factorized_targets = torch.ones(
            factorized_logits.shape[0],
            dtype=torch.long,
            device=factorized_logits.device,
        )
        discriminator_loss = 0.5 * (
            F.cross_entropy(joint_logits, joint_targets)
            + F.cross_entropy(factorized_logits, factorized_targets)
        )
        discriminator_accuracy = 0.5 * (
            (joint_logits.argmax(dim=1) == joint_targets).float().mean()
            + (factorized_logits.argmax(dim=1) == factorized_targets).float().mean()
        )
        joint_logit_gap = (joint_logits[:, 0] - joint_logits[:, 1]).mean()
        factorized_logit_gap = (
            factorized_logits[:, 0] - factorized_logits[:, 1]
        ).mean()
        metrics = {
            "factor_vae_discriminator": discriminator_loss,
            "factor_vae_discriminator_accuracy": discriminator_accuracy,
            "factor_vae_discriminator_joint_logit_gap": joint_logit_gap,
            "factor_vae_discriminator_factorized_logit_gap": factorized_logit_gap,
            "factor_vae_discriminator_logit_separation": (
                joint_logit_gap - factorized_logit_gap
            ),
        }
        return discriminator_loss, metrics

    def compute_loss(
        self,
        *,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        total_correlation = self.total_correlation_loss(z_a=z_a, z_b=z_b)
        discriminator_loss, discriminator_metrics = self.discriminator_loss(
            z_a=z_a,
            z_b=z_b,
        )
        metrics = {
            "factor_vae_tc_weighted": self.gamma * total_correlation,
            **discriminator_metrics,
        }
        return total_correlation, discriminator_loss, metrics


class VICRegModule(BaseSSLModule):
    """
    Self-supervised contrastive training with VICReg/VISReg/SwAV heads.
    """

    def __init__(self, cfg):
        self.data_kind = normalize_data_kind(cfg.data.kind)
        super().__init__(
            cfg,
            module_name="VICRegModule",
        )
        self.cache_warning_prefix = "contrastive"
        self.factor_vae = FactorVAELoss.from_config(
            cfg,
            input_dim=self.vicreg.embed_dim,
        )
        if self.factor_vae.enabled and not (
            self.vicreg.enabled and self.vicreg.weight > 0.0
        ):
            raise ValueError(
                "factor_vae_enabled=true requires an active VICReg objective "
                "(vicreg_enabled=true and vicreg_weight > 0)."
            )
        configured_devices = getattr(cfg, "devices", [0])
        if (
            self.factor_vae.enabled
            and len(configured_devices) > 1
            and not bool(getattr(cfg, "ddp_find_unused_parameters", False))
        ):
            raise ValueError(
                "Multi-GPU FactorVAE alternates model and discriminator backward passes, "
                "so it requires ddp_find_unused_parameters=true."
            )
        self.automatic_optimization = not self.factor_vae.enabled

        self.factor_vae_discriminator_learning_rate = float(
            getattr(cfg, "factor_vae_discriminator_learning_rate", 1.0e-4)
        )
        self.factor_vae_discriminator_update_interval = int(
            getattr(cfg, "factor_vae_discriminator_update_interval", 1)
        )
        discriminator_betas = tuple(
            float(value)
            for value in getattr(cfg, "factor_vae_discriminator_betas", (0.5, 0.9))
        )
        if self.factor_vae.enabled:
            if self.factor_vae_discriminator_learning_rate <= 0.0:
                raise ValueError(
                    "factor_vae_discriminator_learning_rate must be > 0, "
                    f"got {self.factor_vae_discriminator_learning_rate}."
                )
            if self.factor_vae_discriminator_update_interval < 1:
                raise ValueError(
                    "factor_vae_discriminator_update_interval must be >= 1, "
                    f"got {self.factor_vae_discriminator_update_interval}."
                )
            if len(discriminator_betas) != 2 or not all(
                0.0 <= beta < 1.0 for beta in discriminator_betas
            ):
                raise ValueError(
                    "factor_vae_discriminator_betas must contain two values in [0, 1), "
                    f"got {discriminator_betas}."
                )
        self.factor_vae_discriminator_betas = discriminator_betas
        default_gradient_clip = float(getattr(cfg, "gradient_clip_val", 0.0))
        self.factor_vae_model_gradient_clip_val = float(
            getattr(
                cfg,
                "factor_vae_model_gradient_clip_val",
                default_gradient_clip,
            )
        )
        self.factor_vae_discriminator_gradient_clip_val = float(
            getattr(
                cfg,
                "factor_vae_discriminator_gradient_clip_val",
                default_gradient_clip,
            )
        )
        if self.factor_vae.enabled:
            if self.factor_vae_model_gradient_clip_val < 0.0:
                raise ValueError(
                    "factor_vae_model_gradient_clip_val must be >= 0, "
                    f"got {self.factor_vae_model_gradient_clip_val}."
                )
            if self.factor_vae_discriminator_gradient_clip_val < 0.0:
                raise ValueError(
                    "factor_vae_discriminator_gradient_clip_val must be >= 0, "
                    f"got {self.factor_vae_discriminator_gradient_clip_val}."
                )
        self._factor_vae_discriminator_embeddings: (
            tuple[torch.Tensor, torch.Tensor] | None
        ) = None
        if self.factor_vae.enabled:
            self.register_buffer(
                "factor_vae_model_update_count",
                torch.zeros((), dtype=torch.long),
                persistent=True,
            )

    def configure_optimizers(self):
        if not self.factor_vae.enabled:
            return super().configure_optimizers()

        discriminator = self.factor_vae._require_discriminator()
        discriminator_parameters = list(discriminator.parameters())
        discriminator_parameter_ids = {
            id(parameter) for parameter in discriminator_parameters
        }
        model_parameters = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in discriminator_parameter_ids
        ]
        model_optimizers, model_schedulers = get_optimizers_and_scheduler(
            self.hparams,
            model_parameters,
        )
        discriminator_optimizer = torch.optim.Adam(
            discriminator_parameters,
            lr=self.factor_vae_discriminator_learning_rate,
            betas=self.factor_vae_discriminator_betas,
        )
        return [model_optimizers[0], discriminator_optimizer], model_schedulers

    @staticmethod
    def _optimizer_gradient_norm(optimizer) -> torch.Tensor:
        gradient_norms = [
            parameter.grad.detach().float().norm(2)
            for parameter_group in optimizer.optimizer.param_groups
            for parameter in parameter_group["params"]
            if parameter.grad is not None
        ]
        if not gradient_norms:
            raise RuntimeError(
                "Cannot measure FactorVAE optimizer gradient norm because no "
                "parameter gradients were produced."
            )
        return torch.stack(gradient_norms).norm(2)

    def _weighted_total_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = super()._weighted_total_loss(losses)
        if "factor_vae_tc" in losses:
            gamma = self.factor_vae.effective_gamma(
                current_epoch=int(self.current_epoch)
            )
            total_loss = total_loss + gamma * losses["factor_vae_tc"]
        return total_loss

    def _unpack_batch(self, batch):
        if self.data_kind == "static":
            return batch["points"], {}
        if self.data_kind == "synthetic":
            return batch["points"], {
                "class_id": batch["class_id"],
                "instance_id": batch["instance_id"],
                "rotation": batch["rotation"],
            }
        raise RuntimeError(
            "VICRegModule only consumes the repository's static or synthetic batches, "
            f"got data.kind={self.data_kind!r}."
        )

    def _build_contrastive_view_pair(
        self,
        pc: torch.Tensor,
        *,
        view_points: int | None,
    ) -> dict[str, torch.Tensor]:
        use_neighbor_a, use_neighbor_b = self.vicreg._resolve_neighbor_flags(device=pc.device)
        apply_occlusion_a, apply_occlusion_b = self.vicreg._resolve_pair_occlusion_flags(
            use_neighbor_a=use_neighbor_a,
            use_neighbor_b=use_neighbor_b,
            device=pc.device,
        )
        shared_pc = pc
        augment_view_points = view_points
        if view_points is not None and not use_neighbor_a and not use_neighbor_b:
            shared_pc = crop_to_num_points(pc, int(view_points))
            augment_view_points = None
        return {
            "y_a": self.vicreg._augment(
                shared_pc,
                use_neighbor=use_neighbor_a,
                apply_occlusion=apply_occlusion_a,
                view_points=augment_view_points,
            ),
            "y_b": self.vicreg._augment(
                shared_pc,
                use_neighbor=use_neighbor_b,
                apply_occlusion=apply_occlusion_b,
                view_points=augment_view_points,
            ),
        }

    def _encode_contrastive_view_pair(
        self,
        views: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_a = views["y_a"]
        y_b = views["y_b"]
        batch_size = int(y_a.shape[0])

        fused_input = torch.cat([y_a, y_b], dim=0)
        encoded = self.encoder_io.encode(fused_input)
        features = self._shared_invariant(encoded.invariant, encoded.equivariant)

        return features.chunk(2, dim=0)

    def forward(self, pc: torch.Tensor, include_ssl_heads: bool = False):
        pc = self._prepare_model_input(pc).to(device=self.device, dtype=self.dtype)
        encoded = self.encoder_io.encode(pc)
        z_inv_contrastive = self._contrastive_invariant_latent(
            encoded.invariant,
            encoded.equivariant,
        )
        output_representation = self._output_representation(z_inv_contrastive)
        if include_ssl_heads:
            return (
                output_representation,
                encoded.invariant,
                encoded.equivariant,
                self._forward_ssl_heads_for_summary(z_inv_contrastive),
            )
        # Forward returns both invariant branches explicitly:
        # (z_inv_contrastive, z_inv_model, eq_z).
        return output_representation, encoded.invariant, encoded.equivariant

    def _step(
        self,
        batch,
        batch_idx,
        stage: str,
        *,
        compute_factor_vae_discriminator: bool = True,
    ):
        pc_raw, meta = self._unpack_batch(batch)
        batch_size = int(pc_raw.shape[0])
        pc_raw = pc_raw.to(device=self.device, dtype=self.dtype, non_blocking=True)
        losses = {}

        run_vicreg = self.vicreg.should_run(current_epoch=int(self.current_epoch))
        run_swav = self.swav.should_run(current_epoch=int(self.current_epoch))
        run_factor_vae = self.factor_vae.should_run(
            current_epoch=int(self.current_epoch)
        )
        can_share_views = run_vicreg and run_swav and self.vicreg.view_points == self.swav.view_points

        if can_share_views:
            shared_view_pair = self._build_contrastive_view_pair(pc_raw, view_points=self.vicreg.view_points)
            shared_features = self._encode_contrastive_view_pair(shared_view_pair)

        if run_vicreg:
            if can_share_views:
                vicreg_views = shared_view_pair
                z_a, z_b = shared_features
            else:
                vicreg_views = self._build_contrastive_view_pair(
                    pc_raw,
                    view_points=self.vicreg.view_points,
                )
                z_a, z_b = self._encode_contrastive_view_pair(vicreg_views)
            if run_factor_vae:
                projected_z_a = self.vicreg.project_features(z_a)
                projected_z_b = self.vicreg.project_features(z_b)
                vicreg_loss, vicreg_metrics = self.vicreg.compute_loss_from_projected_embeddings(
                    z_a=projected_z_a,
                    z_b=projected_z_b,
                    current_epoch=int(self.current_epoch),
                )
                factor_vae_tc = self.factor_vae.total_correlation_loss(
                    z_a=projected_z_a,
                    z_b=projected_z_b,
                )
                losses["factor_vae_tc"] = factor_vae_tc
                effective_gamma = self.factor_vae.effective_gamma(
                    current_epoch=int(self.current_epoch)
                )
                factor_vae_metrics = {
                    "factor_vae_gamma": factor_vae_tc.new_tensor(effective_gamma),
                    "factor_vae_tc_weighted": effective_gamma * factor_vae_tc,
                }
                if stage == "train":
                    embedding_gradients = torch.autograd.grad(
                        effective_gamma * factor_vae_tc,
                        (projected_z_a, projected_z_b),
                        retain_graph=True,
                    )
                    factor_vae_metrics["factor_vae_tc_embedding_gradient_norm"] = (
                        torch.stack(
                            [
                                gradient.detach().float().norm(2)
                                for gradient in embedding_gradients
                            ]
                        ).norm(2)
                    )
                if compute_factor_vae_discriminator:
                    _, discriminator_metrics = self.factor_vae.discriminator_loss(
                        z_a=projected_z_a,
                        z_b=projected_z_b,
                    )
                    factor_vae_metrics.update(discriminator_metrics)
                else:
                    self._factor_vae_discriminator_embeddings = (
                        projected_z_a.detach(),
                        projected_z_b.detach(),
                    )
                for name, value in factor_vae_metrics.items():
                    self._log_metric(stage, name, value, batch_size=batch_size)
            else:
                vicreg_loss, vicreg_metrics = self.vicreg.compute_loss_from_features(
                    z_a_feat=z_a,
                    z_b_feat=z_b,
                    current_epoch=int(self.current_epoch),
                )
            if vicreg_loss is not None:
                losses[self.vicreg.metric_prefix] = vicreg_loss
            for name, value in vicreg_metrics.items():
                self._log_metric(stage, name, value, batch_size=batch_size)

        if run_swav:
            if can_share_views:
                z_a, z_b = shared_features
            else:
                swav_views = self._build_contrastive_view_pair(
                    pc_raw,
                    view_points=self.swav.view_points,
                )
                z_a, z_b = self._encode_contrastive_view_pair(swav_views)
            swav_loss, swav_metrics = self.swav.compute_loss(
                view_features=[z_a, z_b],
                current_epoch=int(self.current_epoch),
            )
            if swav_loss is not None:
                losses["swav"] = swav_loss
            for name, value in swav_metrics.items():
                self._log_metric(stage, name, value, batch_size=batch_size)

        if self._should_cache_supervised_stage(stage):
            with torch.no_grad():
                pc = self._prepare_model_input(pc_raw)
                encoded = self.encoder_io.encode(pc)
                encoder_features = self._contrastive_invariant_from_eq_latent(
                    encoded.equivariant,
                    z_inv_model=encoded.invariant,
                    stage=stage,
                )
                output_representation = self._output_representation(encoder_features)
            self._cache_supervised_embeddings_if_needed(
                stage=stage,
                meta=meta,
                embeddings=output_representation,
                encoder_features=encoder_features,
            )

        return self._finish_ssl_step(
            stage=stage,
            batch_idx=batch_idx,
            batch_size=batch_size,
            losses=losses,
        )

    def training_step(self, batch, batch_idx, dataloader_idx: int = 0):
        if not self.factor_vae.enabled:
            return super().training_step(batch, batch_idx, dataloader_idx)

        model_optimizer, discriminator_optimizer = self.optimizers()
        self._factor_vae_discriminator_embeddings = None

        with model_optimizer.toggle_model():
            model_optimizer.zero_grad()
            model_loss = self._step(
                batch,
                batch_idx,
                "train",
                compute_factor_vae_discriminator=False,
            )
            self.manual_backward(model_loss)
            discriminator_embeddings = self._factor_vae_discriminator_embeddings
            if discriminator_embeddings is not None:
                model_gradient_norm = self._optimizer_gradient_norm(model_optimizer)
                self._log_metric(
                    "train",
                    "factor_vae_model_total_gradient_norm",
                    model_gradient_norm,
                    batch_size=int(discriminator_embeddings[0].shape[0]),
                )
            if self.factor_vae_model_gradient_clip_val > 0.0:
                self.clip_gradients(
                    model_optimizer,
                    gradient_clip_val=self.factor_vae_model_gradient_clip_val,
                    gradient_clip_algorithm="norm",
                )
            model_optimizer.step()

        self.factor_vae_model_update_count.add_(1)
        self._factor_vae_discriminator_embeddings = None
        update_discriminator = (
            discriminator_embeddings is not None
            and int(self.factor_vae_model_update_count.item())
            % self.factor_vae_discriminator_update_interval
            == 0
        )
        if update_discriminator:
            z_a, z_b = discriminator_embeddings
            with discriminator_optimizer.toggle_model():
                discriminator_optimizer.zero_grad()
                discriminator_loss, discriminator_metrics = (
                    self.factor_vae.discriminator_loss(z_a=z_a, z_b=z_b)
                )
                self.manual_backward(discriminator_loss)
                discriminator_gradient_norm = self._optimizer_gradient_norm(
                    discriminator_optimizer
                )
                discriminator_metrics["factor_vae_discriminator_gradient_norm"] = (
                    discriminator_gradient_norm
                )
                if self.factor_vae_discriminator_gradient_clip_val > 0.0:
                    self.clip_gradients(
                        discriminator_optimizer,
                        gradient_clip_val=self.factor_vae_discriminator_gradient_clip_val,
                        gradient_clip_algorithm="norm",
                    )
                discriminator_optimizer.step()
            batch_size = int(z_a.shape[0])
            for name, value in discriminator_metrics.items():
                self._log_metric("train", name, value, batch_size=batch_size)

        return model_loss.detach()

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if self.factor_vae.enabled:
            scheduler = self.lr_schedulers()
            if isinstance(scheduler, list):
                if len(scheduler) != 1:
                    raise RuntimeError(
                        "FactorVAE manual optimization expects exactly one model scheduler, "
                        f"got {len(scheduler)}."
                    )
                scheduler = scheduler[0]
            scheduler.step()

__all__ = ["FactorVAELoss", "VICRegModule"]
