from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders.ri_mae_encoder import RITransformer
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
from src.utils.training_utils import get_optimizers_and_scheduler


class VICRegMaskedLatentModule(VICRegModule):
    """
    VICReg global training with masked latent-token prediction.

    The online encoder receives visible patch tokens only. The target encoder
    receives the same full patch geometry and provides stop-gradient target
    token latents. Token alignment is kept explicit by sharing RI-MAE patch
    geometry between online and target encoders inside each batch.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cache_warning_prefix = "vicreg_masked_latent"

        encoder_kwargs = getattr(getattr(cfg, "encoder", None), "kwargs", None)
        self.masked_latent_enabled = bool(getattr(cfg, "masked_latent_enabled", True))
        self.masked_latent_weight = float(
            getattr(cfg, "masked_latent_weight", getattr(cfg, "lambda_mask", 1.0))
        )
        self.local_affinity_weight = float(
            getattr(cfg, "local_affinity_weight", getattr(cfg, "lambda_local", 0.0))
        )
        self.masked_latent_start_epoch = int(
            getattr(cfg, "masked_latent_start_epoch", self.vicreg.start_epoch)
        )
        self.masked_latent_mask_ratio = float(
            getattr(
                cfg,
                "masked_latent_mask_ratio",
                getattr(encoder_kwargs, "mask_ratio", 0.5),
            )
        )
        self.masked_latent_target_mode = str(
            getattr(cfg, "masked_latent_target_mode", "ema")
        ).strip().lower()
        self.masked_latent_ema_decay = float(
            getattr(
                cfg,
                "masked_latent_ema_decay",
                getattr(encoder_kwargs, "ema_decay", 0.996),
            )
        )
        self.masked_latent_predictor_depth = int(
            getattr(
                cfg,
                "masked_latent_predictor_depth",
                getattr(encoder_kwargs, "predictor_depth", 2),
            )
        )
        self.masked_latent_predictor_mlp_ratio = float(
            getattr(
                cfg,
                "masked_latent_predictor_mlp_ratio",
                getattr(encoder_kwargs, "mlp_ratio", 4.0),
            )
        )
        self.masked_latent_predictor_dropout = float(
            getattr(
                cfg,
                "masked_latent_predictor_dropout",
                getattr(encoder_kwargs, "dropout", 0.0),
            )
        )
        self.masked_latent_predictor_num_heads = getattr(
            cfg,
            "masked_latent_predictor_num_heads",
            getattr(encoder_kwargs, "num_heads", None),
        )
        self.local_affinity_k = int(getattr(cfg, "local_affinity_k", 6))
        self.local_affinity_temperature = float(
            getattr(cfg, "local_affinity_temperature", 1.0)
        )

        self._masked_latent_initialized = False
        self.target_encoder = None
        self.mask_token = None
        self.masked_latent_predictor = None
        self.masked_latent_pred_head = None

        if self._masked_latent_losses_configured():
            self._init_masked_latent_components(cfg)

    def _masked_latent_losses_configured(self) -> bool:
        return bool(
            self.masked_latent_enabled
            and (self.masked_latent_weight > 0.0 or self.local_affinity_weight > 0.0)
        )

    def _raw_online_encoder(self) -> nn.Module:
        return getattr(self.encoder, "_orig_mod", self.encoder)

    @staticmethod
    def _require_token_encoder(encoder: nn.Module, *, context: str) -> nn.Module:
        required = (
            "prepare_token_geometry",
            "forward_visible_tokens_from_geometry",
            "forward_tokens_from_geometry",
            "token_dim",
            "token_num_heads",
        )
        missing = [name for name in required if not hasattr(encoder, name)]
        if missing:
            raise TypeError(
                f"{context} requires a token-capable encoder exposing {required}, "
                f"but {encoder.__class__.__name__} is missing {missing}. "
                "Use RI_MAE_Invariant or add the same token-latent contract to the encoder."
            )
        return encoder

    def _init_masked_latent_components(self, cfg) -> None:
        if not (0.0 < self.masked_latent_mask_ratio < 1.0):
            raise ValueError(
                "masked_latent_mask_ratio must be in (0, 1) so the online encoder "
                "gets context tokens and at least one target token. "
                f"Got {self.masked_latent_mask_ratio}."
            )
        if self.masked_latent_target_mode not in {"ema", "stop_gradient"}:
            raise ValueError(
                "masked_latent_target_mode must be one of {'ema', 'stop_gradient'}, "
                f"got {self.masked_latent_target_mode!r}."
            )
        if not (0.0 <= self.masked_latent_ema_decay < 1.0):
            raise ValueError(
                "masked_latent_ema_decay must be in [0, 1) for EMA updates, "
                f"got {self.masked_latent_ema_decay}."
            )
        if self.masked_latent_predictor_depth <= 0:
            raise ValueError(
                "masked_latent_predictor_depth must be positive, "
                f"got {self.masked_latent_predictor_depth}."
            )
        if self.local_affinity_weight > 0.0 and self.local_affinity_k <= 0:
            raise ValueError(
                "local_affinity_k must be positive when local_affinity_weight > 0, "
                f"got {self.local_affinity_k}."
            )
        if self.local_affinity_temperature <= 0.0:
            raise ValueError(
                "local_affinity_temperature must be positive, "
                f"got {self.local_affinity_temperature}."
            )

        online_encoder = self._require_token_encoder(
            self._raw_online_encoder(),
            context="VICReg + masked latent prediction",
        )
        token_dim = int(getattr(online_encoder, "token_dim"))
        num_heads = self.masked_latent_predictor_num_heads
        if num_heads is None:
            num_heads = int(getattr(online_encoder, "token_num_heads"))
        num_heads = int(num_heads)
        if token_dim <= 0:
            raise ValueError(f"Token dimension must be positive, got {token_dim}.")
        if num_heads <= 0 or token_dim % num_heads != 0:
            raise ValueError(
                "masked_latent_predictor_num_heads must be positive and divide token_dim. "
                f"Got token_dim={token_dim}, num_heads={num_heads}."
            )

        self.target_encoder = copy.deepcopy(online_encoder)
        self._freeze_target_encoder()
        self._sync_target_encoder_hard()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.masked_latent_predictor = RITransformer(
            embed_dim=token_dim,
            num_heads=num_heads,
            depth=self.masked_latent_predictor_depth,
            mlp_ratio=self.masked_latent_predictor_mlp_ratio,
            dropout=self.masked_latent_predictor_dropout,
        )
        self.masked_latent_pred_head = nn.Linear(token_dim, token_dim)
        self._masked_latent_initialized = True

    def _freeze_target_encoder(self) -> None:
        if self.target_encoder is None:
            raise RuntimeError("Cannot freeze target encoder before it is initialized.")
        self.target_encoder.eval()
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def _sync_target_encoder_hard(self) -> None:
        if self.target_encoder is None:
            raise RuntimeError("Cannot sync target encoder before it is initialized.")
        online_encoder = self._require_token_encoder(
            self._raw_online_encoder(),
            context="hard target encoder sync",
        )
        missing, unexpected = self.target_encoder.load_state_dict(
            online_encoder.state_dict(),
            strict=False,
        )
        if missing or unexpected:
            raise RuntimeError(
                "Target encoder hard sync found incompatible state dictionaries. "
                f"missing={missing}, unexpected={unexpected}."
            )
        self._freeze_target_encoder()

    @torch.no_grad()
    def _update_target_encoder_ema(self) -> None:
        if self.target_encoder is None:
            raise RuntimeError("Cannot EMA-update target encoder before it is initialized.")
        online_encoder = self._require_token_encoder(
            self._raw_online_encoder(),
            context="EMA target encoder update",
        )
        online_params = dict(online_encoder.named_parameters())
        target_params = dict(self.target_encoder.named_parameters())
        if online_params.keys() != target_params.keys():
            raise RuntimeError(
                "Online/target encoder parameter names differ during EMA update. "
                f"online_only={sorted(set(online_params) - set(target_params))}, "
                f"target_only={sorted(set(target_params) - set(online_params))}."
            )
        decay = float(self.masked_latent_ema_decay)
        for name, target_param in target_params.items():
            online_param = online_params[name].detach().to(
                device=target_param.device,
                dtype=target_param.dtype,
            )
            target_param.mul_(decay).add_(online_param, alpha=1.0 - decay)

        online_buffers = dict(online_encoder.named_buffers())
        target_buffers = dict(self.target_encoder.named_buffers())
        if online_buffers.keys() != target_buffers.keys():
            raise RuntimeError(
                "Online/target encoder buffer names differ during EMA update. "
                f"online_only={sorted(set(online_buffers) - set(target_buffers))}, "
                f"target_only={sorted(set(target_buffers) - set(online_buffers))}."
            )
        for name, target_buffer in target_buffers.items():
            online_buffer = online_buffers[name].detach().to(
                device=target_buffer.device,
                dtype=target_buffer.dtype,
            )
            target_buffer.copy_(online_buffer)
        self._freeze_target_encoder()

    def _masked_latent_should_run(self, *, current_epoch: int) -> bool:
        return bool(
            self._masked_latent_initialized
            and self._masked_latent_losses_configured()
            and int(current_epoch) >= self.masked_latent_start_epoch
        )

    def _sample_token_indices(
        self,
        *,
        batch_size: int,
        token_count: int,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_count < 2:
            raise ValueError(
                "Masked latent prediction requires at least two tokens so one can be "
                f"visible while another is masked, got token_count={token_count}."
            )
        mask_count = int(round(float(token_count) * self.masked_latent_mask_ratio))
        mask_count = min(max(1, mask_count), token_count - 1)
        permutation = torch.rand(batch_size, token_count, device=device).argsort(dim=1)
        mask_idx = permutation[:, :mask_count].sort(dim=1).values
        visible_idx = permutation[:, mask_count:].sort(dim=1).values
        token_mask = torch.zeros(batch_size, token_count, dtype=torch.bool, device=device)
        token_mask.scatter_(dim=1, index=mask_idx, value=True)
        return visible_idx, mask_idx, token_mask

    def _build_predictor_input(
        self,
        *,
        context_tokens: torch.Tensor,
        visible_idx: torch.Tensor,
        pos_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.mask_token is None:
            raise RuntimeError("mask_token is not initialized.")
        batch_size, token_count, token_dim = pos_tokens.shape
        if context_tokens.shape[0] != batch_size or context_tokens.shape[-1] != token_dim:
            raise ValueError(
                "Context token shape does not match predictor positional tokens. "
                f"context_tokens={tuple(context_tokens.shape)}, pos_tokens={tuple(pos_tokens.shape)}."
            )
        mask_query = self.mask_token.to(
            device=pos_tokens.device,
            dtype=pos_tokens.dtype,
        ).expand(batch_size, token_count, token_dim)
        full_tokens = mask_query + pos_tokens
        scatter_idx = visible_idx.unsqueeze(-1).expand(-1, -1, token_dim)
        return full_tokens.scatter(dim=1, index=scatter_idx, src=context_tokens)

    @staticmethod
    def _gather_masked_tokens(tokens: torch.Tensor, mask_idx: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"Expected tokens with shape (B, T, C), got {tuple(tokens.shape)}.")
        gather_idx = mask_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        return tokens.gather(dim=1, index=gather_idx)

    def _compute_local_affinity_loss(
        self,
        *,
        pred_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.local_affinity_weight <= 0.0:
            return pred_tokens.new_tensor(0.0)
        if pred_tokens.shape != target_tokens.shape:
            raise ValueError(
                "Local affinity requires predicted and target token tensors with identical shapes, "
                f"got pred={tuple(pred_tokens.shape)}, target={tuple(target_tokens.shape)}."
            )
        if token_positions.dim() != 3 or token_positions.shape[:2] != pred_tokens.shape[:2]:
            raise ValueError(
                "Token positions must have shape (B, T, 3) aligned with token tensors. "
                f"token_positions={tuple(token_positions.shape)}, pred_tokens={tuple(pred_tokens.shape)}."
            )

        pred = F.normalize(pred_tokens.float(), dim=-1, eps=1e-6)
        target = F.normalize(target_tokens.detach().float(), dim=-1, eps=1e-6)
        pred_affinity = torch.matmul(pred, pred.transpose(1, 2)) / self.local_affinity_temperature
        target_affinity = torch.matmul(target, target.transpose(1, 2)) / self.local_affinity_temperature

        batch_size, token_count, _ = pred_affinity.shape
        if token_count < 2:
            raise ValueError(
                "Local affinity requires at least two tokens, "
                f"got token_count={token_count}."
            )
        eye = torch.eye(token_count, dtype=torch.bool, device=pred_tokens.device)
        if self.local_affinity_k >= token_count:
            edge_mask = ~eye.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            dist = torch.cdist(token_positions.float(), token_positions.float())
            dist = dist.masked_fill(eye.unsqueeze(0), float("inf"))
            nn_idx = dist.topk(k=self.local_affinity_k, dim=-1, largest=False).indices
            edge_mask = torch.zeros_like(dist, dtype=torch.bool)
            edge_mask.scatter_(dim=-1, index=nn_idx, value=True)
            edge_mask = edge_mask | edge_mask.transpose(1, 2)

        if not bool(edge_mask.any().item()):
            raise RuntimeError(
                "Local affinity edge mask is empty. "
                f"token_count={token_count}, local_affinity_k={self.local_affinity_k}."
            )
        return (pred_affinity - target_affinity).pow(2)[edge_mask].mean()

    def _compute_masked_latent_losses(
        self,
        *,
        pc_full: torch.Tensor,
        current_epoch: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self._masked_latent_should_run(current_epoch=current_epoch):
            return {}, {}
        if self.masked_latent_predictor is None or self.masked_latent_pred_head is None:
            raise RuntimeError("Masked latent predictor modules are not initialized.")
        if self.target_encoder is None:
            raise RuntimeError("Target encoder is not initialized.")
        if self.masked_latent_target_mode == "stop_gradient":
            self._sync_target_encoder_hard()

        online_encoder = self._require_token_encoder(
            self._raw_online_encoder(),
            context="masked latent online forward",
        )
        geometry = online_encoder.prepare_token_geometry(pc_full)
        token_count = int(geometry["token_positions"].shape[1])
        visible_idx, mask_idx, token_mask = self._sample_token_indices(
            batch_size=int(pc_full.shape[0]),
            token_count=token_count,
            device=pc_full.device,
        )

        online_out = online_encoder.forward_visible_tokens_from_geometry(
            geometry,
            visible_idx=visible_idx,
        )
        with torch.no_grad():
            self.target_encoder.eval()
            target_out = self.target_encoder.forward_tokens_from_geometry(geometry)
            target_tokens = target_out["tokens"].detach()

        predictor_input = self._build_predictor_input(
            context_tokens=online_out["context_tokens"],
            visible_idx=visible_idx,
            pos_tokens=online_out["pos_tokens"],
        )
        pred_tokens = self.masked_latent_predictor(
            predictor_input,
            online_out["attn_bias"].to(dtype=predictor_input.dtype),
        )
        pred_tokens = self.masked_latent_pred_head(pred_tokens)

        pred_masked = self._gather_masked_tokens(pred_tokens, mask_idx).float()
        target_masked = self._gather_masked_tokens(target_tokens, mask_idx).float()
        masked_loss = F.mse_loss(pred_masked, target_masked)
        local_loss = self._compute_local_affinity_loss(
            pred_tokens=pred_tokens,
            target_tokens=target_tokens,
            token_positions=online_out["token_positions"],
        )

        losses = {}
        if self.masked_latent_weight > 0.0:
            losses["masked_latent"] = masked_loss
        if self.local_affinity_weight > 0.0:
            losses["local_affinity"] = local_loss

        metrics = {
            "masked_latent_raw": masked_loss,
            "masked_latent_mask_ratio": pred_tokens.new_tensor(self.masked_latent_mask_ratio),
            "masked_latent_masked_tokens": pred_tokens.new_tensor(mask_idx.shape[1]),
            "masked_latent_visible_tokens": pred_tokens.new_tensor(visible_idx.shape[1]),
        }
        if self.local_affinity_weight > 0.0:
            metrics["local_affinity_raw"] = local_loss
        metrics["masked_latent_token_mask_fraction"] = token_mask.float().mean()
        return losses, metrics

    def _weighted_total_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = None
        contrastive_key = getattr(self.vicreg, "metric_prefix", "vicreg")
        weighted_terms = (
            (contrastive_key, self.vicreg.weight),
            ("swav", self.swav.weight),
            ("masked_latent", self.masked_latent_weight),
            ("local_affinity", self.local_affinity_weight),
        )
        for key, weight in weighted_terms:
            if key not in losses:
                continue
            term = float(weight) * losses[key]
            total_loss = term if total_loss is None else total_loss + term
        if total_loss is None:
            total_loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
        return total_loss

    def _step(self, batch, batch_idx, stage: str):
        pc_raw, meta = self._unpack_batch(batch)
        batch_size = int(pc_raw.shape[0])
        pc_raw = pc_raw.to(device=self.device, dtype=self.dtype, non_blocking=True)
        pc = self._prepare_model_input(pc_raw)
        losses = {}

        run_vicreg = self.vicreg.should_run(current_epoch=int(self.current_epoch))
        run_swav = self.swav.should_run(current_epoch=int(self.current_epoch))
        can_share_views = run_vicreg and run_swav and self.vicreg.view_points == self.swav.view_points

        def process_head(head_name, compute_fn, view_points, shared_features=None, view_pair=None):
            views = view_pair
            if shared_features:
                z_a, z_b = shared_features
            else:
                views = self._build_contrastive_view_pair(pc_raw, view_points=view_points)
                z_a, z_b = self._encode_contrastive_view_pair(views)
            loss, metrics = compute_fn(z_a, z_b, views)
            if loss is not None:
                losses[head_name] = loss
            for k, v in metrics.items():
                self._log_metric(stage, k, v, batch_size=batch_size)
            return z_a, z_b

        shared_features = None
        shared_view_pair = None
        if can_share_views:
            shared_view_pair = self._build_contrastive_view_pair(pc_raw, view_points=self.vicreg.view_points)
            shared_features = self._encode_contrastive_view_pair(shared_view_pair)

        if run_vicreg:
            process_head(
                self.vicreg.metric_prefix,
                lambda a, b, views: self.vicreg.compute_loss_from_features(
                    z_a_feat=a,
                    z_b_feat=b,
                    current_epoch=int(self.current_epoch),
                    overlap_target=None if views is None else views.get("overlap_target"),
                ),
                self.vicreg.view_points,
                shared_features,
                shared_view_pair,
            )

        if run_swav:
            process_head(
                "swav",
                lambda a, b, views: self.swav.compute_loss(
                    view_features=[a, b],
                    current_epoch=int(self.current_epoch),
                ),
                self.swav.view_points,
                shared_features,
                shared_view_pair,
            )

        masked_losses, masked_metrics = self._compute_masked_latent_losses(
            pc_full=pc,
            current_epoch=int(self.current_epoch),
        )
        losses.update(masked_losses)
        for k, v in masked_metrics.items():
            self._log_metric(stage, k, v, batch_size=batch_size)

        if self._should_cache_supervised_stage(stage):
            with torch.no_grad():
                encoded = self.encoder_io.encode(pc)
                z_inv = self._contrastive_invariant_from_eq_latent(
                    encoded.equivariant,
                    z_inv_model=encoded.invariant,
                    stage=stage,
                )
            self._cache_supervised_embeddings_if_needed(stage=stage, meta=meta, embeddings=z_inv)

        return self._finish_ssl_step(
            stage=stage,
            batch_idx=batch_idx,
            batch_size=batch_size,
            losses=losses,
        )

    def configure_optimizers(self):
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for VICRegMaskedLatentModule.")
        return get_optimizers_and_scheduler(self.hparams, trainable_params)

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        if (
            self._masked_latent_should_run(current_epoch=int(self.current_epoch))
            and self.masked_latent_target_mode == "ema"
        ):
            self._update_target_encoder_ema()


__all__ = ["VICRegMaskedLatentModule"]
