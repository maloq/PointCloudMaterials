import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_utils.data_modules.line_jepa import LineJEPADataModule
from src.models import EncoderAdapter
from src.training_methods.base_ssl_module import BaseSSLModule
from src.training_methods.contrastive_learning.vicreg import EvalBatchStatsBatchNorm1d
from src.training_methods.line_jepa.line_jepa import LineJEPALoss


class LineContextPredictor(nn.Module):
    """Transformer predictor for the masked central atom on a sampled line."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        depth = int(depth)
        num_heads = int(num_heads)
        if input_dim <= 0:
            raise ValueError(f"LineContextPredictor input_dim must be > 0, got {input_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"line_jepa_predictor_hidden_dim must be > 0, got {hidden_dim}.")
        if depth <= 0:
            raise ValueError(f"line_jepa_predictor_depth must be > 0, got {depth}.")
        if num_heads <= 0:
            raise ValueError(f"line_jepa_predictor_heads must be > 0, got {num_heads}.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "line_jepa_predictor_hidden_dim must be divisible by line_jepa_predictor_heads. "
                f"Got hidden_dim={hidden_dim}, num_heads={num_heads}."
            )

        self.input_proj = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        self.position_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=max(hidden_dim, int(hidden_dim * float(mlp_ratio))),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.target_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        context_embeddings: torch.Tensor,
        context_positions: torch.Tensor,
        target_position: torch.Tensor,
    ) -> torch.Tensor:
        if context_embeddings.dim() != 3:
            raise ValueError(
                "LineContextPredictor expects context_embeddings with shape (B, C, D), "
                f"got {tuple(context_embeddings.shape)}."
            )
        if context_positions.shape[:2] != context_embeddings.shape[:2] or context_positions.shape[-1] != 2:
            raise ValueError(
                "LineContextPredictor context_positions must have shape (B, C, 2) aligned with "
                f"context_embeddings. Got context_embeddings={tuple(context_embeddings.shape)}, "
                f"context_positions={tuple(context_positions.shape)}."
            )
        if target_position.shape != (context_embeddings.shape[0], 2):
            raise ValueError(
                "LineContextPredictor target_position must have shape (B, 2). "
                f"Got target_position={tuple(target_position.shape)}, batch={context_embeddings.shape[0]}."
            )

        dtype = context_embeddings.dtype
        tokens = self.input_proj(context_embeddings)
        tokens = tokens + self.position_mlp(context_positions.to(device=tokens.device, dtype=dtype))
        encoded = self.context_encoder(tokens)

        query = self.target_query.expand(context_embeddings.shape[0], -1, -1)
        query = query + self.position_mlp(target_position.to(device=tokens.device, dtype=dtype)).unsqueeze(1)
        attended, _ = self.cross_attention(query=query, key=encoded, value=encoded, need_weights=False)
        return self.output(self.norm(attended.squeeze(1)))


class LineJEPAModule(BaseSSLModule):
    """JEPA-style training from line context to the central local-structure embedding."""

    data_module_class = LineJEPADataModule
    test_metrics_on_step = True

    def __init__(self, cfg):
        data_cfg = getattr(cfg, "data", None)
        self.line_atoms = int(getattr(data_cfg, "line_atoms", 0))
        if self.line_atoms <= 1 or self.line_atoms % 2 != 1:
            raise ValueError(
                "LineJEPAModule requires data.line_atoms to be an odd integer > 1. "
                f"Got data.line_atoms={self.line_atoms}."
            )
        self.target_line_index = self.line_atoms // 2
        super().__init__(
            cfg,
            module_name="LineJEPAModule",
            summary_sequence_length=self.line_atoms,
        )
        self.cache_warning_prefix = "line-jepa"
        self.line_jepa = LineJEPALoss.from_config(cfg)
        self.regularize_online_target = bool(
            getattr(cfg, "line_jepa_regularize_online_target", False)
        )
        self.prediction_target = str(getattr(cfg, "line_jepa_prediction_target", "target")).strip().lower()
        if self.prediction_target not in {"target", "residual"}:
            raise ValueError(
                "line_jepa_prediction_target must be 'target' or 'residual', "
                f"got {self.prediction_target!r}."
            )
        self.prediction_positions = str(
            getattr(cfg, "line_jepa_prediction_positions", "center")
        ).strip().lower()
        if self.prediction_positions not in {"center", "all"}:
            raise ValueError(
                "line_jepa_prediction_positions must be 'center' or 'all', "
                f"got {self.prediction_positions!r}."
            )
        self.view_jitter_std = float(getattr(cfg, "line_jepa_view_jitter_std", 0.0))
        if self.view_jitter_std < 0.0:
            raise ValueError(
                "line_jepa_view_jitter_std must be >= 0, "
                f"got {self.view_jitter_std}."
            )
        self.view_scale_range = float(getattr(cfg, "line_jepa_view_scale_range", 0.0))
        if not (0.0 <= self.view_scale_range < 1.0):
            raise ValueError(
                "line_jepa_view_scale_range must be in [0, 1), "
                f"got {self.view_scale_range}."
            )
        self.prediction_hard_weighting_enabled = bool(
            getattr(cfg, "line_jepa_prediction_hard_weighting_enabled", False)
        )
        self.prediction_hard_top_fraction = float(
            getattr(cfg, "line_jepa_prediction_hard_top_fraction", 0.35)
        )
        self.prediction_hard_weight_low = float(
            getattr(cfg, "line_jepa_prediction_hard_weight_low", 0.2)
        )
        self.prediction_hard_weight_high = float(
            getattr(cfg, "line_jepa_prediction_hard_weight_high", 2.0)
        )
        self.prediction_hard_weight_normalize_mean = bool(
            getattr(cfg, "line_jepa_prediction_hard_weight_normalize_mean", True)
        )
        if not (0.0 < self.prediction_hard_top_fraction <= 1.0):
            raise ValueError(
                "line_jepa_prediction_hard_top_fraction must be in (0, 1], "
                f"got {self.prediction_hard_top_fraction}."
            )
        if self.prediction_hard_weight_low < 0.0:
            raise ValueError(
                "line_jepa_prediction_hard_weight_low must be >= 0, "
                f"got {self.prediction_hard_weight_low}."
            )
        if self.prediction_hard_weight_high < 0.0:
            raise ValueError(
                "line_jepa_prediction_hard_weight_high must be >= 0, "
                f"got {self.prediction_hard_weight_high}."
            )
        if self.prediction_hard_weight_high < self.prediction_hard_weight_low:
            raise ValueError(
                "line_jepa_prediction_hard_weight_high must be >= "
                "line_jepa_prediction_hard_weight_low. "
                f"Got high={self.prediction_hard_weight_high}, "
                f"low={self.prediction_hard_weight_low}."
            )
        if self.prediction_hard_weight_high == 0.0 and self.prediction_hard_weight_low == 0.0:
            raise ValueError(
                "At least one of line_jepa_prediction_hard_weight_high or "
                "line_jepa_prediction_hard_weight_low must be positive."
            )
        self.anchor_view_sim_coeff = float(getattr(cfg, "line_jepa_anchor_view_sim_coeff", 0.0))
        if self.anchor_view_sim_coeff < 0.0:
            raise ValueError(
                "line_jepa_anchor_view_sim_coeff must be >= 0, "
                f"got {self.anchor_view_sim_coeff}."
            )
        self.target_encoder_mode = str(getattr(cfg, "line_jepa_target_encoder", "ema")).strip().lower()
        if self.target_encoder_mode not in {"ema", "online"}:
            raise ValueError(
                "line_jepa_target_encoder must be 'ema' or 'online', "
                f"got {self.target_encoder_mode!r}."
            )
        self.target_ema_decay = float(getattr(cfg, "line_jepa_target_ema_decay", 0.996))
        if not (0.0 <= self.target_ema_decay < 1.0):
            raise ValueError(
                "line_jepa_target_ema_decay must be in [0, 1), "
                f"got {self.target_ema_decay}."
            )
        if self.target_encoder_mode == "ema":
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_encoder_io = EncoderAdapter(self.target_encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            self.target_encoder.eval()
        else:
            self.target_encoder = None
            self.target_encoder_io = None

        feature_dim = self._resolve_feature_dim()
        self.regularizer_projector = self._build_regularizer_projector(cfg, input_dim=feature_dim)
        hidden_dim = int(getattr(cfg, "line_jepa_predictor_hidden_dim", feature_dim))
        self.predictor = LineContextPredictor(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            depth=int(getattr(cfg, "line_jepa_predictor_depth", 2)),
            num_heads=int(getattr(cfg, "line_jepa_predictor_heads", 4)),
            mlp_ratio=float(getattr(cfg, "line_jepa_predictor_mlp_ratio", 4.0)),
            dropout=float(getattr(cfg, "line_jepa_predictor_dropout", 0.0)),
        )

    @staticmethod
    def _build_regularizer_projector(cfg, *, input_dim: int) -> nn.Module | None:
        if not bool(getattr(cfg, "line_jepa_regularizer_projector_enabled", False)):
            return None
        input_dim = int(input_dim)
        output_dim = int(getattr(cfg, "line_jepa_regularizer_projector_dim", 120))
        hidden_dim = int(getattr(cfg, "line_jepa_regularizer_projector_hidden_dim", input_dim))
        bn_eval_batch_stats = bool(
            getattr(cfg, "line_jepa_regularizer_projector_bn_eval_batch_stats", False)
        )
        if input_dim <= 0:
            raise ValueError(f"line_jepa regularizer projector input_dim must be > 0, got {input_dim}.")
        if output_dim <= 0:
            raise ValueError(
                "line_jepa_regularizer_projector_dim must be > 0, "
                f"got {output_dim}."
            )
        if hidden_dim <= 0:
            raise ValueError(
                "line_jepa_regularizer_projector_hidden_dim must be > 0, "
                f"got {hidden_dim}."
            )
        norm_layer = EvalBatchStatsBatchNorm1d if bn_eval_batch_stats else nn.BatchNorm1d
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            norm_layer(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            norm_layer(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def load_pretrained_weights_from_checkpoint(self, checkpoint_path: str, *, strict: bool = False) -> None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Expected checkpoint dict from {checkpoint_path}, got {type(checkpoint)!r}.")
        source_state = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
        if not isinstance(source_state, dict):
            raise ValueError(
                "Line-JEPA checkpoint initialization expected 'state_dict' or "
                f"'model_state_dict' in {checkpoint_path}."
            )

        normalized_source = {}
        for key, value in source_state.items():
            new_key = str(key)
            for prefix in ("model.", "module."):
                while new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            new_key = new_key.replace("._orig_mod.", ".")
            if new_key.startswith("_orig_mod."):
                new_key = new_key[len("_orig_mod."):]
            normalized_source[new_key] = value

        def candidate_model_keys(key: str) -> list[str]:
            candidates = [key]
            for prefix in ("encoder.", "target_encoder."):
                if key.startswith(prefix):
                    suffix = key[len(prefix):]
                    candidates.append(f"{prefix}_orig_mod.{suffix}")
            return candidates

        model_state = self.state_dict()
        compatible = {}
        shape_mismatch = []
        for key, value in normalized_source.items():
            target_key = None
            target = None
            for candidate in candidate_model_keys(key):
                maybe_target = model_state.get(candidate)
                if maybe_target is not None:
                    target_key = candidate
                    target = maybe_target
                    break
            if target is None:
                continue
            if tuple(target.shape) != tuple(value.shape):
                shape_mismatch.append(key)
                continue
            if target_key in compatible:
                raise RuntimeError(
                    "Line-JEPA checkpoint initialization found duplicate source tensors for "
                    f"target key {target_key!r} from {checkpoint_path}."
                )
            compatible[target_key] = value
        if not compatible:
            raise RuntimeError(
                f"No compatible tensors found when initializing Line-JEPA from {checkpoint_path}."
            )
        loaded_encoder_keys = sum(1 for key in compatible if key.startswith("encoder."))
        if loaded_encoder_keys == 0:
            raise RuntimeError(
                "Line-JEPA initialization loaded no online encoder tensors. "
                f"checkpoint={checkpoint_path}, compatible_keys={len(compatible)}."
            )

        missing_keys, unexpected_keys = self.load_state_dict(compatible, strict=bool(strict))
        target_sync_message = "No EMA target encoder is active."
        if self.target_encoder_mode == "ema":
            if self.target_encoder is None:
                raise RuntimeError("Line-JEPA target_encoder_mode='ema' but target_encoder is not initialized.")
            self.target_encoder.load_state_dict(self.encoder.state_dict(), strict=True)
            self.target_encoder.eval()
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            target_sync_message = "Synced EMA target encoder to online encoder."
        print(
            "Initialized Line-JEPA from checkpoint: "
            f"{checkpoint_path} | loaded={len(compatible)}/{len(model_state)} "
            f"encoder={loaded_encoder_keys}, shape_mismatch_skipped={len(shape_mismatch)}, "
            f"missing_after_load={len(missing_keys)}, unexpected_after_load={len(unexpected_keys)}, "
            f"strict={bool(strict)}. {target_sync_message}"
        )

    def _resolve_feature_dim(self) -> int:
        if self.vicreg.invariant_head is not None:
            return int(self.vicreg.invariant_head.output_dim)
        latent_dim = getattr(self.encoder, "invariant_dim", None)
        if latent_dim is None:
            latent_dim = getattr(self.encoder, "latent_size", None)
        if latent_dim is None:
            latent_dim = getattr(self.hparams, "latent_size", None)
        if latent_dim is None:
            raise ValueError(
                "LineJEPAModule could not resolve the encoder invariant feature dimension. "
                "Set encoder.invariant_dim/latent_size or cfg.latent_size."
            )
        return int(latent_dim)

    @staticmethod
    def _unpack_batch(batch):
        return batch["points"], {
            "class_id": batch.get("class_id"),
            "target_atom_id": batch.get("target_atom_id"),
            "anchor_atom_id": batch.get("anchor_atom_id"),
            "instance_id": batch.get("instance_id"),
            "frame_indices": batch.get("frame_indices"),
            "timesteps": batch.get("timesteps"),
            "coords": batch.get("coords"),
            "source_path": batch.get("source_path"),
        }

    def _weighted_total_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        line_loss_keys = {
            key
            for key in losses
            if key == "line_jepa" or key.startswith("line_jepa_")
        }
        base_losses = {key: value for key, value in losses.items() if key not in line_loss_keys}
        total = super()._weighted_total_loss(base_losses)
        for key in sorted(line_loss_keys):
            total = total + losses[key]
        return total

    def _anchor_view_similarity_loss(
        self,
        anchor_view_features_flat: torch.Tensor,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if anchor_view_features_flat.dim() != 2:
            raise ValueError(
                "Line-JEPA anchor-view features must have shape (2 * B, D), "
                f"got {tuple(anchor_view_features_flat.shape)}."
            )
        expected_rows = int(batch_size) * 2
        if int(anchor_view_features_flat.shape[0]) != expected_rows:
            raise ValueError(
                "Line-JEPA anchor-view feature row count mismatch. "
                f"expected_rows={expected_rows}, got={int(anchor_view_features_flat.shape[0])}, "
                f"batch_size={int(batch_size)}."
            )
        features = anchor_view_features_flat.reshape(int(batch_size), 2, -1).float()
        sim_loss = F.mse_loss(features[:, 0], features[:, 1])
        pred_cos = F.cosine_similarity(features[:, 0], features[:, 1], dim=-1).mean()
        return sim_loss, pred_cos

    def _prediction_target(
        self,
        *,
        target_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_mean = context_features.detach().mean(dim=1)
        if self.prediction_target == "target":
            return target_features, context_mean
        if self.prediction_target == "residual":
            return target_features - context_mean, context_mean
        raise RuntimeError(f"Unsupported Line-JEPA prediction target mode: {self.prediction_target!r}.")

    def _prediction_hard_weights(
        self,
        *,
        novelty: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if not self.prediction_hard_weighting_enabled:
            return None, {}
        novelty = novelty.detach().float().reshape(-1)
        if int(novelty.numel()) == 0:
            raise ValueError("Line-JEPA hard prediction weighting received an empty novelty tensor.")
        if not torch.isfinite(novelty).all().item():
            raise ValueError("Line-JEPA hard prediction novelty contains non-finite values.")

        task_count = int(novelty.numel())
        hard_count = max(1, int(math.ceil(task_count * self.prediction_hard_top_fraction)))
        hard_count = min(hard_count, task_count)
        hard_values, hard_indices = torch.topk(novelty, k=hard_count, largest=True, sorted=False)

        weights = novelty.new_full((task_count,), self.prediction_hard_weight_low)
        weights.scatter_(0, hard_indices, float(self.prediction_hard_weight_high))
        hard_mask = torch.zeros((task_count,), dtype=torch.bool, device=novelty.device)
        hard_mask.scatter_(0, hard_indices, True)
        if self.prediction_hard_weight_normalize_mean:
            weight_mean = weights.mean()
            if float(weight_mean.detach()) <= 0.0:
                raise RuntimeError(
                    "Line-JEPA hard prediction weights have zero mean after top-k assignment. "
                    f"low={self.prediction_hard_weight_low}, high={self.prediction_hard_weight_high}, "
                    f"hard_count={hard_count}, task_count={task_count}."
                )
            weights = weights / weight_mean

        easy_values = novelty[~hard_mask]
        easy_mean = (
            easy_values.mean()
            if int(easy_values.numel()) > 0
            else novelty.new_tensor(0.0)
        )
        metrics = {
            "line_jepa_prediction_novelty_mean": novelty.mean(),
            "line_jepa_prediction_novelty_hard_mean": hard_values.mean(),
            "line_jepa_prediction_novelty_easy_mean": easy_mean,
            "line_jepa_prediction_hard_threshold": hard_values.min(),
        }
        return weights, metrics

    def _encode_features(self, points: torch.Tensor) -> torch.Tensor:
        prepared = self._prepare_model_input(points)
        encoded = self.encoder_io.encode(prepared)
        features = self._shared_invariant(encoded.invariant, encoded.equivariant)
        if features is None:
            raise RuntimeError(
                "Line-JEPA encoder returned no invariant/equivariant features after invariant reduction. "
                f"input_shape={tuple(points.shape)}."
            )
        if features.dim() != 2:
            raise ValueError(
                "Line-JEPA expects encoder features with shape (B, D), "
                f"got {tuple(features.shape)} for input_shape={tuple(points.shape)}."
            )
        return features

    def _prepare_line_jepa_view(self, points: torch.Tensor) -> torch.Tensor:
        view = self._prepare_model_input(points)
        did_augment = False
        if self.view_scale_range > 0.0:
            scale = (
                1.0
                + (torch.rand((view.shape[0], 1, 1), device=view.device, dtype=view.dtype) * 2.0 - 1.0)
                * self.view_scale_range
            )
            view = view * scale
            did_augment = True
        if self.view_jitter_std > 0.0:
            view = view + torch.randn_like(view) * self.view_jitter_std
            did_augment = True
        if did_augment:
            view = view - view.mean(dim=1, keepdim=True)
        return view

    def _project_regularized_embeddings(
        self,
        embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.regularizer_projector is None:
            return embeddings
        if not embeddings:
            raise ValueError("Line-JEPA regularizer projector received no embeddings to project.")
        projector_dtype = next(self.regularizer_projector.parameters()).dtype
        names = list(embeddings)
        batch_sizes = [int(embeddings[name].shape[0]) for name in names]
        fused = torch.cat(
            [embeddings[name].to(dtype=projector_dtype) for name in names],
            dim=0,
        )
        projected_blocks = self.regularizer_projector(fused).split(batch_sizes, dim=0)
        return {
            name: projected
            for name, projected in zip(names, projected_blocks, strict=True)
        }

    def _encode_prepared_feature_blocks(self, blocks: list[torch.Tensor]) -> list[torch.Tensor]:
        if not blocks:
            raise ValueError("Line-JEPA online encoder fusion received no input blocks.")
        point_counts = {int(block.shape[1]) for block in blocks}
        if len(point_counts) != 1:
            raise ValueError(
                "Line-JEPA online encoder fusion requires all blocks to have the same point count. "
                f"Got point_counts={sorted(point_counts)}. Set data.model_points and "
                "vicreg_view_points consistently, or disable the fused path."
            )
        batch_sizes = [int(block.shape[0]) for block in blocks]
        fused_input = torch.cat(blocks, dim=0)
        encoded = self.encoder_io.encode(fused_input)
        features = self._shared_invariant(encoded.invariant, encoded.equivariant)
        if features is None:
            raise RuntimeError(
                "Line-JEPA fused online encoder returned no invariant/equivariant features after "
                f"invariant reduction. input_shape={tuple(fused_input.shape)}."
            )
        if features.dim() != 2:
            raise ValueError(
                "Line-JEPA fused online encoder expects features with shape (B, D), "
                f"got {tuple(features.shape)} for input_shape={tuple(fused_input.shape)}."
            )
        return list(features.split(batch_sizes, dim=0))

    def _build_contrastive_view_pair(
        self,
        pc: torch.Tensor,
        *,
        view_points: int | None,
    ) -> dict[str, torch.Tensor]:
        if self.vicreg.requires_overlap_target:
            return self.vicreg.build_overlap_view_pair(pc, view_points=view_points)
        use_neighbor_a, use_neighbor_b = self.vicreg._resolve_neighbor_flags(device=pc.device)
        apply_occlusion_a, apply_occlusion_b = self.vicreg._resolve_pair_occlusion_flags(
            use_neighbor_a=use_neighbor_a,
            use_neighbor_b=use_neighbor_b,
            device=pc.device,
        )
        return {
            "y_a": self.vicreg._augment(
                pc,
                use_neighbor=use_neighbor_a,
                apply_occlusion=apply_occlusion_a,
                view_points=view_points,
            ),
            "y_b": self.vicreg._augment(
                pc,
                use_neighbor=use_neighbor_b,
                apply_occlusion=apply_occlusion_b,
                view_points=view_points,
            ),
        }

    def _encode_contrastive_view_pair(
        self,
        views: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_a = views["y_a"]
        y_b = views["y_b"]
        fused_input = torch.cat([y_a, y_b], dim=0)
        encoded = self.encoder_io.encode(fused_input)
        features = self._shared_invariant(encoded.invariant, encoded.equivariant)
        return features.chunk(2, dim=0)

    @torch.no_grad()
    def _encode_ema_target_features(self, points: torch.Tensor) -> torch.Tensor:
        if self.target_encoder is None or self.target_encoder_io is None:
            raise RuntimeError(
                "Line-JEPA EMA target encoding was requested, but the EMA target encoder is not initialized. "
                f"line_jepa_target_encoder={self.target_encoder_mode!r}."
            )
        prepared = self._prepare_model_input(points)
        self.target_encoder.eval()
        encoded = self.target_encoder_io.encode(prepared)
        features = self._shared_invariant(encoded.invariant, encoded.equivariant)
        if features is None:
            raise RuntimeError(
                "Line-JEPA target encoder returned no invariant/equivariant features after invariant reduction. "
                f"input_shape={tuple(points.shape)}."
            )
        if features.dim() != 2:
            raise ValueError(
                "Line-JEPA expects target encoder features with shape (B, D), "
                f"got {tuple(features.shape)} for input_shape={tuple(points.shape)}."
            )
        return features

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        if self.target_encoder_mode != "ema":
            return
        if self.target_encoder is None:
            raise RuntimeError("Line-JEPA target_encoder_mode='ema' but target_encoder is not initialized.")
        decay = float(self.target_ema_decay)
        online_params = dict(self.encoder.named_parameters())
        target_params = dict(self.target_encoder.named_parameters())
        if online_params.keys() != target_params.keys():
            raise RuntimeError(
                "Line-JEPA EMA target encoder parameter names do not match the online encoder. "
                f"online_only={sorted(set(online_params) - set(target_params))[:5]}, "
                f"target_only={sorted(set(target_params) - set(online_params))[:5]}."
            )
        for name, target_param in target_params.items():
            target_param.data.mul_(decay).add_(online_params[name].data, alpha=1.0 - decay)

        online_buffers = dict(self.encoder.named_buffers())
        target_buffers = dict(self.target_encoder.named_buffers())
        if online_buffers.keys() != target_buffers.keys():
            raise RuntimeError(
                "Line-JEPA EMA target encoder buffer names do not match the online encoder. "
                f"online_only={sorted(set(online_buffers) - set(target_buffers))[:5]}, "
                f"target_only={sorted(set(target_buffers) - set(online_buffers))[:5]}."
            )
        for name, target_buffer in target_buffers.items():
            target_buffer.copy_(online_buffers[name])

    def _line_positions(self, batch: dict, *, device, dtype) -> torch.Tensor:
        line_t = batch["line_t"].to(device=device, dtype=dtype, non_blocking=True)
        line_perp = batch["line_perp"].to(device=device, dtype=dtype, non_blocking=True)
        if line_t.shape != line_perp.shape:
            raise ValueError(
                "Line-JEPA batch line_t and line_perp must have identical shapes. "
                f"Got line_t={tuple(line_t.shape)}, line_perp={tuple(line_perp.shape)}."
            )
        target_t = line_t[:, self.target_line_index].unsqueeze(1)
        relative_t = line_t - target_t
        return torch.stack((relative_t, line_perp), dim=-1)

    def _prediction_target_index_values(self) -> list[int]:
        if self.prediction_positions == "center":
            return [self.target_line_index]
        if self.prediction_positions == "all":
            return list(range(self.line_atoms))
        raise RuntimeError(f"Unsupported line_jepa_prediction_positions: {self.prediction_positions!r}.")

    def _line_positions_for_targets(
        self,
        batch: dict,
        *,
        target_indices: list[int],
        device,
        dtype,
    ) -> torch.Tensor:
        line_t = batch["line_t"].to(device=device, dtype=dtype, non_blocking=True)
        line_perp = batch["line_perp"].to(device=device, dtype=dtype, non_blocking=True)
        if line_t.shape != line_perp.shape:
            raise ValueError(
                "Line-JEPA batch line_t and line_perp must have identical shapes. "
                f"Got line_t={tuple(line_t.shape)}, line_perp={tuple(line_perp.shape)}."
            )
        target_t = torch.stack([line_t[:, int(index)] for index in target_indices], dim=1)
        relative_t = line_t.unsqueeze(1) - target_t.unsqueeze(2)
        relative_perp = line_perp.unsqueeze(1).expand(-1, len(target_indices), -1)
        return torch.stack((relative_t, relative_perp), dim=-1)

    def forward(self, pc: torch.Tensor, include_ssl_heads: bool = False):
        if pc.dim() == 4:
            pc = pc[:, self.target_line_index]
        pc = self._prepare_model_input(pc).to(device=self.device, dtype=self.dtype)
        encoded = self.encoder_io.encode(pc)
        z_inv_contrastive = self._contrastive_invariant_latent(
            encoded.invariant,
            encoded.equivariant,
        )
        if include_ssl_heads:
            return (
                z_inv_contrastive,
                encoded.invariant,
                encoded.equivariant,
                self._forward_ssl_heads_for_summary(z_inv_contrastive),
            )
        return z_inv_contrastive, encoded.invariant, encoded.equivariant

    def _step(self, batch, batch_idx, stage: str):
        line_points, meta = self._unpack_batch(batch)
        if line_points.dim() != 4 or line_points.shape[2] <= 0 or line_points.shape[-1] != 3:
            raise ValueError(
                "Line-JEPA batch['points'] must have shape (B, L, P, 3). "
                f"Got {tuple(line_points.shape)}."
            )
        if int(line_points.shape[1]) != self.line_atoms:
            raise ValueError(
                "Line-JEPA batch line length does not match data.line_atoms. "
                f"Expected {self.line_atoms}, got {int(line_points.shape[1])}."
            )

        batch_size, _, points_per_structure, _ = line_points.shape
        line_points = line_points.to(device=self.device, dtype=self.dtype, non_blocking=True)

        target_points = line_points[:, self.target_line_index]

        losses = {}
        current_epoch = int(self.current_epoch)
        line_jepa_active = self.line_jepa.should_run(current_epoch=current_epoch)
        vicreg_active = self.vicreg.should_run(current_epoch=current_epoch)

        line_flat = None
        views = None
        online_blocks = []
        block_names = []
        anchor_view_sim_active = self.anchor_view_sim_coeff > 0.0
        swav_anchor_active = self.swav.should_run(current_epoch=current_epoch)
        anchor_view_objective_active = anchor_view_sim_active or swav_anchor_active
        if line_jepa_active:
            line_flat = line_points.reshape(
                batch_size * self.line_atoms,
                points_per_structure,
                3,
            )
            online_blocks.append(self._prepare_line_jepa_view(line_flat))
            block_names.append("line_views")
            if anchor_view_objective_active:
                anchor_view_points = batch.get("line_anchor_view_points")
                if anchor_view_points is None:
                    raise ValueError(
                        "Line-JEPA anchor-view objectives require the batch to contain "
                        "'line_anchor_view_points'. Set data.line_anchor_views_enabled=true."
                    )
                if (
                    anchor_view_points.dim() != 4
                    or int(anchor_view_points.shape[0]) != batch_size
                    or int(anchor_view_points.shape[1]) != 2
                    or int(anchor_view_points.shape[-1]) != 3
                ):
                    raise ValueError(
                        "Line-JEPA batch['line_anchor_view_points'] must have shape (B, 2, P, 3). "
                        f"Got {tuple(anchor_view_points.shape)} for batch_size={batch_size}."
                    )
                anchor_view_points = anchor_view_points.to(
                    device=self.device,
                    dtype=self.dtype,
                    non_blocking=True,
                )
                anchor_view_flat = anchor_view_points.reshape(
                    batch_size * 2,
                    int(anchor_view_points.shape[2]),
                    3,
                )
                online_blocks.append(self._prepare_line_jepa_view(anchor_view_flat))
                block_names.append("anchor_views")

        if vicreg_active:
            views = self._build_contrastive_view_pair(
                target_points,
                view_points=self.vicreg.view_points,
            )

            online_blocks.extend([views["y_a"], views["y_b"]])
            block_names.extend(["vicreg_a", "vicreg_b"])

        encoded_blocks = {}
        if online_blocks:
            for name, features in zip(
                block_names,
                self._encode_prepared_feature_blocks(online_blocks),
                strict=True,
            ):
                encoded_blocks[name] = features

        if line_jepa_active:
            if line_flat is None:
                raise RuntimeError("Line-JEPA active branch did not build line_flat.")
            line_features_flat = encoded_blocks["line_views"]
            feature_dim = int(line_features_flat.shape[-1])
            line_features = line_features_flat.reshape(
                batch_size,
                self.line_atoms,
                feature_dim,
            )
            target_index_values = self._prediction_target_index_values()
            prediction_task_count = len(target_index_values)
            line_positions = self._line_positions_for_targets(
                batch,
                target_indices=target_index_values,
                device=self.device,
                dtype=line_features.dtype,
            )
            context_feature_blocks = []
            target_feature_blocks = []
            target_point_blocks = []
            context_position_blocks = []
            target_position_blocks = []
            for task_idx, target_index in enumerate(target_index_values):
                context_indices = [index for index in range(self.line_atoms) if index != int(target_index)]
                context_feature_blocks.append(line_features[:, context_indices])
                target_feature_blocks.append(line_features[:, int(target_index)])
                target_point_blocks.append(line_points[:, int(target_index)])
                context_position_blocks.append(line_positions[:, task_idx, context_indices])
                target_position_blocks.append(line_positions[:, task_idx, int(target_index)])

            context_features = torch.stack(context_feature_blocks, dim=1).reshape(
                batch_size * prediction_task_count,
                self.line_atoms - 1,
                feature_dim,
            )
            target_online_features = torch.stack(target_feature_blocks, dim=1).reshape(
                batch_size * prediction_task_count,
                feature_dim,
            )
            target_task_points = torch.stack(target_point_blocks, dim=1).reshape(
                batch_size * prediction_task_count,
                points_per_structure,
                3,
            )
            context_positions = torch.stack(context_position_blocks, dim=1).reshape(
                batch_size * prediction_task_count,
                self.line_atoms - 1,
                2,
            )
            target_position = torch.stack(target_position_blocks, dim=1).reshape(
                batch_size * prediction_task_count,
                2,
            )
            if self.target_encoder_mode == "ema":
                target_features = self._encode_ema_target_features(target_task_points)
            elif self.target_encoder_mode == "online":
                target_features = target_online_features
            else:
                raise RuntimeError(
                    f"Unsupported line_jepa_target_encoder at runtime: {self.target_encoder_mode!r}."
                )
            prediction = self.predictor(context_features, context_positions, target_position)
            prediction_target, context_mean = self._prediction_target(
                target_features=target_features,
                context_features=context_features,
            )
            prediction_novelty = (
                target_features.detach().float() - context_mean.detach().float()
            ).norm(dim=1)
            prediction_weights, hard_weight_metrics = self._prediction_hard_weights(
                novelty=prediction_novelty,
            )
            prediction_batch_size = batch_size * prediction_task_count

            if self.regularize_online_target:
                regularized_embeddings = {"line_views": line_features_flat}
            else:
                regularized_embeddings = {
                    "context": context_features.reshape(
                        batch_size * prediction_task_count * (self.line_atoms - 1),
                        feature_dim,
                    )
                }
            if anchor_view_objective_active:
                anchor_view_features = encoded_blocks.get("anchor_views")
                if anchor_view_features is None:
                    raise RuntimeError(
                        "Line-JEPA anchor-view objective was active, but encoded anchor-view "
                        "features were not produced."
                    )
                regularized_embeddings["anchor_views"] = anchor_view_features
            regularized_embeddings = self._project_regularized_embeddings(regularized_embeddings)

            line_loss, metrics = self.line_jepa.compute_loss(
                prediction=prediction,
                target=prediction_target,
                regularized_embeddings=regularized_embeddings,
                current_epoch=current_epoch,
                global_step=int(self.global_step),
                prediction_weights=prediction_weights,
            )
            if line_loss is not None:
                losses["line_jepa"] = line_loss
            for name, value in metrics.items():
                self._log_metric(stage, name, value, batch_size=prediction_batch_size)
            for name, value in hard_weight_metrics.items():
                self._log_metric(stage, name, value, batch_size=prediction_batch_size)
            if anchor_view_sim_active:
                anchor_view_features = encoded_blocks.get("anchor_views")
                if anchor_view_features is None:
                    raise RuntimeError(
                        "Line-JEPA anchor-view similarity was active, but encoded anchor-view "
                        "features were not produced."
                    )
                anchor_sim, anchor_cos = self._anchor_view_similarity_loss(
                    anchor_view_features,
                    batch_size=batch_size,
                )
                anchor_sim_weighted = self.anchor_view_sim_coeff * anchor_sim
                losses["line_jepa_anchor_view_sim"] = anchor_sim_weighted
                self._log_metric(
                    stage,
                    "line_jepa_anchor_view_sim",
                    anchor_sim,
                    batch_size=batch_size,
                )
                self._log_metric(
                    stage,
                    "line_jepa_anchor_view_sim_weighted",
                    anchor_sim_weighted,
                    batch_size=batch_size,
                )
                self._log_metric(
                    stage,
                    "line_jepa_anchor_view_cos",
                    anchor_cos,
                    batch_size=batch_size,
                )

            if swav_anchor_active:
                anchor_view_features = encoded_blocks.get("anchor_views")
                if anchor_view_features is None:
                    raise RuntimeError(
                        "Line-JEPA SwAV anchor-view objective was active, but encoded anchor-view "
                        "features were not produced."
                    )
                if anchor_view_features.dim() != 2 or int(anchor_view_features.shape[0]) != batch_size * 2:
                    raise ValueError(
                        "Line-JEPA SwAV anchor-view features must have shape (2 * B, D). "
                        f"Got {tuple(anchor_view_features.shape)} for batch_size={batch_size}."
                    )
                swav_view_features = anchor_view_features.reshape(batch_size, 2, -1)
                swav_loss, swav_metrics = self.swav.compute_loss(
                    view_features=[
                        swav_view_features[:, 0],
                        swav_view_features[:, 1],
                    ],
                    current_epoch=current_epoch,
                )
                if swav_loss is not None:
                    losses["swav"] = swav_loss
                for name, value in swav_metrics.items():
                    self._log_metric(stage, name, value, batch_size=batch_size)

            with torch.no_grad():
                pred_cos = F.cosine_similarity(
                    prediction.float(),
                    prediction_target.detach().float(),
                    dim=-1,
                ).mean()
                self._log_metric(stage, "line_jepa_pred_cos", pred_cos, batch_size=prediction_batch_size)
                context_target_cos = F.cosine_similarity(
                    context_mean.float(),
                    target_features.detach().float(),
                    dim=-1,
                ).mean()
                self._log_metric(
                    stage,
                    "line_jepa_context_target_cos",
                    context_target_cos,
                    batch_size=prediction_batch_size,
                )
                recon_cos = prediction.new_tensor(float("nan"))
                if self.prediction_target == "residual":
                    reconstructed_target = prediction.float() + context_mean.float()
                    recon_cos = F.cosine_similarity(
                        reconstructed_target,
                        target_features.detach().float(),
                        dim=-1,
                    ).mean()
                    self._log_metric(
                        stage,
                        "line_jepa_recon_target_cos",
                        recon_cos,
                        batch_size=prediction_batch_size,
                    )
                if stage != "train" and batch_idx == 0:
                    residual_suffix = ""
                    if self.prediction_target == "residual":
                        residual_suffix = f" recon_cos={float(recon_cos.detach()):.6f}"
                    sigreg_value = metrics.get("line_jepa_sigreg")
                    sigreg_suffix = (
                        f"sigreg={float(sigreg_value.detach()):.6f} "
                        if sigreg_value is not None
                        else ""
                    )
                    self._status_print(
                        f"[{stage}-line-jepa] epoch={self.current_epoch} "
                        f"pred={float(metrics.get('line_jepa_pred', prediction.new_tensor(float('nan'))).detach()):.6f} "
                        f"{sigreg_suffix}"
                        f"cos={float(pred_cos.detach()):.6f} "
                        f"context_cos={float(context_target_cos.detach()):.6f} "
                        f"tasks={prediction_task_count}"
                        f"{residual_suffix}"
                    )

        if vicreg_active:
            if views is None:
                raise RuntimeError("VICReg active branch did not build contrastive views.")
            z_a = encoded_blocks["vicreg_a"]
            z_b = encoded_blocks["vicreg_b"]
            vicreg_loss, vicreg_metrics = self.vicreg.compute_loss_from_features(
                z_a_feat=z_a,
                z_b_feat=z_b,
                current_epoch=current_epoch,
                overlap_target=views.get("overlap_target"),
            )
            if vicreg_loss is not None:
                losses[self.vicreg.metric_prefix] = vicreg_loss
            for name, value in vicreg_metrics.items():
                self._log_metric(stage, name, value, batch_size=batch_size)

        if self._should_cache_supervised_stage(stage):
            with torch.no_grad():
                z_inv = self._encode_features(target_points)
            self._cache_supervised_embeddings_if_needed(
                stage=stage,
                meta=meta,
                embeddings=z_inv,
            )

        return self._finish_ssl_step(
            stage=stage,
            batch_idx=batch_idx,
            batch_size=batch_size,
            losses=losses,
            print_first_eval_batch=True,
        )

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        if self.target_encoder_mode == "ema":
            self._update_target_encoder()


__all__ = ["LineJEPAModule", "LineContextPredictor"]
