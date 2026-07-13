import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_utils.data_kinds import normalize_data_kind
from src.data_utils.data_modules.line_jepa import LineJEPADataModule
from src.models import EncoderAdapter
from src.training_methods.base_ssl_module import BaseSSLModule
from src.training_methods.contrastive_learning.vicreg import EvalBatchStatsBatchNorm1d
from src.training_methods.line_jepa.line_jepa import LineJEPALoss
from src.utils.training_utils import get_optimizers_and_scheduler


class CompactSemanticProjector(nn.Module):
    """Trainable compact projection initialized from the frozen teacher PCA basis."""

    def __init__(self, *, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(
                "CompactSemanticProjector dimensions must be positive, "
                f"got input_dim={input_dim}, output_dim={output_dim}."
            )
        if output_dim > input_dim:
            raise ValueError(
                "The compact semantic dimension cannot exceed the encoder feature dimension, "
                f"got output_dim={output_dim}, input_dim={input_dim}."
            )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.linear.weight)
        self.register_buffer("center", torch.zeros(input_dim), persistent=True)
        self.register_buffer("initialized", torch.tensor(False), persistent=True)

    @torch.no_grad()
    def initialize_from_teacher(self, teacher_features: torch.Tensor) -> None:
        if teacher_features.dim() != 2 or int(teacher_features.shape[1]) != self.input_dim:
            raise ValueError(
                "Teacher PCA initialization expects features with shape (B, D), "
                f"got {tuple(teacher_features.shape)}, expected D={self.input_dim}."
            )
        if int(teacher_features.shape[0]) <= self.output_dim:
            raise ValueError(
                "Teacher PCA initialization needs more samples than compact dimensions, "
                f"got samples={int(teacher_features.shape[0])}, output_dim={self.output_dim}."
            )
        with torch.autocast(device_type=teacher_features.device.type, enabled=False):
            x = teacher_features.detach().float()
            if not torch.isfinite(x).all().item():
                raise ValueError("Teacher PCA initialization received non-finite features.")
            center = x.mean(dim=0)
            centered = x - center
            covariance = centered.T @ centered / float(int(x.shape[0]) - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        if not torch.isfinite(eigenvalues).all().item():
            raise RuntimeError("Teacher PCA eigendecomposition produced non-finite eigenvalues.")
        basis = eigenvectors[:, -self.output_dim :].T.flip(0).contiguous()
        self.center.copy_(center.to(device=self.center.device, dtype=self.center.dtype))
        self.linear.weight.copy_(
            basis.to(device=self.linear.weight.device, dtype=self.linear.weight.dtype)
        )
        self.initialized.fill_(True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not bool(self.initialized.item()):
            raise RuntimeError(
                "Compact semantic projector is not initialized. A fresh Line-JEPA run must "
                "initialize it from a frozen-teacher batch before encoding semantic features."
            )
        if features.dim() != 2 or int(features.shape[1]) != self.input_dim:
            raise ValueError(
                "Compact semantic projection expects features with shape (B, D), "
                f"got {tuple(features.shape)}, expected D={self.input_dim}."
            )
        projected = self.linear(features - self.center.to(dtype=features.dtype))
        normalized = F.normalize(projected.float(), dim=-1, eps=1.0e-6)
        return normalized.to(dtype=projected.dtype)


class LineContextPredictor(nn.Module):
    """Transformer predictor for a masked local-structure embedding on a sampled line."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        output_dim: int,
        directional_dim: int = 0,
    ) -> None:
        super().__init__()
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
        if directional_dim < 0:
            raise ValueError(
                f"LineContextPredictor directional_dim must be >= 0, got {directional_dim}."
            )
        if output_dim <= 0:
            raise ValueError(f"LineContextPredictor output_dim must be > 0, got {output_dim}.")

        self.input_proj = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        self.directional_dim = directional_dim
        self.directional_mlp = (
            nn.Sequential(
                nn.Linear(directional_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if directional_dim > 0
            else None
        )
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
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        context_embeddings: torch.Tensor,
        context_positions: torch.Tensor,
        target_position: torch.Tensor,
        context_directional_features: torch.Tensor | None = None,
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
        if self.directional_mlp is None:
            if context_directional_features is not None:
                raise ValueError(
                    "LineContextPredictor received directional features, but directional_dim=0."
                )
        else:
            expected_shape = (*context_embeddings.shape[:2], self.directional_dim)
            if context_directional_features is None:
                raise ValueError(
                    "LineContextPredictor was configured with directional features, but none were provided. "
                    f"Expected shape {expected_shape}."
                )
            if tuple(context_directional_features.shape) != expected_shape:
                raise ValueError(
                    "LineContextPredictor directional feature shape mismatch: "
                    f"expected {expected_shape}, got {tuple(context_directional_features.shape)}."
                )

        dtype = context_embeddings.dtype
        tokens = self.input_proj(context_embeddings)
        tokens = tokens + self.position_mlp(context_positions.to(device=tokens.device, dtype=dtype))
        if self.directional_mlp is not None:
            tokens = tokens + self.directional_mlp(
                context_directional_features.to(device=tokens.device, dtype=dtype)
            )
        encoded = self.context_encoder(tokens)

        query = self.target_query.expand(context_embeddings.shape[0], -1, -1)
        query = query + self.position_mlp(target_position.to(device=tokens.device, dtype=dtype)).unsqueeze(1)
        attended, _ = self.cross_attention(query=query, key=encoded, value=encoded, need_weights=False)
        return self.output(self.norm(attended.squeeze(1)))


class LineJEPAModule(BaseSSLModule):
    """JEPA-style training from line context to the central local-structure embedding."""

    data_module_class = LineJEPADataModule
    test_metrics_on_step = True
    _WANDB_GROUPS = {
        "pred": "prediction",
        "sim": "similarity",
        "context": "context",
        "view": "view_consistency",
        "reg": "regularization",
        "hard": "hard_weighting",
        "manifold": "manifold",
        "cluster": "clustering",
        "mask": "masked_tokens",
    }

    def __init__(self, cfg):
        self.data_kind = normalize_data_kind(cfg.data.kind)
        self.encoder_name = cfg.encoder.name
        self.line_atoms = cfg.data.line_atoms
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
        self.prediction_target = cfg.line_jepa_prediction_target
        if self.prediction_target not in {"target", "residual"}:
            raise ValueError(
                "line_jepa_prediction_target must be 'target' or 'residual', "
                f"got {self.prediction_target!r}."
            )
        self.prediction_normalization = cfg.line_jepa_prediction_normalization
        if self.prediction_normalization not in {"none", "l2"}:
            raise ValueError(
                "line_jepa_prediction_normalization must be 'none' or 'l2', "
                f"got {self.prediction_normalization!r}."
            )
        if self.prediction_normalization != "none" and self.prediction_target != "target":
            raise ValueError(
                "Normalized Line-JEPA prediction is only defined for direct target prediction. "
                f"Got line_jepa_prediction_normalization={self.prediction_normalization!r}, "
                f"line_jepa_prediction_target={self.prediction_target!r}."
            )
        if self.line_jepa.context_match_coeff > 0.0 and (
            self.prediction_target != "target" or self.prediction_normalization != "l2"
        ):
            raise ValueError(
                "Data-agnostic context matching requires direct L2-normalized target prediction. "
                f"Got target={self.prediction_target!r}, normalization={self.prediction_normalization!r}."
            )
        self.prediction_positions = cfg.line_jepa_prediction_positions
        if self.prediction_positions not in {"center", "all", "cycle", "endpoints"}:
            raise ValueError(
                "line_jepa_prediction_positions must be 'center', 'all', 'cycle', or 'endpoints', "
                f"got {self.prediction_positions!r}."
            )
        self.view_jitter_std = cfg.line_jepa_view_jitter_std
        if self.view_jitter_std < 0.0:
            raise ValueError(
                "line_jepa_view_jitter_std must be >= 0, "
                f"got {self.view_jitter_std}."
            )
        self.view_scale_range = cfg.line_jepa_view_scale_range
        if not (0.0 <= self.view_scale_range < 1.0):
            raise ValueError(
                "line_jepa_view_scale_range must be in [0, 1), "
                f"got {self.view_scale_range}."
            )
        self.prediction_hard_enabled = cfg.line_jepa_prediction_hard_enabled
        self.prediction_hard_top_fraction = cfg.line_jepa_prediction_hard_top_fraction
        self.prediction_hard_weight_low = cfg.line_jepa_prediction_hard_weight_low
        self.prediction_hard_weight_high = cfg.line_jepa_prediction_hard_weight_high
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
        self.target_view_sim_coeff = cfg.line_jepa_target_view_sim_coeff
        if self.target_view_sim_coeff < 0.0:
            raise ValueError(
                "line_jepa_target_view_sim_coeff must be >= 0, "
                f"got {self.target_view_sim_coeff}."
            )
        self.shuffle_controls_enabled = cfg.line_jepa_shuffle_controls_enabled
        self.shuffle_control_seed = cfg.line_jepa_shuffle_control_seed
        if self.shuffle_controls_enabled and self.prediction_target != "target":
            raise ValueError(
                "Line-JEPA shuffle controls currently require direct target prediction, "
                f"got line_jepa_prediction_target={self.prediction_target!r}."
            )
        self.target_encoder_mode = cfg.line_jepa_target_encoder
        if self.target_encoder_mode not in {"ema", "frozen", "online"}:
            raise ValueError(
                "line_jepa_target_encoder must be 'ema', 'frozen', or 'online', "
                f"got {self.target_encoder_mode!r}."
            )
        self.target_ema_decay = cfg.line_jepa_target_ema_decay
        if not (0.0 <= self.target_ema_decay < 1.0):
            raise ValueError(
                "line_jepa_target_ema_decay must be in [0, 1), "
                f"got {self.target_ema_decay}."
            )
        if self.target_encoder_mode in {"ema", "frozen"}:
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_encoder_io = EncoderAdapter(self.target_encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            self.target_encoder.eval()
        else:
            self.target_encoder = None
            self.target_encoder_io = None
        self._frozen_target_initialized = self.target_encoder_mode != "frozen"

        self.directional_feature_mode = cfg.line_jepa_directional_feature_mode
        if self.directional_feature_mode not in {"none", "moments", "encoder"}:
            raise ValueError(
                "line_jepa_directional_feature_mode must be 'none', 'moments', or 'encoder', "
                f"got {self.directional_feature_mode!r}."
            )
        if self.directional_feature_mode in {"moments", "encoder"} and self.prediction_positions != "endpoints":
            raise ValueError(
                "Directional features currently require endpoint prediction so every task "
                "has a single unambiguous ray direction. Set "
                "line_jepa_prediction_positions=endpoints."
            )
        if self.directional_feature_mode == "moments":
            directional_feature_dim = 8
        elif self.directional_feature_mode == "encoder":
            encoder_extension = self._encoder_extension()
            if self.encoder_name != "GeoFrameTransformer":
                raise ValueError(
                    "line_jepa_directional_feature_mode='encoder' is produced by the "
                    "GeoFrameTransformer geometry state, "
                    f"got encoder.name={self.encoder_name!r}."
                )
            directional_feature_dim = encoder_extension.directional_feature_dim
        else:
            directional_feature_dim = 0

        self.masked_token_coeff = cfg.line_jepa_masked_token_coeff
        self.masked_token_samples = cfg.line_jepa_masked_token_samples
        if self.masked_token_coeff < 0.0:
            raise ValueError(
                "line_jepa_masked_token_coeff must be >= 0, "
                f"got {self.masked_token_coeff}."
            )
        if self.masked_token_samples <= 0:
            raise ValueError(
                "line_jepa_masked_token_samples must be > 0, "
                f"got {self.masked_token_samples}."
            )
        if self.masked_token_coeff > 0.0:
            if self.encoder_name != "GeoFrameTransformer":
                raise ValueError(
                    "line_jepa_masked_token_coeff > 0 requires the GeoFrameTransformer "
                    f"masked-token producer, got encoder.name={self.encoder_name!r}."
                )

        feature_dim = self._resolve_feature_dim()
        self.semantic_dim = cfg.line_jepa_semantic_dim
        if not (1 <= self.semantic_dim <= feature_dim):
            raise ValueError(
                "line_jepa_semantic_dim must be in [1, encoder_feature_dim], "
                f"got semantic_dim={self.semantic_dim}, encoder_feature_dim={feature_dim}."
            )
        self.semantic_anchor_coeff = cfg.line_jepa_semantic_anchor_coeff
        self.semantic_relation_coeff = cfg.line_jepa_semantic_relation_coeff
        self.semantic_relation_samples = cfg.line_jepa_semantic_relation_samples
        self.prototype_coeff = cfg.line_jepa_prototype_coeff
        self.prototype_temperature = cfg.line_jepa_prototype_temperature
        self.prototype_sinkhorn_epsilon = cfg.line_jepa_prototype_sinkhorn_epsilon
        self.prototype_sinkhorn_iterations = cfg.line_jepa_prototype_sinkhorn_iterations
        self.prototype_counts = tuple(cfg.line_jepa_prototype_counts)
        self.encoder_freeze_epochs = cfg.line_jepa_encoder_freeze_epochs
        self.freeze_encoder = cfg.line_jepa_freeze_encoder
        self.freeze_semantic_projector = cfg.line_jepa_freeze_semantic_projector
        self.encoder_lr_scale = cfg.line_jepa_encoder_lr_scale
        for name, value in (
            ("line_jepa_semantic_anchor_coeff", self.semantic_anchor_coeff),
            ("line_jepa_semantic_relation_coeff", self.semantic_relation_coeff),
            ("line_jepa_prototype_coeff", self.prototype_coeff),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}.")
        if self.semantic_relation_samples < 2:
            raise ValueError(
                "line_jepa_semantic_relation_samples must be >= 2, "
                f"got {self.semantic_relation_samples}."
            )
        if not self.prototype_counts or any(count < 2 for count in self.prototype_counts):
            raise ValueError(
                "line_jepa_prototype_counts must contain cluster counts >= 2, "
                f"got {self.prototype_counts}."
            )
        if len(set(self.prototype_counts)) != len(self.prototype_counts):
            raise ValueError(
                "line_jepa_prototype_counts must not contain duplicates, "
                f"got {self.prototype_counts}."
            )
        if self.prototype_temperature <= 0.0:
            raise ValueError(
                "line_jepa_prototype_temperature must be > 0, "
                f"got {self.prototype_temperature}."
            )
        if self.prototype_sinkhorn_epsilon <= 0.0:
            raise ValueError(
                "line_jepa_prototype_sinkhorn_epsilon must be > 0, "
                f"got {self.prototype_sinkhorn_epsilon}."
            )
        if self.prototype_sinkhorn_iterations < 1:
            raise ValueError(
                "line_jepa_prototype_sinkhorn_iterations must be >= 1, "
                f"got {self.prototype_sinkhorn_iterations}."
            )
        if self.encoder_freeze_epochs < 0:
            raise ValueError(
                "line_jepa_encoder_freeze_epochs must be >= 0, "
                f"got {self.encoder_freeze_epochs}."
            )
        if not (0.0 < self.encoder_lr_scale <= 1.0):
            raise ValueError(
                "line_jepa_encoder_lr_scale must be in (0, 1], "
                f"got {self.encoder_lr_scale}."
            )
        if self.target_encoder_mode == "online" and (
            self.semantic_anchor_coeff > 0.0
            or self.semantic_relation_coeff > 0.0
            or self.prototype_coeff > 0.0
        ):
            raise ValueError(
                "Compact-manifold preservation requires a separate EMA or frozen teacher. "
                "Set line_jepa_target_encoder to 'frozen' or 'ema', or disable semantic and "
                "prototype losses."
            )
        if self.line_jepa.prediction_coeff <= 0.0 and (
            self.semantic_anchor_coeff > 0.0
            or self.semantic_relation_coeff > 0.0
            or self.prototype_coeff > 0.0
        ):
            raise ValueError(
                "Compact-manifold and prototype losses require active Line-JEPA prediction "
                "tasks so online and teacher environments are aligned. Set "
                "line_jepa_prediction_coeff > 0 or disable the added losses."
            )

        self.semantic_projector = CompactSemanticProjector(
            input_dim=feature_dim,
            output_dim=self.semantic_dim,
        )
        if self.freeze_semantic_projector:
            for parameter in self.semantic_projector.parameters():
                parameter.requires_grad_(False)
            self.semantic_projector.eval()
        self.target_semantic_projector = copy.deepcopy(self.semantic_projector)
        for parameter in self.target_semantic_projector.parameters():
            parameter.requires_grad_(False)
        self.target_semantic_projector.eval()
        if self.prototype_coeff > 0.0:
            self.semantic_prototypes = nn.ParameterList(
                [
                    nn.Parameter(F.normalize(torch.randn(count, self.semantic_dim), dim=-1))
                    for count in self.prototype_counts
                ]
            )
        else:
            self.semantic_prototypes = nn.ParameterList()
        self.register_buffer(
            "semantic_prototypes_initialized",
            torch.tensor(self.prototype_coeff == 0.0),
            persistent=True,
        )

        self.regularizer_enabled = self.target_view_sim_coeff > 0.0 or any(
            coefficient > 0.0
            for coefficient in (
                self.line_jepa.sigreg_coeff,
                self.line_jepa.std_coeff,
                self.line_jepa.cov_coeff,
            )
        )
        self.regularizer_projector = (
            self._build_regularizer_projector(cfg, input_dim=feature_dim)
            if self.regularizer_enabled
            else nn.Identity()
        )
        hidden_dim = cfg.line_jepa_predictor_hidden_dim
        self.predictor = LineContextPredictor(
            input_dim=self.semantic_dim,
            hidden_dim=hidden_dim,
            depth=cfg.line_jepa_predictor_depth,
            num_heads=cfg.line_jepa_predictor_heads,
            mlp_ratio=cfg.line_jepa_predictor_mlp_ratio,
            dropout=cfg.line_jepa_predictor_dropout,
            directional_dim=directional_feature_dim,
            output_dim=self.semantic_dim,
        )
        self._encoder_frozen_state: bool | None = None
        # Lightning evaluates example_input_array in its model-summary callback before
        # the first training batch. The semantic PCA basis is deliberately initialized
        # from that first frozen-teacher batch, so a pre-fit forward pass is undefined.
        # Keep the runtime guard loud and disable only the optional summary forward.
        self.example_input_array = None

    def _encoder_extension(self) -> nn.Module:
        module = self.encoder
        while hasattr(module, "_orig_mod"):
            module = module._orig_mod
        return module

    @staticmethod
    def _build_regularizer_projector(cfg, *, input_dim: int) -> nn.Module:
        output_dim = cfg.line_jepa_regularizer_projector_dim
        hidden_dim = cfg.line_jepa_regularizer_projector_hidden_dim
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
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            EvalBatchStatsBatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            EvalBatchStatsBatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    @staticmethod
    @torch.no_grad()
    def _farthest_point_prototypes(
        normalized_features: torch.Tensor,
        *,
        count: int,
    ) -> torch.Tensor:
        if normalized_features.dim() != 2:
            raise ValueError(
                "Prototype initialization expects normalized features with shape (B, D), "
                f"got {tuple(normalized_features.shape)}."
            )
        count = int(count)
        if int(normalized_features.shape[0]) < count:
            raise ValueError(
                "Prototype initialization needs at least one sample per prototype, "
                f"got samples={int(normalized_features.shape[0])}, prototypes={count}."
            )
        selected = [0]
        best_similarity = normalized_features @ normalized_features[0]
        for _ in range(1, count):
            next_index = int(torch.argmin(best_similarity).item())
            selected.append(next_index)
            similarity = normalized_features @ normalized_features[next_index]
            best_similarity = torch.maximum(best_similarity, similarity)
        return normalized_features[selected]

    @torch.no_grad()
    def _initialize_semantic_space(self, teacher_features: torch.Tensor) -> None:
        online_initialized = bool(self.semantic_projector.initialized.item())
        target_initialized = bool(self.target_semantic_projector.initialized.item())
        if online_initialized != target_initialized:
            raise RuntimeError(
                "Online and target semantic projector initialization states disagree: "
                f"online={online_initialized}, target={target_initialized}."
            )
        if online_initialized:
            if not bool(self.semantic_prototypes_initialized.item()):
                raise RuntimeError(
                    "Semantic projector is initialized but prototype initialization state is false."
                )
            return

        self.semantic_projector.initialize_from_teacher(teacher_features)
        self.target_semantic_projector.load_state_dict(
            self.semantic_projector.state_dict(), strict=True
        )
        self.target_semantic_projector.eval()
        for parameter in self.target_semantic_projector.parameters():
            parameter.requires_grad_(False)

        if self.prototype_coeff > 0.0:
            teacher_semantic = self.target_semantic_projector(teacher_features).detach().float()
            teacher_semantic = F.normalize(teacher_semantic, dim=-1, eps=1.0e-6)
            for parameter, count in zip(
                self.semantic_prototypes, self.prototype_counts, strict=True
            ):
                initial = self._farthest_point_prototypes(
                    teacher_semantic,
                    count=count,
                )
                parameter.copy_(initial.to(device=parameter.device, dtype=parameter.dtype))
            self.semantic_prototypes_initialized.fill_(True)
        initialized_prototypes = list(self.prototype_counts) if self.prototype_coeff > 0.0 else []
        self._status_print(
            "[line-jepa/manifold] Initialized compact semantic space from frozen-teacher "
            f"PCA: input_dim={int(teacher_features.shape[1])}, semantic_dim={self.semantic_dim}, "
            f"samples={int(teacher_features.shape[0])}, prototypes={initialized_prototypes}."
        )

    def _online_semantic_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.semantic_projector(features)

    @torch.no_grad()
    def _target_semantic_features(self, features: torch.Tensor) -> torch.Tensor:
        self.target_semantic_projector.eval()
        return self.target_semantic_projector(features)

    def _semantic_geometry_losses(
        self,
        *,
        online: torch.Tensor,
        teacher: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if online.shape != teacher.shape or online.dim() != 2:
            raise ValueError(
                "Semantic geometry losses expect aligned (B, D) tensors, "
                f"got online={tuple(online.shape)}, teacher={tuple(teacher.shape)}."
            )
        online_n = F.normalize(online.float(), dim=-1, eps=1.0e-6)
        teacher_n = F.normalize(teacher.detach().float(), dim=-1, eps=1.0e-6)
        anchor = (
            1.0 - F.cosine_similarity(online_n, teacher_n, dim=-1)
        ).clamp_min(0.0).mean()

        sample_count = min(int(online_n.shape[0]), self.semantic_relation_samples)
        if sample_count < 2:
            raise ValueError(
                "Semantic relational preservation requires at least two samples, "
                f"got {sample_count}."
            )
        if sample_count < int(online_n.shape[0]):
            total_count = int(online_n.shape[0])
            indices = torch.div(
                torch.arange(sample_count, device=online_n.device, dtype=torch.long)
                * total_count,
                sample_count,
                rounding_mode="floor",
            )
            online_n = online_n.index_select(0, indices)
            teacher_n = teacher_n.index_select(0, indices)
        online_centered = F.normalize(
            online_n - online_n.mean(dim=0, keepdim=True), dim=-1, eps=1.0e-6
        )
        teacher_centered = F.normalize(
            teacher_n - teacher_n.mean(dim=0, keepdim=True), dim=-1, eps=1.0e-6
        )
        online_similarity = online_centered @ online_centered.T
        teacher_similarity = teacher_centered @ teacher_centered.T
        relation = F.mse_loss(online_similarity, teacher_similarity)
        return anchor, relation

    @staticmethod
    @torch.no_grad()
    def _balanced_sinkhorn(
        logits: torch.Tensor,
        *,
        epsilon: float,
        iterations: int,
    ) -> torch.Tensor:
        if logits.dim() != 2 or min(logits.shape) < 2:
            raise ValueError(
                "Balanced Sinkhorn expects logits with shape (B, K), B>=2, K>=2, "
                f"got {tuple(logits.shape)}."
            )
        scaled = logits.detach().float() / float(epsilon)
        scaled = scaled - scaled.max()
        assignments = torch.exp(scaled).T
        assignments = assignments / assignments.sum().clamp_min(1.0e-12)
        prototype_count, batch_size = assignments.shape
        for _ in range(int(iterations)):
            assignments = assignments / assignments.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
            assignments = assignments / float(prototype_count)
            assignments = assignments / assignments.sum(dim=0, keepdim=True).clamp_min(1.0e-12)
            assignments = assignments / float(batch_size)
        assignments = assignments * float(batch_size)
        if not torch.isfinite(assignments).all().item():
            raise RuntimeError("Balanced Sinkhorn produced non-finite assignments.")
        return assignments.T

    def _prototype_consistency_loss(
        self,
        *,
        online: torch.Tensor,
        teacher: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if online.shape != teacher.shape or online.dim() != 2:
            raise ValueError(
                "Prototype consistency expects aligned (B, D) semantic features, "
                f"got online={tuple(online.shape)}, teacher={tuple(teacher.shape)}."
            )
        if not bool(self.semantic_prototypes_initialized.item()):
            raise RuntimeError("Prototype consistency ran before prototype initialization.")
        online_n = F.normalize(online.float(), dim=-1, eps=1.0e-6)
        teacher_n = F.normalize(teacher.detach().float(), dim=-1, eps=1.0e-6)
        losses = []
        agreements = []
        usage_entropies = []
        for prototypes, count in zip(
            self.semantic_prototypes, self.prototype_counts, strict=True
        ):
            prototypes_n = F.normalize(prototypes.float(), dim=-1, eps=1.0e-6)
            teacher_logits = teacher_n @ prototypes_n.detach().T
            assignments = self._balanced_sinkhorn(
                teacher_logits,
                epsilon=self.prototype_sinkhorn_epsilon,
                iterations=self.prototype_sinkhorn_iterations,
            )
            online_logits = online_n @ prototypes_n.T / self.prototype_temperature
            losses.append(
                -(assignments * F.log_softmax(online_logits, dim=-1)).sum(dim=-1).mean()
            )
            online_probabilities = F.softmax(online_logits.detach(), dim=-1)
            usage = online_probabilities.mean(dim=0)
            usage_entropies.append(
                -(usage * usage.clamp_min(1.0e-12).log()).sum() / math.log(float(count))
            )
            agreements.append(
                (
                    online_logits.detach().argmax(dim=-1)
                    == assignments.argmax(dim=-1)
                ).float().mean()
            )
        return (
            torch.stack(losses).mean(),
            torch.stack(agreements).mean(),
            torch.stack(usage_entropies).mean(),
        )

    def load_pretrained_weights_from_checkpoint(self, checkpoint_path: str, *, strict: bool = False) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source_state = checkpoint["state_dict"]
        model_state = self.state_dict()
        compatible = {}
        shape_mismatch = []
        for key, value in source_state.items():
            target = model_state.get(key)
            if target is None:
                continue
            if tuple(target.shape) != tuple(value.shape):
                shape_mismatch.append(key)
                continue
            compatible[key] = value
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
        if self.masked_token_coeff > 0.0:
            self._encoder_extension().reset_mask_teacher_from_student()
        target_sync_message = "No separate target encoder is active."
        if self.target_encoder_mode in {"ema", "frozen"}:
            if self.target_encoder is None:
                raise RuntimeError(
                    "Line-JEPA separate target encoder is missing after checkpoint initialization. "
                    f"target_encoder_mode={self.target_encoder_mode!r}."
                )
            self.target_encoder.load_state_dict(self.encoder.state_dict(), strict=True)
            self.target_encoder.eval()
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            target_sync_message = f"Synced {self.target_encoder_mode} target encoder to loaded encoder."
        if self.target_encoder_mode == "frozen":
            self._frozen_target_initialized = True
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
        return self._encoder_extension().invariant_dim

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if self.target_encoder_mode == "frozen":
            state = checkpoint["state_dict"]
            if not any(str(key).startswith("target_encoder.") for key in state):
                raise RuntimeError(
                    "A frozen-target Line-JEPA checkpoint has no target_encoder tensors. "
                    "The checkpoint is incomplete or incompatible."
                )
            if not any(str(key).startswith("target_semantic_projector.") for key in state):
                raise RuntimeError(
                    "A compact-manifold Line-JEPA checkpoint has no target semantic projector "
                    "tensors. The checkpoint is incomplete or predates the compact manifold."
                )
            self._frozen_target_initialized = True

    def on_fit_start(self) -> None:
        if self.target_encoder_mode == "frozen" and not self._frozen_target_initialized:
            raise RuntimeError(
                "line_jepa_target_encoder='frozen' requires init_from_checkpoint for fresh "
                "training, or resume_from_checkpoint for an existing frozen-target run. "
                "A randomly initialized frozen teacher is not meaningful."
            )

    def configure_optimizers(self):
        encoder_parameter_ids = {id(parameter) for parameter in self.encoder.parameters()}
        encoder_parameters = [
            parameter for parameter in self.encoder.parameters() if parameter.requires_grad
        ]
        other_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in encoder_parameter_ids
        ]
        if not encoder_parameters:
            raise RuntimeError("Line-JEPA optimizer found no trainable online encoder parameters.")
        if not other_parameters:
            raise RuntimeError("Line-JEPA optimizer found no trainable predictor/manifold parameters.")
        base_lr = float(self.hparams.learning_rate)
        parameter_groups = [
            {
                "params": encoder_parameters,
                "lr": base_lr * self.encoder_lr_scale,
                "name": "encoder",
            },
            {
                "params": other_parameters,
                "lr": base_lr,
                "name": "predictor_manifold",
            },
        ]
        return get_optimizers_and_scheduler(self.hparams, parameter_groups)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        freeze_encoder = self.freeze_encoder or (
            int(self.current_epoch) < self.encoder_freeze_epochs
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(not freeze_encoder)
        if self.encoder_name == "GeoFrameTransformer":
            self._encoder_extension().enforce_frozen_teacher()
        if freeze_encoder:
            self.encoder.eval()
        if self.freeze_semantic_projector:
            self.semantic_projector.eval()
            for parameter in self.semantic_projector.parameters():
                parameter.requires_grad_(False)
        previous = self._encoder_frozen_state
        if previous is None or bool(previous) != freeze_encoder:
            freeze_reason = (
                "configured permanently"
                if self.freeze_encoder
                else f"warm start through epoch {self.encoder_freeze_epochs - 1}"
            )
            self._status_print(
                "[line-jepa/manifold] Online encoder "
                f"{'frozen' if freeze_encoder else 'unfrozen'} at epoch={int(self.current_epoch)}; "
                f"reason={freeze_reason}; encoder_lr_scale={self.encoder_lr_scale}."
            )
        self._encoder_frozen_state = freeze_encoder

    def _unpack_batch(self, batch):
        if self.data_kind in {"static", "line_static"}:
            return batch["points"], {
                "target_atom_id": batch["target_atom_id"],
                "anchor_atom_id": batch["anchor_atom_id"],
                "instance_id": batch["instance_id"],
                "coords": batch["coords"],
                "line_direction": batch["line_direction"],
                "source_group": batch["source_group"],
                "source_path": batch["source_path"],
            }
        if self.data_kind in {"temporal_lammps", "line_lammps"}:
            return batch["points"], {
                "target_atom_id": batch["target_atom_id"],
                "anchor_atom_id": batch["anchor_atom_id"],
                "instance_id": batch["instance_id"],
                "frame_indices": batch["frame_indices"],
                "timesteps": batch["timesteps"],
                "coords": batch["coords"],
                "line_direction": batch["line_direction"],
                "source_path": batch["source_path"],
            }
        raise RuntimeError(
            "LineJEPAModule received a batch for an unsupported repository data kind: "
            f"{self.data_kind!r}."
        )

    def _weighted_total_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        line_loss_keys = {
            key
            for key in losses
            if key == "line_jepa" or key.startswith("line_jepa_") or key.startswith("_line_jepa")
        }
        base_losses = {key: value for key, value in losses.items() if key not in line_loss_keys}
        total = super()._weighted_total_loss(base_losses)
        for key in sorted(line_loss_keys):
            total = total + losses[key]
        return total

    @classmethod
    def _jepa_metric_path(cls, stage: str, name: str) -> str:
        parts = str(name).split("/")
        if len(parts) < 3 or parts[0] != "jepa":
            raise ValueError(
                "Grouped Line-JEPA metrics must use the internal form "
                f"'jepa/<group>/<metric>', got {name!r}."
            )
        group = cls._WANDB_GROUPS.get(parts[1])
        if group is None:
            raise KeyError(
                f"Unknown Line-JEPA metric group {parts[1]!r}; "
                f"expected one of {sorted(cls._WANDB_GROUPS)}."
            )
        leaf = "_".join(parts[2:])
        return f"{group}/{stage}_{leaf}"

    def _log_jepa_metric(
        self,
        stage: str,
        name: str,
        value,
        *,
        batch_size: int | None = None,
        **kwargs,
    ) -> None:
        self._log_metric(
            stage,
            name,
            value,
            metric_name=self._jepa_metric_path(stage, name),
            batch_size=batch_size,
            **kwargs,
        )

    def _target_view_similarity_loss(
        self,
        first_view_features: torch.Tensor,
        second_view_features: torch.Tensor,
    ) -> torch.Tensor:
        if first_view_features.dim() != 2 or second_view_features.dim() != 2:
            raise ValueError(
                "Line-JEPA target-view similarity expects both feature tensors to have shape (B, D). "
                f"Got first={tuple(first_view_features.shape)}, second={tuple(second_view_features.shape)}."
            )
        if first_view_features.shape != second_view_features.shape:
            raise ValueError(
                "Line-JEPA target-view similarity feature shapes must match. "
                f"Got first={tuple(first_view_features.shape)}, second={tuple(second_view_features.shape)}."
            )
        sim_loss = F.mse_loss(first_view_features.float(), second_view_features.float())
        return sim_loss

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

    def _prediction_loss_inputs(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prediction_normalization == "none":
            return prediction, target
        if self.prediction_normalization == "l2":
            return (
                F.normalize(prediction.float(), dim=-1, eps=1.0e-6),
                F.normalize(target.float(), dim=-1, eps=1.0e-6),
            )
        raise RuntimeError(
            "Unsupported Line-JEPA prediction normalization at runtime: "
            f"{self.prediction_normalization!r}."
        )

    @staticmethod
    def _deterministic_context_shuffle_indices(
        *,
        batch_size: int,
        task_count: int,
        seed: int,
        source_groups: list[str] | None,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = int(batch_size)
        task_count = int(task_count)
        if batch_size <= 1:
            raise ValueError(
                "Line-JEPA shuffle controls require batch_size > 1, "
                f"got {batch_size}."
            )
        if task_count <= 0:
            raise ValueError(
                "Line-JEPA shuffle controls require task_count > 0, "
                f"got {task_count}."
            )
        if source_groups is not None and len(source_groups) != batch_size:
            raise ValueError(
                "Line-JEPA shuffle-control source groups must contain one value per sample. "
                f"Got groups={len(source_groups)}, batch_size={batch_size}."
            )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        row_count = batch_size * task_count
        indices = torch.arange(row_count, dtype=torch.long)
        if source_groups is None:
            group_to_samples = {"__all__": list(range(batch_size))}
        else:
            group_to_samples: dict[str, list[int]] = {}
            for sample_index, group in enumerate(source_groups):
                group_to_samples.setdefault(group, []).append(sample_index)

        for task_index in range(task_count):
            for sample_indices in group_to_samples.values():
                if len(sample_indices) <= 1:
                    continue
                rows = torch.tensor(
                    [sample_index * task_count + task_index for sample_index in sample_indices],
                    dtype=torch.long,
                )
                randomized_rows = rows[
                    torch.randperm(len(sample_indices), generator=generator)
                ]
                indices[randomized_rows] = randomized_rows.roll(1)
        return indices.to(device=device, non_blocking=True)

    def _log_prediction_shuffle_controls(
        self,
        *,
        stage: str,
        batch_idx: int,
        batch_size: int,
        task_count: int,
        context_features: torch.Tensor,
        context_positions: torch.Tensor,
        target_position: torch.Tensor,
        target: torch.Tensor,
        actual_prediction: torch.Tensor,
        source_groups,
        context_directional_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if stage == "train" or not self.shuffle_controls_enabled:
            return {}
        if source_groups is None:
            raise ValueError(
                "Line-JEPA within-source shuffle controls are enabled, but the batch has no "
                "source_group metadata. Add source_group to the dataset batch or disable "
                "line_jepa_shuffle_controls_enabled."
            )
        resolved_groups = source_groups
        base_seed = self.shuffle_control_seed + int(batch_idx) * 2
        permutations = {
            "global": self._deterministic_context_shuffle_indices(
                batch_size=batch_size,
                task_count=task_count,
                seed=base_seed,
                source_groups=None,
                device=context_features.device,
            ),
            "local": self._deterministic_context_shuffle_indices(
                batch_size=batch_size,
                task_count=task_count,
                seed=base_seed + 1,
                source_groups=resolved_groups,
                device=context_features.device,
            ),
        }

        target_float = target.detach().float()
        actual_error = torch.linalg.vector_norm(
            actual_prediction.detach().float() - target_float,
            dim=-1,
        ).mean()
        control_batch_size = batch_size * task_count
        logged = {"jepa/context/error": actual_error}
        self._log_jepa_metric(
            stage,
            "jepa/context/error",
            actual_error,
            batch_size=control_batch_size,
        )
        for control_name, permutation in permutations.items():
            shuffled_context = context_features.detach().index_select(0, permutation)
            if context_directional_features is None:
                shuffled_raw_prediction = self.predictor(
                    shuffled_context,
                    context_positions,
                    target_position,
                )
            else:
                shuffled_directional = context_directional_features.detach().index_select(
                    0, permutation
                )
                shuffled_raw_prediction = self.predictor(
                    shuffled_context,
                    context_positions,
                    target_position,
                    shuffled_directional,
                )
            shuffled_prediction, _ = self._prediction_loss_inputs(
                prediction=shuffled_raw_prediction,
                target=target,
            )
            shuffled_error = torch.linalg.vector_norm(
                shuffled_prediction.float() - target_float,
                dim=-1,
            ).mean()
            contextual_gain = (
                shuffled_error - actual_error
            ) / shuffled_error.clamp_min(1.0e-6)
            metric_name = f"jepa/context/{control_name}_gain"
            logged[metric_name] = contextual_gain
            self._log_jepa_metric(
                stage,
                metric_name,
                contextual_gain,
                batch_size=control_batch_size,
            )
        return logged

    def _prediction_hard_weights(
        self,
        *,
        novelty: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if (
            not self.prediction_hard_enabled
            or self.prediction_hard_weight_high == self.prediction_hard_weight_low
        ):
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
        weight_mean = weights.mean()
        if float(weight_mean.detach()) <= 0.0:
            raise RuntimeError(
                "Line-JEPA hard prediction weights have zero mean after top-k assignment. "
                f"low={self.prediction_hard_weight_low}, high={self.prediction_hard_weight_high}, "
                f"hard_count={hard_count}, task_count={task_count}."
            )
        weights = weights / weight_mean

        metrics = {
            "jepa/hard/mean_novelty": novelty.mean(),
            "jepa/hard/threshold": hard_values.min(),
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
        if not self.regularizer_enabled:
            raise RuntimeError(
                "Line-JEPA regularizer projection was requested while every regularizer "
                "and target-view coefficient is disabled."
            )
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

    def _encode_prepared_feature_blocks(
        self,
        blocks: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor] | None]]:
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
        feature_blocks = list(features.split(batch_sizes, dim=0))
        if self.directional_feature_mode != "encoder":
            return feature_blocks, [None] * len(blocks)

        geometry_state = encoded.aux[0]
        geometry_blocks: list[dict[str, torch.Tensor]] = [dict() for _ in blocks]
        for key, value in geometry_state.items():
            for block_state, block_value in zip(
                geometry_blocks,
                value.split(batch_sizes, dim=0),
                strict=True,
            ):
                block_state[key] = block_value
        return feature_blocks, geometry_blocks

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

    @torch.no_grad()
    def _encode_target_features(self, points: torch.Tensor) -> torch.Tensor:
        if self.target_encoder is None or self.target_encoder_io is None:
            raise RuntimeError(
                "Line-JEPA separate target encoding was requested, but the target encoder is not initialized. "
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

        online_semantic_params = dict(self.semantic_projector.named_parameters())
        target_semantic_params = dict(self.target_semantic_projector.named_parameters())
        if online_semantic_params.keys() != target_semantic_params.keys():
            raise RuntimeError(
                "Line-JEPA EMA semantic projector parameter names do not match."
            )
        for name, target_param in target_semantic_params.items():
            target_param.data.mul_(decay).add_(
                online_semantic_params[name].data, alpha=1.0 - decay
            )
        online_semantic_buffers = dict(self.semantic_projector.named_buffers())
        target_semantic_buffers = dict(self.target_semantic_projector.named_buffers())
        if online_semantic_buffers.keys() != target_semantic_buffers.keys():
            raise RuntimeError("Line-JEPA EMA semantic projector buffer names do not match.")
        for name, target_buffer in target_semantic_buffers.items():
            target_buffer.copy_(online_semantic_buffers[name])

    def _prediction_target_index_values(self, *, batch_idx: int) -> list[int]:
        if self.prediction_positions == "center":
            return [self.target_line_index]
        if self.prediction_positions == "all":
            return list(range(self.line_atoms))
        if self.prediction_positions == "cycle":
            return [int(batch_idx) % self.line_atoms]
        if self.prediction_positions == "endpoints":
            return [0, self.line_atoms - 1]
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

    def _ray_directions_for_targets(
        self,
        line_direction: torch.Tensor,
        *,
        target_indices: list[int],
        device,
        dtype,
    ) -> torch.Tensor:
        line_direction = line_direction.to(device=device, dtype=dtype, non_blocking=True)
        if line_direction.dim() != 2 or line_direction.shape[-1] != 3:
            raise ValueError(
                "Line-JEPA line_direction must have shape (B, 3), "
                f"got {tuple(line_direction.shape)}."
            )
        norms = torch.linalg.vector_norm(line_direction.float(), dim=-1, keepdim=True)
        if torch.any(norms <= 0.0).item():
            raise ValueError("Line-JEPA line_direction contains a zero-length direction.")
        normalized = line_direction / norms.to(dtype=line_direction.dtype)
        directions = []
        for target_index in target_indices:
            if int(target_index) == 0:
                directions.append(normalized)
            elif int(target_index) == self.line_atoms - 1:
                directions.append(-normalized)
            else:
                raise ValueError(
                    "A single ray direction is only defined for endpoint targets. "
                    f"Got target_index={target_index}, line_atoms={self.line_atoms}."
                )
        return torch.stack(directions, dim=1)

    @staticmethod
    def _directional_moment_features(
        context_points: torch.Tensor,
        ray_direction: torch.Tensor,
    ) -> torch.Tensor:
        """Rotation-consistent scalar contractions of local geometry with a query ray."""
        if context_points.dim() != 4 or context_points.shape[-1] != 3:
            raise ValueError(
                "Directional moments expect context_points with shape (B, C, P, 3), "
                f"got {tuple(context_points.shape)}."
            )
        if ray_direction.shape != (context_points.shape[0], 3):
            raise ValueError(
                "Directional moments expect one ray per task. "
                f"Got points={tuple(context_points.shape)}, ray={tuple(ray_direction.shape)}."
            )
        ray_norm = torch.linalg.vector_norm(ray_direction.float(), dim=-1, keepdim=True)
        if torch.any(ray_norm <= 0.0).item():
            raise ValueError("Directional moments received a zero-length ray direction.")
        direction = ray_direction / ray_norm.to(dtype=ray_direction.dtype)
        longitudinal = torch.einsum("bnpc,bc->bnp", context_points, direction)
        radius2 = torch.sum(context_points * context_points, dim=-1)
        longitudinal2 = longitudinal.square()
        transverse2 = torch.clamp_min(radius2 - longitudinal2, 0.0)
        features = torch.stack(
            (
                longitudinal.mean(dim=-1),
                longitudinal.abs().mean(dim=-1),
                longitudinal2.mean(dim=-1).clamp_min(0.0).sqrt(),
                longitudinal.pow(3).mean(dim=-1),
                radius2.mean(dim=-1),
                transverse2.mean(dim=-1),
                (3.0 * longitudinal2 - radius2).mean(dim=-1),
                torch.tanh(4.0 * longitudinal).mean(dim=-1),
            ),
            dim=-1,
        )
        if not torch.isfinite(features).all().item():
            raise RuntimeError(
                "Directional moment features became non-finite. "
                f"points_shape={tuple(context_points.shape)}."
            )
        return features

    def _encoder_directional_features_from_line_state(
        self,
        *,
        line_state: dict[str, torch.Tensor],
        batch_size: int,
        target_indices: list[int],
        task_ray_direction: torch.Tensor,
    ) -> torch.Tensor:
        task_count = len(target_indices)
        context_count = self.line_atoms - 1
        expected_tasks = batch_size * task_count
        if task_ray_direction.shape != (expected_tasks, 3):
            raise ValueError(
                "Encoder directional rays must align with endpoint prediction tasks. "
                f"Expected {(expected_tasks, 3)}, got {tuple(task_ray_direction.shape)}."
            )
        context_state = {}
        for key, value in line_state.items():
            if int(value.shape[0]) != batch_size * self.line_atoms:
                raise ValueError(
                    "Line encoder geometry state is not aligned with the flattened line batch. "
                    f"key={key!r}, state_shape={tuple(value.shape)}, "
                    f"batch_size={batch_size}, line_atoms={self.line_atoms}."
                )
            line_value = value.reshape(batch_size, self.line_atoms, *value.shape[1:])
            task_values = []
            for target_index in target_indices:
                context_indices = [
                    index for index in range(self.line_atoms) if index != int(target_index)
                ]
                task_values.append(line_value[:, context_indices])
            stacked = torch.stack(task_values, dim=1)
            context_state[key] = stacked.reshape(
                expected_tasks * context_count,
                *value.shape[1:],
            )

        repeated_rays = task_ray_direction.unsqueeze(1).expand(-1, context_count, -1).reshape(
            expected_tasks * context_count,
            3,
        )
        encoder_extension = self._encoder_extension()
        directional = encoder_extension.directional_features_from_geometry(
            context_state,
            repeated_rays,
        )
        expected_dim = int(encoder_extension.directional_feature_dim)
        if directional.shape != (expected_tasks * context_count, expected_dim):
            raise ValueError(
                "Encoder directional head returned an unexpected feature shape. "
                f"Expected {(expected_tasks * context_count, expected_dim)}, "
                f"got {tuple(directional.shape)}."
            )
        return directional.reshape(expected_tasks, context_count, expected_dim)

    def forward(self, pc: torch.Tensor, include_ssl_heads: bool = False):
        if pc.dim() == 4:
            pc = pc[:, self.target_line_index]
        pc = self._prepare_model_input(pc).to(device=self.device, dtype=self.dtype)
        encoded = self.encoder_io.encode(pc)
        z_inv_contrastive = self._contrastive_invariant_latent(
            encoded.invariant,
            encoded.equivariant,
        )
        z_semantic = self._online_semantic_features(z_inv_contrastive)
        if include_ssl_heads:
            return (
                z_semantic,
                encoded.invariant,
                encoded.equivariant,
                self._forward_ssl_heads_for_summary(z_inv_contrastive),
            )
        return z_semantic, encoded.invariant, encoded.equivariant

    @torch.inference_mode()
    def encode_directional_environment_batch(
        self,
        points: torch.Tensor,
        *,
        target_encoder: bool = False,
    ) -> torch.Tensor:
        """Encode reusable local environments without directional augmentation."""
        if points.dim() != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected environment points (B, P, 3), got {points.shape}.")
        points = points.to(self.device, dtype=self.dtype, non_blocking=True)
        if target_encoder and self.target_encoder_mode in {"ema", "frozen"}:
            raw_features = self._encode_target_features(points)
            return self._target_semantic_features(raw_features).float()
        if self.target_encoder_mode not in {"ema", "frozen", "online"}:
            raise RuntimeError(f"Unsupported target encoder {self.target_encoder_mode!r}.")
        raw_features = self._encode_features(points)
        return self._online_semantic_features(raw_features).float()

    @torch.inference_mode()
    def evaluate_directional_feature_batch(
        self,
        *,
        line_features: torch.Tensor,
        line_t: torch.Tensor,
        line_perp: torch.Tensor,
        target_features: torch.Tensor,
        target_index: int,
        line_points: torch.Tensor | None = None,
        line_direction: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the directional predictor on cached environment embeddings."""
        if line_features.dim() != 3 or line_features.shape[1] != self.line_atoms:
            raise ValueError(
                f"Expected line features (B, {self.line_atoms}, D), got {line_features.shape}."
            )
        if not 0 <= int(target_index) < self.line_atoms:
            raise IndexError(f"Target index {target_index} is outside the line.")
        line_features = line_features.to(self.device, dtype=self.dtype, non_blocking=True)
        target_features = target_features.to(self.device, dtype=self.dtype, non_blocking=True)
        context_indices = [i for i in range(self.line_atoms) if i != int(target_index)]
        context_features = line_features[:, context_indices]
        positions = self._line_positions_for_targets(
            {"line_t": line_t, "line_perp": line_perp},
            target_indices=[int(target_index)],
            device=self.device,
            dtype=line_features.dtype,
        )[:, 0]
        context_directional_features = None
        directional_mode = self.directional_feature_mode
        if directional_mode in {"moments", "encoder"}:
            if line_points is None or line_direction is None:
                raise ValueError(
                    "Directional evaluation requires line_points and line_direction."
                )
            if line_points.dim() != 4 or line_points.shape[:2] != line_features.shape[:2]:
                raise ValueError(
                    "Directional evaluation line_points must align with line_features. "
                    f"Got points={tuple(line_points.shape)}, features={tuple(line_features.shape)}."
                )
            batch_size, line_atoms, point_count, _ = line_points.shape
            prepared_points = self._prepare_model_input(
                line_points.to(self.device, dtype=self.dtype, non_blocking=True).reshape(
                    batch_size * line_atoms, point_count, 3
                )
            ).reshape(batch_size, line_atoms, -1, 3)
            ray_direction = self._ray_directions_for_targets(
                line_direction,
                target_indices=[int(target_index)],
                device=self.device,
                dtype=prepared_points.dtype,
            )[:, 0]
            if directional_mode == "moments":
                context_directional_features = self._directional_moment_features(
                    prepared_points[:, context_indices], ray_direction
                )
            else:
                encoded = self.encoder_io.encode(
                    prepared_points.reshape(batch_size * line_atoms, -1, 3)
                )
                line_state = encoded.aux[0]
                context_directional_features = (
                    self._encoder_directional_features_from_line_state(
                        line_state=line_state,
                        batch_size=batch_size,
                        target_indices=[int(target_index)],
                        task_ray_direction=ray_direction,
                    )
                )
        if context_directional_features is None:
            raw_prediction = self.predictor(
                context_features,
                positions[:, context_indices],
                positions[:, int(target_index)],
            )
        else:
            raw_prediction = self.predictor(
                context_features,
                positions[:, context_indices],
                positions[:, int(target_index)],
                context_directional_features,
            )
        raw_prediction_target, context_mean = self._prediction_target(
            target_features=target_features, context_features=context_features
        )
        prediction, prediction_target = self._prediction_loss_inputs(
            prediction=raw_prediction,
            target=raw_prediction_target,
        )
        prediction_float = prediction.float()
        target_float = prediction_target.float()
        context_float = context_mean.float()
        if self.prediction_target == "residual":
            true_residual = target_float - context_float
            predicted_residual = prediction_float
            reconstruction = context_float + predicted_residual
            error = torch.linalg.norm(predicted_residual - true_residual, dim=-1)
            baseline_error = torch.linalg.norm(true_residual, dim=-1)
        elif self.prediction_target == "target":
            reconstruction = prediction_float
            if self.prediction_normalization == "l2":
                context_baseline = F.normalize(context_float, dim=-1, eps=1.0e-6)
            else:
                context_baseline = context_float
            error = torch.linalg.norm(reconstruction - target_float, dim=-1)
            baseline_error = torch.linalg.norm(context_baseline - target_float, dim=-1)
        else:
            raise RuntimeError(f"Unsupported prediction target {self.prediction_target!r}.")
        return {
            "residual_prediction_error": error,
            "reconstruction_cosine_error": 1.0 - F.cosine_similarity(
                reconstruction, target_float, dim=-1
            ),
            "residual_norm": baseline_error,
            "relative_residual_error": error / baseline_error.clamp_min(1.0e-6),
            "reconstruction": reconstruction,
            "target_features": target_float,
            "line_online_features": line_features.float(),
        }

    @torch.inference_mode()
    def evaluate_directional_line_batch(
        self,
        batch: dict,
        *,
        target_index: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate deterministic Line-JEPA errors for explicit line contexts."""
        line_points = batch["points"]
        if line_points.dim() != 4 or line_points.shape[1] != self.line_atoms or line_points.shape[-1] != 3:
            raise ValueError(
                f"Expected directional points (B, {self.line_atoms}, P, 3), got {line_points.shape}."
            )
        resolved_target_index = self.target_line_index if target_index is None else int(target_index)
        if not 0 <= resolved_target_index < self.line_atoms:
            raise IndexError(f"Target index {resolved_target_index} is outside the line.")

        batch_size, _, points_per_structure, _ = line_points.shape
        line_features = self.encode_directional_environment_batch(
            line_points.reshape(batch_size * self.line_atoms, points_per_structure, 3)
        ).reshape(batch_size, self.line_atoms, -1)
        target_features = (
            self.encode_directional_environment_batch(
                line_points[:, resolved_target_index], target_encoder=True
            )
            if self.target_encoder_mode in {"ema", "frozen"}
            else line_features[:, resolved_target_index]
        )
        return self.evaluate_directional_feature_batch(
            line_features=line_features,
            line_t=batch["line_t"],
            line_perp=batch["line_perp"],
            target_features=target_features,
            target_index=resolved_target_index,
            line_points=line_points,
            line_direction=batch["line_direction"],
        )

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
        losses = {}
        current_epoch = int(self.current_epoch)
        run_line_jepa = self.line_jepa.should_run(current_epoch=current_epoch)
        run_vicreg = self.vicreg.should_run(current_epoch=current_epoch)
        cache_supervised_stage = self._should_cache_supervised_stage(stage)

        line_flat = None
        views = None
        online_blocks = []
        block_names = []
        line_points_device = None
        prepared_line_views = None
        target_index_values = []
        if run_line_jepa:
            line_points_device = line_points.to(device=self.device, dtype=self.dtype, non_blocking=True)
            if self.line_jepa.prediction_coeff > 0.0 or self.target_view_sim_coeff > 0.0:
                target_index_values = self._prediction_target_index_values(batch_idx=batch_idx)

        target_points = None
        if run_vicreg or cache_supervised_stage:
            if line_points_device is None:
                target_points = line_points[:, self.target_line_index].to(
                    device=self.device,
                    dtype=self.dtype,
                    non_blocking=True,
                )
            else:
                target_points = line_points_device[:, self.target_line_index]

        if run_line_jepa:
            if line_points_device is None:
                raise RuntimeError("Line-JEPA line-view branch did not move line_points to device.")
            line_flat = line_points_device.reshape(
                batch_size * self.line_atoms,
                points_per_structure,
                3,
            )
            prepared_line_views = self._prepare_line_jepa_view(line_flat)
            online_blocks.append(prepared_line_views)
            block_names.append("line_views")
            if self.target_view_sim_coeff > 0.0:
                consistency_points = line_points_device[:, target_index_values].reshape(
                    batch_size * len(target_index_values),
                    points_per_structure,
                    3,
                )
                online_blocks.append(
                    self._prepare_line_jepa_view(consistency_points)
                )
                block_names.append("target_view_extra")

        if run_vicreg:
            if target_points is None:
                raise RuntimeError("VICReg branch needs target_points, but they were not prepared.")
            views = self._build_contrastive_view_pair(
                target_points,
                view_points=self.vicreg.view_points,
            )

            online_blocks.extend([views["y_a"], views["y_b"]])
            block_names.extend(["vicreg_a", "vicreg_b"])

        encoded_blocks = {}
        encoded_geometry_blocks = {}
        if online_blocks:
            feature_blocks, geometry_blocks = self._encode_prepared_feature_blocks(online_blocks)
            for name, features, geometry in zip(
                block_names,
                feature_blocks,
                geometry_blocks,
                strict=True,
            ):
                encoded_blocks[name] = features
                encoded_geometry_blocks[name] = geometry

        if run_line_jepa and self.masked_token_coeff > 0.0:
            if prepared_line_views is None:
                raise RuntimeError(
                    "Masked-token Line-JEPA objective requires prepared line views."
                )
            center_views = prepared_line_views.reshape(
                batch_size,
                self.line_atoms,
                prepared_line_views.shape[1],
                3,
            )[:, self.target_line_index]
            mask_sample_count = min(batch_size, self.masked_token_samples)
            masked_token_loss = self._encoder_extension().masked_token_loss(
                center_views[:mask_sample_count]
            )
            losses["_line_jepa_masked_tokens"] = (
                self.line_jepa.weight * self.masked_token_coeff * masked_token_loss
            )
            self._log_jepa_metric(
                stage,
                "jepa/mask/loss",
                masked_token_loss,
                batch_size=mask_sample_count,
            )

        if run_line_jepa:
            run_prediction = self.line_jepa.prediction_coeff > 0.0
            prediction = None
            prediction_target = None
            prediction_weights = None
            hard_weight_metrics = {}
            prediction_batch_size = batch_size

            if run_prediction:
                if line_flat is None:
                    raise RuntimeError("Line-JEPA prediction branch did not build line_flat.")
                line_features_flat = encoded_blocks["line_views"]
                raw_feature_dim = int(line_features_flat.shape[-1])
                line_features = line_features_flat.reshape(
                    batch_size,
                    self.line_atoms,
                    raw_feature_dim,
                )
                prediction_task_count = len(target_index_values)
                line_positions = self._line_positions_for_targets(
                    batch,
                    target_indices=target_index_values,
                    device=self.device,
                    dtype=line_features.dtype,
                )
                context_feature_blocks = []
                context_point_blocks = (
                    [] if self.directional_feature_mode == "moments" else None
                )
                target_feature_blocks = []
                target_point_blocks = (
                    [] if self.target_encoder_mode in {"ema", "frozen"} else None
                )
                context_position_blocks = []
                target_position_blocks = []
                for task_idx, target_index in enumerate(target_index_values):
                    context_indices = [index for index in range(self.line_atoms) if index != int(target_index)]
                    context_feature_blocks.append(line_features[:, context_indices])
                    if context_point_blocks is not None:
                        if prepared_line_views is None:
                            raise RuntimeError(
                                "Directional moments require prepared line views, but none were built."
                            )
                        context_point_blocks.append(
                            prepared_line_views.reshape(
                                batch_size,
                                self.line_atoms,
                                prepared_line_views.shape[1],
                                3,
                            )[:, context_indices]
                        )
                    target_feature_blocks.append(line_features[:, int(target_index)])
                    if target_point_blocks is not None:
                        if line_points_device is None:
                            raise RuntimeError(
                                "Line-JEPA EMA prediction target branch needs line_points_device, "
                                "but it was not prepared."
                            )
                        target_point_blocks.append(line_points_device[:, int(target_index)])
                    context_position_blocks.append(line_positions[:, task_idx, context_indices])
                    target_position_blocks.append(line_positions[:, task_idx, int(target_index)])

                context_raw_features = torch.stack(context_feature_blocks, dim=1).reshape(
                    batch_size * prediction_task_count,
                    self.line_atoms - 1,
                    raw_feature_dim,
                )
                target_online_raw_features = torch.stack(target_feature_blocks, dim=1).reshape(
                    batch_size * prediction_task_count,
                    raw_feature_dim,
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
                context_directional_features = None
                if self.directional_feature_mode in {"moments", "encoder"}:
                    line_direction = meta["line_direction"]
                    task_ray_direction = self._ray_directions_for_targets(
                        line_direction,
                        target_indices=target_index_values,
                        device=self.device,
                        dtype=line_features.dtype,
                    ).reshape(batch_size * prediction_task_count, 3)
                    if self.directional_feature_mode == "moments":
                        if context_point_blocks is None:
                            raise RuntimeError(
                                "Directional moment context point blocks were not initialized."
                            )
                        context_task_points = torch.stack(context_point_blocks, dim=1).reshape(
                            batch_size * prediction_task_count,
                            self.line_atoms - 1,
                            prepared_line_views.shape[1],
                            3,
                        )
                        context_directional_features = self._directional_moment_features(
                            context_task_points,
                            task_ray_direction.to(dtype=context_task_points.dtype),
                        )
                    else:
                        line_state = encoded_geometry_blocks["line_views"]
                        context_directional_features = (
                            self._encoder_directional_features_from_line_state(
                                line_state=line_state,
                                batch_size=batch_size,
                                target_indices=target_index_values,
                                task_ray_direction=task_ray_direction,
                            )
                        )
                if self.target_encoder_mode in {"ema", "frozen"}:
                    if target_point_blocks is None:
                        raise RuntimeError(
                            "Line-JEPA separate target encoder did not collect target point blocks."
                        )
                    target_task_points = torch.stack(target_point_blocks, dim=1).reshape(
                        batch_size * prediction_task_count,
                        points_per_structure,
                        3,
                    )
                    target_raw_features = self._encode_target_features(target_task_points)
                elif self.target_encoder_mode == "online":
                    target_raw_features = target_online_raw_features.detach()
                else:
                    raise RuntimeError(
                        f"Unsupported line_jepa_target_encoder at runtime: {self.target_encoder_mode!r}."
                    )

                self._initialize_semantic_space(target_raw_features)
                context_features = self._online_semantic_features(
                    context_raw_features.reshape(-1, raw_feature_dim)
                ).reshape(
                    batch_size * prediction_task_count,
                    self.line_atoms - 1,
                    self.semantic_dim,
                )
                target_online_features = self._online_semantic_features(
                    target_online_raw_features
                )
                target_features = self._target_semantic_features(target_raw_features)
                if context_directional_features is None:
                    raw_prediction = self.predictor(
                        context_features, context_positions, target_position
                    )
                else:
                    raw_prediction = self.predictor(
                        context_features,
                        context_positions,
                        target_position,
                        context_directional_features,
                    )
                raw_prediction_target, context_mean = self._prediction_target(
                    target_features=target_features,
                    context_features=context_features,
                )
                prediction, prediction_target = self._prediction_loss_inputs(
                    prediction=raw_prediction,
                    target=raw_prediction_target,
                )
                prediction_novelty = (
                    target_features.detach().float() - context_mean.detach().float()
                ).norm(dim=1)
                prediction_weights, hard_weight_metrics = self._prediction_hard_weights(
                    novelty=prediction_novelty,
                )
                prediction_batch_size = batch_size * prediction_task_count

            regularized_embeddings = {}
            if self.regularizer_enabled:
                line_features_flat = encoded_blocks["line_views"]
                regularized_embeddings["line_views"] = line_features_flat
                if self.target_view_sim_coeff > 0.0:
                    target_view_extra_features = encoded_blocks["target_view_extra"]
                    regularized_embeddings["target_view_extra"] = target_view_extra_features
                regularized_embeddings = self._project_regularized_embeddings(
                    regularized_embeddings
                )

            line_loss, metrics = self.line_jepa.compute_loss(
                prediction=prediction,
                target=prediction_target,
                regularized_embeddings=regularized_embeddings,
                current_epoch=current_epoch,
                global_step=int(self.global_step),
                prediction_weights=prediction_weights,
            )
            if line_loss is not None:
                losses["_line_jepa"] = line_loss
            for name, value in metrics.items():
                self._log_jepa_metric(stage, name, value, batch_size=prediction_batch_size)
            for name, value in hard_weight_metrics.items():
                self._log_jepa_metric(stage, name, value, batch_size=prediction_batch_size)
            manifold_metrics = {}
            if run_prediction and (
                self.semantic_anchor_coeff > 0.0 or self.semantic_relation_coeff > 0.0
            ):
                semantic_anchor, semantic_relation = self._semantic_geometry_losses(
                    online=target_online_features,
                    teacher=target_features,
                )
                manifold_loss = (
                    self.semantic_anchor_coeff * semantic_anchor
                    + self.semantic_relation_coeff * semantic_relation
                )
                losses["_line_jepa_manifold"] = self.line_jepa.weight * manifold_loss
                manifold_metrics.update(
                    {
                        "jepa/manifold/anchor": semantic_anchor,
                        "jepa/manifold/relation": semantic_relation,
                    }
                )
            if run_prediction and self.prototype_coeff > 0.0:
                prototype_loss, prototype_agreement, prototype_usage = (
                    self._prototype_consistency_loss(
                        online=target_online_features,
                        teacher=target_features,
                    )
                )
                losses["_line_jepa_prototypes"] = (
                    self.line_jepa.weight * self.prototype_coeff * prototype_loss
                )
                manifold_metrics.update(
                    {
                        "jepa/cluster/prototype_loss": prototype_loss,
                        "jepa/cluster/assignment_agreement": prototype_agreement,
                        "jepa/cluster/usage_entropy": prototype_usage,
                    }
                )
            for name, value in manifold_metrics.items():
                self._log_jepa_metric(
                    stage,
                    name,
                    value,
                    batch_size=prediction_batch_size,
                )
            if self.target_view_sim_coeff > 0.0:
                line_view_features = regularized_embeddings["line_views"]
                target_view_extra_features = regularized_embeddings["target_view_extra"]
                target_view_main_features = line_view_features.reshape(
                    batch_size,
                    self.line_atoms,
                    -1,
                )[:, target_index_values].reshape(
                    batch_size * len(target_index_values),
                    -1,
                )
                target_sim = self._target_view_similarity_loss(
                    target_view_main_features,
                    target_view_extra_features,
                )
                target_sim_weighted = self.target_view_sim_coeff * target_sim
                losses["_line_jepa_view"] = target_sim_weighted
                self._log_jepa_metric(
                    stage,
                    "jepa/view/loss",
                    target_sim_weighted,
                    batch_size=batch_size * len(target_index_values),
                )

            if run_prediction:
                with torch.no_grad():
                    pred_cos = None
                    if self.line_jepa.prediction_loss != "cosine":
                        pred_cos = F.cosine_similarity(
                            prediction.float(),
                            prediction_target.detach().float(),
                            dim=-1,
                        ).mean()
                        self._log_jepa_metric(
                            stage,
                            "jepa/pred/cos",
                            pred_cos,
                            batch_size=prediction_batch_size,
                        )
                    control_metrics = self._log_prediction_shuffle_controls(
                        stage=stage,
                        batch_idx=batch_idx,
                        batch_size=batch_size,
                        task_count=prediction_task_count,
                        context_features=context_features,
                        context_positions=context_positions,
                        target_position=target_position,
                        target=prediction_target,
                        actual_prediction=prediction,
                        source_groups=(
                            meta["source_group"]
                            if self.shuffle_controls_enabled
                            else None
                        ),
                        context_directional_features=context_directional_features,
                    )
                    residual_gain = None
                    if self.prediction_target == "residual":
                        target_float = target_features.detach().float()
                        context_float = context_mean.float()
                        pred_residual = prediction.float()
                        residual = target_float - context_float
                        residual_norm = residual.norm(dim=-1)
                        pred_error = (pred_residual - residual).norm(dim=-1)
                        residual_gain = (
                            (residual_norm - pred_error) / residual_norm.clamp_min(1e-6)
                        ).mean()
                        for metric_name, metric_value in {
                            "jepa/pred/residual_error": pred_error.mean(),
                            "jepa/pred/residual_gain": residual_gain,
                        }.items():
                            self._log_jepa_metric(
                                stage,
                                metric_name,
                                metric_value,
                                batch_size=prediction_batch_size,
                            )
                    if stage != "train" and batch_idx == 0:
                        status_values = [
                            f"[{stage}-jepa] epoch={self.current_epoch}",
                            f"pred={float(metrics['jepa/pred/loss'].detach()):.5f}",
                        ]
                        sim_loss = metrics.get("jepa/sim/loss")
                        sim_top1 = metrics.get("jepa/sim/top1")
                        if sim_loss is not None:
                            status_values.append(f"sim={float(sim_loss.detach()):.4f}")
                        if sim_top1 is not None:
                            status_values.append(f"top1={float(sim_top1.detach()):.3f}")
                        for name, label in (
                            ("jepa/context/global_gain", "global_gain"),
                            ("jepa/context/local_gain", "local_gain"),
                        ):
                            value = control_metrics.get(name)
                            if value is not None:
                                status_values.append(f"{label}={float(value.detach()):.3f}")
                        if pred_cos is not None:
                            status_values.append(f"pred_cos={float(pred_cos.detach()):.4f}")
                        if residual_gain is not None:
                            status_values.append(f"residual_gain={float(residual_gain.detach()):.3f}")
                        semantic_anchor = manifold_metrics.get("jepa/manifold/anchor")
                        prototype_agreement = manifold_metrics.get(
                            "jepa/cluster/assignment_agreement"
                        )
                        if semantic_anchor is not None:
                            status_values.append(
                                f"anchor={float(semantic_anchor.detach()):.4f}"
                            )
                        if prototype_agreement is not None:
                            status_values.append(
                                f"cluster_agree={float(prototype_agreement.detach()):.3f}"
                            )
                        status_values.append(f"tasks={prediction_task_count}")
                        self._status_print(" ".join(status_values))

        if run_vicreg:
            if views is None:
                raise RuntimeError("VICReg active branch did not build contrastive views.")
            z_a = encoded_blocks["vicreg_a"]
            z_b = encoded_blocks["vicreg_b"]
            vicreg_loss, vicreg_metrics = self.vicreg.compute_loss_from_features(
                z_a_feat=z_a,
                z_b_feat=z_b,
                current_epoch=current_epoch,
                overlap_target=(
                    views["overlap_target"]
                    if self.vicreg.requires_overlap_target
                    else None
                ),
            )
            if vicreg_loss is not None:
                losses[self.vicreg.metric_prefix] = vicreg_loss
            for name, value in vicreg_metrics.items():
                self._log_metric(stage, name, value, batch_size=batch_size)

        if cache_supervised_stage:
            if target_points is None:
                raise RuntimeError(
                    "Supervised cache stage needs target_points, but they were not prepared."
                )
            with torch.no_grad():
                z_inv = self._online_semantic_features(
                    self._encode_features(target_points)
                )
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
        if self.masked_token_coeff > 0.0:
            self._encoder_extension().update_mask_teacher()


__all__ = ["CompactSemanticProjector", "LineJEPAModule", "LineContextPredictor"]
