import os
import sys

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from src.models import EncoderAdapter, build_encoder, resolve_encoder_output_dim
from src.training_methods.contrastive_learning.supervised_cache import (
    cache_limit_for_stage,
    cache_supervised_batch,
    init_supervised_cache,
    log_supervised_metrics,
    reset_supervised_cache,
)
from src.training_methods.contrastive_learning.vicreg import VICRegLoss
from src.utils.model_summary import make_model_summary_point_cloud, resolve_model_summary_batch_size
from src.utils.pointcloud_ops import crop_to_num_points, shift_to_neighbor
from src.utils.training_utils import cached_sample_count, get_optimizers_and_scheduler


class SwAVLoss(nn.Module):
    """Swapped assignment prediction loss for temporal point-cloud views."""

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

        if self.weight < 0.0:
            raise ValueError(f"swav_weight must be >= 0, got {self.weight}.")
        if self.projection_dim <= 0:
            raise ValueError(f"swav_projection_dim must be > 0, got {self.projection_dim}.")
        if self.hidden_dim < 0:
            raise ValueError(f"swav_hidden_dim must be >= 0, got {self.hidden_dim}.")
        if self.num_prototypes <= 0:
            raise ValueError(f"swav_num_prototypes must be > 0, got {self.num_prototypes}.")
        if self.temperature <= 0.0:
            raise ValueError(f"swav_temperature must be > 0, got {self.temperature}.")
        if self.epsilon <= 0.0:
            raise ValueError(f"swav_epsilon must be > 0, got {self.epsilon}.")
        if self.sinkhorn_iterations <= 0:
            raise ValueError(
                f"swav_sinkhorn_iterations must be > 0, got {self.sinkhorn_iterations}."
            )
        valid_view_modes = {"center_adjacent", "center_prev_next", "adjacent"}
        if self.view_mode not in valid_view_modes:
            raise ValueError(
                f"swav_view_mode must be one of {sorted(valid_view_modes)}, got {self.view_mode!r}."
            )
        if self.view_points is not None and self.view_points <= 0:
            raise ValueError(f"swav_view_points must be > 0 when set, got {self.view_points}.")

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
            sinkhorn_iterations=int(getattr(cfg, "swav_sinkhorn_iterations", 3)),
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
    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"SwAV Sinkhorn expects logits with shape (B, K), got {tuple(logits.shape)}.")
        if int(logits.shape[1]) != self.num_prototypes:
            raise ValueError(
                "SwAV Sinkhorn prototype dimension mismatch: "
                f"expected K={self.num_prototypes}, got logits.shape={tuple(logits.shape)}."
            )

        scores = logits.detach().to(dtype=torch.float32) / self.epsilon
        max_score = scores.max()
        max_score = self._distributed_max(max_score)
        q = torch.exp(scores - max_score).t()

        # The max_score subtraction guarantees at least one entry is exp(0)=1,
        # so sum_q > 0 strictly. We rely on the clamp_min(1e-12) in the inner
        # loop for any row-wise degeneracy. Skipping the .item() read here is
        # worth ~1 CPU/GPU sync per SwAV view per step (#3 hot-path cleanup).
        sum_q = q.sum()
        sum_q = self._distributed_sum(sum_q)
        q = q / sum_q.clamp_min(1e-12)

        # q.shape[1] is a Python int (> 0 for any non-empty batch) and the
        # all-reduce across ranks of positive numbers is positive, so the old
        # `.item()` sanity check is dead code under normal operation.
        local_batch = q.new_tensor(float(q.shape[1]))
        global_batch = self._distributed_sum(local_batch)
        num_prototypes = q.new_tensor(float(q.shape[0]))

        for _ in range(self.sinkhorn_iterations):
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
        if len(view_features) < 2:
            raise ValueError(f"SwAV requires at least two views, got {len(view_features)}.")

        batch_size = int(view_features[0].shape[0])
        for view_idx, features in enumerate(view_features):
            if features.dim() != 2:
                raise ValueError(
                    f"SwAV view {view_idx} features must have shape (B, D), got {tuple(features.shape)}."
                )
            if int(features.shape[0]) != batch_size:
                raise ValueError(
                    "SwAV views must share a batch dimension, "
                    f"view0_batch={batch_size}, view{view_idx}_batch={int(features.shape[0])}."
                )
            if self.input_dim is not None and int(features.shape[1]) != self.input_dim:
                raise ValueError(
                    "SwAV feature dimension mismatch: "
                    f"expected D={self.input_dim}, got view{view_idx}.shape={tuple(features.shape)}."
                )

        self.normalize_prototypes()
        logits_by_view = [self._prototype_logits(features) for features in view_features]
        loss_terms = []
        assignment_max_probs = []
        assignment_entropies = []
        for assign_idx, assignment_logits in enumerate(logits_by_view):
            assignments = self._sinkhorn(assignment_logits)
            assignment_max_probs.append(assignments.max(dim=1).values.mean())
            assignment_entropy = -(assignments * assignments.clamp_min(1e-12).log()).sum(dim=1).mean()
            assignment_entropies.append(assignment_entropy)

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
        # Tensor-only nonfinite accounting avoids a per-step CPU sync; we log
        # the flag unconditionally (0.0 when finite, 1.0 otherwise) so
        # downstream dashboards still see the same metric name.
        nonfinite = (~torch.isfinite(loss)).to(dtype=loss.dtype)
        metrics["swav_nonfinite"] = nonfinite
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss, metrics

    def forward(self, features: torch.Tensor, *, profile_logits: bool = False) -> torch.Tensor:
        if not profile_logits:
            raise RuntimeError(
                "SwAVLoss.forward is reserved for PyTorch Lightning model-summary FLOP profiling. "
                "Use SwAVLoss.compute_loss(...) during training."
            )
        if features.dim() != 2:
            raise ValueError(
                "SwAV logits FLOP profiling expects feature tensor with shape (B, D), "
                f"got {tuple(features.shape)}."
            )
        return self._prototype_logits(features)


class TemporalSSLModule(pl.LightningModule):
    """Temporal self-supervised training on local-structure frame sequences."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        data_cfg = getattr(cfg, "data", None)
        if data_cfg is None:
            raise ValueError("TemporalSSLModule requires a data configuration.")
        data_kind = str(getattr(data_cfg, "kind", "")).strip().lower()
        if data_kind != "temporal_lammps":
            raise ValueError(
                "TemporalSSLModule requires data.kind='temporal_lammps', "
                f"got {data_kind!r}."
            )
        self.sequence_length = int(getattr(data_cfg, "sequence_length", 0))
        if self.sequence_length < 3:
            raise ValueError(
                "TemporalSSLModule requires data.sequence_length >= 3 so previous and next frames exist, "
                f"got {self.sequence_length}."
            )
        self.center_frame_index = self.sequence_length // 2
        self.use_temporal_vicreg_views = bool(getattr(cfg, "temporal_vicreg_use_adjacent_views", True))
        self.temporal_vicreg_anchor_frame = str(
            getattr(cfg, "temporal_vicreg_anchor_frame", "center")
        ).strip().lower()
        if self.temporal_vicreg_anchor_frame not in {"center", "last"}:
            raise ValueError(
                "temporal_vicreg_anchor_frame must be one of {'center', 'last'}, "
                f"got {self.temporal_vicreg_anchor_frame!r}."
            )
        if self.use_temporal_vicreg_views and self.temporal_vicreg_anchor_frame != "center":
            raise ValueError(
                "temporal_vicreg_anchor_frame='last' requires temporal_vicreg_use_adjacent_views=false. "
                "Adjacent temporal VICReg views are currently defined around the center frame only."
            )
        # Fused-forward view anchor (#2). When the fused VICReg+SwAV path is
        # active we build just two views per step and reuse them for both
        # losses. "center" keeps the existing semantics (anchor=center,
        # pair=random(prev, next)); "previous" drops the center view entirely
        # and uses (anchor=prev, pair=next) for both heads, saving the
        # center crop and trading it for a harder temporal-only contrast.
        self.fused_views_anchor = str(
            getattr(cfg, "fused_views_anchor", "center")
        ).strip().lower()
        if self.fused_views_anchor not in {"center", "previous"}:
            raise ValueError(
                "fused_views_anchor must be one of {'center', 'previous'}, "
                f"got {self.fused_views_anchor!r}."
            )

        self.encoder = build_encoder(cfg)
        # Resolve latent dim from the uncompiled module (cheap attribute read).
        latent_dim = resolve_encoder_output_dim(cfg, encoder=self.encoder)
        # Optional torch.compile wrap around the encoder (#8). Inductor fuses
        # the patch-encoder Conv1d + BN + ReLU stack and the attention/MLP
        # blocks into larger kernels and removes Python dispatch overhead.
        # The existing `torch.compiler.is_compiling()` guard inside
        # `_normalize_frame_vectors` handles frame-validation graph breaks.
        # Defaults: `compile_encoder=False` so CPU smoke tests + existing
        # runs stay bitwise identical; opt in via the config. First step is
        # slower than usual because of compile warm-up.
        self._compile_encoder = bool(getattr(cfg, "compile_encoder", False))
        self._encoder_compile_mode = str(getattr(cfg, "encoder_compile_mode", "default"))
        self._encoder_compile_fullgraph = bool(getattr(cfg, "encoder_compile_fullgraph", False))
        self._encoder_compile_dynamic = bool(getattr(cfg, "encoder_compile_dynamic", False))
        if self._compile_encoder:
            self.encoder = torch.compile(
                self.encoder,
                mode=self._encoder_compile_mode,
                fullgraph=self._encoder_compile_fullgraph,
                dynamic=self._encoder_compile_dynamic,
            )
        self.encoder_io = EncoderAdapter(self.encoder)

        self.sample_points = int(getattr(data_cfg, "num_points", 0))
        model_points = getattr(data_cfg, "model_points", None)
        if model_points is None:
            model_points = getattr(cfg, "model_points", None)
        if model_points is not None:
            model_points = int(model_points)
            if model_points <= 0:
                model_points = None
        self.model_points = model_points

        if self.model_points is not None and self.sample_points and self.model_points > self.sample_points:
            raise ValueError(
                f"model_points ({self.model_points}) cannot exceed data.num_points ({self.sample_points})"
            )
        summary_points = self.model_points if self.model_points is not None else self.sample_points
        if summary_points <= 0:
            raise ValueError(
                "TemporalSSLModule cannot create a PyTorch Lightning FLOP summary input because "
                f"data.num_points={self.sample_points!r} and data.model_points={self.model_points!r}. "
                "Set data.num_points or data.model_points to a positive point count."
            )
        self.example_input_array = {
            "pc": make_model_summary_point_cloud(
                batch_size=resolve_model_summary_batch_size(cfg),
                num_points=summary_points,
                sequence_length=self.sequence_length,
            ),
            "include_ssl_heads": True,
        }

        self.vicreg = VICRegLoss.from_config(
            cfg,
            input_dim=latent_dim,
        )
        self.swav = SwAVLoss.from_config(
            cfg,
            input_dim=latent_dim,
        )

        init_supervised_cache(self, cfg)
        self.cache_train_supervised_metrics = bool(getattr(cfg, "cache_train_supervised_metrics", False))
        self._warned_cache_eq_fallback = False
        self._consecutive_nan_steps = 0
        self._max_consecutive_nan_steps = int(getattr(cfg, "max_consecutive_nan_steps", 20))
        # Device-side counter used to avoid a per-step `.item()` CPU/GPU sync
        # in the training hot path (#3). We only pull the value to CPU once
        # every `_nonfinite_check_stride` micro-steps; the stride is small
        # enough to still fail loud within O(stride) steps of a divergence.
        self._nonfinite_step_flag: torch.Tensor | None = None
        self._nonfinite_check_stride = max(1, int(getattr(cfg, "nonfinite_check_stride", 8)))

    @staticmethod
    def _load_checkpoint_payload(checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        try:
            return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location="cpu")

    @staticmethod
    def _extract_state_dict(checkpoint, *, checkpoint_path: str) -> dict:
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict")
            if isinstance(state_dict, dict):
                return state_dict
            model_state_dict = checkpoint.get("model_state_dict")
            if isinstance(model_state_dict, dict):
                return model_state_dict
            if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
                return checkpoint
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' does not contain a recognized state dictionary "
            "(`state_dict` or `model_state_dict`)."
        )

    @staticmethod
    def _strip_state_dict_prefixes(state_dict: dict) -> dict:
        prefixes = ("model.", "module.")
        normalized = {}
        for key, value in state_dict.items():
            new_key = key
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        changed = True
            normalized[new_key] = value
        return normalized

    @staticmethod
    def _compatible_state_dict(state_dict: dict, target_state: dict) -> tuple[dict, list[str]]:
        compatible = {}
        shape_mismatch = []
        for key, value in state_dict.items():
            target_value = target_state.get(key)
            if target_value is None:
                continue
            if tuple(target_value.shape) != tuple(value.shape):
                shape_mismatch.append(key)
                continue
            compatible[key] = value
        return compatible, shape_mismatch

    @staticmethod
    def _checkpoint_encoder_name(checkpoint) -> str | None:
        if not isinstance(checkpoint, dict):
            return None
        hparams = checkpoint.get("hyper_parameters")
        if hparams is None:
            return None
        encoder_cfg = getattr(hparams, "encoder", None)
        if encoder_cfg is None and isinstance(hparams, dict):
            encoder_cfg = hparams.get("encoder")
        if encoder_cfg is None:
            return None
        encoder_name = getattr(encoder_cfg, "name", None)
        if encoder_name is None and isinstance(encoder_cfg, dict):
            encoder_name = encoder_cfg.get("name")
        return None if encoder_name is None else str(encoder_name)

    def _resolve_init_checkpoint_target(self) -> str:
        if bool(getattr(self.hparams, "init_from_checkpoint_encoder_only", False)):
            return "encoder"
        raw_target = str(getattr(self.hparams, "init_from_checkpoint_target", "module")).strip().lower()
        if raw_target == "model":
            return "module"
        if raw_target not in {"module", "encoder"}:
            raise ValueError(
                "init_from_checkpoint_target must be one of {'module', 'model', 'encoder'}, "
                f"got {raw_target!r}."
            )
        return raw_target

    def _select_module_state_dict(self, source_state_dict: dict) -> tuple[dict, list[str], bool]:
        model_state = self.state_dict()
        raw_compatible, raw_shape_mismatch = self._compatible_state_dict(source_state_dict, model_state)
        stripped_state_dict = self._strip_state_dict_prefixes(source_state_dict)
        stripped_compatible, stripped_shape_mismatch = self._compatible_state_dict(
            stripped_state_dict,
            model_state,
        )
        use_stripped = len(stripped_compatible) > len(raw_compatible)
        if use_stripped:
            return stripped_compatible, stripped_shape_mismatch, True
        return raw_compatible, raw_shape_mismatch, False

    def _select_encoder_state_dict(self, source_state_dict: dict) -> tuple[str, dict, list[str]]:
        encoder_state = self.encoder.state_dict()
        normalized_state_dict = self._strip_state_dict_prefixes(source_state_dict)
        candidates = [("checkpoint", normalized_state_dict)]
        for prefix in ("encoder.", "backbone."):
            prefixed_state_dict = {
                key[len(prefix):]: value
                for key, value in normalized_state_dict.items()
                if key.startswith(prefix)
            }
            if prefixed_state_dict:
                candidates.append((f"checkpoint stripped '{prefix}'", prefixed_state_dict))

        best_label = candidates[0][0]
        best_compatible: dict = {}
        best_shape_mismatch: list[str] = []
        for label, candidate_state_dict in candidates:
            compatible, shape_mismatch = self._compatible_state_dict(candidate_state_dict, encoder_state)
            if len(compatible) > len(best_compatible):
                best_label = label
                best_compatible = compatible
                best_shape_mismatch = shape_mismatch
        return best_label, best_compatible, best_shape_mismatch

    def load_pretrained_weights_from_checkpoint(self, checkpoint_path: str, *, strict: bool = False) -> None:
        checkpoint = self._load_checkpoint_payload(checkpoint_path)
        source_state_dict = self._extract_state_dict(checkpoint, checkpoint_path=checkpoint_path)

        checkpoint_encoder_name = self._checkpoint_encoder_name(checkpoint)
        current_encoder_name = str(getattr(getattr(self.hparams, "encoder", None), "name", ""))
        if checkpoint_encoder_name and current_encoder_name and checkpoint_encoder_name != current_encoder_name:
            self._status_print(
                "[temporal-ssl] Warning: init checkpoint encoder differs from current config: "
                f"checkpoint={checkpoint_encoder_name}, current={current_encoder_name}. "
                "Only compatible tensors will be loaded."
            )

        target = self._resolve_init_checkpoint_target()
        if target == "module":
            selected_state_dict, shape_mismatch, used_stripped_prefixes = self._select_module_state_dict(
                source_state_dict
            )
            if not selected_state_dict:
                raise RuntimeError(
                    f"No compatible module tensors found when loading checkpoint '{checkpoint_path}'. "
                    "If this checkpoint only contains encoder weights, set "
                    "init_from_checkpoint_target='encoder'."
                )

            missing_keys, unexpected_keys = self.load_state_dict(selected_state_dict, strict=strict)
            model_state = self.state_dict()
            source_encoder_keys = sum(1 for key in source_state_dict if key.startswith("encoder."))
            model_encoder_keys = sum(1 for key in model_state if key.startswith("encoder."))
            loaded_encoder_keys = sum(1 for key in selected_state_dict if key.startswith("encoder."))

            self._status_print(f"[temporal-ssl] Initialized module weights from checkpoint: {checkpoint_path}")
            self._status_print(
                "[temporal-ssl] Checkpoint load summary: "
                f"target=module, "
                f"loaded={len(selected_state_dict)} / model_tensors={len(model_state)}, "
                f"shape_mismatch_skipped={len(shape_mismatch)}, "
                f"missing_after_load={len(missing_keys)}, "
                f"unexpected_after_load={len(unexpected_keys)}, "
                f"stripped_prefixes={used_stripped_prefixes}, "
                f"strict={strict}"
            )
            if source_encoder_keys > 0 and model_encoder_keys > 0 and loaded_encoder_keys == 0:
                self._status_print(
                    "[temporal-ssl] Warning: No encoder weights were loaded from the init checkpoint. "
                    "Current encoder remains randomly initialized."
                )
            return

        candidate_label, selected_state_dict, shape_mismatch = self._select_encoder_state_dict(source_state_dict)
        if not selected_state_dict:
            raise RuntimeError(
                f"No compatible encoder tensors found when loading checkpoint '{checkpoint_path}'. "
                "Expected either encoder-prefixed keys or a raw encoder state_dict with matching tensor shapes."
            )

        missing_keys, unexpected_keys = self.encoder.load_state_dict(selected_state_dict, strict=strict)
        encoder_state = self.encoder.state_dict()
        self._status_print(f"[temporal-ssl] Initialized encoder weights from checkpoint: {checkpoint_path}")
        self._status_print(
            "[temporal-ssl] Checkpoint load summary: "
            f"target=encoder, "
            f"source={candidate_label}, "
            f"loaded={len(selected_state_dict)} / encoder_tensors={len(encoder_state)}, "
            f"shape_mismatch_skipped={len(shape_mismatch)}, "
            f"missing_after_load={len(missing_keys)}, "
            f"unexpected_after_load={len(unexpected_keys)}, "
            f"strict={strict}"
        )

    @property
    def vicreg_projector(self):
        return self.vicreg.projector

    def _shared_invariant(self, z_inv_model, eq_z):
        # Contrastive training always prefers norms(eq_z) when eq_z exists and
        # otherwise falls back to the encoder invariant branch.
        return self.vicreg._invariant(z_inv_model, eq_z)

    def _forward_ssl_heads_for_summary(self, features: torch.Tensor | None) -> dict[str, torch.Tensor]:
        head_outputs = {}
        if self.vicreg.projector is not None:
            if features is None:
                raise RuntimeError(
                    "Cannot profile VICReg FLOPs for the Lightning model summary because "
                    "the encoder did not return invariant contrastive features."
                )
            head_outputs["vicreg_projected"] = self.vicreg(features, profile_projector=True)
        if self.swav.projector is not None or self.swav.prototypes is not None:
            if features is None:
                raise RuntimeError(
                    "Cannot profile SwAV FLOPs for the Lightning model summary because "
                    "the encoder did not return invariant contrastive features."
                )
            head_outputs["swav_logits"] = self.swav(features, profile_logits=True)
        return head_outputs

    def _status_print(self, message: str) -> None:
        if getattr(self, "_trainer", None) is not None:
            self.print(message)
            return
        print(message)

    def _prepare_model_input(self, pc: torch.Tensor) -> torch.Tensor:
        out = pc
        if self.model_points is not None:
            out = crop_to_num_points(out, self.model_points)
        return out

    def _center_frame(self, sequence_points: torch.Tensor) -> torch.Tensor:
        return sequence_points[:, self.center_frame_index]

    def _last_frame(self, sequence_points: torch.Tensor) -> torch.Tensor:
        return sequence_points[:, self.sequence_length - 1]

    def _vicreg_anchor_frame(self, sequence_points: torch.Tensor) -> torch.Tensor:
        if self.temporal_vicreg_anchor_frame == "center":
            return self._center_frame(sequence_points)
        if self.temporal_vicreg_anchor_frame == "last":
            return self._last_frame(sequence_points)
        raise RuntimeError(
            "Unsupported temporal VICReg anchor-frame mode resolved at runtime: "
            f"{self.temporal_vicreg_anchor_frame!r}."
        )

    def _validate_temporal_points(self, sequence_points: torch.Tensor) -> None:
        if not torch.is_tensor(sequence_points):
            raise TypeError(f"Temporal batch points must be a torch.Tensor, got {type(sequence_points)}")
        if sequence_points.dim() != 4 or sequence_points.shape[-1] != 3:
            raise ValueError(
                "TemporalSSLModule expects batch['points'] with shape (B, T, N, 3), "
                f"got {tuple(sequence_points.shape)}."
            )
        if int(sequence_points.shape[1]) != self.sequence_length:
            raise ValueError(
                "Temporal sequence length mismatch between config and batch. "
                f"Configured data.sequence_length={self.sequence_length}, "
                f"batch_sequence_length={int(sequence_points.shape[1])}."
            )

    def _temporal_meta_from_batch(self, batch: dict) -> dict:
        return {
            "class_id": batch.get("class_id"),
            "instance_id": batch.get("instance_id"),
            "rotation": batch.get("rotation"),
            "center_atom_id": batch.get("center_atom_id"),
            "anchor_frame_index": batch.get("anchor_frame_index"),
            "anchor_timestep": batch.get("anchor_timestep"),
            "frame_indices": batch.get("frame_indices"),
            "timesteps": batch.get("timesteps"),
            "source_path": batch.get("source_path"),
        }

    def _unpack_batch(self, batch):
        if not isinstance(batch, dict):
            raise TypeError(
                f"TemporalSSLModule expects dict batches from TemporalLAMMPSDumpDataset, got {type(batch)}."
            )
        if "points" not in batch:
            raise KeyError("Temporal batch is missing required key 'points'.")
        sequence_points = batch["points"]
        self._validate_temporal_points(sequence_points)
        return self._center_frame(sequence_points), self._temporal_meta_from_batch(batch)

    def _unpack_temporal_batch(self, batch):
        center_pc, meta = self._unpack_batch(batch)
        sequence_points = batch["points"]
        prev_pc = sequence_points[:, self.center_frame_index - 1]
        next_pc = sequence_points[:, self.center_frame_index + 1]
        return center_pc, prev_pc, next_pc, meta

    @torch.no_grad()
    def _extract_supervised_features_from_batch(self, batch):
        pc_raw, meta = self._unpack_batch(batch)
        class_id = meta.get("class_id")
        if class_id is None:
            return None, None

        pc_raw = pc_raw.to(device=self.device, dtype=self.dtype, non_blocking=True)
        pc = self._prepare_model_input(pc_raw)
        encoded = self.encoder_io.encode(pc)
        z_inv_contrastive = self._contrastive_invariant_latent(
            encoded.invariant,
            encoded.equivariant,
        )
        features = z_inv_contrastive if z_inv_contrastive is not None else encoded.invariant
        if features is None:
            return None, None
        return features.detach().to(torch.float32), class_id

    def _contrastive_invariant_latent(self, z_inv_model, eq_z):
        return self._shared_invariant(z_inv_model, eq_z)

    def _contrastive_invariant_from_eq_latent(
        self,
        eq_z,
        z_inv_model=None,
        *,
        stage: str | None = None,
    ):
        stage_name = stage if stage is not None else "unknown"
        if eq_z is not None:
            return self._shared_invariant(None, eq_z)
        if z_inv_model is not None and not self._warned_cache_eq_fallback:
            self._status_print(
                f"[temporal-ssl/cache] eq_z is missing at stage='{stage_name}'. "
                "Falling back to encoder invariant output (z_inv_model) for cached z_inv_contrastive."
            )
            self._warned_cache_eq_fallback = True
        return self._shared_invariant(z_inv_model, None)

    def _prepare_explicit_view(self, pc: torch.Tensor, *, target_points: int | None) -> torch.Tensor:
        return crop_to_num_points(pc, target_points)

    def _prepare_swav_view(self, pc: torch.Tensor) -> torch.Tensor:
        target_points = self.swav.view_points
        if target_points is not None:
            return self._prepare_explicit_view(pc, target_points=target_points)
        return self._prepare_model_input(pc)

    def _build_swav_temporal_views(
        self,
        center_pc: torch.Tensor,
        prev_pc: torch.Tensor,
        next_pc: torch.Tensor,
    ) -> list[torch.Tensor]:
        center_view = self._prepare_swav_view(center_pc)
        prev_view = self._prepare_swav_view(prev_pc)
        next_view = self._prepare_swav_view(next_pc)

        if self.swav.view_mode == "center_prev_next":
            return [center_view, prev_view, next_view]
        if self.swav.view_mode == "adjacent":
            return [prev_view, next_view]
        if self.swav.view_mode == "center_adjacent":
            batch_size = int(center_pc.shape[0])
            choose_next_mask = torch.rand((batch_size,), device=center_pc.device) < 0.5
            adjacent_view = torch.where(
                choose_next_mask.view(-1, 1, 1),
                next_view,
                prev_view,
            )
            return [center_view, adjacent_view]
        raise RuntimeError(f"Unsupported SwAV temporal view mode at runtime: {self.swav.view_mode!r}.")

    def _encode_swav_views(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(views) < 2:
            raise ValueError(f"SwAV requires at least two temporal views, got {len(views)}.")
        num_views = len(views)
        batch_size = int(views[0].shape[0])
        concatenated = torch.cat(views, dim=0)
        encoded = self.encoder_io.encode(concatenated)
        features = self._shared_invariant(encoded.invariant, encoded.equivariant)
        if features is None:
            raise RuntimeError(
                "SwAV requires invariant encoder features after temporal view encoding, "
                "but the configured encoder returned none."
            )
        if features.dim() != 2 or int(features.shape[0]) != batch_size * num_views:
            raise ValueError(
                "SwAV encoded temporal views must produce features with shape (B * V, D), "
                f"got {tuple(features.shape)} for batch_size={batch_size}, num_views={num_views}."
            )
        return list(features.chunk(num_views, dim=0))

    def _compute_swav_loss(
        self,
        *,
        center_pc: torch.Tensor,
        prev_pc: torch.Tensor,
        next_pc: torch.Tensor,
    ):
        views = self._build_swav_temporal_views(center_pc, prev_pc, next_pc)
        view_features = self._encode_swav_views(views)
        return self.swav.compute_loss(
            view_features=view_features,
            current_epoch=int(self.current_epoch),
        )

    def _fused_contrastive_applicable(
        self,
        *,
        should_run_vicreg: bool,
        should_run_swav: bool,
    ) -> bool:
        """Decide whether VICReg and SwAV can share a single encoder forward.

        The fused path (#1) requires:
          - both contrastive heads are active
          - temporal VICReg views are enabled
          - SwAV is in a 2-view mode (``center_adjacent`` or ``adjacent``)
          - view crop sizes match between VICReg and SwAV
          - VICReg neighbor_view_mode is one that ``_build_vicreg_temporal_views``
            already supports (``none`` or ``second``)
        Any unsupported combination falls back to the original separate-forward
        paths so the optimisation is a pure opt-in improvement for common
        configs.
        """
        if not (should_run_vicreg and should_run_swav):
            return False
        if not self.use_temporal_vicreg_views:
            return False
        if self.swav.view_mode not in {"center_adjacent", "adjacent"}:
            return False
        if self.vicreg.view_points != self.swav.view_points:
            return False
        neighbor_mode = str(self.vicreg.neighbor_view_mode).lower()
        if self.vicreg.neighbor_view and neighbor_mode not in {"none", "second"}:
            return False
        return True

    def _build_fused_contrastive_views(
        self,
        center_pc: torch.Tensor,
        prev_pc: torch.Tensor,
        next_pc: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build the two shared views consumed by VICReg and SwAV in the fused path.

        When ``fused_views_anchor == "center"`` this reproduces the two views
        built by :meth:`_build_vicreg_temporal_views` (anchor=center crop,
        pair=random adjacent crop optionally mixed with a spatial neighbour
        view). When ``fused_views_anchor == "previous"`` we drop the center
        view entirely (#2) and use (prev, next) as the two views.
        """
        target_points = self.vicreg.view_points
        neighbor_mode = str(self.vicreg.neighbor_view_mode).lower()

        if self.fused_views_anchor == "previous":
            # Temporal-only contrast: both views come from adjacent frames and
            # the center crop is never encoded. SwAV's view_mode is expected
            # to be 'adjacent' but we pass the views list explicitly so
            # either 'adjacent' or 'center_adjacent' would consume them
            # correctly from the caller's perspective.
            anchor_raw = self._prepare_explicit_view(prev_pc, target_points=target_points)
            pair_raw = self._prepare_explicit_view(next_pc, target_points=target_points)
            anchor_view = self.vicreg.apply_view_postprocessing(
                anchor_raw,
                use_neighbor=False,
                apply_occlusion=False,
            )
            pair_view = self.vicreg.apply_view_postprocessing(
                pair_raw,
                use_neighbor=False,
                apply_occlusion=False,
            )
            return {"anchor": anchor_view, "pair": pair_view}

        # Default anchor='center': reuse the VICReg temporal view builder.
        vicreg_views = self._build_vicreg_temporal_views(center_pc, prev_pc, next_pc)
        # Unused in this branch but referenced to keep linter happy.
        del neighbor_mode
        if vicreg_views is None:
            raise RuntimeError(
                "Fused contrastive path requested but _build_vicreg_temporal_views "
                "returned None; this should not happen with "
                "use_temporal_vicreg_views=True."
            )
        return {"anchor": vicreg_views["y_a"], "pair": vicreg_views["y_b"]}

    def _encode_fused_contrastive_views(
        self,
        views: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single encoder forward over ``[anchor ; pair]`` -> (anchor_feat, pair_feat)."""
        anchor_view = views["anchor"]
        pair_view = views["pair"]
        batch_size = int(anchor_view.shape[0])
        if int(pair_view.shape[0]) != batch_size:
            raise ValueError(
                "Fused contrastive anchor and pair views must share a batch dim; "
                f"got anchor_shape={tuple(anchor_view.shape)}, pair_shape={tuple(pair_view.shape)}."
            )
        fused_input = torch.cat([anchor_view, pair_view], dim=0)
        encoded = self.encoder_io.encode(fused_input)
        features = self._shared_invariant(encoded.invariant, encoded.equivariant)
        if features is None:
            raise RuntimeError(
                "Fused contrastive path requires invariant encoder features, "
                "but the configured encoder returned none."
            )
        if features.dim() != 2 or int(features.shape[0]) != 2 * batch_size:
            raise ValueError(
                "Fused contrastive encoded features must have shape (2*B, D); "
                f"got {tuple(features.shape)} for batch_size={batch_size}."
            )
        anchor_feat, pair_feat = features.chunk(2, dim=0)
        return anchor_feat, pair_feat

    def _build_vicreg_temporal_views(
        self,
        center_pc: torch.Tensor,
        prev_pc: torch.Tensor,
        next_pc: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        target_points = self.vicreg.view_points
        neighbor_mode = str(self.vicreg.neighbor_view_mode).lower()
        if self.vicreg.neighbor_view and neighbor_mode not in {"none", "second"}:
            raise ValueError(
                "Temporal VICReg mixed temporal/spatial neighbor sampling currently requires "
                "vicreg_neighbor_view_mode in {'none', 'second'} when "
                "temporal_vicreg_use_adjacent_views=true and vicreg_neighbor_view=true. "
                f"Got vicreg_neighbor_view_mode={neighbor_mode!r}."
            )
        apply_occlusion_a, apply_occlusion_b = self.vicreg._resolve_pair_occlusion_flags(
            use_neighbor_a=False,
            use_neighbor_b=bool(self.vicreg.neighbor_view and neighbor_mode != "none"),
            device=center_pc.device,
        )

        anchor_view = self._prepare_explicit_view(
            center_pc,
            target_points=target_points,
        )
        prev_view = self._prepare_explicit_view(
            prev_pc,
            target_points=target_points,
        )
        next_view = self._prepare_explicit_view(
            next_pc,
            target_points=target_points,
        )

        batch_size = int(center_pc.shape[0])
        choose_next_mask = torch.rand((batch_size,), device=center_pc.device) < 0.5
        temporal_view = torch.where(
            choose_next_mask.view(-1, 1, 1),
            next_view,
            prev_view,
        )
        anchor_view = self.vicreg.apply_view_postprocessing(
            anchor_view,
            use_neighbor=False,
            apply_occlusion=apply_occlusion_a,
        )
        temporal_view = self.vicreg.apply_view_postprocessing(
            temporal_view,
            use_neighbor=False,
            apply_occlusion=apply_occlusion_b,
        )

        if not self.vicreg.neighbor_view or neighbor_mode == "none":
            mixed_second_view = temporal_view
        else:
            spatial_view = self._prepare_explicit_view(
                shift_to_neighbor(
                    center_pc,
                    neighbor_k=self.vicreg.neighbor_k,
                    max_relative_distance=self.vicreg.neighbor_max_relative_distance,
                ),
                target_points=target_points,
            )
            spatial_view = self.vicreg.apply_view_postprocessing(
                spatial_view,
                use_neighbor=True,
                apply_occlusion=apply_occlusion_b,
            )
            temporal_mask = torch.rand((batch_size,), device=center_pc.device) < 0.5
            mixed_second_view = torch.where(
                temporal_mask.view(-1, 1, 1),
                temporal_view,
                spatial_view,
            )
        return {
            "y_a": anchor_view,
            "y_b": mixed_second_view,
        }

    def forward(self, pc: torch.Tensor, include_ssl_heads: bool = False):
        if not isinstance(include_ssl_heads, bool):
            raise TypeError(
                "include_ssl_heads must be a bool. It is intended only for PyTorch Lightning "
                f"model-summary FLOP profiling, got {type(include_ssl_heads)}."
            )
        if torch.is_tensor(pc) and pc.dim() == 4:
            self._validate_temporal_points(pc)
            pc = self._center_frame(pc)
        if torch.is_tensor(pc):
            pc = self._prepare_model_input(pc)
            pc = pc.to(device=self.device, dtype=self.dtype)
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
        if not isinstance(batch, dict):
            raise TypeError(
                f"TemporalSSLModule expects dict batches from TemporalLAMMPSDumpDataset, got {type(batch)}."
            )
        if "points" not in batch:
            raise KeyError("Temporal batch is missing required key 'points'.")

        sequence_points = batch["points"]
        self._validate_temporal_points(sequence_points)
        batch_size = int(sequence_points.shape[0])
        meta = self._temporal_meta_from_batch(batch)
        sequence_points = sequence_points.to(device=self.device, dtype=self.dtype, non_blocking=True)
        pc_raw = self._center_frame(sequence_points)
        prev_pc = sequence_points[:, self.center_frame_index - 1]
        next_pc = sequence_points[:, self.center_frame_index + 1]
        vicreg_pc_raw = self._vicreg_anchor_frame(sequence_points)
        pc = self._prepare_model_input(pc_raw)

        should_cache_stage = stage in self._supervised_cache and (
            stage != "train" or self.cache_train_supervised_metrics
        )
        should_run_vicreg = self.vicreg.should_run(current_epoch=int(self.current_epoch))
        should_run_swav = self.swav.should_run(current_epoch=int(self.current_epoch))

        losses = {}
        center_embeddings = None

        if should_cache_stage:
            with torch.no_grad():
                encoded = self.encoder_io.encode(pc)
                center_embeddings = self._contrastive_invariant_from_eq_latent(
                    encoded.equivariant,
                    z_inv_model=encoded.invariant,
                    stage=stage,
                )

        # Fused path (#1 / #2): when both contrastive heads are active with
        # compatible view configurations, build the two shared views once and
        # run a single encoder forward instead of two or three separate ones.
        use_fused = self._fused_contrastive_applicable(
            should_run_vicreg=should_run_vicreg,
            should_run_swav=should_run_swav,
        )

        if use_fused:
            fused_views = self._build_fused_contrastive_views(pc_raw, prev_pc, next_pc)
            anchor_feat, pair_feat = self._encode_fused_contrastive_views(fused_views)

            vicreg_loss, vicreg_metrics = self.vicreg.compute_loss_from_features(
                z_a_feat=anchor_feat,
                z_b_feat=pair_feat,
                current_epoch=int(self.current_epoch),
            )
            if vicreg_loss is not None:
                losses["vicreg"] = vicreg_loss
            for name, value in vicreg_metrics.items():
                self._log_metric(stage, name, value, batch_size=batch_size)

            swav_loss, swav_metrics = self.swav.compute_loss(
                view_features=[anchor_feat, pair_feat],
                current_epoch=int(self.current_epoch),
            )
            if swav_loss is not None:
                losses["swav"] = swav_loss
            for name, value in swav_metrics.items():
                self._log_metric(stage, name, value, batch_size=batch_size)
        else:
            if should_run_vicreg:
                vicreg_views = (
                    self._build_vicreg_temporal_views(pc_raw, prev_pc, next_pc)
                    if self.use_temporal_vicreg_views
                    else None
                )
                vicreg_loss, vicreg_metrics = self.vicreg.compute_loss(
                    pc=vicreg_pc_raw,
                    encoder=self.encoder,
                    prepare_input=self.encoder_io.prepare_input,
                    split_output=self.encoder_io.split_output,
                    current_epoch=int(self.current_epoch),
                    views=vicreg_views,
                    invariant_transform=self._shared_invariant,
                )
                if vicreg_loss is not None:
                    losses["vicreg"] = vicreg_loss
                for name, value in vicreg_metrics.items():
                    self._log_metric(stage, name, value, batch_size=batch_size)

            if should_run_swav:
                swav_loss, swav_metrics = self._compute_swav_loss(
                    center_pc=pc_raw,
                    prev_pc=prev_pc,
                    next_pc=next_pc,
                )
                if swav_loss is not None:
                    losses["swav"] = swav_loss
                for name, value in swav_metrics.items():
                    self._log_metric(stage, name, value, batch_size=batch_size)

        total_loss = None
        if "vicreg" in losses:
            vicreg_total = self.vicreg.weight * losses["vicreg"]
            total_loss = vicreg_total if total_loss is None else total_loss + vicreg_total
        if "swav" in losses:
            swav_total = self.swav.weight * losses["swav"]
            total_loss = swav_total if total_loss is None else total_loss + swav_total
        if total_loss is None:
            total_loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)

        if stage != "train" and batch_idx == 0:
            parts = [f"[{stage}-diag] epoch={self.current_epoch} batch_idx=0"]
            for k, v in losses.items():
                parts.append(f"{k}={v.item():.6f}")
            parts.append(f"total={total_loss.item():.6f}")
            parts.append(f"active_losses={list(losses.keys())}")
            self._status_print(" | ".join(parts))

        # Tensor-only consecutive nonfinite tracking (#3): no per-step sync.
        nonfinite_step = (~torch.isfinite(total_loss)).to(dtype=torch.int32)
        if self._nonfinite_step_flag is None or self._nonfinite_step_flag.device != total_loss.device:
            self._nonfinite_step_flag = torch.zeros((), dtype=torch.int32, device=total_loss.device)
        self._nonfinite_step_flag = torch.where(
            nonfinite_step > 0,
            self._nonfinite_step_flag + 1,
            torch.zeros_like(self._nonfinite_step_flag),
        )
        # Always clamp NaN/Inf so gradient accumulation does not poison the
        # next optimizer step.
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
        # Log a 0/1 flag instead of gating the log on a Python bool (no
        # sync, same metric name for downstream dashboards).
        self._log_metric(
            stage,
            "loss_nonfinite",
            nonfinite_step.to(dtype=torch.float32),
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        # Pull the counter to CPU only periodically. This still raises fast
        # enough (within O(stride) steps) to halt a diverged run.
        if stage == "train" and ((batch_idx + 1) % self._nonfinite_check_stride == 0):
            observed = int(self._nonfinite_step_flag.item())
            self._consecutive_nan_steps = observed
            if observed >= self._max_consecutive_nan_steps:
                raise RuntimeError(
                    f"Training produced {observed} consecutive non-finite losses "
                    f"(checked every {self._nonfinite_check_stride} steps). "
                    "Halting to prevent silent divergence."
                )

        metrics_to_log = {"loss": total_loss}
        if "vicreg" in losses:
            metrics_to_log["vicreg"] = losses["vicreg"]
        if "swav" in losses:
            metrics_to_log["swav"] = losses["swav"]

        prog_bar_keys = {"loss"}
        for name, value in metrics_to_log.items():
            self._log_metric(
                stage,
                name,
                value,
                prog_bar=(name in prog_bar_keys),
                batch_size=batch_size,
            )

        if should_cache_stage:
            limit = self._cache_limit_for_stage(stage)
            cache = self._supervised_cache.get(stage)
            already_cached = cached_sample_count(cache) if cache is not None else 0
            if limit is None or already_cached < limit:
                if center_embeddings is not None:
                    self._cache_supervised_batch(
                        stage,
                        center_embeddings,
                        meta,
                        encoder_features=center_embeddings,
                    )

        return total_loss

    def training_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return get_optimizers_and_scheduler(self.hparams, self.parameters())

    def on_after_backward(self) -> None:
        if self.swav.should_freeze_prototypes(global_step=int(self.global_step)):
            self.swav.clear_prototype_gradients()

    def _log_metric(
        self,
        stage: str,
        name: str,
        value,
        *,
        on_step=None,
        on_epoch=None,
        batch_size: int | None = None,
        **kwargs,
    ) -> None:
        if on_step is None:
            on_step = True
        if on_epoch is None:
            on_epoch = True
        if stage == "train":
            on_epoch = False
        elif stage == "val":
            on_step = False
        if not on_step and not on_epoch:
            return
        log_kwargs = dict(kwargs)
        if batch_size is not None and "batch_size" not in log_kwargs:
            log_kwargs["batch_size"] = int(batch_size)
        if "sync_dist" not in log_kwargs:
            # Skip DDP all-reduce for per-step train metrics (#7): they are
            # noisy per-rank samples anyway and the all-reduce stalls the
            # training loop in multi-GPU runs. Epoch-reduced and validation
            # metrics keep sync_dist=True for correct aggregation.
            is_train_step_only = (stage == "train") and on_step and not on_epoch
            log_kwargs["sync_dist"] = not is_train_step_only
        if torch.is_tensor(value):
            value = value.detach()
        self.log(f"{stage}/{name}", value, on_step=on_step, on_epoch=on_epoch, **log_kwargs)

    def _handle_epoch_boundary(self, stage: str, is_start: bool):
        if is_start:
            self._reset_supervised_cache(stage)
        else:
            self._log_supervised_metrics(stage)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self._handle_epoch_boundary("train", True)

    def on_train_epoch_end(self) -> None:
        self._handle_epoch_boundary("train", False)
        super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self._handle_epoch_boundary("val", True)

    def on_validation_epoch_end(self) -> None:
        self._handle_epoch_boundary("val", False)
        super().on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self._handle_epoch_boundary("test", True)

    def on_test_epoch_end(self) -> None:
        self._handle_epoch_boundary("test", False)
        super().on_test_epoch_end()

    def _reset_supervised_cache(self, stage: str) -> None:
        reset_supervised_cache(self, stage)

    def _cache_limit_for_stage(self, stage: str):
        return cache_limit_for_stage(self, stage)

    def _cache_supervised_batch(
        self,
        stage: str,
        z_inv_contrastive: torch.Tensor,
        meta: dict,
        encoder_features: torch.Tensor | None = None,
    ) -> None:
        cache_supervised_batch(
            self,
            stage,
            z_inv_contrastive,
            meta,
            encoder_features=encoder_features,
        )

    def _log_supervised_metrics(self, stage: str) -> None:
        log_supervised_metrics(self, stage)


__all__ = ["SwAVLoss", "TemporalSSLModule"]
