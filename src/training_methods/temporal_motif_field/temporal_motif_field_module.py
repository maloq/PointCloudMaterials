from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from src.models import EncoderAdapter, resolve_encoder_output_dim
from src.training_methods.temporal_ssl.temporal_ssl_module import TemporalSSLModule
from src.utils.training_utils import cached_sample_count

from .bridge_mining import score_bridge_candidates, select_bridge_candidate_keys
from .losses import (
    binary_classification_loss,
    kl_from_logits_to_probs,
    masked_mean,
    masked_mse,
    prototype_separation_loss,
    prototype_usage_metrics,
    sample_entropy_loss,
    top1_accuracy_from_logits,
    usage_balance_loss,
)
from .prototype_heads import SoftPrototypeBank
from .temporal_heads import MotifFieldHead, MultiLagFuturePredictor, TemporalContextModel
from .utils import build_mlp, build_sample_keys_from_batch, resolve_lag_weights


class TemporalMotifFieldModule(TemporalSSLModule):
    """End-to-end temporal motif field training on temporal LAMMPS neighborhoods."""

    def __init__(self, cfg):
        super().__init__(cfg)

        tmf_cfg = getattr(cfg, "tmf", None)
        if tmf_cfg is None:
            raise ValueError("TemporalMotifFieldModule requires a tmf configuration section.")

        self.tmf_cfg = tmf_cfg
        self.anchor_policy = str(getattr(tmf_cfg, "anchor_policy", "last_context_frame")).strip().lower()
        if self.anchor_policy != "last_context_frame":
            raise ValueError(
                "TemporalMotifFieldModule currently supports only "
                "tmf.anchor_policy='last_context_frame', "
                f"got {self.anchor_policy!r}."
            )

        self.context_frames = int(getattr(tmf_cfg, "context_frames", 0))
        self.lags = [int(lag) for lag in getattr(tmf_cfg, "lags", [])]
        if self.context_frames <= 0:
            raise ValueError(f"tmf.context_frames must be > 0, got {self.context_frames}.")
        if not self.lags:
            raise ValueError("tmf.lags must contain at least one positive lag.")
        if any(lag <= 0 for lag in self.lags):
            raise ValueError(f"tmf.lags must be positive integers, got {self.lags}.")

        self.anchor_index = self.context_frames - 1
        max_lag = max(self.lags)
        min_sequence_length = self.context_frames + max_lag
        if self.sequence_length < min_sequence_length:
            raise ValueError(
                "Temporal Motif Field requires data.sequence_length >= "
                "tmf.context_frames + max(tmf.lags). "
                f"Got data.sequence_length={self.sequence_length}, "
                f"tmf.context_frames={self.context_frames}, max_lag={max_lag}."
            )
        if self.anchor_index >= self.sequence_length:
            raise ValueError(
                "Anchor index must lie inside the sequence. "
                f"anchor_index={self.anchor_index}, sequence_length={self.sequence_length}."
            )

        self.use_eq_norms_for_invariant = bool(getattr(tmf_cfg, "use_eq_norms_for_invariant", True))
        self.stopgrad_targets = bool(getattr(tmf_cfg, "stopgrad_targets", True))
        self.teacher_ema_decay = float(getattr(tmf_cfg, "teacher_ema_decay", 0.996))
        if not (0.0 < self.teacher_ema_decay < 1.0):
            raise ValueError(
                "tmf.teacher_ema_decay must lie in (0, 1), "
                f"got {self.teacher_ema_decay}."
            )

        invariant_dim = (
            int(self.vicreg.invariant_head.output_dim)
            if getattr(self.vicreg, "invariant_head", None) is not None
            else resolve_encoder_output_dim(cfg, encoder=self.encoder)
        )
        if invariant_dim is None:
            raise ValueError("Unable to resolve encoder invariant dimension for TMF projector.")
        self.encoder_invariant_dim = int(invariant_dim)

        projector_cfg = getattr(tmf_cfg, "projector", None)
        if projector_cfg is None:
            raise ValueError("TemporalMotifFieldModule requires tmf.projector configuration.")
        self.motif_dim = int(getattr(projector_cfg, "output_dim", getattr(cfg, "latent_size", 0)))
        if self.motif_dim <= 0:
            raise ValueError(f"TMF projector output_dim must be > 0, got {self.motif_dim}.")
        self.projector = build_mlp(
            self.encoder_invariant_dim,
            int(getattr(projector_cfg, "hidden_dim", self.motif_dim)),
            self.motif_dim,
            num_layers=int(getattr(projector_cfg, "num_layers", 2)),
            dropout=float(getattr(projector_cfg, "dropout", 0.0)),
        )

        self.ema_teacher = nn.ModuleDict(
            {
                "encoder": deepcopy(self.encoder),
                "projector": deepcopy(self.projector),
            }
        )
        self.ema_teacher_io = EncoderAdapter(self.ema_teacher["encoder"])
        self._freeze_ema_teacher()

        stable_cfg = getattr(tmf_cfg, "stable", None)
        if stable_cfg is None or not bool(getattr(stable_cfg, "enabled", True)):
            raise ValueError("TMF stable prototype bank must be enabled for this implementation.")
        self.stable_num_prototypes = int(getattr(stable_cfg, "num_prototypes", 0))
        self.stable_head = SoftPrototypeBank(
            num_prototypes=self.stable_num_prototypes,
            dim=self.motif_dim,
            temperature=float(getattr(stable_cfg, "temperature", 0.1)),
        )
        self.stable_recon_weight = float(getattr(stable_cfg, "recon_weight", 1.0))
        self.stable_balance_weight = float(getattr(stable_cfg, "balance_weight", 0.05))
        self.stable_sample_entropy_weight = float(getattr(stable_cfg, "sample_entropy_weight", 0.0))

        residual_cfg = getattr(tmf_cfg, "residual", None)
        self.residual_enabled = bool(getattr(residual_cfg, "enabled", True))
        self.residual_dim = int(getattr(residual_cfg, "dim", 64))
        if self.residual_dim <= 0:
            raise ValueError(f"tmf.residual.dim must be > 0, got {self.residual_dim}.")
        self.residual_weight = float(getattr(residual_cfg, "weight", 0.0))
        self.residual_head = build_mlp(
            self.motif_dim,
            max(self.residual_dim, self.motif_dim // 2),
            self.residual_dim,
            num_layers=2,
            dropout=0.0,
        )
        self.residual_decoder = nn.Linear(self.residual_dim, self.motif_dim)

        temporal_cfg = getattr(tmf_cfg, "temporal", None)
        if temporal_cfg is None or not bool(getattr(temporal_cfg, "enabled", True)):
            raise ValueError("TMF temporal prediction must be enabled for this implementation.")
        temporal_hidden_dim = int(getattr(temporal_cfg, "hidden_dim", self.motif_dim))
        temporal_dropout = float(getattr(temporal_cfg, "dropout", 0.0))
        self.temporal_context = TemporalContextModel(
            input_dim=self.motif_dim,
            hidden_dim=temporal_hidden_dim,
            depth=int(getattr(temporal_cfg, "depth", 2)),
            num_heads=int(getattr(temporal_cfg, "num_heads", 8)),
            dropout=temporal_dropout,
            context_frames=self.context_frames,
            model_type=str(getattr(temporal_cfg, "model_type", "transformer")),
        )
        self.lag_weights = resolve_lag_weights(self.lags, getattr(temporal_cfg, "lag_weights", None))
        self.future_predictor = MultiLagFuturePredictor(
            context_dim=self.temporal_context.output_dim,
            hidden_dim=temporal_hidden_dim,
            stable_dim=self.stable_num_prototypes,
            residual_dim=self.residual_dim,
            lags=self.lags,
            dropout=temporal_dropout,
        )
        self.future_stable_weight = float(getattr(temporal_cfg, "future_stable_weight", 1.0))
        self.future_residual_weight = float(getattr(temporal_cfg, "future_residual_weight", 0.0))

        hazard_cfg = getattr(tmf_cfg, "hazard", None)
        if hazard_cfg is None or not bool(getattr(hazard_cfg, "enabled", True)):
            raise ValueError("TMF hazard head must be enabled for this implementation.")
        self.hazard_weight = float(getattr(hazard_cfg, "weight", 0.0))
        self.hazard_focal_gamma = float(getattr(hazard_cfg, "focal_gamma", 0.0))

        bridge_cfg = getattr(tmf_cfg, "bridge", None)
        self.bridge_enabled = bool(bridge_cfg is not None and getattr(bridge_cfg, "enabled", False))
        self.bridge_mining_enabled = bool(getattr(bridge_cfg, "mining_enabled", False)) if bridge_cfg is not None else False
        self.bridge_refresh_epochs = max(1, int(getattr(bridge_cfg, "refresh_epochs", 1))) if bridge_cfg is not None else 1
        self.bridge_candidate_fraction = float(getattr(bridge_cfg, "candidate_fraction", 0.0)) if bridge_cfg is not None else 0.0
        self.bridge_subset_max_samples = int(getattr(bridge_cfg, "train_subset_max_samples", 0)) if bridge_cfg is not None else 0
        self.bridge_weight = float(getattr(bridge_cfg, "weight", 0.0)) if bridge_cfg is not None else 0.0
        self.bridge_separation_weight = float(getattr(bridge_cfg, "separation_weight", 0.0)) if bridge_cfg is not None else 0.0
        self.bridge_separation_margin = float(getattr(bridge_cfg, "separation_margin", 0.0)) if bridge_cfg is not None else 0.0
        self.bridge_score_weights = (
            dict(getattr(bridge_cfg, "score_weights", {}))
            if bridge_cfg is not None and getattr(bridge_cfg, "score_weights", None) is not None
            else {}
        )
        if self.bridge_enabled:
            self.bridge_head = SoftPrototypeBank(
                num_prototypes=int(getattr(bridge_cfg, "num_prototypes", 0)),
                dim=self.motif_dim,
                temperature=float(getattr(bridge_cfg, "temperature", 0.12)),
            )
            self.bridge_context_proj = nn.Linear(self.temporal_context.output_dim, self.motif_dim)
        else:
            self.bridge_head = None
            self.bridge_context_proj = None

        field_cfg = getattr(tmf_cfg, "field", None)
        self.field_enabled = bool(field_cfg is not None and getattr(field_cfg, "enabled", False))
        self.field_weight = float(getattr(field_cfg, "weight", 0.0)) if field_cfg is not None else 0.0
        self.field_lag = int(getattr(field_cfg, "lag", self.lags[0])) if field_cfg is not None else self.lags[0]
        if self.field_enabled:
            if self.field_lag not in self.lags:
                raise ValueError(
                    "tmf.field.lag must be one of tmf.lags. "
                    f"Got tmf.field.lag={self.field_lag}, tmf.lags={self.lags}."
                )
            self.field_head = MotifFieldHead(
                context_dim=self.temporal_context.output_dim,
                stable_dim=self.stable_num_prototypes,
                residual_dim=self.residual_dim,
                hidden_dim=int(getattr(field_cfg, "hidden_dim", temporal_hidden_dim)),
                depth=int(getattr(field_cfg, "depth", 2)),
            )
        else:
            self.field_head = None

        self._bridge_candidate_keys: set[tuple[str, Any, Any]] = set()
        self._sync_ema_teacher_from_student()

        compile_cfg = getattr(cfg, "compile_encoder", False)
        if isinstance(compile_cfg, str):
            compile_mode = str(compile_cfg).strip()
            compile_enabled = compile_mode.lower() not in {"", "false", "off", "no"}
        elif isinstance(compile_cfg, bool):
            compile_enabled = compile_cfg
            compile_mode = "default"
        else:
            compile_enabled = bool(compile_cfg)
            compile_mode = "default"
        if compile_enabled and compile_mode.lower() in {"true", "1", "yes", "on"}:
            compile_mode = "default"

        if compile_enabled:
            compiled_encoder = torch.compile(self.encoder, mode=compile_mode, dynamic=False)
            compiled_teacher_encoder = torch.compile(
                self.ema_teacher["encoder"], mode=compile_mode, dynamic=False
            )
            compiled_modules = {
                "encoder": compiled_encoder,
                "teacher_encoder": compiled_teacher_encoder,
            }
            object.__setattr__(self, "_compiled_modules", compiled_modules)
            self.encoder_io = EncoderAdapter(compiled_encoder)
            self.ema_teacher_io = EncoderAdapter(compiled_teacher_encoder)

    def _freeze_ema_teacher(self) -> None:
        self.ema_teacher.eval()
        for param in self.ema_teacher.parameters():
            param.requires_grad_(False)

    def _sync_ema_teacher_from_student(self) -> None:
        self.ema_teacher["encoder"].load_state_dict(self.encoder.state_dict(), strict=True)
        self.ema_teacher["projector"].load_state_dict(self.projector.state_dict(), strict=True)
        self._freeze_ema_teacher()

    @torch.no_grad()
    def _update_ema_teacher(self) -> None:
        decay = float(self.teacher_ema_decay)
        for teacher_param, student_param in zip(
            self.ema_teacher["encoder"].parameters(),
            self.encoder.parameters(),
        ):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
        for teacher_param, student_param in zip(
            self.ema_teacher["projector"].parameters(),
            self.projector.parameters(),
        ):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)

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

    def _select_named_submodule_state_dict(
        self,
        source_state_dict: dict[str, torch.Tensor],
        *,
        module_name: str,
        target_state: dict[str, torch.Tensor],
    ) -> tuple[str, dict[str, torch.Tensor], list[str]]:
        normalized_state_dict = self._strip_state_dict_prefixes(source_state_dict)
        candidates: list[tuple[str, dict[str, torch.Tensor]]] = [("checkpoint", normalized_state_dict)]
        prefixed = {
            key[len(module_name) + 1 :]: value
            for key, value in normalized_state_dict.items()
            if key.startswith(f"{module_name}.")
        }
        if prefixed:
            candidates.append((f"checkpoint stripped '{module_name}.'", prefixed))

        best_label = candidates[0][0]
        best_state: dict[str, torch.Tensor] = {}
        best_shape_mismatch: list[str] = []
        for label, candidate in candidates:
            compatible, shape_mismatch = self._compatible_state_dict(candidate, target_state)
            if len(compatible) > len(best_state):
                best_label = label
                best_state = compatible
                best_shape_mismatch = shape_mismatch
        return best_label, best_state, best_shape_mismatch

    def load_pretrained_weights_from_checkpoint(self, checkpoint_path: str, *, strict: bool = False) -> None:
        checkpoint = self._load_checkpoint_payload(checkpoint_path)
        source_state_dict = self._extract_state_dict(checkpoint, checkpoint_path=checkpoint_path)

        checkpoint_encoder_name = self._checkpoint_encoder_name(checkpoint)
        current_encoder_name = str(getattr(getattr(self.hparams, "encoder", None), "name", ""))
        if checkpoint_encoder_name and current_encoder_name and checkpoint_encoder_name != current_encoder_name:
            self._status_print(
                "[tmf] Warning: init checkpoint encoder differs from current config: "
                f"checkpoint={checkpoint_encoder_name}, current={current_encoder_name}. "
                "Only compatible tensors will be loaded."
            )

        target = self._resolve_init_checkpoint_target()
        if target == "module":
            super().load_pretrained_weights_from_checkpoint(checkpoint_path, strict=strict)
            self._sync_ema_teacher_from_student()
            return

        encoder_label, encoder_state_dict, encoder_shape_mismatch = self._select_encoder_state_dict(source_state_dict)
        if not encoder_state_dict:
            raise RuntimeError(
                f"No compatible encoder tensors found when loading checkpoint '{checkpoint_path}'."
            )
        enc_missing, enc_unexpected = self.encoder.load_state_dict(encoder_state_dict, strict=strict)

        projector_label, projector_state_dict, projector_shape_mismatch = self._select_named_submodule_state_dict(
            source_state_dict,
            module_name="projector",
            target_state=self.projector.state_dict(),
        )
        if projector_state_dict:
            proj_missing, proj_unexpected = self.projector.load_state_dict(projector_state_dict, strict=False)
        else:
            proj_missing, proj_unexpected = list(self.projector.state_dict().keys()), []
            self._status_print(
                "[tmf] Warning: init checkpoint did not contain compatible TMF projector weights. "
                "The projector remains randomly initialized."
            )

        self._sync_ema_teacher_from_student()
        self._status_print(f"[tmf] Initialized encoder/projector weights from checkpoint: {checkpoint_path}")
        self._status_print(
            "[tmf] Checkpoint load summary: "
            f"target=encoder+projector, "
            f"encoder_source={encoder_label}, encoder_loaded={len(encoder_state_dict)}, "
            f"encoder_shape_mismatch_skipped={len(encoder_shape_mismatch)}, "
            f"encoder_missing_after_load={len(enc_missing)}, encoder_unexpected_after_load={len(enc_unexpected)}, "
            f"projector_source={projector_label}, projector_loaded={len(projector_state_dict)}, "
            f"projector_shape_mismatch_skipped={len(projector_shape_mismatch)}, "
            f"projector_missing_after_load={len(proj_missing)}, projector_unexpected_after_load={len(proj_unexpected)}, "
            f"strict={strict}"
        )

    def _validate_temporal_contract(self, batch: dict[str, Any], sequence_points: torch.Tensor) -> None:
        self._validate_temporal_points(sequence_points)
        if self.sequence_length != int(sequence_points.shape[1]):
            raise ValueError(
                "Temporal sequence length mismatch between config and batch. "
                f"Configured {self.sequence_length}, got {int(sequence_points.shape[1])}."
            )
        timesteps = batch.get("timesteps")
        if timesteps is None:
            return
        if not torch.is_tensor(timesteps):
            timesteps = torch.as_tensor(timesteps)
        if timesteps.ndim != 2 or int(timesteps.shape[1]) != self.sequence_length:
            raise ValueError(
                "Temporal batch timesteps must have shape (B, T) when provided. "
                f"Expected T={self.sequence_length}, got {tuple(timesteps.shape)}."
            )
        deltas = timesteps[:, 1:] - timesteps[:, :-1]
        if torch.any(deltas < 0):
            raise ValueError(
                "Temporal Motif Field requires chronological frame order, but timesteps decreased "
                f"within the batch. timesteps_shape={tuple(timesteps.shape)}."
            )

    def _anchor_frame(self, sequence_points: torch.Tensor) -> torch.Tensor:
        return sequence_points[:, self.anchor_index]

    def _context_frame_window(self, sequence_points: torch.Tensor) -> torch.Tensor:
        return sequence_points[:, : self.context_frames]

    def _resolve_frame_invariant(self, encoded_output) -> torch.Tensor:
        if encoded_output.equivariant is not None and self.use_eq_norms_for_invariant:
            invariant = self._shared_invariant(None, encoded_output.equivariant)
        elif encoded_output.invariant is not None:
            invariant = encoded_output.invariant
        elif encoded_output.equivariant is not None:
            invariant = self._shared_invariant(None, encoded_output.equivariant)
        else:
            raise RuntimeError("Encoder returned neither invariant nor equivariant latent.")

        if invariant is None:
            raise RuntimeError("Unable to resolve a usable invariant latent from the encoder output.")
        if invariant.ndim != 2 or int(invariant.shape[1]) != self.encoder_invariant_dim:
            raise RuntimeError(
                "Resolved invariant latent has an unexpected shape. "
                f"Expected (?, {self.encoder_invariant_dim}), got {tuple(invariant.shape)}."
            )
        return invariant

    def _project_invariant(self, invariant: torch.Tensor, *, use_teacher: bool) -> torch.Tensor:
        projector = self.ema_teacher["projector"] if use_teacher else self.projector
        proj_dtype = next(projector.parameters()).dtype
        return projector(invariant.to(dtype=proj_dtype))

    def _encode_frame_batch(
        self,
        points: torch.Tensor,
        *,
        use_teacher: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(
                "_encode_frame_batch expects points with shape (B, N, 3), "
                f"got {tuple(points.shape)}."
            )

        points = points.to(device=self.device, dtype=self.dtype, non_blocking=True)
        points = self._prepare_model_input(points)
        encoder_io = self.ema_teacher_io if use_teacher else self.encoder_io
        encoded = encoder_io.encode(points)
        invariant = self._resolve_frame_invariant(encoded)
        projected = self._project_invariant(invariant, use_teacher=use_teacher)
        return projected, invariant, encoded.invariant, encoded.equivariant

    def _encode_temporal_frame_sequence(
        self,
        sequence_points: torch.Tensor,
        *,
        use_teacher: bool = False,
        frame_indices: "list[int] | tuple[int, ...] | None" = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(sequence_points):
            raise TypeError(f"sequence_points must be a torch.Tensor, got {type(sequence_points)}.")
        if sequence_points.ndim != 4 or sequence_points.shape[-1] != 3:
            raise ValueError(
                "Temporal Motif Field expects temporal point clouds with shape (B, T, N, 3), "
                f"got {tuple(sequence_points.shape)}."
            )

        sequence_points = sequence_points.to(device=self.device, dtype=self.dtype, non_blocking=True)
        batch_size, num_frames, num_points, _ = sequence_points.shape

        if frame_indices is None:
            selected_points = sequence_points
        else:
            idx_list = [int(i) for i in frame_indices]
            if not idx_list:
                raise ValueError("frame_indices must be a non-empty sequence of frame positions.")
            if any(i < 0 or i >= num_frames for i in idx_list):
                raise ValueError(
                    f"frame_indices out of range for sequence_length={num_frames}: {idx_list}."
                )
            selected_points = sequence_points[:, idx_list]

        num_selected = int(selected_points.shape[1])
        flat_points = selected_points.reshape(batch_size * num_selected, num_points, 3).contiguous()
        flat_points = self._prepare_model_input(flat_points)
        encoder_io = self.ema_teacher_io if use_teacher else self.encoder_io
        encoded = encoder_io.encode(flat_points)
        invariant = self._resolve_frame_invariant(encoded)
        projected = self._project_invariant(invariant, use_teacher=use_teacher)
        return projected.reshape(batch_size, num_selected, -1)

    def _residual_features(self, z: torch.Tensor) -> torch.Tensor:
        residual = self.residual_head(z)
        if not self.residual_enabled:
            return torch.zeros_like(residual)
        return residual

    def _decode_residual(self, residual: torch.Tensor) -> torch.Tensor:
        decoded = self.residual_decoder(residual)
        if not self.residual_enabled:
            return torch.zeros_like(decoded)
        return decoded

    def _bridge_input(self, z_anchor: torch.Tensor, context_repr: torch.Tensor) -> torch.Tensor:
        if self.bridge_context_proj is None:
            return z_anchor
        return z_anchor + self.bridge_context_proj(context_repr)

    def _compute_teacher_targets(self, sequence_points: torch.Tensor) -> dict[str, Any]:
        with torch.no_grad():
            anchor = int(self.anchor_index)
            lag_to_target_frame: dict[int, int] = {}
            unique_frame_indices: list[int] = [anchor]
            for lag in self.lags:
                target_frame_index = anchor + int(lag)
                lag_to_target_frame[lag] = target_frame_index
                if target_frame_index not in unique_frame_indices:
                    unique_frame_indices.append(target_frame_index)
            position_of_frame = {frame_idx: pos for pos, frame_idx in enumerate(unique_frame_indices)}

            teacher_z_seq = self._encode_temporal_frame_sequence(
                sequence_points,
                use_teacher=True,
                frame_indices=unique_frame_indices,
            )
            batch_size, num_selected, latent_dim = teacher_z_seq.shape
            flat_z = teacher_z_seq.reshape(batch_size * num_selected, latent_dim)
            stable_output = self.stable_head(flat_z)
            residual_seq = self._residual_features(flat_z).reshape(batch_size, num_selected, -1)
            stable_probs_seq = stable_output.probs.reshape(batch_size, num_selected, -1)
            stable_logits_seq = stable_output.logits.reshape(batch_size, num_selected, -1)
            stable_recon_seq = stable_output.recon.reshape(batch_size, num_selected, -1)

            anchor_pos = position_of_frame[anchor]
            anchor_probs = stable_probs_seq[:, anchor_pos]
            anchor_logits = stable_logits_seq[:, anchor_pos]
            anchor_recon = stable_recon_seq[:, anchor_pos]
            anchor_residual = residual_seq[:, anchor_pos]

            target_stable_probs: dict[int, torch.Tensor] = {}
            target_residual: dict[int, torch.Tensor] = {}
            change_targets: dict[int, torch.Tensor] = {}
            target_frame_indices: dict[int, int] = {}

            anchor_ids = anchor_probs.argmax(dim=-1)
            stable_id_columns = [anchor_ids]
            for lag in self.lags:
                target_frame_index = lag_to_target_frame[lag]
                target_frame_indices[lag] = target_frame_index
                pos = position_of_frame[target_frame_index]
                target_probs = stable_probs_seq[:, pos]
                target_stable_probs[lag] = target_probs
                target_residual[lag] = residual_seq[:, pos]
                target_ids = target_probs.argmax(dim=-1)
                stable_id_columns.append(target_ids)
                change_targets[lag] = (target_ids != anchor_ids).to(dtype=torch.float32)

            stable_ids = torch.stack(stable_id_columns, dim=1)
            teacher_targets = {
                "z_seq": teacher_z_seq,
                "z_anchor": teacher_z_seq[:, anchor_pos],
                "anchor_stable_probs": anchor_probs,
                "anchor_stable_logits": anchor_logits,
                "anchor_stable_recon": anchor_recon,
                "anchor_residual": anchor_residual,
                "target_stable_probs": target_stable_probs,
                "target_residual": target_residual,
                "change_targets": change_targets,
                "stable_ids": stable_ids,
                "stable_confidence": anchor_probs.max(dim=-1).values,
                "target_frame_indices": target_frame_indices,
            }
            if not self.stopgrad_targets:
                return teacher_targets
            return self._detach_nested_dict(teacher_targets)

    @staticmethod
    def _detach_nested_dict(value):
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, dict):
            return {key: TemporalMotifFieldModule._detach_nested_dict(item) for key, item in value.items()}
        return value

    def _compute_field_logits(
        self,
        batch: dict[str, Any],
        *,
        context_repr: torch.Tensor,
        anchor_stable_probs: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.field_enabled or self.field_head is None:
            return None
        if "neighbor_points" not in batch or "neighbor_offsets" not in batch:
            raise ValueError(
                "tmf.field.enabled=true requires batch keys 'neighbor_points' and 'neighbor_offsets'."
            )

        neighbor_points = batch["neighbor_points"]
        neighbor_offsets = batch["neighbor_offsets"]
        if not torch.is_tensor(neighbor_points):
            neighbor_points = torch.as_tensor(neighbor_points)
        if not torch.is_tensor(neighbor_offsets):
            neighbor_offsets = torch.as_tensor(neighbor_offsets)
        if neighbor_points.ndim != 4 or neighbor_points.shape[-1] != 3:
            raise ValueError(
                "TMF field head expects neighbor_points with shape (R, T, N, 3), "
                f"got {tuple(neighbor_points.shape)}."
            )

        neighbor_points = neighbor_points.to(device=self.device, dtype=self.dtype, non_blocking=True)
        neighbor_offsets = neighbor_offsets.to(device=self.device, dtype=torch.long, non_blocking=True)
        neighbor_anchor_points = neighbor_points[:, self.anchor_index]
        neighbor_z, _, _, _ = self._encode_frame_batch(neighbor_anchor_points, use_teacher=False)
        neighbor_stable = self.stable_head(neighbor_z)
        neighbor_residual = self._residual_features(neighbor_z)
        return self.field_head(
            context_repr=context_repr,
            anchor_stable_probs=anchor_stable_probs,
            neighbor_stable_probs=neighbor_stable.probs,
            neighbor_residual=neighbor_residual,
            neighbor_offsets=neighbor_offsets,
        )

    def forward_sequence(self, batch: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(batch, dict):
            raise TypeError(f"forward_sequence expects a dict batch, got {type(batch)}.")
        if "points" not in batch:
            raise KeyError("Temporal batch is missing required key 'points'.")

        sequence_points = batch["points"]
        self._validate_temporal_contract(batch, sequence_points)
        sequence_points = sequence_points.to(device=self.device, dtype=self.dtype, non_blocking=True)

        student_frame_indices = list(range(self.context_frames))
        z_seq = self._encode_temporal_frame_sequence(
            sequence_points,
            use_teacher=False,
            frame_indices=student_frame_indices,
        )
        z_anchor = z_seq[:, self.anchor_index]
        stable_anchor = self.stable_head(z_anchor)
        residual_anchor = self._residual_features(z_anchor)
        context_repr = self.temporal_context(z_seq)
        future_stable_logits, future_residual_pred, hazard_logits = self.future_predictor(context_repr)

        bridge_output = None
        bridge_logits_anchor = None
        bridge_gate = None
        if self.bridge_head is not None:
            bridge_input = self._bridge_input(z_anchor, context_repr)
            bridge_output = self.bridge_head(bridge_input)
            bridge_logits_anchor = bridge_output.logits
            hazard_stack = torch.stack(
                [torch.sigmoid(hazard_logits[lag]) for lag in self.lags],
                dim=0,
            )
            bridge_gate = hazard_stack.mean(dim=0)

        teacher_targets = self._compute_teacher_targets(sequence_points)
        field_logits = self._compute_field_logits(
            batch,
            context_repr=context_repr,
            anchor_stable_probs=stable_anchor.probs,
        )

        return {
            "z_seq": z_seq,
            "z_anchor": z_anchor,
            "stable_logits_anchor": stable_anchor.logits,
            "stable_probs_anchor": stable_anchor.probs,
            "stable_recon_anchor": stable_anchor.recon,
            "residual_anchor": residual_anchor,
            "context_repr": context_repr,
            "future_stable_logits": future_stable_logits,
            "future_residual_pred": future_residual_pred,
            "hazard_logits": hazard_logits,
            "bridge_logits_anchor": bridge_logits_anchor,
            "bridge_gate": bridge_gate,
            "teacher_targets": teacher_targets,
            "bridge_output": bridge_output,
            "field_logits": field_logits,
        }

    def forward(self, batch_or_points):
        if isinstance(batch_or_points, dict):
            outputs = self.forward_sequence(batch_or_points)
            return outputs["z_anchor"], outputs["z_anchor"], None

        points = batch_or_points
        if not torch.is_tensor(points):
            points = torch.as_tensor(points)
        if points.ndim == 4:
            self._validate_temporal_points(points)
            points = points[:, self.anchor_index]
        z_anchor, z_inv, _, eq_z = self._encode_frame_batch(points, use_teacher=False)
        return z_anchor, z_inv, eq_z

    def _extract_supervised_features_from_batch(self, batch):
        if not isinstance(batch, dict):
            raise TypeError(f"Expected a dict batch, got {type(batch)}.")
        if "points" not in batch:
            raise KeyError("Temporal batch is missing required key 'points'.")
        sequence_points = batch["points"]
        self._validate_temporal_contract(batch, sequence_points)
        sequence_points = sequence_points.to(device=self.device, dtype=self.dtype, non_blocking=True)
        anchor_points = sequence_points[:, self.anchor_index]
        z_anchor, _, _, _ = self._encode_frame_batch(anchor_points, use_teacher=False)
        return z_anchor.detach().to(torch.float32), batch.get("class_id")

    def _current_stage_name(self, epoch: int) -> str:
        stages_cfg = getattr(self.tmf_cfg, "stages", None)
        if stages_cfg is None:
            raise ValueError("TMF stage schedule configuration is missing.")

        stage0_end = int(getattr(stages_cfg, "stage0_warmup_epochs", 0))
        stage1_end = stage0_end + int(getattr(stages_cfg, "stage1_stable_epochs", 0))
        stage2_end = stage1_end + int(getattr(stages_cfg, "stage2_bridge_epochs", 0))

        if int(epoch) < stage0_end:
            return "stage0"
        if int(epoch) < stage1_end:
            return "stage1"
        if int(epoch) < stage2_end:
            return "stage2"
        return "stage3"

    def _stage2_start_epoch(self) -> int:
        stages_cfg = getattr(self.tmf_cfg, "stages", None)
        if stages_cfg is None:
            raise ValueError("TMF stage schedule configuration is missing.")
        return int(getattr(stages_cfg, "stage0_warmup_epochs", 0)) + int(
            getattr(stages_cfg, "stage1_stable_epochs", 0)
        )

    def _loss_weights_for_epoch(self, epoch: int) -> dict[str, float]:
        stage_name = self._current_stage_name(epoch)
        weights = {
            "vicreg": self.vicreg.weight if self.vicreg.should_run(current_epoch=epoch) else 0.0,
            "stable_recon": 0.0,
            "stable_balance": 0.0,
            "stable_entropy": 0.0,
            "residual": 0.0,
            "future_stable": 0.0,
            "future_residual": 0.0,
            "hazard": 0.0,
            "bridge": 0.0,
            "bridge_sep": 0.0,
            "field": 0.0,
        }

        if stage_name == "stage0":
            weights["stable_recon"] = 0.1 * self.stable_recon_weight
            weights["stable_balance"] = 0.25 * self.stable_balance_weight
            weights["residual"] = 0.25 * self.residual_weight
            return weights

        weights["stable_recon"] = self.stable_recon_weight
        weights["stable_balance"] = self.stable_balance_weight
        weights["stable_entropy"] = self.stable_sample_entropy_weight
        weights["residual"] = self.residual_weight
        weights["future_stable"] = self.future_stable_weight
        weights["future_residual"] = self.future_residual_weight
        weights["hazard"] = self.hazard_weight

        if stage_name in {"stage2", "stage3"}:
            weights["bridge"] = self.bridge_weight if self.bridge_enabled else 0.0
            weights["bridge_sep"] = self.bridge_separation_weight if self.bridge_enabled else 0.0
        if stage_name == "stage3":
            weights["field"] = self.field_weight if self.field_enabled else 0.0
        return weights

    def _bridge_mask_from_batch(self, batch: dict[str, Any], batch_size: int) -> torch.Tensor:
        explicit_mask = batch.get("is_bridge_candidate")
        if explicit_mask is not None:
            if not torch.is_tensor(explicit_mask):
                explicit_mask = torch.as_tensor(explicit_mask)
            return explicit_mask.to(device=self.device).reshape(-1).to(dtype=torch.bool)

        if not self._bridge_candidate_keys:
            return torch.zeros((batch_size,), device=self.device, dtype=torch.bool)

        sample_keys = build_sample_keys_from_batch(batch, batch_size=batch_size)
        return torch.as_tensor(
            [key in self._bridge_candidate_keys for key in sample_keys],
            device=self.device,
            dtype=torch.bool,
        )

    def _compute_losses(self, batch: dict[str, Any], outputs: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        teacher_targets = outputs["teacher_targets"]
        batch_size = int(outputs["z_anchor"].shape[0])
        device = outputs["z_anchor"].device
        loss_weights = self._loss_weights_for_epoch(int(self.current_epoch))

        zeros = {
            "vicreg": outputs["z_anchor"].new_zeros(()),
            "stable_recon": outputs["z_anchor"].new_zeros(()),
            "stable_balance": outputs["z_anchor"].new_zeros(()),
            "stable_entropy": outputs["z_anchor"].new_zeros(()),
            "residual": outputs["z_anchor"].new_zeros(()),
            "future_stable": outputs["z_anchor"].new_zeros(()),
            "future_residual": outputs["z_anchor"].new_zeros(()),
            "hazard": outputs["z_anchor"].new_zeros(()),
            "bridge": outputs["z_anchor"].new_zeros(()),
            "bridge_sep": outputs["z_anchor"].new_zeros(()),
            "field": outputs["z_anchor"].new_zeros(()),
        }
        metrics: dict[str, torch.Tensor] = {}

        if loss_weights["vicreg"] > 0.0:
            sequence_points = batch["points"].to(device=self.device, dtype=self.dtype, non_blocking=True)
            anchor_pc = sequence_points[:, self.anchor_index]
            if self.use_temporal_vicreg_views and 0 < self.anchor_index < self.sequence_length - 1:
                prev_pc = sequence_points[:, self.anchor_index - 1]
                next_pc = sequence_points[:, self.anchor_index + 1]
                vicreg_views = self._build_vicreg_temporal_views(anchor_pc, prev_pc, next_pc)
            else:
                vicreg_views = None
            vicreg_loss, vicreg_metrics = self.vicreg.compute_loss(
                pc=anchor_pc,
                encoder=self.encoder,
                prepare_input=self.encoder_io.prepare_input,
                split_output=self.encoder_io.split_output,
                current_epoch=int(self.current_epoch),
                views=vicreg_views,
                invariant_transform=self._shared_invariant,
            )
            if vicreg_loss is not None:
                zeros["vicreg"] = vicreg_loss
            for name, value in vicreg_metrics.items():
                metrics[name] = value.to(dtype=torch.float32)

        decoded_residual_anchor = self._decode_residual(outputs["residual_anchor"])
        stable_plus_residual = outputs["stable_recon_anchor"] + decoded_residual_anchor
        zeros["stable_recon"] = F.mse_loss(
            outputs["stable_recon_anchor"].to(dtype=torch.float32),
            teacher_targets["z_anchor"].to(dtype=torch.float32),
        )
        zeros["residual"] = F.mse_loss(
            stable_plus_residual.to(dtype=torch.float32),
            teacher_targets["z_anchor"].to(dtype=torch.float32),
        )
        zeros["stable_balance"] = usage_balance_loss(outputs["stable_probs_anchor"])
        zeros["stable_entropy"] = sample_entropy_loss(outputs["stable_probs_anchor"])

        future_stable_terms: list[torch.Tensor] = []
        future_residual_terms: list[torch.Tensor] = []
        hazard_terms: list[torch.Tensor] = []
        pred_error_columns: list[torch.Tensor] = []
        hazard_prob_columns: list[torch.Tensor] = []
        stable_id_columns = [teacher_targets["anchor_stable_probs"].argmax(dim=-1)]

        for lag in self.lags:
            lag_weight = float(self.lag_weights[lag])
            pred_logits = outputs["future_stable_logits"][lag]
            target_probs = teacher_targets["target_stable_probs"][lag]
            stable_kl = kl_from_logits_to_probs(pred_logits, target_probs)
            future_stable_terms.append(outputs["z_anchor"].new_tensor(lag_weight) * stable_kl)
            metrics[f"temporal_kl_future_lag_{lag}"] = stable_kl.detach().to(dtype=torch.float32)
            metrics[f"temporal_top1_future_acc_lag_{lag}"] = top1_accuracy_from_logits(
                pred_logits,
                target_probs,
            ).detach().to(dtype=torch.float32)

            residual_pred = outputs["future_residual_pred"][lag]
            target_residual = teacher_targets["target_residual"][lag]
            residual_mse = F.mse_loss(
                residual_pred.to(dtype=torch.float32),
                target_residual.to(dtype=torch.float32),
            )
            future_residual_terms.append(outputs["z_anchor"].new_tensor(lag_weight) * residual_mse)

            change_target = teacher_targets["change_targets"][lag]
            hazard_logits = outputs["hazard_logits"][lag]
            hazard_loss = binary_classification_loss(
                hazard_logits,
                change_target,
                focal_gamma=self.hazard_focal_gamma,
            )
            hazard_terms.append(outputs["z_anchor"].new_tensor(lag_weight) * hazard_loss)

            metrics[f"hazard_pos_fraction_lag_{lag}"] = change_target.mean().detach().to(dtype=torch.float32)
            metrics[f"hazard_pred_mean_lag_{lag}"] = (
                torch.sigmoid(hazard_logits).mean().detach().to(dtype=torch.float32)
            )

            per_sample_pred_error = F.kl_div(
                F.log_softmax(pred_logits.to(dtype=torch.float32), dim=-1),
                target_probs.to(dtype=torch.float32),
                reduction="none",
            ).sum(dim=-1)
            pred_error_columns.append(per_sample_pred_error.detach())
            hazard_prob_columns.append(torch.sigmoid(hazard_logits.detach()).reshape(-1))
            stable_id_columns.append(target_probs.argmax(dim=-1))

        zeros["future_stable"] = (
            torch.stack(future_stable_terms, dim=0).sum()
            if future_stable_terms
            else outputs["z_anchor"].new_zeros(())
        )
        zeros["future_residual"] = (
            torch.stack(future_residual_terms, dim=0).sum()
            if future_residual_terms
            else outputs["z_anchor"].new_zeros(())
        )
        zeros["hazard"] = (
            torch.stack(hazard_terms, dim=0).sum()
            if hazard_terms
            else outputs["z_anchor"].new_zeros(())
        )

        bridge_mask = self._bridge_mask_from_batch(batch, batch_size=batch_size)
        metrics["bridge_candidate_fraction"] = self._current_bridge_candidate_fraction(device=device)
        if self.bridge_head is not None and outputs["bridge_output"] is not None:
            bridge_probs = outputs["bridge_output"].probs
            if bool(bridge_mask.any().item()):
                bridge_recon_loss = masked_mse(
                    outputs["bridge_output"].recon,
                    teacher_targets["z_anchor"],
                    bridge_mask,
                )
                bridge_balance = usage_balance_loss(bridge_probs[bridge_mask])
                bridge_entropy = sample_entropy_loss(bridge_probs[bridge_mask])
                zeros["bridge"] = bridge_recon_loss + 0.1 * bridge_balance + 0.01 * bridge_entropy
                metrics.update(
                    {
                        f"bridge_{name}": value.detach().to(dtype=torch.float32)
                        for name, value in prototype_usage_metrics(bridge_probs[bridge_mask]).items()
                    }
                )
            else:
                metrics["bridge_usage_entropy"] = outputs["z_anchor"].new_zeros(())
                metrics["bridge_active_count"] = outputs["z_anchor"].new_zeros(())
                metrics["bridge_dead_fraction"] = outputs["z_anchor"].new_zeros(())
                metrics["bridge_max_prob_mean"] = outputs["z_anchor"].new_zeros(())
            zeros["bridge_sep"] = prototype_separation_loss(
                self.bridge_head.prototypes,
                self.stable_head.prototypes,
                margin=self.bridge_separation_margin,
            )
        else:
            metrics["bridge_usage_entropy"] = outputs["z_anchor"].new_zeros(())
            metrics["bridge_active_count"] = outputs["z_anchor"].new_zeros(())
            metrics["bridge_dead_fraction"] = outputs["z_anchor"].new_zeros(())
            metrics["bridge_max_prob_mean"] = outputs["z_anchor"].new_zeros(())

        stable_usage = prototype_usage_metrics(outputs["stable_probs_anchor"])
        metrics.update(
            {
                f"stable_{name}": value.detach().to(dtype=torch.float32)
                for name, value in stable_usage.items()
            }
        )

        if self.field_enabled and outputs["field_logits"] is not None:
            zeros["field"] = kl_from_logits_to_probs(
                outputs["field_logits"],
                teacher_targets["target_stable_probs"][self.field_lag],
            )

        if pred_error_columns:
            metrics["bridge_scores_mean"] = score_bridge_candidates(
                {"score_weights": self.bridge_score_weights},
                self.lags,
                torch.stack(pred_error_columns, dim=1),
                torch.stack(stable_id_columns, dim=1),
                teacher_targets["stable_confidence"],
                torch.stack(hazard_prob_columns, dim=1),
            ).mean().detach().to(dtype=torch.float32)
        else:
            metrics["bridge_scores_mean"] = outputs["z_anchor"].new_zeros(())

        total_loss = outputs["z_anchor"].new_zeros(())
        for name, value in zeros.items():
            weight = float(loss_weights.get(name, 0.0))
            if weight == 0.0:
                continue
            total_loss = total_loss + outputs["z_anchor"].new_tensor(weight) * value
        zeros["total"] = total_loss
        metrics["hazard_accuracy_proxy"] = masked_mean(
            (torch.sigmoid(outputs["hazard_logits"][self.lags[0]]).reshape(-1) > 0.5)
            == (teacher_targets["change_targets"][self.lags[0]].reshape(-1) > 0.5)
        ).detach().to(dtype=torch.float32)
        metrics["tmf_stage"] = outputs["z_anchor"].new_tensor(
            {"stage0": 0.0, "stage1": 1.0, "stage2": 2.0, "stage3": 3.0}[self._current_stage_name(int(self.current_epoch))]
        )
        return zeros, metrics

    def _log_core_metrics(
        self,
        *,
        stage: str,
        batch_size: int,
        losses: dict[str, torch.Tensor],
        metrics: dict[str, torch.Tensor],
    ) -> None:
        self._log_metric(
            stage,
            "loss",
            losses["total"],
            prog_bar=True,
            batch_size=batch_size,
        )
        loss_name_map = {
            "total": "loss/total",
            "vicreg": "loss/vicreg",
            "stable_recon": "loss/stable_recon",
            "stable_balance": "loss/stable_balance",
            "stable_entropy": "loss/stable_entropy",
            "residual": "loss/residual",
            "future_stable": "loss/future_stable",
            "future_residual": "loss/future_residual",
            "hazard": "loss/hazard",
            "bridge": "loss/bridge",
            "bridge_sep": "loss/bridge_sep",
            "field": "loss/field",
        }
        for key, log_name in loss_name_map.items():
            self._log_metric(
                stage,
                log_name,
                losses[key],
                prog_bar=(key == "total"),
                batch_size=batch_size,
            )

        metric_name_map = {
            "stable_usage_entropy": "stable/usage_entropy",
            "stable_active_count": "stable/active_count",
            "stable_dead_fraction": "stable/dead_fraction",
            "stable_max_prob_mean": "stable/max_prob_mean",
            "bridge_candidate_fraction": "bridge/candidate_fraction",
            "bridge_usage_entropy": "bridge/usage_entropy",
            "bridge_active_count": "bridge/active_count",
            "bridge_dead_fraction": "bridge/dead_fraction",
            "bridge_max_prob_mean": "bridge/max_prob_mean",
            "bridge_scores_mean": "bridge/score_mean",
            "hazard_accuracy_proxy": "hazard/accuracy",
            "tmf_stage": "tmf/stage_id",
        }
        for key, log_name in metric_name_map.items():
            if key not in metrics:
                continue
            self._log_metric(stage, log_name, metrics[key], batch_size=batch_size)

        for lag in self.lags:
            if f"temporal_top1_future_acc_lag_{lag}" in metrics:
                self._log_metric(
                    stage,
                    f"temporal/top1_future_acc_lag_{lag}",
                    metrics[f"temporal_top1_future_acc_lag_{lag}"],
                    batch_size=batch_size,
                )
            if f"temporal_kl_future_lag_{lag}" in metrics:
                self._log_metric(
                    stage,
                    f"temporal/kl_future_lag_{lag}",
                    metrics[f"temporal_kl_future_lag_{lag}"],
                    batch_size=batch_size,
                )
            if f"hazard_pos_fraction_lag_{lag}" in metrics:
                self._log_metric(
                    stage,
                    f"hazard/pos_fraction_lag_{lag}",
                    metrics[f"hazard_pos_fraction_lag_{lag}"],
                    batch_size=batch_size,
                )
            if f"hazard_pred_mean_lag_{lag}" in metrics:
                self._log_metric(
                    stage,
                    f"hazard/pred_mean_lag_{lag}",
                    metrics[f"hazard_pred_mean_lag_{lag}"],
                    batch_size=batch_size,
                )

    def _step(self, batch, batch_idx: int, stage: str):
        if not isinstance(batch, dict):
            raise TypeError(f"TemporalMotifFieldModule expects dict batches, got {type(batch)}.")
        if "points" not in batch:
            raise KeyError("Temporal batch is missing required key 'points'.")

        sequence_points = batch["points"]
        self._validate_temporal_contract(batch, sequence_points)
        batch_size = int(sequence_points.shape[0])

        outputs = self.forward_sequence(batch)
        losses, metrics = self._compute_losses(batch, outputs)
        total_loss = losses["total"]

        if stage != "train" and batch_idx == 0:
            self._status_print(
                f"[{stage}-diag] epoch={self.current_epoch} "
                f"loss_total={float(total_loss.detach().cpu().item()):.6f} "
                f"stage={self._current_stage_name(int(self.current_epoch))}"
            )

        if not torch.isfinite(total_loss).item():
            self._consecutive_nan_steps += 1
            self._log_metric(stage, "loss_nonfinite", 1.0, on_step=True, on_epoch=True, batch_size=batch_size)
            if self._consecutive_nan_steps >= self._max_consecutive_nan_steps:
                raise RuntimeError(
                    f"Temporal Motif Field produced {self._consecutive_nan_steps} consecutive "
                    "non-finite losses. Halting to avoid silent divergence."
                )
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
            losses["total"] = total_loss
        else:
            self._consecutive_nan_steps = 0

        self._log_core_metrics(stage=stage, batch_size=batch_size, losses=losses, metrics=metrics)

        should_cache_stage = stage in self._supervised_cache and (
            stage != "train" or self.cache_train_supervised_metrics
        )
        if should_cache_stage:
            limit = self._cache_limit_for_stage(stage)
            cache = self._supervised_cache.get(stage)
            already_cached = cached_sample_count(cache) if cache is not None else 0
            if limit is None or already_cached < limit:
                self._cache_supervised_batch(
                    stage,
                    outputs["z_anchor"].detach().to(torch.float32),
                    self._temporal_meta_from_batch(batch),
                    encoder_features=outputs["z_anchor"].detach().to(torch.float32),
                )

        if stage == "test":
            return {
                "loss": total_loss.detach(),
                "stable_probs_anchor": outputs["stable_probs_anchor"].detach(),
                "bridge_gate": None if outputs["bridge_gate"] is None else outputs["bridge_gate"].detach(),
            }
        return total_loss

    def training_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def _log_metric(
        self,
        stage: str,
        name: str,
        value,
        *,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        batch_size: int | None = None,
        **kwargs,
    ) -> None:
        if on_step is None:
            on_step = stage == "train"
        if on_epoch is None:
            on_epoch = True
        if not on_step and not on_epoch:
            return
        log_kwargs = dict(kwargs)
        if batch_size is not None and "batch_size" not in log_kwargs:
            log_kwargs["batch_size"] = int(batch_size)
        if "sync_dist" not in log_kwargs:
            log_kwargs["sync_dist"] = True
        if torch.is_tensor(value):
            value = value.detach()
        self.log(f"{stage}/{name}", value, on_step=on_step, on_epoch=on_epoch, **log_kwargs)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None, *args, **kwargs):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure=optimizer_closure, *args, **kwargs)
        self._update_ema_teacher()

    def _should_refresh_bridge_candidates(self, epoch: int) -> bool:
        if not self.bridge_enabled or not self.bridge_mining_enabled:
            return False
        stage2_start = self._stage2_start_epoch()
        if int(epoch) < stage2_start:
            return False
        if not self._bridge_candidate_keys:
            return True
        return ((int(epoch) - stage2_start) % self.bridge_refresh_epochs) == 0

    @staticmethod
    def _is_global_zero() -> bool:
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def _broadcast_bridge_candidate_keys(
        self,
        candidate_keys: set[tuple[str, Any, Any]],
    ) -> set[tuple[str, Any, Any]]:
        if not dist.is_available() or not dist.is_initialized():
            return candidate_keys
        object_list = [list(candidate_keys) if self._is_global_zero() else None]
        dist.broadcast_object_list(object_list, src=0)
        received = object_list[0]
        if received is None:
            return set()
        return {tuple(item) for item in received}

    def _build_bridge_refresh_dataloader(self):
        if self.trainer is None or self.trainer.datamodule is None:
            raise RuntimeError("Bridge candidate refresh requires an attached trainer datamodule.")
        dm = self.trainer.datamodule
        if hasattr(dm, "_temporal_loader") and hasattr(dm, "train_dataset"):
            return dm._temporal_loader(
                dm.train_dataset,
                shuffle_windows=False,
                shuffle_centers=False,
                drop_last=False,
                mixed_windows_per_batch=None,
            )
        return dm.train_dataloader()

    def _current_bridge_candidate_fraction(self, *, device) -> torch.Tensor:
        if self.trainer is None or self.trainer.datamodule is None:
            return torch.zeros((), device=device, dtype=torch.float32)
        train_dataset = getattr(self.trainer.datamodule, "train_dataset", None)
        if train_dataset is None:
            return torch.zeros((), device=device, dtype=torch.float32)
        total = int(len(train_dataset))
        if total <= 0:
            return torch.zeros((), device=device, dtype=torch.float32)
        return torch.tensor(
            float(len(self._bridge_candidate_keys)) / float(total),
            device=device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def _refresh_bridge_candidates(self) -> None:
        if not self.bridge_enabled or not self.bridge_mining_enabled:
            self._bridge_candidate_keys = set()
            return

        max_samples = self.bridge_subset_max_samples if self.bridge_subset_max_samples > 0 else None
        if self._is_global_zero():
            dataloader = self._build_bridge_refresh_dataloader()
            sample_keys: list[tuple[str, Any, Any]] = []
            score_chunks: list[torch.Tensor] = []
            collected = 0
            was_training = self.training
            self.eval()
            try:
                for batch in dataloader:
                    if not isinstance(batch, dict):
                        raise TypeError(
                            "Bridge mining refresh expects temporal dict batches, "
                            f"got {type(batch)}."
                        )
                    outputs = self.forward_sequence(batch)
                    teacher_targets = outputs["teacher_targets"]
                    batch_size = int(outputs["z_anchor"].shape[0])

                    pred_error_columns: list[torch.Tensor] = []
                    hazard_prob_columns: list[torch.Tensor] = []
                    stable_id_columns = [teacher_targets["anchor_stable_probs"].argmax(dim=-1)]
                    for lag in self.lags:
                        pred_logits = outputs["future_stable_logits"][lag]
                        target_probs = teacher_targets["target_stable_probs"][lag]
                        pred_error_columns.append(
                            F.kl_div(
                                F.log_softmax(pred_logits.to(dtype=torch.float32), dim=-1),
                                target_probs.to(dtype=torch.float32),
                                reduction="none",
                            ).sum(dim=-1).cpu()
                        )
                        hazard_prob_columns.append(
                            torch.sigmoid(outputs["hazard_logits"][lag]).reshape(-1).to(dtype=torch.float32).cpu()
                        )
                        stable_id_columns.append(target_probs.argmax(dim=-1).cpu())

                    scores = score_bridge_candidates(
                        {"score_weights": self.bridge_score_weights},
                        self.lags,
                        torch.stack(pred_error_columns, dim=1),
                        torch.stack(stable_id_columns, dim=1),
                        teacher_targets["stable_confidence"].to(dtype=torch.float32).cpu(),
                        torch.stack(hazard_prob_columns, dim=1),
                    )
                    score_chunks.append(scores.cpu())
                    sample_keys.extend(build_sample_keys_from_batch(batch, batch_size=batch_size))
                    collected += batch_size
                    if max_samples is not None and collected >= int(max_samples):
                        break
            finally:
                if was_training:
                    self.train()

            all_scores = torch.cat(score_chunks, dim=0) if score_chunks else torch.zeros((0,), dtype=torch.float32)
            if max_samples is not None and len(sample_keys) > int(max_samples):
                sample_keys = sample_keys[: int(max_samples)]
                all_scores = all_scores[: int(max_samples)]
            candidate_keys = select_bridge_candidate_keys(
                sample_keys,
                all_scores,
                candidate_fraction=self.bridge_candidate_fraction,
            )
            self._status_print(
                "[tmf] Bridge refresh completed: "
                f"processed_samples={len(sample_keys)}, selected_candidates={len(candidate_keys)}, "
                f"candidate_fraction={self.bridge_candidate_fraction:.4f}."
            )
        else:
            candidate_keys = set()

        self._bridge_candidate_keys = self._broadcast_bridge_candidate_keys(candidate_keys)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if self._should_refresh_bridge_candidates(int(self.current_epoch)):
            self._refresh_bridge_candidates()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["tmf_bridge_candidate_keys"] = list(self._bridge_candidate_keys)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        raw_keys = checkpoint.get("tmf_bridge_candidate_keys", [])
        self._bridge_candidate_keys = {tuple(item) for item in raw_keys}
        state_dict = checkpoint.get("state_dict", {}) if isinstance(checkpoint, dict) else {}
        if not any(str(key).startswith("ema_teacher.") for key in state_dict):
            self._sync_ema_teacher_from_student()


__all__ = ["TemporalMotifFieldModule"]
