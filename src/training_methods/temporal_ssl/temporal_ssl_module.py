import os
import sys
from collections.abc import Sequence

import pytorch_lightning as pl
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from src.models.autoencoders.factory import build_model
from src.training_methods.contrastive_learning.supervised_cache import (
    cache_limit_for_stage,
    cache_supervised_batch,
    init_supervised_cache,
    log_supervised_metrics,
    reset_supervised_cache,
)
from src.training_methods.contrastive_learning.vicreg import VICRegLoss
from src.training_methods.temporal_ssl.lejepa import LeJEPALoss
from src.utils.pointcloud_ops import crop_to_num_points, shift_to_neighbor
from src.utils.training_utils import cached_sample_count, get_optimizers_and_scheduler


def resolve_latent_dim(cfg):
    if hasattr(cfg, "latent_size"):
        return int(cfg.latent_size)
    if hasattr(cfg, "encoder") and hasattr(cfg.encoder, "kwargs"):
        latent_size = cfg.encoder.kwargs.get("latent_size", None)
        if latent_size is not None:
            return int(latent_size)
    return None


class TemporalFrameContrastiveLoss(nn.Module):
    """Contrastive loss on center-frame embeddings for the same tracked atom."""

    def __init__(
        self,
        *,
        enabled: bool,
        weight: float,
        start_epoch: int,
        positive_max_frame_delta: int,
        negative_min_fraction: float,
        negative_margin: float,
        positive_coeff: float,
        negative_coeff: float,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.weight = float(weight)
        self.start_epoch = max(0, int(start_epoch))
        self.positive_max_frame_delta = int(positive_max_frame_delta)
        self.negative_min_fraction = float(negative_min_fraction)
        self.negative_margin = float(negative_margin)
        self.positive_coeff = float(positive_coeff)
        self.negative_coeff = float(negative_coeff)

        if self.positive_max_frame_delta < 0:
            raise ValueError(
                "temporal_contrastive_positive_max_frame_delta must be >= 0, "
                f"got {self.positive_max_frame_delta}."
            )
        if not (0.0 <= self.negative_min_fraction <= 1.0):
            raise ValueError(
                "temporal_contrastive_negative_min_fraction must be in [0, 1], "
                f"got {self.negative_min_fraction}."
            )
        if not (-1.0 <= self.negative_margin <= 1.0):
            raise ValueError(
                "temporal_contrastive_negative_margin must be in [-1, 1], "
                f"got {self.negative_margin}."
            )
        if self.positive_coeff < 0.0:
            raise ValueError(
                "temporal_contrastive_positive_coeff must be >= 0, "
                f"got {self.positive_coeff}."
            )
        if self.negative_coeff < 0.0:
            raise ValueError(
                "temporal_contrastive_negative_coeff must be >= 0, "
                f"got {self.negative_coeff}."
            )

    @classmethod
    def from_config(cls, cfg):
        return cls(
            enabled=bool(getattr(cfg, "temporal_contrastive_enabled", False)),
            weight=float(getattr(cfg, "temporal_contrastive_weight", 0.0)),
            start_epoch=int(getattr(cfg, "temporal_contrastive_start_epoch", 0)),
            positive_max_frame_delta=int(
                getattr(cfg, "temporal_contrastive_positive_max_frame_delta", 2)
            ),
            negative_min_fraction=float(
                getattr(cfg, "temporal_contrastive_negative_min_fraction", 0.5)
            ),
            negative_margin=float(getattr(cfg, "temporal_contrastive_negative_margin", 0.0)),
            positive_coeff=float(getattr(cfg, "temporal_contrastive_positive_coeff", 1.0)),
            negative_coeff=float(getattr(cfg, "temporal_contrastive_negative_coeff", 1.0)),
        )

    def should_run(self, *, current_epoch: int) -> bool:
        return bool(
            self.enabled
            and self.weight > 0.0
            and (self.positive_coeff > 0.0 or self.negative_coeff > 0.0)
            and int(current_epoch) >= self.start_epoch
        )

    def compute_loss(
        self,
        *,
        embeddings: torch.Tensor,
        frame_indices: torch.Tensor,
        center_atom_ids: torch.Tensor,
        source_paths: Sequence[str] | None,
        simulation_frame_span: int,
        current_epoch: int,
    ):
        if not self.should_run(current_epoch=current_epoch):
            return None, {}

        self._validate_inputs(
            embeddings=embeddings,
            frame_indices=frame_indices,
            center_atom_ids=center_atom_ids,
            source_paths=source_paths,
        )
        if simulation_frame_span < 0:
            raise ValueError(
                "simulation_frame_span must be >= 0 for temporal contrastive loss, "
                f"got {simulation_frame_span}."
            )

        normalized_embeddings = F.normalize(
            embeddings.to(dtype=torch.float32),
            dim=1,
            eps=1e-6,
        )
        frame_indices = frame_indices.reshape(-1).to(
            device=normalized_embeddings.device,
            dtype=torch.float32,
        )
        grouped_indices = self._group_sample_indices(
            center_atom_ids=center_atom_ids,
            source_paths=source_paths,
            batch_size=int(normalized_embeddings.shape[0]),
        )
        negative_min_delta = float(simulation_frame_span) * self.negative_min_fraction

        zero = normalized_embeddings.new_zeros(())
        positive_loss_sum = zero.clone()
        negative_loss_sum = zero.clone()
        positive_similarity_sum = zero.clone()
        negative_similarity_sum = zero.clone()
        positive_pair_count = 0
        negative_pair_count = 0
        active_track_count = 0

        for sample_indices in grouped_indices:
            if len(sample_indices) < 2:
                continue
            active_track_count += 1
            index_tensor = torch.as_tensor(
                sample_indices,
                device=normalized_embeddings.device,
                dtype=torch.long,
            )
            track_embeddings = normalized_embeddings.index_select(0, index_tensor)
            track_frame_indices = frame_indices.index_select(0, index_tensor)
            track_size = int(index_tensor.numel())
            upper_triangle = torch.triu(
                torch.ones(
                    (track_size, track_size),
                    device=normalized_embeddings.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            frame_diffs = (track_frame_indices[:, None] - track_frame_indices[None, :]).abs()
            cosine_similarity = track_embeddings @ track_embeddings.transpose(0, 1)

            positive_mask = upper_triangle & (frame_diffs <= float(self.positive_max_frame_delta))
            if self.positive_coeff > 0.0 and positive_mask.any():
                positive_similarity = cosine_similarity[positive_mask]
                positive_loss_sum = positive_loss_sum + (1.0 - positive_similarity).sum()
                positive_similarity_sum = positive_similarity_sum + positive_similarity.sum()
                positive_pair_count += int(positive_similarity.numel())

            negative_mask = upper_triangle & (frame_diffs > negative_min_delta)
            if self.negative_coeff > 0.0 and negative_mask.any():
                negative_similarity = cosine_similarity[negative_mask]
                negative_loss_sum = negative_loss_sum + F.relu(
                    negative_similarity - self.negative_margin
                ).sum()
                negative_similarity_sum = negative_similarity_sum + negative_similarity.sum()
                negative_pair_count += int(negative_similarity.numel())

        loss = None
        metrics = {
            "temporal_contrastive_pos_pairs": zero.new_tensor(float(positive_pair_count)),
            "temporal_contrastive_neg_pairs": zero.new_tensor(float(negative_pair_count)),
            "temporal_contrastive_active_tracks": zero.new_tensor(float(active_track_count)),
            "temporal_contrastive_frame_span": zero.new_tensor(float(simulation_frame_span)),
            "temporal_contrastive_far_threshold": zero.new_tensor(float(negative_min_delta)),
        }

        if positive_pair_count > 0:
            positive_loss = positive_loss_sum / float(positive_pair_count)
            positive_similarity_mean = positive_similarity_sum / float(positive_pair_count)
            metrics["temporal_contrastive_positive"] = positive_loss
            metrics["temporal_contrastive_positive_similarity"] = positive_similarity_mean
            loss = self.positive_coeff * positive_loss if loss is None else loss + self.positive_coeff * positive_loss
        else:
            metrics["temporal_contrastive_positive"] = zero.clone()
            metrics["temporal_contrastive_positive_similarity"] = zero.clone()

        if negative_pair_count > 0:
            negative_loss = negative_loss_sum / float(negative_pair_count)
            negative_similarity_mean = negative_similarity_sum / float(negative_pair_count)
            metrics["temporal_contrastive_negative"] = negative_loss
            metrics["temporal_contrastive_negative_similarity"] = negative_similarity_mean
            loss = self.negative_coeff * negative_loss if loss is None else loss + self.negative_coeff * negative_loss
        else:
            metrics["temporal_contrastive_negative"] = zero.clone()
            metrics["temporal_contrastive_negative_similarity"] = zero.clone()

        if loss is None:
            loss = zero.clone()

        if not torch.isfinite(loss).item():
            metrics["temporal_contrastive_nonfinite"] = zero.new_tensor(1.0)
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        return loss, metrics

    @staticmethod
    def _validate_inputs(
        *,
        embeddings: torch.Tensor,
        frame_indices: torch.Tensor,
        center_atom_ids: torch.Tensor,
        source_paths: Sequence[str] | None,
    ) -> None:
        if not torch.is_tensor(embeddings):
            raise TypeError(f"embeddings must be a torch.Tensor, got {type(embeddings)}.")
        if embeddings.dim() != 2:
            raise ValueError(
                "Temporal contrastive embeddings must have shape (B, D), "
                f"got {tuple(embeddings.shape)}."
            )
        batch_size = int(embeddings.shape[0])
        if not torch.is_tensor(frame_indices):
            raise TypeError(f"frame_indices must be a torch.Tensor, got {type(frame_indices)}.")
        if frame_indices.reshape(-1).shape[0] != batch_size:
            raise ValueError(
                "frame_indices must provide exactly one representative frame index per embedding. "
                f"batch_size={batch_size}, frame_indices_shape={tuple(frame_indices.shape)}."
            )
        if not torch.is_tensor(center_atom_ids):
            raise TypeError(f"center_atom_ids must be a torch.Tensor, got {type(center_atom_ids)}.")
        if center_atom_ids.reshape(-1).shape[0] != batch_size:
            raise ValueError(
                "center_atom_ids must provide exactly one tracked atom id per embedding. "
                f"batch_size={batch_size}, center_atom_ids_shape={tuple(center_atom_ids.shape)}."
            )
        if source_paths is not None and len(source_paths) != batch_size:
            raise ValueError(
                "source_paths must match the embedding batch size when provided. "
                f"batch_size={batch_size}, len(source_paths)={len(source_paths)}."
            )

    @staticmethod
    def _group_sample_indices(
        *,
        center_atom_ids: torch.Tensor,
        source_paths: Sequence[str] | None,
        batch_size: int,
    ) -> list[list[int]]:
        atom_ids = center_atom_ids.reshape(-1).to(dtype=torch.int64, device="cpu").tolist()
        resolved_source_paths = (
            ["<shared-source>"] * batch_size
            if source_paths is None
            else [str(path) for path in source_paths]
        )
        groups: dict[tuple[str, int], list[int]] = {}
        for sample_idx, (source_path, atom_id) in enumerate(
            zip(resolved_source_paths, atom_ids, strict=True)
        ):
            groups.setdefault((source_path, int(atom_id)), []).append(int(sample_idx))
        return [indices for indices in groups.values() if len(indices) > 1]


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

        if bool(getattr(cfg, "vicreg_use_ri_mae_backbone", False)):
            raise ValueError(
                "vicreg_use_ri_mae_backbone is deprecated. "
                "Use encoder.name='RI_MAE_Invariant'. Contrastive training now uses a fixed norms-only invariant path."
            )

        self.encoder, _ = build_model(cfg)
        latent_dim = resolve_latent_dim(cfg)

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

        self.vicreg = VICRegLoss.from_config(
            cfg,
            input_dim=latent_dim,
        )
        self.lejepa = LeJEPALoss.from_config(
            cfg,
            sequence_length=self.sequence_length,
        )
        self.temporal_contrastive = TemporalFrameContrastiveLoss.from_config(cfg)

        init_supervised_cache(self, cfg)
        self.cache_train_supervised_metrics = bool(getattr(cfg, "cache_train_supervised_metrics", False))
        self._warned_cache_eq_fallback = False
        self._warned_temporal_contrastive_no_pairs = False
        self._consecutive_nan_steps = 0
        self._max_consecutive_nan_steps = int(getattr(cfg, "max_consecutive_nan_steps", 20))
        self._temporal_frame_span_cache: dict[str, int] = {}

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

    def _prepare_encoder_input(self, pc: torch.Tensor) -> torch.Tensor:
        if getattr(self.encoder, "expects_channel_first", False):
            return pc.permute(0, 2, 1).contiguous()
        return pc

    def _status_print(self, message: str) -> None:
        if getattr(self, "_trainer", None) is not None:
            self.print(message)
            return
        print(message)

    def _split_encoder_output(self, enc_out):
        if isinstance(enc_out, (tuple, list)):
            if not enc_out:
                raise ValueError("Encoder returned empty output")
            z_inv_model = enc_out[0]
            eq_z = None
            for candidate in enc_out[1:]:
                if not (torch.is_tensor(candidate) and candidate.dim() == 3 and candidate.shape[-1] == 3):
                    continue
                if torch.is_tensor(z_inv_model) and z_inv_model.dim() == 2:
                    if candidate.shape[1] == z_inv_model.shape[1]:
                        eq_z = candidate
                        break
                    if candidate.shape[1] == 3:
                        continue
                if candidate.shape[1] != 3:
                    eq_z = candidate
                    break
            return z_inv_model, eq_z
        return enc_out, None

    def _prepare_model_input(self, pc: torch.Tensor) -> torch.Tensor:
        out = pc
        if self.model_points is not None:
            out = crop_to_num_points(out, self.model_points)
        return out

    def _select_lejepa_frame_window(self, sequence_points: torch.Tensor) -> torch.Tensor:
        if self.lejepa is None:
            raise RuntimeError("LeJEPA frame-window selection was requested, but self.lejepa is not initialized.")
        self._validate_temporal_points(sequence_points)
        target_frame_index = self.sequence_length - 1
        start_frame_index = target_frame_index - int(self.lejepa.context_frames)
        return sequence_points[:, start_frame_index : target_frame_index + 1]

    def _encode_temporal_frame_sequence(self, sequence_points: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(sequence_points):
            raise TypeError(f"sequence_points must be a torch.Tensor, got {type(sequence_points)}.")
        if sequence_points.dim() != 4 or sequence_points.shape[-1] != 3:
            raise ValueError(
                "Expected a temporal point-cloud tensor with shape (B, T, N, 3), "
                f"got {tuple(sequence_points.shape)}."
            )

        sequence_points = sequence_points.to(device=self.device, dtype=self.dtype, non_blocking=True)
        batch_size, num_frames, num_points, _ = sequence_points.shape
        flat_points = sequence_points.reshape(batch_size * num_frames, num_points, 3).contiguous()
        flat_points = self._prepare_model_input(flat_points)
        z_inv_model, eq_z = self._split_encoder_output(
            self.encoder(self._prepare_encoder_input(flat_points))
        )
        z_inv_contrastive = self._contrastive_invariant_latent(z_inv_model, eq_z)
        if z_inv_contrastive is None:
            raise RuntimeError(
                "LeJEPA requires invariant encoder embeddings, but the configured encoder "
                "did not produce a usable invariant latent."
            )
        return z_inv_contrastive.reshape(batch_size, num_frames, -1)

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

    def _resolve_stage_dataset(self, stage: str):
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return None
        dataset_attr = {
            "train": "train_dataset",
            "val": "val_dataset",
            "test": "test_dataset",
        }.get(stage)
        if dataset_attr is None:
            return None
        dataset = getattr(datamodule, dataset_attr, None)
        while isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        return dataset

    def _resolve_temporal_frame_span(self, stage: str) -> int:
        cached = self._temporal_frame_span_cache.get(stage)
        if cached is not None:
            return int(cached)

        dataset = self._resolve_stage_dataset(stage)
        if dataset is not None:
            frame_start = int(getattr(dataset, "frame_start"))
            frame_stop_raw = getattr(dataset, "frame_stop")
            frame_stop = int(getattr(dataset, "frame_count")) if frame_stop_raw is None else int(frame_stop_raw)
            frame_stride = int(getattr(dataset, "frame_stride"))
            sequence_length = int(getattr(dataset, "sequence_length"))
        else:
            data_cfg = getattr(self.hparams, "data", None)
            if data_cfg is None:
                raise RuntimeError(
                    "Temporal contrastive loss could not resolve a temporal dataset or fallback data config."
                )
            frame_start = int(getattr(data_cfg, "frame_start", 0))
            frame_stop_raw = getattr(data_cfg, "frame_stop", None)
            if frame_stop_raw is None:
                raise RuntimeError(
                    "Temporal contrastive loss requires either an attached TemporalLAMMPS dataset or "
                    "data.frame_stop to resolve the simulation frame span."
                )
            frame_stop = int(frame_stop_raw)
            frame_stride = int(getattr(data_cfg, "frame_stride", 1))
            sequence_length = int(getattr(data_cfg, "sequence_length", self.sequence_length))

        min_center_frame = frame_start + self.center_frame_index * frame_stride
        max_center_frame = frame_stop - 1 - (sequence_length - 1 - self.center_frame_index) * frame_stride
        if max_center_frame < min_center_frame:
            raise RuntimeError(
                "Temporal contrastive loss resolved an invalid center-frame span. "
                f"stage={stage!r}, min_center_frame={min_center_frame}, "
                f"max_center_frame={max_center_frame}, frame_start={frame_start}, "
                f"frame_stop={frame_stop}, sequence_length={sequence_length}, frame_stride={frame_stride}."
            )
        frame_span = int(max_center_frame - min_center_frame)
        self._temporal_frame_span_cache[stage] = frame_span
        return frame_span

    @staticmethod
    def _distributed_world_size() -> int:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return 1
        return int(torch.distributed.get_world_size())

    def _gather_all_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        world_size = self._distributed_world_size()
        if world_size <= 1:
            return tensor

        local_size = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_sizes, local_size)
        sizes = [int(size.item()) for size in gathered_sizes]
        max_size = max(sizes)
        if max_size == 0:
            return tensor

        if int(tensor.shape[0]) != max_size:
            pad = tensor.new_zeros((max_size - int(tensor.shape[0]), *tensor.shape[1:]))
            padded = torch.cat([tensor, pad], dim=0)
        else:
            padded = tensor
        padded = padded.contiguous()

        gathered = [padded.new_zeros((max_size, *tensor.shape[1:])) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, padded)

        rank = int(torch.distributed.get_rank())
        gathered[rank] = padded
        trimmed = [chunk[:size] for chunk, size in zip(gathered, sizes, strict=True)]
        return torch.cat(trimmed, dim=0)

    def _gather_all_source_paths(
        self,
        source_paths: Sequence[str] | None,
        *,
        batch_size: int,
    ) -> list[str] | None:
        if source_paths is None:
            local_paths = ["<shared-source>"] * int(batch_size)
        else:
            local_paths = [str(path) for path in source_paths]
            if len(local_paths) != int(batch_size):
                raise ValueError(
                    "source_paths must match the local batch size before distributed gathering. "
                    f"batch_size={batch_size}, len(source_paths)={len(local_paths)}."
                )

        world_size = self._distributed_world_size()
        if world_size <= 1:
            return local_paths

        gathered_paths: list[list[str] | None] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered_paths, local_paths)

        flattened: list[str] = []
        for rank_idx, rank_paths in enumerate(gathered_paths):
            if not isinstance(rank_paths, list):
                raise TypeError(
                    "Distributed source-path gather returned a non-list object. "
                    f"rank={rank_idx}, type={type(rank_paths)}."
                )
            flattened.extend(str(path) for path in rank_paths)
        return flattened

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
        z_inv_model, eq_z = self._split_encoder_output(self.encoder(self._prepare_encoder_input(pc)))
        z_inv_contrastive = self._contrastive_invariant_latent(z_inv_model, eq_z)
        features = z_inv_contrastive if z_inv_contrastive is not None else z_inv_model
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

    def forward(self, pc: torch.Tensor):
        if torch.is_tensor(pc) and pc.dim() == 4:
            self._validate_temporal_points(pc)
            pc = self._center_frame(pc)
        enc_out = self.encoder(self._prepare_encoder_input(pc))
        z_inv_model, eq_z = self._split_encoder_output(enc_out)
        z_inv_contrastive = self._contrastive_invariant_latent(z_inv_model, eq_z)
        return z_inv_contrastive, z_inv_model, eq_z

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
        should_run_temporal_contrastive = self.temporal_contrastive.should_run(
            current_epoch=int(self.current_epoch)
        )
        should_run_vicreg = self.vicreg.should_run(current_epoch=int(self.current_epoch))

        losses = {}
        center_embeddings = None

        need_center_embeddings = should_run_temporal_contrastive or should_cache_stage
        if need_center_embeddings:
            grad_context = torch.enable_grad if should_run_temporal_contrastive else torch.no_grad
            with grad_context():
                z_inv_model, eq_z = self._split_encoder_output(
                    self.encoder(self._prepare_encoder_input(pc))
                )
                center_embeddings = self._contrastive_invariant_from_eq_latent(
                    eq_z,
                    z_inv_model=z_inv_model,
                    stage=stage,
                )

        if should_run_vicreg:
            vicreg_views = (
                self._build_vicreg_temporal_views(pc_raw, prev_pc, next_pc)
                if self.use_temporal_vicreg_views
                else None
            )
            vicreg_loss, vicreg_metrics = self.vicreg.compute_loss(
                pc=vicreg_pc_raw,
                encoder=self.encoder,
                prepare_input=self._prepare_encoder_input,
                split_output=self._split_encoder_output,
                current_epoch=int(self.current_epoch),
                views=vicreg_views,
                invariant_transform=self._shared_invariant,
            )
            if vicreg_loss is not None:
                losses["vicreg"] = vicreg_loss
            for name, value in vicreg_metrics.items():
                self._log_metric(stage, name, value, batch_size=batch_size)

        if self.lejepa is not None and self.lejepa.should_run(current_epoch=int(self.current_epoch)):
            lejepa_frame_embeddings = self._encode_temporal_frame_sequence(
                self._select_lejepa_frame_window(sequence_points)
            )
            lejepa_loss, lejepa_metrics = self.lejepa.compute_loss(
                frame_embeddings=lejepa_frame_embeddings,
                current_epoch=int(self.current_epoch),
                global_step=int(self.global_step),
            )
            if lejepa_loss is not None:
                losses["lejepa"] = lejepa_loss
            for name, value in lejepa_metrics.items():
                self._log_metric(stage, name, value, batch_size=batch_size)

        if should_run_temporal_contrastive:
            if center_embeddings is None:
                raise RuntimeError(
                    "Temporal contrastive loss requires invariant encoder embeddings, "
                    "but the configured encoder did not produce a usable latent."
                )
            frame_indices = meta.get("frame_indices")
            if frame_indices is None:
                raise KeyError("Temporal contrastive loss requires batch['frame_indices'].")
            center_atom_ids = meta.get("center_atom_id")
            if center_atom_ids is None:
                raise KeyError("Temporal contrastive loss requires batch['center_atom_id'].")

            gathered_embeddings = self._gather_all_tensor(center_embeddings)
            gathered_frame_indices = self._gather_all_tensor(
                frame_indices[:, self.center_frame_index].reshape(-1, 1)
            ).reshape(-1)
            gathered_center_atom_ids = self._gather_all_tensor(
                center_atom_ids.reshape(-1, 1)
            ).reshape(-1)
            gathered_source_paths = self._gather_all_source_paths(
                meta.get("source_path"),
                batch_size=batch_size,
            )

            temporal_contrastive_loss, temporal_contrastive_metrics = self.temporal_contrastive.compute_loss(
                embeddings=gathered_embeddings,
                frame_indices=gathered_frame_indices,
                center_atom_ids=gathered_center_atom_ids,
                source_paths=gathered_source_paths,
                simulation_frame_span=self._resolve_temporal_frame_span(stage),
                current_epoch=int(self.current_epoch),
            )
            if temporal_contrastive_loss is not None:
                losses["temporal_contrastive"] = temporal_contrastive_loss
                pos_pairs = int(temporal_contrastive_metrics["temporal_contrastive_pos_pairs"].item())
                neg_pairs = int(temporal_contrastive_metrics["temporal_contrastive_neg_pairs"].item())
                if pos_pairs + neg_pairs == 0 and not self._warned_temporal_contrastive_no_pairs:
                    self._status_print(
                        "[temporal-contrastive] Warning: the current batch did not contain any same-atom "
                        "positive or far-negative temporal pairs. Increase batch mixing across windows if "
                        "this persists."
                    )
                    self._warned_temporal_contrastive_no_pairs = True
            for name, value in temporal_contrastive_metrics.items():
                self._log_metric(stage, name, value, batch_size=batch_size)

        total_loss = None
        if "vicreg" in losses:
            vicreg_total = self.vicreg.weight * losses["vicreg"]
            total_loss = vicreg_total if total_loss is None else total_loss + vicreg_total
        if "lejepa" in losses and self.lejepa is not None:
            lejepa_total = self.lejepa.weight * losses["lejepa"]
            total_loss = lejepa_total if total_loss is None else total_loss + lejepa_total
        if "temporal_contrastive" in losses:
            temporal_contrastive_total = self.temporal_contrastive.weight * losses["temporal_contrastive"]
            total_loss = (
                temporal_contrastive_total
                if total_loss is None
                else total_loss + temporal_contrastive_total
            )
        if total_loss is None:
            total_loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)

        if stage != "train" and batch_idx == 0:
            parts = [f"[{stage}-diag] epoch={self.current_epoch} batch_idx=0"]
            for k, v in losses.items():
                parts.append(f"{k}={v.item():.6f}")
            parts.append(f"total={total_loss.item():.6f}")
            parts.append(f"active_losses={list(losses.keys())}")
            self._status_print(" | ".join(parts))

        if not torch.isfinite(total_loss).item():
            self._consecutive_nan_steps += 1
            self._log_metric(
                stage,
                "loss_nonfinite",
                1.0,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )
            if self._consecutive_nan_steps >= self._max_consecutive_nan_steps:
                raise RuntimeError(
                    f"Training produced {self._consecutive_nan_steps} consecutive "
                    f"non-finite losses. Halting to prevent silent divergence."
                )
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            self._consecutive_nan_steps = 0

        metrics_to_log = {"loss": total_loss}
        if "vicreg" in losses:
            metrics_to_log["vicreg"] = losses["vicreg"]
        if "lejepa" in losses:
            metrics_to_log["lejepa"] = losses["lejepa"]
        if "temporal_contrastive" in losses:
            metrics_to_log["temporal_contrastive"] = losses["temporal_contrastive"]

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
            log_kwargs["sync_dist"] = True
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


__all__ = ["TemporalSSLModule"]
