import copy
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import EncoderAdapter, build_encoder, resolve_encoder_output_dim
from src.training_methods.contrastive_learning.vicreg import VICRegLoss
from src.training_methods.successor_vicreg.data_module import (
    SuccessorTemporalLAMMPSDataModule,
    resolve_temporal_lammps_radius,
)
from src.utils.pointcloud_ops import crop_to_num_points
from src.utils.training_utils import get_optimizers_and_scheduler


class SuccessorPredictor(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"Successor predictor input_dim must be > 0, got {input_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"Successor predictor hidden_dim must be > 0, got {hidden_dim}.")
        if num_layers <= 0:
            raise ValueError(f"Successor predictor num_layers must be > 0, got {num_layers}.")
        if output_dim <= 0:
            raise ValueError(f"Successor predictor output_dim must be > 0, got {output_dim}.")

        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(num_layers) - 1):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.GELU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(
                "Successor predictor expects a 2D latent tensor (B, D), "
                f"got {tuple(x.shape)}."
            )
        return self.net(x)


class SuccessorVICRegModule(pl.LightningModule):
    """Temporal VICReg with a discounted future-field predictor head."""

    data_module_class = SuccessorTemporalLAMMPSDataModule

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        data_cfg = getattr(cfg, "data", None)
        if data_cfg is None:
            raise ValueError("SuccessorVICRegModule requires a data configuration.")
        data_kind = str(getattr(data_cfg, "kind", "")).strip().lower()
        if data_kind != "temporal_lammps":
            raise ValueError(
                "SuccessorVICRegModule requires data.kind='temporal_lammps', "
                f"got {data_kind!r}."
            )

        self.sequence_length = int(getattr(data_cfg, "sequence_length", 0))
        if self.sequence_length <= 0:
            raise ValueError(
                f"SuccessorVICRegModule requires data.sequence_length > 0, got {self.sequence_length}."
            )

        self.enable_successor = bool(getattr(cfg, "enable_successor", False))
        self.successor_horizon_H = int(getattr(cfg, "successor_horizon_H", 1))
        if self.successor_horizon_H <= 0:
            raise ValueError(
                f"successor_horizon_H must be > 0, got {self.successor_horizon_H}."
            )
        if self.enable_successor and self.sequence_length < self.successor_horizon_H + 1:
            raise ValueError(
                "Successor-VICReg requires the temporal batch to include the current frame and H futures. "
                f"Got sequence_length={self.sequence_length}, successor_horizon_H={self.successor_horizon_H}."
            )

        self.successor_gamma = float(getattr(cfg, "successor_gamma", 0.95))
        if not (0.0 < self.successor_gamma <= 1.0):
            raise ValueError(
                f"successor_gamma must be in (0, 1], got {self.successor_gamma}."
            )
        self.successor_lambda = float(getattr(cfg, "successor_lambda", 1.0))
        if self.successor_lambda < 0.0:
            raise ValueError(
                f"successor_lambda must be >= 0, got {self.successor_lambda}."
            )
        self.successor_use_ema_teacher = bool(getattr(cfg, "successor_use_ema_teacher", True))
        self.successor_predictor_hidden_dim = int(
            getattr(cfg, "successor_predictor_hidden_dim", 512)
        )
        self.successor_predictor_num_layers = int(
            getattr(cfg, "successor_predictor_num_layers", 2)
        )
        self.successor_lookup_chunk_size = int(
            getattr(cfg, "successor_lookup_chunk_size", 2048)
        )
        if self.successor_lookup_chunk_size <= 0:
            raise ValueError(
                "successor_lookup_chunk_size must be > 0, "
                f"got {self.successor_lookup_chunk_size}."
            )
        self.successor_use_concat_z_and_successor_for_clustering = bool(
            getattr(cfg, "successor_use_concat_z_and_successor_for_clustering", False)
        )

        self.encoder = build_encoder(cfg)
        self.encoder_io = EncoderAdapter(self.encoder)
        latent_dim = resolve_encoder_output_dim(cfg, encoder=self.encoder)
        if latent_dim is None:
            raise ValueError(
                "Successor-VICReg could not resolve the encoder latent dimension from the config."
            )
        self.latent_dim = int(latent_dim)

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
                f"model_points ({self.model_points}) cannot exceed data.num_points ({self.sample_points})."
            )

        self.vicreg = VICRegLoss.from_config(
            cfg,
            input_dim=self.latent_dim,
        )
        self.successor_predictor = SuccessorPredictor(
            input_dim=self.latent_dim,
            hidden_dim=self.successor_predictor_hidden_dim,
            num_layers=self.successor_predictor_num_layers,
            output_dim=self.latent_dim,
        )

        self.successor_teacher_ema_decay = self._resolve_teacher_ema_decay(cfg)
        self.teacher_encoder = copy.deepcopy(self.encoder) if self.successor_use_ema_teacher else None
        self.teacher_encoder_io = (
            EncoderAdapter(self.teacher_encoder) if self.teacher_encoder is not None else None
        )
        self._freeze_teacher_encoder()

        self._consecutive_nan_steps = 0
        self._max_consecutive_nan_steps = int(getattr(cfg, "max_consecutive_nan_steps", 20))
        self._lookup_dataset = None
        self._lookup_frame_to_slot: dict[int, int] | None = None

    @staticmethod
    def _resolve_teacher_ema_decay(cfg) -> float:
        encoder_cfg = getattr(cfg, "encoder", None)
        encoder_kwargs = getattr(encoder_cfg, "kwargs", None) if encoder_cfg is not None else None
        default_decay = 0.996
        if encoder_kwargs is None:
            return float(default_decay)
        decay = encoder_kwargs.get("ema_decay", default_decay)
        decay = float(decay)
        if not (0.0 < decay < 1.0):
            raise ValueError(
                f"Teacher EMA decay must be in (0, 1), got {decay}."
            )
        return decay

    def _freeze_teacher_encoder(self) -> None:
        if self.teacher_encoder is None:
            return
        self.teacher_encoder.eval()
        for parameter in self.teacher_encoder.parameters():
            parameter.requires_grad_(False)

    def _status_print(self, message: str) -> None:
        if getattr(self, "_trainer", None) is not None:
            self.print(message)
            return
        print(message)

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
                "[successor-vicreg] Warning: init checkpoint encoder differs from current config: "
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
            self._status_print(f"[successor-vicreg] Initialized module weights from checkpoint: {checkpoint_path}")
            self._status_print(
                "[successor-vicreg] Checkpoint load summary: "
                f"target=module, "
                f"loaded={len(selected_state_dict)} / model_tensors={len(model_state)}, "
                f"shape_mismatch_skipped={len(shape_mismatch)}, "
                f"missing_after_load={len(missing_keys)}, "
                f"unexpected_after_load={len(unexpected_keys)}, "
                f"stripped_prefixes={used_stripped_prefixes}, "
                f"strict={strict}"
            )
        else:
            candidate_label, selected_state_dict, shape_mismatch = self._select_encoder_state_dict(source_state_dict)
            if not selected_state_dict:
                raise RuntimeError(
                    f"No compatible encoder tensors found when loading checkpoint '{checkpoint_path}'. "
                    "Expected either encoder-prefixed keys or a raw encoder state_dict with matching tensor shapes."
                )

            missing_keys, unexpected_keys = self.encoder.load_state_dict(selected_state_dict, strict=strict)
            encoder_state = self.encoder.state_dict()
            self._status_print(f"[successor-vicreg] Initialized encoder weights from checkpoint: {checkpoint_path}")
            self._status_print(
                "[successor-vicreg] Checkpoint load summary: "
                f"target=encoder, "
                f"source={candidate_label}, "
                f"loaded={len(selected_state_dict)} / encoder_tensors={len(encoder_state)}, "
                f"shape_mismatch_skipped={len(shape_mismatch)}, "
                f"missing_after_load={len(missing_keys)}, "
                f"unexpected_after_load={len(unexpected_keys)}, "
                f"strict={strict}"
            )

        if self.teacher_encoder is not None:
            self.teacher_encoder.load_state_dict(self.encoder.state_dict())
            self._freeze_teacher_encoder()

    def _prepare_model_input(self, pc: torch.Tensor) -> torch.Tensor:
        out = pc
        if self.model_points is not None:
            out = crop_to_num_points(out, self.model_points)
        return out

    def _shared_invariant(self, z_inv_model, eq_z):
        return self.vicreg._invariant(z_inv_model, eq_z)

    def _encode_invariant_latents(
        self,
        pc: torch.Tensor,
        *,
        teacher: bool,
    ) -> torch.Tensor:
        if pc.dim() != 3 or pc.shape[-1] != 3:
            raise ValueError(
                "Expected point clouds with shape (B, N, 3) when encoding invariants, "
                f"got {tuple(pc.shape)}."
            )
        model_input = self._prepare_model_input(pc)
        encoder_io = self.teacher_encoder_io if teacher and self.teacher_encoder_io is not None else self.encoder_io
        encoded = encoder_io.encode(model_input)
        z_inv = self._shared_invariant(encoded.invariant, encoded.equivariant)
        if z_inv is None:
            raise RuntimeError("Encoder did not produce a usable invariant latent.")
        return z_inv

    def encode_current_invariant(self, pc: torch.Tensor) -> torch.Tensor:
        return self._encode_invariant_latents(pc, teacher=False)

    @torch.no_grad()
    def encode_teacher_invariant(self, pc: torch.Tensor) -> torch.Tensor:
        return self._encode_invariant_latents(pc, teacher=self.successor_use_ema_teacher).detach().to(torch.float32)

    def predict_successor(self, z_inv: torch.Tensor) -> torch.Tensor:
        predictor_dtype = next(self.successor_predictor.parameters()).dtype
        prediction = self.successor_predictor(z_inv.to(dtype=predictor_dtype))
        return prediction.to(torch.float32)

    def predict_successor_from_points(self, pc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pc.dim() == 4:
            self._validate_temporal_points(pc)
            pc = pc[:, 0]
        z_inv = self.encode_current_invariant(pc)
        return z_inv, self.predict_successor(z_inv)

    def forward(self, pc: torch.Tensor):
        if torch.is_tensor(pc) and pc.dim() == 4:
            self._validate_temporal_points(pc)
            pc = pc[:, 0]
        encoded = self.encoder_io.encode(self._prepare_model_input(pc))
        z_inv_contrastive = self._shared_invariant(encoded.invariant, encoded.equivariant)
        return z_inv_contrastive, encoded.invariant, encoded.equivariant

    def _validate_temporal_points(self, sequence_points: torch.Tensor) -> None:
        if not torch.is_tensor(sequence_points):
            raise TypeError(f"sequence_points must be a torch.Tensor, got {type(sequence_points)}.")
        if sequence_points.dim() != 4 or sequence_points.shape[-1] != 3:
            raise ValueError(
                "Expected temporal point clouds with shape (B, T, N, 3), "
                f"got {tuple(sequence_points.shape)}."
            )

    def _unpack_batch(self, batch: dict) -> tuple[torch.Tensor, dict]:
        if not isinstance(batch, dict):
            raise TypeError(
                f"SuccessorVICRegModule expects dict batches from TemporalLAMMPSDumpDataset, got {type(batch)}."
            )
        if "points" not in batch:
            raise KeyError("Temporal batch is missing required key 'points'.")
        if "local_atom_ids" not in batch:
            raise KeyError(
                "Temporal batch is missing required key 'local_atom_ids'. "
                "Successor-VICReg needs neighborhood atom ids to build future-neighborhood targets."
            )
        sequence_points = batch["points"]
        self._validate_temporal_points(sequence_points)
        return sequence_points, {
            "frame_indices": batch.get("frame_indices"),
            "center_atom_id": batch.get("center_atom_id"),
            "source_path": batch.get("source_path"),
            "local_atom_ids": batch.get("local_atom_ids"),
        }

    @staticmethod
    def _extract_neighborhood_atom_ids(local_atom_ids: torch.Tensor) -> list[int]:
        atom_ids_np = np.asarray(local_atom_ids.detach().cpu(), dtype=np.int64).reshape(-1)
        if atom_ids_np.size == 0:
            raise ValueError(
                "local_atom_ids must contain at least one atom id per neighborhood, "
                f"got shape={tuple(local_atom_ids.shape)}."
            )
        return [int(v) for v in atom_ids_np.tolist()]

    def _get_lookup_dataset(self):
        if self._lookup_dataset is not None:
            return self._lookup_dataset

        data_cfg = self.hparams.data
        radius = resolve_temporal_lammps_radius(
            dump_file=getattr(data_cfg, "dump_file"),
            data_cfg=data_cfg,
            frame_start=int(getattr(data_cfg, "frame_start", 0)),
            num_points=int(getattr(data_cfg, "num_points")),
        )
        from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset

        self._lookup_dataset = TemporalLAMMPSDumpDataset(
            dump_file=getattr(data_cfg, "dump_file"),
            sequence_length=1,
            num_points=int(getattr(data_cfg, "num_points")),
            radius=float(radius),
            frame_stride=1,
            window_stride=1,
            frame_start=int(getattr(data_cfg, "frame_start", 0)),
            frame_stop=getattr(data_cfg, "frame_stop", None),
            center_selection_mode="atom_stride",
            center_atom_stride=1,
            normalize=bool(getattr(data_cfg, "normalize", True)),
            center_neighborhoods=bool(getattr(data_cfg, "center_neighborhoods", True)),
            selection_method=str(getattr(data_cfg, "selection_method", "closest")),
            cache_dir=getattr(data_cfg, "cache_dir", None),
            rebuild_cache=False,
            tree_cache_size=int(getattr(data_cfg, "tree_cache_size", 4)),
            precompute_neighbor_indices=False,
            build_lock_timeout_sec=float(getattr(data_cfg, "build_lock_timeout_sec", 7200.0)),
            build_lock_stale_sec=float(getattr(data_cfg, "build_lock_stale_sec", 86400.0)),
        )
        self._lookup_frame_to_slot = {
            int(frame_idx): int(slot)
            for slot, frame_idx in enumerate(self._lookup_dataset.window_start_frames.tolist())
        }
        return self._lookup_dataset

    @torch.no_grad()
    def _lookup_teacher_latents(
        self,
        required_pairs: list[tuple[int, int]],
    ) -> dict[tuple[int, int], torch.Tensor]:
        if not required_pairs:
            return {}

        lookup_dataset = self._get_lookup_dataset()
        if self._lookup_frame_to_slot is None:
            raise RuntimeError("Lookup dataset frame-slot mapping was not initialized.")
        resolved: dict[tuple[int, int], torch.Tensor] = {}
        chunk_size = int(self.successor_lookup_chunk_size)
        for start in range(0, len(required_pairs), chunk_size):
            chunk_pairs = required_pairs[start : start + chunk_size]
            frame_indices = np.asarray([pair[0] for pair in chunk_pairs], dtype=np.int64)
            atom_ids = np.asarray([pair[1] for pair in chunk_pairs], dtype=np.int64)
            atom_positions = np.searchsorted(lookup_dataset.atom_ids, atom_ids)
            valid_atom_positions = (atom_positions >= 0) & (atom_positions < int(lookup_dataset.num_atoms))
            if not np.all(valid_atom_positions):
                missing = atom_ids[~valid_atom_positions]
                raise ValueError(
                    "Some required successor target atom ids were outside the lookup dataset atom range. "
                    f"missing_atom_ids={missing.tolist()}."
                )
            matched_atom_ids = lookup_dataset.atom_ids[atom_positions]
            if not np.array_equal(matched_atom_ids, atom_ids):
                raise ValueError(
                    "Some required successor target atom ids were not found in the lookup dataset. "
                    f"requested_atom_ids={atom_ids.tolist()}."
                )

            sample_indices = []
            for frame_idx, atom_slot in zip(frame_indices.tolist(), atom_positions.tolist(), strict=True):
                frame_slot = self._lookup_frame_to_slot.get(int(frame_idx))
                if frame_slot is None:
                    raise ValueError(
                        "Required successor target frame is outside the lookup dataset frame range. "
                        f"frame_idx={frame_idx}."
                    )
                sample_indices.append(int(frame_slot) * int(lookup_dataset.center_count) + int(atom_slot))

            fetched = lookup_dataset.__getitems__(sample_indices)
            points = fetched["points"]
            if not torch.is_tensor(points):
                points = torch.as_tensor(points)
            if points.dim() != 4 or points.shape[1] != 1 or points.shape[-1] != 3:
                raise ValueError(
                    "Lookup dataset returned an unexpected neighborhood tensor shape. "
                    f"Expected (B, 1, N, 3), got {tuple(points.shape)}."
                )
            teacher_latents = self.encode_teacher_invariant(
                points[:, 0].to(device=self.device, dtype=self.dtype, non_blocking=True)
            )
            for idx, (frame_idx, atom_id) in enumerate(chunk_pairs):
                resolved[(int(frame_idx), int(atom_id))] = teacher_latents[idx]
        return resolved

    @torch.no_grad()
    def _build_batch_teacher_cache(
        self,
        future_points: torch.Tensor,
        future_frame_indices: torch.Tensor,
        center_atom_ids: torch.Tensor,
    ) -> dict[tuple[int, int], torch.Tensor]:
        batch_size, horizon, num_points, coord_dim = future_points.shape
        if coord_dim != 3:
            raise ValueError(
                f"Future points must have last dimension 3, got {tuple(future_points.shape)}."
            )
        teacher_latents = self.encode_teacher_invariant(
            future_points.reshape(batch_size * horizon, num_points, coord_dim)
        )
        flat_frames = future_frame_indices.reshape(-1).detach().cpu().tolist()
        flat_center_atom_ids = (
            center_atom_ids.reshape(-1, 1)
            .expand(-1, horizon)
            .reshape(-1)
            .detach()
            .cpu()
            .tolist()
        )
        cache: dict[tuple[int, int], torch.Tensor] = {}
        for latent_idx, (frame_idx, atom_id) in enumerate(
            zip(flat_frames, flat_center_atom_ids, strict=True)
        ):
            cache[(int(frame_idx), int(atom_id))] = teacher_latents[latent_idx]
        return cache

    def _compute_successor_targets(
        self,
        *,
        future_points: torch.Tensor,
        future_frame_indices: torch.Tensor,
        local_atom_ids: torch.Tensor,
        center_atom_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = int(future_points.shape[0])
        horizon = int(future_points.shape[1])
        if horizon != self.successor_horizon_H:
            raise ValueError(
                "Future-point horizon mismatch while computing successor targets. "
                f"expected={self.successor_horizon_H}, got={horizon}."
            )

        teacher_cache = self._build_batch_teacher_cache(
            future_points=future_points,
            future_frame_indices=future_frame_indices,
            center_atom_ids=center_atom_ids,
        )

        neighborhood_records: list[list[tuple[int, list[int]]]] = []
        missing_pairs: set[tuple[int, int]] = set()
        for batch_idx in range(batch_size):
            per_sample_records = []
            for delta_idx in range(horizon):
                frame_idx = int(future_frame_indices[batch_idx, delta_idx].item())
                neighborhood_atom_ids = self._extract_neighborhood_atom_ids(
                    local_atom_ids[batch_idx, delta_idx]
                )
                per_sample_records.append((frame_idx, neighborhood_atom_ids))
                for atom_id in neighborhood_atom_ids:
                    if (frame_idx, atom_id) not in teacher_cache:
                        missing_pairs.add((frame_idx, atom_id))
            neighborhood_records.append(per_sample_records)

        if missing_pairs:
            teacher_cache.update(self._lookup_teacher_latents(sorted(missing_pairs)))

        target_dim = self.latent_dim
        targets = future_points.new_zeros((batch_size, target_dim), dtype=torch.float32)
        valid_mask = torch.ones((batch_size,), device=future_points.device, dtype=torch.bool)
        discount_values = future_points.new_tensor(
            [self.successor_gamma ** delta_idx for delta_idx in range(horizon)],
            dtype=torch.float32,
        )

        for batch_idx, per_sample_records in enumerate(neighborhood_records):
            sample_target = targets.new_zeros((target_dim,))
            for delta_idx, (frame_idx, neighborhood_atom_ids) in enumerate(
                per_sample_records
            ):
                neighborhood_latents = torch.stack(
                    [teacher_cache[(frame_idx, atom_id)] for atom_id in neighborhood_atom_ids],
                    dim=0,
                )
                successor_field = neighborhood_latents.mean(dim=0)
                sample_target = sample_target + discount_values[delta_idx] * successor_field
            targets[batch_idx] = sample_target

        metrics = {
            "successor_valid_samples": valid_mask.to(torch.float32).sum(),
            "successor_missing_lookup_pairs": future_points.new_tensor(float(len(missing_pairs))),
            "successor_target_norm": targets.norm(dim=1).mean(),
        }
        return targets, valid_mask, metrics

    def _compute_successor_loss(
        self,
        *,
        current_embeddings: torch.Tensor,
        sequence_points: torch.Tensor,
        frame_indices: torch.Tensor,
        center_atom_ids: torch.Tensor,
        local_atom_ids: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if not self.enable_successor or self.successor_lambda <= 0.0:
            return None, {}

        future_points = sequence_points[:, 1 : self.successor_horizon_H + 1]
        future_frame_indices = frame_indices[:, 1 : self.successor_horizon_H + 1]
        future_local_atom_ids = local_atom_ids[:, 1 : self.successor_horizon_H + 1]
        if future_points.shape[1] != self.successor_horizon_H:
            raise ValueError(
                "Temporal batch does not expose enough future frames to compute successor loss. "
                f"required_horizon={self.successor_horizon_H}, available_future_frames={int(future_points.shape[1])}."
            )

        targets, valid_mask, target_metrics = self._compute_successor_targets(
            future_points=future_points,
            future_frame_indices=future_frame_indices,
            local_atom_ids=future_local_atom_ids,
            center_atom_ids=center_atom_ids,
        )
        predictions = self.predict_successor(current_embeddings)
        if predictions.shape != targets.shape:
            raise ValueError(
                "Successor predictor output shape mismatch. "
                f"predictions.shape={tuple(predictions.shape)}, targets.shape={tuple(targets.shape)}."
            )
        if not valid_mask.any():
            zero = predictions.new_zeros(())
            metrics = dict(target_metrics)
            metrics["successor_loss"] = zero
            metrics["successor_prediction_norm"] = zero
            return zero, metrics

        successor_loss = F.smooth_l1_loss(
            predictions[valid_mask],
            targets[valid_mask].detach(),
            reduction="mean",
        )
        metrics = dict(target_metrics)
        metrics["successor_loss"] = successor_loss
        metrics["successor_prediction_norm"] = predictions.norm(dim=1).mean()
        return successor_loss, metrics

    def _step(self, batch, batch_idx, stage: str):
        sequence_points, meta = self._unpack_batch(batch)
        batch_size = int(sequence_points.shape[0])
        sequence_points = sequence_points.to(device=self.device, dtype=self.dtype, non_blocking=True)
        current_points = sequence_points[:, 0]

        frame_indices = meta.get("frame_indices")
        center_atom_ids = meta.get("center_atom_id")
        local_atom_ids = meta.get("local_atom_ids")
        if frame_indices is None:
            raise KeyError("Successor-VICReg requires batch['frame_indices'].")
        if center_atom_ids is None:
            raise KeyError("Successor-VICReg requires batch['center_atom_id'].")
        if local_atom_ids is None:
            raise KeyError("Successor-VICReg requires batch['local_atom_ids'].")
        if not torch.is_tensor(frame_indices):
            frame_indices = torch.as_tensor(frame_indices)
        if not torch.is_tensor(center_atom_ids):
            center_atom_ids = torch.as_tensor(center_atom_ids)
        if not torch.is_tensor(local_atom_ids):
            local_atom_ids = torch.as_tensor(local_atom_ids)
        frame_indices = frame_indices.to(device=self.device, dtype=torch.int64, non_blocking=True)
        center_atom_ids = center_atom_ids.to(device=self.device, dtype=torch.int64, non_blocking=True)
        local_atom_ids = local_atom_ids.to(device=self.device, dtype=torch.int64, non_blocking=True)

        current_embeddings = self.encode_current_invariant(current_points)

        losses = {}
        vicreg_loss, vicreg_metrics = self.vicreg.compute_loss(
            pc=current_points,
            encoder=self.encoder,
            prepare_input=self.encoder_io.prepare_input,
            split_output=self.encoder_io.split_output,
            current_epoch=int(self.current_epoch),
            invariant_transform=self._shared_invariant,
        )
        if vicreg_loss is not None:
            losses["vicreg"] = vicreg_loss
        for name, value in vicreg_metrics.items():
            self._log_metric(stage, name, value, batch_size=batch_size)

        successor_loss, successor_metrics = self._compute_successor_loss(
            current_embeddings=current_embeddings,
            sequence_points=sequence_points,
            frame_indices=frame_indices,
            center_atom_ids=center_atom_ids,
            local_atom_ids=local_atom_ids,
        )
        if successor_loss is not None:
            losses["successor"] = successor_loss
        for name, value in successor_metrics.items():
            self._log_metric(stage, name, value, batch_size=batch_size)

        total_loss = None
        if "vicreg" in losses:
            total_loss = self.vicreg.weight * losses["vicreg"]
        if "successor" in losses:
            successor_total = self.successor_lambda * losses["successor"]
            total_loss = successor_total if total_loss is None else total_loss + successor_total
        if total_loss is None:
            total_loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)

        if stage != "train" and batch_idx == 0:
            parts = [f"[{stage}-diag] epoch={self.current_epoch} batch_idx=0"]
            for key, value in losses.items():
                parts.append(f"{key}={value.item():.6f}")
            parts.append(f"total={total_loss.item():.6f}")
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
                    f"Training produced {self._consecutive_nan_steps} consecutive non-finite losses. "
                    "Halting to prevent silent divergence."
                )
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            self._consecutive_nan_steps = 0

        metrics_to_log = {"loss": total_loss}
        if "vicreg" in losses:
            metrics_to_log["vicreg"] = losses["vicreg"]
        if "successor" in losses:
            metrics_to_log["successor"] = losses["successor"]
        prog_bar_keys = {"loss"}
        for name, value in metrics_to_log.items():
            self._log_metric(
                stage,
                name,
                value,
                prog_bar=(name in prog_bar_keys),
                batch_size=batch_size,
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
            on_step = stage == "train"
        if on_epoch is None:
            on_epoch = stage != "train"
        log_kwargs = dict(kwargs)
        if batch_size is not None and "batch_size" not in log_kwargs:
            log_kwargs["batch_size"] = int(batch_size)
        if "sync_dist" not in log_kwargs and stage != "train":
            log_kwargs["sync_dist"] = True
        self.log(f"{stage}/{name}", value, on_step=on_step, on_epoch=on_epoch, **log_kwargs)

    @torch.no_grad()
    def _update_teacher_encoder(self) -> None:
        if self.teacher_encoder is None:
            return
        decay = self.successor_teacher_ema_decay
        for teacher_param, online_param in zip(
            self.teacher_encoder.parameters(),
            self.encoder.parameters(),
            strict=True,
        ):
            teacher_param.data.mul_(decay).add_(online_param.data, alpha=1.0 - decay)
        for teacher_buffer, online_buffer in zip(
            self.teacher_encoder.buffers(),
            self.encoder.buffers(),
            strict=True,
        ):
            teacher_buffer.copy_(online_buffer)

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        if self.enable_successor and self.successor_use_ema_teacher:
            self._update_teacher_encoder()

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        super().on_save_checkpoint(checkpoint)
        if self.teacher_encoder is not None:
            checkpoint["successor_teacher_encoder_state_dict"] = self.teacher_encoder.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        super().on_load_checkpoint(checkpoint)
        if self.teacher_encoder is None:
            return
        teacher_state_dict = checkpoint.get("successor_teacher_encoder_state_dict")
        if teacher_state_dict is None:
            self.teacher_encoder.load_state_dict(self.encoder.state_dict())
            self._freeze_teacher_encoder()
            self._status_print(
                "[successor-vicreg] Warning: checkpoint did not contain an EMA teacher state. "
                "Resetting the teacher encoder from the online encoder."
            )
            return
        self.teacher_encoder.load_state_dict(teacher_state_dict)
        self._freeze_teacher_encoder()


__all__ = ["SuccessorVICRegModule"]
