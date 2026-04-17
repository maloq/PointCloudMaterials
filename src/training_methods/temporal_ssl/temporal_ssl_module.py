import os
import sys

import pytorch_lightning as pl
import torch

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
from src.training_methods.temporal_ssl.lejepa import LeJEPALoss
from src.utils.pointcloud_ops import crop_to_num_points, shift_to_neighbor
from src.utils.training_utils import cached_sample_count, get_optimizers_and_scheduler


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

        self.encoder = build_encoder(cfg)
        self.encoder_io = EncoderAdapter(self.encoder)
        latent_dim = resolve_encoder_output_dim(cfg, encoder=self.encoder)

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

        init_supervised_cache(self, cfg)
        self.cache_train_supervised_metrics = bool(getattr(cfg, "cache_train_supervised_metrics", False))
        self._warned_cache_eq_fallback = False
        self._consecutive_nan_steps = 0
        self._max_consecutive_nan_steps = int(getattr(cfg, "max_consecutive_nan_steps", 20))

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
        encoded = self.encoder_io.encode(flat_points)
        z_inv_contrastive = self._contrastive_invariant_latent(
            encoded.invariant,
            encoded.equivariant,
        )
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
        encoded = self.encoder_io.encode(pc)
        z_inv_contrastive = self._contrastive_invariant_latent(
            encoded.invariant,
            encoded.equivariant,
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

        total_loss = None
        if "vicreg" in losses:
            vicreg_total = self.vicreg.weight * losses["vicreg"]
            total_loss = vicreg_total if total_loss is None else total_loss + vicreg_total
        if "lejepa" in losses and self.lejepa is not None:
            lejepa_total = self.lejepa.weight * losses["lejepa"]
            total_loss = lejepa_total if total_loss is None else total_loss + lejepa_total
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


__all__ = ["TemporalSSLModule"]
