import os
import sys

import pytorch_lightning as pl
import torch

sys.path.append(os.getcwd())

from src.models.autoencoders.factory import build_model
from src.training_methods.contrastive_learning.barlow_twins import BarlowTwinsLoss
from src.training_methods.contrastive_learning.pointcontrast import PointContrastLoss
from src.training_methods.contrastive_learning.supervised_cache import (
    cache_limit_for_stage,
    cache_supervised_batch,
    init_supervised_cache,
    log_supervised_metrics,
    reset_supervised_cache,
)
from src.training_methods.contrastive_learning.vicreg import VICRegLoss
from src.training_methods.contrastive_learning.wmse import WMSELoss
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

        self.barlow = BarlowTwinsLoss.from_config(
            cfg,
            input_dim=latent_dim,
        )
        vicreg_enabled = bool(getattr(cfg, "vicreg_enabled", False))
        self.vicreg = (
            VICRegLoss.from_config(
                cfg,
                input_dim=latent_dim,
            )
            if vicreg_enabled
            else None
        )
        self.wmse = WMSELoss.from_config(
            cfg,
            input_dim=latent_dim,
        )
        self.pointcontrast = PointContrastLoss.from_config(
            cfg,
            input_dim=latent_dim,
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
    def barlow_projector(self):
        return self.barlow.projector

    @property
    def vicreg_projector(self):
        return self.vicreg.projector if self.vicreg is not None else None

    @property
    def wmse_projector(self):
        return self.wmse.projector if self.wmse is not None else None

    @property
    def pointcontrast_projector(self):
        return self.pointcontrast.projector if self.pointcontrast is not None else None

    def _shared_invariant(self, z_inv_model, eq_z):
        # Contrastive training always prefers norms(eq_z) when eq_z exists and
        # otherwise falls back to the encoder invariant branch.
        return self.barlow._invariant(z_inv_model, eq_z)

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

    def _center_frame(self, sequence_points: torch.Tensor) -> torch.Tensor:
        return sequence_points[:, self.center_frame_index]

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
        if self.vicreg is None:
            return None

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
        pc_raw, prev_pc, next_pc, meta = self._unpack_temporal_batch(batch)
        batch_size = int(pc_raw.shape[0])
        pc_raw = pc_raw.to(device=self.device, dtype=self.dtype, non_blocking=True)
        prev_pc = prev_pc.to(device=self.device, dtype=self.dtype, non_blocking=True)
        next_pc = next_pc.to(device=self.device, dtype=self.dtype, non_blocking=True)
        pc = self._prepare_model_input(pc_raw)

        should_cache_stage = stage in self._supervised_cache and (
            stage != "train" or self.cache_train_supervised_metrics
        )

        losses = {}

        barlow_loss, barlow_metrics = self.barlow.compute_loss(
            pc=pc_raw,
            encoder=self.encoder,
            prepare_input=self._prepare_encoder_input,
            split_output=self._split_encoder_output,
            current_epoch=int(self.current_epoch),
            invariant_transform=self._shared_invariant,
        )
        if barlow_loss is not None:
            losses["barlow"] = barlow_loss
        for name, value in barlow_metrics.items():
            self._log_metric(stage, name, value, batch_size=batch_size)

        if self.vicreg is not None:
            vicreg_views = (
                self._build_vicreg_temporal_views(pc_raw, prev_pc, next_pc)
                if self.use_temporal_vicreg_views
                else None
            )
            vicreg_loss, vicreg_metrics = self.vicreg.compute_loss(
                pc=pc_raw,
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

        if self.wmse is not None:
            wmse_loss, wmse_metrics = self.wmse.compute_loss(
                pc=pc_raw,
                encoder=self.encoder,
                prepare_input=self._prepare_encoder_input,
                split_output=self._split_encoder_output,
                current_epoch=int(self.current_epoch),
                invariant_transform=self._shared_invariant,
            )
            if wmse_loss is not None:
                losses["wmse"] = wmse_loss
            for name, value in wmse_metrics.items():
                self._log_metric(stage, name, value, batch_size=batch_size)

        if self.pointcontrast is not None:
            pointcontrast_loss, pointcontrast_metrics = self.pointcontrast.compute_loss(
                pc=pc_raw,
                encoder=self.encoder,
                prepare_input=self._prepare_encoder_input,
                split_output=self._split_encoder_output,
                current_epoch=int(self.current_epoch),
                invariant_transform=self._shared_invariant,
            )
            if pointcontrast_loss is not None:
                losses["pointcontrast"] = pointcontrast_loss
            for name, value in pointcontrast_metrics.items():
                self._log_metric(stage, name, value, batch_size=batch_size)

        total_loss = None
        if "barlow" in losses:
            total_loss = self.barlow.weight * losses["barlow"]
        if "vicreg" in losses and self.vicreg is not None:
            vicreg_total = self.vicreg.weight * losses["vicreg"]
            total_loss = vicreg_total if total_loss is None else total_loss + vicreg_total
        if "wmse" in losses and self.wmse is not None:
            wmse_total = self.wmse.weight * losses["wmse"]
            total_loss = wmse_total if total_loss is None else total_loss + wmse_total
        if "pointcontrast" in losses and self.pointcontrast is not None:
            pointcontrast_total = self.pointcontrast.weight * losses["pointcontrast"]
            total_loss = pointcontrast_total if total_loss is None else total_loss + pointcontrast_total
        if total_loss is None:
            total_loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)

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
        if "barlow" in losses:
            metrics_to_log["barlow"] = losses["barlow"]
        if "vicreg" in losses:
            metrics_to_log["vicreg"] = losses["vicreg"]
        if "wmse" in losses:
            metrics_to_log["wmse"] = losses["wmse"]
        if "pointcontrast" in losses:
            metrics_to_log["pointcontrast"] = losses["pointcontrast"]

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
                with torch.no_grad():
                    z_inv_model, eq_z = self._split_encoder_output(
                        self.encoder(self._prepare_encoder_input(pc))
                    )
                    z_inv_contrastive = self._contrastive_invariant_from_eq_latent(
                        eq_z, z_inv_model=z_inv_model, stage=stage
                    )
                if z_inv_contrastive is not None:
                    self._cache_supervised_batch(
                        stage,
                        z_inv_contrastive,
                        meta,
                        encoder_features=z_inv_contrastive,
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
