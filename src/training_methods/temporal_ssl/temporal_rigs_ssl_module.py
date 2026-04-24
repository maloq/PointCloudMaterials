from __future__ import annotations

import torch

from src.training_methods.temporal_ssl.temporal_ssl_module import TemporalSSLModule
from src.utils.pointcloud_ops import shift_to_neighbor
from src.utils.training_utils import cached_sample_count


class TemporalRIGSSSLModule(TemporalSSLModule):
    """Temporal SSL with the standard training loop and a RIGS descriptor encoder."""

    def __init__(self, cfg):
        super().__init__(cfg)

        encoder_cfg = getattr(cfg, "encoder", None)
        self.encoder_name = str(getattr(encoder_cfg, "name", "")).strip()
        allowed_encoders = {"RIGS", "RIGS_NN"}
        if self.encoder_name not in allowed_encoders:
            raise ValueError(
                "TemporalRIGSSSLModule requires a RIGS descriptor encoder so local structures are "
                "encoded through scalar RIGS features. "
                f"Expected encoder.name in {sorted(allowed_encoders)}, got encoder.name={self.encoder_name!r}."
            )

    def _batch_has_precomputed_rigs_graph(self, batch: dict) -> bool:
        return self.encoder_name == "RIGS_NN" and isinstance(batch.get("rigs_graph"), dict)

    def _validate_graph_point_count(self, graph: dict[str, torch.Tensor], *, expected_points: int, context: str) -> None:
        got_points = int(graph["radii"].shape[-1])
        if got_points != int(expected_points):
            raise ValueError(
                f"{context} requires precomputed sparse RIGS graphs with {expected_points} points, got {got_points}. "
                "Set data.rigs_num_points to match the encoder input point count."
            )

    def _graph_to_device(self, graph: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for key, value in graph.items():
            if not torch.is_tensor(value):
                raise TypeError(f"Precomputed RIGS graph value {key!r} must be a torch.Tensor, got {type(value)}.")
            if key == "edge_index":
                out[key] = value.to(device=self.device, dtype=torch.long, non_blocking=True)
            else:
                out[key] = value.to(device=self.device, non_blocking=True)
        return out

    def _graph_frame(self, batch: dict, frame_idx: int) -> dict[str, torch.Tensor]:
        graph_seq = batch.get("rigs_graph")
        if not isinstance(graph_seq, dict):
            raise KeyError("Temporal batch does not contain precomputed sparse RIGS graphs under key 'rigs_graph'.")
        frame_graph = {key: value[:, frame_idx] for key, value in graph_seq.items()}
        return self._graph_to_device(frame_graph)

    def _select_graph_samples(self, graph: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if mask.dtype != torch.bool or mask.dim() != 1:
            raise ValueError(f"mask must be a 1D bool tensor, got dtype={mask.dtype} shape={tuple(mask.shape)}.")
        return {key: value[mask] for key, value in graph.items()}

    def _choose_between_graphs(
        self,
        graph_a: dict[str, torch.Tensor],
        graph_b: dict[str, torch.Tensor],
        choose_b_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if choose_b_mask.dtype != torch.bool or choose_b_mask.dim() != 1:
            raise ValueError(
                "choose_b_mask must be a 1D bool tensor when mixing precomputed sparse RIGS graphs, "
                f"got dtype={choose_b_mask.dtype} shape={tuple(choose_b_mask.shape)}."
            )
        out = {}
        for key in graph_a.keys():
            value_a = graph_a[key]
            value_b = graph_b[key]
            view_shape = [int(choose_b_mask.shape[0])] + [1] * (value_a.dim() - 1)
            out[key] = torch.where(choose_b_mask.view(*view_shape), value_b, value_a)
        return out

    def _encode_rigs_input(self, encoder_input) -> torch.Tensor:
        encoded = self.encoder_io.encode(encoder_input)
        invariant = self._contrastive_invariant_latent(
            encoded.invariant,
            encoded.equivariant,
        )
        if invariant is None:
            raise RuntimeError(
                "RIGS temporal SSL requires invariant encoder embeddings, but the configured encoder "
                "did not return a usable invariant latent."
            )
        return invariant

    def _graph_matches_current_encoder_input(self, batch: dict, *, expected_points: int, context: str) -> bool:
        if not self._batch_has_precomputed_rigs_graph(batch):
            return False
        self._validate_graph_point_count(batch["rigs_graph"], expected_points=expected_points, context=context)
        return True

    def _vicreg_precomputed_graph_compatible(self, batch: dict, *, expected_points: int) -> bool:
        if not self.use_temporal_vicreg_views:
            return False
        if not self._graph_matches_current_encoder_input(
            batch,
            expected_points=expected_points,
            context="Temporal RIGS VICReg precompute path",
        ):
            return False
        if self.vicreg.jitter_std > 0:
            return False
        if self.vicreg.drop_ratio > 0:
            return False
        if self.vicreg.strain_std > 0:
            return False
        if self.vicreg.occlusion_mode != "none":
            return False
        return True

    def _compute_vicreg_loss_with_precomputed_graphs(
        self,
        *,
        batch: dict,
        center_pc: torch.Tensor,
        prev_pc: torch.Tensor,
        next_pc: torch.Tensor,
        expected_points: int,
    ):
        if self.vicreg.projector is None:
            raise RuntimeError("VICReg projector is not initialized, so precomputed RIGS VICReg cannot run.")
        if not self._vicreg_precomputed_graph_compatible(batch, expected_points=int(expected_points)):
            raise RuntimeError("Precomputed RIGS VICReg was requested for an incompatible batch/configuration.")

        anchor_graph = self._graph_frame(batch, self.center_frame_index)
        prev_graph = self._graph_frame(batch, self.center_frame_index - 1)
        next_graph = self._graph_frame(batch, self.center_frame_index + 1)

        inv_a = self._encode_rigs_input(anchor_graph)

        batch_size = int(center_pc.shape[0])
        choose_next_mask = torch.rand((batch_size,), device=self.device) < 0.5
        temporal_graph = self._choose_between_graphs(prev_graph, next_graph, choose_next_mask)

        if not self.vicreg.neighbor_view or str(self.vicreg.neighbor_view_mode).lower() == "none":
            inv_b = self._encode_rigs_input(temporal_graph)
        else:
            neighbor_mode = str(self.vicreg.neighbor_view_mode).lower()
            if neighbor_mode != "second":
                raise ValueError(
                    "Precomputed temporal RIGS VICReg currently supports neighbor_view_mode in {'none', 'second'} "
                    f"when temporal_vicreg_use_adjacent_views=true, got {neighbor_mode!r}."
                )
            temporal_mask = torch.rand((batch_size,), device=self.device) < 0.5
            inv_b = torch.empty_like(inv_a)

            if bool(temporal_mask.any().item()):
                temporal_graph_subset = self._select_graph_samples(temporal_graph, temporal_mask)
                inv_b[temporal_mask] = self._encode_rigs_input(temporal_graph_subset)

            spatial_mask = ~temporal_mask
            if bool(spatial_mask.any().item()):
                spatial_view = self._prepare_explicit_view(
                    shift_to_neighbor(
                        center_pc[spatial_mask],
                        neighbor_k=self.vicreg.neighbor_k,
                        max_relative_distance=self.vicreg.neighbor_max_relative_distance,
                    ),
                    target_points=self.vicreg.view_points,
                )
                spatial_view = self.vicreg.apply_view_postprocessing(
                    spatial_view,
                    use_neighbor=True,
                    apply_occlusion=False,
                )
                inv_b[spatial_mask] = self._encode_rigs_input(spatial_view)

        proj_dtype = next(self.vicreg.projector.parameters()).dtype
        z_a = self.vicreg.projector(inv_a.to(dtype=proj_dtype))
        z_b = self.vicreg.projector(inv_b.to(dtype=proj_dtype))
        loss, metrics = self.vicreg._loss(z_a, z_b)
        if not torch.isfinite(loss).item():
            metrics["vicreg_nonfinite"] = center_pc.new_tensor(1.0)
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss, metrics

    def forward(self, pc, include_ssl_heads: bool = False):
        if not isinstance(include_ssl_heads, bool):
            raise TypeError(
                "include_ssl_heads must be a bool. It is intended only for PyTorch Lightning "
                f"model-summary FLOP profiling, got {type(include_ssl_heads)}."
            )
        if isinstance(pc, dict) and "rigs_graph" in pc:
            pc = pc["rigs_graph"]
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
                f"TemporalRIGSSSLModule expects dict batches from TemporalLAMMPSDumpDataset, got {type(batch)}."
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

        expected_points = int(pc.shape[1])
        center_encoder_input = pc
        if self._graph_matches_current_encoder_input(
            batch,
            expected_points=expected_points,
            context="Temporal RIGS center-frame encode",
        ):
            center_encoder_input = self._graph_frame(batch, self.center_frame_index)

        should_cache_stage = stage in self._supervised_cache and (
            stage != "train" or self.cache_train_supervised_metrics
        )
        should_run_vicreg = self.vicreg.should_run(current_epoch=int(self.current_epoch))
        should_run_swav = self.swav.should_run(current_epoch=int(self.current_epoch))

        losses = {}
        center_embeddings = None

        if should_cache_stage:
            with torch.no_grad():
                encoded = self.encoder_io.encode(center_encoder_input)
                center_embeddings = self._contrastive_invariant_from_eq_latent(
                    encoded.equivariant,
                    z_inv_model=encoded.invariant,
                    stage=stage,
                )

        if should_run_vicreg:
            vicreg_expected_points = (
                int(self.vicreg.view_points)
                if self.vicreg.view_points is not None
                else int(vicreg_pc_raw.shape[1])
            )
            if self._vicreg_precomputed_graph_compatible(batch, expected_points=vicreg_expected_points):
                vicreg_loss, vicreg_metrics = self._compute_vicreg_loss_with_precomputed_graphs(
                    batch=batch,
                    center_pc=pc_raw,
                    prev_pc=prev_pc,
                    next_pc=next_pc,
                    expected_points=vicreg_expected_points,
                )
            else:
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


__all__ = ["TemporalRIGSSSLModule"]
