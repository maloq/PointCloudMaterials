import torch

from src.training_methods.base_ssl_module import BaseSSLModule
from src.utils.pointcloud_ops import crop_to_num_points, shift_to_neighbor


class TemporalSSLModule(BaseSSLModule):
    """Temporal self-supervised training on local-structure frame sequences."""

    test_metrics_on_step = True

    def __init__(self, cfg):
        self.sequence_length = cfg.data.sequence_length
        self.center_frame_index = self.sequence_length // 2
        self.use_temporal_vicreg_views = cfg.temporal_vicreg_use_adjacent_views
        self.temporal_vicreg_anchor_frame = cfg.temporal_vicreg_anchor_frame
        self.fused_views_anchor = cfg.fused_views_anchor

        super().__init__(
            cfg,
            module_name="TemporalSSLModule",
            summary_sequence_length=self.sequence_length,
        )
        self.cache_warning_prefix = "temporal-ssl"
        self._validate_current_config()

    def _validate_current_config(self) -> None:
        if self.sequence_length < 3 or self.sequence_length % 2 != 1:
            raise ValueError(
                "TemporalSSLModule requires an odd data.sequence_length >= 3 so the "
                "center frame has both a previous and a next frame, "
                f"got {self.sequence_length}."
            )
        if self.temporal_vicreg_anchor_frame not in {"center", "last"}:
            raise ValueError(
                "temporal_vicreg_anchor_frame must be 'center' or 'last', "
                f"got {self.temporal_vicreg_anchor_frame!r}."
            )
        if self.fused_views_anchor not in {"center", "previous"}:
            raise ValueError(
                "fused_views_anchor must be 'center' or 'previous', "
                f"got {self.fused_views_anchor!r}."
            )
        if self.vicreg.enabled and self.vicreg.view_points is None:
            raise ValueError("Temporal VICReg requires an explicit vicreg_view_points value.")
        if self.swav.enabled and self.swav.view_points is None:
            raise ValueError("Temporal SwAV requires an explicit swav_view_points value.")
        if not (
            self.use_temporal_vicreg_views
            and self.vicreg.enabled
            and self.swav.enabled
        ):
            return
        expected_swav_mode = (
            "center_adjacent" if self.fused_views_anchor == "center" else "adjacent"
        )
        if self.swav.view_mode != expected_swav_mode:
            raise ValueError(
                "Fused temporal views must match their SwAV meaning: "
                f"fused_views_anchor={self.fused_views_anchor!r} requires "
                f"swav_view_mode={expected_swav_mode!r}, got {self.swav.view_mode!r}."
            )
        if self.vicreg.view_points != self.swav.view_points:
            raise ValueError(
                "Fused temporal VICReg and SwAV views require the same point count, "
                f"got vicreg_view_points={self.vicreg.view_points} and "
                f"swav_view_points={self.swav.view_points}."
            )
        if self.vicreg.requires_overlap_target:
            raise ValueError(
                "Temporal fused views do not produce the sampled-index overlap target "
                "required by overlap_vicreg."
            )
        if self.vicreg.neighbor_view and self.vicreg.neighbor_view_mode not in {
            "none",
            "second",
        }:
            raise ValueError(
                "Temporal adjacent views support vicreg_neighbor_view_mode 'none' or "
                f"'second', got {self.vicreg.neighbor_view_mode!r}."
            )

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

    def _temporal_meta_from_batch(self, batch: dict) -> dict:
        return {
            "instance_id": batch["instance_id"],
            "center_atom_id": batch["center_atom_id"],
            "anchor_frame_index": batch["anchor_frame_index"],
            "anchor_timestep": batch["anchor_timestep"],
            "frame_indices": batch["frame_indices"],
            "timesteps": batch["timesteps"],
            "source_path": batch["source_path"],
        }

    def _unpack_batch(self, batch):
        sequence_points = batch["points"]
        return self._center_frame(sequence_points), self._temporal_meta_from_batch(batch)

    def _unpack_temporal_batch(self, batch):
        center_pc, meta = self._unpack_batch(batch)
        sequence_points = batch["points"]
        prev_pc = sequence_points[:, self.center_frame_index - 1]
        next_pc = sequence_points[:, self.center_frame_index + 1]
        return center_pc, prev_pc, next_pc, meta

    def _prepare_explicit_view(self, pc: torch.Tensor, *, target_points: int) -> torch.Tensor:
        return crop_to_num_points(pc, target_points)

    def _prepare_swav_view(self, pc: torch.Tensor) -> torch.Tensor:
        return self._prepare_explicit_view(pc, target_points=self.swav.view_points)

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
        num_views = len(views)
        concatenated = torch.cat(views, dim=0)
        encoded = self.encoder_io.encode(concatenated)
        features = self._shared_invariant(encoded.invariant, encoded.equivariant)
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
        """Return whether both scheduled heads consume the configured shared views."""
        return (
            should_run_vicreg
            and should_run_swav
            and self.use_temporal_vicreg_views
        )

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

        # Anchor='center': reuse the VICReg temporal view builder.
        vicreg_views = self._build_vicreg_temporal_views(center_pc, prev_pc, next_pc)
        return {"anchor": vicreg_views["y_a"], "pair": vicreg_views["y_b"]}

    def _encode_fused_contrastive_views(
        self,
        views: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single encoder forward over ``[anchor ; pair]`` -> (anchor_feat, pair_feat)."""
        fused_input = torch.cat([views["anchor"], views["pair"]], dim=0)
        encoded = self.encoder_io.encode(fused_input)
        features = self._shared_invariant(encoded.invariant, encoded.equivariant)
        return features.chunk(2, dim=0)

    def _build_vicreg_temporal_views(
        self,
        center_pc: torch.Tensor,
        prev_pc: torch.Tensor,
        next_pc: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_points = self.vicreg.view_points
        neighbor_mode = self.vicreg.neighbor_view_mode
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
        if pc.dim() == 4:
            pc = self._center_frame(pc)
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
        sequence_points = batch["points"]
        batch_size = int(sequence_points.shape[0])
        meta = self._temporal_meta_from_batch(batch)
        sequence_points = sequence_points.to(device=self.device, dtype=self.dtype, non_blocking=True)
        pc_raw = self._center_frame(sequence_points)
        prev_pc = sequence_points[:, self.center_frame_index - 1]
        next_pc = sequence_points[:, self.center_frame_index + 1]
        vicreg_pc_raw = self._vicreg_anchor_frame(sequence_points)
        pc = self._prepare_model_input(pc_raw)

        should_cache_stage = self._should_cache_supervised_stage(stage)
        should_run_vicreg = self.vicreg.should_run(current_epoch=int(self.current_epoch))
        should_run_swav = self.swav.should_run(current_epoch=int(self.current_epoch))

        losses = {}
        if should_cache_stage:
            with torch.no_grad():
                encoded = self.encoder_io.encode(pc)
                center_embeddings = self._contrastive_invariant_from_eq_latent(
                    encoded.equivariant,
                    z_inv_model=encoded.invariant,
                    stage=stage,
                )
            self._cache_supervised_embeddings_if_needed(
                stage=stage,
                meta=meta,
                embeddings=center_embeddings,
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
                losses[self.vicreg.metric_prefix] = vicreg_loss
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

        return self._finish_ssl_step(
            stage=stage,
            batch_idx=batch_idx,
            batch_size=batch_size,
            losses=losses,
            print_first_eval_batch=True,
        )


__all__ = ["TemporalSSLModule"]
