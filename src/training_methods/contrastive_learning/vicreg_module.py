import torch

from src.data_utils.data_kinds import normalize_data_kind
from src.training_methods.base_ssl_module import BaseSSLModule


class VICRegModule(BaseSSLModule):
    """
    Self-supervised contrastive training with VICReg/VISReg/SwAV heads.
    """

    def __init__(self, cfg):
        self.data_kind = normalize_data_kind(cfg.data.kind)
        super().__init__(
            cfg,
            module_name="VICRegModule",
        )
        self.cache_warning_prefix = "contrastive"

    def _unpack_batch(self, batch):
        if self.data_kind == "static":
            return batch["points"], {}
        if self.data_kind == "synthetic":
            return batch["points"], {
                "class_id": batch["class_id"],
                "instance_id": batch["instance_id"],
                "rotation": batch["rotation"],
            }
        raise RuntimeError(
            "VICRegModule only consumes the repository's static or synthetic batches, "
            f"got data.kind={self.data_kind!r}."
        )

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
        batch_size = int(y_a.shape[0])

        fused_input = torch.cat([y_a, y_b], dim=0)
        encoded = self.encoder_io.encode(fused_input)
        features = self._shared_invariant(encoded.invariant, encoded.equivariant)

        return features.chunk(2, dim=0)

    def forward(self, pc: torch.Tensor, include_ssl_heads: bool = False):
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
        # Forward returns both invariant branches explicitly:
        # (z_inv_contrastive, z_inv_model, eq_z).
        return z_inv_contrastive, encoded.invariant, encoded.equivariant

    def _step(self, batch, batch_idx, stage: str):
        pc_raw, meta = self._unpack_batch(batch)
        batch_size = int(pc_raw.shape[0])
        pc_raw = pc_raw.to(device=self.device, dtype=self.dtype, non_blocking=True)
        pc = self._prepare_model_input(pc_raw)
        losses = {}

        run_vicreg = self.vicreg.should_run(current_epoch=int(self.current_epoch))
        run_swav = self.swav.should_run(current_epoch=int(self.current_epoch))
        can_share_views = run_vicreg and run_swav and self.vicreg.view_points == self.swav.view_points

        if can_share_views:
            shared_view_pair = self._build_contrastive_view_pair(pc_raw, view_points=self.vicreg.view_points)
            shared_features = self._encode_contrastive_view_pair(shared_view_pair)

        if run_vicreg:
            if can_share_views:
                vicreg_views = shared_view_pair
                z_a, z_b = shared_features
            else:
                vicreg_views = self._build_contrastive_view_pair(
                    pc_raw,
                    view_points=self.vicreg.view_points,
                )
                z_a, z_b = self._encode_contrastive_view_pair(vicreg_views)
            overlap_target = (
                vicreg_views["overlap_target"]
                if self.vicreg.requires_overlap_target
                else None
            )
            vicreg_loss, vicreg_metrics = self.vicreg.compute_loss_from_features(
                z_a_feat=z_a,
                z_b_feat=z_b,
                current_epoch=int(self.current_epoch),
                overlap_target=overlap_target,
            )
            if vicreg_loss is not None:
                losses[self.vicreg.metric_prefix] = vicreg_loss
            for name, value in vicreg_metrics.items():
                self._log_metric(stage, name, value, batch_size=batch_size)

        if run_swav:
            if can_share_views:
                z_a, z_b = shared_features
            else:
                swav_views = self._build_contrastive_view_pair(
                    pc_raw,
                    view_points=self.swav.view_points,
                )
                z_a, z_b = self._encode_contrastive_view_pair(swav_views)
            swav_loss, swav_metrics = self.swav.compute_loss(
                view_features=[z_a, z_b],
                current_epoch=int(self.current_epoch),
            )
            if swav_loss is not None:
                losses["swav"] = swav_loss
            for name, value in swav_metrics.items():
                self._log_metric(stage, name, value, batch_size=batch_size)

        if self._should_cache_supervised_stage(stage):
            with torch.no_grad():
                encoded = self.encoder_io.encode(pc)
                z_inv = self._contrastive_invariant_from_eq_latent(encoded.equivariant, z_inv_model=encoded.invariant, stage=stage)
            self._cache_supervised_embeddings_if_needed(stage=stage, meta=meta, embeddings=z_inv)

        return self._finish_ssl_step(stage=stage, batch_idx=batch_idx, batch_size=batch_size, losses=losses)

__all__ = ["VICRegModule"]
