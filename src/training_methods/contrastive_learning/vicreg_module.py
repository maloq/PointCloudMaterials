import torch

from src.training_methods.base_ssl_module import BaseSSLModule


class VICRegModule(BaseSSLModule):
    """
    Self-supervised contrastive training with VICReg/VISReg/SwAV heads.
    """

    def __init__(self, cfg):
        super().__init__(
            cfg,
            module_name="VICRegModule",
            require_summary_points=getattr(cfg, "data", None) is not None,
        )
        self.cache_warning_prefix = "contrastive"

    @staticmethod
    def _unpack_batch(batch):
        return batch["points"], {
            "class_id": batch.get("class_id"),
            "instance_id": batch.get("instance_id"),
            "rotation": batch.get("rotation"),
        }

    def _build_contrastive_view_pair(
        self,
        pc: torch.Tensor,
        *,
        view_points: int | None,
    ) -> dict[str, torch.Tensor]:
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
        shared_views = run_vicreg and run_swav and self.vicreg.view_points == self.swav.view_points

        def process_head(head_name, compute_fn, view_points, shared_features=None):
            if shared_features:
                z_a, z_b = shared_features
            else:
                views = self._build_contrastive_view_pair(pc_raw, view_points=view_points)
                z_a, z_b = self._encode_contrastive_view_pair(views)
            loss, metrics = compute_fn(z_a, z_b)
            if loss is not None: losses[head_name] = loss
            for k, v in metrics.items(): self._log_metric(stage, k, v, batch_size=batch_size)
            return z_a, z_b

        shared_features = None
        if shared_views:
            views = self._build_contrastive_view_pair(pc_raw, view_points=self.vicreg.view_points)
            shared_features = self._encode_contrastive_view_pair(views)

        if run_vicreg:
            process_head(
                self.vicreg.metric_prefix,
                lambda a, b: self.vicreg.compute_loss_from_features(z_a_feat=a, z_b_feat=b, current_epoch=int(self.current_epoch)),
                self.vicreg.view_points,
                shared_features
            )

        if run_swav:
            process_head(
                "swav",
                lambda a, b: self.swav.compute_loss(view_features=[a, b], current_epoch=int(self.current_epoch)),
                self.swav.view_points,
                shared_features
            )

        if self._should_cache_supervised_stage(stage):
            with torch.no_grad():
                encoded = self.encoder_io.encode(pc)
                z_inv = self._contrastive_invariant_from_eq_latent(encoded.equivariant, z_inv_model=encoded.invariant, stage=stage)
            self._cache_supervised_embeddings_if_needed(stage=stage, meta=meta, embeddings=z_inv)

        return self._finish_ssl_step(stage=stage, batch_idx=batch_idx, batch_size=batch_size, losses=losses)

__all__ = ["VICRegModule"]
