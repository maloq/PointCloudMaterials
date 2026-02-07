import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import sys
from typing import Dict, Optional
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.getcwd())
from src.models.autoencoders.factory import build_model
from src.utils.spd_utils import get_optimizers_and_scheduler


class ProjectionMLP(nn.Module):
    """Projector MLP matching the contrastive (Barlow/VICReg) head."""

    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(embed_dim), bias=False),
            nn.BatchNorm1d(int(embed_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(embed_dim), int(embed_dim), bias=False),
            nn.BatchNorm1d(int(embed_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(embed_dim), int(embed_dim), bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SupervisedEncoder(pl.LightningModule):
    """
    Supervised pretraining module for encoder.

    Trains encoder to:
    1. Predict class labels from the encoder invariant code using a contrastive-style MLP projector
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        # Build encoder (decoder not needed for pretraining)
        self.encoder, _ = build_model(cfg)

        # Get encoder latent size
        encoder_kwargs = self.hparams.encoder.get('kwargs', {})
        self.encoder_latent_size = encoder_kwargs.get('latent_size', self.hparams.latent_size)

        # Supervised classifier head (projector + linear classifier)
        self.projector_dim = self._resolve_projector_dim(cfg, self.encoder_latent_size)
        self.projector = ProjectionMLP(self.encoder_latent_size, self.projector_dim)

        self.num_classes = int(getattr(cfg, "num_classes", 0) or getattr(cfg, "num_phases", 0) or 0)
        if self.num_classes <= 0:
            self.num_classes = 1
        self.classifier = nn.Linear(self.projector_dim, self.num_classes)
        self.class_id_to_name: Optional[Dict[int, str]] = None

        # Loss weights
        self.class_loss_weight = float(getattr(cfg, "class_loss_weight", getattr(cfg, "phase_loss_weight", 1.0)))

        # Metrics cache
        self._supervised_cache = {
            "train": {"preds": [], "labels": []},
            "val": {"preds": [], "labels": []},
        }

    @staticmethod
    def _cfg_get(cfg, key: str, default=None):
        return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)

    @staticmethod
    def _resolve_projector_dim(cfg, fallback: int) -> int:
        explicit = SupervisedEncoder._cfg_get(cfg, "supervised_projector_dim", None)
        if explicit is not None:
            return int(explicit)
        if bool(SupervisedEncoder._cfg_get(cfg, "vicreg_enabled", False)):
            vicreg_dim = SupervisedEncoder._cfg_get(cfg, "vicreg_embed_dim", None)
            if vicreg_dim is not None:
                return int(vicreg_dim)
        if bool(SupervisedEncoder._cfg_get(cfg, "barlow_enabled", False)):
            barlow_dim = SupervisedEncoder._cfg_get(cfg, "barlow_embed_dim", None)
            if barlow_dim is not None:
                return int(barlow_dim)
        vicreg_dim = SupervisedEncoder._cfg_get(cfg, "vicreg_embed_dim", None)
        if vicreg_dim is not None:
            return int(vicreg_dim)
        barlow_dim = SupervisedEncoder._cfg_get(cfg, "barlow_embed_dim", None)
        if barlow_dim is not None:
            return int(barlow_dim)
        return int(fallback)

    @staticmethod
    def _unwrap_dataset(dataset):
        visited = set()
        ds = dataset
        while ds is not None and id(ds) not in visited:
            visited.add(id(ds))
            if hasattr(ds, "num_classes") or hasattr(ds, "class_names"):
                return ds
            next_ds = getattr(ds, "dataset", None)
            if next_ds is not None and next_ds is not ds:
                ds = next_ds
                continue
            base_ds = getattr(ds, "base_dataset", None)
            if base_ds is not None and base_ds is not ds:
                ds = base_ds
                continue
            break
        return ds

    def _resolve_class_info(self):
        dm = getattr(self.trainer, "datamodule", None)
        if dm is None:
            return None, None
        ds = getattr(dm, "train_dataset", None) or getattr(dm, "val_dataset", None)
        ds = self._unwrap_dataset(ds)
        if ds is None:
            return None, None
        class_id_to_name = None
        if hasattr(ds, "class_names"):
            try:
                class_id_to_name = dict(ds.class_names)
            except Exception:
                class_id_to_name = None
        num_classes = None
        if hasattr(ds, "num_classes"):
            try:
                num_classes = int(ds.num_classes)
            except Exception:
                num_classes = None
        if num_classes is None and class_id_to_name is not None:
            num_classes = len(class_id_to_name)
        return class_id_to_name, num_classes

    def _rebuild_classifier(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        device = next(self.projector.parameters()).device
        dtype = next(self.projector.parameters()).dtype
        self.classifier = nn.Linear(self.projector_dim, self.num_classes).to(device=device, dtype=dtype)

    def setup(self, stage=None):
        """Resolve class mapping and update classifier if needed."""
        class_id_to_name, num_classes = self._resolve_class_info()
        if num_classes is not None and num_classes > 0 and num_classes != self.num_classes:
            print(f"Updating classifier: {self.num_classes} -> {num_classes} classes")
            self._rebuild_classifier(num_classes)
        if class_id_to_name is not None:
            self.class_id_to_name = class_id_to_name

    def _prepare_encoder_input(self, pc: torch.Tensor) -> torch.Tensor:
        if getattr(self.encoder, "expects_channel_first", False):
            return pc.permute(0, 2, 1).contiguous()
        return pc

    @staticmethod
    def _split_encoder_output(enc_out):
        if isinstance(enc_out, (tuple, list)):
            if not enc_out:
                raise ValueError("Encoder returned empty output")
            inv_z = enc_out[0]
            eq_z = None
            if len(enc_out) > 1:
                candidate = enc_out[1]
                if torch.is_tensor(candidate) and candidate.dim() == 3 and candidate.shape[-1] == 3:
                    if inv_z is not None and inv_z.dim() == 2 and candidate.shape[1] == inv_z.shape[1]:
                        eq_z = candidate
                    elif candidate.shape[1] != 3:
                        eq_z = candidate
            return inv_z, eq_z
        return enc_out, None

    @staticmethod
    def _select_inv_z(inv_z, eq_z):
        if inv_z is not None:
            if torch.is_tensor(inv_z) and inv_z.dim() == 3 and inv_z.shape[-1] == 3:
                return inv_z.norm(dim=-1)
            return inv_z
        if eq_z is None:
            return None
        if torch.is_tensor(eq_z) and eq_z.dim() == 3 and eq_z.shape[-1] == 3:
            return eq_z.norm(dim=-1)
        if torch.is_tensor(eq_z) and eq_z.dim() > 2:
            return eq_z.reshape(eq_z.shape[0], -1)
        return eq_z

    def _encode(self, pc: torch.Tensor):
        enc_out = self.encoder(self._prepare_encoder_input(pc))
        inv_z, eq_z = self._split_encoder_output(enc_out)
        inv_z = self._select_inv_z(inv_z, eq_z)
        return inv_z, eq_z

    def _ensure_projector_input_dim(self, inv_z: torch.Tensor) -> None:
        if inv_z is None or inv_z.dim() != 2:
            return
        current_in = self.projector.net[0].in_features
        target_in = int(inv_z.shape[1])
        if current_in == target_in:
            return
        print(f"Updating projector input dim: {current_in} -> {target_in}")
        device = next(self.projector.parameters()).device
        dtype = next(self.projector.parameters()).dtype
        self.projector = ProjectionMLP(target_in, self.projector_dim).to(device=device, dtype=dtype)

    def forward(self, pc: torch.Tensor):
        """
        Args:
            pc: (B, N, 3) input point cloud
        Returns:
            inv_z: (B, latent_size) invariant latent code
            class_logits: (B, num_classes) class prediction logits
        """
        inv_z, eq_z = self._encode(pc)
        if inv_z is None:
            raise ValueError("Encoder did not provide invariant features (inv_z)")
        self._ensure_projector_input_dim(inv_z)
        proj_dtype = next(self.projector.parameters()).dtype
        proj = self.projector(inv_z.to(dtype=proj_dtype))
        class_logits = self.classifier(proj)
        return inv_z, class_logits

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, dict):
            pc = batch["points"]
            meta = {
                "class_id": batch.get("class_id"),
            }
            return pc, meta
        if not isinstance(batch, (tuple, list)):
            return batch, {}
        meta = {}
        if len(batch) > 1:
            meta["class_id"] = batch[1]
        return batch[0], meta

    def _step(self, batch, batch_idx, stage: str):
        pc, meta = self._unpack_batch(batch)
        pc = pc.to(device=self.device, dtype=self.dtype, non_blocking=True)

        class_id = meta.get("class_id")
        if class_id is None:
            raise ValueError("Supervised encoder requires class_id labels in the batch")
        if not torch.is_tensor(class_id):
            class_id = torch.as_tensor(class_id)
        class_id = class_id.to(device=self.device, dtype=torch.long, non_blocking=True).view(-1)

        # Forward pass
        inv_z, class_logits = self(pc)

        # Classification loss
        class_loss = nn.functional.cross_entropy(class_logits, class_id)

        # Total loss
        total_loss = self.class_loss_weight * class_loss

        # Compute accuracy
        with torch.no_grad():
            class_preds = torch.argmax(class_logits, dim=1)
            accuracy = (class_preds == class_id).float().mean()

        # Cache predictions for epoch-end metrics
        if stage in self._supervised_cache:
            self._supervised_cache[stage]["preds"].append(class_preds.detach().cpu())
            self._supervised_cache[stage]["labels"].append(class_id.detach().cpu())

        # Log metrics
        self.log(f"{stage}/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/class_loss", class_loss, on_step=True, on_epoch=True)
        self.log(f"{stage}/phase_loss", class_loss, on_step=True, on_epoch=True)
        self.log(f"{stage}/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def on_train_epoch_start(self):
        self._reset_cache("train")

    def on_validation_epoch_start(self):
        self._reset_cache("val")

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def _reset_cache(self, stage: str):
        if stage in self._supervised_cache:
            self._supervised_cache[stage]["preds"].clear()
            self._supervised_cache[stage]["labels"].clear()

    def _log_epoch_metrics(self, stage: str):
        cache = self._supervised_cache.get(stage)
        if cache is None or not cache["preds"]:
            return

        # Concatenate all predictions and labels
        all_preds = torch.cat(cache["preds"]).numpy()
        all_labels = torch.cat(cache["labels"]).numpy()

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        # Log metrics
        self.log(f"{stage}/f1_macro", f1_macro, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/f1_weighted", f1_weighted, on_epoch=True, sync_dist=True)

        # Clear cache
        self._reset_cache(stage)

    def configure_optimizers(self):
        return get_optimizers_and_scheduler(self.hparams, self.parameters())


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    def _parse_data_yaml(yaml_path: Path):
        """Very light parser to extract data_path and radius from a YAML file.

        Avoids adding a new dependency. Falls back to None if keys not found.
        """
        data_path = None
        radius = None
        try:
            text = yaml_path.read_text()
        except Exception:
            return data_path, radius
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data_path:") and data_path is None:
                data_path = line.split(":", 1)[1].strip()
            if line.startswith("radius:") and radius is None:
                try:
                    radius = float(line.split(":", 1)[1].strip())
                except Exception:
                    radius = None
        return data_path, radius

    def _compute_stats(clouds: list[np.ndarray]):
        means = []
        max_norms = []
        for arr in clouds:
            if arr.size == 0:
                continue
            c = arr.mean(axis=0)
            means.append(float(np.linalg.norm(c)))
            max_norms.append(float(np.linalg.norm(arr - c, axis=1).max()))
        return {
            "count": len(max_norms),
            "centroid_l2_mean": float(np.mean(means)) if means else 0.0,
            "max_norm_mean": float(np.mean(max_norms)) if max_norms else 0.0,
            "max_norm_std": float(np.std(max_norms)) if max_norms else 0.0,
        }

    parser = argparse.ArgumentParser(description="Verify reference PC normalization vs. dataset samples")
    parser.add_argument("--data_path", type=str, default=None, help="Dataset directory with atoms.npy + metadata.json")
    parser.add_argument("--radius", type=float, default=None, help="Sampling radius used to normalize samples")
    parser.add_argument("--data_yaml", type=str, default="configs/data/data_synth_no_perturb.yaml", help="Data YAML to infer defaults")
    parser.add_argument("--write_fixed", type=str, default=None, help="Optional output path to write radius-normalized references")
    args = parser.parse_args()

    data_path = args.data_path
    radius = args.radius

    # Backfill from YAML if needed
    if data_path is None or radius is None:
        yaml_path = Path(args.data_yaml)
        if yaml_path.exists():
            y_data_path, y_radius = _parse_data_yaml(yaml_path)
            data_path = data_path or y_data_path
            radius = radius or y_radius

    if not data_path:
        print("[verify] data_path not provided and could not be inferred. Use --data_path.")
        raise SystemExit(2)
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"[verify] data_path {data_dir} does not exist")
        raise SystemExit(2)

    # Locate reference file and metadata
    meta_path = data_dir / "metadata.json"
    ref_path = data_dir / "reference_point_clouds.npy"
    meta = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = None
    if meta and "reference_point_clouds" in meta:
        ref_section = meta["reference_point_clouds"]
        if isinstance(ref_section, dict):
            if not ref_path.exists() and ref_section.get("point_clouds_file"):
                ref_path = data_dir / ref_section["point_clouds_file"]

    if not ref_path.exists():
        print(f"[verify] reference point clouds not found at {ref_path}")
        raise SystemExit(2)

    ref_dict = np.load(str(ref_path), allow_pickle=True).item()
    ref_clouds = [np.asarray(v, dtype=np.float32) for v in ref_dict.values()]

    # Compute reference stats
    ref_stats = _compute_stats(ref_clouds)
    ref_norm_flag = None
    if meta and isinstance(meta.get("reference_point_clouds"), dict):
        ref_norm_flag = meta["reference_point_clouds"].get("point_cloud_normalized")

    print("[verify] Reference clouds:")
    print(f" - count: {ref_stats['count']}")
    print(f" - centroid L2 mean: {ref_stats['centroid_l2_mean']:.4f}")
    print(f" - max_norm mean ± std: {ref_stats['max_norm_mean']:.4f} ± {ref_stats['max_norm_std']:.4f}")
    if ref_norm_flag is not None:
        print(f" - metadata.point_cloud_normalized: {ref_norm_flag}")

    # Compare against expected sample scaling
    if radius is None:
        print("[verify] radius unknown; cannot compare to sample scaling. Use --radius or a data YAML.")
        need_fix = False
    else:
        # Samples scaled by radius -> expected unit ball scale
        expected = 1.0
        diff = abs(ref_stats["max_norm_mean"] - expected)
        need_fix = diff > 0.2  # tolerant threshold
        status = "OK" if not need_fix else "MISMATCH"
        print(f"[verify] Expected sample scale (after divide by radius {radius}): ~{expected}. Status: {status}")

    # Optionally write a radius-normalized copy for references
    if need_fix and args.write_fixed:
        if radius is None:
            print("[verify] --radius required to write fixed references")
        else:
            out_path = Path(args.write_fixed)
            fixed = {}
            for k, pts in ref_dict.items():
                arr = np.asarray(pts, dtype=np.float32)
                c = arr.mean(axis=0, keepdims=True)
                fixed[k] = ((arr - c) / float(radius)).astype(np.float32)
            np.save(str(out_path), fixed, allow_pickle=True)
            print(f"[verify] Wrote radius-normalized references to {out_path}")
    elif need_fix:
        print("[verify] Consider normalizing references to match sample scaling (center + divide by radius).\n"
              "         To write a fixed copy, rerun with --radius R --write_fixed <out.npy>.")
    else:
        print("[verify] Reference normalization appears consistent with sample scaling.")
