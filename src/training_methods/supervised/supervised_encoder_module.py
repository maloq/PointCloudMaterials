import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.getcwd())
from src.models.autoencoders.factory import build_model
from src.utils.spd_utils import get_optimizers_and_scheduler
from src.training_methods.spd.rot_heads import build_rot_head
from src.loss.reconstruction_loss import chamfer_distance as _cd_fn


class PhasePredictionHead(nn.Module):
    """MLP head for phase prediction from invariant latent code."""

    def __init__(self, latent_size: int, num_phases: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_phases),
        )

    def forward(self, z_inv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_inv: (B, latent_size) invariant features
        Returns:
            logits: (B, num_phases)
        """
        return self.mlp(z_inv)


class SupervisedEncoder(pl.LightningModule):
    """
    Supervised pretraining module for encoder.

    Trains encoder to:
    1. Predict phase labels using Z_inv and a small MLP
    2. Predict rotation to align input point cloud to reference point cloud using rotation network
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        # Build encoder (decoder not needed for pretraining)
        self.encoder, _ = build_model(cfg)

        # Get encoder latent size
        encoder_kwargs = self.hparams.encoder.get('kwargs', {})
        self.encoder_latent_size = encoder_kwargs.get('latent_size', self.hparams.latent_size)

        # Phase prediction head (will be updated with actual number of phases in setup)
        self.num_phases = cfg.get("num_phases", 5)  # Default, will be updated
        self.phase_head = PhasePredictionHead(
            latent_size=self.encoder_latent_size,
            num_phases=self.num_phases,
            hidden_dim=cfg.get("phase_head_hidden", 128)
        )

        # Rotation network to predict rotation to reference
        rotation_mode = getattr(cfg, "rotation_mode", None)
        if rotation_mode is None or rotation_mode == "":
            raise ValueError("rotation_mode is required")

        self.rotation_mode = str(rotation_mode).lower()
        self._use_rot_head = self.rotation_mode in {"sixd_head", "matrix_head"}
        self.rot_net = build_rot_head(cfg, in_features=self.encoder_latent_size * 3) if self._use_rot_head else None

        # Loss weights
        self.phase_loss_weight = cfg.get("phase_loss_weight", 1.0)
        self.rotation_loss_weight = cfg.get("rotation_loss_weight", 1.0)
        # Optional geodesic supervision to orientation labels
        self.geodesic_loss_weight = cfg.get("geodesic_loss_weight", 0.0)
        self.geodesic_use_transpose = cfg.get("geodesic_use_transpose", True)
        self.geodesic_softmin_tau = cfg.get("geodesic_softmin_tau", None)
        # Symmetry soft-min temperature for rotation alignment (0 -> hard min)
        self.symmetry_softmin_tau = cfg.get("symmetry_softmin_tau", 0.02)

        # Augmentation settings
        self.use_rotation_augmentation = cfg.get("use_rotation_augmentation", True)

        # Reference point clouds (will be loaded in setup)
        self.reference_pcs = None
        self.phase_to_idx = {}

        # Metrics cache
        self._supervised_cache = {
            "train": {"preds": [], "labels": []},
            "val": {"preds": [], "labels": []},
        }

    @staticmethod
    def _group_phase(phase_id: str) -> str:
        """Group amorphous phases (but not intermediate) into one class."""
        if phase_id.startswith('amorphous_') and not phase_id.startswith('intermediate_'):
            return 'amorphous'
        return phase_id

    def setup(self, stage=None):
        """Load reference point clouds and determine number of phases from metadata."""
        if self.reference_pcs is None:
            # Determine data directory
            if hasattr(self.hparams, 'data') and hasattr(self.hparams.data, 'data_path'):
                data_dir = self.hparams.data.data_path
            elif hasattr(self.hparams, 'synthetic') and hasattr(self.hparams.synthetic, 'data_dir'):
                data_dir = self.hparams.synthetic.data_dir
            else:
                raise ValueError("Cannot determine data path")

            # Load reference point clouds
            ref_path = os.path.join(data_dir, 'reference_point_clouds.npy')
            if os.path.exists(ref_path):
                self.reference_pcs = np.load(ref_path, allow_pickle=True).item()
                print(f"Loaded reference point clouds from {ref_path}")
                print(f"Reference phases: {list(self.reference_pcs.keys())}")
            else:
                print(f"Warning: Reference point clouds not found at {ref_path}")
                self.reference_pcs = {}

            # Load all phases from metadata.json (includes main + intermediate phases)
            metadata_path = os.path.join(data_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Collect all unique phases
                all_phases = set()
                for grain in metadata.get('grains', []):
                    all_phases.add(grain['base_phase_id'])
                for region in metadata.get('intermediate_regions', []):
                    all_phases.add(region.get('intermediate_phase_id', 'intermediate'))

                # Apply the same grouping logic as the dataset
                grouped_phases = set()
                for phase in all_phases:
                    grouped_phase = self._group_phase(phase)
                    grouped_phases.add(grouped_phase)

                # Create phase to index mapping (sorted for consistency)
                phase_names = sorted(grouped_phases)
                self.phase_to_idx = {phase: idx for idx, phase in enumerate(phase_names)}
                actual_num_phases = len(phase_names)

                print(f"Loaded {len(all_phases)} original phases, grouped into {actual_num_phases} classes")
                print(f"Grouped phases: {phase_names}")
                print(f"Phase to index mapping: {self.phase_to_idx}")

                # Update phase head if number of phases changed
                if actual_num_phases != self.num_phases:
                    print(f"Updating phase head: {self.num_phases} -> {actual_num_phases} phases")
                    self.num_phases = actual_num_phases
                    self.phase_head = PhasePredictionHead(
                        latent_size=self.encoder_latent_size,
                        num_phases=self.num_phases,
                        hidden_dim=self.hparams.get("phase_head_hidden", 128)
                    )
            else:
                raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    def forward(self, pc: torch.Tensor):
        """
        Args:
            pc: (B, N, 3) input point cloud
        Returns:
            inv_z: (B, latent_size) invariant latent code
            eq_z: (B, latent_size, 3) equivariant latent code
            phase_logits: (B, num_phases) phase prediction logits
            rot: (B, 3, 3) predicted rotation matrix
        """
        inv_z, eq_z, _ = self.encoder(pc)
        phase_logits = self.phase_head(inv_z)
        rot = self.rot_net(eq_z) if self.rot_net is not None else None
        return inv_z, eq_z, phase_logits, rot

    def _get_reference_pc(self, phase_names):
        """Get reference point clouds for given phase names (assumes all phase names have references)."""
        if not self.reference_pcs:
            return None

        batch_refs = []
        for phase_name in phase_names:
            # Handle grouped "amorphous" phase - use amorphous_mixed as reference
            if phase_name == 'amorphous':
                # Try to use any available amorphous reference
                ref_phase = 'amorphous_mixed' if 'amorphous_mixed' in self.reference_pcs else \
                           'amorphous_random' if 'amorphous_random' in self.reference_pcs else \
                           'amorphous_repeat' if 'amorphous_repeat' in self.reference_pcs else None
                if ref_phase:
                    batch_refs.append(self.reference_pcs[ref_phase])
                else:
                    raise ValueError(f"No amorphous reference point cloud found")
            else:
                batch_refs.append(self.reference_pcs[phase_name])

        return torch.tensor(np.stack(batch_refs), dtype=torch.float32)

    @staticmethod
    def _random_rotation_matrix(batch_size: int, device, dtype) -> torch.Tensor:
        """Generate random rotation matrices using uniform sampling on SO(3).

        Uses the method from "Uniform Random Rotations" by Ken Shoemake.
        """
        # Generate random quaternions
        u1 = torch.rand(batch_size, device=device, dtype=dtype)
        u2 = torch.rand(batch_size, device=device, dtype=dtype)
        u3 = torch.rand(batch_size, device=device, dtype=dtype)

        # Convert to quaternion using Shoemake's method
        sqrt1_u1 = torch.sqrt(1.0 - u1)
        sqrt_u1 = torch.sqrt(u1)

        w = sqrt1_u1 * torch.sin(2 * np.pi * u2)
        x = sqrt1_u1 * torch.cos(2 * np.pi * u2)
        y = sqrt_u1 * torch.sin(2 * np.pi * u3)
        z = sqrt_u1 * torch.cos(2 * np.pi * u3)

        # Convert quaternion to rotation matrix
        # R = [[1-2(y^2+z^2), 2(xy-wz), 2(xz+wy)],
        #      [2(xy+wz), 1-2(x^2+z^2), 2(yz-wx)],
        #      [2(xz-wy), 2(yz+wx), 1-2(x^2+y^2)]]

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        R = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        R[:, 0, 0] = 1 - 2 * (yy + zz)
        R[:, 0, 1] = 2 * (xy - wz)
        R[:, 0, 2] = 2 * (xz + wy)
        R[:, 1, 0] = 2 * (xy + wz)
        R[:, 1, 1] = 1 - 2 * (xx + zz)
        R[:, 1, 2] = 2 * (yz - wx)
        R[:, 2, 0] = 2 * (xz - wy)
        R[:, 2, 1] = 2 * (yz + wx)
        R[:, 2, 2] = 1 - 2 * (xx + yy)

        return R

    @staticmethod
    def _apply_rotation(points: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
        """Apply rotation matrices to batched point clouds.

        Args:
            points: (B, N, 3) point clouds
            rot: (B, 3, 3) rotation matrices
        Returns:
            rotated_points: (B, N, 3)
        """
        return (rot @ points.transpose(1, 2)).transpose(1, 2).contiguous()

    def _rotation_chamfer_loss(self, pred_rot: torch.Tensor, pc: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Compute chamfer distance between rotated input and reference.

        Args:
            pred_rot: (B, 3, 3) predicted rotation matrix
            pc: (B, N, 3) input point cloud
            reference: (B, M, 3) reference point cloud
        """
        # Apply rotation to input
        rotated_pc = self._apply_rotation(pc, pred_rot)  # (B, N, 3)

        # Compute chamfer distance
        # For each point in rotated_pc, find nearest in reference
        dist1 = torch.cdist(rotated_pc, reference)  # (B, N, M)
        min_dist1 = dist1.min(dim=2)[0]  # (B, N)

        # For each point in reference, find nearest in rotated_pc
        min_dist2 = dist1.min(dim=1)[0]  # (B, M)

        # Chamfer distance
        chamfer = min_dist1.mean(dim=1) + min_dist2.mean(dim=1)  # (B,)
        return chamfer.mean()

    def _chamfer_distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Wrapper around project-wide Chamfer distance implementation.

        Returns a scalar tensor (mean over batch).
        """
        dist, _ = _cd_fn(pred, target)
        return dist

    def _euler_to_matrix(self, x_deg: float, y_deg: float, z_deg: float, device, dtype) -> torch.Tensor:
        """Create rotation matrix from XYZ Euler angles in degrees.

        Uses R = Rz @ Ry @ Rx convention.
        """
        rad = np.pi / 180.0
        x = x_deg * rad
        y = y_deg * rad
        z = z_deg * rad

        cx, sx = np.cos(x), np.sin(x)
        cy, sy = np.cos(y), np.sin(y)
        cz, sz = np.cos(z), np.sin(z)

        Rx = torch.tensor([[1.0, 0.0, 0.0],
                           [0.0, cx, -sx],
                           [0.0, sx, cx]], device=device, dtype=dtype)
        Ry = torch.tensor([[cy, 0.0, sy],
                           [0.0, 1.0, 0.0],
                           [-sy, 0.0, cy]], device=device, dtype=dtype)
        Rz = torch.tensor([[cz, -sz, 0.0],
                           [sz,  cz, 0.0],
                           [0.0, 0.0, 1.0]], device=device, dtype=dtype)

        return Rz @ Ry @ Rx

    def _get_symmetry_rotations(self, device, dtype):
        """Generate rotations approximating the cubic symmetry group without Python loops.

        Returns a tensor of shape (64, 3, 3) with all combinations of
        0/90/180/270° rotations around X, Y, and Z using R = Rz @ Ry @ Rx.
        """
        angles_deg = torch.tensor([0.0, 90.0, 180.0, 270.0], device=device, dtype=torch.float32)
        angles = angles_deg * (np.pi / 180.0)

        x, y, z = torch.meshgrid(angles, angles, angles, indexing='ij')  # each (4,4,4)
        x = x.reshape(-1)  # (64,)
        y = y.reshape(-1)
        z = z.reshape(-1)

        cx, sx = torch.cos(x), torch.sin(x)
        cy, sy = torch.cos(y), torch.sin(y)
        cz, sz = torch.cos(z), torch.sin(z)

        ones = torch.ones_like(cx)
        zeros = torch.zeros_like(cx)

        # Build Rx, Ry, Rz in a vectorized fashion: each is (64, 3, 3)
        Rx_row0 = torch.stack([ones,  zeros,  zeros], dim=-1)
        Rx_row1 = torch.stack([zeros,   cx,    -sx], dim=-1)
        Rx_row2 = torch.stack([zeros,   sx,     cx], dim=-1)
        Rx = torch.stack([Rx_row0, Rx_row1, Rx_row2], dim=-2)

        Ry_row0 = torch.stack([  cy, zeros,   sy], dim=-1)
        Ry_row1 = torch.stack([zeros,  ones, zeros], dim=-1)
        Ry_row2 = torch.stack([ -sy, zeros,   cy], dim=-1)
        Ry = torch.stack([Ry_row0, Ry_row1, Ry_row2], dim=-2)

        Rz_row0 = torch.stack([  cz,  -sz, zeros], dim=-1)
        Rz_row1 = torch.stack([  sz,   cz, zeros], dim=-1)
        Rz_row2 = torch.stack([zeros, zeros,  ones], dim=-1)
        Rz = torch.stack([Rz_row0, Rz_row1, Rz_row2], dim=-2)

        # Final rotation R = Rz @ Ry @ Rx  (batched matmul)
        R = Rz @ Ry @ Rx  # (64, 3, 3)
        return R.to(dtype)

    def _symmetry_aware_chamfer_loss(self, pred_rot: torch.Tensor, pc: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Vectorized minimum Chamfer distance over a set of symmetry rotations.

        Args:
            pred_rot: (B, 3, 3)
            pc:       (B, N, 3)
            reference:(B, M, 3)
        Returns:
            Scalar tensor: mean over batch of the minimum CD across symmetries.
        """
        # Rotate inputs once using the predicted rotation
        rotated_pc = self._apply_rotation(pc, pred_rot)  # (B, N, 3)

        B, N, _ = rotated_pc.shape
        M = reference.shape[1]

        # Generate candidate symmetry rotations as a tensor (S, 3, 3)
        sym_rots = self._get_symmetry_rotations(rotated_pc.device, rotated_pc.dtype)  # (S,3,3)
        S = sym_rots.shape[0]

        # Tile reference and rotations across the symmetry dimension
        ref_bS = reference.unsqueeze(1).expand(B, S, M, 3).reshape(B * S, M, 3)
        rot_bS = sym_rots.unsqueeze(0).expand(B, S, 3, 3).reshape(B * S, 3, 3)
        ref_rot_bS = self._apply_rotation(ref_bS, rot_bS)  # (B*S, M, 3)

        # Tile rotated pc across the symmetry dimension
        pc_bS = rotated_pc.unsqueeze(1).expand(B, S, N, 3).reshape(B * S, N, 3)  # (B*S, N, 3)

        # Chamfer distance for all (B*S) pairs in one go (compute in fp32 for stability)
        dists = torch.cdist(pc_bS.to(torch.float32), ref_rot_bS.to(torch.float32))  # (B*S, N, M)
        min1 = dists.min(dim=2)[0]             # (B*S, N)
        min2 = dists.min(dim=1)[0]             # (B*S, M)
        cd_pairs = min1.mean(dim=1) + min2.mean(dim=1)  # (B*S,)

        # Reshape to (B, S) and take min over S per batch element
        cd_b_s = cd_pairs.view(B, S)
        tau = float(self.symmetry_softmin_tau) if self.symmetry_softmin_tau is not None else 0.0
        if tau > 0.0:
            # Soft-min: -tau * logsumexp(-x/tau)
            min_per_sample = (-tau * torch.logsumexp(-cd_b_s / tau, dim=1))  # (B,)
        else:
            min_per_sample = cd_b_s.min(dim=1)[0]  # (B,)
        return min_per_sample.mean()

    def _symmetry_aware_geodesic_loss(self, R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        """Symmetry-aware geodesic loss between predicted rotations and orientation labels.

        If the input cloud is generated as P = R_gt @ C (C canonical), the aligning
        rotation is R_gt^T. We thus compare R_pred to R_gt^T by default, optionally
        considering all symmetry-equivalent targets S^T @ R_gt^T.
        """
        if R_pred is None:
            return torch.tensor(0.0, device=self.device)

        B = R_pred.shape[0]
        device = R_pred.device
        dtype = R_pred.dtype

        # Target orientation (invert if configured)
        R_target = R_gt.transpose(-1, -2) if self.geodesic_use_transpose else R_gt  # (B,3,3)

        # Symmetry candidates
        sym_rots = self._get_symmetry_rotations(device, dtype)  # (S,3,3)
        S = sym_rots.shape[0]
        S_T = sym_rots.transpose(-1, -2)

        # Build all candidate targets: (B,S,3,3)
        R_target_b = R_target.unsqueeze(1).expand(B, S, 3, 3)
        S_T_b = S_T.unsqueeze(0).expand(B, S, 3, 3)
        R_cands = torch.matmul(S_T_b, R_target_b).to(torch.float32)

        # Predicted rotations tiled: (B,S,3,3)
        R_pred_b = R_pred.unsqueeze(1).expand(B, S, 3, 3).to(torch.float32)

        # Geodesic angles theta = arccos( (trace(Rp^T R*) - 1)/2 )
        RtR = torch.matmul(R_pred_b.transpose(-1, -2), R_cands)  # (B,S,3,3)
        tr = RtR.diagonal(dim1=-2, dim2=-1).sum(-1)              # (B,S)
        tr = tr.clamp(min=-1.0, max=3.0)
        cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.arccos(cos_theta)                           # (B,S)

        tau = self.geodesic_softmin_tau
        if tau is None:
            tau = self.symmetry_softmin_tau
        tau = float(tau) if tau is not None else 0.0

        if tau > 0.0:
            per_sample = (-tau * torch.logsumexp(-theta / tau, dim=1))  # (B,)
        else:
            per_sample = theta.min(dim=1)[0]
        return per_sample.mean()

    def _step(self, batch, batch_idx, stage: str):
        pc, phase_tensor, grain_tensor, orientation_tensor, quaternion_tensor = batch
        pc = pc.to(device=self.device, dtype=self.dtype, non_blocking=True)
        phase_tensor = phase_tensor.to(device=self.device, non_blocking=True)
        orientation_tensor = orientation_tensor.to(device=self.device, dtype=self.dtype, non_blocking=True)

        # Apply random rotation augmentation during training
        if stage == "train" and self.use_rotation_augmentation:
            batch_size = pc.shape[0]
            random_rot = self._random_rotation_matrix(batch_size, pc.device, pc.dtype)
            pc = self._apply_rotation(pc, random_rot)

        # Forward pass
        inv_z, eq_z, phase_logits, pred_rot = self(pc)

        # Phase classification loss (all samples)
        phase_loss = nn.functional.cross_entropy(phase_logits, phase_tensor)

        # Rotation alignment loss (only for samples with reference point clouds)
        rot_loss = torch.tensor(0.0, device=self.device)
        geo_loss = torch.tensor(0.0, device=self.device)
        if pred_rot is not None and self.reference_pcs:
            # Get phase names from phase indices
            phase_indices = phase_tensor.cpu().numpy()
            idx_to_phase = {v: k for k, v in self.phase_to_idx.items()}
            phase_names = [idx_to_phase.get(int(idx), "unknown") for idx in phase_indices]

            # Filter to only samples that have reference point clouds
            has_ref_mask = torch.tensor(
                [phase_name in self.reference_pcs for phase_name in phase_names],
                dtype=torch.bool,
                device=self.device
            )

            if has_ref_mask.sum() > 0:
                # Get reference point clouds for samples that have them
                pc_with_ref = pc[has_ref_mask]
                pred_rot_with_ref = pred_rot[has_ref_mask]
                phase_names_with_ref = [pn for pn, has_ref in zip(phase_names, has_ref_mask.cpu().numpy()) if has_ref]

                reference_pcs = self._get_reference_pc(phase_names_with_ref)
                if reference_pcs is not None:
                    reference_pcs = reference_pcs.to(device=self.device, dtype=pc_with_ref.dtype)
                    # Use symmetry-aware alignment loss to account for equivalent orientations
                    rot_loss = self._symmetry_aware_chamfer_loss(pred_rot_with_ref, pc_with_ref, reference_pcs)

        # Optional symmetry-aware geodesic supervision w.r.t. orientation labels
        if pred_rot is not None and self.geodesic_loss_weight > 0.0:
            geo_loss = self._symmetry_aware_geodesic_loss(pred_rot, orientation_tensor)

        # Total loss
        total_loss = (self.phase_loss_weight * phase_loss +
                     self.rotation_loss_weight * rot_loss +
                     self.geodesic_loss_weight * geo_loss)

        # Compute accuracy
        with torch.no_grad():
            phase_preds = torch.argmax(phase_logits, dim=1)
            accuracy = (phase_preds == phase_tensor).float().mean()

        # Cache predictions for epoch-end metrics
        if stage in self._supervised_cache:
            self._supervised_cache[stage]["preds"].append(phase_preds.detach().cpu())
            self._supervised_cache[stage]["labels"].append(phase_tensor.detach().cpu())

        # Log metrics
        self.log(f"{stage}/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/phase_loss", phase_loss, on_step=True, on_epoch=True)
        self.log(f"{stage}/rotation_loss", rot_loss, on_step=True, on_epoch=True)
        if self.geodesic_loss_weight > 0.0:
            self.log(f"{stage}/geodesic_loss", geo_loss, on_step=True, on_epoch=True)
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
