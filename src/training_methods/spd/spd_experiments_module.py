import torch
import numpy as np
from src.training_methods.spd.spd_module import ShapePoseDisentanglement
from src.training_methods.spd.rot_heads import kabsch_rotation
from src.utils.spd_utils import order_points_for_kabsch, rotation_geodesic
from src.utils.spd_metrics import (
    test_rotation_equivariance_sample,
    compute_canonical_consistency_metrics
)

class SPDExperimentsModule(ShapePoseDisentanglement):
    """
    Specialized SPD module for ModelNet40 experiments.
    Adapts batch unpacking and adds specific metrics.
    """
    
    @staticmethod
    def _normalize_pc(pc):
        """
        Normalize point cloud to zero mean and unit sphere.
        pc: (B, N, 3)
        """
        # Centering
        centroid = torch.mean(pc, dim=1, keepdim=True)
        pc = pc - centroid
        
        # Scaling to unit sphere
        # max distance from origin
        m = torch.max(torch.sqrt(torch.sum(pc**2, dim=2, keepdim=True)), dim=1, keepdim=True)[0]
        
        # Avoid div by zero
        m = torch.maximum(m, torch.tensor(1e-6, device=pc.device, dtype=pc.dtype))
        pc = pc / m
        return pc

    def _unpack_batch(self, batch):
        """
        Unpack ModelNet batch: (points, label, class_name)
        """
        if not isinstance(batch, (tuple, list)):
            return batch, {}
            
        pc = batch[0]
        
        if torch.is_tensor(pc):
            pc = SPDExperimentsModule._normalize_pc(pc)
        
        labels = {}
        
        # ModelNet loader returns (pc, label_idx, class_name)
        if len(batch) >= 2:
            labels["phase"] = batch[1] # Use class label as phase
        
        # No ground truth orientation/grain for ModelNet
        labels["grain"] = None
        labels["orientation"] = None 
        labels["quaternion"] = None
        
        return pc, labels

    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Augmentation parameters for ModelNet experiments
        self.aug_noise_scale = 0.0
        self.aug_rotation_scale = 0.0
        if hasattr(cfg, 'augmentation'):
            self.aug_noise_scale = cfg.augmentation.noise_scale
            self.aug_rotation_scale = cfg.augmentation.rotation_scale
            self.aug_jitter_scale = getattr(cfg.augmentation, 'jitter_scale', 0.0)
            self.aug_scaling_range = getattr(cfg.augmentation, 'scaling_range', 0.0)
        else:
            self.aug_jitter_scale = 0.0
            self.aug_scaling_range = 0.0

    def _apply_augmentations(self, pc, labels):
        """Apply noise and rotation augmentations."""
        # Noise
        if self.aug_noise_scale > 0:
            noise = torch.randn_like(pc) * self.aug_noise_scale
            pc = pc + noise
            
        # Jitter (Translation)
        if self.aug_jitter_scale > 0:
            jitter = torch.randn(pc.shape[0], 1, 3, device=pc.device) * self.aug_jitter_scale
            pc = pc + jitter
            
        # Scaling
        if self.aug_scaling_range > 0:
            # Scale factor between [1-scale, 1+scale]
            scales = (torch.rand(pc.shape[0], 1, 1, device=pc.device) * 2.0 - 1.0) * self.aug_scaling_range + 1.0
            pc = pc * scales
        
        # Rotation
        if self.aug_rotation_scale > 0:
            # Generate random rotations
            B = pc.shape[0]
            rand_mat = torch.randn(B, 3, 3, device=pc.device, dtype=pc.dtype)
            q, r = torch.linalg.qr(rand_mat)
            d = torch.diagonal(r, dim1=-2, dim2=-1).sign()
            q *= d.unsqueeze(-1)
            # Ensure det is 1 (rotation, not reflection)
            det = torch.det(q)
            mask = det < 0
            if mask.any():
                q[mask, :, 0] *= -1
            
            rot_aug = q
            
            # Apply rotation
            pc = (rot_aug @ pc.transpose(1, 2)).transpose(1, 2).contiguous()
            
            # Update ground truth orientation
            if labels.get("orientation") is None:
                # If no orientation exists, assume identity (input was canonical)
                labels["orientation"] = torch.eye(3, device=pc.device, dtype=pc.dtype).unsqueeze(0).expand(pc.shape[0], -1, -1)
            
            # New orientation = R_aug @ Old_orientation
            labels["orientation"] = rot_aug @ labels["orientation"]
                
        return pc, labels

    def _compute_losses(self, recon, cano, rot, pc, inv_z, vq_loss=0.0, labels=None):
        """Override to add new metrics for ModelNet experiments."""
        # Call parent to get base losses
        losses, sinkhorn_blur = super()._compute_losses(recon, cano, rot, pc, inv_z, vq_loss)
        
        from src.utils.spd_utils import to_float32
        from src.loss.reconstruction_loss import chamfer_distance, sinkhorn_distance
        
        recon_f32, cano_f32, pc_f32 = to_float32(recon, cano, pc)
        
        # Add new metrics within no_grad context
        with torch.no_grad():
            # "Before rotation" metrics
            point_reduction = self._loss_param("chamfer", "point_reduction", "mean")
            losses['emd_pred_canonical_vs_input'], _ = sinkhorn_distance(cano_f32.contiguous(), pc_f32, blur=sinkhorn_blur)
            losses['chamfer_pred_canonical_vs_input'], _ = chamfer_distance(cano_f32, pc_f32, squared=False, point_reduction=point_reduction)

            if labels is not None:
                gt_rot = labels.get("orientation")
                
                # If no orientation provided, assume identity (input is canonical)
                if gt_rot is None:
                    B = pc_f32.shape[0]
                    gt_rot = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
                else:
                    gt_rot = gt_rot.to(device=self.device, dtype=torch.float32)
                
                # pc_unrotated_gt = R_gt^T @ pc (The original object in canonical frame)
                pc_unrotated_gt = (gt_rot.transpose(1, 2) @ pc_f32.transpose(1, 2)).transpose(1, 2).contiguous()
                
                # Pred Canonical vs GT Canonical
                losses['emd_pred_canonical_vs_gt_canonical'], _ = sinkhorn_distance(cano_f32.contiguous(), pc_unrotated_gt, blur=sinkhorn_blur)
                losses['chamfer_pred_canonical_vs_gt_canonical'], _ = chamfer_distance(cano_f32, pc_unrotated_gt, squared=False, point_reduction=point_reduction)
                
                # Rotation Correctness
                if rot is not None:
                    pc_derotated = (rot.transpose(1, 2).float() @ pc_f32.transpose(1, 2)).transpose(1, 2).contiguous()
                    losses['rot_correctness'], _ = chamfer_distance(pc_derotated, pc_unrotated_gt, squared=False, point_reduction=point_reduction)
        
        return losses, sinkhorn_blur

    def _compute_kabsch_consistency(self, cano, pc, rot):
        """
        Compute consistency between predicted rotation and Kabsch rotation
        derived from canonical and input point clouds (ordered).
        """
        # Order points to establish correspondence for Kabsch
        cano_ordered = order_points_for_kabsch(cano)
        pc_ordered = order_points_for_kabsch(pc)
        
        # Compute Kabsch rotation: R that takes cano to pc
        R_kabsch = kabsch_rotation(cano_ordered, pc_ordered)
        
        # Compare with predicted rotation
        # rot is predicted R s.t. pc ~ rot @ cano
        error = rotation_geodesic(rot, R_kabsch)
        return error

    def _step(self, batch, batch_idx, stage: str):
        # Override _step to add Kabsch consistency metric logging
        pc, labels = self._unpack_batch(batch)
        pc = pc.to(device=self.device, dtype=self.dtype, non_blocking=True)

        if stage == "train":
            pc, labels = self._apply_augmentations(pc, labels)

        # Forward pass
        inv_z, recon, cano, rot, vq_loss = self(pc)

        # Cache for metrics if needed
        if stage in self._supervised_cache:
            self._cache_supervised_batch(stage, inv_z, labels, recon, cano, rot, pc)

        # Compute all losses (calls overridden _compute_losses with new metrics)
        losses, sinkhorn_blur = self._compute_losses(recon, cano, rot, pc, inv_z, vq_loss, labels=labels)

        # Build total loss
        total_loss = losses['recon'] + self.ortho_scale * losses['ortho']
        if 'kl' in losses:
            total_loss += self.kl_latent_loss_scale * losses['kl']
        if 'vq' in losses:
            total_loss += losses['vq']
        total_loss = total_loss.to(self.dtype)

        # Prepare metrics for logging
        metrics_to_log = {
            'loss': total_loss,
            f'{self.loss_name}_loss': losses['recon'],
            'chamfer': losses.get('chamfer', 0.0),
            'emd': losses['emd_after'],
            'emd_before_rot': losses.get('emd_pred_canonical_vs_input', 0.0),
            'chamfer_before_rot': losses.get('chamfer_pred_canonical_vs_input', 0.0),
            'ortho': losses['ortho'],
        }
        if 'emd_pred_canonical_vs_gt_canonical' in losses:
            metrics_to_log['emd_pred_canonical_vs_gt_canonical'] = losses['emd_pred_canonical_vs_gt_canonical']
        if 'chamfer_pred_canonical_vs_gt_canonical' in losses:
            metrics_to_log['chamfer_pred_canonical_vs_gt_canonical'] = losses['chamfer_pred_canonical_vs_gt_canonical']
        if 'rot_correctness' in losses:
            metrics_to_log['rot_correctness'] = losses['rot_correctness']
        if 'kl' in losses:
            metrics_to_log['kl_loss'] = losses['kl']
        if 'vq' in losses:
            metrics_to_log['vq_loss'] = losses['vq']
            
        # Add Kabsch consistency metric
        if rot is not None:
            kabsch_error = self._compute_kabsch_consistency(cano, pc, rot)
            metrics_to_log['kabsch_consistency_deg'] = kabsch_error * (180.0 / np.pi)

        # Log all metrics
        self._log_metrics(stage, metrics_to_log, prog_bar_keys={'loss'}, batch_size=pc.shape[0])

        return total_loss

    def _run_expensive_test_metrics(self, stage, labels):
        """
        Override to ensure we run equivariance tests even without reference_pcs
        by sampling from the batch/cache if needed, or just skip if strictly required.
        But for ModelNet experiments, we want to run these.
        """
        # If reference_pcs is None, we can't run these tests easily as they expect
        # a dictionary of {phase_name: pc}.
        # We can construct a temporary reference set from the cache.
        
        if self.reference_pcs is None:
            # Build reference PCs from cache
            cache = self._supervised_cache.get(stage)
            if cache and cache["originals"] and cache["phase"]:
                originals = torch.cat(cache["originals"], dim=0).numpy()
                phase_ids = torch.cat(cache["phase"], dim=0).numpy()
                
                self.reference_pcs = {}
                unique_phases = np.unique(phase_ids)
                for pid in unique_phases:
                    # Pick first sample of this phase
                    idx = np.where(phase_ids == pid)[0][0]
                    # We don't have class names here easily unless we store them, 
                    # but spd_metrics expects phase names or we can just use IDs if we adapt it.
                    # spd_metrics.test_rotation_equivariance_sample maps IDs to names.
                    # We might need to monkeypatch or update spd_metrics to handle int IDs.
                    # For now, let's assume we can pass a dict with int keys if we modify spd_metrics
                    # OR we rely on the fact that we will pass phase_labels as ints.
                    
                    # Actually spd_metrics.py has a hardcoded phase_map!
                    # phase_map = {0: 'crystal_fcc', ...}
                    # This is bad for ModelNet.
                    # I should override test_rotation_equivariance_sample or modify spd_metrics.py.
                    pass
        
        # Since spd_metrics.py has hardcoded phase map, I should probably 
        # reimplement the test loop here or modify spd_metrics.py to be generic.
        # Modifying spd_metrics.py is better for long term, but I can also just 
        # copy the function here and adapt it to avoid modifying shared code too much.
        # Let's copy and adapt locally to be safe and specific.
        
        self._run_adapted_equivariance_test(stage, labels)
        self._run_adapted_consistency_test(stage, labels)

    def _run_adapted_equivariance_test(self, stage, labels):
        # Adapted from spd_metrics.py but without hardcoded phase map
        metrics = {'equivariance_errors': []}
        
        # Get reference PCs from cache
        cache = self._supervised_cache.get(stage)
        if not cache or not cache["originals"]:
            return

        originals = torch.cat(cache["originals"], dim=0)
        phase_ids = torch.cat(cache["phase"], dim=0).cpu().numpy()
        
        unique_phases = np.unique(phase_ids)
        
        self.eval()
        with torch.no_grad():
            for pid in unique_phases:
                # Get a few samples for this phase
                indices = np.where(phase_ids == pid)[0]
                if len(indices) == 0:
                    continue
                
                # Test on up to 3 samples per class
                test_indices = indices[:3]
                
                for idx in test_indices:
                    X_base_t = originals[idx].unsqueeze(0).to(self.device) # (1, N, 3)
                    
                    # Get prediction for original
                    _, _, _, R_pred_orig, _ = self(X_base_t)
                    R_pred_orig_np = R_pred_orig[0].cpu().numpy()
                    
                    # Test with random rotations
                    for _ in range(5): # 5 rotations per sample
                        from scipy.spatial.transform import Rotation
                        R_test = Rotation.random().as_matrix().astype(np.float32)
                        R_test_t = torch.from_numpy(R_test).to(self.device)
                        
                        # Apply rotation: X_rot = X @ R_test.T
                        X_rotated_t = (R_test_t @ X_base_t.transpose(1, 2)).transpose(1, 2)
                        
                        # Get prediction
                        _, _, _, R_pred_rot, _ = self(X_rotated_t)
                        R_pred_rot_np = R_pred_rot[0].cpu().numpy()
                        
                        # Expected: R_pred_rot = R_test @ R_pred_orig
                        R_expected = R_test @ R_pred_orig_np
                        
                        # Error
                        error = rotation_geodesic(
                            torch.from_numpy(R_pred_rot_np).unsqueeze(0),
                            torch.from_numpy(R_expected).unsqueeze(0)
                        )
                        metrics['equivariance_errors'].append(error.item() * (180.0 / np.pi))

        if metrics['equivariance_errors']:
            self._log_metric(stage, "equivariance/mean_deg", np.mean(metrics['equivariance_errors']), on_step=False, on_epoch=True)
            self._log_metric(stage, "equivariance/std_deg", np.std(metrics['equivariance_errors']), on_step=False, on_epoch=True)

    def _run_adapted_consistency_test(self, stage, labels):
        # Adapted reconstruction consistency
        metrics = {}
        reconstruction_errors_all = []
        
        cache = self._supervised_cache.get(stage)
        if not cache or not cache["originals"]:
            return

        originals = torch.cat(cache["originals"], dim=0)
        phase_ids = torch.cat(cache["phase"], dim=0).cpu().numpy()
        unique_phases = np.unique(phase_ids)
        
        self.eval()
        with torch.no_grad():
            for pid in unique_phases:
                indices = np.where(phase_ids == pid)[0]
                if len(indices) == 0:
                    continue
                
                test_indices = indices[:2] # 2 samples per class
                phase_errors = []
                
                for idx in test_indices:
                    X_base_t = originals[idx].unsqueeze(0).to(self.device)
                    
                    for _ in range(5): # 5 rotations
                        from scipy.spatial.transform import Rotation
                        R_test = Rotation.random().as_matrix().astype(np.float32)
                        R_test_t = torch.from_numpy(R_test).to(self.device)
                        
                        X_rotated_t = (R_test_t @ X_base_t.transpose(1, 2)).transpose(1, 2)
                        
                        _, recon, _, _, _ = self(X_rotated_t)
                        
                        # Reconstruction error (Sinkhorn)
                        from src.loss.reconstruction_loss import sinkhorn_distance
                        emd, _ = sinkhorn_distance(recon.contiguous(), X_rotated_t)
                        phase_errors.append(emd.item())
                
                if phase_errors:
                    reconstruction_errors_all.extend(phase_errors)
                    metrics[f'recon_consistency/mean_phase_{int(pid)}'] = np.mean(phase_errors)
                    metrics[f'recon_consistency/std_phase_{int(pid)}'] = np.std(phase_errors)

        if reconstruction_errors_all:
            self._log_metric(stage, "recon_consistency/mean_all", np.mean(reconstruction_errors_all), on_step=False, on_epoch=True)
            self._log_metric(stage, "recon_consistency/std_all", np.std(reconstruction_errors_all), on_step=False, on_epoch=True)
            
        # Log per-phase metrics
        for k, v in metrics.items():
            self._log_metric(stage, k, v, on_step=False, on_epoch=True)
