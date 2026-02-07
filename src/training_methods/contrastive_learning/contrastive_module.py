import itertools
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import pytorch_lightning as pl

import sys, os
sys.path.append(os.getcwd())

from src.models.autoencoders.factory import build_model
from src.utils.pointcloud_ops import crop_to_num_points
from src.utils.spd_metrics import (
    compute_embedding_quality_metrics,
    compute_cluster_metrics,
)
from src.utils.spd_utils import (
    apply_rotation,
    cached_sample_count,
    get_optimizers_and_scheduler,
)
from src.training_methods.equivariant_autoencoder.idec import resolve_latent_dim
from src.training_methods.contrastive_learning.barlow_twins import BarlowTwinsLoss
from src.training_methods.contrastive_learning.vicreg import VICRegLoss
from src.training_methods.spd.rot_heads import sixd_to_so3
from src.models.autoencoders.encoders.vn_encoders import VNLinearLeakyReLU, VNMaxPool


class PoseRotationHead(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        hidden = max(8, int(hidden))
        vn_width = max(8, hidden // 2)
        self.vn_stem = nn.Sequential(
            VNLinearLeakyReLU(1, vn_width, dim=4, negative_slope=0.1, use_batchnorm=False),
            VNLinearLeakyReLU(vn_width, hidden, dim=4, negative_slope=0.1, use_batchnorm=False),
            VNLinearLeakyReLU(hidden, hidden, dim=4, negative_slope=0.1, use_batchnorm=False),
        )
        self.vn_pool = VNMaxPool(hidden)
        self.out = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 6),
        )

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        if cov.dim() != 3 or cov.shape[-2:] != (3, 3):
            raise ValueError(f"Expected cov of shape (B,3,3). Got {tuple(cov.shape)}")
        x = cov.transpose(1, 2).unsqueeze(1)  # (B, 1, 3, 3): 3 points, 1 VN channel
        x = self.vn_stem(x)
        x = self.vn_pool(x)  # (B, C, 3)
        return self.out(x.reshape(x.shape[0], -1))


def _autocast_disabled_context(tensor: torch.Tensor):
    device_type = tensor.device.type
    if device_type == "cuda" and torch.is_autocast_enabled():
        return torch.autocast(device_type=device_type, enabled=False)
    if device_type == "cpu" and hasattr(torch, "is_autocast_cpu_enabled") and torch.is_autocast_cpu_enabled():
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


class BarlowTwinsModule(pl.LightningModule):
    """
    Self-supervised Barlow Twins / VICReg training for point cloud encoders.
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        # Build encoder (decoder is not used for contrastive training)
        self.encoder, _ = build_model(cfg)

        latent_dim = resolve_latent_dim(cfg)
        self.barlow = BarlowTwinsLoss.from_config(cfg, input_dim=latent_dim)
        vicreg_enabled = bool(getattr(cfg, "vicreg_enabled", False))
        self.vicreg = VICRegLoss.from_config(cfg, input_dim=latent_dim) if vicreg_enabled else None

        data_cfg = getattr(cfg, "data", None)
        self.sample_points = int(getattr(data_cfg, "num_points", 0)) if data_cfg is not None else 0
        model_points = getattr(data_cfg, "model_points", None) if data_cfg is not None else None
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

        pose_cfg = getattr(cfg, "pose", None)
        self.pose_enabled = bool(getattr(cfg, "pose_enabled", self._cfg_get(pose_cfg, "enabled", False)))
        self.pose_weight = float(getattr(cfg, "pose_weight", self._cfg_get(pose_cfg, "weight", 0.0)))
        self.pose_start_epoch = int(getattr(cfg, "pose_start_epoch", self._cfg_get(pose_cfg, "start_epoch", 0)))
        self.pose_jitter_std = float(getattr(cfg, "pose_jitter_std", self._cfg_get(pose_cfg, "jitter_std", 0.0)))
        self.pose_seeded_forward = bool(
            getattr(cfg, "pose_seeded_forward", self._cfg_get(pose_cfg, "seeded_forward", True))
        )
        self.pose_head_hidden = int(
            getattr(cfg, "pose_head_hidden", self._cfg_get(pose_cfg, "head_hidden", 128))
        )
        self.pose_view_rotation_mode = str(
            getattr(
                cfg,
                "pose_view_rotation_mode",
                self._cfg_get(pose_cfg, "view_rotation_mode", "full"),
            )
        ).lower()
        self.pose_view_rotation_deg = float(
            getattr(
                cfg,
                "pose_view_rotation_deg",
                self._cfg_get(pose_cfg, "view_rotation_deg", 0.0),
            )
        )
        self.pose_unambiguous_rotation = bool(
            getattr(
                cfg,
                "pose_unambiguous_rotation",
                self._cfg_get(pose_cfg, "unambiguous_rotation", True),
            )
        )
        self.pose_unambiguous_margin_deg = float(
            getattr(
                cfg,
                "pose_unambiguous_margin_deg",
                self._cfg_get(pose_cfg, "unambiguous_margin_deg", 1.0),
            )
        )

        sym_cfg = getattr(cfg, "symmetry", None)
        if sym_cfg is None and pose_cfg is not None:
            sym_cfg = self._cfg_get(pose_cfg, "symmetry", None)
        self.symmetry_type = str(self._cfg_get(sym_cfg, "type", "cubic24")).lower()
        self.symmetry_beta = float(self._cfg_get(sym_cfg, "beta", 30.0))
        self.symmetry_angle_eps = float(self._cfg_get(sym_cfg, "angle_eps", 1e-6))
        sym_mats = self._build_symmetry_mats(sym_cfg)
        self.register_buffer("symmetry_mats", sym_mats)
        self.pose_unambiguous_cap_rad = self._compute_unambiguous_cap_rad()

        self.pose_head = PoseRotationHead(hidden=self.pose_head_hidden) if self.pose_enabled else None

        # Caches for embedding metrics
        self._supervised_cache = {
            "train": {"latents": [], "class_id": []},
            "val": {"latents": [], "class_id": []},
            "test": {"latents": [], "class_id": []},
        }
        self.max_supervised_samples = cfg.max_supervised_samples if hasattr(cfg, 'max_supervised_samples') else 8192
        self.max_test_samples = cfg.max_test_samples if hasattr(cfg, 'max_test_samples') else 1000

    @property
    def barlow_projector(self):
        return self.barlow.projector

    @property
    def vicreg_projector(self):
        return self.vicreg.projector if self.vicreg is not None else None

    @staticmethod
    def _cfg_get(obj, name: str, default=None):
        if obj is None:
            return default
        if hasattr(obj, "get"):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @staticmethod
    def _perm_parity(perm: tuple[int, int, int]) -> int:
        inv = 0
        for i in range(len(perm)):
            for j in range(i + 1, len(perm)):
                if perm[i] > perm[j]:
                    inv += 1
        return -1 if (inv % 2) else 1

    @classmethod
    def _cubic_symmetry_mats(cls) -> torch.Tensor:
        mats = []
        for perm in itertools.permutations([0, 1, 2]):
            parity = cls._perm_parity(perm)
            for signs in itertools.product([-1, 1], repeat=3):
                det = parity * (signs[0] * signs[1] * signs[2])
                if det <= 0:
                    continue
                mat = torch.zeros(3, 3, dtype=torch.float32)
                mat[0, perm[0]] = signs[0]
                mat[1, perm[1]] = signs[1]
                mat[2, perm[2]] = signs[2]
                mats.append(mat)
        if not mats:
            return torch.eye(3, dtype=torch.float32).unsqueeze(0)
        return torch.stack(mats, dim=0)

    @staticmethod
    def _ensure_identity(mats: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if mats.numel() == 0:
            return torch.eye(3, dtype=torch.float32).unsqueeze(0)
        identity = torch.eye(3, dtype=mats.dtype).unsqueeze(0)
        diffs = (mats - identity).abs().max(dim=-1).values.max(dim=-1).values
        if (diffs <= eps).any():
            return mats
        return torch.cat([identity, mats], dim=0)

    def _build_symmetry_mats(self, sym_cfg) -> torch.Tensor:
        sym_type = self.symmetry_type
        if sym_type == "none":
            mats = torch.eye(3, dtype=torch.float32).unsqueeze(0)
            return mats
        if sym_type == "cubic24":
            mats = self._cubic_symmetry_mats()
            return mats
        else:
            raise ValueError(f"Unknown symmetry.type '{sym_type}'. Expected none, cubic24.")

    @staticmethod
    def _random_rotation_matrices(batch_size: int, *, device, dtype) -> torch.Tensor:
        rot_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        rand = torch.randn(batch_size, 3, 3, device=device, dtype=rot_dtype)
        q, r = torch.linalg.qr(rand)
        d = torch.diagonal(r, dim1=-2, dim2=-1).sign()
        q = q * d.unsqueeze(-1)
        det = torch.det(q)
        neg = det < 0
        if neg.any():
            q[neg, :, 0] *= -1
        return q.to(dtype=dtype)

    def _sample_axis_angle_rotation(
        self,
        batch_size: int,
        *,
        max_rad: float,
        device,
        dtype,
    ) -> torch.Tensor:
        if max_rad <= 0:
            return torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        axis = self.barlow._random_unit_vectors(batch_size, device=device, dtype=dtype)
        angle = (torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 1.0) * float(max_rad)
        return self.barlow._axis_angle_to_matrix(axis, angle)

    def _compute_unambiguous_cap_rad(self) -> float:
        mats = self.symmetry_mats.detach().to(device="cpu", dtype=torch.float32)
        if mats.shape[0] <= 1:
            return math.pi

        eye = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        diffs = (mats - eye).abs().amax(dim=(-2, -1))
        non_identity = mats[diffs > 1e-6]
        if non_identity.numel() == 0:
            return math.pi

        eye_n = eye.expand(non_identity.shape[0], -1, -1)
        angles, _ = self._rotation_geodesic_angles(
            eye_n,
            non_identity,
            eps=float(self.symmetry_angle_eps),
        )
        min_sym_angle = float(angles.min().item())
        margin = math.radians(max(0.0, float(self.pose_unambiguous_margin_deg)))
        return max(0.0, 0.5 * min_sym_angle - margin)

    def _sample_pose_relative_rotation(self, batch_size: int, *, device, dtype) -> torch.Tensor:
        mode = self.pose_view_rotation_mode
        if mode == "none":
            return torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

        if mode == "full":
            if not self.pose_unambiguous_rotation:
                return self._random_rotation_matrices(batch_size, device=device, dtype=dtype)
            max_rad = math.pi
        else:
            max_deg = float(self.pose_view_rotation_deg)
            max_rad = math.radians(max(0.0, max_deg))

        if self.pose_unambiguous_rotation:
            max_rad = min(max_rad, float(self.pose_unambiguous_cap_rad))

        return self._sample_axis_angle_rotation(
            batch_size,
            max_rad=max_rad,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _prepare_eq_latent(eq_z: torch.Tensor | None) -> torch.Tensor | None:
        if eq_z is None:
            return None
        if eq_z.dim() == 3 and eq_z.shape[-1] == 3:
            return eq_z
        if eq_z.dim() == 4:
            if eq_z.shape[-1] == 3:
                return eq_z.mean(dim=-2)
            if eq_z.shape[2] == 3:
                return eq_z.mean(dim=-1)
        return None

    @staticmethod
    def _rotation_geodesic_angles(
        pred: torch.Tensor, target: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = pred.transpose(-1, -2) @ target
        trace = delta.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_theta = (0.5 * (trace - 1.0))
        cos_clamped = cos_theta.clamp(-1.0 + eps, 1.0 - eps)
        angle = torch.arccos(cos_clamped)
        clamped = cos_clamped != cos_theta
        return angle, clamped

    def _seeded_encoder_output(self, pc: torch.Tensor, seed: int):
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if pc.is_cuda else None
        torch.manual_seed(seed)
        if pc.is_cuda:
            torch.cuda.manual_seed_all(seed)
        try:
            enc_out = self.encoder(self._prepare_encoder_input(pc))
            return self._split_encoder_output(enc_out)
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _prepare_encoder_input(self, pc: torch.Tensor) -> torch.Tensor:
        if getattr(self.encoder, "expects_channel_first", False):
            return pc.permute(0, 2, 1).contiguous()
        return pc

    def _split_encoder_output(self, enc_out):
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

    def _prepare_model_input(self, pc: torch.Tensor) -> torch.Tensor:
        out = pc
        if self.model_points is not None:
            out = crop_to_num_points(out, self.model_points)
        return out

    def _pose_should_run(self) -> bool:
        return bool(
            self.pose_enabled
            and self.pose_weight > 0
            and self.pose_head is not None
            and int(self.current_epoch) >= self.pose_start_epoch
        )

    def _compute_pose_loss(
        self,
        pc: torch.Tensor,
        batch_idx: int,
    ):
        if not self._pose_should_run():
            return None, {}
        if pc.dim() != 3 or pc.shape[-1] != 3:
            return None, {}

        # Pose supervision uses two rotations of the same base point cloud.
        xa_base = self._prepare_model_input(pc)
        if self.pose_jitter_std > 0:
            jitter = torch.randn_like(xa_base) * float(self.pose_jitter_std)
            xa_base = xa_base + jitter

        bsz = xa_base.shape[0]
        device = xa_base.device
        dtype = xa_base.dtype

        if self.pose_view_rotation_mode == "none":
            rot_anchor = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1, -1)
        else:
            rot_anchor = self._random_rotation_matrices(bsz, device=device, dtype=dtype)

        rot_rel = self._sample_pose_relative_rotation(bsz, device=device, dtype=dtype)
        rot_a = rot_anchor
        rot_b = rot_rel @ rot_anchor
        xa = apply_rotation(xa_base, rot_a)
        xb = apply_rotation(xa_base, rot_b)
        rot = rot_rel
        device = xa.device
        dtype = xa.dtype

        seed = int(self.global_step) + int(batch_idx) * 100003
        if self.pose_seeded_forward:
            _, eq_a = self._seeded_encoder_output(xa, seed)
            _, eq_b = self._seeded_encoder_output(xb, seed)
        else:
            eq_a = self._split_encoder_output(self.encoder(self._prepare_encoder_input(xa)))[1]
            eq_b = self._split_encoder_output(self.encoder(self._prepare_encoder_input(xb)))[1]

        eq_a = self._prepare_eq_latent(eq_a)
        eq_b = self._prepare_eq_latent(eq_b)
        if eq_a is None or eq_b is None:
            return None, {}

        cov = torch.einsum("bci,bcj->bij", eq_b, eq_a)
        r6 = self.pose_head(cov)
        rot_hat = sixd_to_so3(r6, eps=1e-6)

        sym_mats = self.symmetry_mats.to(device=device, dtype=dtype)
        rot_targets = rot.unsqueeze(1) @ sym_mats.unsqueeze(0)

        with _autocast_disabled_context(rot_hat):
            rot_hat_f = rot_hat.to(dtype=torch.float32)
            rot_targets_f = rot_targets.to(dtype=torch.float32)

            angles_all, _ = self._rotation_geodesic_angles(
                rot_hat_f.unsqueeze(1), rot_targets_f, eps=float(self.symmetry_angle_eps)
            )
            angles_sq = angles_all.pow(2)
            if self.symmetry_beta > 0:
                beta = float(self.symmetry_beta)
                loss_sym = -torch.logsumexp(-beta * angles_sq, dim=1) / beta
            else:
                loss_sym = angles_sq.min(dim=1).values

        pose_loss = loss_sym.mean()

        if not torch.isfinite(pose_loss).item():
            # pose_nonfinite: indicator that pose loss produced NaN/Inf on this step.
            metrics = {"pose_nonfinite": xa.new_tensor(1.0)}
            pose_loss = torch.nan_to_num(pose_loss, nan=0.0, posinf=0.0, neginf=0.0)
            return pose_loss, metrics

        theta_sym = angles_all.min(dim=1).values
        rad_to_deg = 180.0 / math.pi
        theta_sym_deg = (theta_sym * rad_to_deg).to(dtype=torch.float32)
        metrics = {
            "pose_theta_sym_mean": theta_sym_deg.mean(),
        }

        with torch.no_grad():
            rot_f = rot.to(dtype=torch.float32)
            eye = torch.eye(3, device=rot_f.device, dtype=rot_f.dtype).unsqueeze(0).expand_as(rot_f)
            target_ang, _ = self._rotation_geodesic_angles(eye, rot_f, eps=float(self.symmetry_angle_eps))
            metrics["pose_target_angle_mean_deg"] = (target_ang * (180.0 / math.pi)).mean()
        return pose_loss, metrics

    @staticmethod
    def _unpack_batch(batch):
        """Unpack batch dict into points and metadata."""
        if isinstance(batch, dict):
            pc = batch["points"]
            meta = {
                "class_id": batch.get("class_id"),
                "instance_id": batch.get("instance_id"),
                "rotation": batch.get("rotation"),
            }
            return pc, meta
        if not isinstance(batch, (tuple, list)):
            return batch, {}
        return batch[0], {}

    def _invariant_latent(self, inv_z, eq_z):
        return self.barlow._invariant(inv_z, eq_z)

    def forward(self, pc: torch.Tensor):
        enc_out = self.encoder(self._prepare_encoder_input(pc))
        inv_z, eq_z = self._split_encoder_output(enc_out)
        z = self._invariant_latent(inv_z, eq_z)
        return z, inv_z, eq_z

    def _step(self, batch, batch_idx, stage: str):
        pc_raw, meta = self._unpack_batch(batch)
        pc_raw = pc_raw.to(device=self.device, dtype=self.dtype, non_blocking=True)
        pc = self._prepare_model_input(pc_raw)

        # Embeddings for metrics
        inv_z, eq_z = self._split_encoder_output(self.encoder(self._prepare_encoder_input(pc)))
        z = self._invariant_latent(inv_z, eq_z)
        if z is not None and stage in self._supervised_cache:
            self._cache_supervised_batch(stage, z, meta)
        # Barlow Twins loss (self-supervised)
        losses = {}
        barlow_loss, barlow_metrics = self.barlow.compute_loss(
            pc=pc_raw,
            encoder=self.encoder,
            prepare_input=self._prepare_encoder_input,
            split_output=self._split_encoder_output,
            current_epoch=int(self.current_epoch),
        )
        if barlow_loss is not None:
            losses["barlow"] = barlow_loss
        # Barlow metrics come from BarlowTwinsLoss (invariance/redundancy diagnostics).
        for name, value in barlow_metrics.items():
            self._log_metric(stage, name, value)

        # VICReg loss (self-supervised)
        if self.vicreg is not None:
            vicreg_loss, vicreg_metrics = self.vicreg.compute_loss(
                pc=pc_raw,
                encoder=self.encoder,
                prepare_input=self._prepare_encoder_input,
                split_output=self._split_encoder_output,
                current_epoch=int(self.current_epoch),
            )
            if vicreg_loss is not None:
                losses["vicreg"] = vicreg_loss
            # VICReg metrics come from VICRegLoss (variance/invariance/covariance diagnostics).
            for name, value in vicreg_metrics.items():
                self._log_metric(stage, name, value)

        # Pose loss (symmetry-aware)
        pose_loss, pose_metrics = self._compute_pose_loss(
            pc,
            batch_idx,
        )
        if pose_loss is not None:
            losses["pose"] = pose_loss
            for name, value in pose_metrics.items():
                self._log_metric(stage, name, value)

        total_loss = None
        if "barlow" in losses:
            total_loss = self.barlow.weight * losses["barlow"]
        if "vicreg" in losses and self.vicreg is not None:
            vicreg_total = self.vicreg.weight * losses["vicreg"]
            total_loss = vicreg_total if total_loss is None else total_loss + vicreg_total
        if "pose" in losses:
            pose_total = self.pose_weight * losses["pose"]
            total_loss = pose_total if total_loss is None else total_loss + pose_total
        if total_loss is None:
            total_loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)

        if not torch.isfinite(total_loss).item():
            # loss_nonfinite: indicator that the aggregated loss produced NaN/Inf.
            self._log_metric(stage, "loss_nonfinite", 1.0, on_step=True, on_epoch=False)
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)

        metrics_to_log = {
            # Total weighted optimization objective.
            "loss": total_loss,
        }
        if "barlow" in losses:
            # Unweighted Barlow objective term.
            metrics_to_log["barlow"] = losses["barlow"]
        if "vicreg" in losses:
            # Unweighted VICReg objective term.
            metrics_to_log["vicreg"] = losses["vicreg"]
        if "pose" in losses:
            # Unweighted pose-regression objective term.
            metrics_to_log["pose"] = losses["pose"]

        prog_bar_keys = {"loss"}
        for name, value in metrics_to_log.items():
            self._log_metric(stage, name, value, prog_bar=(name in prog_bar_keys))

        return total_loss

    def training_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return get_optimizers_and_scheduler(self.hparams, self.parameters())

    def _log_metric(self, stage: str, name: str, value, *, on_step=None, on_epoch=None, **kwargs) -> None:
        if on_step is None:
            on_step = stage == "train"
        if on_epoch is None:
            on_epoch = stage != "train"
        log_kwargs = dict(kwargs)
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
        cache = self._supervised_cache.get(stage)
        if cache is None:
            return
        for key in cache:
            cache[key].clear()

    def _cache_limit_for_stage(self, stage: str):
        if stage == "test":
            return self.max_test_samples
        if stage in {"train", "val"}:
            return self.max_supervised_samples
        return None

    def _cache_supervised_batch(self, stage: str, z: torch.Tensor, meta: dict) -> None:
        cache = self._supervised_cache.get(stage)
        if cache is None:
            return

        limit = self._cache_limit_for_stage(stage)
        remaining = None
        if limit is not None and limit > 0:
            cached = cached_sample_count(cache)
            remaining = int(limit - cached)
            if remaining <= 0:
                return

        if z is None:
            return

        batch_size = int(z.shape[0])
        effective_batch = batch_size if remaining is None else min(batch_size, remaining)
        if effective_batch <= 0:
            return

        class_id = meta.get("class_id")
        if class_id is None:
            return
        if not torch.is_tensor(class_id):
            class_id = torch.as_tensor(class_id)
        class_id = class_id.detach().view(-1)
        effective_batch = min(effective_batch, class_id.shape[0])
        if effective_batch <= 0:
            return

        cache["latents"].append(z[:effective_batch].detach().to(torch.float32).cpu())
        cache["class_id"].append(class_id[:effective_batch].cpu())

    def _log_supervised_metrics(self, stage: str) -> None:
        cache = self._supervised_cache.get(stage)
        if cache is None:
            return

        if not cache["latents"] or not cache["class_id"]:
            for key in cache:
                cache[key].clear()
            return

        latents = torch.cat(cache["latents"], dim=0).numpy()
        labels = torch.cat(cache["class_id"], dim=0).numpy()

        metrics = compute_cluster_metrics(latents, labels, stage)
        if metrics:
            # class/* metrics: clustering/classification quality against class_id labels.
            for name, value in metrics.items():
                self._log_metric(
                    stage,
                    f"class/{name.lower()}",
                    value,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        try:
            emb_metrics = compute_embedding_quality_metrics(latents, labels, include_expensive=(stage == "test"))
            # embedding/* metrics: geometry/quality scores of latent embedding with class labels.
            for name, value in emb_metrics.items():
                self._log_metric(
                    stage,
                    f"embedding/{name}",
                    value,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        except Exception as e:
            print(f"Error computing embedding quality metrics: {e}")

        for key in cache:
            cache[key].clear()
