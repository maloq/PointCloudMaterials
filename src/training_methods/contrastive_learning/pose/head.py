import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseRotationHead(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        *,
        delta_max_rad: float = 0.5,
        generic_in_dim: int = 18,
    ) -> None:
        super().__init__()
        hidden = max(16, int(hidden))
        w_hidden = max(8, hidden // 2)
        self.delta_max_rad = max(0.0, float(delta_max_rad))
        self.generic_in_dim = max(1, int(generic_in_dim))
        self.generic_svd_k = 4
        # Compact generic summary dimension: avoids large dense layers on flattened Gram inputs.
        self.generic_summary_dim = 26

        # Channel confidence branch for weighted cross-covariance:
        # w_i = softplus(MLP([||a_i||, ||b_i||, a_i.b_i, ||a_i x b_i||])).
        self.weight_mlp = nn.Sequential(
            nn.Linear(4, w_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(w_hidden, w_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(w_hidden, 1),
        )

        # Covariance summary branch (vec(M) + singular values -> residual + uncertainty).
        self.cov_stem = nn.Sequential(
            nn.Linear(12, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.cov_delta = nn.Linear(hidden, 3)
        self.cov_logvar = nn.Linear(hidden, 3)
        nn.init.zeros_(self.cov_delta.weight)
        nn.init.zeros_(self.cov_delta.bias)
        nn.init.zeros_(self.cov_logvar.weight)
        nn.init.zeros_(self.cov_logvar.bias)

        # Generic fallback branch (e.g. Gram features).
        self.generic_stem = nn.Sequential(
            nn.Linear(self.generic_summary_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.generic_rot = nn.Linear(hidden, 6)
        self.generic_logvar = nn.Linear(hidden, 3)

    def _module_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        p = next(self.parameters())
        return p.device, p.dtype

    @staticmethod
    def _procrustes_rotation_from_cov(cov: torch.Tensor) -> torch.Tensor:
        u, _, vh = torch.linalg.svd(cov, full_matrices=False)
        rot = u @ vh
        det = torch.det(rot)
        neg = det < 0
        if neg.any():
            sfix = torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0).expand(cov.shape[0], -1, -1).clone()
            sfix[neg, -1, -1] = -1.0
            rot = u @ sfix @ vh
        return rot

    @staticmethod
    def _matrix_to_sixd(rot: torch.Tensor) -> torch.Tensor:
        return torch.cat([rot[..., :, 0], rot[..., :, 1]], dim=-1)

    @staticmethod
    def _skew(v: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros(v.shape[0], device=v.device, dtype=v.dtype)
        return torch.stack(
            [
                torch.stack([zero, -v[:, 2], v[:, 1]], dim=-1),
                torch.stack([v[:, 2], zero, -v[:, 0]], dim=-1),
                torch.stack([-v[:, 1], v[:, 0], zero], dim=-1),
            ],
            dim=1,
        )

    @classmethod
    def _axis_angle_to_matrix(cls, omega: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if omega.dim() != 2 or omega.shape[-1] != 3:
            raise ValueError(f"Expected omega of shape (B,3). Got {tuple(omega.shape)}")
        theta = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
        axis = omega / theta.clamp_min(eps)
        k = cls._skew(axis)
        eye = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0).expand(omega.shape[0], -1, -1)
        sin_t = torch.sin(theta).unsqueeze(-1)
        cos_t = torch.cos(theta).unsqueeze(-1)
        k2 = k @ k
        rot = eye + sin_t * k + (1.0 - cos_t) * k2

        k_omega = cls._skew(omega)
        rot_small = eye + k_omega + 0.5 * (k_omega @ k_omega)
        small = theta.squeeze(-1).abs() < 1e-4
        return torch.where(small[:, None, None], rot_small, rot)

    def _clip_delta(self, delta_omega: torch.Tensor) -> torch.Tensor:
        if self.delta_max_rad <= 0:
            return delta_omega
        norm = torch.linalg.vector_norm(delta_omega, dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(self.delta_max_rad / norm, max=1.0)
        return delta_omega * scale

    @staticmethod
    def _stats6(v: torch.Tensor) -> torch.Tensor:
        if v.dim() != 2:
            raise ValueError(f"Expected 2D tensor for statistics. Got {tuple(v.shape)}")
        rms = torch.sqrt((v * v).mean(dim=-1).clamp_min(1e-12))
        return torch.stack(
            [
                v.mean(dim=-1),
                v.std(dim=-1, unbiased=False),
                v.min(dim=-1).values,
                v.max(dim=-1).values,
                v.abs().mean(dim=-1),
                rms,
            ],
            dim=-1,
        )

    def _generic_summary_from_flat(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected flattened generic input (B,D). Got {tuple(x.shape)}")
        bsz, dim = x.shape

        # Preferred path: x = [vec(Ga) || vec(Gb)] with Ga,Gb in R^{C x C}.
        if dim % 2 == 0:
            half = dim // 2
            c = int(math.isqrt(half))
            if c * c == half:
                ga = x[:, :half].reshape(bsz, c, c)
                gb = x[:, half:].reshape(bsz, c, c)
                diag_a = ga.diagonal(dim1=-2, dim2=-1)
                diag_b = gb.diagonal(dim1=-2, dim2=-1)

                tr_a = diag_a.sum(dim=-1) / float(max(c, 1))
                tr_b = diag_b.sum(dim=-1) / float(max(c, 1))
                fro_a = torch.sqrt((ga * ga).mean(dim=(-2, -1)).clamp_min(1e-12))
                fro_b = torch.sqrt((gb * gb).mean(dim=(-2, -1)).clamp_min(1e-12))
                sym_a = (ga - ga.transpose(-1, -2)).pow(2).mean(dim=(-2, -1))
                sym_b = (gb - gb.transpose(-1, -2)).pow(2).mean(dim=(-2, -1))
                off_denom = float(max(c * c - c, 1))
                off_abs_a = (ga.abs().sum(dim=(-2, -1)) - diag_a.abs().sum(dim=-1)) / off_denom
                off_abs_b = (gb.abs().sum(dim=(-2, -1)) - diag_b.abs().sum(dim=-1)) / off_denom
                delta_mean_abs = (ga - gb).abs().mean(dim=(-2, -1))
                delta_fro = torch.sqrt(((ga - gb).pow(2)).mean(dim=(-2, -1)).clamp_min(1e-12))

                sv_a = torch.linalg.svdvals(ga)
                sv_b = torch.linalg.svdvals(gb)
                k = min(self.generic_svd_k, sv_a.shape[-1])
                sv_a_k = sv_a[:, :k]
                sv_b_k = sv_b[:, :k]
                if k < self.generic_svd_k:
                    pad = torch.zeros(bsz, self.generic_svd_k - k, device=x.device, dtype=x.dtype)
                    sv_a_k = torch.cat([sv_a_k, pad], dim=-1)
                    sv_b_k = torch.cat([sv_b_k, pad], dim=-1)

                diag_a_stats = torch.stack(
                    [
                        diag_a.mean(dim=-1),
                        diag_a.std(dim=-1, unbiased=False),
                        diag_a.min(dim=-1).values,
                        diag_a.max(dim=-1).values,
                    ],
                    dim=-1,
                )
                diag_b_stats = torch.stack(
                    [
                        diag_b.mean(dim=-1),
                        diag_b.std(dim=-1, unbiased=False),
                        diag_b.min(dim=-1).values,
                        diag_b.max(dim=-1).values,
                    ],
                    dim=-1,
                )
                summary = torch.cat(
                    [
                        tr_a.unsqueeze(-1),
                        tr_b.unsqueeze(-1),
                        diag_a_stats,
                        diag_b_stats,
                        fro_a.unsqueeze(-1),
                        fro_b.unsqueeze(-1),
                        off_abs_a.unsqueeze(-1),
                        off_abs_b.unsqueeze(-1),
                        sym_a.unsqueeze(-1),
                        sym_b.unsqueeze(-1),
                        delta_mean_abs.unsqueeze(-1),
                        delta_fro.unsqueeze(-1),
                        sv_a_k,
                        sv_b_k,
                    ],
                    dim=-1,
                )
                return summary

        # Fallback: compact statistics from split halves.
        half = dim // 2
        xa = x[:, :half] if half > 0 else x
        xb = x[:, half:] if half > 0 else x
        if xa.shape[1] == 0:
            xa = x
        if xb.shape[1] == 0:
            xb = x

        sa = self._stats6(xa)
        sb = self._stats6(xb)
        d = xa[:, : min(xa.shape[1], xb.shape[1])] - xb[:, : min(xa.shape[1], xb.shape[1])]
        sd = self._stats6(d)
        corr_num = (xa[:, : d.shape[1]] * xb[:, : d.shape[1]]).mean(dim=-1)
        corr_den = (
            xa[:, : d.shape[1]].pow(2).mean(dim=-1).sqrt() * xb[:, : d.shape[1]].pow(2).mean(dim=-1).sqrt()
        ).clamp_min(1e-6)
        corr = (corr_num / corr_den).unsqueeze(-1)
        return torch.cat([sa, sb, sd, corr], dim=-1)

    def _rotation_from_cov_with_uncertainty(
        self,
        cov: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Keep rotation/SVD numerics in float32, but feed MLPs in module dtype.
        cov_f = cov.to(dtype=torch.float32)
        cov_flat = cov_f.reshape(cov_f.shape[0], -1)
        svals = torch.linalg.svdvals(cov_f)

        _, module_dtype = self._module_device_dtype()
        stem_in = torch.cat([cov_flat, svals], dim=-1).to(dtype=module_dtype)
        h = self.cov_stem(stem_in)
        delta_omega = self._clip_delta(self.cov_delta(h)).to(dtype=torch.float32)
        logvar = self.cov_logvar(h).to(dtype=torch.float32)

        rot_base = self._procrustes_rotation_from_cov(cov_f)
        rot_delta = self._axis_angle_to_matrix(delta_omega)
        rot_hat = rot_delta @ rot_base
        return rot_hat, logvar, rot_base, delta_omega

    def forward_pair_with_uncertainty(
        self,
        eq_a: torch.Tensor,
        eq_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if eq_a.dim() != 3 or eq_b.dim() != 3 or eq_a.shape != eq_b.shape or eq_a.shape[-1] != 3:
            raise ValueError(
                "Expected eq_a and eq_b with identical shape (B,C,3). "
                f"Got {tuple(eq_a.shape)} and {tuple(eq_b.shape)}"
            )
        module_device, module_dtype = self._module_device_dtype()
        if eq_a.device != module_device or eq_a.dtype != module_dtype:
            eq_a = eq_a.to(device=module_device, dtype=module_dtype)
        if eq_b.device != module_device or eq_b.dtype != module_dtype:
            eq_b = eq_b.to(device=module_device, dtype=module_dtype)

        na = torch.linalg.vector_norm(eq_a, dim=-1)
        nb = torch.linalg.vector_norm(eq_b, dim=-1)
        dot = (eq_a * eq_b).sum(dim=-1)
        cross_n = torch.linalg.vector_norm(torch.cross(eq_a, eq_b, dim=-1), dim=-1)
        pair_feat = torch.stack([na, nb, dot, cross_n], dim=-1)
        w = F.softplus(self.weight_mlp(pair_feat).squeeze(-1)) + 1e-6
        w_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        cov = torch.einsum("bc,bci,bcj->bij", w, eq_b, eq_a) / w_sum.unsqueeze(-1)

        rot_hat, logvar, rot_base, delta_omega = self._rotation_from_cov_with_uncertainty(cov)
        aux = {
            "weights": w,
            "cov": cov,
            "rot_base": rot_base,
            "delta_omega": delta_omega,
        }
        return rot_hat, logvar, aux

    def forward_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3 and x.shape[-2:] == (3, 3):
            rot, logvar, _, _ = self._rotation_from_cov_with_uncertainty(x)
            return self._matrix_to_sixd(rot), logvar

        if x.dim() >= 3:
            x = x.reshape(x.shape[0], -1)
        elif x.dim() != 2:
            raise ValueError(f"Expected input with shape (B,3,3) or (B,D). Got {tuple(x.shape)}")
        if x.shape[-1] != self.generic_in_dim:
            raise ValueError(
                f"PoseRotationHead generic input dim mismatch: expected {self.generic_in_dim}, got {x.shape[-1]}"
            )

        summary = self._generic_summary_from_flat(x)
        _, module_dtype = self._module_device_dtype()
        h = self.generic_stem(summary.to(dtype=module_dtype))
        return self.generic_rot(h).to(dtype=torch.float32), self.generic_logvar(h).to(dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rot6d, _ = self.forward_with_uncertainty(x)
        return rot6d
