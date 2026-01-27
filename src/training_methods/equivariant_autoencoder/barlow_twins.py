import torch
import torch.nn as nn


class BarlowTwinsLoss(nn.Module):
    def __init__(
        self,
        *,
        enabled: bool,
        weight: float,
        lambda_: float,
        embed_dim: int,
        start_epoch: int,
        jitter_std: float,
        drop_ratio: float,
        input_dim,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.weight = float(weight)
        self.lambda_ = float(lambda_)
        self.embed_dim = int(embed_dim)
        self.start_epoch = max(0, int(start_epoch))
        self.jitter_std = float(jitter_std)
        self.drop_ratio = float(drop_ratio)

        self.projector = None
        if input_dim is None:
            if self.enabled and self.weight > 0:
                raise ValueError("Barlow Twins requires latent_size to set projector input dim")
        else:
            self.projector = nn.Sequential(
                nn.Linear(int(input_dim), self.embed_dim, bias=False),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim, bias=False),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            )

    @classmethod
    def from_config(cls, cfg, *, input_dim):
        return cls(
            enabled=bool(getattr(cfg, "barlow_enabled", False)),
            weight=float(getattr(cfg, "barlow_weight", 0.0)),
            lambda_=float(getattr(cfg, "barlow_lambda", 5e-3)),
            embed_dim=int(getattr(cfg, "barlow_embed_dim", 8192)),
            start_epoch=int(getattr(cfg, "barlow_start_epoch", 0)),
            jitter_std=float(getattr(cfg, "barlow_jitter_std", 0.01)),
            drop_ratio=float(getattr(cfg, "barlow_drop_ratio", 0.2)),
            input_dim=input_dim,
        )

    def should_run(self, *, current_epoch: int) -> bool:
        return bool(
            self.enabled
            and self.weight > 0
            and self.projector is not None
            and int(current_epoch) >= self.start_epoch
        )

    def compute_loss(
        self,
        *,
        pc: torch.Tensor,
        encoder,
        prepare_input,
        split_output,
        current_epoch: int,
    ):
        if not self.should_run(current_epoch=current_epoch):
            return None, {}
        y_a = self._augment(pc)
        y_b = self._augment(pc)

        enc_a = encoder(prepare_input(y_a))
        inv_a, eq_a = split_output(enc_a)
        inv_a = self._invariant(inv_a, eq_a)

        enc_b = encoder(prepare_input(y_b))
        inv_b, eq_b = split_output(enc_b)
        inv_b = self._invariant(inv_b, eq_b)

        if inv_a is None or inv_b is None:
            return None, {}

        proj_dtype = next(self.projector.parameters()).dtype
        z_a = self.projector(inv_a.to(dtype=proj_dtype))
        z_b = self.projector(inv_b.to(dtype=proj_dtype))
        loss = self._loss(z_a, z_b)
        metrics = {}
        if not torch.isfinite(loss).item():
            metrics["barlow_nonfinite"] = pc.new_tensor(1.0)
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss, metrics

    def _augment(self, pc: torch.Tensor) -> torch.Tensor:
        x = pc
        if self.jitter_std > 0:
            x = x + torch.randn_like(x) * self.jitter_std
        if self.drop_ratio > 0:
            bsz, num_points, _ = x.shape
            keep = (torch.rand(bsz, num_points, device=x.device) > self.drop_ratio)
            keep[:, 0] = True
            w = keep.float()
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
            idx = torch.multinomial(w, num_samples=num_points, replacement=True)
            x = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
        return x

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        if n != m:
            raise ValueError("Input must be a square matrix")
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        if z_a.dtype != torch.float32:
            z_a = z_a.float()
        if z_b.dtype != torch.float32:
            z_b = z_b.float()
        n, d = z_a.shape
        if n < 2:
            return z_a.new_tensor(0.0)

        z_a_mean = z_a.mean(0)
        z_b_mean = z_b.mean(0)
        z_a_std = z_a.std(0, unbiased=False).clamp_min(1e-4)
        z_b_std = z_b.std(0, unbiased=False).clamp_min(1e-4)
        z_a_norm = (z_a - z_a_mean) / z_a_std
        z_b_norm = (z_b - z_b_mean) / z_b_std

        c = (z_a_norm.T @ z_b_norm) / n
        c_diff = (c - torch.eye(d, device=c.device, dtype=c.dtype)).pow(2)

        off = self._off_diagonal(c_diff)
        off.mul_(self.lambda_)

        loss = torch.diagonal(c_diff).sum() + off.sum()
        return loss

    @staticmethod
    def _invariant(inv_z, eq_z):
        if eq_z is None and inv_z is not None and inv_z.dim() == 3 and inv_z.shape[-1] == 3:
            eq_z = inv_z
            inv_z = None
        if eq_z is not None:
            return eq_z.norm(dim=-1)
        return inv_z
