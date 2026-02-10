import torch
import torch.nn as nn


class TensorProductInvariantHead(nn.Module):
    """
    Builds scalar invariants from equivariant vector channels with a strict
    dimensionality cap.

    Modes:
    - norms: per-channel L2 norms (legacy behavior)
    - tensor_product: norms + grouped 2nd-order (power-spectrum-like) +
      grouped 3rd-order (bispectrum-like) invariants
    - passthrough: prefer inv_z as-is, fallback to norms(eq_z)
    """

    def __init__(
        self,
        channels: int,
        *,
        mode: str = "norms",
        max_factor: float = 4.0,
        groups: int = 0,
        include_third_order: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channels = max(0, int(channels))
        self.mode = self._normalize_mode(mode)
        self.eps = float(eps)

        self.max_dim = self.channels
        if self.channels > 0:
            self.max_dim = max(self.channels, int(float(max_factor) * self.channels))

        self.num_groups = 0
        self.num_second_order = 0
        self.num_third_order = 0
        self.output_dim = self.channels

        self.register_buffer("group_weights", torch.empty(0, 0), persistent=False)
        self.register_buffer("second_order_indices", torch.empty(0, 2, dtype=torch.long), persistent=False)
        self.register_buffer("third_order_indices", torch.empty(0, 3, dtype=torch.long), persistent=False)

        if self.mode == "tensor_product" and self.channels > 0:
            base_dim = self.channels
            budget = max(0, self.max_dim - base_dim)
            max_groups = self.channels if int(groups) <= 1 else min(self.channels, int(groups))
            g, n2, n3 = self._plan_orders(
                budget=budget,
                max_groups=max_groups,
                include_third_order=bool(include_third_order),
            )
            self.num_groups = int(g)
            self.num_second_order = int(n2)
            self.num_third_order = int(n3)
            if self.num_groups > 1 and self.num_second_order > 0:
                self.group_weights = self._build_group_weights(self.channels, self.num_groups)
                self.second_order_indices = self._build_second_order_indices(
                    self.num_groups,
                    self.num_second_order,
                )
                self.third_order_indices = self._build_third_order_indices(
                    self.num_groups,
                    self.num_third_order,
                )
            self.output_dim = self.channels + self.num_second_order + self.num_third_order

        self.output_norm = nn.LayerNorm(self.output_dim) if self.mode == "tensor_product" else nn.Identity()

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        value = str(mode).lower().strip()
        if value in {"tensor", "tp", "power_bispectrum", "tensor_product"}:
            return "tensor_product"
        if value in {"pass", "passthrough", "raw"}:
            return "passthrough"
        return "norms"

    @staticmethod
    def _plan_orders(
        *,
        budget: int,
        max_groups: int,
        include_third_order: bool,
    ) -> tuple[int, int, int]:
        if budget <= 0 or max_groups < 2:
            return 0, 0, 0

        candidates: list[tuple[tuple[int, int, int, int], int, int, int]] = []
        for groups in range(2, max_groups + 1):
            second_order = groups * (groups + 1) // 2
            if second_order > budget:
                break
            third_total = (groups * (groups - 1) * (groups - 2) // 6) if include_third_order else 0
            third_order = min(third_total, budget - second_order)
            used = second_order + third_order
            score = (used, third_order, second_order, groups)
            candidates.append((score, groups, second_order, third_order))

        if not candidates:
            return 0, 0, 0

        if include_third_order:
            with_third = [item for item in candidates if item[3] > 0]
            if with_third:
                candidates = with_third

        best = max(candidates, key=lambda item: item[0])
        _, groups, second_order, third_order = best
        return groups, second_order, third_order

    @staticmethod
    def _build_group_weights(channels: int, groups: int) -> torch.Tensor:
        weights = torch.zeros(groups, channels, dtype=torch.float32)
        for g in range(groups):
            start = (g * channels) // groups
            end = ((g + 1) * channels) // groups
            if end <= start:
                continue
            weights[g, start:end] = 1.0 / float(end - start)
        return weights

    @staticmethod
    def _build_second_order_indices(groups: int, count: int) -> torch.Tensor:
        idx = torch.triu_indices(groups, groups, offset=0).T.contiguous()
        if count < idx.shape[0]:
            idx = idx[:count]
        return idx

    @staticmethod
    def _build_third_order_indices(groups: int, count: int) -> torch.Tensor:
        if count <= 0 or groups < 3:
            return torch.empty((0, 3), dtype=torch.long)
        triples = []
        for i in range(groups - 2):
            for j in range(i + 1, groups - 1):
                for k in range(j + 1, groups):
                    triples.append((i, j, k))
                    if len(triples) >= count:
                        return torch.tensor(triples, dtype=torch.long)
        if not triples:
            return torch.empty((0, 3), dtype=torch.long)
        return torch.tensor(triples, dtype=torch.long)

    def _coerce_eq_latent(self, eq_z: torch.Tensor | None) -> torch.Tensor | None:
        if eq_z is None:
            return None
        if eq_z.dim() == 3:
            if eq_z.shape[-1] == 3:
                return eq_z
            if eq_z.shape[1] == 3:
                return eq_z.transpose(1, 2).contiguous()
            return None
        if eq_z.dim() == 4 and eq_z.shape[-1] == 3:
            if eq_z.shape[1] == self.channels:
                return eq_z.mean(dim=2)
            if eq_z.shape[2] == self.channels:
                return eq_z.mean(dim=1)
            return eq_z.mean(dim=1)
        return None

    def _fit_output_dim(self, feat: torch.Tensor | None) -> torch.Tensor | None:
        if feat is None:
            return None
        if feat.dim() > 2:
            feat = feat.reshape(feat.shape[0], -1)
        if feat.dim() != 2:
            return feat
        if feat.shape[1] == self.output_dim:
            return feat
        if feat.shape[1] > self.output_dim:
            return feat[:, : self.output_dim]
        pad = feat.new_zeros((feat.shape[0], self.output_dim - feat.shape[1]))
        return torch.cat([feat, pad], dim=1)

    def _norms(self, eq_z: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((eq_z * eq_z).sum(dim=-1) + self.eps)

    def _tensor_product_features(self, eq_z: torch.Tensor) -> torch.Tensor:
        norms = self._norms(eq_z)
        parts = [norms]

        if self.num_groups > 1 and eq_z.shape[1] == self.channels and self.num_second_order > 0:
            w = self.group_weights.to(device=eq_z.device, dtype=eq_z.dtype)
            grouped = torch.einsum("gc,bck->bgk", w, eq_z)

            idx2 = self.second_order_indices
            if idx2.numel() > 0:
                gram = torch.einsum("bgi,bhi->bgh", grouped, grouped)
                second = gram[:, idx2[:, 0], idx2[:, 1]]
                parts.append(second)

            idx3 = self.third_order_indices
            if idx3.numel() > 0:
                a = grouped[:, idx3[:, 0], :]
                b = grouped[:, idx3[:, 1], :]
                c = grouped[:, idx3[:, 2], :]
                third = (a * torch.cross(b, c, dim=-1)).sum(dim=-1)
                parts.append(third)

        return torch.cat(parts, dim=-1)

    def forward(self, inv_z: torch.Tensor | None, eq_z: torch.Tensor | None) -> torch.Tensor | None:
        if eq_z is None and inv_z is not None and inv_z.dim() == 3 and inv_z.shape[-1] == 3:
            eq_z = inv_z
            inv_z = None

        if self.mode == "passthrough":
            if inv_z is not None:
                return self._fit_output_dim(inv_z)
            eq = self._coerce_eq_latent(eq_z)
            if eq is None:
                return None
            return self._fit_output_dim(self._norms(eq))

        eq = self._coerce_eq_latent(eq_z)
        if eq is not None:
            if self.mode == "tensor_product":
                feat = self._tensor_product_features(eq)
                feat = self._fit_output_dim(feat)
                return self.output_norm(feat)
            return self._fit_output_dim(self._norms(eq))

        if inv_z is not None:
            return self._fit_output_dim(inv_z)
        return None
