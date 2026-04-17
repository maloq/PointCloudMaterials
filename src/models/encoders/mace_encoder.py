from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .base import Encoder
from .registry import register_encoder

try:
    from e3nn import o3
except ImportError as exc:  # pragma: no cover - exercised only without e3nn installed
    o3 = None
    _E3NN_IMPORT_ERROR: Exception | None = exc
else:
    _E3NN_IMPORT_ERROR = None

try:
    from mace.modules.blocks import (
        EquivariantProductBasisBlock,
        LinearNodeEmbeddingBlock,
        RadialEmbeddingBlock,
        RealAgnosticInteractionBlock,
        RealAgnosticResidualInteractionBlock,
    )
except ImportError as exc:  # pragma: no cover - exercised only without mace-torch installed
    EquivariantProductBasisBlock = None
    LinearNodeEmbeddingBlock = None
    RadialEmbeddingBlock = None
    RealAgnosticInteractionBlock = None
    RealAgnosticResidualInteractionBlock = None
    _MACE_IMPORT_ERROR: Exception | None = exc
else:
    _MACE_IMPORT_ERROR = None


def _require_backends() -> None:
    if _E3NN_IMPORT_ERROR is not None:
        raise ImportError(
            "MACE_Backbone requires the optional dependency 'e3nn'. "
            "Install it with `pip install e3nn` or `pip install -r requirements.txt`."
        ) from _E3NN_IMPORT_ERROR
    if _MACE_IMPORT_ERROR is not None:
        raise ImportError(
            "MACE_Backbone requires the optional dependency 'mace-torch'. "
            "Install it with `pip install mace-torch` or `pip install -r requirements.txt`."
        ) from _MACE_IMPORT_ERROR


def _radius_graph(
    points: torch.Tensor,
    *,
    r_max: float,
    max_neighbors: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a (sender, receiver) radius graph for each point cloud in a batch, and
    flatten batched nodes into a single [B*N, ...] node index space.

    Returns (edge_src, edge_dst, edge_vec, edge_len, batch_index) where
    edge_vec[e] = points_flat[edge_dst[e]] - points_flat[edge_src[e]] so that a
    message from sender `src` to receiver `dst` uses the vector pointing from
    src to dst. This matches the convention used internally by MACE's
    InteractionBlock (it indexes `node_feats[edge_index[0]]` as the sender).
    """
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected points shape [B, N, 3], got {tuple(points.shape)}")
    if float(r_max) <= 0.0:
        raise ValueError(f"r_max must be > 0, got {r_max}")

    batch_size, num_points, _ = points.shape
    if num_points < 2:
        raise ValueError(
            f"MACE_Backbone requires at least 2 points per cloud to build edges, got {num_points}."
        )
    if max_neighbors is not None:
        max_neighbors = int(max_neighbors)
        if max_neighbors <= 0:
            raise ValueError(f"max_neighbors must be > 0, got {max_neighbors}")

    edge_dst_parts: list[torch.Tensor] = []
    edge_src_parts: list[torch.Tensor] = []
    batch_index = torch.arange(batch_size, device=points.device, dtype=torch.long).repeat_interleave(
        num_points
    )

    for batch_idx in range(batch_size):
        pos = points[batch_idx]
        dist = torch.cdist(pos.float(), pos.float(), p=2.0)
        eye = torch.eye(num_points, dtype=torch.bool, device=points.device)
        valid = (dist < float(r_max)) & (~eye)

        if max_neighbors is None:
            dst_local, src_local = torch.nonzero(valid, as_tuple=True)
        else:
            k_eff = min(max_neighbors, num_points - 1)
            masked = dist.masked_fill(~valid, float("inf"))
            values, indices = torch.topk(
                masked, k=k_eff, dim=-1, largest=False, sorted=False
            )
            keep = torch.isfinite(values)
            dst_local = (
                torch.arange(num_points, device=points.device, dtype=torch.long)
                .unsqueeze(1)
                .expand_as(indices)[keep]
            )
            src_local = indices[keep]

        if dst_local.numel() == 0:
            raise ValueError(
                "Radius graph produced zero edges. "
                f"Try increasing r_max (currently {r_max}) or max_neighbors "
                f"(currently {max_neighbors}) for input shape {tuple(points.shape)}."
            )

        offset = batch_idx * num_points
        edge_dst_parts.append(dst_local + offset)
        edge_src_parts.append(src_local + offset)

    edge_dst = torch.cat(edge_dst_parts, dim=0)
    edge_src = torch.cat(edge_src_parts, dim=0)

    points_flat = points.reshape(batch_size * num_points, 3)
    edge_vec = points_flat[edge_dst] - points_flat[edge_src]
    edge_len = torch.linalg.norm(edge_vec, dim=-1, keepdim=True)
    return edge_src, edge_dst, edge_vec, edge_len, batch_index


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.dim() != 2:
        raise ValueError(f"Expected src with shape [N, F], got {tuple(src.shape)}")
    if index.dim() != 1:
        raise ValueError(f"Expected index with shape [N], got {tuple(index.shape)}")
    out = src.new_zeros((int(dim_size), src.shape[1]))
    out.index_add_(0, index, src)
    return out


def _batch_mean_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    pooled = _scatter_sum(x, batch, dim_size=num_graphs)
    counts = torch.bincount(batch, minlength=num_graphs).to(dtype=x.dtype).unsqueeze(-1)
    counts = counts.clamp_min(1.0)
    return pooled / counts


def _build_hidden_irreps(
    *,
    num_features: int,
    max_ell: int,
    include_parity_odd_scalars: bool,
):
    """
    Build MACE hidden irreps `Nx0e + Nx1o + Nx2e + ...` up to `max_ell`.

    `include_parity_odd_scalars` controls whether we additionally allow
    `Nx0o + Nx1e + ...` terms. For plain point clouds (no pseudoscalars)
    the standard MACE choice `False` is what the reference foundation
    models use.
    """
    _require_backends()
    if int(num_features) <= 0:
        raise ValueError(f"num_features must be > 0, got {num_features}")
    if int(max_ell) < 0:
        raise ValueError(f"max_ell must be >= 0, got {max_ell}")
    parts = []
    for l in range(int(max_ell) + 1):
        p = 1 if l % 2 == 0 else -1
        parts.append((int(num_features), (l, p)))
        if include_parity_odd_scalars:
            parts.append((int(num_features), (l, -p)))
    return o3.Irreps(parts)


@register_encoder("MACE_Backbone")
class MACEBackboneEncoder(Encoder):
    """
    MACE encoder adapted to this repository's graph-level latent contract.

    This uses the building blocks from the reference `mace-torch` package
    (https://github.com/ACEsuit/mace):

    - `LinearNodeEmbeddingBlock` for the initial invariant node features,
    - `RadialEmbeddingBlock` with Bessel basis + polynomial cutoff,
    - `o3.SphericalHarmonics` for edge angular attributes,
    - `RealAgnosticInteractionBlock` + `EquivariantProductBasisBlock` for the
      first MACE layer (two-body → many-body via symmetric contraction),
    - `RealAgnosticResidualInteractionBlock` + `EquivariantProductBasisBlock`
      for the subsequent layers.

    The last layer keeps both `0e` scalars and `1o` vectors so we can emit a
    graph-level invariant latent `inv_z` (shape `[B, latent_size]`) and a
    graph-level equivariant latent `eq_z` (shape `[B, latent_size, 3]`), plus
    the cloud centroid `center` (shape `[B, 3]`), matching the encoder
    contract used across this repo.

    Single-type point clouds only, because the encoder interface here
    passes coordinates only (no per-point atom types).
    """

    def __init__(
        self,
        latent_size: int = 128,
        *,
        r_max: float = 2.0,
        num_interactions: int = 2,
        max_ell: int = 2,
        num_features: int = 32,
        correlation: int | Sequence[int] = 3,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 6,
        radial_type: str = "bessel",
        radial_MLP: Sequence[int] | None = None,
        avg_num_neighbors: float | None = None,
        max_neighbors: int | None = None,
        num_types: int = 1,
        pool: str = "mean",
        include_parity_odd_scalars: bool = False,
    ) -> None:
        super().__init__()
        _require_backends()

        if int(latent_size) <= 0:
            raise ValueError(f"latent_size must be > 0, got {latent_size}")
        if float(r_max) <= 0.0:
            raise ValueError(f"r_max must be > 0, got {r_max}")
        if int(num_interactions) < 2:
            raise ValueError(
                "MACE_Backbone requires num_interactions >= 2 so that the last layer "
                "can be a residual MACE layer that retains both scalars and vectors. "
                f"Got num_interactions={num_interactions}."
            )
        if int(max_ell) < 1:
            raise ValueError(
                f"max_ell must be >= 1 to produce vector (l=1) latents, got {max_ell}."
            )
        if int(num_types) != 1:
            raise ValueError(
                "MACE_Backbone currently supports num_types=1 only because the encoder "
                "API in this repo passes coordinates only, not per-point atom types."
            )

        pool_mode = str(pool).lower().strip()
        if pool_mode not in {"mean", "sum"}:
            raise ValueError(f"Unsupported pool={pool!r}; expected 'mean' or 'sum'.")

        if isinstance(correlation, int):
            correlation_list = [int(correlation)] * int(num_interactions)
        else:
            correlation_list = [int(v) for v in correlation]
            if len(correlation_list) != int(num_interactions):
                raise ValueError(
                    "correlation must be an int or a list of length num_interactions "
                    f"({num_interactions}), got {correlation_list}."
                )
        if any(c < 1 for c in correlation_list):
            raise ValueError(f"correlation entries must be >= 1, got {correlation_list}.")

        if avg_num_neighbors is None:
            if max_neighbors is not None:
                avg_num_neighbors = float(max_neighbors)
            else:
                avg_num_neighbors = 1.0
        if float(avg_num_neighbors) <= 0.0:
            raise ValueError(f"avg_num_neighbors must be > 0, got {avg_num_neighbors}")

        self.latent_size = int(latent_size)
        self.invariant_dim = self.latent_size
        self.equivariant_dim = self.latent_size
        self.r_max = float(r_max)
        self.num_interactions = int(num_interactions)
        self.max_ell = int(max_ell)
        self.num_features = int(num_features)
        self.correlation_list = correlation_list
        self.pool = pool_mode
        self.max_neighbors = None if max_neighbors is None else int(max_neighbors)
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.num_types = int(num_types)

        node_attrs_irreps = o3.Irreps(f"{self.num_types}x0e")
        node_feats_irreps = o3.Irreps(f"{self.num_features}x0e")
        sh_irreps = o3.Irreps.spherical_harmonics(self.max_ell)
        interaction_irreps = (sh_irreps * self.num_features).sort()[0].simplify()
        hidden_irreps = _build_hidden_irreps(
            num_features=self.num_features,
            max_ell=self.max_ell,
            include_parity_odd_scalars=bool(include_parity_odd_scalars),
        )

        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.sh_irreps = sh_irreps
        self.interaction_irreps = interaction_irreps
        self.hidden_irreps = hidden_irreps

        if not any(ir == o3.Irrep("0e") for _, ir in hidden_irreps):
            raise ValueError(
                "Final MACE hidden irreps do not contain 0e scalars, so inv_z cannot be produced. "
                f"hidden_irreps={hidden_irreps}."
            )
        if not any(ir == o3.Irrep("1o") for _, ir in hidden_irreps):
            raise ValueError(
                "Final MACE hidden irreps do not contain 1o vectors, so eq_z cannot be produced. "
                f"hidden_irreps={hidden_irreps}."
            )

        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attrs_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=self.r_max,
            num_bessel=int(num_bessel),
            num_polynomial_cutoff=int(num_polynomial_cutoff),
            radial_type=str(radial_type),
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        radial_mlp = [64, 64, 64] if radial_MLP is None else [int(v) for v in radial_MLP]

        interactions: list[nn.Module] = []
        products: list[nn.Module] = []

        # First layer: no skip connection on inputs (single-type scalars only).
        inter_first = RealAgnosticInteractionBlock(
            node_attrs_irreps=node_attrs_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            radial_MLP=radial_mlp,
        )
        prod_first = EquivariantProductBasisBlock(
            node_feats_irreps=inter_first.target_irreps,
            target_irreps=hidden_irreps,
            correlation=correlation_list[0],
            num_elements=self.num_types,
            use_sc=False,
        )
        interactions.append(inter_first)
        products.append(prod_first)

        for layer_idx in range(1, self.num_interactions):
            inter = RealAgnosticResidualInteractionBlock(
                node_attrs_irreps=node_attrs_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=self.avg_num_neighbors,
                radial_MLP=radial_mlp,
            )
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps,
                correlation=correlation_list[layer_idx],
                num_elements=self.num_types,
                use_sc=True,
            )
            interactions.append(inter)
            products.append(prod)

        self.interactions = nn.ModuleList(interactions)
        self.products = nn.ModuleList(products)

        # Heads: independently project the scalar (0e) and vector (1o) parts
        # of the final hidden irreps to the requested latent size.
        self._scalar_slice = self._irreps_slice_for(hidden_irreps, o3.Irrep("0e"))
        self._vector_slice = self._irreps_slice_for(hidden_irreps, o3.Irrep("1o"))
        scalar_mul = hidden_irreps.count(o3.Irrep("0e"))
        vector_mul = hidden_irreps.count(o3.Irrep("1o"))

        self.inv_head = nn.Sequential(
            nn.Linear(scalar_mul, self.latent_size),
            nn.SiLU(),
            nn.Linear(self.latent_size, self.latent_size),
        )
        # For the equivariant head we only apply a scalar linear mixing over the
        # channel axis so the resulting tensor remains an equivariant vector.
        self.eq_head = nn.Linear(vector_mul, self.latent_size, bias=False)

    @staticmethod
    def _irreps_slice_for(irreps, target_irrep) -> tuple[int, int]:
        """Return the (start, stop) slice in the flat irreps layout for the
        contiguous block that matches `target_irrep`. Raises if no or multiple
        blocks match, which should never happen for the simplified hidden
        irreps we construct internally."""
        start = 0
        matches: list[tuple[int, int]] = []
        for mul, ir in irreps:
            block_dim = mul * ir.dim
            if ir == target_irrep:
                matches.append((start, start + block_dim))
            start += block_dim
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one contiguous block of {target_irrep} in "
                f"irreps={irreps}, found {len(matches)}."
            )
        return matches[0]

    def _pool(self, x: torch.Tensor, batch: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.pool == "sum":
            return _scatter_sum(x, batch, dim_size=batch_size)
        return _batch_mean_pool(x, batch, num_graphs=batch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected input shape [B, N, 3], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError(
                f"Expected floating-point input for MACE_Backbone, got dtype={x.dtype}."
            )

        batch_size, num_points, _ = x.shape
        if num_points < 2:
            raise ValueError(
                f"MACE_Backbone requires at least 2 points per cloud, got {num_points}."
            )

        center = x.mean(dim=1)
        compute_dtype = self.node_embedding.linear.weight.dtype
        pos = (x - center.unsqueeze(1)).to(dtype=compute_dtype)

        edge_src, edge_dst, edge_vec, edge_len, batch = _radius_graph(
            pos, r_max=self.r_max, max_neighbors=self.max_neighbors
        )
        edge_index = torch.stack([edge_src, edge_dst], dim=0)

        node_count = batch_size * num_points
        atom_types = torch.zeros(node_count, dtype=torch.long, device=x.device)
        node_attrs = torch.zeros(
            (node_count, self.num_types), dtype=compute_dtype, device=x.device
        )
        node_attrs[torch.arange(node_count, device=x.device), atom_types] = 1.0

        node_feats = self.node_embedding(node_attrs)
        edge_attrs = self.spherical_harmonics(edge_vec)
        # RadialEmbeddingBlock expects (edge_lengths, node_attrs, edge_index, atomic_numbers).
        # The last two are only used when a distance_transform is configured
        # (not the case here), so we pass the node_attrs/edge_index we have
        # and None for atomic_numbers.
        edge_feats, _ = self.radial_embedding(edge_len, node_attrs, edge_index, None)

        for inter, prod in zip(self.interactions, self.products):
            node_feats, sc = inter(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
            node_feats = prod(node_feats=node_feats, sc=sc, node_attrs=node_attrs)

        # node_feats is now flat [N, hidden_irreps.dim]. Extract the 0e and 1o
        # blocks for the invariant / equivariant heads.
        s0, s1 = self._scalar_slice
        v0, v1 = self._vector_slice
        scalar_feats = node_feats[:, s0:s1]
        vector_feats = node_feats[:, v0:v1].view(node_count, -1, 3)

        pooled_scalars = self._pool(scalar_feats, batch, batch_size)
        pooled_vectors = self._pool(
            vector_feats.reshape(node_count, -1), batch, batch_size
        ).view(batch_size, -1, 3)

        inv_z = self.inv_head(pooled_scalars)
        # Channel-mix the equivariant vectors with a single scalar linear;
        # this preserves the O(3) equivariance of the 1o block.
        eq_z = self.eq_head(pooled_vectors.transpose(-1, -2)).transpose(-1, -2).contiguous()

        expected_inv_shape = (batch_size, self.latent_size)
        if tuple(inv_z.shape) != expected_inv_shape:
            raise RuntimeError(
                f"Unexpected inv_z shape {tuple(inv_z.shape)}, expected {expected_inv_shape}."
            )
        expected_eq_shape = (batch_size, self.latent_size, 3)
        if tuple(eq_z.shape) != expected_eq_shape:
            raise RuntimeError(
                f"Unexpected eq_z shape {tuple(eq_z.shape)}, expected {expected_eq_shape}."
            )

        return inv_z, eq_z, center
