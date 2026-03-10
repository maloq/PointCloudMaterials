from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from ..base import Encoder
from ..registry import register_encoder

try:
    from e3nn import o3
    from e3nn.nn import Gate
except ImportError as exc:  # pragma: no cover - exercised only without e3nn installed
    o3 = None
    Gate = None
    _E3NN_IMPORT_ERROR = exc
else:
    _E3NN_IMPORT_ERROR = None


def _require_e3nn() -> None:
    if _E3NN_IMPORT_ERROR is not None:
        raise ImportError(
            "NequIP_Backbone requires the optional dependency 'e3nn'. "
            "Install it with `pip install e3nn` or `pip install -r requirements.txt`."
        ) from _E3NN_IMPORT_ERROR


def _shifted_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x) - math.log(2.0)


_ACTS = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "ssp": _shifted_softplus,
    "silu": torch.nn.functional.silu,
}


def _resolve_activation(name: str):
    key = str(name).lower().strip()
    if key not in _ACTS:
        raise ValueError(
            f"Unsupported activation {name!r}. Expected one of {sorted(_ACTS)}."
        )
    return _ACTS[key]


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.dim() != 2:
        raise ValueError(f"Expected src with shape [E, F], got {tuple(src.shape)}")
    if index.dim() != 1:
        raise ValueError(f"Expected index with shape [E], got {tuple(index.shape)}")
    if src.shape[0] != index.shape[0]:
        raise ValueError(
            f"Edge/message mismatch: src has {src.shape[0]} rows, index has {index.shape[0]}."
        )
    out = src.new_zeros((int(dim_size), src.shape[1]))
    out.index_add_(0, index, src)
    return out


def _batch_mean_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    pooled = _scatter_sum(x, batch, dim_size=num_graphs)
    counts = torch.bincount(batch, minlength=num_graphs).to(dtype=x.dtype).unsqueeze(-1)
    counts = counts.clamp_min(1.0)
    return pooled / counts


def _tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    _require_e3nn()
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def _build_feature_irreps(
    *,
    l_max: int,
    parity: bool,
    num_features: int | Sequence[int],
):
    _require_e3nn()
    if isinstance(num_features, int):
        feature_counts = [int(num_features)] * (int(l_max) + 1)
    else:
        feature_counts = [int(v) for v in num_features]
    if len(feature_counts) != int(l_max) + 1:
        raise ValueError(
            f"num_features must have length l_max + 1 ({int(l_max) + 1}), "
            f"got {feature_counts}."
        )

    irreps = []
    for l, mul in enumerate(feature_counts):
        if mul <= 0:
            raise ValueError(f"num_features entries must be > 0, got {feature_counts}.")
        parities = (1, -1) if parity else ((1,) if l % 2 == 0 else (-1,))
        for p in parities:
            irreps.append((mul, (l, p)))
    return o3.Irreps(irreps)


def _resolve_hidden_irreps(
    *,
    num_layers: int,
    feature_irreps_hidden,
    l_max: int,
    parity: bool,
    num_features: int | Sequence[int],
):
    _require_e3nn()
    if feature_irreps_hidden is None:
        base = _build_feature_irreps(l_max=l_max, parity=parity, num_features=num_features)
        return [base for _ in range(int(num_layers))]

    if isinstance(feature_irreps_hidden, str):
        base = o3.Irreps(feature_irreps_hidden)
        return [base for _ in range(int(num_layers))]

    if hasattr(feature_irreps_hidden, "dim") and hasattr(feature_irreps_hidden, "simplify"):
        base = o3.Irreps(feature_irreps_hidden)
        return [base for _ in range(int(num_layers))]

    if not isinstance(feature_irreps_hidden, Sequence):
        raise TypeError(
            "feature_irreps_hidden must be None, an Irreps/string, or a sequence "
            f"of Irreps/string entries, got {type(feature_irreps_hidden)}."
        )

    hidden_list = [o3.Irreps(ir) for ir in feature_irreps_hidden]
    if len(hidden_list) != int(num_layers):
        raise ValueError(
            f"feature_irreps_hidden must have one entry per layer ({num_layers}), "
            f"got {len(hidden_list)}."
        )
    return hidden_list


def _scalar_irreps(mul: int):
    _require_e3nn()
    if int(mul) <= 0:
        raise ValueError(f"Scalar irrep multiplicity must be > 0, got {mul}")
    return o3.Irreps([(int(mul), (0, 1))])


def _radius_graph(
    points: torch.Tensor,
    *,
    r_max: float,
    max_neighbors: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected points shape [B, N, 3], got {tuple(points.shape)}")
    if float(r_max) <= 0.0:
        raise ValueError(f"r_max must be > 0, got {r_max}")

    batch_size, num_points, _ = points.shape
    if num_points < 2:
        raise ValueError(
            f"NequIP_Backbone requires at least 2 points per cloud to build edges, got {num_points}."
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
                masked,
                k=k_eff,
                dim=-1,
                largest=False,
                sorted=False,
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
    edge_vec = points_flat[edge_src] - points_flat[edge_dst]
    edge_len = torch.linalg.norm(edge_vec, dim=-1, keepdim=True)
    return edge_dst, edge_src, edge_vec, edge_len, batch_index


class PolynomialCutoff(nn.Module):
    def __init__(self, p: float = 6.0):
        super().__init__()
        if float(p) < 2.0:
            raise ValueError(f"Polynomial cutoff requires p >= 2, got {p}")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p
        out = 1.0
        out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
        out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
        out = out - ((p * (p + 1.0) / 2.0) * torch.pow(x, p + 2.0))
        return out * (x < 1.0)


class BesselEdgeLengthEncoding(nn.Module):
    def __init__(
        self,
        *,
        r_max: float,
        num_bessels: int = 8,
        trainable: bool = False,
        cutoff: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if float(r_max) <= 0.0:
            raise ValueError(f"r_max must be > 0, got {r_max}")
        if int(num_bessels) <= 0:
            raise ValueError(f"num_bessels must be > 0, got {num_bessels}")

        self.r_max = float(r_max)
        self.num_bessels = int(num_bessels)
        self.cutoff = cutoff if cutoff is not None else PolynomialCutoff(6.0)
        self.factor = (2.0 * math.pi) / (self.r_max * self.r_max)

        weights = torch.linspace(
            1.0,
            float(self.num_bessels),
            steps=self.num_bessels,
            dtype=torch.get_default_dtype(),
        ).unsqueeze(0)
        if bool(trainable):
            self.bessel_weights = nn.Parameter(weights)
        else:
            self.register_buffer("bessel_weights", weights, persistent=False)

    def forward(self, distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if distances.dim() != 2 or distances.shape[1] != 1:
            raise ValueError(
                f"Expected distances with shape [E, 1], got {tuple(distances.shape)}"
            )
        x = distances.to(dtype=torch.get_default_dtype()) / self.r_max
        bessel = torch.sinc(x * self.bessel_weights) * self.bessel_weights
        cutoff = self.cutoff(x)
        return (bessel * cutoff) * self.factor, cutoff


class ScalarMLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_layers_depth: int = 1,
        hidden_layers_width: int = 128,
        activation: str = "silu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0 or int(output_dim) <= 0:
            raise ValueError(
                f"input_dim/output_dim must be > 0, got {input_dim}/{output_dim}"
            )
        depth = int(hidden_layers_depth)
        width = int(hidden_layers_width)
        if depth < 0:
            raise ValueError(f"hidden_layers_depth must be >= 0, got {hidden_layers_depth}")
        if depth > 0 and width <= 0:
            raise ValueError(
                f"hidden_layers_width must be > 0 when hidden_layers_depth > 0, got {hidden_layers_width}"
            )

        layers: list[nn.Module] = []
        dims = [int(input_dim)] + ([width] * depth) + [int(output_dim)]
        act = nn.SiLU if activation == "silu" else None
        if act is None:
            raise ValueError(
                f"Unsupported radial activation {activation!r}; expected 'silu'."
            )
        for layer_idx, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim, bias=bool(bias)))
            if layer_idx != len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NequIPConvolution(nn.Module):
    def __init__(
        self,
        *,
        feature_irreps_in,
        node_attr_irreps,
        edge_attr_irreps,
        edge_embedding_dim: int,
        irreps_out,
        radial_mlp_depth: int,
        radial_mlp_width: int,
        use_sc: bool,
        avg_num_neighbors: float,
    ) -> None:
        super().__init__()
        _require_e3nn()

        self.feature_irreps_in = o3.Irreps(feature_irreps_in)
        self.node_attr_irreps = o3.Irreps(node_attr_irreps)
        self.edge_attr_irreps = o3.Irreps(edge_attr_irreps)
        self.irreps_out = o3.Irreps(irreps_out)

        if float(avg_num_neighbors) <= 0.0:
            raise ValueError(f"avg_num_neighbors must be > 0, got {avg_num_neighbors}")
        self.avg_num_neighbors = float(avg_num_neighbors)

        self.linear_1 = o3.Linear(self.feature_irreps_in, self.feature_irreps_in)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.feature_irreps_in):
            for j, (_, ir_edge) in enumerate(self.edge_attr_irreps):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        idx = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, idx, "uvu", True))

        if not irreps_mid:
            raise ValueError(
                "No valid tensor-product paths were found for the requested NequIP layer. "
                f"feature_irreps_in={self.feature_irreps_in}, "
                f"edge_attr_irreps={self.edge_attr_irreps}, irreps_out={self.irreps_out}."
            )

        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, permutation, _ = irreps_mid.sort()
        instructions = [
            (i_in1, i_in2, permutation[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        self.tp = o3.TensorProduct(
            self.feature_irreps_in,
            self.edge_attr_irreps,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.edge_mlp = ScalarMLP(
            input_dim=int(edge_embedding_dim),
            output_dim=int(self.tp.weight_numel),
            hidden_layers_depth=int(radial_mlp_depth),
            hidden_layers_width=int(radial_mlp_width),
            activation="silu",
            bias=False,
        )
        self.linear_2 = o3.Linear(irreps_mid.simplify(), self.irreps_out)
        self.sc = (
            o3.FullyConnectedTensorProduct(
                self.feature_irreps_in,
                self.node_attr_irreps,
                self.irreps_out,
            )
            if bool(use_sc)
            else None
        )

    def forward(
        self,
        node_features: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if node_features.dim() != 2:
            raise ValueError(
                f"Expected node_features with shape [num_nodes, F], got {tuple(node_features.shape)}"
            )

        x = self.linear_1(node_features)
        sc = self.sc(node_features, node_attrs) if self.sc is not None else None

        weights = self.edge_mlp(edge_embedding)
        edge_features = self.tp(x[edge_src], edge_attr, weights)
        out = _scatter_sum(edge_features, edge_dst, dim_size=x.shape[0])
        out = out * (1.0 / math.sqrt(self.avg_num_neighbors))
        out = self.linear_2(out)

        if sc is not None:
            out = out + sc
        return out


class NequIPConvNetLayer(nn.Module):
    def __init__(
        self,
        *,
        feature_irreps_in,
        feature_irreps_hidden,
        edge_attr_irreps,
        node_attr_irreps,
        edge_embedding_dim: int,
        radial_mlp_depth: int,
        radial_mlp_width: int,
        avg_num_neighbors: float,
        use_sc: bool,
        resnet: bool,
        nonlinearity_scalars: dict[str, str] | None = None,
        nonlinearity_gates: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        _require_e3nn()

        nonlinearity_scalars = (
            {"e": "silu", "o": "tanh"}
            if nonlinearity_scalars is None
            else dict(nonlinearity_scalars)
        )
        nonlinearity_gates = (
            {"e": "silu", "o": "tanh"}
            if nonlinearity_gates is None
            else dict(nonlinearity_gates)
        )
        if set(nonlinearity_scalars) != {"e", "o"}:
            raise ValueError(
                "nonlinearity_scalars must provide exactly the keys {'e', 'o'}."
            )
        if set(nonlinearity_gates) != {"e", "o"}:
            raise ValueError(
                "nonlinearity_gates must provide exactly the keys {'e', 'o'}."
            )

        self.feature_irreps_in = o3.Irreps(feature_irreps_in)
        self.feature_irreps_hidden = o3.Irreps(feature_irreps_hidden)
        self.edge_attr_irreps = o3.Irreps(edge_attr_irreps)
        self.node_attr_irreps = o3.Irreps(node_attr_irreps)

        irreps_scalars = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l == 0 and _tp_path_exists(self.feature_irreps_in, self.edge_attr_irreps, ir)
            ]
        )
        irreps_gated = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l > 0 and _tp_path_exists(self.feature_irreps_in, self.edge_attr_irreps, ir)
            ]
        )

        if len(irreps_scalars) == 0 and len(irreps_gated) == 0:
            raise ValueError(
                "No valid hidden irreps survive tensor-product path filtering. "
                f"feature_irreps_in={self.feature_irreps_in}, "
                f"feature_irreps_hidden={self.feature_irreps_hidden}, "
                f"edge_attr_irreps={self.edge_attr_irreps}."
            )

        gate_irrep = "0e" if _tp_path_exists(self.feature_irreps_in, self.edge_attr_irreps, "0e") else "0o"
        irreps_gates = o3.Irreps([(mul, gate_irrep) for mul, _ in irreps_gated])

        self.equivariant_nonlin = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[_resolve_activation(nonlinearity_scalars["e" if ir.p == 1 else "o"]) for _, ir in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[_resolve_activation(nonlinearity_gates["e" if ir.p == 1 else "o"]) for _, ir in irreps_gates],
            irreps_gated=irreps_gated,
        )
        conv_irreps_out = self.equivariant_nonlin.irreps_in.simplify()

        self.conv = NequIPConvolution(
            feature_irreps_in=self.feature_irreps_in,
            node_attr_irreps=self.node_attr_irreps,
            edge_attr_irreps=self.edge_attr_irreps,
            edge_embedding_dim=int(edge_embedding_dim),
            irreps_out=conv_irreps_out,
            radial_mlp_depth=int(radial_mlp_depth),
            radial_mlp_width=int(radial_mlp_width),
            use_sc=bool(use_sc),
            avg_num_neighbors=float(avg_num_neighbors),
        )

        self.irreps_out = self.equivariant_nonlin.irreps_out
        self.resnet = bool(resnet and self.irreps_out == self.feature_irreps_in)

    def forward(
        self,
        node_features: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_embedding: torch.Tensor,
    ) -> torch.Tensor:
        old_features = node_features
        node_features = self.conv(
            node_features=node_features,
            node_attrs=node_attrs,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_attr,
            edge_embedding=edge_embedding,
        )
        node_features = self.equivariant_nonlin(node_features)
        if self.resnet:
            node_features = node_features + old_features
        return node_features


@register_encoder("NequIP_Backbone")
class NequIPBackboneEncoder(Encoder):
    """
    NequIP encoder adapted to this repository's graph-level latent contract.

    The message-passing core follows the original NequIP design: node-type
    embeddings, spherical-harmonic edge attributes, Bessel radial basis with a
    polynomial cutoff, tensor-product interaction blocks, and gate
    nonlinearities. The adaptation here is the readout: pooled node features are
    projected to a graph-level invariant latent `inv_z` and equivariant latent
    `eq_z` so the encoder fits this repository's `(inv_z, eq_z, center)` API.

    This encoder currently supports single-type point clouds only because the
    repo's encoder interface provides coordinates but not per-point atom types.
    """

    def __init__(
        self,
        latent_size: int = 128,
        *,
        r_max: float = 2.0,
        num_layers: int = 4,
        l_max: int = 1,
        parity: bool = True,
        num_features: int | Sequence[int] = 32,
        type_embed_num_features: int | None = None,
        radial_mlp_depth: int = 1,
        radial_mlp_width: int = 128,
        feature_irreps_hidden=None,
        num_bessels: int = 8,
        bessel_trainable: bool = False,
        polynomial_cutoff_p: int = 6,
        avg_num_neighbors: float | None = None,
        max_neighbors: int | None = None,
        pool: str = "mean",
        num_types: int = 1,
        resnet: bool = False,
        convnet_sc: bool = True,
        final_scalar_layer: bool = True,
        nonlinearity_scalars: dict[str, str] | None = None,
        nonlinearity_gates: dict[str, str] | None = None,
        # Backwards-compatible aliases / traps for the previous simplified encoder.
        cutoff: float | None = None,
        num_neighbors: int | None = None,
        radial_basis: int | None = None,
        pooling: str | None = None,
        scalar_dim: int | None = None,
        vector_dim: int | None = None,
        hidden_dim: int | None = None,
        dropout_rate: float | None = None,
        invariant_eps: float | None = None,
    ) -> None:
        super().__init__()
        _require_e3nn()

        if scalar_dim is not None or vector_dim is not None:
            raise ValueError(
                "scalar_dim/vector_dim belong to the previous simplified encoder and are not "
                "part of faithful NequIP. Use l_max, parity, and num_features instead."
            )
        if hidden_dim is not None:
            raise ValueError(
                "hidden_dim belongs to the previous simplified encoder and is ambiguous here. "
                "Use radial_mlp_width and num_features instead."
            )
        if dropout_rate is not None:
            raise ValueError(
                "dropout_rate is not part of the original NequIP interaction blocks implemented here."
            )
        if invariant_eps is not None:
            raise ValueError(
                "invariant_eps belongs to the previous simplified encoder and is not used here."
            )

        if cutoff is not None:
            r_max = float(cutoff)
        if num_neighbors is not None:
            max_neighbors = int(num_neighbors)
        if radial_basis is not None:
            num_bessels = int(radial_basis)
        if pooling is not None:
            pool = str(pooling)

        if int(latent_size) <= 0:
            raise ValueError(f"latent_size must be > 0, got {latent_size}")
        if float(r_max) <= 0.0:
            raise ValueError(f"r_max must be > 0, got {r_max}")
        if int(num_layers) <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if int(l_max) < 1:
            raise ValueError(
                f"l_max must be >= 1 to produce vector latents, got {l_max}."
            )
        if int(num_types) != 1:
            raise ValueError(
                "NequIP_Backbone currently supports num_types=1 only because the encoder "
                "API in this repo passes coordinates only, not per-point atom types."
            )
        if type_embed_num_features is not None and int(type_embed_num_features) <= 0:
            raise ValueError(
                f"type_embed_num_features must be > 0 when set, got {type_embed_num_features}"
            )

        pool = str(pool).lower().strip()
        if pool not in {"mean", "sum"}:
            raise ValueError(f"Unsupported pool={pool!r}; expected 'mean' or 'sum'.")

        self.latent_size = int(latent_size)
        self.r_max = float(r_max)
        self.num_layers = int(num_layers)
        self.l_max = int(l_max)
        self.parity = bool(parity)
        self.pool = pool
        self.max_neighbors = None if max_neighbors is None else int(max_neighbors)
        self.final_scalar_layer = bool(final_scalar_layer)

        hidden_feature_counts = (
            [int(v) for v in num_features]
            if isinstance(num_features, Sequence) and not isinstance(num_features, (str, bytes))
            else num_features
        )
        if type_embed_num_features is None:
            if isinstance(hidden_feature_counts, int):
                type_embed_num_features = int(hidden_feature_counts)
            else:
                type_embed_num_features = int(hidden_feature_counts[0])
        self.type_embed_num_features = int(type_embed_num_features)

        if avg_num_neighbors is None:
            if self.max_neighbors is not None:
                avg_num_neighbors = float(self.max_neighbors)
            else:
                avg_num_neighbors = 1.0
        if float(avg_num_neighbors) <= 0.0:
            raise ValueError(
                f"avg_num_neighbors must be > 0, got {avg_num_neighbors}"
            )
        self.avg_num_neighbors = float(avg_num_neighbors)

        scalar_readout_mul = (
            int(hidden_feature_counts)
            if isinstance(hidden_feature_counts, int)
            else int(hidden_feature_counts[0])
        )
        if self.final_scalar_layer:
            if self.num_layers < 2:
                raise ValueError(
                    "NequIP_Backbone with final_scalar_layer=True requires num_layers >= 2 "
                    "so the equivariant latent can be read from a penultimate equivariant layer."
                )
            if feature_irreps_hidden is None:
                full_hidden = _build_feature_irreps(
                    l_max=self.l_max,
                    parity=self.parity,
                    num_features=num_features,
                )
                hidden_irreps_list = [full_hidden for _ in range(self.num_layers - 1)]
                hidden_irreps_list.append(_scalar_irreps(scalar_readout_mul))
            elif isinstance(feature_irreps_hidden, Sequence) and not isinstance(
                feature_irreps_hidden, (str, bytes)
            ):
                hidden_irreps_list = _resolve_hidden_irreps(
                    num_layers=self.num_layers,
                    feature_irreps_hidden=feature_irreps_hidden,
                    l_max=self.l_max,
                    parity=self.parity,
                    num_features=num_features,
                )
            else:
                full_hidden = o3.Irreps(feature_irreps_hidden)
                hidden_irreps_list = [full_hidden for _ in range(self.num_layers - 1)]
                hidden_irreps_list.append(_scalar_irreps(scalar_readout_mul))
        else:
            hidden_irreps_list = _resolve_hidden_irreps(
                num_layers=self.num_layers,
                feature_irreps_hidden=feature_irreps_hidden,
                l_max=self.l_max,
                parity=self.parity,
                num_features=num_features,
            )

        self.node_attr_irreps = o3.Irreps([(self.type_embed_num_features, (0, 1))])
        self.edge_attr_irreps = o3.Irreps.spherical_harmonics(lmax=self.l_max)
        self.type_embedding = nn.Embedding(
            num_embeddings=int(num_types),
            embedding_dim=self.type_embed_num_features,
        )
        self.edge_sh = o3.SphericalHarmonics(
            self.edge_attr_irreps,
            normalize=True,
            normalization="component",
        )
        self.edge_length_encoding = BesselEdgeLengthEncoding(
            r_max=self.r_max,
            num_bessels=int(num_bessels),
            trainable=bool(bessel_trainable),
            cutoff=PolynomialCutoff(float(polynomial_cutoff_p)),
        )

        self.layers = nn.ModuleList()
        current_irreps = self.node_attr_irreps
        self.eq_source_layer_index = self.num_layers - 2 if self.final_scalar_layer else self.num_layers - 1
        self.eq_feature_irreps = None
        for layer_idx, hidden_irreps in enumerate(hidden_irreps_list):
            layer = NequIPConvNetLayer(
                feature_irreps_in=current_irreps,
                feature_irreps_hidden=hidden_irreps,
                edge_attr_irreps=self.edge_attr_irreps,
                node_attr_irreps=self.node_attr_irreps,
                edge_embedding_dim=int(num_bessels),
                radial_mlp_depth=int(radial_mlp_depth),
                radial_mlp_width=int(radial_mlp_width),
                avg_num_neighbors=self.avg_num_neighbors,
                use_sc=bool(convnet_sc and layer_idx != 0),
                resnet=bool(resnet and layer_idx != 0),
                nonlinearity_scalars=nonlinearity_scalars,
                nonlinearity_gates=nonlinearity_gates,
            )
            self.layers.append(layer)
            current_irreps = layer.irreps_out
            if layer_idx == self.eq_source_layer_index:
                self.eq_feature_irreps = layer.irreps_out

        self.feature_irreps_out = current_irreps
        self.inv_irreps = o3.Irreps([(self.latent_size, (0, 1))])
        self.eq_irreps = o3.Irreps([(self.latent_size, (1, -1))])
        if self.eq_feature_irreps is None:
            raise RuntimeError(
                "Internal error: eq_feature_irreps was not set during NequIP layer construction."
            )

        if not any(ir == o3.Irrep("0e") for _, ir in self.feature_irreps_out):
            raise ValueError(
                "Final NequIP feature irreps do not contain scalars, so inv_z cannot be produced. "
                f"Final irreps: {self.feature_irreps_out}."
            )
        if not any(ir == o3.Irrep("1o") for _, ir in self.eq_feature_irreps):
            raise ValueError(
                "NequIP eq source features do not contain odd vectors, so eq_z cannot be produced. "
                f"eq_source_irreps={self.eq_feature_irreps}."
            )

        self.inv_head = o3.Linear(self.feature_irreps_out, self.inv_irreps)
        self.eq_head = o3.Linear(self.eq_feature_irreps, self.eq_irreps)

    def _pool(self, x: torch.Tensor, batch: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.pool == "sum":
            return _scatter_sum(x, batch, dim_size=batch_size)
        return _batch_mean_pool(x, batch, num_graphs=batch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected input shape [B, N, 3], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError(
                f"Expected floating-point input for NequIP_Backbone, got dtype={x.dtype}."
            )

        batch_size, num_points, _ = x.shape
        if num_points < 2:
            raise ValueError(
                f"NequIP_Backbone requires at least 2 points per cloud, got {num_points}."
            )

        center = x.mean(dim=1)
        compute_dtype = self.type_embedding.weight.dtype
        pos = (x - center.unsqueeze(1)).to(dtype=compute_dtype)

        edge_dst, edge_src, edge_vec, edge_len, batch = _radius_graph(
            pos,
            r_max=self.r_max,
            max_neighbors=self.max_neighbors,
        )
        node_count = batch_size * num_points
        atom_types = torch.zeros(node_count, dtype=torch.long, device=x.device)
        node_attrs = self.type_embedding(atom_types)
        node_features = node_attrs

        edge_attr = self.edge_sh(edge_vec)
        edge_embedding, _ = self.edge_length_encoding(edge_len)

        eq_source_features = None
        for layer_idx, layer in enumerate(self.layers):
            node_features = layer(
                node_features=node_features,
                node_attrs=node_attrs,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_attr,
                edge_embedding=edge_embedding,
            )
            if layer_idx == self.eq_source_layer_index:
                eq_source_features = node_features

        if eq_source_features is None:
            raise RuntimeError(
                "eq_source_features is missing after NequIP forward pass; "
                f"eq_source_layer_index={self.eq_source_layer_index}, num_layers={len(self.layers)}."
            )

        pooled = self._pool(node_features, batch, batch_size)
        pooled_eq = self._pool(eq_source_features, batch, batch_size)
        inv_z = self.inv_head(pooled)
        eq_z = self.eq_head(pooled_eq)

        expected_inv_shape = (batch_size, self.latent_size)
        if tuple(inv_z.shape) != expected_inv_shape:
            raise RuntimeError(
                f"Unexpected inv_z shape {tuple(inv_z.shape)}, expected {expected_inv_shape}."
            )

        expected_eq_dim = self.latent_size * 3
        if eq_z.dim() != 2 or eq_z.shape[1] != expected_eq_dim:
            raise RuntimeError(
                f"Unexpected raw eq_z shape {tuple(eq_z.shape)}, expected [B, {expected_eq_dim}]."
            )
        eq_z = eq_z.view(batch_size, self.latent_size, 3)

        return inv_z, eq_z, center
