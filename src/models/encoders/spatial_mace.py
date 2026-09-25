"""Tensor-only local MACE backbone; layout belongs to the spatial runtime.

No causal/velocity/denoising model is constructed to obtain spatial blocks.
The scalar export is independent of internal mul_ir versus ir_mul storage.
"""
import numpy as np
import torch
from torch import nn
from e3nn import o3
from mace import modules
from .mace_backend import mace_backend_config


class SpatialMACE(nn.Module):
    def __init__(self, *, d0, n_ref, radius, channels=128, code_dim=128,
                 cutoff=5., backend='cueq', layout=None, conv_fusion=None):
        super().__init__()
        self.channels, self.radius, self.cutoff = channels, radius, cutoff
        self.layout = ('ir_mul' if backend == 'cueq' else 'mul_ir') if layout is None else layout
        self.conv_fusion = backend == 'cueq' if conv_fusion is None else conv_fusion
        config = mace_backend_config(backend, layout=self.layout, conv_fusion=self.conv_fusion)
        irreps = o3.Irreps(f'{channels}x0e + {channels}x1o + {channels}x2e')
        backbone = modules.MACE(r_max=cutoff, num_bessel=6, num_polynomial_cutoff=5,
            max_ell=2, interaction_cls=modules.RealAgnosticResidualInteractionBlock,
            interaction_cls_first=modules.RealAgnosticInteractionBlock, num_interactions=2,
            num_elements=1, hidden_irreps=irreps, MLP_irreps=o3.Irreps(f'{channels}x0e'),
            atomic_energies=np.zeros(1), avg_num_neighbors=12., atomic_numbers=[13],
            correlation=2, gate=torch.nn.functional.silu, radial_MLP=[32],
            keep_last_layer_irreps=True, cueq_config=config).float()
        for name in ('node_embedding', 'radial_embedding', 'spherical_harmonics', 'interactions', 'products'):
            setattr(self, name, getattr(backbone, name))
        self.register_buffer('atomic_numbers', backbone.atomic_numbers)
        self.register_buffer('d0', torch.tensor(float(d0)))
        self.register_buffer('n_ref', torch.tensor(float(n_ref)))
        self.center_embedding = nn.Linear(1, channels, bias=False)

    def atom_features(self, graph):
        """Equivariant channels before scalar export, in the recorded layout."""
        attrs, weight = graph['attrs'], graph['weight']
        h = self.node_embedding(attrs) + self.center_embedding(graph['center'])
        for k, (interaction, product) in enumerate(zip(self.interactions, self.products, strict=True)):
            message, skip = interaction(node_attrs=attrs, node_feats=h,
                edge_attrs=graph['angular'], edge_feats=graph['radial'],
                edge_index=graph['edge'], cutoff=None, first_layer=k == 0)
            h = product(message, sc=skip, node_attrs=attrs)
            # Contract ALL irreps before extracting scalars. This is invariant
            # to layout and preserves the old tensor-dependent normalization.
            h = h / torch.sqrt(1 + h.square().mean(-1, keepdim=True)) * weight[:, None]
        return h

    def pooled_graph(self, graph):
        h = self.atom_features(graph)
        weight = graph['weight']
        scalar = h[:, :self.channels]
        pooled = scalar.new_zeros(graph['size'], self.channels).index_add(
            0, graph['group'], scalar * weight[:, None]) / self.n_ref
        return torch.cat((scalar[graph['centers']], pooled), -1)

    def forward(self, graph):
        return self.export_pooled(self.pooled_graph(graph))


def compile_spatial_encoder(encoder, example):
    """Compile the tensor graph only, retaining ordinary checkpoint names.

    NumPy batch planning remains outside this call. Lazy cuEq bases are warmed
    first. Fullgraph rejects silent graph breaks; compilation failures propagate.
    """
    import torch._functorch.config
    import torch.backends.opt_einsum
    torch.backends.opt_einsum.enabled = False
    torch._functorch.config.backward_pass_autocast = 'off'
    with torch.no_grad():
        encoder(example)
    encoder.to(next(encoder.parameters()).device)
    torch._dynamo.config.recompile_limit = 8
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    encoder.compile(fullgraph=True, dynamic=True, options={'emulate_precision_casts': True})
