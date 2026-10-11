"""Optional execution acceleration for the unchanged spatial eSEN parameters."""
from types import MethodType
import torch


def _permutation(x, indices):
    # The original one-hot matrix einsum is autocast. Preserve that rounding.
    if torch.is_autocast_enabled('cuda'):
        x = x.to(torch.get_autocast_dtype('cuda'))
    return x.index_select(1, indices)


def _so2_forward(self, x, x_edge):
    """FAIR SO2_Convolution with exact index permutations instead of dense GEMMs.

    Derived from esen_vendor/nn/so2_layers.py under its retained MIT license.
    All learned projections, radial factors and +/-m arithmetic are unchanged.
    """
    edges = len(x_edge)
    x = _permutation(x, self._esen_to_m)
    if self.rad_func is not None:
        x_edge = self.rad_func(x_edge)
    scalar = x.narrow(1, 0, self.mappingReduced.m_size[0]).reshape(edges, -1)
    if self.rad_func is not None:
        scalar = scalar * x_edge.narrow(1, 0, self.fc_m0.in_features)
    scalar = self.fc_m0(scalar)
    extra = None
    if self.extra_m0_output_channels is not None:
        extra = scalar.narrow(-1, 0, self.extra_m0_output_channels)
        scalar = scalar.narrow(-1, self.extra_m0_output_channels,
                              self.fc_m0.out_features-self.extra_m0_output_channels)
    out = [scalar.reshape(edges, -1, self.m_output_channels)]
    offset = self.mappingReduced.m_size[0]
    radial_offset = self.fc_m0.in_features
    for m in range(1, self.mmax+1):
        width = self.so2_m_conv[m-1].fc.in_features
        value = x.narrow(1, offset, 2*self.mappingReduced.m_size[m]).reshape(edges, 2, -1)
        if self.rad_func is not None:
            value = value*x_edge.narrow(1, radial_offset, width).reshape(edges, 1, width)
        value = self.so2_m_conv[m-1](value)
        out.append(value.reshape(edges, -1, self.m_output_channels))
        offset += 2*self.mappingReduced.m_size[m]
        radial_offset += width
    result = _permutation(torch.cat(out, dim=1), self._esen_from_m)
    return (result, extra) if self.extra_m0_output_channels is not None else result


def index_permutations(geometry):
    from .esen_vendor.nn.so2_layers import SO2_Convolution
    for module in geometry.modules():
        if not isinstance(module, SO2_Convolution):
            continue
        matrix = module.mappingReduced.to_m
        if not torch.equal(matrix.sum(0), torch.ones_like(matrix.sum(0))) or not torch.equal(matrix.sum(1), torch.ones_like(matrix.sum(1))):
            raise ValueError('SO(2) mapping is not the expected permutation matrix')
        module.register_buffer('_esen_to_m', matrix.argmax(1), persistent=False)
        module.register_buffer('_esen_from_m', matrix.argmax(0), persistent=False)
        module.forward = MethodType(_so2_forward, module)


def direct_backward(model, q, target, weights, seed):
    """Same complete 256-row objective, retain both graphs instead of replay."""
    from src.research.structural_response.model import views, spatial_loss
    with torch.autocast('cuda', dtype=torch.bfloat16):
        latent = [model.geometry(x, keep) for x, keep in views(q, 'vicreg_descriptors', seed)]
    loss, stats = spatial_loss(model, latent, target, weights, 'vicreg_descriptors')
    loss.backward()
    return float(loss.detach()), {k: float(v.detach()) for k, v in stats.items()}


def compile_blocks(geometry):
    """Compile tensor blocks; keep variable-size neighbor construction eager.

    Module.compile preserves parameter/checkpoint names. Emulate the original
    BF16 casts, and use FP32 backward context as in the ordinary training loop.
    """
    import torch._functorch.config
    import torch._inductor.config
    torch._functorch.config.backward_pass_autocast = 'off'
    torch._inductor.config.compile_threads = 2
    options = {'emulate_precision_casts': True}
    geometry.degree_embedding.compile(fullgraph=True, dynamic=True, options=options)
    for block in geometry.blocks:
        block.compile(fullgraph=True, dynamic=True, options=options)
    geometry.norm.compile(fullgraph=True, dynamic=True, options=options)


def configure(geometry, architecture, variant):
    if architecture == 'tensornet':
        if variant not in ('eager', 'direct', 'compiled-direct'):
            raise ValueError(f'Unsupported TensorNet execution variant: {variant}')
        if variant == 'compiled-direct':
            geometry.compile_blocks()
        return
    if variant not in ('eager', 'compiled', 'direct', 'compiled-direct', 'indexed',
                       'compiled-indexed', 'indexed-direct', 'compiled-indexed-direct'):
        raise ValueError(f'Unknown execution variant: {variant}')
    if architecture == 'mace' and variant not in ('eager', 'direct'):
        raise ValueError('MACE retains its cuEquivariance backend')
    if 'indexed' in variant:
        index_permutations(geometry)
    if variant.startswith('compiled'):
        compile_blocks(geometry)
