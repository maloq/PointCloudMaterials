"""Explicit MACE backend selection; custom history layers retain e3nn layout."""
from importlib.metadata import version


def mace_backend_config(backend):
    if backend == 'e3nn':
        return None
    if backend != 'cueq':
        raise ValueError(f'Unknown MACE backend {backend!r}; choose e3nn or cueq')
    try:
        import cuequivariance_ops_torch  # noqa: F401: require the compiled CUDA extension
        from mace.modules.wrapper_ops import CUET_AVAILABLE, CuEquivarianceConfig
    except ImportError as error:
        raise RuntimeError('The cueq backend requires cuequivariance, cuequivariance-torch '
                           'and cuequivariance-ops-torch-cu13 in the training environment') from error
    if not CUET_AVAILABLE:
        raise RuntimeError('MACE could not import cuEquivariance; refusing an e3nn fallback')
    # Invariants, velocity injections and temporal attention consume mul_ir.
    # O3_e3nn preserves MACE/e3nn Clebsch-Gordan conventions, including parity.
    return CuEquivarianceConfig(enabled=True, layout='mul_ir', group='O3_e3nn',
                                optimize_all=True, conv_fusion=False)


def mace_backend_metadata(backend):
    result = dict(name=backend, layout='mul_ir', mace_version=version('mace-torch'),
                  e3nn_version=version('e3nn'))
    if backend == 'cueq':
        result.update(cuequivariance_version=version('cuequivariance'),
                      cuequivariance_torch_version=version('cuequivariance-torch'),
                      cuda_ops_version=version('cuequivariance-ops-torch-cu13'))
    return result


def with_mace_backend(encoder, backend):
    """Map encoder weights before creating an optimizer, or for frozen inference.

    AdamW's diagonal moments cannot be exactly rotated to the CuEq parameter
    basis. This function deliberately does not accept or migrate an optimizer.
    The source module is untouched and remains the numerical reference.
    """
    config = mace_backend_config(backend)
    if encoder.mace_backend == backend:
        return encoder
    if encoder.mace_backend != 'e3nn' or backend != 'cueq':
        raise ValueError('Only e3nn-to-cueq weight mapping is supported')
    import copy
    import torch
    from mace import modules
    from mace.cli.convert_e3nn_cueq import transfer_weights

    # Both backbones initialize on CPU; conversion must not consume the caller's
    # RNG stream or alter initialization of subsequent heads.
    with torch.random.fork_rng(devices=[]):
        target = copy.deepcopy(encoder)
        backbone = modules.MACE(**encoder._mace_kwargs, cueq_config=config).float()
        for name in ('node_embedding', 'radial_embedding', 'spherical_harmonics', 'interactions', 'products'):
            setattr(target, name, getattr(backbone, name))
        original = next(encoder.parameters())
        target.to(device=original.device, dtype=original.dtype)
        source_params, target_params = dict(encoder.named_parameters()), dict(target.named_parameters())
        common = {name for name in source_params if '.symmetric_contractions.' not in name}
        expected = {name for name in target_params if '.symmetric_contractions.' not in name}
        if common != expected:
            raise ValueError(f'CuEq conversion parameter mismatch: {sorted(common ^ expected)}')
        for name in common:
            source_shape = tuple(n for n in source_params[name].shape if n != 1)
            target_shape = tuple(n for n in target_params[name].shape if n != 1)
            if source_shape != target_shape:
                raise ValueError(f'CuEq conversion parameter shape mismatch: {name}')
            target_params[name].requires_grad_(source_params[name].requires_grad)
        for layer in range(encoder.num_layers):
            prefix = f'products.{layer}.symmetric_contractions.'
            flags = {p.requires_grad for name,p in source_params.items() if name.startswith(prefix)}
            if len(flags) != 1:
                raise ValueError(f'CuEq requires uniform trainability within product {layer}')
            target_params[prefix+'weight'].requires_grad_(flags.pop())
        transfer_weights(encoder, target, num_product_irreps=2, correlation=encoder._mace_kwargs['correlation'],
                         num_layers=encoder.num_layers, use_reduced_cg=True, keep_last_layer_irreps=True)
        for name in common:
            if not torch.equal(source_params[name], target_params[name].reshape_as(source_params[name])):
                raise RuntimeError(f'MACE failed to transfer parameter {name}')
        target.mace_backend = backend
        target.train(encoder.training)
    return target
