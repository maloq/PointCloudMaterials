"""Compile variable-size encoders without changing checkpoint parameter names."""
import torch
import torch._functorch.config
import torch.backends.opt_einsum
from gatr.utils.einsum import enable_cached_einsum


def compile_encoder(encoder,example,precision='float32'):
    # The pinned GATr exposes this switch specifically for torch.compile.
    # Its opt_einsum cache otherwise specializes on atom counts in Python.
    enable_cached_einsum(False)
    # torch.einsum's separate opt_einsum optimizer otherwise converts symbolic
    # atom counts to Python integers in contract_path, specializing EVERY N.
    # Disabling GATr's cache alone does not disable this second optimizer.
    torch.backends.opt_einsum.enabled = False
    # Both gradient-cache backward passes run outside autocast. AOTAutograd's
    # default "same_as_forward" would silently compile a different backward.
    torch._functorch.config.backward_pass_autocast = 'off'
    # cuEquivariance and GATr lazily create cached contraction graphs/bases.
    # Initialize them before tracing; include generated buffers in device moves.
    device=next(encoder.parameters()).device
    with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=precision=='bf16'):
        encoder(example)
    encoder.to(device)
    # Compile one complete encoder graph per grad/shape regime. Graph breaks
    # previously compiled shared EquiLinear.forward code independently across
    # all layer widths and exhausted its per-code-object cache after two steps.
    torch._dynamo.config.recompile_limit = 8
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    encoder.compile(dynamic=True, fullgraph=True,
                    options={'emulate_precision_casts': True})


def compilation_counters():
    return {name: dict(values) for name, values in torch._dynamo.utils.counters.items()
            if name in ('stats', 'frames', 'graph_break', 'unimplemented')}
