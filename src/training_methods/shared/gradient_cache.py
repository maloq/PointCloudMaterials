"""Full-batch representation gradients with exact stochastic microbatch replay."""
import torch


@torch.no_grad()
def cache_encoder(encode, batches, device):
    states, rngs = [], []
    for batch in batches:
        rngs.append((torch.get_rng_state(),
                     torch.cuda.get_rng_state(device) if device.type == 'cuda' else None))
        states.append(encode(batch))
    return torch.cat(states).detach().requires_grad_(True), rngs


def backward_cached(encode, batches, features, rngs):
    """Replay encoder chunks without advancing the caller's RNG a second time."""
    device = features.device
    devices = [device.index] if device.type == 'cuda' else []
    derivative = features.grad.detach()
    offset = 0
    for batch, (cpu_rng, cuda_rng) in zip(batches, rngs, strict=True):
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            current = encode(batch)
            current.backward(derivative[offset:offset + len(current)])
            offset += len(current)
