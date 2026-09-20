"""Multi-GPU encoder replay with one unchanged global VICReg/bond objective.

Packed graphs are indivisible microbatches. Only encoder computation is replicated;
heads, within-domain statistics, gradient clipping and optimizer updates are global.
StructuralMACE has no random layers or mutable training normalization buffers.
"""
from concurrent.futures import ThreadPoolExecutor

import torch

from src.data.structural_pretraining.batches import move
from src.models.encoders.structural import StructuralMACE
from src.models.encoders.mixed_mace import MixedSnapshotMACE
from src.training_methods.structural_pretraining.train import target_batch
from .compilation import compile_encoder


class ParallelMACE:
    def __init__(self, model, devices, example, precision, compile_models=False):
        if not isinstance(model, MixedSnapshotMACE) or not isinstance(model.encoder, StructuralMACE):
            raise TypeError('Parallel replay requires the deterministic structural MACE encoder')
        self.devices = [torch.device('cuda', i) for i in devices]
        if len(set(devices)) != len(devices) or len(devices) < 2:
            raise ValueError('Parallel MACE requires at least two distinct CUDA devices')
        if next(model.parameters()).device != self.devices[0]:
            raise ValueError('Global model and heads must be on the first requested device')
        self.precision = precision
        self.encoders = [model.encoder]
        # Construct cuEquivariance kernels on their own device; deepcopy is invalid.
        # Preserve the training RNG, although the actual encoder forward is deterministic.
        with torch.random.fork_rng(devices=devices):
            for device in self.devices[1:]:
                with torch.cuda.device(device):
                    encoder = StructuralMACE(channels=model.encoder.channels,
                        backend=model.encoder.backend).to(device)
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,
                                                         enabled=precision == 'bf16'):
                        encoder(move(example, device))
                    encoder.load_state_dict(model.encoder.state_dict(), strict=True)
                    if compile_models:
                        compile_encoder(encoder, move(example, device), precision)
                    self.encoders.append(encoder)
        self.pool = ThreadPoolExecutor(max_workers=len(devices))

    def close(self):
        self.pool.shutdown(wait=True)

    def _cache(self, replica, work):
        device = self.devices[replica]
        encoder = self.encoders[replica]
        encoder.train()
        encoder.zero_grad(set_to_none=True)
        result = []
        with torch.cuda.device(device), torch.no_grad():
            for index, batch in work:
                batch = move(batch, device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.precision == 'bf16'):
                    states = encoder(batch, return_equivariant=True).float()
                result.append((index, batch, states))
            torch.cuda.synchronize(device)
        return result

    def _replay(self, replica, work):
        device = self.devices[replica]
        with torch.cuda.device(device):
            for batch, derivative in work:
                derivative = derivative.to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.precision == 'bf16'):
                    states = self.encoders[replica](batch, return_equivariant=True).float()
                states.backward(derivative)
            torch.cuda.synchronize(device)

    def update(self, model, objective, batches, optimizer, temporal, delta, extra):
        if len(batches) < len(self.devices):
            raise ValueError('Insufficient packed microbatches to use all requested GPUs')
        device = self.devices[0]
        optimizer.zero_grad(set_to_none=True)
        work = [list(enumerate(batches))[i::len(self.devices)] for i in range(len(self.devices))]
        futures = [self.pool.submit(self._cache, i, rows) for i, rows in enumerate(work)]
        cached = [future.result() for future in futures]
        ordered = sorted((row for rows in cached for row in rows), key=lambda row: row[0])
        z = torch.cat([states.to(device) for _, _, states in ordered]).detach().requires_grad_(True)
        targets = target_batch(batches, device)
        targets['bond_order'] = torch.cat([batch['bond_order'] for batch in batches]).to(device)
        targets.update({key: torch.as_tensor(value, device=device) for key, value in extra.items()})
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.precision == 'bf16'):
            loss, terms = objective(model, z, targets, temporal, torch.as_tensor(delta, device=device, dtype=z.dtype))
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite parallel MACE loss: {terms}')
        loss.backward()
        derivative = z.grad.detach()
        offsets = [0]
        for _, _, states in ordered:
            offsets.append(offsets[-1]+len(states))
        torch.cuda.synchronize(device)
        futures = [self.pool.submit(self._replay, i,
            [(batch, derivative[offsets[index]:offsets[index+1]]) for index, batch, _ in rows])
            for i, rows in enumerate(cached)]
        for future in futures:
            future.result()
        # Sum, do not average: each local derivative already belongs to the one
        # global mean loss, including correlations that cross GPU boundaries.
        for replica in self.encoders[1:]:
            for primary, secondary in zip(model.encoder.parameters(), replica.parameters(), strict=True):
                if secondary.grad is not None:
                    grad = secondary.grad.to(device)
                    if primary.grad is None:
                        primary.grad = grad
                    else:
                        primary.grad.add_(grad)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for replica in self.encoders[1:]:
                for primary, secondary in zip(model.encoder.parameters(), replica.parameters(), strict=True):
                    secondary.copy_(primary)
        for device in self.devices:
            torch.cuda.synchronize(device)
        return {key: float(value.detach()) for key, value in terms.items()} | dict(gradient_norm=float(norm))
