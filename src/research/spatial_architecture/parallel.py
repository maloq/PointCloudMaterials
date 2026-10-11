"""Two local GPU geometry replicas with one exact global-batch spatial loss.

Only the primary model/optimizer is checkpointed. Each update broadcasts its
geometry weights, splits the already-generated views, gathers differentiable
embeddings, and sums the two geometry gradients before clipping/AdamW. No
per-device VICReg statistics or independent per-device optimizer is used.
"""
from concurrent.futures import ThreadPoolExecutor

import torch

from src.models.encoders.spatial_esen import SpatialESEN
from src.models.encoders.esen_runtime import configure, direct_backward
from src.research.structural_response.model import spatial_loss, views


def devices(c):
    count = c['runtime'].get('gpus', 1)
    if count not in (1, 2) or torch.cuda.device_count() < count:
        raise ValueError(f'Require {count} local GPUs; visible={torch.cuda.device_count()}')
    return list(range(count))


def synchronize(c):
    for device in devices(c):
        torch.cuda.synchronize(device)


def create(c, architecture, model):
    if len(devices(c)) == 1:
        return None
    if architecture != 'esen' or not c['execution_variants'][architecture].endswith('direct'):
        raise ValueError('Two-GPU spatial execution currently requires direct eSEN')
    if c['spatial_training']['microbatch'] != c['spatial_training']['batch_size']:
        raise ValueError('Two-GPU execution splits one full global microbatch')
    return ParallelDirect(c, model)


def _thread_setup():
    from torch._dynamo import config as dynamo_config
    from torch._functorch import config as functorch_config
    from torch._inductor import config as inductor_config
    # Compiler configuration is thread-local. Install the same execution
    # profile in the main thread and both CUDA forward-dispatch threads.
    dynamo_config.recompile_limit = 32
    functorch_config.backward_pass_autocast = 'off'
    inductor_config.compile_threads = 2


class ParallelDirect:
    def __init__(self, c, model):
        # Two device guards, both grad/no-grad contexts, and full/partial
        # batches share the vendor block's Python code object. Keep all of
        # these compiled variants; the default eight-entry limit is too small.
        _thread_setup()
        if next(model.parameters()).device != torch.device('cuda:0'):
            raise ValueError('The primary spatial model must be on cuda:0')
        self.model = model
        self.config_rows = c['data']['training_rows']
        with torch.random.fork_rng(devices=devices(c)), torch.cuda.device(1):
            replica = SpatialESEN(**c['encoder']).cuda(1)
            replica.load_state_dict(model.geometry.state_dict())
            configure(replica, 'esen', c['execution_variants']['esen'])
        self.replica = replica
        self.primary_parameters = tuple(model.geometry.parameters())
        self.replica_parameters = tuple(replica.parameters())
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='spatial-gpu',
                                           initializer=_thread_setup)
        self.warmed = False

    def _encode(self, device, inputs):
        geometry = self.model.geometry if device == 0 else self.replica
        with torch.cuda.device(device), torch.autocast('cuda', dtype=torch.bfloat16):
            return [geometry(x, keep) for x, keep in inputs]

    def backward(self, q, target, weights, seed):
        if len(q) < 2:
            raise ValueError('VICReg requires at least two global observations')
        self.replica.train(self.model.geometry.training)
        self.replica.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.cuda.device(1):
            for source, dest in zip(self.primary_parameters, self.replica_parameters, strict=True):
                dest.copy_(source, non_blocking=True)
        # Generate the SAME two global jitter tensors as the single-GPU recipe.
        # Splitting before random generation would change its observation views.
        inputs = [[], []]
        for x, keep in views(q, 'vicreg_descriptors', seed):
            assert keep is None
            for device, part in enumerate(x.tensor_split(2)):
                inputs[device].append((part.to(device=torch.device('cuda', device), non_blocking=True), None))
        if self.warmed:
            work = [self.executor.submit(self._encode, d, inputs[d]) for d in range(2)]
            local = [item.result() for item in work]
        else:
            # Initial compilation is sequential; steady-state launches overlap.
            local = [self._encode(d, inputs[d]) for d in range(2)]
            self.warmed = True
        latent = [torch.cat((local[0][v], local[1][v].to(q.device))) for v in range(2)]
        loss, stats = spatial_loss(self.model, latent, target, weights, 'vicreg_descriptors')
        loss.backward()
        # The global loss already applies the batch mean. SUM the replica
        # contributions; averaging them would halve the geometry gradient.
        with torch.no_grad():
            if any(p.grad is None for p in self.primary_parameters + self.replica_parameters):
                raise RuntimeError('Missing gradient in two-GPU spatial geometry')
            incoming = torch.cat([p.grad.flatten() for p in self.replica_parameters]).to(q.device)
            offset = 0
            for p in self.primary_parameters:
                p.grad.add_(incoming[offset:offset+p.numel()].view_as(p))
                offset += p.numel()
        return float(loss.detach()), {k: float(v.detach()) for k, v in stats.items()}

    def audit(self, q, target, weights, seed):
        """Real-batch value/gradient parity before any scientific optimizer step."""
        records = []
        # Include the actual unequal final split, e.g. 21 rows -> 11 + 10.
        remainder = self.config_rows % len(q)
        for n in dict.fromkeys([len(q), remainder] if remainder > 1 else [len(q)]):
            self.model.zero_grad(set_to_none=True)
            reference, reference_stats = direct_backward(self.model, q[:n], target[:n], weights, seed)
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            reference_gradient = torch.cat([p.grad.flatten() for p in parameters]).clone()
            self.model.zero_grad(set_to_none=True)
            actual, actual_stats = self.backward(q[:n], target[:n], weights, seed)
            gradient = torch.cat([p.grad.flatten() for p in parameters])
            relative = float((gradient-reference_gradient).norm()/reference_gradient.norm().clamp_min(1e-12))
            error = abs(actual-reference)/max(abs(reference), 1e-12)
            record = dict(rows=n, device_rows=[(n+1)//2, n//2], loss_relative_error=error,
                gradient_relative_l2=relative, reference_loss=reference, parallel_loss=actual,
                reference_components=reference_stats, parallel_components=actual_stats,
                all_gradients_finite=bool(torch.isfinite(gradient).all()))
            if not record['all_gradients_finite'] or error > .02 or relative > .05:
                raise FloatingPointError(f'Two-GPU global-batch objective mismatch: {record}')
            print('two-GPU batch parity', record, flush=True)
            records.append(record)
        self.model.zero_grad(set_to_none=True)
        self.replica.zero_grad(set_to_none=True)
        return records

    def close(self):
        self.executor.shutdown(wait=True)
