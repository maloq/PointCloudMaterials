"""Verified graph packing, persistent inputs and vectorized physical inference."""
import numpy as np
import torch

from src.models.encoders.mace_causal_batch import pack_histories, validate_history
from .objective import targets


class CausalRuntime:
    def __init__(self, samples, norm, model, device, spec=None):
        spec = dict(spec or {})
        unknown = set(spec)-{'encoding', 'residency', 'batch_size'}
        if unknown:
            raise ValueError(f'Unknown causal runtime settings: {sorted(unknown)}')
        self.encoding = spec.get('encoding', 'packed')
        self.residency = spec.get('residency', 'host')
        self.batch_size = int(spec.get('batch_size', 8))
        self.device = torch.device(device)
        if self.encoding not in ('packed', 'sequential') or self.residency not in ('host', 'device', 'train_device') or self.batch_size < 1:
            raise ValueError(f'Invalid causal execution settings: {spec}')
        self.samples = samples
        self.indices = {id(s): i for i, s in enumerate(samples)}
        if len(self.indices) != len(samples):
            raise ValueError('Runtime source list contains the same sample object more than once')
        offsets = samples[0]['history'].offsets_ps
        for sample in samples:
            try:
                validate_history(sample['history'], model)
                if not torch.equal(sample['history'].offsets_ps, offsets):
                    raise ValueError('All samples in a runtime must have identical physical offsets')
            except ValueError as error:
                raise ValueError(f'Invalid history: source={sample["source_id"]}, center={sample["center_atom_id"]}, '
                                 f'anchor={sample["anchor_ps"]}: {error}') from error
        resident = [self.residency == 'device' or (self.residency == 'train_device' and s['split'] == 'train') for s in samples]
        tensor_bytes = lambda h: sum(v.numel()*v.element_size() for v in vars(h).values() if isinstance(v, torch.Tensor))
        requested = sum(tensor_bytes(s['history']) for s, keep in zip(samples, resident, strict=True) if keep)
        if self.device.type == 'cuda' and requested:
            free, _ = torch.cuda.mem_get_info(self.device)
            if requested >= free:
                raise MemoryError(f'History residency needs {requested/2**30:.2f} GiB before activations; '
                                  f'only {free/2**30:.2f} GiB free. Choose train_device/host or fewer simultaneous fits.')
        self.histories = [s['history'].to(self.device) if keep else s['history']
                          for s, keep in zip(samples, resident, strict=True)]
        # These are immutable labels, never model inputs. Normalization was fitted
        # on training sources; indexed training batches cannot consume test rows.
        self.target_bank = {k: v.to(self.device) for k, v in targets(samples, norm, 'cpu').items()}
        self.summary = dict(encoding=self.encoding, residency=self.residency, batch_size=self.batch_size,
                            history_bytes=requested, samples=len(samples), device=str(self.device))

    def sample_indices(self, samples):
        return [self.indices[id(s)] for s in samples]

    def target(self, samples):
        indices = torch.tensor(self.sample_indices(samples), dtype=torch.long, device=self.device)
        return {k: v.index_select(0, indices) for k, v in self.target_bank.items()}

    def encode(self, model, samples):
        indices = self.sample_indices(samples)
        pieces = []
        for start in range(0, len(indices), self.batch_size):
            histories = [self.histories[i] for i in indices[start:start+self.batch_size]]
            if self.encoding == 'sequential':
                pieces.extend(model(h.to(self.device), validate=False) for h in histories)
            else:
                # A mixed train/validation request with train-only residency is
                # uncommon, but must preserve sample order and physical meaning.
                if len({h.positions.device for h in histories}) > 1:
                    histories = [h.to(self.device) for h in histories]
                packed = pack_histories(histories, validated=True).to(self.device)
                pieces.append(model(packed, validate=False))
        return torch.cat(pieces)

    @torch.no_grad()
    def extract(self, model, heads, samples):
        model.eval(); heads.eval()
        collected = {k: [] for k in ('z', 'present', 'future', 'scale', 'hazard')}
        for start in range(0, len(samples), self.batch_size):
            batch = samples[start:start+self.batch_size]
            z = self.encode(model, batch)
            prediction = heads(z, self.target(batch)['temperature'])
            collected['z'].append(z)
            for key in ('present', 'future', 'scale', 'hazard'):
                if prediction[key] is not None:
                    collected[key].append(prediction[key])
        result = {k: torch.cat(v).cpu().numpy() if v else None for k, v in collected.items()}
        if any(v is not None and not np.isfinite(v).all() for v in result.values()):
            raise FloatingPointError('Nonfinite embedding/physical predictions during batched extraction')
        return result
