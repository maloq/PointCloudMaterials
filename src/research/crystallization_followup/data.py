"""Existing paired observations and strictly past dense MD descriptors."""
import copy
import json
import numpy as np
import torch

from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import file_hash
from src.research.context_night.context import LOCAL_COLUMNS
from src.research.structured_context.reuse_data import ReusePaths


DENSE_OFFSETS = (-16, -8, -4, -1, 0)  # frame spacing 0.75 ps
EXTRA_BLOCKS = {'spans': slice(0, 2), 'dense': slice(2, 487),
                'quench': slice(487, 584), 'front': slice(584, 668)}
EXTRA_DIM = 668


def history_information(values, offsets, rates):
    """Two backward secants at actual physical lags; retain the present and shells."""
    if values.shape[1:] != (3, 97) or offsets.shape != values.shape[:2]:
        raise ValueError('Expected three chronological 97-channel observations and their times')
    if torch.any(offsets[:, -1] != 0) or torch.any(torch.diff(offsets, dim=1) <= 0):
        raise ValueError('History must end at the origin and have strictly increasing times')
    current = values[:, -1]
    changes = [current[:, :93] - values[:, j, :93] for j in (-2, -3)]
    if rates:
        changes = [value / (-offsets[:, j, None]) for value, j in zip(changes, (-2, -3))]
    return torch.cat((current[:, :93], *changes, torch.zeros_like(current[:, :93]),
                      current[:, 93:], current.new_zeros(len(current), 128)), -1)


def dense_observations(timeline, origins, centers):
    """Gather real MD frames only. Future timeline contents cannot enter this join."""
    frames = np.asarray(origins)[:, None] + np.asarray(DENSE_OFFSETS)[None]
    if frames.min() < 0 or frames.max() >= timeline.shape[1]:
        raise ValueError('Dense observed history lies outside the archived timeline')
    if timeline.shape[-1] != 97:
        raise ValueError(f'Expected physical/order/shell width 97, got {timeline.shape}')
    return timeline[np.asarray(centers)[:, None], frames].reshape(len(frames), -1)


def balanced_moments(values, groups):
    """One unit of mass per training source, natural windows within each source."""
    mean = torch.zeros(values.shape[-1], dtype=torch.float64, device=values.device)
    second = mean.clone()
    for ids in groups.values():
        x = values[ids].double()
        mean += x.mean(0) / len(groups)
        second += x.square().mean(0) / len(groups)
    return mean.float(), (second - mean.square()).clamp_min(1e-6).sqrt().float()


def extra_mask(spec, device):
    mask = torch.zeros(EXTRA_DIM, device=device)
    if spec['history_rates']:
        mask[EXTRA_BLOCKS['spans']] = 1
    if spec['dense_mode'] != 'off':
        mask[EXTRA_BLOCKS['dense']] = 1
    if spec['quench_descriptors']:
        mask[EXTRA_BLOCKS['quench']] = 1
    if spec['front_mode'] != 'off':
        from .front import RADII, NAMES
        radii = range(3) if spec['front_radius_A'] == 0 else [RADII.index(spec['front_radius_A'])]
        columns = {'all': list(range(len(NAMES))), 'coherence': [0, 1, 6, 7, 8, 9],
                   'geometry': [2, 3, 4, 5, 10, 11, 12, 13]}[spec['front_features']]
        for time in range(2 if spec['front_mode'] in ('rates', 'repeat') else 1):
            for radius in radii:
                mask[[584+time*42+radius*14+c for c in columns]] = 1
    return mask


class FollowupPaths:
    """One resident paired dataset per worker, reused by all matched ablations."""
    def __init__(self, plan, spec, device='cuda'):
        self.plan = plan
        self.device = torch.device(device)
        self.domains = {}
        for domain in ('relaxed', 'observed'):
            base = dict(spec, observation_domain=domain)
            self.domains[domain] = ReusePaths(plan, base, device)
        cold, hot = self.domains['relaxed'], self.domains['observed']
        if cold.corpus.rows != hot.corpus.rows:
            raise ValueError('Original/relaxed populations differ')
        for field in ('history', 'history_offsets', 'rows', 'mean', 'scale', 'states', 'onsets'):
            torch.testing.assert_close(getattr(cold, field), getattr(hot, field), rtol=0, atol=0)
        self.corpus = cold.corpus
        for field in ('rows', 'states', 'onsets', 'mean', 'scale', 'sources', 'full_training_indices',
                      'full_training_groups', 'history', 'history_offsets'):
            setattr(self, field, getattr(cold, field))
        # These immutable original MD descriptors are a separate input array, never
        # read from the normalized future-target tensor.
        assay_plan = json.loads(resolve_path(plan['config']['reuse_plan']).read_text())
        assay = resolve_path(assay_plan['config']['assay_cache'])
        dense = np.empty((len(self.rows), 485), np.float32)
        for source in self.sources:
            sid = source['id']; a = self.corpus.arrays[sid]
            path = assay / source['shard']
            if file_hash(path) != source['shard_sha256']:
                raise ValueError(f'Original descriptor shard changed: {sid}')
            with np.load(path) as archived:
                np.testing.assert_array_equal(archived['atom_ids'], source['center_atom_ids'])
                local = np.concatenate((a['packet'], a['order']), -1)[..., LOCAL_COLUMNS]
                timeline = np.concatenate((local, archived['shell'][..., [0, 1, 6, 7]]), -1)
            ids = np.flatnonzero(self.corpus.source_ids == sid)
            meta = [self.corpus.rows[i] for i in ids]
            origins = [plan['anchors'][r[1]] for r in meta]
            dense[ids] = dense_observations(timeline, origins, [r[2] for r in meta])
        self.dense = torch.tensor(dense, device=device)
        self.front = torch.zeros((len(self.rows), 2, 42), device=device)
        if 'front_cache' in plan['followup_config']:
            cache = resolve_path(plan['followup_config']['front_cache'])
            for item in self.sources:
                folder = cache/str(item['id']); receipt = json.loads((folder/'complete.json').read_text())
                from . import front
                if receipt['producer_sha256'] != file_hash(front.__file__):
                    raise ValueError(f'Front producer changed: {item["id"]}')
                if receipt['manifest_sha256'] != item['manifest_sha256'] or file_hash(folder/'front.npz') != receipt['sha256']:
                    raise ValueError(f'Observed front cache changed: {item["id"]}')
                with np.load(folder/'front.npz') as a:
                    np.testing.assert_array_equal(a['atom_ids'], item['center_atom_ids'])
                    ids = np.flatnonzero(self.corpus.source_ids == item['id'])
                    rows = [self.corpus.rows[i] for i in ids]
                    origins = np.array([plan['anchors'][r[1]] for r in rows])
                    frames = np.stack((origins, origins-4), -1)
                    indices = np.searchsorted(a['frames'], frames)
                    np.testing.assert_array_equal(a['frames'][indices], frames)
                    values = a['features'][indices, np.array([r[2] for r in rows])[:, None]]
                    self.front[ids] = torch.tensor(values, device=device)
        self.set_context(spec)

    def set_context(self, spec):
        self.spec = copy.deepcopy(spec)
        if spec['dense_mode'] not in ('off', 'real', 'repeat'):
            raise ValueError(spec['dense_mode'])
        if spec['secondary_domain'] not in ('off', 'observed', 'relaxed'):
            raise ValueError(spec['secondary_domain'])
        if spec['front_mode'] not in ('off', 'current', 'repeat', 'rates'):
            raise ValueError(spec['front_mode'])
        if spec['front_mode'] != 'off' and 'front_cache' not in self.plan['followup_config']:
            raise ValueError('A front experiment requires a verified observed-front cache')
        self.primary = self.domains[spec['observation_domain']]
        c = self.rows[:, 2]
        info = self.primary.information[self.history, c[:, None]]
        self.information_values = history_information(info, self.history_offsets, spec['history_rates'])
        cold = self.domains['relaxed'].information[self.history[:, -1], c]
        hot = self.domains['observed'].information[self.history[:, -1], c]
        dense = self.dense
        if spec['dense_mode'] == 'repeat':
            dense = dense[:, -97:].repeat(1, len(DENSE_OFFSETS))
        current = self.front[:, 0]
        past = current if spec['front_mode'] == 'repeat' else (current-self.front[:, 1])/3.
        self.extra_values = torch.cat((-self.history_offsets[:, [1, 0]], dense, hot - cold, current, past), -1)
        groups = self.full_training_groups
        self.information_mean, self.information_scale = balanced_moments(self.information_values, groups)
        self.extra_mean, self.extra_scale = balanced_moments(self.extra_values, groups)

    def observed(self, indices):
        primary = self.primary.observed(indices)
        primary['information'] = self.information_values[indices]
        primary['extra'] = self.extra_values[indices]
        if not self.spec['absolute_clock']:
            primary['condition'][:, -2:] = 0
        if self.spec['secondary_domain'] != 'off':
            secondary = self.domains[self.spec['secondary_domain']].observed(indices)
            primary['secondary_features'] = secondary['features']
            primary['secondary_geometry'] = secondary['geometry']
        return primary

    def targets(self, indices, normalize=True):
        return self.domains['relaxed'].targets(indices, normalize)

    def baseline(self, indices):
        return self.domains['relaxed'].baseline(indices)

    def event_bins(self, indices):
        return self.domains['relaxed'].event_bins(indices)
