"""Matched history treatments and a separate fixed-support physical task."""
import numpy as np
import torch
from torch import nn
from .data import read, root, cache, sha, digest, write_json, resolve_path
from .physical import load as load_physical
from .model import DirectProcess, feature_weights

FAMILY = 'measurement_followup'
GROUPS = [(0, 74), (74, 83), (83, 87), (87, 91)]
GROUP_NAMES = ['density', 'velocity_shells', 'bond_rates', 'flow']


def prepare(c):
    path = resolve_path(c['source_run'])/'technical/plan.json'
    if sha(path) != c['source_plan_sha256']: raise ValueError('Changed direct-measurement source plan')
    source = read(path); original = source['config']
    packet = read(cache(original)/'data.json')
    if packet['identity'] != source['identity']: raise ValueError('Mismatched native measurement identity')
    plan = dict(config=c, source_identity=source['identity'], source_config=original,
                data_sha256=packet['sha256'], rows=source['rows'], columns=source['columns'],
                native_origin_index=13, native_origin_shift_ps=1., native_future_ps=5.,
                roles=source['roles'])
    plan['identity'] = digest(plan)
    destination = root(c)/'technical/plan.json'
    if destination.exists() and read(destination) != plan: raise ValueError('Changed overnight recipe; use a new revision')
    write_json(destination, plan)
    write_json(root(c)/'technical/prediction-context.json', dict(
        observation='244 fixed BO/TDA/density/velocity measurements; no learned spatial encoder',
        native='14400 fixed-center sequences; common shifted origin +1ps; history4/history14/summary14; exact0.1ps targets through5ps',
        physical='300 existing fixed-box BAOAB anchors; current observation only; initial80 identities; two shots at5..100fs',
        perturbation='only present observation changes; real Q/V input secants; analytic91-channel output mean response',
        conditions='no temperature/age/absolute-time/box/species inputs; ordinal slots only',
        selection='native selection NLL or physical selection NLL; no AP or held-out selection',
        expansion='more history and three seeds; no new MD or added independent ancestors',
        physical_gate='short physical fits are bounded diagnostic comparisons, not automatic transfer into native forecasts'))
    return plan


def load(c, *, device='cpu'):
    p = read(root(c)/'technical/plan.json'); original = p['source_config']
    path = cache(original)/'data.pt'
    if sha(path) != p['data_sha256']: raise ValueError('Changed native measurements')
    data = torch.load(path, map_location='cpu', weights_only=False)
    bank = load_physical(original)
    mean, scale = data['mean'].to(device), data['scale'].to(device)
    x = (data['x'].to(device)-mean)/scale
    physical = {key:(torch.stack([b[key] for b in bank]).to(device)-mean)/scale
                for key in ['history', 'pairs', 'half_pairs', 'target']}
    for key in ['response', 'initial_response']:
        physical[key] = torch.stack([b[key] for b in bank]).to(device)/scale[153:, None]
    train = np.flatnonzero([b['role'] == 'train' for b in bank])
    path = torch.cat((physical['history'][:, None, -1:, :].expand(-1, 2, -1, -1), physical['target']), 2)
    fitted = path[train]
    rho = (fitted[:, :, :-1]*fitted[:, :, 1:]).mean((0, 1, 2))/fitted[:, :, :-1].square().mean((0, 1, 2)).clamp_min(1e-5)
    physical['rate'] = -rho.clamp(.01, .9995).log()/.005*(scale > 1e-4)
    physical['increment_scale'] = (fitted[:, :, 1:]-fitted[:, :, :-1]).square().mean((0, 1, 2)).sqrt().clamp_min(.05)
    physical['normalizers'] = torch.stack([physical['response'][train, :, a:b].square().mean((0, 2)).clamp_min(1e-5)
                                         for a,b in GROUPS])
    return p, data, x, bank, physical


class HistoryTransition(nn.Module):
    def __init__(self, network, memory, summary):
        super().__init__()
        self.network, self.memory, self.summary = network, memory, summary

    def forward(self, flat):
        if self.summary:
            history = flat.reshape(*flat.shape[:-1], self.memory, 244)
            past = history[..., :-1, :]
            tokens = torch.stack((past.mean(-2), past.var(-2, unbiased=False).clamp_min(1e-10).sqrt(), history[..., -1, :]), -2)
            flat = tokens.flatten(-2)
        return self.network(flat)


class MemoryProcess(DirectProcess):
    def __init__(self, c, rate, arm):
        # Construct identical shared weights for every history treatment at a
        # paired seed. Extend only ordinal slots; no extra random body draws.
        super().__init__(c, rate)
        self.memory = arm['history']; network = self.transition
        tokens = 3 if arm['summary'] else self.memory
        slots = torch.zeros(tokens, network.slots.shape[1], device=rate.device)
        n = min(tokens, 4)
        slots[-n:] = network.slots.detach()[-n:].to(rate.device)
        network.slots = nn.Parameter(slots); network.memory = tokens
        self.transition = HistoryTransition(network, self.memory, arm['summary'])


def native_history(x, arm): return x[:, 14-arm['history']:14]


def settings(data, physical, arm, device):
    rate = physical['rate'] if arm['task'] == 'physical' else data['rate'].to(device)
    increments = physical['increment_scale'] if arm['task'] == 'physical' else data['increment_scale'].to(device)
    return rate, increments, feature_weights(data['scale'].to(device))
