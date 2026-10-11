"""BAOAB with paired finite differences and an explicit, shared noise tape."""
import math
import time
import numpy as np
import torch
from ase import units

from src.research.local_response.oracle import LocalFeatures


class FrozenFeatures(LocalFeatures):
    """Reuse the actual local observable, with frozen CUDA-generated RFF weights."""
    def __init__(self, payload):
        self.cutoff, self.scale = payload['cutoff'], payload['scale']
        self.centers = payload['centers'].double()
        self.omega = [x.double() for x in payload['omega']]
        self.phase = [x.double() for x in payload['phase']]


class ForceOracle:
    def __init__(self, engine, features, config):
        self.engine, self.features, self.config = engine, features, config

    def path_value(self, initial, noise):
        c = self.config
        steps = max(c['horizons_steps'])
        if noise.shape != (steps + 1, *initial.shape):
            raise ValueError(f'Noise shape {noise.shape} does not match {initial.shape}')
        mass = 26.9815385
        dt = c['timestep_fs'] * units.fs
        decay = math.exp(-c['timestep_fs'] / c['friction_time_fs'])
        kbt = units.kB * c['temperature_K']
        q = initial.copy()
        p = math.sqrt(mass * kbt) * noise[0]
        _, force = self.engine(q)
        points = []
        for step in range(1, steps + 1):
            p += dt / 2 * force
            q += dt / 2 * p / mass
            p = decay * p + math.sqrt(mass * kbt * (1 - decay * decay)) * noise[step]
            q += dt / 2 * p / mass
            _, force = self.engine(q)
            p += dt / 2 * force
            if step in c['horizons_steps']:
                points.append(q.copy())
        path = torch.from_numpy(np.stack(points)).reshape(1, len(points), -1)
        with torch.no_grad():
            # Each displaced path uses its OWN displaced initial anchor.
            value = self.features(torch.from_numpy(initial).reshape(1, -1), path)[0]
        if not torch.isfinite(value).all() or not np.isfinite(q).all():
            raise FloatingPointError('Nonfinite force-only trajectory/observable')
        return value

    def query(self, q, basis, noise, epsilons):
        begin = time.monotonic()
        total_begin = begin
        value = self.path_value(q, noise)
        value_seconds = time.monotonic() - begin
        responses, timings = {}, {}
        for eps in epsilons:
            begin = time.monotonic()
            columns = []
            for j in range(basis.shape[-1]):
                plus = self.path_value(q + eps * basis[..., j], noise)
                minus = self.path_value(q - eps * basis[..., j], noise)
                columns.append((plus - minus) / (2 * eps))
            responses[str(eps)] = torch.stack(columns, -1)
            timings[str(eps)] = value_seconds + time.monotonic() - begin
        return dict(values=value, responses=responses, value_seconds=value_seconds,
                    bundle_seconds=timings, total_seconds=time.monotonic() - total_begin)
