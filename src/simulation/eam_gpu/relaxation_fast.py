"""Fixed-box native FIRE with one synchronization per FastNi force callback.

Uses the existing EAM kernels and conservative neighbor displacement bound.
The graph includes the neighbor decision, conditional rebuild, forces and scalar
reductions. Pinned buffers combine transfers. This adapter accepts no dynamics
or cell-changing commands; float64 LAMMPS still owns the FIRE minimizer.
"""
import argparse
import gc
import functools
import math
import os
from pathlib import Path
import shlex
import time
import traceback

import numpy as np
import torch

from src.experiment_runner.artifacts import file_hash as sha, json_digest, write_json
from .potential import FastNi


@functools.lru_cache(maxsize=1)
def implementation():
    import lammps
    folder = Path(__file__).parent
    return dict(files={name: sha(folder/name) for name in
        ('relaxation_fast.py', 'potential.py', 'tables.py', 'kernels.cu')},
        native_library_sha256=sha(Path(lammps.__file__).parent/'liblammps.so'))


class FixedBoxCallback:
    def __init__(self, native, potential, directory):
        self.native = native; self.directory = Path(directory)
        self.atoms = int(native.get_natoms()); self.calls = 0; self.tags = None
        low, high, xy, yz, xz, periodic, _ = native.extract_box()
        if (xy, yz, xz) != (0., 0., 0.) or list(periodic) != [1, 1, 1]:
            raise ValueError('Fixed-box Ni relaxation requires an orthorhombic periodic cell')
        self.low = np.asarray(low); self.box = np.subtract(high, low)
        m = self.model = FastNi(potential, self.atoms, box=self.box, dtype=torch.float32)
        self.grid = (m.box/m.cutoff).floor().long().tolist()
        if min(self.grid) < 3:
            raise ValueError('Periodic lengths must each be >= 3*(cutoff+skin)')
        n = self.atoms; device = m.device
        integer = dict(device=device, dtype=torch.int32)
        floating = dict(device=device, dtype=torch.float32)
        # Last input scalar forces a rebuild on initial call or tag reordering.
        self.host_in = torch.empty(n*3+1, dtype=torch.float64, pin_memory=True)
        self.input_array = self.host_in.numpy()
        self.device_in = self.host_in.to(device)
        self.x = torch.zeros(n, 3, **floating); self.reference = torch.zeros_like(self.x)
        self.neighbors = torch.empty(n, 256, **integer); self.count = torch.empty(n, **integer)
        cells = math.prod(self.grid)
        self.bin_counts = torch.empty(cells, **integer); self.bins = torch.empty(cells, 128, **integer)
        self.atom_cells = torch.empty(n, 3, **integer)
        self.error = torch.zeros(1, **integer); self.rebuild = torch.zeros(1, **integer)
        self.rebuild_count = torch.zeros(1, **integer)
        self.origin = torch.zeros(3, **floating); self.fp = torch.empty(n, **floating)
        self.empty = torch.empty(0, **floating); self.f = torch.empty_like(self.x)
        self.e = torch.empty(n, device=device, dtype=torch.float64); self.w = torch.empty_like(self.e)
        self.packed = torch.empty(n*3+4, device=device, dtype=torch.float64)
        self.host_out = torch.empty_like(self.packed, device='cpu', pin_memory=True)
        self.output_array = self.host_out.numpy()
        self.graph = None

    def _evaluate(self):
        m = self.model
        self.x.copy_(self.device_in[:-1].view(self.atoms, 3))
        self.error.zero_()
        d = self.x-self.reference
        d -= m.box*torch.round(d/m.box)
        bound = m.cutoff-2*d.square().sum(-1).max().sqrt()
        self.rebuild.copy_((bound <= m.data.cutoff+1e-4) | (self.device_in[-1] != 0))
        m.native.dynamic_neighbors(self.x, self.reference, m.box, self.origin,
            *self.grid, m.atoms, m.cutoff, m.skin, self.bin_counts, self.bins,
            self.atom_cells, self.neighbors, self.count, self.rebuild, self.rebuild_count, self.error)
        m.native.evaluate(self.x, self.empty, m.box, True, self.neighbors, self.count,
            m.rho, m.embed, m.pair, m.data.dr, m.data.drho, m.data.cutoff,
            self.fp, self.empty, self.e, self.w, self.f, self.empty, self.error)
        self.packed[:-4].copy_(self.f.flatten())
        self.packed[-4].copy_(self.e.sum()); self.packed[-3].copy_(self.w.sum())
        self.packed[-2:].copy_(torch.cat((self.error, self.rebuild_count)))

    @torch.no_grad()
    def compute(self, tags, positions):
        q = self.input_array[:-1].reshape(self.atoms, 3)
        np.subtract(positions, self.low, out=q); np.remainder(q, self.box, out=q)
        if not np.isfinite(q).all():
            raise ValueError('Nonfinite Ni coordinates')
        changed = self.tags is None or not np.array_equal(self.tags, tags)
        self.input_array[-1] = changed
        if changed:
            self.tags = np.array(tags, copy=True)
        self.device_in.copy_(self.host_in, non_blocking=True)
        if self.graph is None:
            stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._evaluate()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._evaluate()
            self.rebuild_count.zero_()
        self.graph.replay()
        self.host_out.copy_(self.packed, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        error = int(self.output_array[-2])
        if error:
            raise RuntimeError(f'Ni relaxation force/neighbor error {error}; result rejected')
        self.calls += 1
        return (self.output_array[:-4].reshape(self.atoms, 3),
                float(self.output_array[-4]), float(self.output_array[-3]))

    def __call__(self, caller, step, nlocal, tags, positions, forces):
        try:
            if nlocal != self.atoms:
                raise ValueError('Exactly one LAMMPS rank is required')
            f, e, w = self.compute(tags, positions)
            forces[:] = f
            self.native.fix_external_set_energy_global('gpu_ni', e)
            self.native.fix_external_set_virial_global('gpu_ni', [w/3]*3+[0., 0., 0.])
        except BaseException:
            error = traceback.format_exc()
            write_json(self.directory/'callback-failure.json', dict(step=int(step), error=error))
            print(error, flush=True)
            os._exit(72)  # ctypes otherwise swallows callback exceptions.


def execute(input_file, log_file, potential, *, expected_sha256):
    """One full native minimization; callable repeatedly in a persistent process."""
    from lammps import lammps
    actual = implementation()
    if json_digest(actual) != expected_sha256:
        raise ValueError('Pinned optimized Ni implementation or native library changed')
    started = time.monotonic(); callback = None; saw_pair = False
    native = lammps(cmdargs=['-log', str(log_file), '-screen', 'none'])
    try:
        for line in Path(input_file).read_text().splitlines():
            if line == 'pair_style eam/fs':
                saw_pair = True; continue
            if line.startswith('pair_coeff '):
                if not saw_pair or shlex.split(line) != ['pair_coeff', '*', '*', potential, 'Ni']:
                    raise ValueError(f'Unexpected physical pair command: {line}')
                native.commands_string('pair_style zero 1.0\npair_coeff * *\n'
                    'fix gpu_ni all external pf/callback 1 1\nfix_modify gpu_ni energy yes virial yes')
                callback = FixedBoxCallback(native, potential, Path.cwd())
                native.set_fix_external_callback('gpu_ni', callback)
                continue
            # Fail closed: fixed-box quench/reuse grammar, no run/fix/box mutation.
            allowed = {'units', 'atom_style', 'boundary', 'read_data', 'read_restart',
                'mass', 'neighbor', 'neigh_modify', 'thermo', 'thermo_style',
                'thermo_modify', 'timestep', 'min_style', 'min_modify', 'minimize',
                'variable', 'print', 'reset_timestep', 'write_dump', 'write_restart'}
            if line.strip() and line.split()[0] not in allowed:
                raise ValueError(f'Not a fixed-box relaxation command: {line}')
            if line.strip(): native.command(line)
        if callback is None:
            raise ValueError('No Ni GPU force callback was installed')
        result = dict(state='complete', implementation=actual, gpu=torch.cuda.get_device_name(),
            force_calls=callback.calls, neighbor_rebuilds=int(callback.output_array[-1]) if callback.calls else 0,
            seconds=time.monotonic()-started, force_dtype='float32',
            coordinates_and_minimizer_dtype='float64', scalar_energy_dtype='float64',
            backend='FastNi fixed-box captured callback with native LAMMPS FIRE')
        write_json(Path(input_file).with_suffix('.gpu.json'), result)
        return result
    finally:
        native.close()
        if callback is not None: callback.native = None
        del callback, native
        gc.collect()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--implementation-sha256', required=True)
    parser.add_argument('--potential', required=True)
    parser.add_argument('-in', dest='input', required=True)
    parser.add_argument('-log', default='log.lammps')
    args = parser.parse_args()
    execute(args.input, args.log, args.potential, expected_sha256=args.implementation_sha256)


if __name__ == '__main__': main()
