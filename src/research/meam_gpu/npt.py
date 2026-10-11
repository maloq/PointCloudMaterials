"""GPU MEAM forces and diagonal virials inside native LAMMPS isotropic NPT.

Only orthorhombic isotropic NPT is supported. The three normal virials are
analytic strain derivatives; shear components are not exposed by this adapter.
The original fixed-cell force/HVP implementation is unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from lammps import lammps

from src.project_runtime.transfer import write_json
from .energy import energy
from .potential import FastAl


class IsotropicAl(FastAl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normal_strain = torch.ones(3, device=self.table.device, dtype=self.table.dtype)
        self.compiled_stress = torch.compile(
            torch.func.grad_and_value(self.strained_energy, argnums=(0, 1)),
            dynamic=True, fullgraph=True)

    def strained_energy(self, q, strain, support):
        return energy(self, q, support, strain)

    def force_normal_virial(self, q, box):
        self.periodic_box.copy_(torch.as_tensor(box, device=q.device, dtype=self.table.dtype))
        # Box deformation and LAMMPS atom sorting both invalidate the list.
        # Rebuilding every callback is conservative and measured in wall time.
        self.topology.reference = None
        (gradient, stress), value = self.compiled_stress(q, self.normal_strain, self.support(q))
        return value.detach(), -gradient.detach(), -stress.detach()


class Callback:
    def __init__(self, native, directory):
        self.native = native
        self.directory = Path(directory)
        self.atoms = int(native.get_natoms())
        low, high, xy, yz, xz, periodic, _ = native.extract_box()
        if (xy, yz, xz) != (0., 0., 0.) or list(periodic) != [1, 1, 1]:
            raise ValueError('GPU dense adapter requires an orthorhombic periodic cell')
        files = dict(library=self.directory/'potential/Lee2003_Al.library.meam',
                     parameters=self.directory/'potential/Lee2003_Al.meam')
        self.model = IsotropicAl(files, self.atoms, dtype=torch.float32,
                                 box=np.subtract(high, low), compile_tangents=False)
        self.calls = 0
        self.started = time.monotonic()

    def __call__(self, caller, step, nlocal, tags, positions, forces):
        try:
            if nlocal != self.atoms:
                raise ValueError('Use exactly one LAMMPS rank per GPU cell')
            low, high, xy, yz, xz, _, _ = self.native.extract_box()
            if (xy, yz, xz) != (0., 0., 0.):
                raise ValueError('A tilted cell is unsupported')
            box = np.subtract(high, low)
            q = torch.as_tensor(np.remainder(positions-low, box), device='cuda', dtype=torch.float64)
            value, force, normal = self.model.force_normal_virial(q, box)
            forces[:] = force.cpu().numpy()
            self.native.fix_external_set_energy_global('gpu_meam', float(value))
            # The unchanged iso barostat uses trace/3; no shear observable is requested.
            self.native.fix_external_set_virial_global('gpu_meam', normal.tolist()+[0., 0., 0.])
            self.calls += 1
            if self.calls % 500 == 0:
                write_json(self.directory/'technical/gpu-progress.json', dict(
                    step=int(step), calls=self.calls, elapsed_seconds=time.monotonic()-self.started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    box_A=box.tolist(), device=torch.cuda.get_device_name()))
        except BaseException:
            # ctypes otherwise swallows Python callback exceptions and continues MD.
            failure = traceback.format_exc()
            write_json(self.directory/'technical/gpu-callback-failure.json',
                       dict(step=int(step), error=failure))
            print(failure, flush=True)
            os._exit(72)


def install(native, directory):
    native.commands_string('pair_style zero 1.0\npair_coeff * *\nfix gpu_meam all external pf/callback 1 1\nfix_modify gpu_meam energy yes virial yes')
    callback = Callback(native, directory)
    native.set_fix_external_callback('gpu_meam', callback)
    return callback


def run(directory):
    directory = Path(directory).resolve()
    os.chdir(directory)
    original = (directory/'source.in.lammps').read_text()
    if 'fix ensemble all npt' not in original or ' iso 0 0 3\n' not in original:
        raise ValueError('Only the declared original isotropic NPT input is supported')
    head, tail = original.split('pair_style meam\n', 1)
    pair_line, tail = tail.split('\n', 1)
    if pair_line != 'pair_coeff * * potential/Lee2003_Al.library.meam Al potential/Lee2003_Al.meam Al':
        raise ValueError('Unexpected potential mapping')
    native = lammps(cmdargs=['-log', 'none'])
    native.commands_string(head)
    callback = install(native, directory)
    # A real input file retains jump SELF semantics and all original run/fix commands.
    source = directory/'gpu-tail.in.lammps'
    source.write_text(tail)
    native.file(str(source))
    write_json(directory/'technical/gpu-execution.json', dict(
        state='complete', lammps_version=native.version(), calls=callback.calls,
        seconds=time.monotonic()-callback.started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), device=torch.cuda.get_device_name(),
        force_dtype='float32', integration_dtype='float64', pressure='analytic normal-strain derivatives',
        shear_stress='not computed; adapter restricted to isotropic NPT'))
    native.close()


def validate(directory, output, steps):
    """Local full-cell execution audit; never a training or independent MD sample."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((directory/'input_metadata.json').read_text())
    base = f'''units metal
atom_style atomic
boundary p p p
read_data {directory}/prepared_liquid.lammps.data
mass 1 26.9815
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep 0.002
velocity all create {metadata['temperature_K']} {metadata['velocity_seed']} mom yes rot no dist gaussian loop all
'''
    controls = f'''fix remove_drift all momentum 150 linear 1 1 1
fix ensemble all npt temp {metadata['temperature_K']} {metadata['temperature_K']} 0.3 iso 0 0 3
thermo 100
thermo_style custom step temp press vol pe pxx pyy pzz
run 0
'''
    def observe(l):
        ids = l.numpy.extract_atom('id')[:70304].copy()
        order = np.argsort(ids)
        return dict(q=l.numpy.extract_atom('x')[:70304][order].copy(),
                    f=l.numpy.extract_atom('f')[:70304][order].copy(),
                    thermo=np.array([l.get_thermo(k) for k in ['temp','press','vol','pe','pxx','pyy','pzz']]))
    results = {}
    for mode in ['native', 'gpu']:
        l = lammps(cmdargs=['-screen', str(output/f'{mode}.log'), '-log', 'none'])
        l.commands_string(base)
        if mode == 'native':
            l.commands_string(f'pair_style meam\npair_coeff * * {directory}/potential/Lee2003_Al.library.meam Al {directory}/potential/Lee2003_Al.meam Al')
        else:
            callback = install(l, directory)
            callback.directory = output
        l.commands_string(controls)
        before = observe(l)
        start = time.monotonic()
        l.command(f'run {steps}')
        seconds = time.monotonic()-start
        after = observe(l)
        results[mode] = dict(before=before, after=after, seconds=seconds)
        np.savez(output/f'{mode}.npz', **{f'{phase}_{key}': val for phase in ['before','after'] for key,val in results[mode][phase].items()})
        l.close()
    a, b = results['native'], results['gpu']
    box = b['after']['thermo'][2]**(1/3)
    displacement = a['after']['q']-b['after']['q']
    displacement -= box*np.rint(displacement/box)
    metrics = dict(force_rms_eV_A=float(np.sqrt(np.mean((a['before']['f']-b['before']['f'])**2))),
        energy_error_eV_atom=float(abs(a['before']['thermo'][3]-b['before']['thermo'][3])/70304),
        normal_pressure_max_error_bar=float(np.max(abs(a['before']['thermo'][4:]-b['before']['thermo'][4:]))),
        final_position_rms_A=float(np.sqrt(np.mean(displacement**2))),
        final_volume_relative_error=float(abs(a['after']['thermo'][2]/b['after']['thermo'][2]-1)),
        native_seconds=a['seconds'], gpu_seconds=b['seconds'])
    limits = dict(force_rms_eV_A=5e-4, energy_error_eV_atom=2e-5,
        normal_pressure_max_error_bar=2., final_position_rms_A=1e-3, final_volume_relative_error=1e-5)
    passed = all(metrics[k] <= v for k,v in limits.items())
    receipt = dict(state='passed' if passed else 'failed', atoms=70304, steps=steps,
        duration_ps=steps*.002, source_id=metadata['source_id'], metrics=metrics, limits=limits,
        purpose='Full-cell isotropic-NPT numerical adapter audit, excluded from scientific sample counts')
    write_json(output/'validation.json', receipt)
    print(json.dumps(receipt), flush=True)
    if not passed:
        raise ValueError('GPU NPT adapter failed numerical gates')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['run','validate'])
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--steps', type=int, default=100)
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.action == 'run': run(a.directory)
    else: validate(a.directory, a.output, a.steps)


if __name__ == '__main__':
    main()
