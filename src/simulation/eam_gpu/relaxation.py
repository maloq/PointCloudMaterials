"""LAMMPS FIRE command adapter using the existing FastNi CUDA force callback.

Implements the -in/-log interface consumed by the shared relax_frame producer.
The physical pair commands remain in the recipe; only their execution is routed
to the pinned GPU implementation. No thermostat, barostat or MD is installed.
"""
import argparse
import json
from pathlib import Path
import shlex
import time

from src.experiment_runner.artifacts import file_hash as sha, json_digest, write_json


def implementation():
    folder = Path(__file__).parent
    files = {name: sha(folder / name) for name in
             ('relaxation.py', 'npt.py', 'potential.py', 'tables.py', 'kernels.cu')}
    import lammps
    library = Path(lammps.__file__).parent / 'liblammps.so'
    return dict(files=files, native_library_sha256=sha(library))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--implementation-sha256', required=True)
    parser.add_argument('--potential', required=True)
    parser.add_argument('-in', dest='input', required=True)
    parser.add_argument('-log', default='log.lammps')
    args = parser.parse_args()
    actual = implementation()
    if json_digest(actual) != args.implementation_sha256:
        raise ValueError('FastNi relaxation implementation or LAMMPS library changed')
    import torch
    from lammps import lammps
    from .npt import install
    started = time.monotonic()
    callback = None
    saw_pair = False
    native = lammps(cmdargs=['-log', args.log, '-screen', 'none'])
    try:
        for line in Path(args.input).read_text().splitlines():
            if line == 'pair_style eam/fs':
                saw_pair = True
                continue
            if line.startswith('pair_coeff '):
                if not saw_pair or shlex.split(line) != ['pair_coeff', '*', '*', args.potential, 'Ni']:
                    raise ValueError(f'Unexpected physical pair command: {line}')
                callback = install(native, args.potential, Path.cwd())
                continue
            if line.startswith(('run ', 'fix ')):
                raise ValueError(f'This adapter accepts relaxation/reuse only: {line}')
            if line.strip():
                native.command(line)
        if callback is None:
            raise ValueError('No Ni GPU force callback was installed')
        write_json(Path(args.input).with_suffix('.gpu.json'), dict(
            state='complete', implementation=actual, gpu=torch.cuda.get_device_name(),
            force_calls=callback.calls, seconds=time.monotonic()-started,
            force_dtype='float32', coordinates_and_minimizer_dtype='float64',
            scalar_energy_dtype='float64', backend='FastNi CUDA with native LAMMPS FIRE'))
        print(json.dumps(dict(state='complete', force_calls=callback.calls)), flush=True)
    finally:
        native.close()


if __name__ == '__main__':
    main()
