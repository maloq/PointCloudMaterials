"""Fixed-cell energy minimization of complete repository trajectory frames."""
import hashlib
from pathlib import Path
import subprocess
import time
import numpy as np
from src.data_utils.temporal_campaign import write_json


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def relax_frame(trajectory, frame, directory, settings):
    # Existing CG experiments retain their original minimization protocol.
    minimizer = settings.get('minimizer', 'cg')
    if minimizer not in ('cg', 'fire'):
        raise ValueError(f'Unsupported full-cell minimizer: {minimizer!r}')
    minimization = f'min_style {minimizer}\nmin_modify norm inf'
    if minimizer == 'fire':
        minimization = f"timestep {settings['timestep_ps']}\n" + minimization
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'metadata.json').exists():
        raise FileExistsError(f'Relaxation already exists: {directory}')
    low=trajectory.box_low[frame].astype(np.float64)
    high=trajectory.box_high[frame].astype(np.float64)
    positions=np.mod(trajectory.positions[frame].astype(np.float64)-low,high-low)+low
    path=directory/'input.data'
    with path.open('w') as handle:
        handle.write(f'Complete periodic frame {frame}\n\n{len(positions)} atoms\n1 atom types\n\n')
        for i,axis in enumerate('xyz'):
            handle.write(f'{low[i]:.17g} {high[i]:.17g} {axis}lo {axis}hi\n')
        handle.write('\nAtoms # atomic\n\n')
        np.savetxt(handle,np.column_stack((trajectory.atom_ids,trajectory.atom_types,positions)),
                   fmt=['%d','%d','%.17g','%.17g','%.17g'])
    commands='\n'.join(settings['pair_commands'])
    text=f'''units metal
atom_style atomic
boundary p p p
read_data {path.resolve()}
mass 1 {settings['mass']}
{commands}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
thermo 100
thermo_style custom step pe fmax fnorm
thermo_modify flush yes format float %.16g
{minimization}
minimize 0.0 {settings['force_tolerance']} {settings['max_iterations']} {settings['max_evaluations']}
variable final_force equal fmax
variable final_energy equal pe
print "RELAXED_FORCE ${{final_force}}"
print "RELAXED_ENERGY ${{final_energy}}"
reset_timestep {int(trajectory.timesteps[frame])}
write_dump all custom relaxed.dump id type x y z modify sort id format line "%d %d %.17g %.17g %.17g"
print "RELAXATION_COMPLETE"
'''
    (directory/'in.lammps').write_text(text)
    started=time.monotonic()
    command=[*settings['lammps_command'],'-in','in.lammps','-log','log.lammps']
    with (directory/'stdout.log').open('w') as log:
        subprocess.run(command,cwd=directory,stdout=log,stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL,check=True,timeout=settings['frame_timeout_seconds'])
    lines=(directory/'log.lammps').read_text().splitlines()
    force=float(next(line.split()[1] for line in reversed(lines) if line.startswith('RELAXED_FORCE ')))
    energy=float(next(line.split()[1] for line in reversed(lines) if line.startswith('RELAXED_ENERGY ')))
    if 'RELAXATION_COMPLETE' not in lines or force>settings['force_tolerance']:
        raise RuntimeError(f'Unconverged full-cell relaxation: fmax={force} eV/Å in {directory}')
    result=dict(state='relaxed',source=str(trajectory.root),source_frame=frame,
        source_timestep=int(trajectory.timesteps[frame]),source_manifest_sha256=sha256(trajectory.root/'manifest.json'),
        input_sha256=sha256(path),atom_count=trajectory.atom_count,box_low=low.tolist(),box_high=high.tolist(),
        fmax_eV_per_A=force,energy_eV=energy,seconds=time.monotonic()-started,settings=settings,
        potential_checksums={p:sha256(p) for p in settings['potential_files']},
        protocol='Full periodic cell, fixed box, generating potential; infinity-norm force convergence, no isolated-patch relaxation.')
    write_json(directory/'metadata.json',result)
    return result
