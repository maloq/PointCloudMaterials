"""LAMMPS CPU Lennard-Jones dynamics without potential or trajectory files."""
import math
import hashlib
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .common import command_info, summarize


@dataclass
class CPUSettings:
    cells: int = 20
    warmup: int = 100
    steps: int = 500
    repeats: int = 3
    mpi_ranks: int = 1
    threads: int = 1
    seed: int = 1729
    timeout_seconds: int = 1800


def validate(settings):
    for name in vars(settings):
        if type(getattr(settings, name)) is not int or getattr(settings, name) < 1:
            raise ValueError(f"cpu.{name} must be a positive integer")
    if settings.cells < 4 or settings.seed >= 900000000:
        raise ValueError("LAMMPS requires cells >= 4 and 0 < seed < 900000000")


def input_text(settings):
    validate(settings)
    return f"""# Synthetic LJ/FCC CPU workload; not a calibrated metal potential.
units lj
atom_style atomic
boundary p p p
lattice fcc 0.8442
region box block 0 {settings.cells} 0 {settings.cells} 0 {settings.cells}
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 {settings.seed} mom yes rot no dist gaussian
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
neighbor 0.3 bin
neigh_modify delay 0 every 1 check yes
fix integrate all nve
timestep 0.005
thermo {settings.steps}
thermo_style custom step atoms temp pe ke etotal press
thermo_modify lost error flush yes
run {settings.warmup}
""" + "".join(f"run {settings.steps}\n" for _ in range(settings.repeats))


def parse_loop_times(text, settings):
    matches = re.findall(r"^Loop time of (\S+) on (\d+) procs for (\d+) steps with (\d+) atoms\s*$",
                         text, flags=re.MULTILINE)
    if len(matches) != settings.repeats + 1:
        raise RuntimeError(f"Expected {settings.repeats + 1} LAMMPS loop reports (including warmup); got {matches}")
    expected_atoms = 4 * settings.cells**3
    for index, (seconds, procs, steps, atoms) in enumerate(matches):
        expected_steps = settings.warmup if index == 0 else settings.steps
        if (int(procs), int(steps), int(atoms)) != (
                settings.mpi_ranks * settings.threads, expected_steps, expected_atoms):
            raise RuntimeError(f"LAMMPS run {index} has unexpected procs/steps/atoms: {matches[index]}; "
                               f"expected {settings.mpi_ranks * settings.threads}/{expected_steps}/{expected_atoms}")
        if not math.isfinite(float(seconds)) or float(seconds) <= 0:
            raise RuntimeError(f"Invalid LAMMPS loop time: {seconds}")
    if re.search(r"(?im)(?:^|\s)(?:nan|[+-]?inf)(?:\s|$)", text):
        raise RuntimeError("Nonfinite thermodynamic output in LAMMPS log")
    return [float(match[0]) for match in matches[1:]]


def run(settings, technical, executable, launcher):
    validate(settings)
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"LAMMPS executable {executable!r} not found; activate pointnet or pass --lammps")
    prefix = shlex.split(launcher)
    if settings.mpi_ranks != 1 and not prefix:
        raise ValueError("Multiple MPI ranks require --launcher, e.g. 'mpiexec -n 8' or 'srun -n 8'")
    source = technical / "lammps.in"
    source.write_text(input_text(settings))
    log = technical / "lammps.log"
    screen = technical / "lammps-screen.txt"
    command = prefix + [resolved, "-in", str(source), "-log", str(log)]
    if settings.threads > 1:
        command += ["-sf", "omp", "-pk", "omp", str(settings.threads)]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS=str(settings.threads), CUDA_VISIBLE_DEVICES="",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       LC_ALL="C")
    # Same conda MPI library setup as src/simulation/atomistic/lammps_shooting.py.
    environment["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib") + (
        ":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    print(f"cpu: {4 * settings.cells**3} atoms, {settings.mpi_ranks} MPI ranks x {settings.threads} threads", flush=True)
    started = time.perf_counter()
    with screen.open("w") as stream:
        try:
            # Own one process group so timeout/interrupt also stops MPI children.
            with subprocess.Popen(command, cwd=technical, env=environment, stdout=stream,
                                  stderr=subprocess.STDOUT, start_new_session=True) as process:
                try:
                    returncode = process.wait(timeout=settings.timeout_seconds)
                except BaseException:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # The process group already exited.
                    process.wait()
                    raise
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"LAMMPS exceeded {settings.timeout_seconds}s; see {screen}") from error
    wall = time.perf_counter() - started
    if returncode:
        raise RuntimeError(f"LAMMPS exited {returncode}; command={shlex.join(command)}; "
                           f"see {screen}\n{screen.read_text()[-4000:]}")
    output = log.read_text()
    samples = parse_loop_times(output, settings)
    metrics = summarize(samples, settings.steps, "steps")
    metrics["atom_steps_per_second"] = 4 * settings.cells**3 * metrics["steps_per_second"]
    metrics["process_wall_seconds"] = wall
    # Keep build details separate from the measured interval.
    help_result = subprocess.run([resolved, "-help"], env=environment, capture_output=True,
                                 text=True, timeout=30, check=True)
    (technical / "lammps-build.txt").write_text(help_result.stdout)
    return dict(metrics=metrics, atoms=4 * settings.cells**3, command=command,
                executable_sha256=hashlib.sha256(Path(resolved).read_bytes()).hexdigest(),
                build_info=help_result.stdout, launcher=prefix,
                launcher_version=command_info([prefix[0], "--version"]) if prefix else None,
                version=output.splitlines()[0], environment={key: environment.get(key) for key in (
                    "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "OMP_PROC_BIND", "OMP_PLACES")},
                protocol="lj/cut NVE; initialization and warmup excluded from loop timings")
