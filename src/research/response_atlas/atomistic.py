"""Complete-cell, fixed-MACE ASE-unit BAOAB responses; numerical pilot only."""
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.build import bulk

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path
from .common import output, read, table
from .reference import BAOABOracle, ShotBundle, unbiased_gram


def configuration(c, index):
    atoms = bulk('Al', 'fcc', a=c['lattice_A'], cubic=True).repeat((4, 4, 4))
    rng = np.random.default_rng(c['seed'] + index)
    sigma = c['parent_displacement_sd_A'][index % len(c['parent_displacement_sd_A'])]
    displacement = rng.normal(0, sigma, (len(atoms), 3))
    displacement -= displacement.mean(0)
    atoms.positions += displacement
    atoms.wrap()
    return atoms


class MACEPotential:
    """Rebuild exact periodic graph every call, then attach the live coordinates."""
    def __init__(self, c, atoms, device):
        from mace.calculators import MACECalculator
        if sha(resolve_path(c['potential'])) != c['potential_sha256']:
            raise ValueError('Fixed potential hash changed')
        self.atoms = atoms.copy()
        self.calculator = MACECalculator(model_paths=str(resolve_path(c['potential'])),
            device=device, default_dtype='float64', enable_cueq=False, enable_oeq=False)
        if len(self.calculator.models) != 1:
            raise ValueError('Response oracle requires one fixed potential')
        self.model = self.calculator.models[0].eval()
        self.model.requires_grad_(False)
        self.calls = 0

    def __call__(self, q):
        self.atoms.positions = q.detach().reshape(-1, 3).cpu().numpy()
        graph = self.calculator._atoms_to_batch(self.atoms).to_dict()
        # Detached positions above determine discrete graph membership only.
        # Energy, vectors and periodic shifts consume this live tensor.
        graph['positions'] = q.reshape(-1, 3)
        self.calls += 1
        return self.model(graph, training=True, compute_force=False)['energy'].sum()


class PathFeatures:
    """Smooth periodic radial mean/variance; fixed RFF of joint change prefixes."""
    def __init__(self, box, horizons, c, device):
        self.box = torch.as_tensor(box, dtype=torch.float64, device=device)
        self.horizons = horizons
        self.cutoff = c['feature_cutoff_A']
        if self.cutoff >= float(self.box.min()) / 2:
            raise ValueError('Feature cutoff must lie below half every box length')
        self.centers = torch.linspace(2., 5., 8, dtype=torch.float64, device=device)
        self.features = c['rff_features']
        self.scale = c['path_coordinate_scale']
        gen = torch.Generator(device=device).manual_seed(c['seed'] + 500)
        self.omega, self.phase = [], []
        for h in range(len(horizons)):
            self.omega.append(torch.randn(16 * (h + 1), self.features, generator=gen, dtype=torch.float64, device=device)
                              / math.sqrt(16 * (h + 1)))
            self.phase.append(2 * math.pi * torch.rand(self.features, generator=gen, dtype=torch.float64, device=device))

    def observed(self, q):
        x = q.reshape(-1, 3)
        delta = x[:, None] - x[None]
        delta = delta - self.box * torch.round(delta / self.box)
        eye = torch.eye(len(x), dtype=torch.bool, device=x.device)
        # Avoid a norm derivative at diagonal zero; exclude self edges explicitly.
        distance = (delta.square().sum(-1) + eye.to(x.dtype)).sqrt()
        inside = (distance < self.cutoff) & ~eye
        r = distance / self.cutoff
        envelope = torch.where(inside, (1 - r).clamp_min(0).pow(4) * (1 + 4 * r), 0.)
        radial = torch.exp(-.5 * ((distance[..., None] - self.centers) / .35).square()) * envelope[..., None]
        local = radial.sum(1)
        return torch.cat((local.mean(0), local.var(0, unbiased=False)))

    def __call__(self, initial, path):
        anchor = self.observed(initial)
        delta = torch.stack([self.observed(q) - anchor for q in path]) / self.scale
        return torch.cat([math.sqrt(2 / self.features) * torch.cos(delta[:h + 1].flatten() @ self.omega[h] + self.phase[h])
                          for h in range(len(self.horizons))])


def basis_for(q, directions, seed):
    gen = torch.Generator(device=q.device).manual_seed(seed)
    x = torch.randn(len(q) // 3, 3, directions, dtype=q.dtype, device=q.device, generator=gen)
    x -= x.mean(0)
    return torch.linalg.qr(x.reshape(len(q), directions), mode='reduced').Q


def oracle_for(c, potential, features, q, horizons):
    # ASE internal time = sqrt(amu A^2 / eV); units.fs converts physical fs.
    # This explicitly supplies a consistent unit system to the reference core.
    return BAOABOracle(potential, features, mass=torch.full_like(q, 26.9815385),
        dt=c['timestep_fs'] * units.fs, steps=max(horizons), kbt=units.kB * c['temperature_K'],
        friction=1 / (c['friction_time_fs'] * units.fs), horizon_steps=horizons)


def admissible(q, box, minimum):
    x = q.detach().reshape(-1, 3)
    delta = x[:, None] - x[None]
    delta -= box * torch.round(delta / box)
    distance = delta.square().sum(-1).sqrt()
    distance.fill_diagonal_(math.inf)
    if float(distance.min()) < minimum:
        raise ValueError(f'Overlapping pilot configuration: {float(distance.min())} A')


def physical_screen(potential, q):
    point = q.detach().requires_grad_(True)
    energy = potential(point)
    force, = torch.autograd.grad(-energy, point)
    per_atom = float(energy.detach()) / (len(q) // 3)
    peak = float(force.detach().reshape(-1, 3).norm(dim=1).max())
    if not -5 < per_atom < 0 or not math.isfinite(peak) or peak > 20:
        raise ValueError(f'Pilot state outside declared energy/force bounds: {per_atom} eV/atom, {peak} eV/A')
    return dict(energy_eV_per_atom=per_atom, maximum_force_eV_per_A=peak,
                limits='-5 < energy/atom < 0 eV; force norm <= 20 eV/A; numerical screen only')


def gate(c, device='cuda'):
    from .common import protocol
    torch.set_num_threads(2)
    atoms = configuration(c, 0)
    q = torch.tensor(atoms.positions.flatten(), dtype=torch.float64, device=device, requires_grad=True)
    box = torch.tensor(atoms.cell.lengths(), dtype=q.dtype, device=device)
    admissible(q, box, c['minimum_separation_A'])
    potential = MACEPotential(c, atoms, device)
    began = time.monotonic()
    energy = potential(q)
    force, = torch.autograd.grad(-energy, q, create_graph=True)
    atoms.calc = potential.calculator
    ref_energy = atoms.get_potential_energy(); ref_force = atoms.get_forces()
    energy_error = abs(float(energy.detach()) - ref_energy)
    force_error = float(np.max(np.abs(force.detach().cpu().numpy().reshape(-1, 3) - ref_force)))
    if energy_error > 1e-7 or force_error > 1e-7:
        raise ValueError(f'Energy/force disagreement with calculator: {energy_error}/{force_error}')
    basis = basis_for(q, 2, c['seed'])
    df = torch.stack([torch.autograd.grad(force, q, grad_outputs=basis[:, j], retain_graph=True)[0]
                      for j in range(2)], -1).detach()
    rows = []
    for epsilon in c['gate_epsilon_A']:
        for j in range(2):
            forces = []
            for sign in (-1, 1):
                point = (q.detach() + sign * epsilon * basis[:, j]).requires_grad_(True)
                f, = torch.autograd.grad(-potential(point), point)
                forces.append(f.detach())
            fd = (forces[1] - forces[0]) / (2 * epsilon)
            relative = float((fd - df[:, j]).norm() / df[:, j].norm().clamp_min(1e-12))
            rows.append(dict(kind='force_hvp', epsilon_A=epsilon, direction=j, relative_error=relative))
    if any(min(r['relative_error'] for r in rows if r['direction'] == j) > c['hvp_relative_tolerance'] for j in range(2)):
        raise ValueError(f'No converged HVP finite-difference window: {rows}')
    # Whole-cell lattice-vector translations exercise rebuilt periodic images.
    translated = q.detach().reshape(-1, 3).clone(); translated[0] += box
    translated = translated.flatten().requires_grad_(True)
    translated_energy = potential(translated)
    translated_force, = torch.autograd.grad(-translated_energy, translated)
    periodic_error = float((translated_force - force.detach()).abs().max())
    if periodic_error > 1e-7 or abs(float(translated_energy.detach() - energy.detach())) > 1e-7:
        raise ValueError('Periodic image/neighbor rebuild changes physical force')
    hs = [2, 5]
    features = PathFeatures(atoms.cell.lengths(), hs, c, device)
    oracle = oracle_for(c, potential, features, q, hs)
    tangent = oracle.query(q.detach(), basis, [8_000_000, 8_000_001])
    for epsilon in c['gate_epsilon_A']:
        for j in range(2):
            values = [oracle.query(q.detach() + s * epsilon * basis[:, j], q.new_empty(len(q), 0), tangent.seeds).values
                      for s in (-1, 1)]
            fd = (values[1] - values[0]) / (2 * epsilon)
            relative = float((fd - tangent.responses[:, :, j]).norm() / tangent.responses[:, :, j].norm().clamp_min(1e-12))
            rows.append(dict(kind='path_response', epsilon_A=epsilon, direction=j, relative_error=relative))
    for j in range(2):
        if min(r['relative_error'] for r in rows if r['kind'] == 'path_response' and r['direction'] == j) > c['path_relative_tolerance']:
            raise ValueError(f'No reliable path-response finite-difference window: {rows}')
    receipt = dict(passed=True, binding=digest(protocol(c)), energy_abs_error=energy_error, force_max_abs_error=force_error,
        periodic_force_max_abs_error=periodic_error, checks=rows, seconds=time.monotonic() - began,
        force_calls=potential.calls, potential_sha256=c['potential_sha256'], precision='float64', backend='eager torch, no CuEq',
        units=dict(position='angstrom', energy='eV', mass='amu', internal_time='ASE sqrt(amu*A^2/eV)', fs_conversion=units.fs),
        feature_anchor='differentiated explicitly', neighbor_policy='rebuild all periodic edges every energy call')
    write_json(output(c) / 'technical/atomistic-gate.json', receipt)
    return receipt


def run(c, device='cuda'):
    from .common import protocol
    gate_record = read(output(c) / 'technical/atomistic-gate.json')
    if not gate_record['passed'] or gate_record['potential_sha256'] != c['potential_sha256'] or gate_record['binding'] != digest(protocol(c)):
        raise ValueError('Current fixed-potential gate missing')
    scratch = resolve_path(c['simulation_scratch']); scratch.mkdir(parents=True, exist_ok=True)
    archive = resolve_path(c['simulation_archive']); archive.mkdir(parents=True, exist_ok=True)
    binding = digest(protocol(c))
    rows, fd_rows = [], []
    indices = c.get('parent_indices', list(range(c['pilot_parents'])))
    if len(indices) != c['pilot_parents'] or len(set(indices)) != len(indices):
        raise ValueError('Parent indices must be unique and match pilot_parents')
    for index in indices:
        root = scratch / f'parent-{index:03d}'; root.mkdir(exist_ok=True)
        try:
            receipt_path = archive / root.name / 'complete.json'
            if receipt_path.exists():
                done = read(receipt_path)
                if done['binding'] != binding or sha(receipt_path.parent / 'query.pt') != done['query_sha256']:
                    raise ValueError('Changed completed pilot parent')
                rows.extend(done['rows']); fd_rows.extend(done['fd_rows']); continue
            atoms = configuration(c, index)
            q = torch.tensor(atoms.positions.flatten(), dtype=torch.float64, device=device)
            box = torch.tensor(atoms.cell.lengths(), dtype=q.dtype, device=device)
            admissible(q, box, c['minimum_separation_A'])
            potential = MACEPotential(c, atoms, device)
            parent_screen = physical_screen(potential, q)
            basis = basis_for(q, c['directions'], c['seed'] + 100 + index)
            horizons = c['horizons_steps'] if index < c['long_horizon_parents'] else c['horizons_steps'][:2]
            features = PathFeatures(atoms.cell.lengths(), horizons, c, device)
            oracle = oracle_for(c, potential, features, q, horizons)
            torch.save(dict(q=q.cpu(), box=box.cpu(), basis=basis.cpu(), horizons_steps=horizons,
                            config=c, binding=binding, screen=parent_screen,
                            ancestor='synthetic-FCC-256-prototype'), root / 'parent-restart.pt')
            (archive / root.name).mkdir(exist_ok=True)
            shutil.copy2(root / 'parent-restart.pt', archive / root.name / 'parent-restart.pt')
            torch.cuda.reset_peak_memory_stats()
            began = time.monotonic()
            base_seed = c.get('branch_seed_base', 10_000_000) + index * 1000
            seeds = tuple(range(base_seed, base_seed + c['discovery_branches']))
            # Save each query branch before proceeding; failures retain completed evidence.
            pairs = []
            discovery_seconds = 0.
            for seed in seeds:
                path = root / f'branch-{seed}.pt'
                if path.exists():
                    saved = torch.load(path, map_location=device, weights_only=False)
                    if saved['binding'] != binding:
                        raise ValueError('Changed cached response branch')
                    bundle = ShotBundle(q, basis, saved['values'], saved['responses'], (seed,), saved['force_calls'], saved['hvp_calls'])
                    bundle.validate()
                else:
                    before = time.monotonic()
                    bundle = oracle.query(q, basis, [seed])
                    saved = dict(binding=binding, seed=seed, values=bundle.values.cpu(), responses=bundle.responses.cpu(),
                                 force_calls=bundle.force_calls, hvp_calls=bundle.hvp_calls, seconds=time.monotonic() - before)
                    torch.save(saved, path)
                discovery_seconds += saved['seconds']
                pairs.append(bundle)
                (archive / root.name).mkdir(exist_ok=True)
                shutil.copy2(path, archive / root.name / path.name)
                write_json(root / 'progress.json', dict(parent=index, completed_discovery=len(pairs), seconds=time.monotonic() - began))
            values = torch.cat([b.values for b in pairs]); responses = torch.cat([b.responses for b in pairs])
            parent_rows, parent_fd_rows = [], []
            verification = []
            # Fresh CRN finite differences, independent of discovery. Only first two
            # pilot parents receive this costly horizon/epsilon audit; all are saved.
            if index < c['verification_parents']:
                fresh = tuple(range(base_seed + 100, base_seed + 100 + c['verification_branches']))
                for epsilon in c['verification_epsilon_A']:
                    for j in range(c['directions']):
                        points = [q + s * epsilon * basis[:, j] for s in (-1, 1)]
                        for point in points:
                            admissible(point, box, c['minimum_separation_A'])
                            physical_screen(potential, point)
                        differences = []; pair_seconds = 0.; pair_calls = 0
                        for seed in fresh:
                            path = root / f'verify-{epsilon:g}-{j}-{seed}.pt'
                            if path.exists():
                                saved = torch.load(path, map_location=device, weights_only=False)
                                if saved['binding'] != binding:
                                    raise ValueError('Changed cached verification pair')
                            else:
                                before = time.monotonic()
                                minus, plus = [oracle.query(point, q.new_empty(len(q), 0), [seed]) for point in points]
                                saved = dict(binding=binding, response=((plus.values - minus.values) / (2 * epsilon)).cpu(),
                                    seconds=time.monotonic() - before, force_calls=minus.force_calls + plus.force_calls)
                                torch.save(saved, path)
                            differences.append(saved['response'].to(q))
                            pair_seconds += saved['seconds']; pair_calls += saved['force_calls']
                            shutil.copy2(path, archive / root.name / path.name)
                            write_json(root / 'progress.json', dict(parent=index, verification_direction=j,
                                epsilon_A=epsilon, completed_pairs=len(differences), total_pairs=len(fresh)))
                        fd = torch.cat(differences)
                        verification.append(dict(epsilon_A=epsilon, direction=j, seeds=fresh, response=fd.cpu(),
                                                 seconds=pair_seconds, force_calls=pair_calls))
            for h, steps in enumerate(horizons):
                H = responses[:, h * c['rff_features']:(h + 1) * c['rff_features']]
                signal = float(torch.trace(unbiased_gram(H)))
                noise = float(H.var(0, unbiased=True).sum())
                parent_rows.append(dict(parent=index, horizon_steps=steps, horizon_fs=steps * c['timestep_fs'],
                    mean_response_squared_corrected=signal, branch_response_variance=noise,
                    relative_mc_noise=noise / len(seeds) / max(abs(signal), 1e-12),
                    discovery_seconds=discovery_seconds, force_calls=sum(b.force_calls for b in pairs),
                    hvp_calls=sum(b.hvp_calls for b in pairs), verification_force_calls=sum(v['force_calls'] for v in verification),
                    verification_seconds=sum(v['seconds'] for v in verification), peak_memory_bytes=torch.cuda.max_memory_allocated()))
                for v in verification:
                    fd = v['response'][:, h * c['rff_features']:(h + 1) * c['rff_features']].to(q)
                    ad = H[:, :, v['direction']]
                    difference = float((ad.mean(0) - fd.mean(0)).square().sum())
                    corrected = difference - float(ad.var(0, unbiased=True).sum() / len(ad) + fd.var(0, unbiased=True).sum() / len(fd))
                    parent_fd_rows.append(dict(parent=index, horizon_steps=steps, direction=v['direction'], epsilon_A=v['epsilon_A'],
                        mean_difference_squared=difference, mean_difference_squared_corrected=corrected,
                        fd_mean_response_squared_corrected=float(unbiased_gram(fd[:, :, None])[0, 0])))
            torch.save(dict(q=q.cpu(), box=box.cpu(), species='Al', mass_amu=26.9815385, basis=basis.cpu(),
                values=values.cpu(), responses=responses.cpu(), discovery_seeds=seeds, verification=verification,
                horizons_steps=horizons,
                parent_id=root.name, ancestor_id='synthetic-FCC-256-prototype', split='development-only',
                role='oracle-feasibility', binding=binding, config=c, coordinate_precision='float64 restart/query state',
                rng='torch Philox CUDA generator; identical draw order within +/-',
                noise_coupling='explicit generator per branch; layout fixed 256x3; no MPI'), root / 'query.pt')
            write_json(root / 'complete.json', dict(binding=binding, query_sha256=sha(root / 'query.pt'), rows=parent_rows, fd_rows=parent_fd_rows,
                                                   seconds=time.monotonic() - began))
            rows.extend(parent_rows)
            fd_rows.extend(parent_fd_rows)
        except BaseException as error:
            write_json(root / 'failure.json', dict(error=repr(error), binding=binding))
            raise
        finally:
            # Copy successful and stopped-failure artifacts to durable storage;
            # no integration or query state is reduced to float16.
            if list(root.iterdir()):
                shutil.copytree(root, archive / root.name, dirs_exist_ok=True)
        table(c, 'atomistic-v1', 'horizon-noise-cost', rows)
        table(c, 'atomistic-v1', 'independent-fd-comparison', fd_rows)
        write_json(output(c) / 'technical/atomistic-progress.json', dict(completed_parents=index + 1, total=c['pilot_parents']))
    table(c, 'atomistic-v1', 'horizon-noise-cost', rows)
    table(c, 'atomistic-v1', 'independent-fd-comparison', fd_rows)
    write_json(output(c) / 'analyses/atomistic-v1/technical/complete.json', dict(parents=c['pilot_parents'],
        scope='oracle feasibility on perturbed FCC; no atomistic student training or equilibrium generalization',
        later_campaign='five-arm learning and MEAM bridge remain gated on scientific review'))
