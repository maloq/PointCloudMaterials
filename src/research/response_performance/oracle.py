"""Conditional derivatives, GPU periodic graphs, and independent BAOAB replicas.

The historical reference and completed label bank are deliberately unchanged.
All new arithmetic is validated against that float64 reference by the benchmark.
"""
import math
import torch
from ase import units

from src.project_runtime.paths import resolve_path
from src.experiment_runner.artifacts import file_hash
from src.research.response_atlas.reference import BAOABOracle, ShotBundle
from src.research.response_atlas.atomistic import PathFeatures


def force_and_tangent(potential, q, dq):
    with torch.enable_grad():
        x = q.detach().requires_grad_(True)
        energy = potential(x)
        count = dq.shape[-1]
        grad, = torch.autograd.grad(energy, x, create_graph=count > 0)
        columns = []
        for j in range(count):
            hv, = torch.autograd.grad(grad, x, grad_outputs=dq[..., j],
                                      retain_graph=j + 1 < count)
            columns.append(-hv.detach())
        tangent = torch.stack(columns, -1) if columns else x.new_empty(*x.shape, 0)
    return -grad.detach(), tangent


class ConditionalOracle(BAOABOracle):
    def _force_and_tangent(self, q, dq):
        return force_and_tangent(self.potential, q, dq)


class GPUPotential:
    """Fixed MACE energy with disconnected replicas and exact GPU edge selection.

    Only orthorhombic cells with cutoff+skin below half the shortest box are
    supported. The optional Verlet list is rebuilt before any atom moves skin/2;
    retained candidates are filtered to the actual teacher cutoff every call.
    """
    def __init__(self, config, atoms, *, backend='e3nn', dtype=torch.float64, skin=0.):
        from mace.calculators import MACECalculator
        if file_hash(resolve_path(config['potential'])) != config['potential_sha256']:
            raise ValueError('Fixed teacher checkpoint changed')
        if backend not in ('e3nn', 'cueq'):
            raise ValueError(backend)
        calculator = MACECalculator(model_paths=str(resolve_path(config['potential'])),
            device='cuda', default_dtype=str(dtype).split('.')[-1],
            enable_cueq=backend == 'cueq', enable_oeq=False)
        self.model = calculator.models[0].eval().requires_grad_(False)
        self.atoms = len(atoms)
        self.box = torch.tensor(atoms.cell.lengths(), device='cuda', dtype=dtype)
        if not torch.allclose(torch.tensor(atoms.cell.array, device='cuda', dtype=dtype),
                              torch.diag(self.box)):
            raise ValueError('GPU graph requires an orthorhombic cell')
        self.cutoff = float(self.model.r_max)
        self.skin = skin
        if skin < 0 or self.cutoff + skin >= float(self.box.min())/2:
            raise ValueError('Teacher cutoff+skin must be below half every box length')
        template = calculator._atoms_to_batch(atoms).to_dict()
        self.node_attrs = template['node_attrs'].detach().to(dtype)
        self.head = template['head'].detach()
        self.static = {}
        self.candidates = None
        self.reference = None
        self.calls = 0
        self.rebuilds = 0

    def graph(self, q):
        x = q.reshape(-1, self.atoms, 3)
        batch, atoms, _ = x.shape
        with torch.no_grad():
            rebuild = self.reference is None or self.reference.shape != x.shape or self.skin == 0
            if not rebuild:
                # Unwrapped displacement is conservative for periodic crossings.
                rebuild = bool((x.detach()-self.reference).norm(dim=-1).max()*2 >= self.skin)
            if rebuild:
                delta = x.detach()[:, None, :, :] - x.detach()[:, :, None, :]
                delta -= self.box*torch.round(delta/self.box)
                allowed = delta.square().sum(-1) < (self.cutoff+self.skin)**2
                allowed &= ~torch.eye(atoms, device=x.device, dtype=torch.bool)[None]
                self.candidates = allowed.nonzero(as_tuple=True)
                self.reference = x.detach().clone()
                self.rebuilds += 1
            b, i, j = self.candidates
            raw = x.detach()[b, j]-x.detach()[b, i]
            images = -torch.round(raw/self.box)
            inside = (raw+images*self.box).square().sum(-1) < self.cutoff**2
            b, i, j, images = b[inside], i[inside], j[inside], images[inside]
            edge = torch.stack((b*atoms+i, b*atoms+j))
        if batch not in self.static:
            self.static[batch] = dict(node_attrs=self.node_attrs.repeat(batch, 1),
                head=self.head.repeat(batch), cell=torch.diag(self.box).repeat(batch, 1),
                batch=torch.arange(batch, device=x.device).repeat_interleave(atoms),
                ptr=torch.arange(batch+1, device=x.device)*atoms)
        return dict(self.static[batch], positions=x.flatten(0, 1), edge_index=edge,
                    shifts=images*self.box, unit_shifts=images)

    def __call__(self, q):
        self.calls += 1
        return self.model(self.graph(q), training=True, compute_force=False)['energy'].sum()


class BatchedFeatures:
    """The original smooth radial statistics and fixed Fourier maps, per replica."""
    def __init__(self, box, config, dtype):
        original = PathFeatures(box, config['horizons_steps'], config, 'cuda')
        self.box, self.centers = original.box.to(dtype), original.centers.to(dtype)
        self.omega = [x.to(dtype) for x in original.omega]
        self.phase = [x.to(dtype) for x in original.phase]
        self.cutoff, self.scale, self.features = original.cutoff, original.scale, original.features

    def observed(self, q):
        x = q.reshape(len(q), -1, 3)
        delta = x[:, :, None] - x[:, None, :]
        delta -= self.box*torch.round(delta/self.box)
        eye = torch.eye(x.shape[1], dtype=torch.bool, device=x.device)[None]
        distance = (delta.square().sum(-1)+eye.to(x.dtype)).sqrt()
        inside = (distance < self.cutoff) & ~eye
        r = distance/self.cutoff
        envelope = torch.where(inside, (1-r).clamp_min(0).pow(4)*(1+4*r), 0.)
        radial = torch.exp(-.5*((distance[..., None]-self.centers)/.35).square())*envelope[..., None]
        local = radial.sum(2)
        return torch.cat((local.mean(1), local.var(1, unbiased=False)), -1)

    def __call__(self, initial, path):
        anchor = self.observed(initial)
        delta = torch.stack([self.observed(path[:, h])-anchor for h in range(path.shape[1])], 1)/self.scale
        return torch.cat([math.sqrt(2/self.features)*torch.cos(
            delta[:, :h+1].flatten(1)@omega+self.phase[h]) for h,omega in enumerate(self.omega)], -1)


class BatchedOracle:
    """Same per-seed RNG draws and BAOAB order, with replica-batched forces/HVPs."""
    def __init__(self, potential, features, config):
        self.potential, self.features, self.config = potential, features, config

    def simulate(self, initial, basis, seeds):
        c = self.config
        if len(initial) != len(seeds) or basis.shape[:2] != initial.shape:
            raise ValueError('Replica inputs and seed count differ')
        generators = [torch.Generator(device=initial.device).manual_seed(int(seed)) for seed in seeds]
        # Retain the historical float64 random draws even in the separate float32
        # numerical experiment; a float32 RNG draw would change the noise stream.
        def noise():
            return torch.stack([torch.randn(initial.shape[1:], generator=g,
                device=initial.device, dtype=torch.float64) for g in generators])
        mass64 = torch.full((initial.shape[1],), 26.9815385, device=initial.device, dtype=torch.float64)
        mass = mass64.to(initial.dtype)
        kbt = units.kB*c['temperature_K']
        dt = c['timestep_fs']*units.fs
        decay = math.exp(-dt/(c['friction_time_fs']*units.fs))
        scale64 = (mass64*kbt*(1-decay*decay)).sqrt()
        q = initial.detach().clone()
        p = ((mass64*kbt).sqrt()*noise()).to(initial.dtype)
        dq, dp = basis.detach().clone(), torch.zeros_like(basis)
        force, tangent = force_and_tangent(self.potential, q, dq)
        points, tangents = [], []
        for step in range(1, max(c['horizons_steps'])+1):
            p, dp = p + dt/2*force, dp + dt/2*tangent
            q, dq = q + dt/2*p/mass, dq + dt/2*dp/mass[None, :, None]
            p, dp = decay*p + (scale64*noise()).to(initial.dtype), decay*dp
            q, dq = q + dt/2*p/mass, dq + dt/2*dp/mass[None, :, None]
            force, tangent = force_and_tangent(self.potential, q, dq)
            p, dp = p + dt/2*force, dp + dt/2*tangent
            if step in c['horizons_steps']:
                points.append(q.clone()); tangents.append(dq.clone())
        path, path_dq = torch.stack(points, 1), torch.stack(tangents, 1)
        value = self.features(initial, path)
        columns = []
        for j in range(basis.shape[-1]):
            _, column = torch.autograd.functional.jvp(self.features, (initial, path),
                                                      (basis[..., j], path_dq[..., j]))
            columns.append(column)
        responses = torch.stack(columns, -1) if columns else value.new_empty(*value.shape, 0)
        return value.detach(), responses.detach()

    def query(self, q, basis, seeds):
        count = len(seeds)
        values, responses = self.simulate(q[None].expand(count, -1),
                                          basis[None].expand(count, -1, -1), seeds)
        calls = count*(max(self.config['horizons_steps'])+1)
        bundle = ShotBundle(q.detach().clone(), basis.detach().clone(), values, responses,
                            tuple(seeds), force_calls=calls, hvp_calls=calls*basis.shape[-1])
        bundle.validate()
        return bundle
