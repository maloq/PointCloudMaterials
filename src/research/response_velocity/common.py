"""Immutable scientific inputs and implementation for the velocity comparison."""
from pathlib import Path
import torch
from src.experiment_runner.execution import allocation_deadline
from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read, save, sha, digest, write_json

FAMILY = 'response_velocity'


def root(c):
    return resolve_path(c['output'])


def deadline(c, *, reserve_seconds):
    end = allocation_deadline(reserve_seconds=reserve_seconds)
    if 'execution' in c:
        end = min(end, read(root(c)/'technical/budget.json')['deadline_unix']-reserve_seconds)
    return end


def dependencies():
    return ['src/research/response_meam/potential.py',
            'src/research/response_performance/oracle.py',
            'src/research/response_geometry/features.py',
            'src/research/response_geometry/train.py',
            'src/research/response_geometry/clusters.py',
            'src/research/local_response/model.py',
            'src/research/supervised_onset/model.py',
            'src/models/encoders/spatial_mace.py',
            'docs/metrics/response_velocity.md']


def bind(c):
    repo = Path(__file__).resolve().parents[3]
    reference = resolve_path(c['reference'])
    old = read(reference/'technical/binding.json')
    if old['identity'] != c['reference_identity']:
        raise ValueError('Original response input identity changed')
    benchmark = resolve_path(c['meam_benchmark'])
    if not read(benchmark/'technical/validation-complete.json')['passed']:
        raise ValueError('MEAM AD benchmark has not passed')
    benchmark_binding = read(benchmark/'technical/binding.json')
    potential = 'src/research/response_meam/potential.py'
    if sha(repo/potential) != benchmark_binding['implementation'][potential]:
        raise ValueError('MEAM implementation differs from validated version')
    files = list(Path(__file__).parent.glob('*.py'))+[repo/p for p in dependencies()]
    backend_evidence=None
    if c.get('teacher_backend')=='cuda_meam':
        benchmark_root=resolve_path(c['fast_meam_reference'])
        for name,expected in c['fast_meam_receipts'].items():
            if sha(benchmark_root/name)!=expected:raise ValueError('Fast MEAM evidence changed: '+name)
        contract=read(benchmark_root/'analyses/fast-response-comparison-v1/technical/metric-contracts/meam_gpu_response_speed.json')
        for name in ['potential.py','energy.py','topology.cu']:
            relative='src/research/meam_gpu/'+name
            if sha(repo/relative)!=contract['files'][relative]:raise ValueError('Fast MEAM implementation changed: '+relative)
            files.append(repo/relative)
        backend_evidence=dict(root=str(benchmark_root),receipts=c['fast_meam_receipts'],
            validation='Existing completed CUDA MEAM response and periodic-MD evidence; no additional MEAM validation')
    geometry = resolve_path(c['geometry_reference'])/'technical'
    if read(geometry/'binding.json')['identity'] != c['geometry_reference_identity']:
        raise ValueError('Observable-normalization ancestry changed')
    record = dict(config=c, implementation={str(p.relative_to(repo)):sha(p) for p in files},
                  parents_sha256=sha(reference/'technical/parents.pt'),
                  observable_normalizer_sha256=sha(geometry/'feature-normalization.pt'),
                  benchmark_identity=benchmark_binding['identity'])
    if backend_evidence is not None:record['fast_meam_evidence']=backend_evidence
    record['identity'] = digest(record)
    path = root(c)/'technical/binding.json'
    if path.exists() and read(path) != record:
        raise ValueError('Frozen velocity experiment changed; use a new revision')
    write_json(path, record)
    native=c.get('initial_velocity')=='saved_md'
    write_json(root(c)/'technical/prediction-context.json', dict(
        encoder_inputs=dict(positions='center-relative unrelaxed nearest80, support8A, edges5A',
            velocities=('QV arms only: actual same-frame MD velocities relative to central atom, fixed scale0.004 A/fs' if native else
                'QV arms only: sampled initial velocities relative to central atom, fixed scale0.004 A/fs'),
            velocity_authorization='User 2026-10-04: add V with longer observations',
            channels=128, embedding=128, constant_atom_channel=True, center_indicator=True,
            history=0, halo=False, conditions=[], relaxation=False),
        predictor_inputs=dict(embedding='z128 only', conditions=[], history=0),
        teacher=dict(potential='Lee2003 Al MEAM, same potential as parent MD',
            backend=c.get('teacher_backend','reference'),force_dtype=c.get('force_dtype','float64'),
            integration_and_observables_dtype=c.get('state_dtype','float64'),
            energy_reduction_dtype='float64',
            temperature_K='per-source original setpoint, physical simulator only' if c.get('temperature_mode')=='source' else c['temperature_K'],
            dt_fs=c['timestep_fs'], friction_fs=c['friction_fs'], horizons_fs=c['horizons_fs'],
            initial_velocity='one actual same-frame full-cell MD field; no redraw/rescale/COM removal' if native else f"{c['velocity_fields']} independently sampled full-environment fields per geometry; fixed across noise replicas",
            halo='Full periodic70304-atom parent; no halo enters the encoder',
            positional_directions=2, velocity_directions=2, reconstruction=False, crystal_labels=False),
        cohort=dict(fixed_identity=c['fixed_identity'], roles=c.get('parent_counts',[90,15,30]),
            selected_parent_indices=c.get('parent_indices',list(range(135))),
            ancestry='Subset of original135 queries, one geometry per ancestor; velocity/noise replicas are not new ancestors',
            evaluation='all64 unrelaxed transfer with actual same-atom MD velocities' if native else 'all64 unrelaxed static transfer with declared sampled velocities; not native MD velocity evaluation'),
        deviations=dict(batch_size=c['training']['batch_size'],microbatch=c['training']['microbatch'],
            reason='Higher derivative memory; native 90-state cohort sampled with replacement' if native else 'Small explicitly selected pilot cohort and higher derivative memory' if 'parent_indices' in c else 'Higher derivative memory',
            initial_velocity='Saved MD rather than Maxwell sampling' if native else 'Sampled and fixed across shots; new protocol')))
    return record


def load(c):
    r = read(root(c)/'technical/data-complete.json')
    p = resolve_path(c['cache'])/'training.pt'
    if r['identity'] != read(root(c)/'technical/binding.json')['identity'] or sha(p) != r['sha256']:
        raise ValueError('Sealed velocity bank changed')
    return torch.load(p, map_location='cpu', weights_only=False)
