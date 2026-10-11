"""Freeze, start locally, and continue one owned spatial-comparison GPU worker."""
import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, allocation_deadline, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import REPO, resolve_path
from src.research.structural_response.data import read, root, cache, digest, sha, write_json

MODULE = 'src.research.spatial_architecture.workflow'
OWNER = 'Response research chat: spatial MACE/eSEN/TensorNet comparisons, authorized 2026-10-07/08'


@contextmanager
def lease(path):
    with path.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prepare(c):
    contracts = check_metric_docs(family='spatial_architecture')['spatial_architecture']['files']
    if 'comparison_reference' in c:
        from .comparison import validate_reference
        validate_reference(c)
    if 'baseline_reference' in c:
        previous = resolve_path(c['baseline_reference']['output'])
        checkpoint = previous / 'analyses/spatial-vicreg_descriptors/technical/best.pt'
        if sha(checkpoint) != c['baseline_reference']['checkpoint_sha256']:
            raise ValueError('Changed completed MACE reference')
    if 'monitoring' in c:
        from .data import prepare_monitoring
        prepare_monitoring(c)
    complete = read(cache(c) / 'complete.json')
    for name in ('spatial.json', 'spatial-selection.npz'):
        if sha(cache(c)/name) != complete['files'][name]:
            raise ValueError('Changed reusable spatial target or selection data')
    packed = read(resolve_path(c['loader']['packed_cache'])/'manifest.json')
    fixed = read(resolve_path(c['fixed_dataset']['root'])/'splits.json')
    if fixed['identity'] != c['fixed_dataset']['identity']:
        raise ValueError('Changed fixed Al64 ancestry/split reference')
    roles = {s['lineage']:s['role'] for s in fixed['sources']}
    for shard in packed['shards']:
        task = shard['task']
        if task['role'] != 'train' or roles.get(task['lineage'], 'train') != 'train':
            raise ValueError('Held-out ancestry in spatial training')
    prepared = read(cache(c)/'spatial.json')
    from src.research.structural_response.data import spatial_target_names
    if [v['name'] for v in prepared['descriptor_columns']]!=spatial_target_names(c):
        raise ValueError('Prepared target panel differs from requested columns; prepare a new derived cache')
    if prepared['packed_identity'] != packed['identity'] or prepared['rows'] != c['data']['training_rows']:
        raise ValueError('Training population changed')
    plan = dict(config=c, implementations=contracts, packed_identity=packed['identity'],
        packed_manifest_sha256=sha(resolve_path(c['loader']['packed_cache'])/'manifest.json'),
        selection_sha256=complete['files']['spatial-selection.npz'])
    if 'monitoring' in c:
        plan['monitoring_files'] = complete['files']
    plan['identity'] = digest(plan)
    p = root(c)/'technical';p.mkdir(parents=True, exist_ok=True)
    if (p/'plan.json').exists() and read(p/'plan.json') != plan:
        raise ValueError('Changed prepared experiment; choose a new revision')
    write_json(p/'plan.json', plan)
    inputs = dict(
        encoder_inputs='unrelaxed nearest80 center-relative positions in fixed Al-equivalent units; constant atom channel and center indicator',
        species=False, explicit_conditions=[], velocities=False, history=False, relaxation=False,
        support_A=8, halo=False, taper_A=[6,8],
        edge_cutoffs_A={a: c['reference_encoder' if a == 'mace' else 'encoder']['cutoff'] for a in c['architectures']},
        predictor_inputs='128-dimensional exported spatial embedding only',
        training_targets=dict(columns=prepared['descriptor_columns'],weights=prepared['weights'],
            crystal_or_nucleus_labels=False),
        training_rows=prepared['rows'], material_rows=prepared['material_rows'],
        architectures=c['architectures'],
        architecture_settings={a: c['reference_encoder' if a == 'mace' else 'encoder'] for a in c['architectures']},
        execution_variants=c.get('execution_variants', {}),
        packed_identity=packed['identity'], selection_rows=prepared['validation_rows'],
        raw_histories_opened=False, retired_raw_campaign_selected=False,
        reused_input='preserved, checksum-bound spatial patch collection, not the retired raw Al histories',
        selector='fixed final epoch12; structural validation diagnostics only',
        precision='BF16 neural matmul; FP32 geometry, normalization, pooling, losses, optimizer')
    if c['data']['domain'] == 'relaxed':
        pairing = c['paired_relaxation']
        if not pairing['original_required_for_dynamic'] or pairing['original_is_model_input']:
            raise ValueError('Relaxed spatial training requires audit-only paired originals')
        inputs.update(
            encoder_inputs='relaxed center-relative positions in fixed Al-equivalent units; dynamic pairs preserve original nearest80 IDs, static clouds use their current nearest80; constant atom channel and center indicator',
            relaxation=True, paired_relaxation=pairing,
            reused_input='checksum-bound paired dynamic relaxations and user-confirmed relaxed static Mg/Al/Zr/Ta',
            raw_histories_opened=False,
            preparation_inputs='retained paired patches and available source frames; full-cell native restarts are read on CPU without force evaluation',
            training_only_teachers='same BO/CNA/TDA target panel computed on relaxed coordinates; unrelaxed originals excluded')
    elif c['data']['domain'] != 'hot':
        raise ValueError('Unrecognized spatial input domain: '+c['data']['domain'])
    if 'esen' in c['architectures']:
        inputs['eSEN'] = dict(upstream_revision='977a80328f2be44649b414a9907a1d6ef2f81e95',
            adaptation='one constant channel, center flag, local crop and MACE-matched scalar pooling; deterministic edge gauge',
            activation='SiLU gated spectral, no sampled-grid nonlinearity',
            smooth_envelope=True, correlation_order='not an eSEN parameter')
    if c['runtime'].get('gpus', 1) == 2:
        inputs['parallel_execution'] = dict(gpus=2, global_batch=c['spatial_training']['batch_size'],
            geometry_rows_per_device=c['spatial_training']['batch_size']//2,
            global_views='same seeded global views, split after jitter generation',
            loss='both views gathered on GPU0; full global-batch VICReg and descriptor mean',
            gradient='replica geometry gradients summed before clipping and one primary AdamW step',
            monitoring='unchanged diagnostic panels evaluated on GPU0')
    if 'monitoring' in c:
        inputs['monitoring'] = {k:prepared[k] for k in ('evaluation_role', 'evaluation_pool_rows', 'panels')}
        inputs['dataset_sampling'] = c['data']['sampling']
    if 'tensornet' in c['architectures']:
        inputs['TensorNet'] = dict(upstream_revision='7e22a231f34a855f894daaeb218c8b393068f846',
            adaptation='one learned constant atom channel and center flag, local crop and common scalar pooling',
            representations='O(3) rank2 Cartesian tensors; scalar/antisymmetric/symmetric-traceless parts',
            self_edges=True, charges='fixed internal zero, not an input',
            radial='32 fixed exponential-normal basis functions with upstream cosine cutoff',
            precision='BF16 neural matmul; FP32 geometry, norm reductions, normalization, pooling and losses',
            derivative_scope='ordinary PyTorch blocks support nested AD away from graph/support boundaries; fixed local atom list')
    write_json(p/'input-contract.json', inputs)
    return plan


def environment():
    return dict(PCM_PROJECT_ROOT=str(REPO), TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        CUEQUIVARIANCE_OPS_PARALLEL_THREADS='4', PYTHONWARNINGS='ignore::FutureWarning',
        PYTORCH_ALLOC_CONF='expandable_segments:True', TORCHINDUCTOR_COMPILE_THREADS='2')


def worker(c, lane):
    import torch
    from .parallel import devices
    from .train import fit, folder, identity
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = root(c)/'technical'
    probe = read(p/'probe.json')
    if not probe['passed'] or probe['config_identity'] != digest(c):
        raise ValueError('Numerical/timing probe does not match the training recipe')
    if torch.cuda.get_device_name() != probe['gpu']:
        raise ValueError('Matched training-speed comparison requires the benchmark GPU model')
    if len(devices(c)) != probe.get('gpu_count', 1):
        raise ValueError('Training GPU count differs from numerical/timing probe')
    if len(devices(c)) == 2 and [torch.cuda.get_device_name(d) for d in devices(c)] != probe['gpu_names']:
        raise ValueError('Training devices differ from the two-GPU numerical/timing probe')
    deadline = allocation_deadline(reserve_seconds=180)
    receipt = p/f'worker-{lane}-{os.getpid()}.json'
    with recorded_stage(receipt, owner=OWNER, pid=os.getpid(), host=os.uname().nodename,
            allocation=os.environ.get('SLURM_JOB_ID'), lane=lane, gpu=torch.cuda.get_device_name(),
            gpu_count=len(devices(c)),
            deadline_unix=deadline, command=sys.argv) as stage, lease(p/'worker.lock'):
        for architecture in c['architectures']:
            target = folder(c, architecture)
            if (target/'complete.json').exists():
                if read(target/'complete.json')['identity'] != identity(c, architecture):
                    raise ValueError('Changed completed model identity')
                continue
            stage.update(architecture=architecture)
            finished = fit(c, architecture, deadline)
            torch.cuda.empty_cache()
            if not finished:
                stage.update(continuation_required=True)
                if lane in ('successor', 'queued'):
                    job = os.environ['SLURM_JOB_ID']
                    if read(p/'launch.json')['jobs'][lane] != job:
                        raise ValueError('Refuse to requeue a job not owned by this launch')
                    subprocess.run(['scontrol', 'requeue', job], check=True)
                return
        completed = {a: read(folder(c,a)/'complete.json') for a in c['architectures']}
        report = dict(state='complete', models=completed)
        if 'esen' in completed and 'mace' in completed:
            report['esen_over_mace_training_time'] = completed['esen']['timing']['training_seconds']/completed['mace']['timing']['training_seconds']
        if 'comparison_reference' in c:
            from .comparison import publish
            report['comparison'] = publish(c)
        write_json(p/'complete.json', report)


def queued_worker(c, config_path):
    """Run the local numerical probe on the assigned idle GPU before fitting."""
    p = root(c)/'technical'
    with recorded_stage(p/'queued-stage.json', owner=OWNER, job=os.environ['SLURM_JOB_ID']) as stage:
        import torch
        from .parallel import devices
        probe_path = p/'probe.json'
        gpu_names = [torch.cuda.get_device_name(d) for d in devices(c)]
        if probe_path.exists() and read(probe_path).get('gpu_names', [read(probe_path)['gpu']]) != gpu_names:
            if not c['runtime'].get('reprobe_on_gpu_change', False):
                raise ValueError('Matched speed protocol cannot resume on a different GPU')
            archive = p/'probes'
            archive.mkdir(exist_ok=True)
            probe_path.rename(archive/(sha(probe_path)+'.json'))
        if not probe_path.exists():
            stage.update(phase='numerical_and_speed_probe')
            with (p/'probe.log').open('ab') as log:
                subprocess.run([sys.executable, '-u', '-m', MODULE, 'probe', '--config', str(config_path)],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
        stage.update(phase='scientific_training')
        worker(c, 'queued')


def queue_only(c, after_job=None):
    """Freeze and submit one worker with the requested local GPU count."""
    prepare(c)
    p = root(c)/'technical'
    if (p/'launch.json').exists():
        raise FileExistsError('This experiment already has a launch receipt')
    bundle = ExecutionBundle.freeze(REPO, p/'code', c,
        directories=('src','docs/metrics','configs/spatial_architecture','configs/data'))
    receipt = dict(owner=OWNER, jobs={}, bundle=str(bundle.root), launch_unix=time.time(),
        local=None, architectures=c['architectures'], after_job=after_job,
        execution='queued numerical probe, then online scientific training')
    queue = SlurmQueue(p,bundle,MODULE,environment(),p/'launch.json',receipt,'spatial-arch')
    with queue.submission():
        queue.submit('queued', [f"--gpus={c['runtime'].get('gpus', 1)}", f"--cpus-per-task={c['runtime']['cpu_threads']}",
            f"--mem={c['runtime']['memory_GB']}G", f"--time={c['runtime']['queue_hours']:02d}:00:00", '--requeue'],
            dependency='afterok:'+after_job if after_job else None,
            partition=c['runtime']['partition'], command_stage='queued-worker')
    print(receipt, flush=True)


def launch(c):
    if c['runtime'].get('gpus', 1) == 2:
        # The allocation supervisor already handles matched multi-GPU
        # continuation probes and cancels only its unused owned successor.
        from .allocation import launch as use_allocation
        queue_only(c, after_job=os.environ['SLURM_JOB_ID'])
        use_allocation(root(c))
        return
    prepare(c)
    p = root(c)/'technical'
    if (p/'launch.json').exists():
        raise FileExistsError('This experiment has a launch receipt; inspect/resume it, do not duplicate it')
    probe = read(p/'probe.json')
    if not probe['passed'] or probe['config_identity'] != digest(c):
        raise ValueError('Current recipe has not passed its numerical and timing probe')
    bundle = ExecutionBundle.freeze(REPO, p/'code', c,
        directories=('src','docs/metrics','configs/spatial_architecture'))
    env = environment()
    receipt = dict(owner=OWNER, jobs={}, bundle=str(bundle.root), allocation=os.environ.get('SLURM_JOB_ID'),
        launch_unix=time.time(), local=None, architectures=c['architectures'])
    queue = SlurmQueue(p,bundle,MODULE,env,p/'launch.json',receipt,'spatial-arch')
    with queue.submission():
        # One successor for the existing local lane, not one reservation per GPU type.
        dependency = 'afterany:'+os.environ['SLURM_JOB_ID'] if os.environ.get('SLURM_JOB_ID') else None
        queue.submit('successor', ['--gpus=1','--cpus-per-task=4','--mem=32G',
            '--time=24:00:00','--requeue'], dependency=dependency, partition=c['runtime']['partition'],
            command_stage='worker', arguments=('--lane','successor'))
        with (p/'local.log').open('ab') as log:
            process = subprocess.Popen(queue.command('worker','--lane','local'), cwd=bundle.root,
                env=dict(os.environ,**env,CUDA_VISIBLE_DEVICES='0'), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt['local'] = dict(pid=process.pid, host=os.uname().nodename, gpu='0', log=str(p/'local.log'))
        write_json(p/'launch.json',receipt)
    print(receipt,flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare','probe','launch','worker','queue','queued-worker'])
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--lane', choices=['local','successor','queued'])
    parser.add_argument('--after-job', help='Successful Slurm prerequisite for queue mode')
    args = parser.parse_args();c=read(args.config)
    if args.stage == 'prepare':prepare(c)
    elif args.stage == 'probe':
        from .probe import run
        run(c)
    elif args.stage == 'launch':launch(c)
    elif args.stage == 'queue':
        if args.after_job is not None and not args.after_job.isdigit():
            parser.error('--after-job must be numeric when supplied')
        queue_only(c,args.after_job)
    elif args.stage == 'queued-worker':queued_worker(c,args.config)
    else:worker(c,args.lane)


if __name__ == '__main__':
    main()
