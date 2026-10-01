"""Operational Slurm batching; each fit uses the unchanged frozen worker."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def worker(bundle, kind, lanes):
    sys.path.insert(0, str(bundle))
    from src.research.predictive_followup.common import jobs
    c = json.loads((bundle/'config.json').read_text())
    lane = int(os.environ['SLURM_ARRAY_TASK_ID'])
    for index in range(lane, len(jobs(c,kind)), lanes):
        subprocess.run([sys.executable,'-u','-m','src.research.predictive_followup.queue',kind,
            '--config',str(bundle/'config.json'),'--index',str(index)],cwd=bundle,check=True)


def submit(config_path, gpu_lanes=1):
    from src.data.fixed_cohort.protocol import sha, write_json
    from src.experiment_runner.execution import ExecutionBundle, SlurmQueue
    from src.experiment_runner.metric_docs import check_metric_docs
    from src.research.predictive_followup.common import FAMILY, jobs, prepare, read, output, folder
    c = read(config_path)
    binding = prepare(c)
    check_metric_docs(family=FAMILY)
    root = output(c)
    tech = root/'technical'
    repo = Path(os.environ['PCM_PROJECT_ROOT']) if 'PCM_PROJECT_ROOT' in os.environ else Path(__file__).resolve().parents[3]
    gate = read(tech/'preflight.json')
    if gate['state'] != 'complete' or gate['binding'] != binding['identity']:
        raise ValueError('Matching full-batch gate required')
    if (tech/'launch.json').exists():
        receipt = read(tech/'launch.json')
        if 'collect' in receipt['jobs']:
            raise FileExistsError('All stages already submitted; use recorded scripts to resume')
        if receipt['binding'] != binding['identity']:
            raise ValueError('Existing scientific submission binding changed')
        if not (tech/'initial-launch.json').exists():
            write_json(tech/'initial-launch.json',receipt)
        if 'submission_error' in receipt:
            receipt['recovered_submission_error'] = receipt.pop('submission_error')
        bundle = ExecutionBundle(Path(receipt['code']))
        if read(bundle.config_path) != c:
            raise ValueError('Frozen configuration differs from requested recovery')
    else:
        bundle = ExecutionBundle.freeze(repo,tech/'code',c,
            directories=('src','docs/metrics','configs/predictive_baseline'),
            files=((repo/c['shooting_config'],c['shooting_config']),
                   (repo/c['encoders'][0]['pretraining_recipe'],c['encoders'][0]['pretraining_recipe'])))
        receipt = dict(jobs={},binding=binding['identity'],code=str(bundle.root),
            joint_arms=jobs(c,'joint'),probe_arms=jobs(c,'probe'),
            scientific_online_runs=len(jobs(c,'joint')),local_diagnostic_heads=len(jobs(c,'probe')))
    execution = tech/'execution-lanes-v1'
    execution.mkdir(exist_ok=True)
    script = execution/'lanes.py'
    if script.exists():
        if sha(script) != receipt['execution_lanes']['sha256']:
            raise ValueError('Existing frozen operational wrapper changed')
    else:
        shutil.copy2(__file__,script)
    receipt['execution_lanes'] = dict(script=str(script),sha256=sha(script),
        cpu_lanes=c['probe_concurrency'],gpu_lanes=gpu_lanes,
        assignment='arm indices lane, lane+lane_count, ...; each invokes original frozen worker',
        reason='Bound Slurm submission count; original 48 scientific/diagnostic fits preserved')
    environment = dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
                       WANDB_MODE='online',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')

    class LaneQueue(SlurmQueue):
        def command(self,stage,*arguments):
            lanes = gpu_lanes if stage=='joint' else c['probe_concurrency']
            return [sys.executable,'-u',str(script),'worker','--bundle',str(bundle.root),
                    '--kind',stage,'--lanes',str(lanes)]

    regular = SlurmQueue(tech,bundle,'src.research.predictive_followup.queue',environment,tech/'launch.json',receipt,'PRED-FOLLOW')
    lane_queue = LaneQueue(tech,bundle,'src.research.predictive_followup.queue',environment,tech/'launch.json',receipt,'PRED-FOLLOW')
    with regular.submission():
        if 'controls' not in receipt['jobs']:
            regular.submit('controls',['--cpus-per-task=4','--mem=16G','--time=00:30:00'])
        controls = receipt['jobs']['controls']
        if 'probe-lanes' not in receipt['jobs']:
            lane_queue.submit('probe-lanes',['--cpus-per-task=4','--mem=12G','--time=01:30:00',
                f'--array=0-{c["probe_concurrency"]-1}'],command_stage='probe')
        probes = receipt['jobs']['probe-lanes']
        if 'joint-lanes' not in receipt['jobs']:
            lane_queue.submit('joint-lanes',['--gpus=1','--cpus-per-task=4','--mem=24G','--time=02:00:00',
                '--exclude='+','.join(c['exclude_nodes']),f'--array=0-{gpu_lanes-1}'],
                partition=c['gpu_partition'],command_stage='joint')
        joint = receipt['jobs']['joint-lanes']
        dependencies = []
        completed = []
        controls_done = tech/'controls-complete.json'
        if controls_done.exists() and read(controls_done)['state']=='complete':
            completed.append(str(controls_done))
        else:
            dependencies.append(controls)
        for kind,job in (('probe',probes),('joint',joint)):
            receipts = [folder(c,arm)/'technical/complete.json' for arm in jobs(c,kind)]
            if all(p.exists() and read(p)['state']=='complete' for p in receipts):
                completed.extend(map(str,receipts))
            else:
                dependencies.append(job)
        # Completed jobs can age out of Slurm while another stage runs. Durable
        # completion receipts replace only those already satisfied dependencies;
        # the numerical collector independently verifies predictions and identities.
        receipt['collector_dependencies'] = dependencies
        receipt['completed_stage_receipts_used'] = completed
        regular.submit('collect',['--cpus-per-task=4','--mem=16G','--time=01:00:00'],
                       'afterok:'+':'.join(dependencies) if dependencies else None)
    local = repo/'output/predictive_baseline'/root.name
    local.parent.mkdir(parents=True,exist_ok=True)
    if not local.exists():
        local.symlink_to(root,target_is_directory=True)
    (root/'README.md').write_text(
        '# Predictive baseline follow-up with MM-TDA-BLOCK-DIRECT-FULL\n\n'
        '12 joint MACE128 fits and 36 local frozen-head diagnostics use three seeds and a '
        '2x2 full/moment target versus free/nonnegative variance design. Frozen references '
        'include VICReg128, Epi128 and the exact MM-TDA epoch20 z256 export.\n\n'
        'All new fits select on the same18-moment validation likelihood. Targets and source '
        'roles are inherited unchanged from the historical Al480 baseline.\n\n'
        f'Execution is grouped into four CPU and {gpu_lanes} GPU workers to respect scheduler job '
        'limits. See `technical/launch.json`, per-arm `analyses/*/technical/progress.json` '
        'and `technical/prediction-context.json`. Each scientific fit has independent '
        'resumable checkpoints and an online W&B ID; frozen heads remain local.\n\n'
        'The dependent collector writes `analyses/comparison-v1/tables/`, paired contrasts '
        'and plots. Missing full-feature predictions for moment-only heads are never filled. '
        'MM-TDA retains its historical capacity/pretraining and archived ancestry limitations.\n')
    return receipt


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=('submit','worker'))
    p.add_argument('--config')
    p.add_argument('--bundle',type=Path)
    p.add_argument('--kind',choices=('joint','probe'))
    p.add_argument('--lanes',type=int)
    p.add_argument('--gpu-lanes',type=int,default=1)
    a = p.parse_args()
    if a.stage=='worker':
        worker(a.bundle,a.kind,a.lanes)
    else:
        print(json.dumps(submit(a.config,a.gpu_lanes)['jobs']),flush=True)


if __name__=='__main__':
    main()
