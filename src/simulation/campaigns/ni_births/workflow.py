"""Prepare, validate and detach the Ni nucleus-birth collection/search queue."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from src.experiment_runner.artifacts import file_hash as sha,json_digest as digest,write_json
from src.experiment_runner.execution import ExecutionBundle,SlurmQueue,allocation_deadline
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import resolve_path,REPO,machine
from src.simulation.campaigns.birth_sources import register_entry


def extension_parent(c):
    extension=c['extension_of'];launch=resolve_path(extension['launch'])
    parent=json.loads((launch/'plan.json').read_text())
    if parent['identity']!=extension['identity'] or parent['config']['campaign_id']!=extension['dataset']:
        raise ValueError('Ni extension parent identity changed')
    operational={'campaign_id','launch','scratch','archive','analysis','cache','seed','replicate_start',
                 'gpu_workers','walltime','analysis_workers','analysis_walltime','analysis_memory',
                 'extension_of','combined_analysis','scientific_scope'}
    scientific={k:v for k,v in parent['config'].items() if k not in operational}
    if scientific!={k:v for k,v in c.items() if k not in operational}:
        raise ValueError('Ni extension changed the parent scientific protocol')
    return parent


def prepare(c):
    launch=resolve_path(c['launch']);launch.mkdir(parents=True,exist_ok=True)
    if (launch/'plan.json').exists():
        old=json.loads((launch/'plan.json').read_text())
        if old['config']!=c:raise ValueError('Ni campaign configuration changed')
        return old
    if sha(resolve_path(c['potential']))!=c['potential_sha256']:raise ValueError('Ni potential changed')
    records=[]
    for replicate,role in enumerate(c['replicate_roles'],start=c.get('replicate_start',0)):
        for temperature in c['temperatures_K']:
            index=len(records);records.append(dict(id=f'ni-T{temperature}-melt{replicate:02d}',
                temperature_K=temperature,replicate=replicate,role=role,preparation_seed=c['seed']+index*101))
    workers=c.get('gpu_workers',1)
    if type(workers)!=int or not 1<=workers<=len(records):raise ValueError('Invalid Ni GPU worker count')
    if 'extension_of' in c:
        parent=extension_parent(c)
        if ({r['id'] for r in records}&{r['id'] for r in parent['records']}
                or {r['preparation_seed'] for r in records}&{r['preparation_seed'] for r in parent['records']}):
            raise ValueError('Ni extension repeats a parent source identity or preparation seed')
    p=dict(config=c,records=records,created_at=datetime.now(timezone.utc).isoformat());p['identity']=digest(p)
    resolve_path(c['archive']).mkdir(parents=True,exist_ok=True)
    entry=dict(root='archive',path='simulations/'+c['campaign_id'],kind='simulation',dependencies=['potential-ni-mendelev2012'],
        metadata=dict(title='Independent Ni NPT nucleus-birth sources, 0.10 ps Q/V',role='raw_dynamics',classification='planned',
            materials=['Ni'],potential_ids=['ni-mendelev2012-eam'],description=c['scientific_scope'],
            planned_independent_melts=len(records),timestep_ps=c['timestep_ps'],sample_interval_ps=.1,
            velocity_units='angstrom_per_ps',staging_location=c['scratch'],evidence=[str(launch/'plan.json')]))
    if 'extension_of' in c:
        entry['metadata'].update(campaign_extension_of=c['extension_of']['dataset'],
            ancestry='New independent full-cell melts; no reused parent configurations or velocities')
    register_entry(c['campaign_id'],entry)
    write_json(launch/'plan.json',p);write_json(resolve_path(c['archive'])/'campaign.json',p)
    if 'extension_of' in c:
        combined=[dict(r,dataset=source['config']['campaign_id'],archive=source['config']['archive'])
                  for source in [parent,p] for r in source['records']]
        write_json(launch/'combined-source-plan.json',dict(sources=combined,planned_independent_melts=len(combined),
            component_identities=[parent['identity'],p['identity']],analysis=c['combined_analysis']))
    return p


def queue(c,*,frozen):
    launch=resolve_path(c['launch']);settings=machine()
    if frozen:
        code=ExecutionBundle.freeze(REPO,launch/'code',c,directories=('src','configs','scripts','docs/metrics'))
        settings['catalog']=str(code.root/'configs/datasets.json')
        write_json(launch/'machine.json',settings)
        receipt=dict(identity=json.loads((launch/'plan.json').read_text())['identity'],jobs={},
            code=str(code.root),machine_sha256=sha(launch/'machine.json'),
            launched_at=datetime.now(timezone.utc).isoformat(),purpose='User requested independent Ni nucleus-birth campaign, current chat')
        receipt_path=launch/'launch.json'
    else:
        folder=launch/'audit-entry';folder.mkdir(exist_ok=True);write_json(folder/'config.json',c)
        code=ExecutionBundle(folder)
        receipt=dict(jobs={},purpose='Current-chat Ni isotropic NPT adapter numerical validation')
        receipt_path=launch/'audit-launch.json'
    cuda=settings['execution']['gpu_meam_portable_cuda_root']
    env=dict(PCM_PROJECT_ROOT=str(REPO),PYTHONPATH=str(code.root if frozen else REPO),
        UCX_TLS='tcp,self',CUDA_HOME=cuda,PATH=cuda+'/bin:'+str(Path(sys.executable).parent)+':/usr/bin:/bin',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MAX_JOBS='2',
        OVITO_THREAD_COUNT='1',QT_QPA_PLATFORM='offscreen')
    if frozen:env['PCM_MACHINE_CONFIG']=str(launch/'machine.json')
    # Audit runs from the repository because the maintained converter is a script.
    if not frozen:code=ExecutionBundle(REPO)
    q=SlurmQueue(launch,code,'src.simulation.campaigns.ni_births.workflow',env,receipt_path,receipt,'NI-BIRTH')
    return q


def submit_audit(c,path):
    launch=resolve_path(c['launch'])
    from .validation import reuse
    try:
        proof=reuse(c,destination=launch/'validation-reuse.json')
    except ValueError as error:
        print(str(error),flush=True)  # The explicitly requested audit is needed.
    else:
        print(json.dumps(proof));return
    if (launch/'audit-launch.json').exists():raise FileExistsError('Ni audit already submitted')
    q=queue(c,frozen=False)
    # The audit uses the explicit recipe, not a repository-root config.json.
    command=[sys.executable,'-u','-m','src.simulation.campaigns.ni_births.workflow','audit','--config',str(resolve_path(path))]
    import shlex
    text=q.render('audit',['--gres=gpu:1','--cpus-per-task=4','--mem=24G','--time=01:00:00','--exclude='+c['exclude_nodes']],partition=c['partitions'])
    text=text.replace(shlex.join(q.command('audit')),shlex.join(command))
    script=launch/'audit.sbatch';script.write_text(text)
    job=subprocess.check_output(['sbatch','--parsable',str(script)],text=True).strip().split(';')[0]
    q.receipt['jobs']['audit']=job;write_json(q.receipt_path,q.receipt);print(json.dumps(q.receipt))


def submit(c):
    launch=resolve_path(c['launch'])
    if (launch/'launch.json').exists():raise FileExistsError('Ni campaign already launched')
    from .validation import reuse
    reuse(c,destination=launch/'validation-reuse.json')
    check_metric_docs(family='ni_eam_gpu')
    check_metric_docs(family='ni_births');q=queue(c,frozen=True)
    with q.submission():
        workers=c.get('gpu_workers',1);jobs=[]
        for lane in range(workers):
            name='worker' if workers==1 else f'worker-{lane:02d}'
            jobs.append(q.submit(name,['--gres=gpu:1','--cpus-per-task=4','--mem=24G','--time='+c['walltime'],
                                      '--exclude='+c['exclude_nodes']],partition=c['partitions'],
                                 command_stage='worker',arguments=('--lane',str(lane))))
        analysis=q.submit('analyse',['--cpus-per-task='+str(c['analysis_workers']),
            '--mem='+c.get('analysis_memory','48G'),'--time='+c['analysis_walltime']],dependency='afterany:'+':'.join(jobs))
        if 'extension_of' in c:
            parent_launch=resolve_path(c['extension_of']['launch'])
            parent_receipt=json.loads((parent_launch/'launch.json').read_text())
            if parent_receipt['identity']!=c['extension_of']['identity']:raise ValueError('Parent launch identity changed')
            q.submit('combine',['--cpus-per-task=1','--mem=16G','--time=01:00:00'],
                     dependency='afterok:'+parent_receipt['jobs']['analyse']+':'+analysis)
    print(json.dumps(q.receipt),flush=True)


def worker(c,path,lane=0):
    launch=resolve_path(c['launch']);plan=json.loads((launch/'plan.json').read_text());deadline=allocation_deadline(reserve_seconds=600)
    workers=c.get('gpu_workers',1)
    if not 0<=lane<workers:raise ValueError('Ni GPU lane outside configured range')
    status=launch/('status.json' if workers==1 else f'status-{lane:02d}.json')
    outcomes=[]
    for index,record in enumerate(plan['records']):
        if index%workers!=lane:continue
        if time.time()+600>deadline:
            outcomes.append(dict(id=record['id'],state='not_started',reason='allocation time remaining'));continue
        command=[sys.executable,'-u','-m','src.simulation.campaigns.ni_births.workflow','source','--config',str(path),'--index',str(index),'--deadline',str(deadline)]
        with (launch/(record['id']+'.log')).open('a') as log:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        archive=resolve_path(c['archive'])/record['id'];outcome=archive/'outcome.json'
        if process.returncode:
            archive.mkdir(parents=True,exist_ok=True)
            result=dict(id=record['id'],state='failed',exit_code=process.returncode,log=str(launch/(record['id']+'.log')))
            if (archive/'complete.json').exists():
                # Preserve the distinction between finished MD and a subsequent
                # housekeeping failure, including the original nonzero exit.
                write_json(archive/'postprocessing-failure.json',result)
                result=dict(json.loads((archive/'complete.json').read_text()),postprocessing_state='failed')
                write_json(outcome,result)
            else:
                write_json(outcome,result)
                # Preserve this source's partial payload and native checkpoints on STORE.
                root=resolve_path(c['scratch'])/record['id']
                if root.exists():shutil.copytree(root,archive/'failed-staging',dirs_exist_ok=True)
        else:result=json.loads(outcome.read_text())
        outcomes.append(result)
        write_json(status,dict(state='running',outcomes=outcomes,lane=lane,workers=workers,planned_sources=len(plan['records']),job=os.environ.get('SLURM_JOB_ID')))
    state='complete' if all(x['state']=='complete' for x in outcomes) else 'incomplete'
    write_json(status,dict(state=state,outcomes=outcomes,lane=lane,workers=workers,planned_sources=len(plan['records']),job=os.environ.get('SLURM_JOB_ID')))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['prepare','submit-audit','audit','submit','worker','source','analyse','combine','status'])
    p.add_argument('--config',required=True);p.add_argument('--index',type=int);p.add_argument('--deadline',type=float);p.add_argument('--lane',type=int,default=0)
    a=p.parse_args();c=json.loads(resolve_path(a.config).read_text())
    if a.action=='prepare':print(json.dumps(prepare(c)))
    elif a.action=='submit-audit':submit_audit(c,a.config)
    elif a.action=='submit':submit(c)
    elif a.action=='worker':worker(c,a.config,a.lane)
    elif a.action=='analyse':
        from src.postprocessing.protocols.ni import analyse
        analyse(c)
    elif a.action=='combine':
        from src.postprocessing.protocols.ni import combine
        combine(c)
    elif a.action=='status':
        launch=resolve_path(c['launch'])
        for name in ['audit-launch.json','launch.json','status.json','verified-status.json']:
            if (launch/name).exists():print(name,(launch/name).read_text())
        for path in sorted(launch.glob('status-*.json')):print(path.name,path.read_text())
    else:
        from . import md
        import torch
        torch.set_num_threads(1)
        if a.action=='audit':md.audit(c)
        else:
            record=json.loads((resolve_path(c['launch'])/'plan.json').read_text())['records'][a.index]
            md.run(c,record,a.deadline)


if __name__=='__main__':main()
