"""Matched full-cell MEAM CUDA relaxation: throughput and target fidelity.

This is a standalone benchmark, never a trainer hook. No production receipt is
modified. The source release and each binary are pinned in the result manifest.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import traceback
from types import SimpleNamespace
import numpy as np
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json, file_hash
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.data.relaxed_targets.worker import AbsolutePositions, publish
from src.data.conversion.relaxation import read_relaxed, convert
from src.simulation.relaxation import relax_frame
from src.research.relaxed_encoder.prepare import settings, paired_clouds
from src.research.relaxed_encoder.benchmark import fidelity
from src.experiment_runner.metric_docs import write_metric_table


def read_force_dump(path, atom_ids):
    """Only our explicit sorted force-dump schema is accepted."""
    lines=Path(path).read_text().splitlines()
    if lines[8] != 'ITEM: ATOMS id type fx fy fz':
        raise ValueError(f'Unexpected force dump schema in {path}: {lines[8]}')
    table=np.loadtxt(lines[9:])
    if table.shape != (len(atom_ids),5):raise ValueError(f'Force shape {table.shape}')
    np.testing.assert_array_equal(table[:,0],atom_ids)
    np.testing.assert_array_equal(table[:,1],np.ones(len(atom_ids)))
    if not np.isfinite(table).all():raise ValueError(f'Nonfinite forces: {path}')
    return table[:,2:]


def force_metrics(reference, candidate):
    if reference.shape!=candidate.shape:raise ValueError('Unpaired force arrays')
    delta=candidate-reference
    rms=float(np.sqrt(np.mean(delta**2)))
    ref_rms=float(np.sqrt(np.mean(reference**2)))
    return dict(force_component_rms_error_eV_per_A=rms,
                force_component_max_error_eV_per_A=float(np.abs(delta).max()),
                force_relative_rms_error=rms/max(ref_rms,1e-30))


def force_probe(work, command, atom_ids, timeout):
    # Evaluate exactly the input coordinates, not the just-minimized coordinates.
    text=(work/'in.lammps').read_text().split('minimize 0.0')[0]
    text+='run 0\nvariable probe_energy equal pe\nprint "PROBE_ENERGY ${probe_energy}"\n'
    text+='write_dump all custom forces.dump id type fx fy fz modify sort id format line "%d %d %.17g %.17g %.17g"\n'
    (work/'probe.lammps').write_text(text)
    with (work/'probe.stdout').open('w') as log:
        subprocess.run([*command,'-in','probe.lammps','-log','probe.log'],cwd=work,
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    lines=(work/'probe.log').read_text().splitlines()
    energy=float(next(x.split()[1] for x in reversed(lines) if x.startswith('PROBE_ENERGY ')))
    if not np.isfinite(energy):raise ValueError('Nonfinite initial energy')
    return read_force_dump(work/'forces.dump',atom_ids),energy


def progress(root,backend,**values):
    save_json(root/'technical'/f'status-{backend}.json',dict(updated_at=time.time(),**values))


def run(config,backend,binary,*,first_case_only=False):
    root=resolve_path(config['output']);technical=root/'technical';technical.mkdir(parents=True,exist_ok=True)
    plan=json.loads(resolve_path(config['paired_plan']).read_text())
    variants=['cpu-current','cpu-legacy'] if backend=='cpu' else [backend]
    hardware=dict(host=socket.gethostname(),allocation=os.environ.get('SLURM_JOB_ID'),
                  source_commit=config['source_commit'],binary=str(binary),binary_sha256=file_hash(binary))
    if backend!='cpu':
        hardware['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    save_json(technical/f'hardware-{backend}.json',hardware)
    for case in (config['cases'][:1] if first_case_only else config['cases']):
        source=next(s for s in plan['sources'] if s['id']==case['source'])
        raw=ShootingBinaryTrajectory.load(resolve_path(source['path']));frame=case['frame']
        if file_hash(raw.root/'manifest.json')!=source['manifest_sha256']:raise ValueError('Source changed')
        low=raw.box_low[frame].astype(np.float64);box=raw.box_high[frame].astype(np.float64)-low
        hot=raw.positions[frame].astype(np.float64);queries=np.searchsorted(raw.atom_ids,source['pool_atom_ids'])
        np.testing.assert_array_equal(raw.atom_ids[queries],source['pool_atom_ids'])
        absolute=SimpleNamespace(**vars(raw),atom_count=raw.atom_count);absolute.positions=AbsolutePositions(raw)
        for variant in variants:
            for repeat in range(1 if first_case_only else config['repeats']):
                name=f'{variant}-{case["name"]}-r{repeat}';receipt=technical/'results'/f'{name}.json'
                if receipt.exists():continue
                work=resolve_path(config['scratch'])/name;archive=resolve_path(config['archive'])/name
                cfg=settings(plan,config['cpu_ranks'])
                cfg.update({k:config[k] for k in ('max_iterations','max_evaluations','frame_timeout_seconds')})
                if variant=='cpu-current':cfg['lammps_command'][-1]=str(binary)
                elif variant!='cpu-legacy':
                    cfg['lammps_command']=[str(binary),'-k','on','g','1','-sf','kk','-pk','kokkos','neigh','half','newton','on','gpu/aware','off']
                binary_path=cfg['lammps_command'][0] if backend!='cpu' else cfg['lammps_command'][-1]
                row=dict(case=case,variant=variant,repeat=repeat,atoms=raw.atom_count,
                         binary_sha256=file_hash(binary_path),hardware=hardware,
                         source_manifest_sha256=source['manifest_sha256'],archive=str(archive))
                progress(root,backend,state='running',case=case['name'],variant=variant,repeat=repeat)
                started=time.monotonic()
                try:
                    meta=relax_frame(absolute,frame,work,cfg)
                    pos,_=read_relaxed(work)
                    _,clouds,_=paired_clouds(hot,pos-low,box,queries)
                    receipt.parent.mkdir(exist_ok=True)
                    np.save(technical/'results'/f'{name}-clouds.npy',clouds)
                    row.update(state='complete',seconds=meta['seconds'],fmax_eV_per_A=meta['fmax_eV_per_A'],
                               energy_eV_per_atom=meta['energy_eV']/raw.atom_count,input_sha256=meta['input_sha256'])
                    log=(work/'log.lammps').read_text()
                    count=re.search(r'Iterations, force evaluations\s*=\s*(\d+)\s+(\d+)',log)
                    if count:row.update(iterations=int(count[1]),force_evaluations=int(count[2]))
                    if backend!='cpu' and ('meam/kk' not in log or not re.search(r'KOKKOS mode(?: with Kokkos version [0-9.]+)? is enabled',log)):
                        raise RuntimeError('GPU accelerated MEAM was not confirmed in the log')
                    # One untimed initial force/energy equivalence check per case.
                    if repeat==0:
                        forces,energy=force_probe(work,cfg['lammps_command'],raw.atom_ids,180)
                        np.save(technical/'results'/f'{name}-forces.npy',forces)
                        row['initial_energy_eV_per_atom']=energy/raw.atom_count
                    convert(work,delete_source=False,local_cloud_dtype='float32')
                except Exception as exc:
                    row.update(state='failed',error=repr(exc),traceback=traceback.format_exc(),attempt_seconds=time.monotonic()-started)
                    print(row['traceback'],flush=True)
                if work.exists():publish(work,archive)
                save_json(receipt,row);print(json.dumps({k:v for k,v in row.items() if k not in ('hardware','traceback')}),flush=True)
                report(config)
    rows=[json.loads(p.read_text()) for p in (technical/'results').glob('*.json')]
    own=[r for r in rows if r['variant'] in variants]
    failures=sum(r['state']!='complete' for r in own)
    progress(root,backend,state='complete' if not failures else 'completed_with_failures',completed=len(own)-failures,failed=failures)
    report(config)


def report(config):
    root=resolve_path(config['output']);technical=root/'technical';technical.mkdir(parents=True,exist_ok=True)
    with (technical/'report.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=[json.loads(p.read_text()) for p in (technical/'results').glob('*.json')]
        plan=json.loads(resolve_path(config['paired_plan']).read_text());c=plan['config']
        norm=json.loads(resolve_path(c['normalization_manifest']).read_text())['normalization']
        order=json.loads(resolve_path(c['order_manifest']).read_text())
        table={};summaries=[]
        for case in config['cases']:
            groups={v:[r for r in rows if r['case']==case and r['variant']==v] for v in ('cpu-current','cpu-legacy','v100','a100','h100')}
            reference=next((r for r in groups['cpu-current'] if r['state']=='complete' and r['repeat']==0),None)
            for variant,rr in groups.items():
                good=[r for r in rr if r['state']=='complete']
                if not rr:continue
                row=dict(completed=len(good),failed=len(rr)-len(good))
                if good:
                    seconds=float(np.median([r['seconds'] for r in good]));row.update(median_seconds=seconds,cells_per_hour=3600/seconds)
                    for base in ('cpu-current','cpu-legacy'):
                        b=[r['seconds'] for r in groups[base] if r['state']=='complete']
                        if b:row[f'speedup_vs_{base}']=float(np.median(b)/seconds)
                if reference:
                    refname=f'cpu-current-{case["name"]}-r0';ref=np.load(technical/'results'/f'{refname}-clouds.npy')
                    for r in good:
                        name=f'{variant}-{case["name"]}-r{r["repeat"]}'
                        detail=technical/'fidelity'/f'{name}.json'
                        if not detail.exists():
                            f=fidelity(ref,np.load(technical/'results'/f'{name}-clouds.npy'),np.array(norm['physical']['std']),np.array(norm['tda']['std']),np.array(order['std']),c['scale'])
                            f['energy_difference_eV_per_atom']=r['energy_eV_per_atom']-reference['energy_eV_per_atom']
                            if r['repeat']==0:
                                f.update(force_metrics(np.load(technical/'results'/f'{refname}-forces.npy'),np.load(technical/'results'/f'{name}-forces.npy')))
                                f['initial_energy_difference_eV_per_atom']=r['initial_energy_eV_per_atom']-reference['initial_energy_eV_per_atom']
                            save_json(detail,f)
                        row[f'repeat_{r["repeat"]}']=json.loads(detail.read_text())
                table[f'{case["name"]}/{variant}']=row
                summaries.append(f'| {case["name"]} | {variant} | {len(good)}/{config["repeats"]} | {row.get("median_seconds",float("nan")):.1f} | {row.get("speedup_vs_cpu-current",float("nan")):.2f} |')
        save_json(technical/'comparison.json',table)
        write_metric_table(table,root,family='relaxation_cuda',name='throughput-fidelity')
        (root/'RESULTS.md').write_text('# Matched MEAM CPU/GPU relaxation benchmark\n\n'
            'Partial until every backend status is complete. One GPU per frame; CPU uses 32 MPI ranks. '
            'Both repeats start from identical original MD coordinates. Median end-to-end LAMMPS invocation seconds excludes build, input writing, target extraction, force probe and archiving. '
            'Failed attempts remain in receipts; speedups describe successful runs only.\n\n'
            '| Case | Backend | Completed | Seconds | Speedup vs same-release CPU |\n|---|---|---:|---:|---:|\n'+'\n'.join(summaries)+'\n\n'
            'See tables/throughput-fidelity.csv and tables/METRICS.md for force equivalence and final target differences. '
            'Case labels describe sampled local onset context, not a full-cell phase certification.\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--backend',choices=['cpu','v100','a100','h100'],required=True);p.add_argument('--binary',required=True);p.add_argument('--first-case-only',action='store_true');args=p.parse_args()
    config=json.loads(resolve_path(args.config).read_text())
    try:run(config,args.backend,Path(args.binary),first_case_only=args.first_case_only)
    except Exception as exc:
        progress(resolve_path(config['output']),args.backend,state='failed',error=repr(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
