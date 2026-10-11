"""Ni-specific source inventory using the established full-cell lineage tracker."""
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
from src.experiment_runner.artifacts import file_hash as sha,json_digest as digest,write_json
from src.experiment_runner.metric_docs import write_metric_rows,check_metric_docs
from src.project_runtime.paths import resolve_path
from src.postprocessing import extraction as search_clusters, lineage as lineage_tracker
from src.postprocessing.sources import part
from src.data.trajectories.shooting import ShootingBinaryTrajectory


def extraction_config(c):
    return dict(cache=c['cache'],cadences={'010':.1},detectors=['ptm','coherent_loose','coherent','coherent_strict'],
        ptm_rmsd_cutoff=c['ptm_rmsd_cutoff'],neighbor_cutoff_A=c['neighbor_cutoff_A'],minimum_tracked_atoms=4,
        bond_order=dict(l=6,thresholds=[[.65,6],[.7,7],[.75,8]]),chunk_frames=16)


def track_config(c):
    return dict(output=c['analysis'],minimum_shared_atoms=4,continuation_fraction=.5,minor_parent_fraction=.2,
        embryo_atoms=8,reset_atoms=4,reset_ps=1.5,persistence_ps=1.5,establishment_atoms=64,growth_atoms=128,
        thresholds=[8,32,64,128],durations_ps=[1.5,3.0],prehistory_ps=1.5,interface_distance_A=c['interface_distance_A'])


def source(c,r):
    """Validate one completed native trajectory; an active simulation is never read."""
    folder=resolve_path(c['archive'])/r['id'];outcome=json.loads((folder/'complete.json').read_text())
    path=folder/'trajectory_binary_float16';binary=ShootingBinaryTrajectory.load(path)
    if sha(path/'manifest.json')!=outcome['trajectory_manifest_sha256']:raise ValueError('Changed completed Ni trajectory')
    binary.verify_checksums();interval=round(.1/c['timestep_ps'])
    if not np.array_equal(binary.timesteps,np.arange(binary.frame_count)*interval):raise ValueError('Ni timeline is not exact 0.1 ps')
    p=part(path,c['timestep_ps']*1000);frames=[[0,i] for i in range(binary.frame_count)]
    s=dict(id=r['id'],dataset=c['campaign_id'],parts=[p],frames=frames,
        times_ps=(binary.timesteps*c['timestep_ps']).tolist(),sampled_frames=list(range(len(frames))),
        views={'010':list(range(len(frames)))},frame_count=len(frames),atom_count=binary.atom_count,
        material='Ni',potential=['ni-mendelev2012-eam'],ancestry_group=r['id'],role=r['role'],
        origin='independent_melt_cooling_and_npt_hold',native_cadence_ps=.1,temperature_K=r['temperature_K'])
    s['input_identity']=digest(dict(arrays=p['array_sha256'],frames=frames))
    return s


def plans(c,sources,coverage):
    parent=dict(config=extraction_config(c),sources=sources,coverage=coverage);parent['identity']=digest(parent)
    plan=dict(config=track_config(c),parent_identity=parent['identity'],producer_files=check_metric_docs(family='ni_births')['ni_births']['files'])
    plan['identity']=digest(plan)
    return dict(parent=parent,tracking=plan,campaign=c)


def prepare(c):
    root=resolve_path(c['analysis']);archive=resolve_path(c['archive']);records=json.loads((resolve_path(c['launch'])/'plan.json').read_text())['records']
    sources=[];coverage=[]
    for r in records:
        folder=archive/r['id']
        if not (folder/'complete.json').exists():
            status=json.loads((folder/'outcome.json').read_text()) if (folder/'outcome.json').exists() else dict(state='not_started')
            coverage.append(dict(source=r['id'],state=status['state'],frames=0,temperature_K=r['temperature_K']));continue
        s=source(c,r);sources.append(s)
        coverage.append(dict(source=r['id'],state='complete',frames=s['frame_count'],temperature_K=r['temperature_K']))
    payload=plans(c,sources,coverage);target=root/'technical/plans.json'
    if target.exists() and json.loads(target.read_text())!=payload:raise ValueError('Ni search population changed; use a new analysis revision')
    write_json(target,payload)
    return payload


def resume_plan(c,expected_sha256):
    """Resume an explicitly pinned population, retaining original chunk identities."""
    path=resolve_path(c['analysis'])/'technical/plans.json'
    if sha(path)!=expected_sha256:raise ValueError(f'Ni resume plan changed: {path}')
    return resume_payload(c,json.loads(path.read_text()))


def resume_payload(c,payload):
    if payload['campaign']!=c:raise ValueError('Ni resume recipe differs from the pinned campaign')
    for key in ('parent','tracking'):
        plan=payload[key]
        if digest({k:v for k,v in plan.items() if k!='identity'})!=plan['identity']:
            raise ValueError(f'Changed Ni {key} identity')
    if payload['parent']['config']!=extraction_config(c) or payload['tracking']['config']!=track_config(c):
        raise ValueError('Ni resumed descriptor or lineage definitions changed')
    if payload['tracking']['parent_identity']!=payload['parent']['identity']:
        raise ValueError('Ni resumed parent identity mismatch')
    return payload


def analyse(c,*,plan_sha256=None,workers=None,wait_for_jobs=()):
    from .ni_queue import run
    return run(c,workers=workers or c['analysis_workers'],plan_sha256=plan_sha256,wait_for_jobs=wait_for_jobs)


def collect(c,payload):
    root=resolve_path(c['analysis']);sources=payload['parent']['sources']
    events=[];counts=[];dest=root/'analyses/births-v1'
    for s in sources:
        for detector in payload['parent']['config']['detectors']:
            folder=root/'technical/sources'/s['id']/'010'/detector
            receipt=json.loads((folder/'complete.json').read_text())
            execution=payload
            if 'source_plans' in payload:
                ref=payload['source_plans'][s['id']]
                if sha(ref['path'])!=ref['sha256']:raise ValueError('Changed Ni source plan')
                execution=resume_payload(c,json.loads(Path(ref['path']).read_text()))
            task=dict(source=s['id'],cadence='010',detector=detector)
            if receipt['identity']!=digest(dict(plan=execution['tracking']['identity'],task=task)):
                raise ValueError('Ni tracking receipt belongs to a different plan')
            if any(sha(folder/n)!=v for n,v in receipt['files'].items()):raise ValueError('Changed Ni tracking result')
            rows=json.loads((folder/'events.json').read_text());summary=Counter()
            for row in rows:
                phase='cooling' if row['first_time_ps']<c['cooling_ps'] else 'isothermal'
                e={k:v for k,v in row.items() if k!='first_members'}
                e.update(material='Ni',target_temperature_K=s['temperature_K'],onset_phase=phase,
                         onset_time_from_hold_ps=row['first_time_ps']-c['cooling_ps'])
                events.append(e)
                summary[(phase,row['origin'],'established' if row['establishment_confirmation_ps'] is not None else 'not_established')]+=1
            for (phase,origin,state),count in summary.items():
                counts.append(dict(source=s['id'],temperature_K=s['temperature_K'],detector=detector,phase=phase,origin=origin,establishment=state,count=count))
    write_metric_rows(payload['parent']['coverage'],dest,family='ni_births',name='coverage')
    if events:write_metric_rows(events,dest,family='ni_births',name='events',columns=tuple(dict.fromkeys(k for e in events for k in e)))
    if counts:write_metric_rows(counts,dest,family='ni_births',name='counts')
    result=dict(state='complete' if len(sources)==len(payload['parent']['coverage']) else 'partial_campaign_analysed',
        completed_sources=len(sources),planned_sources=len(payload['parent']['coverage']),event_records=len(events),
        established_ptm_liquid_births=sum(e['detector']=='ptm' and e['origin']=='emergence_from_liquid' and e['establishment_confirmation_ps'] is not None for e in events),
        ancestry='One independent preparation seed per source; detector records and cluster families are not additional independent melts')
    write_json(root/'technical/complete.json',result)
    (dest/'README.md').write_text('# Ni nucleus-birth search\n\n'+json.dumps(result,indent=2)+'\n\n[Coverage](tables/coverage.csv) · [Definitions](tables/METRICS.md)\n\nCooling and constant-temperature births remain separate. The 64-atom/1.5-ps establishment rule is operational, not a measured critical nucleus.\n')
    print(json.dumps(result),flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'analyse', 'combine'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--plan-sha256', help='Resume exactly this existing plans.json checksum; analyse only')
    parser.add_argument('--workers',type=int,help='Allocated one-thread CPU workers')
    parser.add_argument('--wait-for-jobs',nargs='+',default=[],help='MD jobs to observe while completed sources become available')
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    if args.plan_sha256 and args.stage!='analyse':parser.error('--plan-sha256 requires analyse')
    if args.stage=='analyse':analyse(config,plan_sha256=args.plan_sha256,workers=args.workers,wait_for_jobs=args.wait_for_jobs)
    else:{'prepare': prepare, 'combine': combine}[args.stage](config)


def combine(c):
    """Publish both independent source populations from their verified birth tables."""
    from src.simulation.campaigns.ni_births.workflow import extension_parent
    original=extension_parent(c);root=resolve_path(c['combined_analysis'])
    plans=[original,json.loads((resolve_path(c['launch'])/'plan.json').read_text())]
    combined={'coverage':[],'events':[],'counts':[]};inputs=[];seen=set();statuses=[];definitions=None
    for plan in plans:
        config=plan['config'];component=resolve_path(config['analysis']);bundle=component/'analyses/births-v1'
        result=json.loads((component/'technical/complete.json').read_text())
        prepared=json.loads((component/'technical/plans.json').read_text())
        if prepared['campaign']!=config:raise ValueError(f'Changed Ni analysis recipe: {component}')
        current=dict(extraction={k:v for k,v in prepared['parent']['config'].items() if k!='cache'},
                     tracking={k:v for k,v in prepared['tracking']['config'].items() if k!='output'})
        if definitions is not None and current!=definitions:raise ValueError('Ni component event definitions differ')
        definitions=current;source_ids={r['id'] for r in plan['records']}
        if seen & source_ids:raise ValueError('Repeated independent source in Ni combined collection')
        seen.update(source_ids);statuses.append(result)
        record=dict(dataset=config['campaign_id'],plan_identity=plan['identity'],analysis=str(component),
            completion_sha256=sha(component/'technical/complete.json'),files={})
        for name in combined:
            path=bundle/'tables'/f'{name}.csv'
            if not path.exists():
                if name=='coverage' or result['event_records']!=0:raise FileNotFoundError(path)
                continue
            binding=json.loads((bundle/'technical/table-contracts'/f'{name}.json').read_text())
            if binding['family']!='ni_births' or sha(path)!=binding['sha256']:raise ValueError(f'Changed Ni table: {path}')
            contract_path=bundle/binding['contract'];contract=json.loads(contract_path.read_text())
            if contract['family']!='ni_births':raise ValueError('Ni table has a foreign metric contract')
            # Record the exact historical implementations of each input table;
            # no event extraction or metric recalculation is performed here.
            record['files'][name]=dict(path=str(path),sha256=binding['sha256'],
                                      metric_contract=str(contract_path),contract_sha256=sha(contract_path))
            rows=list(csv.DictReader(path.open()))
            if any(r['source'] not in source_ids for r in rows):raise ValueError('Foreign source in Ni event table')
            if name=='coverage' and (len(rows)!=len(source_ids) or {r['source'] for r in rows}!=source_ids):
                raise ValueError('Ni source coverage is incomplete or duplicated')
            combined[name].extend(dict(row,dataset=config['campaign_id']) for row in rows)
        inputs.append(record)
    provenance=dict(components=inputs,definitions=definitions,planned_independent_melts=len(seen))
    receipt=root/'technical/inputs.json'
    if receipt.exists() and json.loads(receipt.read_text())!=provenance:raise ValueError('Ni combined inputs changed; use a new revision')
    write_json(receipt,provenance)
    for name,rows in combined.items():
        if rows:write_metric_rows(rows,root,family='ni_births',name=name,columns=tuple(dict.fromkeys(k for row in rows for k in row)))
    result=dict(state='complete' if all(s['state']=='complete' for s in statuses) else 'partial_campaign_analysed',
        completed_sources=sum(s['completed_sources'] for s in statuses),planned_sources=len(seen),
        event_records=sum(s['event_records'] for s in statuses),
        established_ptm_liquid_births=sum(s['established_ptm_liquid_births'] for s in statuses),
        component_datasets=[p['config']['campaign_id'] for p in plans],
        operation='Concatenate verified component tables with unique source identities; no recomputation or resplitting')
    write_json(root/'technical/complete.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
