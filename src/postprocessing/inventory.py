"""Verify completed Al MEAM production runs before a lineage search."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import shlex

import numpy as np

from src.experiment_runner.artifacts import file_hash as sha, write_json
from src.project_runtime.dataset_registry import build_registry
from src.project_runtime.paths import REPO, machine, resolve_path, path_snapshot
from src.postprocessing.io import read
from src.postprocessing.sources import prepare


def commands(path):
    names={'pair_style','pair_coeff','read_data','read_restart','fix','DENSE_AL_COMPLETE'}
    return [shlex.split(line, comments=True) for line in path.read_text().splitlines()
            if line.split() and line.split()[0] in names
            and line.split()[:3]!=['read_data','CPU','=']]


def audit_run(identifier, entry, directory, potential, protocol, *, campaign_run=None):
    """Bind producer intent, executed commands, potential bytes and liquid origin."""
    def require(condition, detail):
        if not condition:
            raise ValueError(f'{identifier}: {detail}')

    meta=read(directory/'input_metadata.json');outcome=read(directory/'outcome.json')
    require(meta['run_id']==identifier and outcome['run_id']==identifier, 'run identity differs')
    require(meta['protocol']==protocol and outcome['termination']['protocol']==protocol, 'unexpected MD protocol')
    require(outcome['state']=='complete' and outcome['phase']=='converted', 'production not sealed')
    require(meta['material']=='Al' and meta['ensemble']=='NPT', 'unexpected material/ensemble')
    require(meta['timestep_ps']==.002 and meta['sample_interval_steps']==50 and meta['sampling_ps']==.1,
            'expected native 2 fs integration and 0.10 ps observations')
    require(meta['root_lineage']==entry['metadata']['ancestry'] and
            meta['root_lineage']==f'independent_melt_{meta["preparation_seed"]}', 'melt ancestry differs')
    deck=commands(directory/'source.in.lammps');executed=commands(directory/'equilibration.lammps.log')
    expected_pair=['pair_coeff','*','*','potential/Lee2003_Al.library.meam','Al','potential/Lee2003_Al.meam','Al']
    gpu = campaign_run is not None
    for label,lines in [('input',deck), *([] if gpu else [('executed equilibration',executed)])]:
        require([x for x in lines if x[0]=='pair_style']==[['pair_style','meam']], f'{label}: not native MEAM')
        require([x for x in lines if x[0]=='pair_coeff']==[expected_pair], f'{label}: unexpected potential mapping')
        require([x for x in lines if x[0] in ('read_data','read_restart')]==[['read_data','prepared_liquid.lammps.data']],
                f'{label}: not the declared prepared-liquid start')
        fixes=[x for x in lines if x[0]=='fix']
        require(len(fixes)==2 and {(x[1],x[3]) for x in fixes}=={('remove_drift','momentum'),('ensemble','npt')},
                f'{label}: unexpected restraint/control fixes')
    execution_evidence = []
    if gpu:
        require(meta['execution_backend']=='torch_meam_lammps_iso_npt_v1', 'unexpected GPU adapter')
        require(meta['sampling_protocol']=='prepared_liquid_velocity_reshoot_v1', 'not a liquid velocity reshoot')
        for key in ('run_id','protocol','root_lineage','split','preparation_seed','velocity_seed',
                    'shot_index','collection_id','sampling_protocol','execution_backend','stopping'):
            require(meta[key]==campaign_run[key], f'campaign provenance differs: {key}')
        for name, expected in campaign_run['input_sha256'].items():
            require(sha(directory/name)==expected, f'campaign input checksum differs: {name}')
        expected_execution = [
            ['pair_style','zero','1.0'], ['pair_coeff','*','*'],
            ['fix','gpu_meam','all','external','pf/callback','1','1'],
        ]
        require(executed==[deck[0], *expected_execution, *deck[3:]],
                'executed commands differ from GPU callback plus declared NPT/drift fixes')
        execution=read(directory/'technical/gpu-execution.json')
        require(execution['state']=='complete', 'GPU execution incomplete')
        for field, expected in [('force_dtype','float32'),('integration_dtype','float64')]:
            require(execution[field]==meta['execution'][field]==expected, f'GPU precision differs: {field}')
        require(meta['execution']['energy_reduction_dtype']=='float64', 'unexpected energy reduction')
        if protocol == 'al_birth_focused_010ps_v1':
            require(meta['equilibration_steps']==0 and meta['observation_start_step']==0
                    and meta['initial_settling_steps_recorded']==7500,
                    'birth-focused observations must include initialization and settling')
            require(outcome['termination']['recorded_from_initialization'] is True,
                    'birth-focused termination does not certify the full initial prefix')
        else:
            require(meta['equilibration_steps']==7500, 'unexpected unrecorded equilibration duration')
        validation=Path(meta['execution']['validation'])
        require(read(validation)['state']=='passed', 'recorded GPU numerical audit did not pass')
        execution_evidence=[str(directory/'technical/gpu-execution.json'),str(validation)]
    require(['DENSE_AL_COMPLETE',identifier] in commands(directory/'measurement.lammps.log'), 'missing executed completion marker')
    potential_files={}
    for item in potential['files']:
        path=directory/'potential'/Path(item['path']).name;observed=sha(path)
        require(observed==item['sha256'], f'potential checksum differs: {path}')
        potential_files[str(path)]=observed
    for name in ('prepared_liquid.lammps.data','melt_final.restart.bin'):
        require(sha(directory/name)==meta['parent_input_sha256'][name], f'prepared liquid provenance differs: {name}')
    binary=directory/'trajectory_binary_float16';manifest=read(binary/'manifest.json')
    conversion=read(directory/'paired_conversion.json')
    require(manifest['state']=='complete' and conversion['state']=='complete', 'binary conversion incomplete')
    require(sha(binary/'manifest.json')==conversion['float16_manifest_sha256'], 'converted manifest differs')
    require(manifest['frame_count']==outcome['frame_count']==conversion['frame_count'], 'frame count differs')
    require(manifest['atom_count']==meta['atom_count']==conversion['atom_count'], 'atom count differs')
    steps=np.load(binary/'timesteps.npy',mmap_mode='r',allow_pickle=False)
    require(len(steps)==manifest['frame_count'] and len(steps)>1 and steps[0]==0 and np.all(np.diff(steps)==50),
            'not an exact native 0.10 ps timeline starting at production time zero')
    require(int(steps[-1])==outcome['termination']['final_step'], 'binary does not reach declared final step')
    evidence=['input_metadata.json','outcome.json','source.in.lammps','equilibration.lammps.log',
              'measurement.lammps.log','paired_conversion.json','trajectory_binary_float16/manifest.json']
    return dict(dataset=identifier,source_directory=str(directory),purpose=entry['metadata']['role'],
                producer_protocol=protocol,potential='al-lee2003-meam',
                engine=meta['execution']['engine'] if gpu else 'native LAMMPS MEAM',
                ensemble='NPT',ancestor=meta['root_lineage'],parent_dataset=meta['parent_dataset'],
                parent_directory=meta['parent_directory'],split=meta['split'],temperature_K=meta['temperature_K'],
                frames=len(steps),atoms=meta['atom_count'],duration_ps=float(steps[-1])*.002,
                native_cadence_ps=.1,potential_files=potential_files,prepared_liquid_sha256=meta['parent_input_sha256'],
                observation_start_after_equilibration_ps=meta['equilibration_steps']*.002,
                execution=meta['execution'] if gpu else {'engine':'native LAMMPS MEAM'},
                evidence={str(directory/n):sha(directory/n) for n in evidence+execution_evidence})


@path_snapshot()
def run(config_path):
    c=read(config_path);root=resolve_path(c['output']);inventory=root/'technical/inventory'
    if (root/'technical/plan.json').exists():raise ValueError('Population already prepared; use a new release')
    settings=machine();catalog_path=Path(settings['catalog'])
    if not catalog_path.is_absolute():catalog_path=REPO/catalog_path
    catalog=read(catalog_path);request=c['production_inventory'];old_sources=[];parents=[]
    for reference in request['previous_plans']:
        path=resolve_path(reference);p=read(path);old_sources.extend(p['sources'])
        parents.append(dict(path=str(path),sha256=sha(path),identity=p['identity']))
    known={s['dataset'] for s in old_sources};known_inputs={s['input_identity'] for s in old_sources}
    potential=catalog['potential_registry'][request['potential_id']]
    requested=request.get('source_ids')
    if requested is not None and (not requested or len(requested)!=len(set(requested)) or set(requested)-catalog['datasets'].keys()):
        raise ValueError('source_ids must be a nonempty unique list of registered datasets')
    campaign_runs={};campaign_evidence=None
    if request.get('campaign_manifest'):
        if requested is None:raise ValueError('GPU shooting inventory requires explicit source_ids')
        campaign_path=resolve_path(request['campaign_manifest']);campaign=read(campaign_path)
        campaign_runs={r['run_id']:r for r in campaign['runs']}
        if not set(requested)<=campaign_runs.keys():raise ValueError('Requested sources absent from campaign')
        campaign_evidence=dict(path=str(campaign_path),sha256=sha(campaign_path))
    selected={};coverage=[];audits=[]
    for identifier,entry in sorted(catalog['datasets'].items()):
        meta=entry.get('metadata',{})
        if 'Al' not in meta.get('materials',[]):continue
        if requested is not None and identifier not in requested:
            coverage.append(dict(dataset=identifier,state='outside_explicit_source_selection',role=meta.get('role')))
            continue
        reason=('already_in_prior_release' if identifier in known else
                'outside_production_role' if meta.get('role')!='raw_dynamics' else
                'outside_native_010_production_protocol' if meta.get('protocol')!=request['producer_protocol'] or meta.get('sampling_ps')!=.1 else
                'production_not_complete' if entry.get('state')!='complete' else
                'different_or_unknown_potential' if meta.get('potential_ids')!=[request['potential_id']] else None)
        if reason:
            if requested is not None:raise ValueError(f'{identifier}: requested source rejected: {reason}')
            coverage.append(dict(dataset=identifier,state=reason,role=meta.get('role')));continue
        directory=(Path(settings['roots'][entry['root']])/entry['path']).resolve()
        audit=audit_run(identifier,entry,directory,potential,request['producer_protocol'],
                        campaign_run=campaign_runs.get(identifier))
        selected[identifier]=entry;audits.append(audit)
        print(f'VERIFIED {identifier}: MEAM production, {audit["frames"]} frames',flush=True)
    if not selected:raise ValueError('No new completed verified MEAM production trajectories')
    write_json(inventory/'production-audit.json',dict(generated_at=datetime.now(timezone.utc).isoformat(),
        catalog_sha256=sha(catalog_path),implementation_sha256=sha(Path(__file__)),previous_plans=parents,
        campaign_manifest=campaign_evidence,requested_sources=requested,
        verified_sources=audits,exclusions=coverage,independence='Source melt ancestry retained; not new independent sources relative to historical data'))
    subset=dict(catalog,datasets={**selected,'potentials':catalog['datasets']['potentials']},
        potential_registry={request['potential_id']:potential},registry_discovery_roots=[])
    write_json(inventory/'catalog.json',subset)
    build_registry(inventory,settings=settings,catalog_path=inventory/'catalog.json')
    registry=read(inventory/'registry.json')
    for d in registry['datasets']:d['records_file']=str(inventory/d['records_file'])
    registry.update(selection_audit=coverage,production_audit_sha256=sha(inventory/'production-audit.json'),
        refresh_scope='Fresh producer/header inventory of the verified completed MEAM sources in production-audit.json; other Al collections explicitly excluded or already searched.')
    write_json(resolve_path(c['registry']),registry)
    p,summary=prepare(c)
    if {s['dataset'] for s in p['sources']}!=set(selected):raise ValueError('Prepared population differs from verified production sources')
    if any(s['input_identity'] in known_inputs for s in p['sources']):raise ValueError('Prior input duplicated')
    for s in p['sources']:
        if '030' in s['views'] and '010' in s['views'] and s['views']['030']!=s['views']['010'][::3]:
            raise ValueError('Incorrect paired cadence')
    write_json(root/'technical/production-verification.json',dict(plan_identity=p['identity'],
        audit_sha256=registry['production_audit_sha256'],previous_plans=parents,sources=len(p['sources']),
        input_duplicates=0,temperature_counts=dict(Counter(a['temperature_K'] for a in audits)),
        ancestors=len({a['ancestor'] for a in audits}),
        frames=sum(a['frames'] for a in audits),duration_ps=sum(a['duration_ps'] for a in audits)))
    return summary


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',required=True)
    print(json.dumps(run(parser.parse_args().config),indent=2))


if __name__=='__main__':main()
