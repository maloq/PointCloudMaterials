"""Relaxed-MACE structured-context preparation, with precise full-cell provenance.

This protocol keeps original MD outcomes but relaxes every encoder observation,
including future latent targets. It never recovers local clouds from float16 cells.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
import torch

from src.project_runtime.paths import resolve_path, dataset_path
from src.data.structural_pretraining.prepare import file_hash, save_json, digest
from src.data.structural_pretraining.support import REFERENCE_RADIUS, support_weights
from src.data.structural_pretraining.batches import collate, move
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.data.relaxed_targets.worker import AbsolutePositions, publish, verify_archive, lock
from src.data.conversion.relaxation import read_relaxed, convert
from src.simulation.relaxation import relax_frame
from src.research.relaxed_encoder.prepare import settings, paired_clouds, graph_arrays
from src.research.relaxed_encoder.accelerated import accelerator_settings
from .geometry import stencil, representatives


def freeze(config):
    root = resolve_path(config['output'])/'technical'
    root.mkdir(parents=True, exist_ok=True)
    path = root/'plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if plan['relaxed_config'] != config:
            raise ValueError('Frozen relaxed context configuration changed')
        return plan
    reference = resolve_path(config['reference_plan'])
    plan = json.loads(reference.read_text())
    training = json.loads(resolve_path(config['relaxation_plan']).read_text())
    selection = json.loads(resolve_path(config['selection']).read_text())
    checkpoint = resolve_path(training['config']['output'])/'technical/runs'/selection['name']/'best.pt'
    from src.research.relaxed_encoder.prepare import arm_for_name
    if arm_for_name(training['config'], selection['name']) != 'relaxed_to_relaxed':
        raise ValueError('Selected encoder was not trained with relaxed inputs and targets')
    if training['config']['scale'] != plan['scale']:
        raise ValueError('Relaxed encoder material scale differs from forecast scale')
    manifest = json.loads(resolve_path(training['config']['normalization_manifest']).read_text())
    exposed = set(manifest['train_roots'] + manifest['selection_roots'])
    exposed |= {s['lineage'] for s in training['sources'] if s['pilot_fit']}
    protected = {s['lineage'] for s in plan['sources'] if s.get('validation_role', s['split']) in ('test','calibration')}
    if exposed & protected:
        raise ValueError(f'Relaxed encoder ancestry leakage: {exposed & protected}')
    plan['relaxed_config'] = config
    plan['relaxed_encoder'] = dict(checkpoint=str(checkpoint), sha256=file_hash(checkpoint), selection=selection,
        protected_overlap=[], support='80 nearest observed atom identities, retained after quench; radius 8 normalized')
    plan['structured_config'] = dict(plan['structured_config'], output=config['output'], context_cache=config['context_cache'])
    plan['config'] = dict(plan['config'], output=config['output'])
    plan['structured_identity'] = digest(dict(config=config, reference=file_hash(reference), encoder=plan['relaxed_encoder']))
    plan['identity'] = digest(plan)
    save_json(path, plan)
    tasks = []
    for spec in json.loads((reference.parent/'queue.json').read_text()):
        if spec['encoder'] != 'mace':
            continue
        spec = copy.deepcopy(spec)
        spec['name'] = 'relaxed-' + spec['name']
        tasks.append(spec)
    save_json(root/'queue.json', tasks)
    return plan


def load_encoder(plan, device):
    from src.training_methods.neighborhood_jepa.regularization.model import Encoder
    record = plan['relaxed_encoder']
    path = resolve_path(record['checkpoint'])
    if file_hash(path) != record['sha256']:
        raise ValueError('Selected relaxed checkpoint changed')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = Encoder(saved['manifest']['config']['encoder_channels'], saved['spec']['export_norm'])
    model.load_state_dict({k.removeprefix('encoder.'):v for k,v in saved['model'].items() if k.startswith('encoder.')}, strict=True)
    return model.to(device).eval().requires_grad_(False)


def samples(clouds, scale):
    arrays, _ = graph_arrays(clouds, scale)
    result = []
    for g in range(len(arrays['offsets'])-1):
        lo, hi = arrays['offsets'][g:g+2]
        elo, ehi = arrays['edge_offsets'][g:g+2]
        x = arrays['positions'][lo:hi]
        result.append(dict(positions=x[None], weights=support_weights(x)[None], center=0,
            times=np.array([0.],np.float32), species=1, log_scale=np.log(scale/REFERENCE_RADIUS),
            physical=np.zeros(85,np.float32), tda=np.zeros(144,np.float32), tda_valid=False,
            edges=arrays['edges'][:,elo:ehi].astype(np.int64)))
    return result


@torch.no_grad()
def encode(model, clouds, scale, device='cuda', batch_size=128):
    graphs = samples(clouds, scale)
    values = []
    for start in range(0,len(graphs),batch_size):
        batch = move(collate(graphs[start:start+batch_size],'mace'),device)
        # Same invariant prefix as the checkpoint's frozen relaxed assay producer.
        with torch.autocast(torch.device(device).type,dtype=torch.bfloat16,enabled=device!='cpu'):
            z = model(batch).float()[:,:128]
        if not torch.isfinite(z).all() or z.shape != (min(batch_size,len(graphs)-start),128):
            raise ValueError(f'Invalid relaxed invariant export: {z.shape}')
        values.append(z.cpu().numpy())
    return np.concatenate(values)


def observed_information(cold, box, centers):
    """Same geometric descriptor definitions as the reference, evaluated cold.

    Only velocity-independent columns are exported. Zero velocities are temporary
    arguments to the original packet producer, never claimed as measured motion.
    """
    from src.analysis.liquid_structure import bond_order
    from src.data.predictive_memory.targets import physical_packet
    from src.research.local_predictability.data import shell_features
    from src.research.context_night.context import LOCAL_COLUMNS
    points = np.mod(cold,box)
    tree = cKDTree(points,boxsize=box)
    _, nearest = tree.query(points[centers],k=13)
    _, neighbors = tree.query(points[nearest],k=13)
    bonds = points[neighbors[:,:,1:]]-points[nearest][:,:,None,:]
    bonds -= box*np.rint(bonds/box)
    order, _ = bond_order(bonds,3.5)
    values = []
    for i, center in enumerate(centers):
        rows = tree.query_ball_point(points[center],25.)
        x = points[rows]-points[center]
        x -= box*np.rint(x/box)
        x = x.astype(np.float32)
        u = np.zeros_like(x)
        local = np.r_[physical_packet(x,u),order[i]][LOCAL_COLUMNS]
        values.append(np.r_[local,shell_features(x,u)[[0,1,6,7]]])
    result = np.array(values,np.float32)
    if result.shape != (len(centers),97) or not np.isfinite(result).all():
        raise ValueError('Invalid relaxed geometric auxiliary inputs')
    return result


def extract_cell(plan, source, frame, archive, model, device='cuda'):
    """Read a verified precise archive; choose symmetric query atoms in cold geometry."""
    verify_archive(archive)
    cold, meta = read_relaxed(archive)
    raw = ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
    if file_hash(raw.root/'manifest.json') != source['manifest_sha256']:
        raise ValueError('Raw trajectory changed')
    if meta['source_manifest_sha256'] != source['manifest_sha256'] or meta['source_frame'] != frame:
        raise ValueError('Relaxed full-cell source/frame identity differs')
    if meta['fmax_eV_per_A'] > .01 or meta['settings']['minimizer'] != 'fire':
        raise ValueError('Relaxation does not satisfy the encoder training protocol')
    training = json.loads(resolve_path(plan['relaxed_config']['relaxation_plan']).read_text())
    expected = training['config']['potential_sha256']
    if ([file_hash(p) for p in meta['settings']['potential_files']] != expected
            or [meta['potential_checksums'][p] for p in meta['settings']['potential_files']] != expected):
        raise ValueError('Relaxed cell generating potential differs')
    low = raw.box_low[frame].astype(float)
    box = raw.box_high[frame].astype(float)-low
    cold = np.mod(cold-low,box)
    hot = raw.positions[frame].astype(float)
    centers = np.searchsorted(raw.atom_ids,source['center_atom_ids'])
    np.testing.assert_array_equal(raw.atom_ids[centers],source['center_atom_ids'])
    tree = cKDTree(cold,boxsize=box)
    queries = stencil(plan['structured_config']['shell_radii_A'])
    if frame <= 664:
        assigned = [representatives(cold,c,tree,box,queries,plan['structured_config']['max_query_offset_A']) for c in centers]
        ids = np.stack([a for a,r in assigned]); relative = np.stack([r for a,r in assigned])
    else:
        ids = centers[:,None]; relative = np.zeros((len(centers),1,3),np.float32)
    unique, mapping = np.unique(ids,return_inverse=True)
    _, clouds, neighbor_rows = paired_clouds(hot,cold,box,unique)
    features = encode(model,clouds,plan['scale'],device,plan['relaxed_config']['extraction_batch'])[mapping].reshape(*ids.shape,128)
    folder = resolve_path(plan['relaxed_config']['context_cache'])/'cells'/f'{source["id"]}-{frame}'
    folder.mkdir(parents=True,exist_ok=True)
    # Centered float32 observations saved before any full-cell precision reduction.
    np.savez(folder/'observations.npz',clouds=clouds,neighbor_atom_ids=raw.atom_ids[neighbor_rows],
        query_atom_ids=raw.atom_ids[ids],unique_query_atom_ids=raw.atom_ids[unique],mapping=mapping.reshape(ids.shape),
        features=features,relative=relative,information=observed_information(cold,box,centers))
    receipt = dict(identity=plan['structured_identity'],source=source['id'],frame=frame,
        checkpoint_sha256=plan['relaxed_encoder']['sha256'],archive=str(archive),
        precise_dump_sha256=file_hash(Path(archive)/'relaxed.dump'),relaxation=meta,
        observations_sha256=file_hash(folder/'observations.npz'))
    save_json(folder/'complete.json',receipt)
    return receipt


def prepare_cell(plan,source,frame,model,profile,device='cuda'):
    c = plan['relaxed_config'];key=f'{source["id"]}-{frame}'
    training = json.loads(resolve_path(c['relaxation_plan']).read_text())
    work = resolve_path(c['scratch'])/'cells'/key
    archive = resolve_path(c['archive'])/'cells'/key
    if not archive.exists():
        raw = ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
        if file_hash(raw.root/'manifest.json') != source['manifest_sha256']:
            raise ValueError('Trajectory changed before relaxation')
        for p,h in zip(training['potential_files'],training['config']['potential_sha256'],strict=True):
            if file_hash(p) != h:raise ValueError('Generating potential changed')
        absolute=SimpleNamespace(**vars(raw),atom_count=raw.atom_count)
        absolute.positions=AbsolutePositions(raw)
        execution=accelerator_settings(settings(training,1),profile)
        if not (work/'metadata.json').exists():
            relax_frame(absolute,frame,work,execution)
        # Retain precise dump in the archive until observations have been exported.
        if not (work/'conversion.json').exists():
            convert(work,delete_source=False,local_cloud_dtype='float32')
        publish(work,archive)
    return extract_cell(plan,source,frame,archive,model,device)


def assemble(plan,source):
    cache=resolve_path(plan['relaxed_config']['context_cache']);folder=cache/str(source['id'])
    folder.mkdir(parents=True,exist_ok=True)
    values={k:[] for k in ('relative','mace_features','mace_center','information')}
    for frame in range(0,793,4):
        cell=cache/'cells'/f'{source["id"]}-{frame}'
        receipt=json.loads((cell/'complete.json').read_text())
        if receipt['identity'] != plan['structured_identity'] or file_hash(cell/'observations.npz') != receipt['observations_sha256']:
            raise ValueError(f'Relaxed cell changed: {cell}')
        with np.load(cell/'observations.npz') as a:
            values['mace_center'].append(a['features'][:,0])
            if frame<=664:
                values['relative'].append(a['relative']);values['mace_features'].append(a['features']);values['information'].append(a['information'])
    for k,v in values.items():np.save(folder/f'{k}.npy',np.stack(v))
    save_json(folder/'complete.json',dict(identity=plan['structured_identity'],files={f'{k}.npy':file_hash(folder/f'{k}.npy') for k in values}))


def check_heads(plan,source,frame):
    """Execution only: repeated real cold inputs with synthetic future targets."""
    from src.research.crystallization_paths.runtime import make_model
    root=resolve_path(plan['relaxed_config']['output'])/'technical'
    cell=resolve_path(plan['relaxed_config']['context_cache'])/'cells'/f'{source}-{frame}'
    with np.load(cell/'observations.npz') as a:
        features=torch.tensor(np.tile(a['features'][:4,None],(1,4,1,1)).reshape(4,100,128),device='cuda')
        r=np.tile(a['relative'][:4,None],(1,4,1,1))
    dt=np.broadcast_to(np.array([-48,-12,-3,0])[None,:,None,None],(4,4,25,1))
    geometry=torch.tensor(np.concatenate((r,dt),-1).reshape(4,100,4),device='cuda',dtype=torch.float32)
    obs=dict(features=features,geometry=geometry,condition=torch.zeros(4,7,device='cuda'),information=torch.zeros(4,504,device='cuda'))
    torch.manual_seed(20260921)
    target=dict(state=torch.randn(4,32,265,device='cuda'),present=torch.randn(4,265,device='cuda'),
        event=torch.tensor([3,15,64,128],device='cuda'),occurred=torch.zeros(4,32,4,device='cuda'))
    results=[]
    for spec in json.loads((root/'queue.json').read_text()):
        spec=dict(spec,training_event_cdf=np.linspace(.001,.5,128).tolist())
        model=make_model(spec).cuda();x,w=model.context.inputs(features,geometry);model.context.normalization.calibrate(x,w)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        loss=model.loss(obs,target,0.).mean();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
        model.eval()
        with torch.no_grad():paths,cdf=model.forecast(obs,samples=4,diffusion_steps=4)
        if not torch.isfinite(paths).all() or not torch.isfinite(cdf).all() or torch.any(cdf[:,1:]<cdf[:,:-1]-1e-6):
            raise ValueError(f'Invalid {spec["name"]} smoke rollout')
        results.append(dict(name=spec['name'],loss=float(loss.detach()),gradient_norm=float(norm),path_shape=list(paths.shape)))
    save_json(root/'head-preflight.json',dict(passed=True,checks=results,meaning='Execution only; repeated observed inputs and synthetic targets, not scientific performance.'))


def worker(plan,lane,max_cells=0):
    """Cell-level locks and bounded admission; no GPU waiting for peer producers."""
    from src.training_methods.shared_pretraining.queue import deadline_for_job
    from src.research.relaxed_encoder.accelerated import wait_for_benchmark
    from src.research.crystallization_transfer.runtime import setup
    from src.research.crystallization_paths.runtime import fit
    from .data import StructuredPaths
    setup();root=resolve_path(plan['relaxed_config']['output'])/'technical'
    deadline=deadline_for_job();cache=resolve_path(plan['relaxed_config']['context_cache'])
    status=root/f'lane-{lane}.json'
    def state(value,**kw):save_json(status,dict(state=value,pid=os.getpid(),updated_at=time.time(),**kw))
    try:
        gpu=torch.cuda.get_device_name()
        if 'H100' in gpu:backend='h100'
        elif 'A100' in gpu:backend='a100'
        else:raise ValueError(f'No force-validated relaxation backend for {gpu}')
        config=json.loads(resolve_path('configs/analysis/relaxed_encoder_accelerated.json').read_text())
        profile=wait_for_benchmark(config,backend,deadline,root/f'backend-{lane}.json')
        model=load_encoder(plan,'cuda');done=0
        for source in plan['sources']:
            for frame in range(0,793,4):
                cell=cache/'cells'/f'{source["id"]}-{frame}'
                receipt=cell/'complete.json'
                if receipt.exists():
                    record=json.loads(receipt.read_text())
                    if record['identity']!=plan['structured_identity'] or file_hash(cell/'observations.npz')!=record['observations_sha256']:
                        raise ValueError(f'Corrupt completed cell: {cell}')
                    continue
                if time.time()>deadline-3000 or (max_cells and done>=max_cells):
                    state('checkpointed',completed_cells=done);return
                with lock(cell/'worker.lock') as acquired:
                    if not acquired:continue
                    if receipt.exists():continue
                    state('relaxing',source=source['id'],frame=frame,completed_cells=done)
                    prepare_cell(plan,source,frame,model,profile);done+=1
                    print(json.dumps(dict(source=source['id'],frame=frame,completed_cells=done)),flush=True)
            with lock(cache/str(source['id'])/'assemble.lock') as acquired:
                if acquired and all((cache/'cells'/f'{source["id"]}-{f}'/'complete.json').exists() for f in range(0,793,4)):
                    assemble(plan,source)
        if not all((cache/str(s['id'])/'complete.json').exists() for s in plan['sources']):
            state('preparation_pending',completed_cells=done);return
        del model;torch.cuda.empty_cache();data=None
        for spec in json.loads((root/'queue.json').read_text()):
            folder=root/'runs'/spec['name'];fit_status=folder/'status.json'
            if fit_status.exists() and json.loads(fit_status.read_text())['state']=='complete':continue
            with lock(folder/'worker.lock') as acquired:
                if not acquired:continue
                if time.time()>deadline-1800:state('checkpointed',stage='training');return
                if data is None:data=StructuredPaths(plan,spec)
                save_json(folder/'spec.json',spec);state('training',fit=spec['name'])
                if not fit(plan,spec,data,deadline):state('checkpointed',fit=spec['name']);return
        state('finished_lane',completed_cells=done)
    except Exception as exc:
        state('failed',error=repr(exc),traceback=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','preflight','worker']);p.add_argument('--config',required=True)
    p.add_argument('--lane',default='manual');p.add_argument('--max-cells',type=int,default=0)
    p.add_argument('--archive');p.add_argument('--source',type=int);p.add_argument('--frame',type=int)
    a=p.parse_args();config=json.loads(resolve_path(a.config).read_text());plan=freeze(config)
    if a.stage=='preflight':
        torch.set_num_threads(1)
        source=next(s for s in plan['sources'] if s['id']==a.source)
        result=extract_cell(plan,source,a.frame,Path(a.archive),load_encoder(plan,'cuda'))
        check_heads(plan,a.source,a.frame)
        save_json(resolve_path(config['output'])/'technical/preflight.json',dict(passed=True,cell=result))
        print(json.dumps(dict(passed=True,source=a.source,frame=a.frame)),flush=True)
    elif a.stage=='worker':worker(plan,a.lane,a.max_cells)

if __name__=='__main__':main()
