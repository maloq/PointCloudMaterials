"""Extract measured controls without modifying checkpoints or source trajectories."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.special import sph_harm_y
import torch

from src.analysis.liquid_structure import persistence_image
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.registry import sha256, write_json
from src.project_runtime.paths import load_json, resolve_path
from src.training_methods.embedding_forecast.data import load_snapshot_encoder, source_records


def offsets(points, centers, neighbors, lengths):
    x = points[neighbors] - points[centers, None]
    return x - lengths * np.round(x / lengths)


def frame_clouds(trajectory, frame, centers, *, round_global=False):
    low = trajectory.box_low[frame].astype(np.float64)
    lengths = trajectory.box_high[frame].astype(np.float64) - low
    raw = trajectory.positions[frame]
    if round_global:
        raw = raw.astype(np.float16)
    points = np.mod(raw.astype(np.float64) - low, lengths)
    neighbors = cKDTree(points, boxsize=lengths).query(points[centers], k=84, workers=1)[1]
    np.testing.assert_array_equal(neighbors[:, 0], centers)
    return offsets(points, centers, neighbors, lengths), neighbors, points, lengths


def structural(clouds, cutoff):
    """PTM best-template RMSD/margin, q4/q6, nearest-shell density and distances."""
    from ovito.data import DataCollection, Particles
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    n, k, _ = clouds.shape
    points = clouds.astype(np.float64).copy()
    # Separation greater than twice the maximum cloud diameter.
    points[:, :, 0] += 4 * np.linalg.norm(points, axis=-1).max() * np.arange(n)[:, None]
    data = DataCollection()
    particles = Particles(count=n*k)
    particles.create_property('Position', data=points.reshape(-1, 3))
    data.objects.append(particles)
    # Cutoff zero exposes the best template even for subsequently rejected centers.
    data.apply(PolyhedralTemplateMatchingModifier(rmsd_cutoff=0, output_rmsd=True))
    rmsd = np.asarray(data.particles['RMSD'])[::k].copy()
    best = np.asarray(data.particles['Structure Type'])[::k].copy()
    # No matching template is a valid liquid outcome; zero is OVITO's sentinel,
    # not a perfect structural fit. Preserve this as an explicitly missing margin.
    rmsd[best == 0] = np.nan
    labels = np.where(rmsd <= cutoff, best, 0)
    vec = clouds[:, 1:13].astype(np.float64)
    distances = np.linalg.norm(vec, axis=-1)
    theta = np.arccos(np.clip(vec[..., 2] / distances, -1, 1))
    phi = np.arctan2(vec[..., 1], vec[..., 0])
    q = [np.sqrt(4*np.pi/(2*l+1))*np.linalg.norm(np.stack([
        sph_harm_y(l, m, theta, phi).mean(1) for m in range(-l, l+1)], -1), axis=1)
        for l in (4, 6)]
    return np.column_stack([rmsd, cutoff-rmsd, *q,
        12/(4*np.pi*distances[:, -1]**3/3), distances.mean(1)]), labels


class Assay:
    def __init__(self, config):
        self.cfg = config
        self.root = result_folders(config['output'])
        self.tech = self.root / 'technical'
        self.encoder, self.radius = load_snapshot_encoder(config['checkpoint'], config['device'])
        self.encoder.eval()
        self.rng = np.random.default_rng(config['seed'])

    def encode(self, clouds, encoder=None, batch=None):
        model = self.encoder if encoder is None else encoder
        batch = self.cfg['batch_size'] if batch is None else batch
        # Matches local_views: divide before casting to float32.
        x = (clouds / self.radius).astype(np.float32)
        with torch.inference_mode():
            z = np.concatenate([model(torch.from_numpy(x[i:i+batch]).to(self.cfg['device']))
                .float().cpu().numpy() for i in range(0, len(x), batch)])
        if z.shape != (len(x), 256) or not np.isfinite(z).all():
            raise ValueError(f'Invalid encoder output: {z.shape}')
        return z

    def save(self, name, **arrays):
        path = self.tech / f'{name}.npz'
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, **arrays)
        print(f'SAVED {name}: {len(next(iter(arrays.values())))} rows', flush=True)

    def probes(self):
        cache = Path(self.cfg['training_cache'])
        manifest = load_json(cache/'manifest.json')
        sources = source_records(dict(sources_config=self.cfg['sources_config'], cadence_ps=.75))
        raw_targets = np.load(cache/'targets.npy', mmap_mode='r')
        start = 0
        values = {k: [] for k in ('z', 'hot', 'relaxed', 'geometry', 'source', 'context', 'split', 'temperature', 'atom_id', 'clouds')}
        for context, record in enumerate(manifest['shards']):
            directory = resolve_path(record['source_directory'])
            rows = np.sort(self.rng.choice(record['samples'], self.cfg['probe_centers_per_context'], replace=False))
            path = cache/record['views']
            if sha256(path) != manifest['checksums'][record['views']]:
                raise ValueError(f'Changed training input: {path}')
            x = np.load(path, mmap_mode='r')[rows, 0, -1].astype(np.float32)
            y = np.load(directory/'targets.npy')[rows]
            np.testing.assert_array_equal(y, raw_targets[start+rows])
            start += record['samples']
            source = sources[record['source']]
            if source['split'] != record['split']:
                raise ValueError('Source split mismatch')
            obs, _ = structural(x, self.cfg['ptm_rmsd_cutoff'])
            # Sorted radial distances are a strong simple geometry control.
            geometry = np.column_stack([np.linalg.norm(x[:, 1:], axis=-1), obs[:, 2:]])
            for key, value in dict(z=self.encode(x), hot=np.load(directory/'hot_targets.npy')[rows],
                    relaxed=y, geometry=geometry, source=np.full(len(x), record['source']),
                    context=np.full(len(x), context), split=np.full(len(x), record['split']),
                    temperature=np.full(len(x), record['temperature_K']),
                    atom_id=np.load(directory/'neighbor_ids.npy')[rows, 0], clouds=x).items():
                values[key].append(value)
            print(f'PROBE {context+1}/{len(manifest["shards"])} {record["split"]}', flush=True)
        self.save('probes', **{k:np.concatenate(v) for k,v in values.items()})

    def temporal(self):
        sources = source_records(dict(sources_config=self.cfg['sources_config'], cadence_ps=.75))
        values = {k:[] for k in ('z', 'fixed_z', 'hot', 'observables', 'labels', 'clouds', 'fixed_clouds', 'neighbors', 'source', 'context', 'atom_id')}
        for source_index, source in enumerate(sources):
            if source['split'] != 'test':
                continue
            trajectory = ShootingBinaryTrajectory.load(source['path'])
            centers = np.sort(self.rng.choice(trajectory.atom_count, self.cfg['temporal_centers_per_source'], replace=False))
            for context, anchor in enumerate(source['anchors']):
                frames = np.arange(anchor-8, anchor-8+self.cfg['temporal_frames'])
                np.testing.assert_allclose(np.diff(trajectory.timesteps[frames])*source['timestep_ps'], .75)
                cloud_list, fixed_list, neighbor_list = [], [], []
                for column, frame in enumerate(frames):
                    x, ids, points, lengths = frame_clouds(trajectory, frame, centers)
                    if column == 0:
                        initial = ids[:, :80].copy()
                    cloud_list.append(x[:, :80]); neighbor_list.append(trajectory.atom_ids[ids[:, :80]])
                    fixed_list.append(offsets(points, centers, initial, lengths))
                x = np.stack(cloud_list, 1); fixed = np.stack(fixed_list, 1)
                flat = x.reshape(-1, 80, 3)
                obs, labels = structural(flat, self.cfg['ptm_rmsd_cutoff'])
                n, t = x.shape[:2]
                arrays = dict(z=self.encode(flat).reshape(n,t,256), fixed_z=self.encode(fixed.reshape(-1,80,3)).reshape(n,t,256),
                    hot=np.stack([persistence_image(c) for c in flat]).reshape(n,t,144),
                    observables=obs.reshape(n,t,-1), labels=labels.reshape(n,t), clouds=x,
                    fixed_clouds=fixed, neighbors=np.stack(neighbor_list,1), source=np.full(n,source_index),
                    context=np.full(n,context), atom_id=trajectory.atom_ids[centers])
                for key,value in arrays.items(): values[key].append(value)
                print(f'TEMPORAL source={source_index} context={context}', flush=True)
        self.save('temporal', **{k:np.concatenate(v) for k,v in values.items()})

    def controls(self):
        data = np.load(self.tech/'temporal.npz')
        x = data['clouds'][:, 8]
        z = self.encode(x)
        order = self.rng.permutation(len(x))
        permutation = np.argsort(self.rng.random((len(x), 80)), axis=1)
        rotated = np.einsum('bni,bij->bnj', x, Rotation.random(len(x), random_state=self.rng).as_matrix())
        variants = dict(repeat1=self.encode(x), repeat2=self.encode(x),
            reordered=self.encode(x[order])[np.argsort(order)], singleton=self.encode(x,batch=1),
            batch17=self.encode(x,batch=17), rotation=self.encode(rotated),
            permutation=self.encode(np.take_along_axis(x,permutation[:,:,None],axis=1)),
            translation=self.encode(x+np.array([1.,-2.,.5])),
            local_float16=self.encode(x.astype(np.float16).astype(np.float64)),
            embedding_float16=z.astype(np.float16).astype(np.float32))
        # Same trained tensors and backend; only radial matrix arithmetic changes.
        from src.models.encoders.mace_bf16 import BF16RadialLayer
        originals=[]
        for interaction in self.encoder.mace.backbone.interactions:
            for name, layer in list(interaction.conv_tp_weights.named_children()):
                if not isinstance(layer, BF16RadialLayer):
                    raise TypeError(f'Expected checkpoint compensated radial layer, got {type(layer)}')
                # Keep normalization/activation exactly, replacing only the product.
                originals.append((interaction.conv_tp_weights,name,layer))
                interaction.conv_tp_weights._modules[name] = ExactRadial(layer)
        try:
            variants['radial_fp32'] = self.encode(x)
        finally:
            for module,name,layer in originals:
                module._modules[name]=layer
        for sigma in (.001, .005, .02, .1):
            noise = self.rng.normal(size=x.shape)*sigma/np.sqrt(3)
            noise[:,0] = 0
            variants[f'jitter_{sigma}A'] = self.encode(x+noise)
        self.save('controls', baseline=z, clouds=x, **variants)

    def boundaries(self):
        data = np.load(self.tech/'temporal.npz')
        sources = load_json(self.cfg['sources_config'])['sources']
        original, following, fixed_next, swapped = [], [], [], []
        retained, displacement, source_ids = [], [], []
        for i in range(len(data['source'])):
            s = int(data['source'][i]); source = sources[s]
            tr = ShootingBinaryTrajectory.load(source['path'])
            center = np.searchsorted(tr.atom_ids, data['atom_id'][i:i+1])
            anchor = source['anchors'][int(data['context'][i])]
            x, ids, _, _ = frame_clouds(tr, anchor, center)
            y, jds, points, lengths = frame_clouds(tr, anchor+1, center)
            matched = offsets(points, center, ids[:,:80], lengths)[0]
            original.append(x[0,:80]); following.append(y[0,:80]); fixed_next.append(matched)
            changes=[]
            for count in (1,2,4):
                variant=x[0,:80].copy(); variant[-count:]=x[0,80:80+count]; changes.append(variant)
            swapped.append(changes)
            retained.append(len(np.intersect1d(ids[0,1:80],jds[0,1:80]))/79)
            displacement.append(np.sqrt(np.mean(np.sum((matched[1:]-x[0,1:80])**2,axis=-1))))
            source_ids.append(s)
        x,y,fixed,swap = map(np.array,(original,following,fixed_next,swapped))
        self.save('boundaries', z=self.encode(x), next_z=self.encode(y), fixed_next_z=self.encode(fixed),
            swapped_z=self.encode(swap.reshape(-1,80,3)).reshape(len(x),3,256),
            retained=np.array(retained), displacement_A=np.array(displacement), source=np.array(source_ids),
            hot=np.stack([persistence_image(c) for c in x]),
            swapped_hot=np.stack([persistence_image(c) for c in swap.reshape(-1,80,3)]).reshape(len(x),3,144))

    def siblings(self):
        campaign=Path(self.cfg['sibling_campaign']); manifest=load_json(campaign/'manifest.json')
        branches=manifest['branches']; frames=np.array(self.cfg['sibling_frames'])
        centers=np.sort(self.rng.choice(manifest['atom_count'],self.cfg['sibling_centers'],replace=False))
        data={k:[] for k in ('z','hot','obs','labels','clouds','neighbor_ids','momentum','noise')}
        reference=None
        for branch in branches:
            tr=ShootingBinaryTrajectory.load(campaign/branch['branch_dir']/'trajectory_binary_float32')
            if tr.positions.dtype != np.float32:
                raise ValueError(f'Precision reference must retain original float32: {tr.root}')
            if reference is None:
                reference=tr.positions[0].copy(); ref_ids=tr.atom_ids.copy(); ref_box=tr.box_high[0].copy()
            np.testing.assert_array_equal(tr.positions[0],reference)
            np.testing.assert_array_equal(tr.atom_ids,ref_ids)
            np.testing.assert_array_equal(tr.box_high[0],ref_box)
            clouds=[]; quantized=[]; fixed_quantized=[]; ids=[]
            for frame in frames:
                x,neighbors,_,_=frame_clouds(tr,frame,centers)
                q,qids,points,lengths=frame_clouds(tr,frame,centers,round_global=True)
                clouds.append(x[:,:80]); quantized.append(q[:,:80]); ids.append(tr.atom_ids[neighbors[:,:80]])
                fixed_quantized.append(offsets(points,centers,neighbors[:,:80],lengths))
            x=np.stack(clouds,1); flat=x.reshape(-1,80,3); n,t=x.shape[:2]
            obs,labels=structural(flat,self.cfg['ptm_rmsd_cutoff'])
            for key,value in dict(z=self.encode(flat).reshape(n,t,256),hot=np.stack([persistence_image(c) for c in flat]).reshape(n,t,144),
                obs=obs.reshape(n,t,-1),labels=labels.reshape(n,t),clouds=x,neighbor_ids=np.stack(ids,1),
                momentum=np.array(branch['momentum_index']),noise=np.array(branch['thermostat_replica_index'])).items():
                data[key].append(value)
            if branch['branch_index']==0:
                q=np.stack(quantized,1).reshape(-1,80,3); fq=np.stack(fixed_quantized,1).reshape(-1,80,3)
                qobs,qlabels=structural(q,self.cfg['ptm_rmsd_cutoff'])
                self.save('global_precision', z=self.encode(flat), quantized_z=self.encode(q),fixed_quantized_z=self.encode(fq),
                    hot=np.stack([persistence_image(c) for c in flat]),quantized_hot=np.stack([persistence_image(c) for c in q]),
                    obs=obs,quantized_obs=qobs,labels=labels,quantized_labels=qlabels,
                    displacement_A=np.sqrt(np.mean(np.sum((fq-flat)**2,axis=-1),axis=1)))
            print(f'SIBLING {branch["branch_index"]+1}/{len(branches)}',flush=True)
        self.save('siblings', **{k:np.stack(v) for k,v in data.items()}, time_ps=frames*manifest['protocol']['sample_interval_ps'])


class ExactRadial(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weight=source.weight; self.act=source.act
        self.h_in=source.h_in; self.var_in=source.var_in; self.var_out=source.var_out

    def forward(self,x):
        denominator=(self.h_in*self.var_in/(1 if self.act is not None else self.var_out))**.5
        x=x@(self.weight/denominator)
        return x if self.act is None else self.act(x)*self.var_out**.5


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',nargs='+',choices=['probes','temporal','controls','boundaries','siblings','all'],default=['all'])
    args=parser.parse_args()
    os.environ['OVITO_THREAD_COUNT']='2'
    torch.set_num_threads(4)
    config=load_json(args.config)
    assay=Assay(config)
    identity=dict(config=config,checkpoint_sha256=sha256(Path(config['checkpoint'])),
        source_sha256=sha256(Path(__file__)), source_selection_sha256=sha256(Path(config['sources_config'])))
    provenance=assay.tech/'extraction.json'
    if provenance.exists():
        original=load_json(provenance)
        if any(original[k]!=identity[k] for k in ('config','checkpoint_sha256','source_selection_sha256')):
            raise ValueError('Extraction configuration/checkpoint/source selection changed; choose a fresh output.')
    else:
        write_json(provenance,identity)
    stages=['probes','temporal','controls','boundaries','siblings'] if args.stage==['all'] else args.stage
    for stage in stages:
        if (assay.tech/f'{stage}.npz').exists():
            raise FileExistsError(f'Stage already exists: {stage}')
        started=time.monotonic()
        write_json(assay.tech/f'{stage}.extraction.json',identity)
        (assay.tech/f'{stage}.source.py').write_bytes(Path(__file__).read_bytes())
        assay.rng=np.random.default_rng(np.random.SeedSequence([config['seed'],
            ['probes','temporal','controls','boundaries','siblings'].index(stage)]))
        getattr(assay,stage)()
        print(f'COMPLETE {stage} {time.monotonic()-started:.1f}s',flush=True)


if __name__=='__main__':
    main()
