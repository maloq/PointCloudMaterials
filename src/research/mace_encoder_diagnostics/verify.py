"""Check label producers and replay stored forecast inputs through the exact encoder."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.analysis.liquid_structure import persistence_image
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.registry import sha256, write_json
from src.project_runtime.paths import load_json, resolve_path
from src.research.smooth_temporal_encoder.prepare import ptm_labels
from .extract import Assay, frame_clouds, structural


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();cfg=load_json(args.config);torch.set_num_threads(4)
    root=Path(cfg['output']);tech=root/'technical'
    p=np.load(tech/'probes.npz');sample=np.linspace(0,len(p['z'])-1,24,dtype=int)
    recomputed=np.stack([persistence_image(x) for x in p['clouds'][sample]])
    np.testing.assert_array_equal(recomputed,p['hot'][sample])
    t=np.load(tech/'temporal.npz');x=t['clouds'][:,8]
    obs,labels=structural(x,cfg['ptm_rmsd_cutoff'])
    expected=ptm_labels(x[:,1:]/10.,cfg['ptm_rmsd_cutoff'])
    np.testing.assert_array_equal(labels,expected)
    np.testing.assert_array_equal(labels,t['labels'][:,8])
    cache=Path(cfg['forecast_cache']);manifest=json.loads((cache/'manifest.json').read_text())
    if manifest['protocol']['checkpoint_sha256']!=sha256(Path(cfg['checkpoint'])):
        raise ValueError('Forecast cache was encoded with a different checkpoint')
    selection=json.loads(resolve_path(manifest['protocol']['config']['sources_config']).read_text())['sources']
    chosen=[]
    for temp in (400.,450.,510.):
        chosen.append(next(s for s in manifest['shards'] if s['split']=='test' and s['temperature_K']==temp))
    assay=Assay(cfg);replay=[]
    for shard in chosen:
        directory=cache/shard['directory'];source=selection[shard['source_index']]
        if source['name']!=shard['name']:raise ValueError('Forecast source identity mismatch')
        for name in ('atom_ids.npy','frames.npy','timesteps.npy','embeddings.npy'):
            if sha256(directory/name)!=shard['checksums'][name]:raise ValueError(f'Changed cache shard {directory/name}')
        tr=ShootingBinaryTrajectory.load(source['path']);atom_ids=np.load(directory/'atom_ids.npy')[:8]
        centers=np.searchsorted(tr.atom_ids,atom_ids);np.testing.assert_array_equal(tr.atom_ids[centers],atom_ids)
        frames=np.load(directory/'frames.npy');columns=np.array([40,240,640])
        np.testing.assert_array_equal(tr.timesteps[frames[columns]],np.load(directory/'timesteps.npy')[columns])
        x=np.concatenate([frame_clouds(tr,int(frames[c]),centers)[0][:,:80] for c in columns])
        z=assay.encode(x)
        stored=np.load(directory/'embeddings.npy',mmap_mode='r')[:8,columns].transpose(1,0,2).reshape(-1,256).astype(np.float32)
        replay.append(dict(source=shard['name'],n=len(z),raw_mse=float(np.mean((z-stored)**2)),
            max_absolute_error=float(np.max(np.abs(z-stored))),
            fp16_bin_match_fraction=float(np.mean(z.astype(np.float16).astype(np.float32)==stored)),
            expected_fp16_mse=float(np.mean((z-z.astype(np.float16).astype(np.float32))**2))))
        np.savez_compressed(tech/f'replay-{shard["source_index"]}.npz',z=z,stored=stored,atom_ids=atom_ids,frames=frames[columns])
    campaign=Path(cfg['sibling_campaign']);spec=json.loads((campaign/'manifest.json').read_text())
    momenta={};velocity_checks=[];initial_box=None
    for branch in spec['branches']:
        trajectory=ShootingBinaryTrajectory.load(campaign/branch['branch_dir']/'trajectory_binary_float32')
        group=branch['momentum_index'];v=trajectory.velocities[0]
        box=np.stack([trajectory.box_low[0],trajectory.box_high[0]])
        if initial_box is None:initial_box=box
        np.testing.assert_array_equal(box,initial_box)
        np.testing.assert_allclose(np.diff(trajectory.timesteps)*spec['protocol']['timestep_fs']/1000,
            spec['protocol']['sample_interval_ps'],rtol=0,atol=1e-12)
        if group in momenta:
            np.testing.assert_array_equal(v,momenta[group])
            velocity_checks.append(group)
        else:momenta[group]=v.copy()
    groups=sorted(momenta)
    for i,a in enumerate(groups):
        for b in groups[i+1:]:
            if np.array_equal(momenta[a],momenta[b]):raise ValueError('Distinct momentum groups have identical velocities')
    result=dict(hot_label_recomputed_bitwise=24,ptm_label_matches=len(t['source']),
        sibling_identical_momentum_pairs=velocity_checks,sibling_distinct_momentum_groups=len(groups),
        ptm_missing_margin_count=int(np.isnan(obs[:,0]).sum()),cache_replay=replay,
        forecast_manifest_sha256=sha256(cache/'manifest.json'),implementation_sha256=sha256(Path(__file__)),
        extraction_artifacts={name:sha256(tech/f'{name}.npz') for name in
            ('probes','temporal','controls','boundaries','siblings','global_precision')})
    write_json(tech/'verification.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
