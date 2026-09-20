"""Trace native graph/query identities to existing raw trajectories and parent targets."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import source_arrays,chart,offsets,file_hash,save_json
from src.data.structural_pretraining.support import REFERENCE_RADIUS,OUTER_RADIUS


def audit(config):
    cache=resolve_path(config['cache'])
    manifest=json.loads((cache/'manifest.json').read_text())
    parent=Path(manifest['parent'])
    plan=json.loads((parent/'plan.json').read_text())
    tasks={t['shard']['task']['id']:t for t in plan['tasks']}
    target_parent=resolve_path(plan['config']['parent_release'])
    checked=[]
    for record in manifest['shards']:
        task=tasks[record['id']]
        source=task['source']
        raw=source_arrays(source)
        frame=record['frame']
        if raw['box_low'].shape[1:]!=(3,) or raw['box_high'].shape[1:]!=(3,):
            raise ValueError('Only the registered orthorhombic box producer is supported')
        folder=parent/'shards'/record['id']
        a={p.stem:np.load(p,mmap_mode='r') for p in folder.glob('*.npy')}
        target_folder=target_parent/'shards'/record['id']
        parent_views=np.load(target_folder/'views.npy',mmap_mode='r')[task['rows']]
        parent_ids=np.load(target_folder/'center_ids.npy',mmap_mode='r')
        np.testing.assert_array_equal(parent_ids[parent_views[:,2]],a['query_atom_ids'][:,0])
        for name in ('physical','tda'):
            original=np.load(target_folder/f'{name}.npy',mmap_mode='r')
            np.testing.assert_array_equal(original[parent_views[:,[2,3]]],a[name])
        lookup={int(atom):i for i,atom in enumerate(raw['atom_ids'])}
        chosen=np.array([lookup[int(atom)] for atom in a['query_atom_ids'][0]])
        factor=REFERENCE_RADIUS/record['scale']
        for ti,f in enumerate((frame-1,frame,frame+1)):
            points,tree,box=chart(raw,f,False)
            if ti==1:
                np.testing.assert_allclose(offsets(points,chosen[0],chosen,box)*factor,a['query_positions'][0],rtol=0,atol=1e-6)
            for j,center in enumerate(chosen):
                ids=np.array(sorted(tree.query_ball_point(points[center],OUTER_RADIUS/factor)))
                ids=np.r_[center,ids[ids!=center]]
                local=offsets(points,center,ids,box)*factor
                local=local[np.linalg.norm(local,axis=-1)<OUTER_RADIUS]
                view=a['views'][0,ti,j]
                lo,hi=a['offsets'][view:view+2]
                np.testing.assert_allclose(local,a['positions'][lo:hi],rtol=0,atol=1e-6)
                if j==0 and len(local)<80: raise ValueError('TDA target support is outside observation')
        # The stored float16 ULP is an upper coordinate quantization resolution,
        # not a reconstruction of unknown full-precision MD coordinates.
        example=np.array(raw['positions'][frame])
        spacing=np.spacing(np.abs(example)).astype(np.float64)
        checked.append(dict(shard=record['id'],source=record['source'],frame=frame,
            atom_ids=a['query_atom_ids'][0].tolist(),dtype=str(example.dtype),
            maximum_coordinate_ulp_A=float(spacing.max()),audited_views=21,
            source_manifest_sha256=source['manifest_sha256']))
    return dict(state='passed',cache_identity=manifest['identity'],shards=len(checked),
        views=sum(r['audited_views'] for r in checked),evidence=checked,
        periodic_contract='registered wrapped orthorhombic coordinates; exact minimum-image reconstruction',
        precision='Sub-ULP perturbations cannot be interpreted as resolved MD motion',
        tda='nearest80 inclusion verified; a separate continuity assay remains deferred')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    save_json(args.output,audit(json.loads(Path(args.config).read_text())))
