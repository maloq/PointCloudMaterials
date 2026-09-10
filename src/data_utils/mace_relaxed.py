"""Matched hot/relaxed 80-atom views from converged full periodic cells."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
from src.analysis.liquid_structure import persistence_image
from src.data_utils.conversion.relaxation import read_relaxed
from src.data_utils.temporal_lammps_binary import TemporalLAMMPSBinaryTrajectory
from src.data_utils.temporal_campaign import write_json
from src.simulation.relaxation import relax_frame,sha256


def paired_clouds(hot,relaxed,lengths,centers):
    """Choose membership once in the hot frame and keep exactly those identities."""
    hot=np.mod(hot,lengths);relaxed=np.mod(relaxed,lengths)
    tree=cKDTree(hot,boxsize=lengths)
    neighbors=tree.query(hot[centers],k=80,workers=1)[1]
    views=[];errors=[]
    for positions in (hot,relaxed):
        offsets=positions[neighbors]-positions[centers,None]
        offsets-=lengths*np.round(offsets/lengths)
        stored=offsets.astype(np.float16)
        errors.append(float(np.max(np.abs(stored.astype(float)-offsets))))
        views.append(stored)
    np.testing.assert_array_equal(neighbors[:,0],centers)
    return views[0],views[1],neighbors,errors


def prepare_shard(record,cfg,pool):
    root=Path(cfg['cache']);directory=root/record['name'];directory.mkdir(parents=True,exist_ok=True)
    if (directory/'manifest.json').exists():
        saved=json.loads((directory/'manifest.json').read_text())
        for name,digest in saved['checksums'].items():
            if sha256(directory/name)!=digest:raise ValueError(f'Changed paired cache: {directory/name}')
        return saved
    source=Path(record['directory']);all_frames=np.load(source/'frames.npy')
    selected=np.flatnonzero(np.isin(all_frames[:,0],record['anchors'][:cfg['relaxation']['anchor_frames_per_source']]))
    ids=np.load(source/'ids.npy')[selected];frames=all_frames[selected]
    conditions=np.load(source/'condition.npy')[selected]
    for name,array in (('ids',ids),('frames',frames),('condition',conditions)):
        np.save(directory/(name+'.npy'),array)
    count=len(ids);mode='r+' if (directory/'clouds.npy').exists() else 'w+'
    clouds=np.lib.format.open_memmap(directory/'clouds.npy',mode=mode,dtype=np.float16,shape=(count,8,80,3))
    targets=np.lib.format.open_memmap(directory/'tda.npy',mode=mode,dtype=np.float32,shape=(count,8,144))
    identities=np.lib.format.open_memmap(directory/'neighbor_ids.npy',mode=mode,dtype=np.int32,shape=(count,4,80))
    trajectory=TemporalLAMMPSBinaryTrajectory.load(record['path'])
    settings=cfg['relaxation'];material=('Al','Mg','Ta')[record['material']]
    physics={**settings['controls'],**settings['materials'][material]}
    frame_reports=[];started=time.monotonic()
    for frame in np.unique(frames):
        frame=int(frame);work=Path(settings['output'])/record['name']/str(frame)
        completion=work/'paired_complete.json'
        if completion.exists():
            frame_reports.append(json.loads(completion.read_text()));continue
        write_json(root/'status.json',dict(state='relaxing_full_cell',shard=record['name'],frame=frame))
        if not (work/'metadata.json').exists():relax_frame(trajectory,frame,work,physics)
        relaxed,metadata=read_relaxed(work)
        low=trajectory.box_low[frame].astype(float);lengths=trajectory.box_high[frame].astype(float)-low
        hot=np.mod(trajectory.positions[frame].astype(float)-low,lengths)
        row,view=np.nonzero(frames==frame);centers,inverse=np.unique(ids[row,view],return_inverse=True)
        a,b,neighbors,errors=paired_clouds(hot,relaxed-low,lengths,centers)
        tda=np.stack(list(pool.map(persistence_image,(x.astype(np.float32) for x in b),chunksize=32)))
        clouds[row,view]=a[inverse];clouds[row,view+4]=b[inverse]
        targets[row,view]=tda[inverse];targets[row,view+4]=tda[inverse]
        identities[row,view]=trajectory.atom_ids[neighbors[inverse]]
        clouds.flush();targets.flush();identities.flush()
        subprocess.run([sys.executable,'scripts/convert_trajectory.py','relaxation',str(work),'--delete-source'],check=True)
        report=dict(frame=frame,patches=len(centers),fmax_eV_per_A=metadata['fmax_eV_per_A'],
                    relaxation_seconds=metadata['seconds'],local_offset_quantization_max_A=errors,
                    mean_hot_relaxed_patch_displacement_A=float(np.linalg.norm(a.astype(float)-b.astype(float),axis=-1).mean()))
        write_json(completion,report);frame_reports.append(report)
        print('PAIRED_FRAME',record['name'],json.dumps(report),flush=True)
    files=('clouds.npy','tda.npy','neighbor_ids.npy','ids.npy','frames.npy','condition.npy')
    saved=dict(record,directory=str(directory),anchors_count=count,anchors=np.unique(frames[:,0]).tolist(),
        source_directory=str(source),tda_points=80,training_views=6,stored_views=8,frame_reports=frame_reports,
        source_manifest_sha256=sha256(source/'manifest.json'),checksums={name:sha256(directory/name) for name in files},
        seconds=time.monotonic()-started,
        view_order=['hot_anchor','hot_spatial','hot_temporal','hot_future','relaxed_anchor','relaxed_spatial','relaxed_temporal','relaxed_future'],
        target='Both states predict TDA of the relaxed version of their identical 80 atom identities; computed from centered float16 offsets.')
    write_json(directory/'manifest.json',saved)
    return saved


def prepare(cfg):
    root=Path(cfg['cache']);root.mkdir(parents=True,exist_ok=True)
    records=json.loads(Path(cfg['source_manifest']).read_text())['shards']
    records=[r for r in records if r['name'] in cfg['relaxation']['shards']]
    if len(records)!=len(cfg['relaxation']['shards']):raise ValueError('Configured relaxed source shards are missing')
    try:
        with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
            result=[prepare_shard(r,cfg,pool) for r in records]
        write_json(root/'manifest.json',dict(shards=result,config=cfg,tda_points=80,protocol='thermal80'))
        write_json(root/'status.json',dict(state='complete',shards=len(result),anchors=sum(r['anchors_count'] for r in result)))
    except BaseException:
        import traceback
        write_json(root/'status.json',dict(state='failed',traceback=traceback.format_exc()));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args()
    prepare(json.loads(Path(args.config).read_text()))


if __name__=='__main__':main()
