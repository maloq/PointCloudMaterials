"""PTM-oriented ideal lattice overlays for the existing saved local examples."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from src.analysis.representative_structures import _build_ovito_data_collection
from src.data.fixed_cohort.protocol import sha, write_json
from .viewer_payload import read_asset, write_asset


def grid(kind, distance, rotation, radius, hcp_branch=1):
    n=int(np.ceil(radius/distance))+2
    cells=np.array(list(product(range(-n,n+1),repeat=3)),float)
    if kind==1:
        basis=np.array([[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]])
        points=(cells[:,None,:]+basis).reshape(-1,3)*distance*np.sqrt(2)
    elif kind==3:
        points=(cells[:,None,:]+np.array([[0,0,0],[.5,.5,.5]])).reshape(-1,3)*distance*2/np.sqrt(3)
    elif kind==2:
        unit=np.array([[1,0,0],[.5,np.sqrt(3)/2,0],[0,0,np.sqrt(8/3)]])
        points=(cells@unit)[:,None,:]+np.array([[0,0,0],[0,-hcp_branch*np.sqrt(3)/3,np.sqrt(2/3)]])
        points=points.reshape(-1,3)*distance
    else:raise ValueError(f'Unsupported PTM crystal: {kind}')
    points=points[np.linalg.norm(points,axis=1)<=radius+1e-8]
    return rotation.apply(points)


def fit_batch(patches):
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    xyz=np.asarray(patches,float);r=float(np.linalg.norm(xyz,axis=2).max());pitch=4*r+4
    side=int(np.ceil(len(xyz)**(1/3)))+1
    shifts=np.array(list(product(range(side),repeat=3)))[:len(xyz)]*pitch
    data=_build_ovito_data_collection((xyz+shifts[:,None,:]).reshape(-1,3))
    data.apply(PolyhedralTemplateMatchingModifier(rmsd_cutoff=0,output_rmsd=True,output_orientation=True,output_interatomic_distance=True))
    centers=np.arange(len(xyz))*80
    types=np.asarray(data.particles['Structure Type'])[centers]
    rmsd=np.asarray(data.particles['RMSD'])[centers]
    orientations=np.asarray(data.particles['Orientation'])[centers]
    distances=np.asarray(data.particles['Interatomic Distance'])[centers]
    results=[]
    for points,kind,error,quaternion,distance in zip(xyz,types,rmsd,orientations,distances):
        if kind==0:
            results.append(dict(candidate='None',accepted=False,rmsd=None,grid=[],edges=[],matched=[],shell=[]));continue
        if not np.isfinite(error) or distance<=0:raise ValueError('Invalid central PTM fit')
        candidates=[grid(int(kind),float(distance),Rotation.from_quat(quaternion),float(np.linalg.norm(points,axis=1).max()+distance),branch)
                    for branch in ([1,-1] if kind==2 else [1])]
        # PTM's hexagonal orientation has a 60-degree basis ambiguity. Resolve
        # the two AB offsets against the fitted first shell, keeping a perfect lattice.
        shell_points=points[np.argsort(np.linalg.norm(points,axis=1))[:13]]
        ideal=min(candidates,key=lambda lattice:np.sum(cKDTree(lattice).query(shell_points)[0]**2))
        origin=int(np.argmin(np.linalg.norm(ideal,axis=1)));others=np.delete(np.arange(len(ideal)),origin)
        costs=np.sum((points[1:,None,:]-ideal[None,others,:])**2,axis=2)
        row,col=linear_sum_assignment(costs)
        if not np.array_equal(row,np.arange(79)):raise ValueError('Incomplete one-to-one lattice correspondence')
        matched=np.r_[origin,others[col]]
        residual=np.linalg.norm(points-ideal[matched],axis=1)
        edges=sorted(cKDTree(ideal).query_pairs(float(distance)*1.05))
        shell=np.argsort(np.linalg.norm(points,axis=1))[:15 if kind==3 else 13]
        results.append(dict(candidate={1:'FCC',2:'HCP',3:'BCC'}[int(kind)],accepted=bool(error<=.1),rmsd=float(error),
            distance=float(distance),grid=np.round(ideal,6).tolist(),edges=edges,matched=matched.tolist(),shell=shell.tolist(),
            residual_A=np.round(residual,6).tolist(),extended_rms_A=float(np.sqrt(np.mean(residual[1:]**2)))))
    return results


def export_snapshot(publication,key):
    dest=Path(publication);manifest=json.loads((dest/'sample-data/manifest.json').read_text())[key]
    assets={};patches={}
    for kind,spaces in manifest.items():
        for identity,entry in spaces.items():
            path=dest/'sample-data'/Path(entry['asset']).name
            data=read_asset(path)
            assets[(kind,identity)]=(entry,data,path);patches.update(data['patches'])
    rows=sorted(patches,key=int);fits={}
    for start in range(0,len(rows),128):
        selected=rows[start:start+128]
        for row,fit in zip(selected,fit_batch([patches[r]['xyz'] for r in selected])):fits[row]=fit
    folder=dest/'lattice-data';folder.mkdir(exist_ok=True);result={kind:{} for kind in manifest};receipts={}
    for (kind,identity),(entry,data,path) in assets.items():
        asset_key=entry['key'];out=folder/(asset_key+'.js')
        write_asset(out,asset_key,{row:fits[row] for row in data['patches']},'LATTICE_SAMPLES')
        result[kind][identity]=dict(key=asset_key,asset='../lattice-data/'+out.name)
        receipts[out.name]=dict(input_payload_sha256=sha(path.with_suffix('.json')),asset_sha256=sha(out))
    write_json(folder/f'manifest-{key}.json',{key:result})
    write_json(dest/f'technical/rendering/lattice-{key}.json',dict(assets=receipts,examples=len(fits),implementation_sha256=sha(__file__),
        fit='central-atom PTM FCC/HCP/BCC; rigid orientation and isotropic spacing; accepted at RMSD <= 0.1',
        grid='ideal periodic lattice extended across sample; no affine deformation; one-to-one 80-atom assignment with center fixed',
        limitation='A poor best candidate is not a crystal identification; outer-grid mismatch is distinct from local PTM RMSD'))
    print(f'Lattice {key}: {len(fits)} examples',flush=True)
    return {key:result}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--publication',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    dest=Path(a.publication);keys=list(json.loads((dest/'sample-data/manifest.json').read_text()))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(export_snapshot,str(dest),key) for key in keys];results=[f.result() for f in futures]
    write_json(dest/'lattice-data/manifest.json',{k:v for r in results for k,v in r.items()})

if __name__=='__main__':main()
