"""Cache buffered graphs and label-free descriptor baselines for the benchmark."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import json
import multiprocessing as mp
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from numpy.lib.format import open_memmap
import torch
from src.models.encoders.smooth_density import SmoothDensity
from experiments.smooth_temporal_encoder_20260905.evaluate import load_geoframe
from experiments.smooth_temporal_encoder_20260905.prepare import write_json


def soap_chunk(clouds):
    from ase import Atoms
    from dscribe.descriptors import SOAP
    descriptor=SOAP(species=[13],periodic=False,r_cut=6.5,n_max=8,l_max=6,sigma=.3,sparse=False)
    return np.stack([descriptor.create(Atoms(numbers=np.full(len(x),13),positions=x),centers=[0])[0] for x in clouds]).astype(np.float32)


@torch.no_grad()
def fit_pca(values,train):
    x=torch.tensor(values,device='cuda',dtype=torch.float64)
    mean=x[train].mean(0);std=x[train].std(0)
    std=std.clamp_min(std.median()*.01)
    z=(x-mean)/std
    eig,vec=torch.linalg.eigh(z[train].T@z[train]/(int(train.sum())-1))
    v=vec[:,-128:].flip(1)
    return (z@v).float().cpu().numpy(),dict(mean=mean.cpu(),std=std.cpu(),vectors=v.cpu(),eigenvalues=eig[-128:].flip(0).cpu())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text());out=ROOT/cfg['output']
    assert json.loads((out/'prepare_status.json').read_text())['state']=='complete'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    clouds=np.load(out/'clouds.npy');meta=dict(np.load(out/'metadata.npz'))
    train=torch.tensor(meta['split']=='train',device='cuda')
    (out/'embeddings').mkdir(exist_ok=False)
    # Diagonal exclusion is by node identity, never by cdist > 0 (roundoff).
    edge_chunks=[];counts=[]
    for start in range(0,len(clouds),128):
        x=torch.tensor(clouds[start:start+128],device='cuda')
        mask=(torch.cdist(x,x)<cfg['graph_cutoff_A']+cfg['graph_skin_A'])&~torch.eye(193,device='cuda',dtype=torch.bool)[None]
        for row in mask:
            edges=row.nonzero().to(torch.uint8).cpu().numpy()
            edge_chunks.append(edges);counts.append(len(edges))
    counts=np.array(counts,dtype=np.int32)
    cap=int(counts.max())
    edges=open_memmap(out/'edges.npy',mode='w+',dtype=np.uint8,shape=(len(clouds),cap,2))
    for i,value in enumerate(edge_chunks):edges[i,:len(value)]=value
    edges.flush();np.save(out/'edge_counts.npy',counts)
    del edge_chunks,edges
    print('Graph cache',len(clouds),'max edges',cap,flush=True)
    write_json(out/'graph_protocol.json',dict(cutoff_A=4.,skin_A=cfg['graph_skin_A'],max_edges=cap,
        max_jitter_pair_displacement_A=float(2*np.sqrt(3)*cfg['training']['jitter_clip_A']),
        central_readout='Both reference MACE layers; two-hop support completely inside atom buffer',
        cutoff_application='Reference MACE apply_cutoff=False: apply envelope after radial MLP; buffered edges outside 4 A have exactly zero messages'))
    density=SmoothDensity().cuda();powers=[]
    with torch.no_grad():
        for start in range(0,len(clouds),512):
            x=torch.tensor(clouds[start:start+512,1:]/8.,device='cuda')
            powers.append(density.power(density(x)).cpu().numpy())
    powers=np.concatenate(powers);np.save(out/'density_power.npy',powers)
    t=torch.tensor(powers[meta['split']=='train'],device='cuda')
    std=t.std(0);torch.save(dict(mean=t.mean(0).cpu(),std=std.clamp_min(std.median()*.01).cpu()),out/'density_scaling.pt')
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn')) as pool:
        soap=np.concatenate(list(pool.map(soap_chunk,[clouds[i:i+128] for i in range(0,len(clouds),128)])))
    np.save(out/'soap_raw.npy',soap)
    for name,values in [('SOAP',soap),('TDA',np.load(out/'tda.npy')),('DensityPCA',powers)]:
        z,scaling=fit_pca(values,train);np.save(out/'embeddings'/f'{name}.npy',z);torch.save(scaling,out/f'{name}.pca.pt')
        if name=='TDA':np.save(out/'embeddings/TDA_16.npy',z[:,:16])
        print('Baseline',name,z.shape,flush=True)
    pcfg=json.loads((ROOT/cfg['pilot']/'config.json').read_text());gf=load_geoframe(pcfg)
    source_manifest=json.loads(Path(pcfg['views_manifest']).read_text())
    scales=torch.tensor([next(s['radius'] for s in source_manifest['sources'] if s['material']==m) for m in ('Al','Mg','Ta')],device='cuda')
    outputs=[]
    with torch.no_grad():
        for start in range(0,len(clouds),512):
            x=torch.tensor(clouds[start:start+512,:80],device='cuda')
            material=torch.tensor(meta['material'][start:start+512],device='cuda')
            outputs.append(gf.encoder.forward_features(x/scales[material,None,None]).cpu().numpy())
    np.save(out/'embeddings/GeoFrame_pretrained.npy',np.concatenate(outputs))
    write_json(out/'config.json',cfg)
    write_json(out/'features_status.json',dict(state='complete',finished_at=datetime.now(timezone.utc).isoformat(),
        soap='DScribe nmax=8, lmax=6, sigma=.3 A, rcut=6.5 A, one shared geometry species; Gaussian padding lies inside complete input support',
        normalization='PCA and feature standardization fitted on training inputs only'))


if __name__=='__main__':
    main()
