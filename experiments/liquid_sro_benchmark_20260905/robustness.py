"""Physical perturbation controls, including the hard nearest-neighbor TDA window."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.analysis.liquid_structure import persistence_image
from src.models.encoders.smooth_density import SmoothDensity
from experiments.liquid_sro_benchmark_20260905.features import soap_chunk
from experiments.liquid_sro_benchmark_20260905.train import construct
from experiments.liquid_sro_benchmark_20260905.evaluate import FIXED,LEARNED,save_csv
from experiments.smooth_temporal_encoder_20260905.evaluate import load_geoframe
from experiments.smooth_temporal_encoder_20260905.prepare import write_json


def sort_clouds(x):
    order=np.argsort(np.square(x).sum(-1),axis=1)
    return np.take_along_axis(x,order[:,:,None],axis=1)


def descriptors(x):
    # The index-selected point set changes with perturbation; measure it explicitly.
    tda=np.stack([persistence_image(c[:65]) for c in sort_clouds(x)])
    return soap_chunk(x),tda


def prepare(cfg,out,directory):
    meta=dict(np.load(out/'metadata.npz'));rng=np.random.default_rng(20260909)
    indices=[]
    for material in range(3):
        eligible=np.flatnonzero((meta['split']=='test')&(meta['material']==material))
        indices.extend(rng.choice(eligible,128 if material==0 else 64,replace=False))
    indices=np.array(indices);x=np.load(out/'clouds.npy')[indices]
    rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
    noise=rng.normal(size=x.shape).astype(np.float32);noise[:,0]=0
    cases={'original':x,'rotation':(x@rotation).astype(np.float32),
        'noise_0.02A':x+np.clip(noise*.02,-.06,.06),
        'noise_0.0001A':x+np.clip(noise*.0001,-.0003,.0003)}
    np.savez(directory/'inputs.npz',indices=indices,material=meta['material'][indices],**cases)
    tasks=[value[i:i+64] for value in cases.values() for i in range(0,len(x),64)]
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn')) as pool:
        results=list(pool.map(descriptors,tasks))
    for case_index,name in enumerate(cases):
        part=results[case_index*4:(case_index+1)*4]
        np.save(directory/f'{name}_soap.npy',np.concatenate([v[0] for v in part]))
        np.save(directory/f'{name}_tda.npy',np.concatenate([v[1] for v in part]))
    boundary={}
    initial=np.load(directory/'original_tda.npy')
    for name in ('noise_0.02A','noise_0.0001A'):
        y=cases[name];order=np.argsort(np.square(y).sum(-1),axis=1)
        churn=np.array([len(set(ids[:65])-set(range(65))) for ids in order])
        with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn')) as pool:
            fixed=np.stack(list(pool.map(persistence_image,[c[:65] for c in y])))
        reranked=np.load(directory/f'{name}_tda.npy')
        boundary[name]=dict(fraction_windows_changed=float((churn>0).mean()),
            mean_changed_atoms=float(churn.mean()),
            fixed_window_mean_squared_change=float(np.square(fixed-initial).sum(1).mean()),
            reranked_window_mean_squared_change=float(np.square(reranked-initial).sum(1).mean()))
        np.save(directory/f'{name}_tda_fixed_window.npy',fixed)
    write_json(directory/'tda_boundary.json',boundary)
    write_json(directory/'prepare_status.json',dict(state='complete',count=len(indices),noise='independent Cartesian Gaussian displacements per neighbor; center fixed; same noise direction for refinement'))


def edges(x):
    mask=(torch.cdist(x,x)<4.25)&~torch.eye(x.shape[1],device=x.device,dtype=torch.bool)[None]
    values=[m.nonzero() for m in mask]
    counts=torch.tensor([len(v) for v in values],device=x.device)
    padded=torch.zeros(len(x),int(counts.max()),2,device=x.device,dtype=torch.long)
    for i,value in enumerate(values):padded[i,:len(value)]=value
    return padded,counts


def pca(values,out,name,dimensions=128):
    saved=torch.load(out/f'{name}.pca.pt',weights_only=True)
    mean,std,basis=(saved[k].numpy() for k in ('mean','std','vectors'))
    return ((values-mean)/std)@basis[:,:dimensions]


@torch.no_grad()
def evaluate(cfg,out,directory):
    assert json.loads((out/'training_status.json').read_text())['state']=='complete'
    data=dict(np.load(directory/'inputs.npz'));material=torch.tensor(data['material'],device='cuda')
    meta=dict(np.load(out/'metadata.npz'));names=list(FIXED)+[f'{m}_seed{s}' for m in LEARNED for s in cfg['training']['seeds']]
    outputs={name:{} for name in names}
    for case in ('original','rotation','noise_0.02A','noise_0.0001A'):
        outputs['SOAP'][case]=pca(np.load(directory/f'{case}_soap.npy'),out,'SOAP')
        outputs['TDA'][case]=pca(np.load(directory/f'{case}_tda.npy'),out,'TDA')
        outputs['TDA_16'][case]=outputs['TDA'][case][:,:16]
    density=SmoothDensity().cuda()
    for case in ('original','rotation','noise_0.02A','noise_0.0001A'):
        x=torch.tensor(data[case],device='cuda')
        powers=density.power(density(x[:,1:]/8.)).cpu().numpy()
        outputs['DensityPCA'][case]=pca(powers,out,'DensityPCA')
    pcfg=json.loads((ROOT/cfg['pilot']/'config.json').read_text());gf=load_geoframe(pcfg)
    manifest=json.loads(Path(pcfg['views_manifest']).read_text())
    scale=torch.tensor([next(s['radius'] for s in manifest['sources'] if s['material']==m) for m in ('Al','Mg','Ta')],device='cuda')
    for case in ('original','rotation','noise_0.02A','noise_0.0001A'):
        x=torch.tensor(sort_clouds(data[case])[:,:80],device='cuda')
        outputs['GeoFrame_pretrained'][case]=gf.encoder.forward_features(x/scale[material,None,None]).cpu().numpy()
    del gf,density
    for model_name in LEARNED:
        for seed in cfg['training']['seeds']:
            name=f'{model_name}_seed{seed}'
            model=construct(model_name,out).eval().requires_grad_(False)
            saved=torch.load(out/'models'/name/'selected.pt',map_location='cuda',weights_only=False)
            model.load_state_dict(saved['state_dict'])
            for case in ('original','rotation','noise_0.02A','noise_0.0001A'):
                x=torch.tensor(data[case],device='cuda');edge,count=edges(x)
                outputs[name][case]=model(x.clone(),material,edge,count).cpu().numpy()
            del model,saved;torch.cuda.empty_cache()
    rows=[]
    for name,cases in outputs.items():
        training=np.load(out/'embeddings'/f'{name}.npy')
        for material_id,label in enumerate(('Al','Mg','Ta')):
            scale=np.sqrt(training[(meta['split']=='train')&(meta['material']==material_id)].var(0).sum())
            keep=data['material']==material_id
            for case in ('rotation','noise_0.02A','noise_0.0001A'):
                norm=np.linalg.norm(cases[case][keep]-cases['original'][keep],axis=1)/scale
                rows.append(dict(model=name,material=label,case=case,n=int(keep.sum()),
                    median_relative_change=float(np.median(norm)),p95_relative_change=float(np.quantile(norm,.95))))
        np.savez(directory/f'{name}_outputs.npz',**cases)
    save_csv(directory/'perturbation_results.csv',rows)
    write_json(directory/'status.json',dict(state='complete',normalization='L2 change / sqrt(sum training within-material feature variances); report with fidelity to avoid rewarding collapse'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('prepare','evaluate'),required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());out=ROOT/cfg['output']
    directory=out/'robustness';directory.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    if args.stage=='prepare':prepare(cfg,out,directory)
    else:evaluate(cfg,out,directory)


if __name__=='__main__':main()
