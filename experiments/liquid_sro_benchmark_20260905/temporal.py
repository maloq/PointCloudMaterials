"""Real 0.3 ps trajectory continuity controls on six held-out float32 Al sources."""
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
from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.models.encoders.smooth_density import SmoothDensity
from experiments.liquid_sro_benchmark_20260905.prepare import frame_geometry
from experiments.liquid_sro_benchmark_20260905.robustness import descriptors,edges,pca,sort_clouds
from experiments.liquid_sro_benchmark_20260905.train import construct
from experiments.liquid_sro_benchmark_20260905.evaluate import FIXED,LEARNED,save_csv
from experiments.smooth_temporal_encoder_20260905.evaluate import load_geoframe
from experiments.smooth_temporal_encoder_20260905.prepare import write_json


def parent_sequence(item):
    parent,out=item
    manifest=sorted(parent['input_manifests'])[0]
    trajectory=ShootingBinaryTrajectory.load(Path(manifest).parent)
    ids=np.load(out/'parents'/f"parent_{parent['parent_index']:03d}.npz")['rows'][:32]
    box=trajectory.box_high[0].astype(np.float64)-trajectory.box_low[0]
    sequence=[frame_geometry(trajectory.positions[f],box,ids,3.7)[0] for f in range(8)]
    return np.stack(sequence,axis=1),ids


def prepare(cfg,out,directory):
    parents=json.loads((out/'parent_manifest.json').read_text())
    sources=json.loads((out/'data_summary.json').read_text())['shooting_source_splits']['test']
    chosen=[min((p for p in parents if p['source_index']==s),key=lambda p:p['parent_index']) for s in sources]
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
        results=list(pool.map(parent_sequence,[(p,out) for p in chosen]))
    x=np.concatenate([r[0] for r in results]);flat=x.reshape(-1,193,3)
    np.savez(directory/'inputs.npz',clouds=x,center_ids=np.concatenate([r[1] for r in results]),source=np.repeat(sources,32),times_ps=np.arange(8)*.3)
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn')) as pool:
        results=list(pool.map(descriptors,[flat[i:i+64] for i in range(0,len(flat),64)]))
    np.save(directory/'soap.npy',np.concatenate([v[0] for v in results]))
    np.save(directory/'tda.npy',np.concatenate([v[1] for v in results]))
    write_json(directory/'prepare_status.json',dict(state='complete',source_count=len(sources),centers=len(x),frames=8,dt_ps=.3,position_dtype='float32'))


@torch.no_grad()
def evaluate(cfg,out,directory):
    assert json.loads((out/'training_status.json').read_text())['state']=='complete'
    data=dict(np.load(directory/'inputs.npz'));x=data['clouds'].reshape(-1,193,3)
    meta=dict(np.load(out/'metadata.npz'))
    outputs={'SOAP':pca(np.load(directory/'soap.npy'),out,'SOAP'),
             'TDA':pca(np.load(directory/'tda.npy'),out,'TDA')}
    outputs['TDA_16']=outputs['TDA'][:,:16]
    density=SmoothDensity().cuda();powers=[]
    for start in range(0,len(x),256):
        c=torch.tensor(x[start:start+256,1:]/8.,device='cuda')
        powers.append(density.power(density(c)).cpu().numpy())
    outputs['DensityPCA']=pca(np.concatenate(powers),out,'DensityPCA');del density
    pcfg=json.loads((ROOT/cfg['pilot']/'config.json').read_text());gf=load_geoframe(pcfg)
    manifest=json.loads(Path(pcfg['views_manifest']).read_text());scale=next(s['radius'] for s in manifest['sources'] if s['material']=='Al')
    zz=[]
    for start in range(0,len(x),256):
        c=torch.tensor(sort_clouds(x[start:start+256])[:,:80]/scale,device='cuda')
        zz.append(gf.encoder.forward_features(c).cpu().numpy())
    outputs['GeoFrame_pretrained']=np.concatenate(zz);del gf
    for model_name in LEARNED:
        for seed in cfg['training']['seeds']:
            name=f'{model_name}_seed{seed}';model=construct(model_name,out).eval().requires_grad_(False)
            model.load_state_dict(torch.load(out/'models'/name/'selected.pt',map_location='cuda',weights_only=False)['state_dict'])
            zz=[]
            for start in range(0,len(x),256):
                c=torch.tensor(x[start:start+256],device='cuda');edge,count=edges(c)
                zz.append(model(c.clone(),torch.zeros(len(c),device='cuda',dtype=torch.long),edge,count).cpu().numpy())
            outputs[name]=np.concatenate(zz);del model;torch.cuda.empty_cache()
    records=[]
    for name,z in outputs.items():
        train=np.load(out/'embeddings'/f'{name}.npy')[(meta['split']=='train')&(meta['material']==0)]
        scale=np.sqrt(train.var(0).sum())
        sequence=z.reshape(len(data['clouds']),8,-1)
        step=np.linalg.norm(np.diff(sequence,axis=1),axis=-1)/scale
        records.append(dict(model=name,dt_ps=.3,median_relative_step=float(np.median(step)),p95_relative_step=float(np.quantile(step,.95))))
        np.save(directory/f'{name}.npy',sequence)
    save_csv(directory/'continuity.csv',records)
    write_json(directory/'status.json',dict(state='complete',normalization='L2 step / sqrt(sum training Al feature variances); thermal motion is real and lower step alone is not evidence of a better embedding'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('prepare','evaluate'),required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());out=ROOT/cfg['output']
    directory=out/'temporal';directory.mkdir(exist_ok=True);torch.set_num_threads(4)
    if args.stage=='prepare':prepare(cfg,out,directory)
    else:evaluate(cfg,out,directory)


if __name__=='__main__':main()
