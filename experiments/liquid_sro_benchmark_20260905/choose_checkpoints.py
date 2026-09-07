"""Select retained checkpoints on full-population validation VICReg before probes."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from experiments.liquid_sro_benchmark_20260905.train import construct,jitter,encode
from experiments.smooth_temporal_encoder_20260905.run import make_loss
from experiments.smooth_temporal_encoder_20260905.prepare import write_json


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());out=ROOT/cfg['output']
    assert json.loads((out/'training_status.json').read_text())['state']=='complete'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    x=torch.tensor(np.load(out/'clouds.npy'),device='cuda')
    meta=dict(np.load(out/'metadata.npz'));material=torch.tensor(meta['material'],device='cuda')
    edge=torch.tensor(np.load(out/'edges.npy'),device='cuda');count=torch.tensor(np.load(out/'edge_counts.npy'),device='cuda')
    val=torch.tensor(np.flatnonzero(meta['split']=='val'),device='cuda')
    rng=torch.Generator(device='cuda').manual_seed(777)
    a=jitter(x[val],cfg['training'],rng);b=jitter(x[val],cfg['training'],rng)
    loss=make_loss(cfg['training']['projector_dim'])
    records=[];legacy=out/'embeddings_legacy_selection';legacy.mkdir(exist_ok=False)
    for name in ('MACE','SchNet','DensityMLP'):
        for seed in cfg['training']['seeds']:
            full_name=f'{name}_seed{seed}';directory=out/'models'/full_name
            model=construct(name,out).eval()
            projector=nn.Sequential(nn.Linear(128,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Linear(512,256)).cuda().eval()
            candidates=[]
            for filename in ('best.pt','final.pt'):
                payload=torch.load(directory/filename,map_location='cuda',weights_only=False)
                model.load_state_dict(payload['state_dict']);projector.load_state_dict(payload['projector'])
                pa=[];pb=[]
                for local in torch.arange(len(val),device='cuda').split(256):
                    ids=val[local].repeat(2)
                    z=model(torch.cat((a[local],b[local])).clone(),material[ids],edge[ids],count[ids])
                    zz=projector(z);pa.append(zz[:len(local)]);pb.append(zz[len(local):])
                value,parts=loss._loss(torch.cat(pa),torch.cat(pb))
                if not torch.isfinite(value):raise FloatingPointError(f'Nonfinite whole-validation loss for {full_name}/{filename}')
                candidates.append(dict(filename=filename,epoch=int(payload['epoch']),validation_loss=float(value),parts={k:float(v) for k,v in parts.items()}))
            selected=min(candidates,key=lambda r:r['validation_loss'])
            shutil.copyfile(directory/selected['filename'],directory/'selected.pt')
            model.load_state_dict(torch.load(directory/'selected.pt',map_location='cuda',weights_only=False)['state_dict'])
            shutil.copyfile(out/'embeddings'/f'{full_name}.npy',legacy/f'{full_name}.npy')
            np.save(out/'embeddings'/f'{full_name}.npy',encode(model,x,material,edge,count))
            record=dict(model=full_name,candidates=candidates,selected=selected)
            records.append(record);write_json(out/'checkpoint_selection.json',records)
            print(json.dumps(record),flush=True)
            del model,projector,payload;torch.cuda.empty_cache()
    for seed in cfg['training']['seeds']:
        name=f'MACE_untrained_seed{seed}';directory=out/'models'/name;directory.mkdir(exist_ok=False)
        torch.manual_seed(seed);model=construct('MACE_untrained',out).eval()
        torch.save(dict(state_dict=model.state_dict(),seed=seed,trained=False),directory/'selected.pt')
        np.save(out/'embeddings'/f'{name}.npy',encode(model,x,material,edge,count))
        del model;torch.cuda.empty_cache()
    write_json(out/'selection_status.json',dict(state='complete',criterion='Whole 2304-center held-out validation projected VICReg; fixed seed-777 jitter views; selection only between retained best (legacy ordered validation batches) and final epoch; no physical-target or test labels used',control='Also export three untrained reference MACE initializations to distinguish architecture from self-supervised training'))


if __name__=='__main__':main()
