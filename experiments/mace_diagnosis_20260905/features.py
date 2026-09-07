"""Audit the trained product branch, intermediate features, and input perturbations."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
import torch
from experiments.smooth_temporal_encoder_20260905.evaluate import load_models, power_pca, structure_fit, spectrum
from experiments.smooth_temporal_encoder_20260905.prepare import write_json
from experiments.smooth_temporal_encoder_20260905.run import construct, load_split, make_loss, paired_loss
from experiments.mace_diagnosis_20260905.readouts import metrics, predict


@torch.no_grad()
def layers(model,q):
    power=(model.density.power(q)-model.power_mean)/model.power_std
    block=q[...,:25]/model.moment_scale[None,:,None]
    mixed=model.mix(block.transpose(-1,-2)).transpose(-1,-2)
    product=model.product(mixed,None,q.new_ones(len(q),1))
    joined=torch.cat((power,product),1)
    h1=model.head[:2](joined)
    h2=model.head[2:4](h1)
    return dict(power=power,product=product,prehead=joined,hidden1=h1,hidden2=h2,output=model.head[4](h2),
                power_only_output=model.head(torch.cat((power,torch.zeros_like(product)),1)))


@torch.no_grad()
def batch_layers(model,q):
    chunks=[layers(model,b) for b in q.split(4096)]
    return {k:torch.cat([c[k] for c in chunks]) for k in chunks[0]}


@torch.no_grad()
def static_samples(cfg,pilot,out,model):
    coverage=json.loads((pilot/'full_static_Al/coverage.json').read_text())
    coords=np.load(pilot/'full_static_Al/coords.npy',mmap_mode='r')
    meta=dict(np.load(pilot/'full_static_Al/metadata.npz'))
    rng=np.random.default_rng(cfg['seed'])
    buckets={k:[] for k in ('original','quantized16','jitter_0.02','jitter_0.05','jitter_0.1')}
    selected=[]
    quantization=[]
    for frame in coverage['frames']:
        ids=rng.choice(frame['count'],cfg['static_samples_per_source'],replace=False)+frame['offset']
        selected.append(ids)
        points=np.load(frame['path'])
        distances,neighbors=cKDTree(points,balanced_tree=False).query(coords[ids],k=194,workers=4)
        assert distances[:,0].max()<1e-5
        assert distances[:,-1].min()>frame['radius_A']+0.7
        local=points[neighbors[:,1:-1]]
        centers=points[neighbors[:,0]]
        offsets=local-centers[:,None]
        quantized=local.astype(np.float16).astype(np.float32)-centers.astype(np.float16).astype(np.float32)[:,None]
        quantization.append(dict(source=frame['source'],offset_rms_A=float(np.sqrt(np.mean((quantized-offsets)**2)))))
        versions={'original':offsets,'quantized16':quantized}
        for sigma in (.02,.05,.1):
            # Fixed-neighbor assay: bounded independent offset perturbations.
            noise=rng.normal(0,sigma,offsets.shape).clip(-3*sigma,3*sigma).astype(np.float32)
            versions[f'jitter_{sigma}']=offsets+noise
        for name,x in versions.items():
            q=model.density(torch.tensor(x/frame['radius_A'],device='cuda'))
            buckets[name].append(q.cpu().numpy())
        print('Prepared static sensitivity:',frame['source'],flush=True)
    selected=np.concatenate(selected)
    np.savez(out/'static_sample_metadata.npz',rows=selected,labels=meta['ptm_labels'][selected],source=meta['source_ids'][selected])
    for name,values in buckets.items():
        np.save(out/f'static_{name}.moments.npy',np.concatenate(values))
    write_json(out/'quantization.json',quantization)
    return {k:torch.tensor(np.concatenate(v),device='cuda') for k,v in buckets.items()},meta['ptm_labels'][selected],selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    pilot,out=ROOT/cfg['pilot'],ROOT/cfg['output']
    out.mkdir(exist_ok=True)
    pcfg=json.loads((pilot/'config.json').read_text())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    scaling,models=load_models(pcfg,pilot)
    model=models['mace_product']
    qs,static_labels,static_ids=static_samples(cfg,pilot,out,model)
    manifest=json.loads((pilot/'data/manifest.json').read_text())
    q,meta={},{}
    audit_batch=None
    for split in ('train','val','test'):
        raw,meta[split]=load_split(pilot,manifest,split)
        q[split]=torch.tensor(raw[:,0],device='cuda')
        if split=='train':
            ids=np.random.default_rng(cfg['seed']).choice(len(raw),4096,replace=False)
            audit_batch=torch.tensor(raw[ids],device='cuda')
        del raw
    model.requires_grad_(True)
    loss=paired_loss(make_loss(128),model.forward_moments(audit_batch.flatten(0,1)).reshape(-1,3,128),.25)
    loss.backward()
    torch.manual_seed(456)
    initial=construct(pcfg,'mace_product',scaling)
    groups={}
    for group in ('mix','product','head'):
        trained=dict(getattr(model,group).named_parameters())
        reference=dict(getattr(initial,group).named_parameters())
        groups[group]=dict(parameters=sum(p.numel() for p in trained.values()),
            gradient_l2=float(torch.cat([p.grad.flatten() for p in trained.values()]).norm()),
            relative_weight_change=float(torch.cat([(p-reference[n]).detach().flatten() for n,p in trained.items()]).norm()/torch.cat([p.detach().flatten() for p in reference.values()]).norm()))
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    del initial,audit_batch
    feats={s:batch_layers(model,x) for s,x in q.items()}
    sf=batch_layers(model,qs['original'])
    with torch.no_grad():
        first=model.head[0]
        cp=feats['train']['power']@first.weight[:,:260].T
        cm=feats['train']['product']@first.weight[:,260:].T
        branch=dict(power_contribution_rms=float(cp.square().mean().sqrt()),product_contribution_rms=float(cm.square().mean().sqrt()),
            power_contribution_variance=float(cp.var(0).sum()),product_contribution_variance=float(cm.var(0).sum()),
            output_relative_change_if_product_zero=float((feats['train']['output']-feats['train']['power_only_output']).square().mean().sqrt()/feats['train']['output'].square().mean().sqrt()))
    report=dict(parameter_audit=groups,branch=branch,layers={},sensitivity={})
    for name in sf:
        selection,probe,p=structure_fit(feats['train'][name],meta['train']['labels'],feats['val'][name],meta['val']['labels'],sf[name],static_labels)
        report['layers'][name]=dict(selection=selection,static=metrics(static_labels,p),
             md_test=metrics(meta['test']['labels'],predict(probe,feats['test'][name])),
             static_spectrum=spectrum(sf[name]),train_spectrum=spectrum(feats['train'][name]))
        torch.save(probe,out/f'layer_{name}.probe.pt')
        print('Layer',name,'static F1',report['layers'][name]['static']['macro_f1'],flush=True)
        write_json(out/'features.json',report)
    for name in ('density_pca','power_mlp','mace_product'):
        probe=torch.load(pilot/'embeddings'/f'{name}_structure_probe.pt',weights_only=True)
        report['sensitivity'][name]={}
        for version,x in qs.items():
            with torch.no_grad():
                z=power_pca(model.density,x,scaling) if name=='density_pca' else torch.cat([models[name].forward_moments(b) for b in x.split(4096)])
            p=predict(probe,z)
            report['sensitivity'][name][version]=metrics(static_labels,p)
        write_json(out/'features.json',report)
    print(json.dumps(dict(parameter_audit=groups,branch=branch)),flush=True)


if __name__=='__main__':
    main()
