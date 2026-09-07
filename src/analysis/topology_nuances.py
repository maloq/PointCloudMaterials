"""Training-only topology reliability audit and frozen MACE decoder comparison."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
from src.data_utils.temporal_campaign import write_json


def perturb_patch(args):
    from src.analysis.liquid_structure import persistence_image
    x,seed,sigmas=args;rng=np.random.default_rng(seed);base=persistence_image(x[:65])
    result={'base':base};changes={}
    for sigma in sigmas:
        noise=rng.normal(0,sigma,x.shape);noise[0]=0
        y=x+noise;order=np.argsort(np.square(y).sum(1))
        result[f'fixed_{sigma}']=persistence_image(y[:65])
        result[f'resorted_{sigma}']=persistence_image(y[order[:65]])
        changes[str(sigma)]=int(len(set(order[:65])-set(range(65))))
    q,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(q)<0:q[:,0]*=-1
    # Restrict rotated input to the same 65 atoms, just as all other descriptors.
    result['rotated']=persistence_image((x@q)[:65])
    y=x+rng.normal(0,.005,x.shape);y-=y[0]
    result['precision_reference']=persistence_image(y[:65])
    result['precision_float16']=persistence_image(y[:65].astype(np.float16).astype(np.float64))
    return result,changes


def stability(plan):
    from src.data_utils.pretrained_mace import Quadruplets
    cfg=json.loads(Path(plan['source_config']).read_text());data=Quadruplets(cfg)
    source=Path(cfg['output']);out=Path(plan['output']);out.mkdir(parents=True,exist_ok=True)
    with np.load(source/'scaling.npz') as f:sc=dict(f)
    rng=np.random.default_rng(plan['seed'])
    ids=np.concatenate([p[rng.choice(len(p),plan['audit_anchors_per_material'],replace=False)] for p in data.pools['train']])
    x,t,c,m=data.get(ids)
    jobs=[(a.astype(np.float64),plan['seed']+i,plan['perturbation_sigmas_A']) for i,a in enumerate(x[:,0])]
    with ProcessPoolExecutor(max_workers=plan['audit_workers'],mp_context=mp.get_context('spawn')) as pool:results=list(pool.map(perturb_patch,jobs,chunksize=8))
    arrays={k:np.stack([r[0][k] for r in results]) for k in results[0][0]}
    transform=lambda a:((a-sc['tda_mean'])@sc['tda_components'].T)/sc['tda_std']
    y=transform(t[:,0]);b=transform(arrays['base'])
    signal=np.mean([np.var(y[m==i],axis=0) for i in range(3)],axis=0)
    errors={k:np.mean(np.square(transform(v)-b),axis=0) for k,v in arrays.items() if k not in ('base','precision_reference','precision_float16')}
    errors['cache_target']=np.mean(np.square(y-b),axis=0)
    errors['float16']=np.mean(np.square(transform(arrays['precision_float16'])-transform(arrays['precision_reference'])),axis=0)
    sigma=plan['reliability_sigma_A']
    # This suppresses sensitivity at an explicit small geometric tolerance. It
    # is not a claim that physical vibrations at this amplitude are meaningless.
    nuisance=np.maximum.reduce([errors[f'resorted_{sigma}'],errors['cache_target'],errors['float16'],errors['rotated']])
    weights=np.clip(signal/(signal+plan['noise_penalty']*nuisance+1e-12),.05,1.)
    all_y=transform(t)
    distance=lambda a,b:np.sum(np.square(a-b)*weights,axis=-1)/weights.sum()
    eligible=data.temporal_mask(ids);spatial=[];temporal=[]
    for i in range(3):
        spatial.append(float(np.median(distance(all_y[m==i,0],all_y[m==i,1]))))
        use=(m==i)&eligible;temporal.append(float(np.median(distance(all_y[use,0],all_y[use,2]))))
    if min(spatial+temporal)<=0:raise ValueError('Topology attraction calibration requires positive within-material median distances')
    result=dict(train_anchors=len(ids),sigmas_A=plan['perturbation_sigmas_A'],reliability_sigma_A=sigma,component_signal_variance=signal.tolist(),component_perturbation_mse={k:v.tolist() for k,v in errors.items()},reliability_weights=weights.tolist(),mean_whitened_mse={k:float(v.mean()) for k,v in errors.items()},neighbor_replacement_fraction={str(s):float(np.mean([r[1][str(s)]>0 for r in results])) for s in plan['perturbation_sigmas_A']},spatial_distance_medians=spatial,temporal_distance_medians=temporal,scope='Training data only. Local 80-atom pool; tests 65-neighbor membership changes within that pool. Not a liquid-only or full-trajectory precision audit.')
    write_json(out/'target_stability.json',result)
    print('TARGET_STABILITY',json.dumps(result),flush=True)


def decoder(plan):
    import torch
    from torch import nn
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from src.data_utils.pretrained_mace import Quadruplets
    from src.training_methods.pretrained_mace import Learner
    from src.analysis.pretrained_mace_ablation import features
    cfg=json.loads(Path(plan['source_config']).read_text());out=Path(plan['output']);data=Quadruplets(cfg)
    torch.set_num_threads(4);torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed'])
    model=Learner(cfg).cuda().eval();saved=torch.load(Path(cfg['output'])/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True)
    with np.load(Path(cfg['output'])/'scaling.npz') as f:sc=dict(f)
    # Inner decoder tuning is split by center IDs, keeping all views of an
    # anchor together. Repository validation sources remain final evaluation.
    indices=np.concatenate([p[rng.choice(len(p),plan['decoder_anchors_per_material'],replace=False)] for p in data.pools['train']])
    val=data.validation_indices(cfg['validation_anchors_per_material'],np.random.default_rng(cfg['seed']))
    z,y,c,m=features(model,data,indices,sc,cfg);v,w,d,n=features(model,data,val,sc,cfg)
    inner=[]
    for shard,row in indices:
        atom=int(np.load(Path(data.records[shard]['directory'])/'ids.npy',mmap_mode='r')[row,0]);inner.append(atom%5==0)
    inner=np.array(inner);scaler=StandardScaler().fit(z[~inner].reshape(-1,256))
    x=torch.tensor(scaler.transform(z[~inner].reshape(-1,256)),device='cuda',dtype=torch.float32);t=torch.tensor(y[~inner].reshape(-1,32),device='cuda')
    ix=torch.tensor(scaler.transform(z[inner].reshape(-1,256)),device='cuda',dtype=torch.float32);it=torch.tensor(y[inner].reshape(-1,32),device='cuda')
    vx=torch.tensor(scaler.transform(v.reshape(-1,256)),device='cuda',dtype=torch.float32)
    with torch.no_grad():original=model.tda(torch.tensor(v,device='cuda')).cpu().numpy()
    with torch.no_grad():original_inner_mse=float((model.tda(torch.tensor(z[inner],device='cuda'))-torch.tensor(y[inner],device='cuda')).square().mean())
    baseline=Ridge(alpha=10).fit(x.cpu().numpy(),t.cpu().numpy()).predict(vx.cpu().numpy()).reshape(w.shape)
    head=nn.Sequential(nn.Linear(256,512),nn.SiLU(),nn.Linear(512,512),nn.SiLU(),nn.Linear(512,32)).cuda()
    optimizer=torch.optim.AdamW(head.parameters(),lr=plan['decoder_lr'],weight_decay=1e-4)
    best=float('inf');bad=0;history=[]
    for epoch in range(plan['decoder_epochs']):
        head.train();order=torch.randperm(len(x),device='cuda')
        for ids in order.split(1024):
            optimizer.zero_grad(set_to_none=True);loss=(head(x[ids])-t[ids]).square().mean();loss.backward();optimizer.step()
        head.eval()
        with torch.no_grad():score=float((head(ix)-it).square().mean())
        history.append(dict(epoch=epoch+1,inner_validation_mse=score))
        if score<best:
            best=score;bad=0;state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        else:bad+=1
        if bad>=10:break
    head.load_state_dict(state)
    with torch.no_grad():prediction=head(vx).cpu().numpy().reshape(w.shape)
    def metrics(p):
        return dict(mse=float(np.square(p-w).mean()),by_material={name:float(np.square(p[n==i]-w[n==i]).mean()) for i,name in enumerate(('Al','Mg','Ta'))})
    result=dict(original_training_head=metrics(original),matched_ridge=metrics(baseline),larger_frozen_mlp=metrics(prediction),relative_mse_improvement=1-float(np.square(prediction-w).mean()/np.square(original-w).mean()),training_anchors=int((~inner).sum()),inner_validation_anchors=int(inner.sum()),held_out_source_anchors=len(val),best_inner_mse=best,original_inner_mse=original_inner_mse,epochs=len(history),history=history,interpretation='Single-seed decoder diagnostic, not a capacity proof. Inner split groups anchor atom IDs; different spatial views can still overlap. Held-out repository validation sources never select decoder epochs.')
    torch.save(dict(state_dict=state,feature_mean=scaler.mean_,feature_scale=scaler.scale_,config=plan),out/'frozen_decoder.pt')
    write_json(out/'decoder_comparison.json',result);print('DECODER_COMPARISON',json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--stage',choices=['stability','decoder'],required=True);args=p.parse_args();plan=json.loads(Path(args.plan).read_text())
    (stability if args.stage=='stability' else decoder)(plan)


if __name__=='__main__':main()
