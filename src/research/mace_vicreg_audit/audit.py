"""Numerical VICReg audit and TDA-free readouts on fixed, real MACE features."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn
from src.data_utils.pretrained_mace import Quadruplets
from src.data_utils.temporal_campaign import write_json
from src.models.encoders.pretrained_mace import PretrainedMACEEncoder
from src.training_methods.contrastive_learning.vicreg import VICRegLoss
from src.training_methods.mace_objective import objective, variance_covariance, cached_step
from src.training_methods.mace_performance import encode_views, replay_chunks


def reference(dim=256, mode='identity'):
    cfg=OmegaConf.create(dict(vicreg_enabled=True,vicreg_weight=1.,vicreg_embed_dim=dim,
        vicreg_projector_mode=mode,vicreg_drop_ratio=0.,vicreg_jitter_std=0.))
    return VICRegLoss.from_config(cfg,input_dim=256)


def ref_loss(ref,z,mask):
    # The actual repository pair objective, with the producer's 0.1 ps mask.
    spatial,_=ref._loss(z[:,0],z[:,1])
    temporal,_=ref._loss(z[mask,0],z[mask,2])
    return .5*(spatial+temporal)


def conditional_loss(ref,z,mask,materials):
    """Same pair weights; variance/covariance within elements, sample weighted."""
    sim=.5*((z[:,0]-z[:,1]).square().mean()+(z[mask,0]-z[mask,2]).square().mean())
    terms=[]
    for x,m in ((z[:,0],materials),(z[:,1],materials),(z[mask,0],materials[mask]),(z[mask,2],materials[mask])):
        value=x.new_zeros(())
        for i in range(3):
            selected=m==i
            value+=selected.float().mean()*(25*ref._variance_loss(x[selected])+ref._covariance_loss(x[selected]))
        terms.append(value)
    return 25*sim+torch.stack(terms).mean()


def math_audit(out):
    torch.manual_seed(19);ref=reference();settings=dict(invariance=25.,variance=25.,covariance=1.,tda=1.,tda_start_epoch=6)
    results=[]
    for count in (12,1536):
        # Different view scales expose weighting differences that iid inputs hide.
        z=(torch.randn(count,3,256)*torch.tensor([.25,.5,.8])[None,:,None]).requires_grad_()
        mask=torch.ones(count,dtype=torch.bool)
        actual,parts=objective(None,z,None,mask,settings,1)
        values=[variance_covariance(z[:,i]) for i in range(3)]
        variance=torch.stack([ref._variance_loss(z[:,i]) for i in range(3)])
        covariance=torch.stack([ref._covariance_loss(z[:,i]) for i in range(3)])
        cov_error=float((torch.stack([v[1] for v in values])-covariance).abs().max().detach())
        same_weights=25*.5*((z[:,0]-z[:,1]).square().mean()+(z[:,0]-z[:,2]).square().mean())+25*variance.mean()+covariance.mean()
        reference_pairs=ref_loss(ref,z,mask)
        # Replace only the known sample-vs-population variance difference.
        corrected=actual-25*torch.stack([v[0] for v in values]).mean()+25*variance.mean()
        torch.testing.assert_close(corrected,same_weights,rtol=2e-6,atol=2e-6)
        ga=torch.autograd.grad(corrected,z,retain_graph=True)[0]
        gb=torch.autograd.grad(same_weights,z,retain_graph=True)[0]
        torch.testing.assert_close(ga,gb,rtol=2e-5,atol=2e-7)
        results.append(dict(batch=count,current_loss=float(actual.detach()),reference_equal_view_loss=float(same_weights.detach()),
            reference_pair_loss=float(reference_pairs.detach()),covariance_max_error=cov_error,
            corrected_gradient_max_error=float((ga-gb).abs().max()),
            variance_convention_loss_difference=float((actual-same_weights).detach())))
    write_json(out/'math_audit.json',results)


def data_audit(data,out):
    rows=[]
    for r in data.records:
        d=Path(r['directory']);ids=np.load(d/'ids.npy');frames=np.load(d/'frames.npy')
        np.testing.assert_array_equal(ids[:,0],ids[:,2])
        np.testing.assert_array_equal(frames[:,0],frames[:,1])
        lag=(frames[:,2]-frames[:,0])*r['cadence_ps']
        np.testing.assert_allclose(lag,r['lags'][0]*r['cadence_ps'],rtol=0,atol=1e-9)
        assert np.all(ids[:,0]!=ids[:,1]),r['name']
        rows.append(dict(shard=r['name'],split=r['split'],anchors=len(ids),lag_ps=float(lag[0]),
            temporal_eligible=bool(abs(lag[0]-.1)<1e-9)))
    write_json(out/'data_audit.json',rows)


def load_encoder(cfg,which):
    model=PretrainedMACEEncoder(cfg['pretrained_checkpoint'],performance=cfg['performance']).cuda()
    root=Path(cfg['output'])
    if which=='initial':state=torch.load(root/'initial_encoder.pt',map_location='cpu',weights_only=False)['encoder']
    else:
        state=torch.load(root/'best.pt',map_location='cpu',weights_only=False)['model']
        state={k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')}
    model.load_state_dict(state,strict=True)
    return model


def gradient_audit(model,data,cfg,out):
    rows=np.concatenate([p[:4] for p in data.pools['train']]);x,t,c,m=data.get(rows)
    batch=tuple(torch.from_numpy(a).cuda() for a in (x,t,c,m))
    mask=torch.tensor(data.temporal_mask(rows),device='cuda')
    learner=nn.Module();learner.encoder=model;learner.tda=nn.Linear(256,32).cuda()
    learner.tda.requires_grad_(False);local=dict(cfg,microbatch_size=7)
    learner.zero_grad(set_to_none=True);cached,parts=cached_step(learner,batch,mask,local,1)
    expected={k:p.grad.detach().clone() for k,p in learner.named_parameters() if p.requires_grad}
    assert all(p.grad is None for p in learner.tda.parameters())
    learner.zero_grad(set_to_none=True)
    z=encode_views(learner,batch[0][:,:3],batch[3],7)
    direct,_=objective(learner,z,batch[1][:,:3],mask,cfg['loss'],1);direct.backward()
    errors={}
    for name,p in learner.named_parameters():
        if not p.requires_grad:continue
        torch.testing.assert_close(p.grad,expected[name],rtol=1e-3,atol=1e-4,msg=name)
        errors[name]=float((p.grad-expected[name]).norm()/expected[name].norm().clamp_min(1e-12))
    # Same clipped AdamW update from both gradients, with no TDA gradients.
    a=copy.deepcopy(learner);b=copy.deepcopy(learner)
    oa=torch.optim.AdamW(a.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    ob=torch.optim.AdamW(b.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    direct_grads={k:p.grad for k,p in learner.named_parameters()}
    for k,p in a.named_parameters():p.grad=expected[k].clone() if p.requires_grad else None
    for k,p in b.named_parameters():p.grad=direct_grads[k].clone() if p.requires_grad else None
    for item,opt in ((a,oa),(b,ob)):
        torch.nn.utils.clip_grad_norm_(item.parameters(),cfg['gradient_clip'],error_if_nonfinite=True);opt.step()
    maximum=0.
    for (name,p),q in zip(a.named_parameters(),b.parameters()):
        torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6,msg=name)
        maximum=max(maximum,float((p-q).abs().max().detach()))
    fp32=PretrainedMACEEncoder(cfg['pretrained_checkpoint'],performance=dict(cfg['performance'],bf16_mode='none',compile_radial_mlp=False)).cuda()
    fp32.load_state_dict(model.state_dict(),strict=True)
    z32=encode_views(SimpleNamespace(encoder=fp32),batch[0][:,:3],batch[3],7)
    loss32,_=objective(None,z32,None,mask,cfg['loss'],1);loss32.backward()
    g32={k:p.grad for k,p in fp32.named_parameters() if p.requires_grad}
    precision_errors={name:float((p.grad-g32[name.removeprefix('encoder.')]).norm()/g32[name.removeprefix('encoder.')].norm().clamp_min(1e-12))
        for name,p in learner.named_parameters() if p.requires_grad}
    assert max(precision_errors.values())<1e-3,precision_errors
    write_json(out/'gradient_audit.json',dict(cached_loss=cached,direct_loss=float(direct.detach()),
        parameter_gradient_relative_errors=errors,adamw_parameter_max_error=maximum,tda_gradients_absent=True,
        fp32_loss=float(loss32.detach()),compensated_bf16_vs_fp32_gradient_relative_errors=precision_errors,
        forward_chunk_size=7,anchors=len(rows),views=3))
    model.zero_grad(set_to_none=True)


@torch.no_grad()
def extract(model,data,rows,cfg):
    blocks=[]
    for start in range(0,len(rows),512):
        x,_,_,m=data.get(rows[start:start+512])
        blocks.append(encode_views(SimpleNamespace(encoder=model),torch.from_numpy(x[:,:3]).cuda(),torch.from_numpy(m).cuda(),cfg['microbatch_size']).cpu())
    return torch.cat(blocks).cuda()


@torch.no_grad()
def stats(z,m,mask,ref):
    v=torch.stack([ref._variance_loss(view) for view in z.unbind(1)]).mean()
    c=torch.stack([ref._covariance_loss(view) for view in z.unbind(1)]).mean()
    a=z[:,0];total=a.var(0,unbiased=False).sum();between=a.new_zeros(())
    by={}
    for i,name in enumerate(('Al','Mg','Ta')):
        x=a[m==i];cov=torch.cov(x.double().T);e=torch.linalg.eigvalsh(cov).clamp_min(0);p=e/e.sum()
        s=z[m==i,1];eligible=(m==i)&mask;u=z[eligible,0];t=z[eligible,2]
        rng=np.random.default_rng(100+i);shuffle=torch.tensor(rng.permutation(len(x)),device=z.device)
        tshuffle=torch.tensor(rng.permutation(len(t)),device=z.device)
        between+=(m==i).float().mean()*(x.mean(0)-a.mean(0)).square().sum()
        by[name]=dict(rank=float(torch.exp(-(p*p.clamp_min(1e-20).log()).sum())),
            spatial_ratio=float((x-s).square().mean()/(x-s[shuffle]).square().mean()),
            temporal_ratio=float((u-t).square().mean()/(u-t[tshuffle]).square().mean()),mean_std=float(x.std(0,unbiased=False).mean()))
    return dict(reference_loss=float(ref_loss(ref,z,mask)),variance_penalty=float(v),covariance_penalty=float(c),
        mean_std=float(a.std(0,unbiased=False).mean()),between_element_variance_fraction=float(between/total),by_material=by)


def readout(ref,z):return torch.stack([ref.project_features(view) for view in z.unbind(1)],1)


def shrinkage_audit(tz,vz,tmask,vmask,cfg,out):
    results={};scales=torch.linspace(.01,1.2,11901,device=tz.device)
    for split,z,mask in (('train',tz,tmask),('val',vz,vmask)):
        loss,parts=objective(None,z,None,mask,cfg['loss'],1)
        variance=z.var(0,unbiased=True)
        penalty=torch.relu(1-torch.sqrt(scales[:,None,None].square()*variance[None]+1e-4)).mean((1,2))
        curve=parts['loss_invariance']*scales.square()+25*penalty+parts['loss_covariance']*scales.pow(4)
        best=int(curve.argmin());results[split]=dict(initial_loss=float(loss),best_scale=float(scales[best]),best_loss=float(curve[best]))
        if split=='train':chosen=float(scales[best])
        else:
            measured,_=objective(None,z*chosen,None,mask,cfg['loss'],1)
            results[split]['loss_at_train_selected_scale']=float(measured)
    write_json(out/'scalar_shrinkage.json',results)


class NormalizedGeometryEncoder(nn.Module):
    """A diagnostic geometry encoder using one pretrained Al species channel."""
    def __init__(self,base,radii):
        super().__init__();self.base=base
        self.register_buffer('radii',torch.tensor(radii,device='cuda'))

    def build_geometry(self,x,m):
        scaled=x*(self.radii[0]/self.radii[m])[:,None,None]
        return self.base.build_geometry(scaled,torch.zeros_like(m))

    def forward_from_geometry(self,g):return self.base.forward_from_geometry(g)
    def forward(self,x,m):return self.forward_from_geometry(self.build_geometry(x,m))


def joint_controls(config,cfg,data,train,val,tm,vm,tmask,vmask,out):
    """Matched short end-to-end controls; full-view projector BN outside replay."""
    x,_,_,_=data.get(train);clouds=torch.from_numpy(x[:,:3]).cuda()
    results={}
    for mode in config['joint_modes']:
        torch.manual_seed(config['seed']);model=load_encoder(cfg,'initial')
        if mode=='normalized_geometry_mlp':
            old=OmegaConf.load(config['geoframe_config']);radii={s.name:float(s.radius) for s in old.data.data_sources}
            model=NormalizedGeometryEncoder(model,[radii[m] for m in ('Al','Mg','Ta')])
            model.base.feature_mean.zero_();model.base.feature_std.fill_(1.)
            with torch.no_grad():
                fitted=extract(model,data,train[:4096],cfg)[:,0]
                model.base.feature_mean.copy_(fitted.mean(0));model.base.feature_std.copy_(fitted.std(0,unbiased=False).clamp_min(.01))
                a=clouds[:8,0];m=torch.zeros(8,dtype=torch.long,device='cuda')
                torch.testing.assert_close(model(a,m),model(a*(radii['Mg']/radii['Al']),m+1),rtol=1e-4,atol=2e-5)
        ref=reference(256,'identity' if mode=='reference_raw' else 'mlp').cuda()
        holder=SimpleNamespace(encoder=model)
        opt=torch.optim.AdamW([dict(params=[p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate']),
            dict(params=ref.parameters(),lr=config['readout_learning_rate'])],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,config['joint_steps'],eta_min=1e-6)
        rng=np.random.default_rng(config['seed']);history=[]
        for step in range(config['joint_steps']):
            model.train();ref.train();ids=torch.tensor(rng.choice(len(train),config['joint_batch_size'],replace=False),device='cuda')
            batch=clouds[ids];material=tm[ids];mask=tmask[ids]
            opt.zero_grad(set_to_none=True);geometries=[]
            with torch.no_grad():z=encode_views(holder,batch,material,cfg['microbatch_size'],geometries)
            z.requires_grad_(True);projected=readout(ref,z)
            loss=conditional_loss(ref,projected,mask,material) if mode=='conditional_mlp' else ref_loss(ref,projected,mask)
            if not bool(torch.isfinite(loss)):raise FloatingPointError(f'{mode} step {step}: {loss}')
            loss.backward()
            for actual,sl in replay_chunks(holder,batch.flatten(0,1),material.repeat_interleave(3),cfg['microbatch_size'],geometries):
                actual.backward(z.grad.flatten(0,1)[sl])
            norm=torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(ref.parameters()),cfg['gradient_clip'],error_if_nonfinite=True)
            opt.step();scheduler.step()
            if step%100==0 or step+1==config['joint_steps']:
                model.eval();ref.eval();v=extract(model,data,val,cfg)
                with torch.no_grad():
                    projected=readout(ref,v);record=dict(step=step+1,train_loss=float(loss.detach()),gradient_norm=float(norm),val=stats(projected,vm,vmask,ref),backbone_val=stats(v,vm,vmask,ref))
                    record['conditional_validation_loss']=float(conditional_loss(ref,projected,vmask,vm))
                history.append(record);write_json(out/(mode+'_joint.json'),history);print('JOINT',mode,json.dumps(record),flush=True)
        results[mode]=history[-1]
        torch.save(dict(encoder=model.state_dict(),projector=ref.projector.state_dict(),config=config),out/(mode+'_joint.pt'))
        del model,ref,opt,z,projected;torch.cuda.empty_cache()
    write_json(out/config['joint_results_file'],results)


@torch.no_grad()
def normalization_audit(config,cfg,data,out):
    """Separate geometric length scale from explicit atomic-number conditioning."""
    old=OmegaConf.load(config['geoframe_config'])
    radii={source.name:float(source.radius) for source in old.data.data_sources}
    radius=torch.tensor([radii[m] for m in ('Al','Mg','Ta')],device='cuda')
    rng=np.random.default_rng(config['seed']);pool=data.all_indices('train')
    train=pool[rng.choice(len(pool),config['normalization_train_anchors'],replace=False)]
    val=np.concatenate([p[rng.choice(len(p),min(config['normalization_val_per_material'],len(p)),replace=False)] for p in data.pools['val']])
    np.savez(out/'normalization_indices.npz',train=train,val=val)
    vm=torch.tensor([data.records[i]['material'] for i,j in val],device='cuda');mask=torch.tensor(data.temporal_mask(val),device='cuda')
    model=load_encoder(cfg,'initial').eval();model.feature_mean.zero_();model.feature_std.fill_(1.)
    ref=reference().cuda();results={}
    for normalized,common_element in ((False,False),(True,False),(False,True),(True,True)):
        blocks={};geometry={}
        for split,rows in (('train',train),('val',val)):
            features=[]
            for start in range(0,len(rows),512):
                x,_,_,m=data.get(rows[start:start+512]);x=torch.from_numpy(x[:,:3]).cuda();m=torch.from_numpy(m).cuda()
                if normalized:x=x*(radius[0]/radius[m])[:,None,None,None]
                if split=='val':
                    for i,name in enumerate(('Al','Mg','Ta')):
                        geometry.setdefault(name,[]).extend(x[m==i,0,1].norm(dim=-1).cpu().tolist())
                encoder_material=torch.zeros_like(m) if common_element else m
                features.append(encode_views(SimpleNamespace(encoder=model),x,encoder_material,cfg['microbatch_size']))
            blocks[split]=torch.cat(features)
        mean=blocks['train'][:,0].mean(0);std=blocks['train'][:,0].std(0,unbiased=False).clamp_min(.01)
        value=(blocks['val']-mean)/std
        name=f'{"common_length" if normalized else "physical_length"}_{"common_Al_channel" if common_element else "actual_elements"}'
        results[name]=dict(val=stats(value,vm,mask,ref),median_nearest_distance_A={k:float(np.median(v)) for k,v in geometry.items()})
        write_json(out/'normalization_audit.json',dict(source_radii_A=radii,reference_radius_A=radii['Al'],
            protocol='Diagnostic intervention only: x * R_Al/R_material keeps an Al-scale angstrom range; common_Al_channel changes node attributes to Al, not physical chemistry. No production weights or caches changed.',results=results))
        print('NORMALIZATION',name,json.dumps(results[name]),flush=True)


def run(config,stage):
    out=Path(config['output']);out.mkdir(exist_ok=True,parents=True)
    cfg=json.loads(Path(config['training_config']).read_text());torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    data=Quadruplets(cfg)
    if stage=='normalization':
        normalization_audit(config,cfg,data,out);return
    if stage=='gradients':
        gradient_audit(load_encoder(cfg,'initial'),data,cfg,out);return
    if stage=='all':math_audit(out);data_audit(data,out)
    rng=np.random.default_rng(config['seed']);pool=data.all_indices('train');train=pool[rng.choice(len(pool),config['train_anchors'],replace=False)];val=data.all_indices('val')
    np.savez(out/'sample_indices.npz',train=train,val=val)
    tm=torch.tensor([data.records[i]['material'] for i,j in train],device='cuda');vm=torch.tensor([data.records[i]['material'] for i,j in val],device='cuda')
    tmask=torch.tensor(data.temporal_mask(train),device='cuda');vmask=torch.tensor(data.temporal_mask(val),device='cuda')
    if stage=='joint':
        joint_controls(config,cfg,data,train,val,tm,vm,tmask,vmask,out);return
    if stage=='all':
        model=load_encoder(cfg,'initial');gradient_audit(model,data,cfg,out);model.eval()
        tz=extract(model,data,train,cfg);vz=extract(model,data,val,cfg)
        np.savez(out/'initial_features.npz',train=tz.cpu().numpy(),val=vz.cpu().numpy())
        ref=reference().cuda();results={'initial':dict(train=stats(tz,tm,tmask,ref),val=stats(vz,vm,vmask,ref))}
        del model;torch.cuda.empty_cache()
        model=load_encoder(cfg,'best').eval();best=extract(model,data,val,cfg)
        results['selected_epoch6']=dict(val=stats(best,vm,vmask,ref));del best,model;torch.cuda.empty_cache()
        write_json(out/'features_audit.json',results);print('FEATURE_AUDIT',json.dumps(results),flush=True)
    else:
        cache=np.load(out/'initial_features.npz');tz=torch.from_numpy(cache['train']).cuda();vz=torch.from_numpy(cache['val']).cuda()
        results=json.loads((out/'readout_results.json').read_text())
    shrinkage_audit(tz,vz,tmask,vmask,cfg,out)
    for name in config['readouts']:
        dim=128 if name.startswith('mlp128') else 256
        for seed in config['readout_seeds']:
            torch.manual_seed(seed);ref=reference(dim,'mlp' if name.startswith('mlp') else 'identity').cuda()
            if name=='linear256':
                ref.projector=nn.Linear(256,256,bias=False).cuda();nn.init.eye_(ref.projector.weight)
                ref.projector_mode='mlp'
            opt=torch.optim.AdamW(ref.parameters(),lr=config['readout_learning_rate'],weight_decay=cfg['weight_decay'])
            schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,config['readout_steps'],eta_min=1e-5)
            generator=np.random.default_rng(seed);history=[]
            for step in range(config['readout_steps']):
                ref.train();ids=torch.tensor(generator.choice(len(tz),config['batch_size'],replace=False),device='cuda')
                opt.zero_grad(set_to_none=True);z=readout(ref,tz[ids])
                loss=conditional_loss(ref,z,tmask[ids],tm[ids]) if name.endswith('_conditional') else ref_loss(ref,z,tmask[ids])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ref.parameters(),cfg['gradient_clip'],error_if_nonfinite=True);opt.step();schedule.step()
                if step%100==0 or step+1==config['readout_steps']:
                    ref.eval();projected=readout(ref,vz)
                    record=dict(step=step+1,train_loss=float(loss.detach()),val=stats(projected,vm,vmask,ref))
                    if name.endswith('_conditional'):record['conditional_validation_loss']=float(conditional_loss(ref,projected,vmask,vm).detach())
                    history.append(record);print(name,seed,json.dumps(record),flush=True)
            key=f'{name}_seed{seed}';results[key]=history[-1]
            write_json(out/(key+'.json'),history);torch.save(ref.projector.state_dict(),out/(key+'.pt'))
            write_json(out/'readout_results.json',results)
    write_json(out/'status.json',dict(state='complete',protocol='Frozen real MACE features; no encoder updates in readout controls; no TDA targets used in any objective.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--stage',choices=['all','readouts','joint','gradients','normalization'],default='all');args=parser.parse_args()
    run(json.loads(Path(args.config).read_text()),args.stage)
