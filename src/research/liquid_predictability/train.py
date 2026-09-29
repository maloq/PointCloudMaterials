"""Resumable one-GPU fits and local controls using one predictive selector."""
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from src.data.fixed_cohort.protocol import sha,digest,write_json
from src.project_runtime.paths import resolve_path
from src.research.crystal_vector.train import deadline,compile_model
from src.research.crystal_vector.model import vcreg
from src.research.distance_encoder.model import loss_terms
from src.research.spatial_distance.model import capped_mean,cdf
from src.research.supervised_onset.tracking import tracked_run,local_evaluation,update_training_summary
from src.research.equivariant_context.cache import RetainedCache
from .data import config,population,masks,features
from .models import DescriptorMixture,DistanceMACE,initialize_descriptor

class Data:
    def __init__(self,c,arm,device):
        root,manifest,plan,self.meta,self.base=population(c);self.manifest=manifest;self.plan=plan
        self.split,self.weights,self.sources=masks(self.meta,self.base,arm,c);self.device=device
        self.arm=arm;self.root=root;self.encoded=None;self.positions=None
        self.distance=torch.tensor(self.meta['crystal_distance'],device=device)
        if arm['model'] in ('prior','linear','mlp'):
            x,_,_=features(c,self.meta);self.x=torch.tensor(x,device=device)
        else:
            ids=np.concatenate(list(self.split.values()));patches=np.unique(self.meta['indices'][ids]);self.patches=patches
            self.indices=torch.tensor(np.searchsorted(patches,self.meta['indices']),device=device)
            self.actual=torch.tensor(self.meta['actual'],device=device)
            self.positions=torch.empty((len(patches),80,3),dtype=torch.float32,device=device)
            offset=0
            for item in manifest['sources']:
                path=root/'sources'/str(item['source'])/'positions.npy'
                if sha(path)!=item['files']['positions.npy']:raise ValueError('Changed geometry')
                lo,hi=np.searchsorted(patches,[offset,offset+item['patches']]);bank=np.load(path,mmap_mode='r')
                for start in range(lo,hi,8192):
                    end=min(start+8192,hi);self.positions[start:end].copy_(torch.tensor(bank[patches[start:end]-offset],device=device))
                offset+=item['patches']
    def batch(self,ids):
        ix=torch.as_tensor(ids,device=self.device);b=dict(distance=self.distance[ix])
        if self.arm['model'] in ('prior','linear','mlp'):b['features']=self.x[ix]
        elif self.encoded is not None:
            value=self.encoded[self.indices[ix]];b.update(z=value[...,:128],v=value[...,128:].reshape(-1,25,16,3),actual=self.actual[ix])
        else:
            patches,inverse=torch.unique(self.indices[ix].flatten(),return_inverse=True)
            b.update(positions=self.positions[patches],inverse=inverse.reshape(-1,25),actual=self.actual[ix])
        return b

def study_record(c,arm,data):
    root=resolve_path(c['output'])/arm['name'];tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[3]
    files=list(Path(__file__).parent.glob('*.py'))+[repo/p for p in (
        'src/research/crystal_vector/model.py','src/research/crystal_vector/parallel.py',
        'src/research/equivariant_context/model.py','src/research/equivariant_context/features.py',
        'src/models/encoders/spatial_mace.py','src/research/encoder_context/geometry.py',
        'src/research/supervised_onset/model.py','src/research/spatial_distance/model.py',
        'src/research/distance_encoder/model.py')]
    binding=dict(config=c,arm=arm,dataset=data.manifest['identity'],train_sources=data.sources.tolist(),
        implementation={str(p.relative_to(repo)):sha(p) for p in files})
    identity=digest(binding)
    if (tech/'identity.json').exists() and json.loads((tech/'identity.json').read_text())!=binding:raise ValueError('Fit identity changed')
    write_json(tech/'identity.json',binding)
    tracking=dict(c,wandb=dict(c['wandb'],display_name=f'Liquid information | {arm["name"]} | NLL | seed {c["seed"]}'))
    return SimpleNamespace(config=tracking,root=root,technical=tech,identity=identity)

@torch.no_grad()
def initialize(c,arm,data,device):
    torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    if arm['model'] in ('prior','linear','mlp'):
        model=DescriptorMixture(arm['model']).to(device);ids=data.split['train']
        initialize_descriptor(model,data.x[ids].cpu().numpy().astype(float),data.meta['crystal_distance'][ids],data.weights['train'])
        return model,None
    path=resolve_path(c['initial_encoder'])
    if sha(path)!=c['initial_encoder_sha256']:raise ValueError('Changed parent encoder')
    saved=torch.load(path,map_location=device,weights_only=False);enc=saved['encoder_config']
    model=DistanceMACE(enc,c).to(device)
    if arm['initialization']=='snapshot_parent':model.encoder.load_state_dict(saved['encoder'],strict=True)
    elif arm['initialization']!='scratch':raise ValueError('Unknown encoder initialization')
    for p in list(model.direction_channels.parameters())+list(model.direction_offset.parameters()):p.requires_grad_(False)
    if arm['model']=='frozen':
        for p in list(model.encoder.parameters())+list(model.vector_export.parameters()):p.requires_grad_(False)
    rng=np.random.default_rng(c['seed']);ids=rng.choice(data.split['train'],256,replace=True,p=data.weights['train'])
    patches=np.unique(data.indices[torch.as_tensor(ids,device=device)].cpu().numpy())
    # Random subset, not the first sorted source/patch IDs.
    patches=rng.choice(patches,min(4096,len(patches)),replace=False);model.eval()
    with torch.autocast('cuda',dtype=torch.bfloat16):z,v=model.encode(data.positions[torch.as_tensor(patches,device=device)])
    model.scalar_mean.copy_(z.mean(0));model.scalar_scale.copy_(z.std(0,unbiased=False).clamp_min(1e-4))
    model.vector_scale.copy_(v.square().mean((0,2)).sqrt().clamp_min(1e-4))
    return model,enc

@torch.no_grad()
def score(model,data,c,role):
    model.eval();total=np.zeros(5);ids=data.split[role];w=data.weights[role]
    for begin in range(0,len(ids),c['batch_size']):
        b=data.batch(ids[begin:begin+c['batch_size']])
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=data.device.type=='cuda'):out=model(b)
        _,nll,_=loss_terms(out['parts'],b['distance'],c['loss']);mean=capped_mean(out['parts'],64)
        radii=b['distance'].new_tensor([20,32,48]);prob=cdf(out['parts'],radii)
        values=torch.cat((nll[:,None],(mean-b['distance'].clamp(max=64)).square()[:,None],(prob-(b['distance'][:,None]<=radii).to(prob.dtype)).square()),1)
        total+=(values.double().cpu().numpy()*w[begin:begin+len(b['distance']),None]).sum(0)
    return dict(distance_nll=float(total[0]),distance_rmse_A=float(np.sqrt(total[1])),
        brier20A=float(total[2]),brier32A=float(total[3]),brier48A=float(total[4]))

def frozen_features(model,data,c,study,stop):
    """Hold an LRU lease until the frozen readout and export finish."""
    metadata=dict(study=study.identity,parent=c['initial_encoder_sha256'],type='liquid-typed-patches')
    cache=RetainedCache(resolve_path(c['cache_policy']['features']),6)
    return cache.lease(digest(metadata),deadline=stop,metadata=metadata)

@torch.no_grad()
def load_features(model,data,folder,study):
    file=folder/'features.npy';receipt=folder/'complete.json'
    if not receipt.exists():
        model.eval();a=np.lib.format.open_memmap(folder/'features.building.npy',mode='w+',dtype=np.float32,shape=(len(data.positions),176))
        for begin in range(0,len(a),4096):
            with torch.autocast('cuda',dtype=torch.bfloat16):z,v=model.encode(data.positions[begin:begin+4096])
            a[begin:begin+len(z)]=torch.cat((z,v.flatten(1)),1).cpu().numpy()
        a.flush();del a;(folder/'features.building.npy').replace(file)
        write_json(receipt,dict(rows=len(data.positions),sha256=sha(file)))
    done=json.loads(receipt.read_text())
    if done['rows']!=len(data.positions) or sha(file)!=done['sha256']:raise ValueError('Changed frozen cache')
    data.encoded=torch.empty((done['rows'],176),device=data.device);a=np.load(file,mmap_mode='r')
    for start in range(0,len(a),8192):data.encoded[start:start+8192].copy_(torch.tensor(np.array(a[start:start+8192]),device=data.device))
    data.positions=None;torch.cuda.empty_cache()
    write_json(study.technical/'feature-cache.json',dict(path=str(folder),**done))

def run(path,name,preflight=False):
    from contextlib import nullcontext
    c=config(path);arm=next(a for a in c['arms'] if a['name']==name)
    torch.set_num_threads(2 if arm['lane']=='cpu' else 1);torch.set_float32_matmul_precision('high')
    device=torch.device('cpu' if arm['lane']=='cpu' else 'cuda')
    data=Data(c,arm,device);model,enc=initialize(c,arm,data,device)
    if preflight:
        ids=np.random.default_rng(c['seed']).choice(data.split['train'],c['batch_size'],p=data.weights['train']);b=data.batch(ids)
        model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):out=model(b)
        _,nll,_=loss_terms(out['parts'],b['distance'],c['loss']);loss=nll.mean()
        if arm['model']=='joint':loss=loss+vcreg(out,c['regularization'])[0]
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite preflight')
        return dict(arm=name,batch=c['batch_size'],loss=float(loss.detach()),gradient_norm=float(norm),rows={r:len(v) for r,v in data.split.items()},parameters=sum(p.numel() for p in model.parameters()))
    study=study_record(c,arm,data);tech=study.technical
    if (study.root/'analyses/predictability-v1/technical/complete.json').exists():return
    encoder_ids={id(p) for p in model.encoder.parameters()}|{id(p) for p in model.vector_export.parameters()} if enc else set()
    groups=[dict(params=[p for p in model.parameters() if p.requires_grad and id(p) in encoder_ids],lr=c['training']['encoder_lr']),
            dict(params=[p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids],lr=c['training']['head_lr'])]
    opt=torch.optim.AdamW(groups,weight_decay=c['training']['weight_decay'],fused=device.type=='cuda')
    epoch=update=step=0;best=float('inf');rng=np.random.default_rng(c['seed'])
    if (tech/'last.pt').exists():
        old=torch.load(tech/'last.pt',map_location=device,weights_only=False)
        if old['identity']!=study.identity:raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(old['model']);opt.load_state_dict(old['optimizer']);epoch=old['epoch'];update=old['update'];step=old['step'];best=old['best'];rng.bit_generator.state=old['rng']
    if enc:compile_model(model,data,c)
    encoder_input=dict(type='native_MACE',geometry_only=True,radius_A=8,nearest_candidates=80,layers=2,channels=128,
        embedding=128,trainable=arm['model']=='joint',history=False,motion=False,conditions=[]) if enc else None
    predictor_input=dict(type='vector_messages' if enc else arm['model'],patches=0 if arm['model']=='prior' else 25,
        max_support_A=0 if arm['model']=='prior' else 32,conditions=[],training_only_teacher=None,
        features='scalar128/vector16/relative offsets' if enc else ('none' if arm['model']=='prior' else '224 invariant radial, bond-order and context-gradient summaries'))
    write_json(tech/'prediction-context.json',dict(encoder=encoder_input,predictor=predictor_input,
        population=arm['population'],target='nearest established crystal distance; direction excluded',
        labels_as_inputs=False,clearance_as_input=False,encoder_initialization=arm['initialization'],train_sources=data.sources.tolist(),fixed_cohort=c['fixed_dataset']))
    def save(file):
        tmp=file.with_suffix('.building.pt');torch.save(dict(identity=study.identity,model=model.state_dict(),optimizer=opt.state_dict(),
            epoch=epoch,step=step,update=update,best=best,rng=rng.bit_generator.state,config=c,arm=arm,encoder_config=enc),tmp);tmp.replace(file)
    stop=deadline(c);total=c['training']['epochs']*c['training']['updates_per_epoch']
    lease=frozen_features(model,data,c,study,stop) if arm['model']=='frozen' else nullcontext(None)
    if (tech/'complete.json').exists():
        with lease as cache:
            if cache is not None:load_features(model,data,cache,study)
            selected=torch.load(tech/'best.pt',map_location=device,weights_only=False);model.load_state_dict(selected['model'])
            from .evaluate import export
            fields=export(model,data,c,study)
            if arm['model']=='joint':update_training_summary(study,'fit',fields,evaluation='predictability-v1')
            write_json(tech/'state.json',dict(state='complete',updates=update))
        return
    tracking=tracked_run(study,'fit',job_type='encoder') if arm['model']=='joint' else local_evaluation(study,'fit','readout')
    with lease as cache,tracking as log:
        if cache is not None:load_features(model,data,cache,study)
        log.summary.update({'model/parameters':sum(p.numel() for p in model.parameters()),'data/train_rows':len(data.split['train']),
            'data/train_sources':len(data.sources),'training/global_batch':c['batch_size'],'checkpoint/selector':'minimum validation distance NLL, epoch 1 onward'})
        while epoch<c['training']['epochs']:
            model.train();started=time.time()
            while step<c['training']['updates_per_epoch']:
                if time.time()>stop:
                    save(tech/'last.pt');write_json(tech/'state.json',dict(state='checkpointed',update=update));return
                ids=rng.choice(data.split['train'],c['batch_size'],p=data.weights['train']);b=data.batch(ids);opt.zero_grad(set_to_none=True)
                factor=min((update+1)/c['training']['warmup_updates'],1)*(.05+.95*.5*(1+math.cos(math.pi*update/total)))
                for g,base in zip(opt.param_groups,(c['training']['encoder_lr'],c['training']['head_lr'])):g['lr']=base*factor
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):out=model(b)
                _,nll,_=loss_terms(out['parts'],b['distance'],c['loss']);penalty=nll.new_zeros(())
                if arm['model']=='joint':penalty=vcreg(out,c['regularization'])[0]*min((update+1)/c['regularization']['warmup_updates'],1)
                loss=nll.mean()+penalty
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite fit {name}/{update}')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();step+=1;update+=1
                if update%c['training']['log_every']==0:
                    row=dict(optimizer_update=update,**{'train/distance_nll':float(nll.mean().detach()),'train/vcreg':float(penalty.detach()),
                        'train/gradient_norm_before_clip':float(norm),'train/epoch':epoch+step/c['training']['updates_per_epoch'],
                        'train/encoder_lr':opt.param_groups[0]['lr'],'train/predictor_lr':opt.param_groups[1]['lr']})
                    log.log(row)
                    with (tech/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    write_json(tech/'state.json',dict(state='training',epoch=epoch,update=update,total_updates=total))
                if update%c['training']['save_every']==0:save(tech/'last.pt')
            metrics=score(model,data,c,'selection');epoch+=1;step=0
            if not np.isfinite(metrics['distance_nll']):raise FloatingPointError('Nonfinite validation')
            if metrics['distance_nll']<best:
                best=metrics['distance_nll'];save(tech/'best.pt');log.summary.update({'checkpoint/selected_epoch':epoch,'checkpoint/validation_distance_nll':best})
            save(tech/f'epoch-{epoch:02d}.pt');save(tech/'last.pt')
            row=dict(optimizer_update=update,epoch=epoch,seconds=time.time()-started,**{'validation/'+k:v for k,v in metrics.items()})
            log.log(row)
            with (tech/'validation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(dict(arm=name,**row)),flush=True)
        write_json(tech/'complete.json',dict(identity=study.identity,updates=update,best_sha256=sha(tech/'best.pt')))
        selected=torch.load(tech/'best.pt',map_location=device,weights_only=False);model.load_state_dict(selected['model'])
        from .evaluate import export
        fields=export(model,data,c,study)
        log.summary.update(fields)
        write_json(tech/'state.json',dict(state='complete',updates=update))
