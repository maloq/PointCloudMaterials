"""Fixed-budget physical trajectory fits with optional initial density rates."""
import math
import os
import time
import numpy as np
import torch
from torch import nn
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import online_training
from src.research.ni_latent_sim.transformer import TemporalTransformer
from .data import root, digest, atomic_save, write_json, sha
from .rate_data import load, FAMILY
from .model import feature_weights
from .learning import value_scores
from .train import objective, response_loss


class RateProcess(nn.Module):
    """Evolve244 measured coordinates, conditioned on an immutable initial token."""
    def __init__(self, c, rate, arm):
        super().__init__()
        self.h_fs = arm['step_fs']; self.use_rates = arm['rates']
        self.network = TemporalTransformer(244,2,c['model'])
        self.rate_projection = nn.Linear(74,c['model']['width'],bias=False)
        nn.init.zeros_(self.network.head.weight); nn.init.zeros_(self.network.head.bias)
        self.register_buffer('decay',torch.exp(-rate*self.h_fs/1000))
        self.register_buffer('noise_scale',(1-self.decay.square()).sqrt())
        self.register_buffer('active',(rate>0).float())

    def forward(self, initial, rates, particles, seed):
        batch = len(initial); width = self.network.slots.shape[1]
        first = self.network.projection(initial)
        first = first+self.rate_projection(rates*float(self.use_rates))
        first = first[:,None].expand(-1,particles,-1).reshape(-1,width)
        current = initial[:,None].expand(-1,particles,-1)
        generator = torch.Generator(device=initial.device).manual_seed(seed)
        noises = torch.randn(100//self.h_fs,batch,particles,244,device=initial.device,generator=generator)
        output = []
        for step,noise in enumerate(noises,1):
            tokens = torch.stack((first,self.network.projection(current).reshape(-1,width)),1)+self.network.slots
            for block in self.network.blocks: tokens = block(tokens)
            parameters = self.network.head(self.network.norm(tokens[:,-1])).reshape(batch,particles,488)
            drift,diffusion = parameters.float().chunk(2,-1)
            current = self.decay*current+self.active*(self.h_fs/1000)*20*torch.tanh(drift/20)
            current = current+self.noise_scale*torch.exp(2*torch.tanh(diffusion/2))*noise
            if step*self.h_fs % 5 == 0: output.append(current)
        return torch.stack(output,2)


def response_prediction(model, data, ids, epsilon, seed, particles, *, half=False):
    pairs = data['half_pairs' if half else 'pairs'][ids]
    rates = data['half_pair_rates' if half else 'pair_rates'][ids]
    predictions = []
    for k in range(2):
        a = model(pairs[:,k,1],rates[:,k,1],particles,seed)[...,153:].mean(1)
        b = model(pairs[:,k,0],rates[:,k,0],particles,seed)[...,153:].mean(1)
        predictions.append((a-b)/(2*epsilon))
    return torch.stack(predictions,-1)


def fit(c, arm, deadline):
    folder = root(c)/'analyses'/arm['name']/'technical'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists(): return True
    plan,records,data,n,norm = load(c,arm['bank'],device='cuda')
    cfg = c['training']; torch.manual_seed(arm['seed'])
    model = RateProcess(c,data['decay_rate'],arm).cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-5)
    weights = feature_weights(norm['scale'].cuda()); count = cfg['updates']; batch = cfg['batch_size']
    micro = min(batch,cfg['microbatch'])
    identity = digest([plan['identity'],arm]); rng = np.random.default_rng(arm['seed'])
    start = 0; history = []
    if (folder/'last.pt').exists():
        old = torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if old['identity'] != identity: raise ValueError('Changed resumed density-rate fit')
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        start,history = old['update'],old['history']; rng.bit_generator.state = old['rng']
    def save(update):
        atomic_save(folder/'last.pt',dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
            update=update,history=history,rng=rng.bit_generator.state,arm=arm,normalization=norm))
    execution = dict(allocation=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),
        train_states=n,train_ancestors=len({r['ancestor'] for r in records}),effective_batch=batch,
        microbatch=micro,neural_step_fs=arm['step_fs'],resumed_update=start,precision='FP32 neural states, network, measurements and losses',
        checkpoint_rule='Final predeclared update only; no test access while fitting')
    write_json(folder/'execution-segments'/f'{time.time_ns()}.json',execution)
    with online_training(c['wandb'],run_id='rates-'+identity[:16],name=arm['name'],config=dict(c,arm=arm),
        folder=folder,receipt_path=folder/'wandb.json',job_type='predictor',group=c['protocol'],receipt_fields=dict(identity=identity)) as tracking:
        for update in range(start+1,count+1):
            if time.time()>deadline-600: save(update-1); return False
            begin = time.monotonic(); model.train(); optimizer.zero_grad(set_to_none=True)
            warmup = min(update/cfg['warmup'],1.)
            cosine = .01+.99*.5*(1+math.cos(math.pi*max(0,update-cfg['warmup'])/(count-cfg['warmup'])))
            for group in optimizer.param_groups: group['lr'] = cfg['lr']*warmup*cosine
            chosen = rng.integers(n,size=batch); seed = int(rng.integers(2**31)); totals = {}; derivative = 0.
            for offset in range(0,batch,micro):
                ids = chosen[offset:offset+micro]; factor = len(ids)/batch
                prediction = model(data['initial'][ids],data['rates'][ids],cfg['particles'],seed+offset)
                scores = value_scores(prediction,data['target'][ids],data['initial'][ids],weights,
                                      data['increment_scale'],cfg,physical=True)
                (factor*objective(scores,cfg)).backward()
                for key,value in scores.items(): totals[key] = totals.get(key,0.)+factor*float(value.detach().mean())
                del prediction,scores
                if arm['response']:
                    predicted = response_prediction(model,data,ids,cfg['response_epsilon'],seed+offset,cfg['particles'])
                    loss = response_loss(predicted,data['response'][ids],data['normalizers'])
                    (factor*cfg['response_weight']*loss).backward(); derivative += factor*float(loss.detach())
                    del predicted,loss
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
            optimizer.step()
            row = dict(update=update,**totals,response_loss=derivative,gradient_norm=float(gradient))
            history.append(row); tracking.log(row,step=update)
            if update==start+1 or update%cfg['save_every']==0: save(update)
            write_json(folder/'progress.json',dict(state='training',**row,total=count,seconds_per_update=time.monotonic()-begin))
            if update==start+1 or update%50==0: print(arm['name'],row,flush=True)
        tracking.summary.update(dict(final_update=count,checkpoint_rule='fixed final; no selection split'))
    save(count)
    atomic_save(folder/'final.pt',dict(identity=identity,model=model.state_dict(),update=count,arm=arm,normalization=norm))
    write_metric_rows(history,folder.parent,family=FAMILY,name='learning')
    write_json(folder/'complete.json',dict(state='complete',updates=count,checkpoint_sha256=sha(folder/'final.pt'),identity=identity))
    return True
