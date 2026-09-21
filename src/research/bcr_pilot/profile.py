"""Bounded B256 throughput measurement, separate from scientific training."""
import copy
import json
import time
import torch
from src.project_runtime.paths import resolve_path
from src.training_methods.bcr.runtime import load_data
from src.training_methods.bcr.model import BCR
from src.training_methods.bcr.data import pack,corrupt
from src.training_methods.bcr.objective import per_environment


def measure(config,training,device='cuda'):
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.manual_seed(config['seed'])
    patches,manifest=load_data(resolve_path(config['data']));patches=[p for p,r in zip(patches,manifest['records']) if r['split']=='train']
    clean=pack(patches[:config['batch_size']],device);out=[]
    for arm in ('bcr','unconditional','frozen_random'):
        cfg=copy.deepcopy(training);cfg['arm']=arm;model=BCR(cfg).to(device);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
        generator=torch.Generator().manual_seed(23);timings=[]
        for step in range(6):
            torch.cuda.synchronize();started=time.monotonic();noisy,eps,sigma,_=corrupt(clean,manifest['noise_levels'],manifest['d0'],generator);optimizer.zero_grad(set_to_none=True)
            for start in range(0,len(sigma),config['microbatch']):
                sl=slice(start,start+config['microbatch']);a={k:v[sl] for k,v in clean.items()};b={k:v[sl] for k,v in noisy.items()}
                (per_environment(model(a,b,sigma[sl])[0],eps[sl],a,config['radius_A']).sum()/len(sigma)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();torch.cuda.synchronize();timings.append(time.monotonic()-started)
        seconds=sum(timings[2:])/4
        out.append(dict(arm=arm,seconds_per_update=seconds,ten_thousand_update_hours=seconds*config['updates']/3600,
            gpu=torch.cuda.get_device_name(),batch_size=config['batch_size'],microbatch=config['microbatch']))
        del model,optimizer;torch.cuda.empty_cache()
    path=resolve_path(config['output'])/'technical/profile.json';path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
    return out
