"""Two-device deterministic encoder replay, one global SIGReg and prediction objective.

Uses the established ParallelMACE scheme: cache independent graph microbatches on
both GPUs, differentiate one global objective, replay exact derivatives, SUM the
encoder gradients, then update and synchronize parameters. No per-device SIGReg.
"""
from concurrent.futures import ThreadPoolExecutor
import torch
import resource
from src.data.structural_pretraining.batches import move
from src.training_methods.shared_pretraining.compilation import compile_encoder
from .model import Encoder


def configure_host():
    # Spawned loader workers inherit this limit. A 2,048-anchor batch carries
    # 224 graph microbatches and thousands of shared-memory tensor descriptors.
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard!=resource.RLIM_INFINITY and hard<65536:
        raise RuntimeError(f'Dual-GPU large batches require RLIMIT_NOFILE >= 65536; hard limit is {hard}')
    resource.setrlimit(resource.RLIMIT_NOFILE,(65536,hard))


class ParallelEncoder:
    def __init__(self,model,example,precision,compiled):
        if torch.cuda.device_count()!=2:
            raise ValueError('This run requires exactly two allocated visible GPUs')
        self.devices=[torch.device('cuda',0),torch.device('cuda',1)]
        self.encoders=[model.encoder]
        self.precision=precision
        with torch.random.fork_rng(devices=[0,1]),torch.cuda.device(1):
            secondary=Encoder(channels=model.encoder.base.channels).cuda(1)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                secondary(move(example,self.devices[1]))
            secondary.load_state_dict(model.encoder.state_dict(),strict=True)
            if compiled:compile_encoder(secondary,move(example,self.devices[1]),precision)
            self.encoders.append(secondary)
        self.pool=ThreadPoolExecutor(max_workers=2)

    def close(self):
        self.pool.shutdown(wait=True)

    def cache(self,index,work):
        device=self.devices[index]
        encoder=self.encoders[index]
        encoder.train()
        encoder.zero_grad(set_to_none=True)
        result=[]
        with torch.cuda.device(device),torch.no_grad():
            for ordinal,batch in work:
                batch=move(batch,device)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.precision=='bf16'):
                    states=encoder(batch).float()
                result.append((ordinal,batch,states))
            torch.cuda.synchronize(device)
        return result

    def replay(self,index,work):
        device=self.devices[index]
        with torch.cuda.device(device):
            for batch,derivative in work:
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.precision=='bf16'):
                    states=self.encoders[index](batch).float()
                states.backward(derivative.to(device))
            torch.cuda.synchronize(device)

    def step(self,model,objective,batches,target,diagnose=False):
        if len(batches)<2:raise ValueError('Both GPUs need at least one graph microbatch')
        futures=[self.pool.submit(self.cache,i,list(enumerate(batches))[i::2]) for i in range(2)]
        cached=[f.result() for f in futures]
        ordered=sorted((row for rows in cached for row in rows),key=lambda row:row[0])
        encoded=torch.cat([z.to(self.devices[0]) for _,_,z in ordered]).detach().requires_grad_(True)
        loss,terms=objective(model,encoded,target)
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite dual-GPU JEPA loss: {terms}')
        diagnostics=dict(objective.diagnostics)
        if diagnose:
            for name,value in terms.items():
                diagnostics[f'export_gradient/{name}']=float(torch.autograd.grad(value,encoded,retain_graph=True)[0].norm())
        loss.backward()
        offsets=[0]
        for _,_,z in ordered:offsets.append(offsets[-1]+len(z))
        torch.cuda.synchronize(0)
        futures=[self.pool.submit(self.replay,i,[(batch,encoded.grad[offsets[j]:offsets[j+1]]) for j,batch,_ in rows])
                 for i,rows in enumerate(cached)]
        for future in futures:future.result()
        for primary,secondary in zip(model.encoder.parameters(),self.encoders[1].parameters(),strict=True):
            if secondary.grad is not None:
                contribution=secondary.grad.to(self.devices[0])
                if primary.grad is None:primary.grad=contribution
                else:primary.grad.add_(contribution)
        return float(loss.detach()),{k:float(v.detach()) for k,v in terms.items()},diagnostics

    @torch.no_grad()
    def synchronize(self):
        for primary,secondary in zip(self.encoders[0].parameters(),self.encoders[1].parameters(),strict=True):
            secondary.copy_(primary)
        for device in self.devices:torch.cuda.synchronize(device)
