"""Batched typed-field export and bounded, ordered CPU preparation.

Only the consumer owns CUDA work. Producers return immutable frame geometry;
no learned features are cached across checkpoints.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import time
import numpy as np
import torch
from torch import nn
from e3nn import o3
from src.models.encoders.graph_bank import GraphBank
from src.models.encoders.spatial_mace import compile_spatial_encoder
from src.research.structural_state.data import graph_arrays
from .model import envelope


def prepare_graphs(patches, cutoff, *, pin_memory):
    arrays=graph_arrays(patches,cutoff)
    arrays['groups']=np.repeat(np.arange(len(patches)),np.diff(arrays['offsets']))
    # Pin in the producer, not on the GPU consumer's critical path.
    for name in ('positions','edges','groups'):
        value=torch.from_numpy(arrays[name])
        arrays[name]=value.pin_memory() if pin_memory else value
    return arrays


@contextmanager
def prepared_frames(items, prepare, *, workers=2, capacity=2):
    """At most capacity futures plus the consumed frame; preserve input order.

    Consumer failures, producer failures and early exits cancel queued work and
    join active producers. Exceptions propagate; no frame is silently skipped.
    """
    if workers<1 or capacity<1:
        raise ValueError('Frame preparation requires positive workers and capacity')
    pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='context-geometry')
    pending=deque();items=iter(items)
    def submit():
        try:item=next(items)
        except StopIteration:return
        pending.append(pool.submit(prepare,item))
    def consume():
        for _ in range(capacity):submit()
        while pending:
            value=pending.popleft().result()
            submit()
            yield value
    frames=consume()
    try:yield frames
    finally:
        frames.close()
        for future in pending:future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)


class TypedPatchExport(nn.Module):
    """Compile the actual typed-feature forward, not encoder.forward's scalar z."""
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder

    def forward(self,graph):
        encoder=self.encoder;c=encoder.channels
        h=encoder.atom_features(graph);weight=graph['weight']
        def pool(value):
            weighted=value*weight.reshape((-1,)+(1,)*(value.ndim-1))
            return value.new_zeros((graph['size'],)+value.shape[1:]).index_add(
                0,graph['group'],weighted)/encoder.n_ref
        scalar=h[:,:c]
        values=[encoder.export_pooled(torch.cat((scalar[graph['centers']],pool(scalar)),-1))]
        offset=c
        for l in (1,2):
            width=2*l+1;part=h[:,offset:offset+c*width];offset+=c*width
            field=part.reshape(-1,width,c).transpose(1,2) if encoder.layout=='ir_mul' else part.reshape(-1,c,width)
            values.append(torch.cat((field[graph['centers']],pool(field)),1).flatten(1))
        return torch.cat(values,-1)


def bond_fields(positions,groups,center,n_ref,size):
    """Current-coordinate bonds with the original center exclusion and taper."""
    radius=positions.norm(dim=-1)
    weights=torch.stack((envelope(radius,5.),envelope(radius,8.)),1)*(1-center)
    fields=[]
    for l in (4,6):
        y=o3.spherical_harmonics(l,positions,normalize=True,normalization='component')
        weighted=weights[...,None]*y[:,None,:]
        fields.append(weighted.new_zeros(size,2,2*l+1).index_add(0,groups,weighted)/n_ref)
    return fields


class FeatureExtractor:
    """One reusable exporter per frozen encoder/checkpoint, with one D2H per frame."""
    def __init__(self,encoder,device,chunk=256,*,compile=True):
        if chunk<1:raise ValueError('Positive feature extraction chunk required')
        self.encoder=encoder;self.device=torch.device(device);self.chunk=chunk
        self.exporter=TypedPatchExport(encoder).eval()
        self.compile=compile;self.compiled=False
        self.last_timing={}

    @torch.no_grad()
    def __call__(self,arrays):
        start=time.perf_counter()
        uploaded=dict(arrays)
        for name in ('positions','edges','groups'):
            uploaded[name]=arrays[name].to(self.device,non_blocking=True)
        bank=GraphBank(uploaded,self.encoder,self.device,plan_cache_bytes=0,node_capacity=80*self.chunk)
        geometry_done=time.perf_counter();encoded=[]
        count=len(arrays['offsets'])-1
        for begin in range(0,count,self.chunk):
            graph=bank.batch(np.arange(begin,min(begin+self.chunk,count)))
            if self.compile and not self.compiled:
                compile_spatial_encoder(self.exporter,graph)
                self.compiled=True
            encoded.append(self.exporter(graph))
        encoded=torch.cat(encoded)
        encoder_done=time.perf_counter()
        fields=bond_fields(uploaded['positions'],uploaded['groups'],bank.center,self.encoder.n_ref,count)
        # Keep chunk results on device and transfer the complete frame once.
        packed=torch.cat((encoded,*(f.flatten(1) for f in fields)),-1)
        fields_done=time.perf_counter();host=packed.cpu().numpy();done=time.perf_counter()
        c=self.encoder.channels;zdim=encoded.shape[-1]-16*c
        cuts=np.cumsum([0,zdim,6*c,10*c,18,26])
        result={name:host[:,cuts[i]:cuts[i+1]].reshape((count,)+shape) for i,(name,shape) in
                enumerate((('z',(zdim,)),('f1',(2*c,3)),('f2',(2*c,5)),('f4',(2,9)),('f6',(2,13))))}
        self.last_timing=dict(patches=count,atoms=len(arrays['positions']),
            upload_geometry_submit_s=geometry_done-start,encoder_submit_s=encoder_done-geometry_done,
            bonds_pack_submit_s=fields_done-encoder_done,download_wait_s=done-fields_done,total_s=done-start)
        return result
