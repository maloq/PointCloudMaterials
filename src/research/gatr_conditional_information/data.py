"""Verified native observations and an explicit radial-only encoder control."""
import json
from pathlib import Path
import socket
import time
from functools import lru_cache

import numpy as np
import torch

from src.data.predictive_memory.targets import taper
from src.data.structural_pretraining.prepare import file_hash, save_json, REFERENCE_RADIUS
from src.research.gatr_equivariant.model import Capture
from src.research.gatr_equivariant.run import encode_rows
from src.research.forecast_crystallization.local_metrics import first_sustained_onset, risk_windows


def require_hardware(config):
    if socket.gethostname().split('.')[0] != config['required_hostname']:
        raise RuntimeError(f'Requires {config["required_hostname"]}, found {socket.gethostname()}')
    if not torch.cuda.is_available() or config['required_gpu'] not in torch.cuda.get_device_name(0):
        raise RuntimeError('Requires the requested A100')
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')


@lru_cache(None)
def fibonacci_directions(n):
    rank = np.arange(n,dtype=float)
    z = 1-2*(rank+.5)/n
    angle = rank*np.pi*(3-np.sqrt(5))
    xy = np.sqrt(1-z*z)
    return np.column_stack((xy*np.cos(angle),xy*np.sin(angle),z))


def radial_control(local,scale,quantiles):
    from src.data.structural_pretraining.support import INNER_RADIUS, OUTER_RADIUS
    r = np.linalg.norm(local.astype(float),axis=-1)
    r = r[r*REFERENCE_RADIUS/scale < OUTER_RADIUS]
    if np.count_nonzero(r==0)!=1:
        raise ValueError('Exactly one center is required')
    r = np.sort(r[r>0])
    if len(r)<80:
        raise ValueError('Incomplete first 80 neighbor radii')
    q = np.quantile(r,np.linspace(0,1,quantiles))
    weights = taper(r*REFERENCE_RADIUS/scale,INNER_RADIUS,OUTER_RADIUS)
    moments = np.array([len(r),weights.sum(),*(weights@(r**k)/weights.sum() for k in (1,2,3))])
    # A function of the radius multiset only, with a fixed deterministic angular layout.
    replaced = np.vstack((np.zeros((1,3)),r[:,None]*fibonacci_directions(len(r)))).astype(np.float32)
    error = np.max(np.abs(np.linalg.norm(replaced[1:].astype(float),axis=-1)-r))
    return replaced,r[:80],q,moments,float(error)


def prepare(config):
    require_hardware(config)
    root = Path(config['output']); technical=root/'technical'; technical.mkdir(parents=True,exist_ok=True)
    if (technical/'config.json').exists() and json.loads((technical/'config.json').read_text())!=config:
        raise ValueError('Existing conditional protocol configuration differs')
    save_json(technical/'config.json',config)
    parent = json.loads((Path(config['parent_audit'])/'technical/plan.json').read_text())
    checkpoint=Path(config['parent_audit'])/'technical/gatr.pt'
    model = Capture(checkpoint,config['checkpoint_sha256']).cuda().eval()
    save_json(technical/'environment.json',dict(host=socket.gethostname(),gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,checkpoint_sha256=file_hash(checkpoint),parent_plan_sha256=file_hash(Path(config['parent_audit'])/'technical/plan.json')))
    for source in parent['sources']:
        if source['split']!='test': continue
        folder=Path(config['parent_audit'])/'technical/sources'/str(source['id'])
        dest=technical/'sources'/str(source['id']);dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists(): continue
        started=time.monotonic()
        receipt=json.loads((folder/'complete.json').read_text())
        for name,digest in receipt['hashes'].items():
            if file_hash(folder/name)!=digest: raise ValueError(f'Changed observation file: {folder/name}')
        encoder_receipts={}
        for name in ('gatr','mace'):
            rec=json.loads((folder/f'{name}-verification.json').read_text())
            if file_hash(folder/f'{name}.npy')!=rec['features_sha256']: raise ValueError(f'Changed {name} embeddings')
            if name=='gatr' and rec['checkpoint_sha256']!=config['checkpoint_sha256']: raise ValueError('Wrong GATr checkpoint')
            encoder_receipts[name]=rec
        a=dict(np.load(folder/'observations.npz'));x=np.load(folder/'positions.npy',mmap_mode='r')
        replacements=[];near=[];quantiles=[];moments=[];errors=[]
        for lo,hi in zip(a['offsets'][:-1],a['offsets'][1:],strict=True):
            p,r,q,m,e=radial_control(x[lo:hi],model.scale,config['reference_quantiles'])
            replacements.append(p);near.append(r);quantiles.append(q);moments.append(m);errors.append(e)
        zrad,_=encode_rows(model,replacements,[0]*len(replacements),config['batch_size'])
        originals=[x[lo:hi] for lo,hi in zip(a['offsets'][:16],a['offsets'][1:17],strict=True)]
        check,_=encode_rows(model,originals,a['center_indices'][:16],config['batch_size'])
        gatr=np.load(folder/'gatr.npy');np.testing.assert_allclose(check,gatr[:16],atol=2e-6,rtol=2e-5)
        radial=np.concatenate((a['geometry'][:,:32],near,quantiles,moments,a['order'][:,6:8]),axis=1)
        nf,nc=len(a['frames']),len(a['centers'])
        if not np.array_equal(a['frames'],np.arange(nf)): raise ValueError('Future outcomes require every frame')
        context=np.column_stack((np.full(len(radial),source['temperature_K']),np.repeat(a['times_ps'],nc)))
        np.savez(dest/'observations.npz',radial=radial,radii80=np.array(near),radial_quantiles=np.array(quantiles),
            context=context,gatr=gatr,radial_gatr=zrad,mace=np.load(folder/'mace.npy'),tda=a['tda'],soap=a['soap'],
            bond=a['bond_order'],angular=a['geometry'][:,64:80],labels=a['labels'],order=a['order'],
            frames=a['frames'],centers=a['centers'],times_ps=a['times_ps'])
        save_json(dest/'complete.json',dict(source=source['id'],rows=len(gatr),sha256=file_hash(dest/'observations.npz'),
            radius_preservation_max_A=max(errors),native_z_max_abs=float(np.max(np.abs(check-gatr[:16]))),
            original_encoder_receipts=encoder_receipts,parent_observation_sha256=receipt['hashes']['observations.npz'],
            seconds=time.monotonic()-started))
        print(f'Source {source["id"]}: radial control {len(gatr)} rows, {time.monotonic()-started:.1f}s',flush=True)


def future_labels(labels,cadence,config):
    """Frame x center labels -> prospective eligibility and first sustained onsets."""
    nf,nc=labels.shape
    crystal=np.isin(labels.T,[1,2,3])
    onset=first_sustained_onset(crystal,config['confirmation_frames'])
    horizons=np.asarray(config['future_horizons_ps'])/cadence
    if not np.array_equal(horizons,horizons.astype(int)): raise ValueError('Horizons must be exact frame multiples')
    horizons=horizons.astype(int)
    anchor=np.arange(config['negative_history_frames']-1,nf-int(horizons.max())-config['confirmation_frames']+1)
    risk=risk_windows(crystal,onset,anchor,config['negative_history_frames']).T
    eligible=np.zeros((nf,nc),bool);eligible[anchor]=risk
    delay=onset[None,:]-np.arange(nf)[:,None]
    y=np.stack([(delay>0)&(delay<=h) for h in horizons],axis=-1)
    return eligible,y,onset


def load(config):
    root=Path(config['output'])/'technical'
    parent=json.loads((Path(config['parent_audit'])/'technical/plan.json').read_text())
    parts=[];source_list=[]
    for source in parent['sources']:
        if source['split']!='test':continue
        folder=root/'sources'/str(source['id']);rec=json.loads((folder/'complete.json').read_text())
        if file_hash(folder/'observations.npz')!=rec['sha256']:raise ValueError('Conditional extraction changed')
        a=dict(np.load(folder/'observations.npz'));nf,nc=len(a['frames']),len(a['centers'])
        eligible,future,onset=future_labels(a['labels'].reshape(nf,nc),parent['cadence_ps'],config)
        a.update(source=np.full(nf*nc,source['id']),temperature=np.full(nf*nc,source['temperature_K']),
            frame=np.repeat(a['frames'],nc),atom=np.tile(a['centers'],nf),future_eligible=eligible.ravel(),
            future=future.reshape(nf*nc,-1),onset_frame=np.tile(onset,nf))
        parts.append(a);source_list.append(source)
    keys=['radial','radii80','radial_quantiles','context','gatr','radial_gatr','mace','tda','soap','bond','angular','labels','order',
          'source','temperature','frame','atom','future_eligible','future','onset_frame']
    return {k:np.concatenate([a[k] for a in parts]) for k in keys},source_list,parent
