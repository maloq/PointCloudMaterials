"""Explicit native producers. Invoked as a file in each pinned producer checkout.

Only this inference driver is shared: imports named src resolve to the declared
training producer. No state-dict adaptation or implicit architecture fallback.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--record',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--static-only',action='store_true')
    args=p.parse_args(); task=json.loads(Path(args.record).read_text())
    sys.path[:] = [task['producer']] + [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    for path,expected in task['producer_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError(f'Producer changed: {path}')
    checkpoint=Path(task['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=task['checkpoint_sha256']:raise ValueError(f'Checkpoint changed: {checkpoint}')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    kind=task['kind']; precision=task['precision']; names=['encoder']
    if kind=='geometry':
        from src.research.structural_state.model import GeometryEncoder,GraphBank
        from src.research.structural_state.data import graph_arrays
        model=GeometryEncoder(**saved['encoder_config'])
        model.load_state_dict({k.removeprefix('encoder.'):v for k,v in saved['model'].items() if k.startswith('encoder.')},strict=True)
    elif kind=='neighborhood':
        from src.training_methods.neighborhood_jepa.regularization.model import Encoder
        model=Encoder(saved['manifest']['config']['encoder_channels'],saved['spec']['export_norm'])
        model.load_state_dict({k.removeprefix('encoder.'):v for k,v in saved['model'].items() if k.startswith('encoder.')},strict=True)
    elif kind=='shared':
        from src.models.encoders.structural import StructuralGATr,StructuralMACE
        model=StructuralGATr() if task['architecture']=='gatr' else StructuralMACE(backend='cueq')
        model.load_state_dict({k.removeprefix('encoder.'):v for k,v in saved['model'].items() if k.startswith('encoder.')},strict=True)
        from gatr.utils.einsum import enable_cached_einsum
        import torch.backends.opt_einsum
        enable_cached_einsum(False);torch.backends.opt_einsum.enabled=False
    elif kind=='geoframe':
        from src.research.geoframe_continuity.analysis import load_model
        model=load_model(checkpoint); names=['encoder','projector']
    else:raise ValueError(kind)
    model=model.cuda().eval().requires_grad_(False)
    if kind in ('shared','neighborhood'):
        from src.data.structural_pretraining.batches import collate,move
        from src.data.structural_pretraining.support import support_weights,REFERENCE_RADIUS
        from src.data.structural_pretraining.prepare import ELEMENTS

    compiled = False

    def samples(patches,material,scale):
        factor=REFERENCE_RADIUS/scale; result=[]
        for physical in patches:
            x=physical*factor; keep=np.linalg.norm(x,axis=-1)<8.
            x=x[keep]; center=np.flatnonzero((x==0).all(1))
            if len(center)!=1:raise ValueError('Missing/duplicate exact center')
            pairs=cKDTree(x).query_pairs(5.,output_type='ndarray')
            result.append(dict(positions=x[None],weights=support_weights(x)[None],center=int(center[0]),
                times=np.array([0.],np.float32),species=list(ELEMENTS).index(material),log_scale=np.log(scale/REFERENCE_RADIUS),
                physical=np.zeros(85,np.float32),tda=np.zeros(144,np.float32),tda_valid=False,
                edges=np.concatenate((pairs,pairs[:,::-1]),axis=0).T))
        return result

    @torch.no_grad()
    def encode(patches,material,scale):
        nonlocal compiled
        if kind=='geoframe':
            x=torch.as_tensor(np.stack(patches)/scale,device='cuda',dtype=torch.float32)
            f=model.encoder.forward_features(x)
            return {'encoder':f.float().cpu().numpy(),'projector':model.vicreg.project_features(f).float().cpu().numpy()}
        if kind=='geometry':
            graph=GraphBank(graph_arrays(patches,model.cutoff),model,'cuda')
            z=model(graph.batch(np.arange(len(patches))))
        else:
            batch=move(collate(samples(patches,material,scale),task['architecture']),'cuda')
            if kind=='shared' and task['architecture']=='mace' and not compiled:
                from src.training_methods.shared_pretraining.compilation import compile_encoder
                compile_encoder(model,batch,precision);compiled=True
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                z=model(batch)
            if kind=='neighborhood':z=z[:,:128]
        if not torch.isfinite(z).all():raise FloatingPointError('Nonfinite native embeddings')
        return {'encoder':z.float().cpu().numpy()}

    def extract(patches,material,scale):
        values={n:[] for n in names}; size=task['batch_size']
        for a in range(0,len(patches),size):
            out=encode(patches[a:a+size],material,scale)
            for n,v in out.items():values[n].append(v)
        return {n:np.concatenate(v) for n,v in values.items()}

    inputs=Path(task['inputs']); manifest=json.loads((inputs/'manifest.json').read_text())
    destination=Path(task['destination']);destination.mkdir(parents=True,exist_ok=True)
    timings={}; checks={}
    for record in manifest['frames']:
        if record['material'] not in task['materials']:continue
        i=record['frame_index']; start=time.monotonic(); a=np.load(inputs/f'frame-{i:02d}.npz')
        material=record['material']; scale=task['scales'][material]
        if task['support']=='nearest80': patches=list(a['nearest80'])
        else:
            radius=8. if kind=='geometry' else 8*scale/9.192189
            if radius>10:raise ValueError('Cached context insufficient')
            patches=[]
            for first,last,center in zip(a['offsets'][:-1],a['offsets'][1:],a['centers']):
                x=a['positions'][first:last]; keep=np.linalg.norm(x,axis=-1)<radius
                if kind=='geometry':
                    order=np.r_[center,np.flatnonzero(keep & (np.arange(len(x))!=center))];x=x[order]
                else:x=x[keep]
                patches.append(x)
        if args.smoke:patches=patches[:task['batch_size']]
        out=extract(patches,material,scale)
        m=min(task['batch_size'],len(patches)); repeat=encode(patches[:m],material,scale)
        # Record native reduction noise; keep the same batch for perturbations.
        checks[str(i)]={n:dict(repeat_max_abs=float(np.max(abs(out[n][:m]-repeat[n])))) for n in names}
        count=0 if args.static_only else min(256,len(patches)); control={} if args.static_only else extract(patches[:count],material,scale)
        rng=np.random.default_rng(20260923+i)
        noise=[rng.normal(size=x.shape).astype(np.float32) for x in patches[:count]]
        # The tracked center is fixed. Membership in the original candidate list
        # is fixed; edges and tapered support are recomputed after displacement.
        for x,epsilon in zip(patches[:count],noise):epsilon[(x==0).all(1)]=0
        extra={}
        for n in names:
            if not args.static_only:extra[n+'_control']=control[n]
        for amplitude in (() if args.static_only else (1e-4,.01,.1)):
            perturbed=extract([x+epsilon*amplitude for x,epsilon in zip(patches[:count],noise)],material,scale)
            for n in names:extra[n+'_'+str(amplitude)]=perturbed[n]
        if args.smoke:
            # Explicit singleton/reorder diagnostics; BF16 batch effects remain
            # visible and are not silently counted as structural sensitivity.
            one=encode(patches[:1],material,scale)
            reverse=encode(patches[:m][::-1],material,scale)
            for n in names:checks[str(i)][n].update(singleton_max_abs=float(np.max(abs(out[n][:1]-one[n]))),reorder_max_abs=float(np.max(abs(out[n][:m]-reverse[n][::-1]))))
        np.savez(destination/f'frame-{i:02d}.npz',**out,**extra)
        timings[str(i)]=time.monotonic()-start
        print(json.dumps(dict(frame=i,rows=len(patches),seconds=timings[str(i)],checks=checks[str(i)])),flush=True)
        if args.smoke:break
    if not args.smoke and not args.static_only:
        start=time.monotonic();a=np.load(manifest['future_graphs']);patches=[]
        scale=task['scales']['Al'];radius=8 if kind=='geometry' else 8*scale/9.192189
        if task['support']=='radius' and radius>8.+1e-6:raise ValueError('Future cache lacks required native halo')
        for first,last in zip(a['offsets'][:-1],a['offsets'][1:]):
            x=a['positions'][first:last]
            if task['support']=='nearest80':x=x[np.argsort(np.square(x).sum(1),kind='stable')[:80]]
            elif kind!='geometry':x=x[np.linalg.norm(x,axis=-1)<radius]
            patches.append(x)
        np.savez(destination/'future.npz',**extract(patches,'Al',scale))
        timings['future']=time.monotonic()-start
    receipt=dict(state='complete',checkpoint_sha256=task['checkpoint_sha256'],timings=timings,checks=checks,
        precision=precision,execution='compiled' if compiled else 'eager',representations=names,smoke=args.smoke,
        task_sha256=hashlib.sha256(Path(args.record).read_bytes()).hexdigest(),
        inputs_sha256=hashlib.sha256((inputs/'manifest.json').read_bytes()).hexdigest(),
        perturbation='tracked center fixed; fixed candidate membership, recomputed support and edges')
    (destination/'extraction.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
