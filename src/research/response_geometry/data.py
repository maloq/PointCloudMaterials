"""Eight unrelaxed inputs per train ancestor; original selection/test queries."""
import hashlib
import numpy as np
import torch
from src.data.fixed_cohort.dataset import read_release
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.project_runtime.paths import resolve_path,dataset_path
from .common import root,read,sha,save,write_json,legacy
from .features import observed


def prepare(c):
    out=root(c)/'technical';out.mkdir(parents=True,exist_ok=True)
    receipt=out/'parents.json'
    if receipt.exists():
        r=read(receipt)
        if sha(out/'parents.pt')!=r['sha256']:raise ValueError('Changed expanded parents')
        return torch.load(out/'parents.pt',map_location='cpu',weights_only=False)
    oldroot=resolve_path(c['reference'])/'technical';old=read(oldroot/'parents.json')
    if sha(oldroot/'parents.pt')!=old['sha256']:raise ValueError('Changed original parents')
    states=torch.load(oldroot/'parents.pt',map_location='cpu',weights_only=False)
    _,plan=read_release(c['fixed_release'])
    if plan['identity']!=c['fixed_identity']:raise ValueError('Changed source roles')
    for state in states:state['original_index']=state['index']
    for rank,item in enumerate(s for s in plan['sources']if s['role']=='train'):
        raw=ShootingBinaryTrajectory.load(dataset_path(item['dataset'])/item['relative_trajectory_path'])
        if sha(raw.root/'manifest.json')!=item['manifest_sha256']:raise ValueError('Changed source manifest')
        ids=np.asarray(raw.atom_ids)
        if not np.all(raw.atom_types==1):raise ValueError('Non-Al parent')
        for extra in range(1,c['train_patches_per_source']):
            frame=c['sampling_frames'][(rank+extra)%len(c['sampling_frames'])]
            atom=item['center_atom_ids'][(rank*17+extra*7)%64]
            row=int(np.searchsorted(ids,atom));xyz=np.asarray(raw.positions[frame],np.float64)
            box=np.asarray(raw.box_high[frame]-raw.box_low[frame],np.float64)
            if ids[row]!=atom or box.min()/2<=36:raise ValueError('Invalid parent atom or box')
            x=xyz-xyz[row];x-=box*np.round(x/box);distance=np.linalg.norm(x,axis=1)
            order=np.lexsort((ids,distance));selected=order[distance[order]<36]
            if selected[0]!=row or distance[order[79]]>=8 or distance[order[79]]<=5.5:
                raise ValueError('Nearest80 must cover feature support and fit encoder support')
            seed=np.random.SeedSequence([c['seed'],item['id'],frame,int(atom)])
            rng=np.random.default_rng(seed);v=rng.normal(size=(79,3,2));v-=v.mean(0,keepdims=True)
            v=np.linalg.qr(v.reshape(-1,2))[0].reshape(79,3,2)
            basis=torch.zeros(80,3,2,dtype=torch.float64);basis[1:]=torch.from_numpy(v)
            states.append(dict(index=len(states),original_index=None,source=item['id'],role='train',lineage=item['lineage'],
                frame=frame,center_atom_id=int(atom),original_atoms=len(xyz),q=torch.from_numpy(x[selected].copy()),
                atom_rows=torch.from_numpy(selected.copy()),atom_ids=torch.from_numpy(ids[selected].copy()),
                basis=basis,box=torch.from_numpy(box),source_manifest_sha256=item['manifest_sha256'],
                source_frame_sha256=hashlib.sha256(np.ascontiguousarray(raw.positions[frame]).tobytes()).hexdigest(),
                parent_potential='Lee2003 MEAM',parent_storage_dtype=str(raw.positions.dtype),
                time_ps=float(raw.timesteps[frame]*item['timestep_fs']/1000)))
        print('Prepared source',item['id'],'parents',len(states),flush=True)
    if [sum(s['role']==r for s in states)for r in ['train','selection','test']]!=[720,15,30]:
        raise ValueError('Expanded sample contract changed')
    keys=[(s['source'],s['frame'],s['center_atom_id'])for s in states]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate sampled geometry identity')
    training=torch.stack([s['q'][:80]for s in states if s['role']=='train'])
    if (training[:,79].norm(dim=-1)<=5.5).any():raise ValueError('Feature support extends past nearest80')
    torch.set_num_threads(2);value=torch.cat([observed(x.flatten(1))for x in training.split(64)])
    normalizer=dict(mean=value.mean(0),scale=value.std(0,unbiased=False).clamp_min(.01),
        fit_role='train initial observations only; no reconstruction target',floor=.01)
    save(out/'feature-normalization.pt',normalizer);save(out/'parents.pt',states)
    write_json(receipt,dict(sha256=sha(out/'parents.pt'),normalizer_sha256=sha(out/'feature-normalization.pt'),
        fixed_identity=plan['identity'],parents=[{k:v for k,v in s.items()if not torch.is_tensor(v)}for s in states],
        counts=dict(train=720,selection=15,test=30),selection='fixed center ranks and frame cycle; no outcomes'))
    return states
