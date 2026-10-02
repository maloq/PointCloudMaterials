"""Local numerical checks and train-only environment-convergence gates."""
import gc
import time
import numpy as np
import torch
from ase import Atoms

from src.experiment_runner.metric_docs import write_metric_rows
from src.research.response_performance.oracle import force_and_tangent
from .common import FAMILY,root,read,sha,save,write_json
from .data import prepare
from .oracle import Teacher,make


def error(a,b):
    a,b=a.double(),b.double();d=a-b
    return dict(relative=float(d.norm()/b.norm().clamp_min(1e-12)),
        rms=float(d.square().mean().sqrt()),maximum_absolute=float(d.abs().max()))


def run(c):
    out=root(c)/'technical/gates';out.mkdir(parents=True,exist_ok=True)
    states=prepare(c);binding=read(root(c)/'technical/binding.json')
    if (out/'complete.json').exists():
        receipt=read(out/'complete.json')
        if receipt['identity']!=binding['identity']:raise ValueError('Changed gate identity')
        return receipt
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    from .model import Predictor,initialize,responses
    torch.set_default_dtype(torch.float32);torch.manual_seed(c['seed'])
    q=torch.stack([s['q'][:80] for s in states[:8]]).float().cuda()
    v=torch.stack([s['basis'] for s in states[:8]]).float().cuda()
    model=Predictor(c).cuda();initialize(model,q,4)
    h=responses(model,q[:2],v[:2],create_graph=True)
    (h-.5*h.detach()).square().mean().backward()
    gradient=float(torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),10,error_if_nonfinite=True))
    if gradient<=0:raise ValueError('Response loss does not train the local encoder')
    numerical=[]
    for eps in (.03,.01):
        with torch.no_grad():fd=(model(q[:2]+eps*v[:2,...,0])-model(q[:2]-eps*v[:2,...,0]))/(2*eps)
        numerical.append(dict(epsilon=eps,**error(fd,h[...,0].detach())))
    if min(r['relative'] for r in numerical)>.02:raise ValueError(f'Student JVP mismatch: {numerical}')
    with torch.no_grad():
        reference=model(q[:2]);perm=torch.cat((torch.zeros(1,dtype=torch.long,device='cuda'),torch.randperm(79,device='cuda')+1))
        invariance=dict(translation=error(model(q[:2]+q.new_tensor([.2,.4,.3])),reference),
            neighbor_permutation=error(model(q[:2,perm]),reference))
    if any(v['maximum_absolute']>2e-5 for v in invariance.values()):raise ValueError(invariance)
    write_json(out/'student.json',dict(jvp_fd=numerical,encoder_gradient=gradient,invariance=invariance))
    del model,h;gc.collect();torch.cuda.empty_cache()
    teacher=Teacher(c['oracle']);state=states[c['halo']['gate_parent_indices'][0]]
    oracle,q,v=make(c,state,c['halo']['candidates_A'][0],teacher)
    # Independent ASE neighbor producer checks GPU skin graph, forces and HVPs.
    atoms=Atoms('Al'+str(len(q)//3),positions=q.reshape(-1,3).cpu().numpy(),pbc=False)
    template=teacher.calculator._atoms_to_batch(atoms).to_dict()
    def ase_energy(x):
        return teacher.model(dict(template,positions=x.reshape(-1,3)),training=True,compute_force=False)['energy'].sum()
    f,h=force_and_tangent(oracle.potential,q,v);af,ah=force_and_tangent(ase_energy,q,v)
    graph=dict(force=error(f,af),hvp=error(h,ah))
    if max(x['relative'] for x in graph.values())>.001:raise ValueError(f'GPU/ASE graph mismatch: {graph}')
    write_json(out/'graph.json',graph)
    del template,af,ah,f,h
    # A physical100fs AD/FD check with the same full-cell random stream.
    seed=c['branch_seed_base']-100
    path=out/'numerical32.pt'
    if path.exists():a=torch.load(path,weights_only=False)
    else:
        start=time.monotonic();ad=oracle.query(q,v,[seed]);no=oracle.query(q,q.new_empty(len(q),0),[seed])
        if not torch.allclose(ad.values,no.values,atol=1e-6,rtol=1e-5):raise ValueError('Response execution changed values')
        fd_rows=[]
        for eps in c['halo']['fd_epsilon_A']:
            plus=oracle.query(q+eps*v[:,0],q.new_empty(len(q),0),[seed])
            minus=oracle.query(q-eps*v[:,0],q.new_empty(len(q),0),[seed])
            fd_rows.append(dict(epsilon_A=eps,**error((plus.values-minus.values)/(2*eps),ad.responses[:,:,0])))
        a=dict(values=ad.values.cpu(),responses=ad.responses.cpu(),fd=fd_rows,seconds=time.monotonic()-start)
        save(path,a)
    if min(r['relative'] for r in a['fd'])>c['halo']['fd_relative_tolerance']:
        raise ValueError(f'Physical AD/FD gate failed: {a["fd"]}')
    del oracle,q,v;gc.collect();torch.cuda.empty_cache()
    # The Al256 float32 result is not assumed to apply to these new local targets.
    teacher64=Teacher(c['oracle'],torch.float64)
    oracle64,q64,v64=make(c,state,c['halo']['candidates_A'][0],teacher64)
    p64=out/'numerical64.pt'
    if p64.exists():b=torch.load(p64,weights_only=False)
    else:
        ad64=oracle64.query(q64,v64,[seed]);b=dict(values=ad64.values.cpu(),responses=ad64.responses.cpu());save(p64,b)
    precision={k:error(a[k],b[k]) for k in ('values','responses')}
    if any(e['relative']>.01 or e['maximum_absolute']>1e-4 for e in precision.values()):
        raise ValueError(f'Local float32 budget exceeded: {precision}')
    write_json(out/'numerical.json',dict(fd=a['fd'],precision=precision))
    del teacher64,oracle64,q64,v64;gc.collect();torch.cuda.empty_cache();torch.set_default_dtype(torch.float32)
    radii=c['halo']['candidates_A']+[c['halo']['reference_A']];outputs={};timing=[]
    for index in c['halo']['gate_parent_indices']:
        state=states[index]
        if state['role']!='train':raise ValueError('Environment selection may use training ancestors only')
        for radius in radii:
            path=out/f'parent{index}-radius{radius}.pt'
            if path.exists():payload=torch.load(path,weights_only=False)
            else:
                oracle,q,v=make(c,state,radius,teacher);torch.cuda.reset_peak_memory_stats()
                seeds=[c['branch_seed_base']-1000-index*10-j for j in range(c['halo']['gate_shots'])]
                print(dict(stage='halo-gate',parent=index,radius_A=radius,atoms=len(q)//3),flush=True)
                start=time.monotonic();result=oracle.query(q,v,seeds);torch.cuda.synchronize()
                payload=dict(values=result.values.cpu(),responses=result.responses.cpu(),seconds=time.monotonic()-start,
                    atoms=len(q)//3,peak_GiB=torch.cuda.max_memory_allocated()/2**30,seeds=seeds)
                save(path,payload);del oracle,q,v,result;gc.collect();torch.cuda.empty_cache()
            outputs[index,radius]=payload
            timing.append(dict(parent=index,radius_A=radius,seconds=payload['seconds'],atoms=payload['atoms'],peak_GiB=payload['peak_GiB']))
    rows=[];accepted=[]
    for radius in c['halo']['candidates_A']:
        passed=True
        for index in c['halo']['gate_parent_indices']:
            for reference_radius in [r for r in radii if r>radius]:
                a,b=outputs[index,radius],outputs[index,reference_radius]
                for horizon,lo,hi in [('20fs',0,128),('100fs_joint',128,256)]:
                    ve=error(a['values'][:,lo:hi],b['values'][:,lo:hi])
                    he=error(a['responses'][:,lo:hi],b['responses'][:,lo:hi])
                    ok=ve['rms']<=c['halo']['value_rms_tolerance'] and he['relative']<=c['halo']['response_relative_tolerance']
                    passed &= ok
                    rows.append(dict(parent=index,radius_A=radius,reference_A=reference_radius,horizon=horizon,
                        value_rms=ve['rms'],response_relative=he['relative'],passed=ok))
        if passed:accepted.append(radius)
    table=root(c)/'analyses/environment-gate-v1'
    write_metric_rows(rows,table,family=FAMILY,name='environment-convergence')
    write_metric_rows(timing,table,family=FAMILY,name='timing')
    if not accepted:raise ValueError('No environment passed every larger-radius comparison; training remains blocked by scientific gate')
    result=dict(state='complete',identity=binding['identity'],selected_radius_A=min(accepted),
        reference_radius_A=c['halo']['reference_A'],gate_parents=c['halo']['gate_parent_indices'],
        precision=precision,fd=read(out/'numerical.json')['fd'],timings=timing,
        qualification='agreement with nested moving open environments, not verified exact full-periodic dynamics')
    write_json(out/'complete.json',result);print(result,flush=True)
    return result
