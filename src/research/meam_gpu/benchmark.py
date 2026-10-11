"""Physical gates and complete response timings on the same allocated GPU."""
import argparse
import json
import time
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read,write_json
from src.research.response_meam.benchmark import inputs,query_data,features,error,make
from src.research.response_meam.potential import Lee2003Al
from src.research.response_performance.oracle import force_and_tangent
from src.research.meam_md.potential import PeriodicAl
from .potential import FastAl
from .oracle import FastOracle


def timed(call,repeats):
    call();torch.cuda.synchronize();samples=[]
    for _ in range(repeats):
        torch.cuda.synchronize();begin=time.perf_counter();result=call();torch.cuda.synchronize()
        samples.append(time.perf_counter()-begin)
    return samples,result


def periodic(c):
    out=resolve_path(c['output'])/'analyses/numerical-v1/technical';out.mkdir(parents=True,exist_ok=True)
    with np.load(resolve_path(c['ancestor'])/'initial.npz') as data:x=torch.tensor(data['q'],device='cuda');box=data['box']
    files={k:resolve_path(v) for k,v in c['meam_files'].items()}
    ref=PeriodicAl(files,len(x),box,dtype=torch.float64,compile_force=True);e0,f0=ref.force(x)
    timings,_=timed(lambda:ref.force(x),30)
    rows=[dict(engine='old-compiled-f64',seconds=float(np.median(timings)),samples=timings,energy_error=0.,force_rms=0.,pressure_error_bar=0.)]
    virial64=None
    for dtype in (torch.float64,torch.float32):
        m=FastAl(files,len(x),box=box,dtype=dtype,compile_tangents=False)
        timings,(e,f)=timed(lambda:m.force(x),30)
        virial=float(m.virial_trace(x))
        if virial64 is None:virial64=virial
        row=dict(engine='cuda-'+str(dtype).split('.')[-1],seconds=float(np.median(timings)),samples=timings,
            energy_error=abs(float(e-e0))/len(x),force_rms=error(f,f0)['rms'],
            pressure_error_bar=abs(virial-virial64)/3/np.prod(box)*1.602176634e6,
            configurational_pressure_bar=virial/3/np.prod(box)*1.602176634e6)
        tolerance='double' if dtype==torch.float64 else 'mixed'
        if row['energy_error']>c['gates'][tolerance+'_energy_eV_per_atom'] or row['force_rms']>c['gates'][tolerance+'_force_rms_eV_A']:
            raise ValueError(f'Periodic precision gate failed: {row}')
        if row['pressure_error_bar']>c['gates']['mixed_virial_pressure_bar']:raise ValueError(f'Virial gate failed: {row}')
        rows.append(row);print(dict(stage='periodic',**{k:v for k,v in row.items() if k!='samples'}),flush=True)
    write_json(out/'periodic.json',rows)


def response(c):
    b=read(c['response_reference_config']);payload,norm,original=inputs(b)
    out=resolve_path(c['output'])/'analyses/response-v2/technical';out.mkdir(parents=True,exist_ok=True)
    files={k:resolve_path(v) for k,v in c['meam_files'].items()};validation=[];timings=[]
    for state in payload['states']:
        q,basis,atom_rows,seeds,_=query_data(b,state,original)
        x=q.cuda().flatten()[None];dq=basis.cuda().reshape(1,-1,2)
        old=Lee2003Al(files,len(q));f0,h0=force_and_tangent(old,x,dq)
        for batch in b['batch_sizes']:
            initial=x.repeat(batch,1);directions=dq.repeat(batch,1,1);shot_seeds=seeds
            def all_shots(oracle):
                values=[];responses=[]
                for offset in range(0,len(seeds),batch):
                    v,r=oracle.simulate(initial,directions,seeds[offset:offset+batch]);values.append(v);responses.append(r)
                return torch.cat(values),torch.cat(responses)
            reference=make(b,state,q,atom_rows,payload['oracle'],norm,'meam',torch.float64)
            samples,(v0,r0)=timed(lambda:all_shots(reference),b['repeats'])
            timings.append(dict(parent=state['index'],atoms=len(q),batch=batch,engine='old-meam-f64',seconds=float(np.median(samples)),samples=samples,shots=len(seeds)))
            for dtype in (torch.float64,torch.float32):
                name='cuda-'+str(dtype).split('.')[-1]
                model=FastAl(files,len(q),dtype=dtype)
                f,h=model.force_tangent(x,dq);he=error(h,h0)['relative'];fe=error(f,f0)['rms']
                oracle=FastOracle(model,features(payload['oracle'],norm,torch.float64),payload['oracle'],atom_rows.cuda(),state['original_atoms'])
                samples,(value,response)=timed(lambda:all_shots(oracle),b['repeats'])
                ve=error(value,v0)['rms'];re=error(response,r0)['relative']
                row=dict(parent=state['index'],atoms=len(q),batch=batch,engine=name,force_rms=fe,hvp_relative=he,value_rms=ve,response_relative=re,
                         features=value.shape[1],directions=response.shape[-1])
                if he>c['gates']['hvp_relative'] or ve>c['gates']['value_rms'] or re>c['gates']['response_relative']:
                    raise ValueError(f'Analytic response gate failed: {row}')
                validation.append(row);timings.append(dict(parent=state['index'],atoms=len(q),batch=batch,engine=name,seconds=float(np.median(samples)),samples=samples,shots=len(seeds)))
                print(dict(stage='response',**row,seconds_per_shot=float(np.median(samples))/len(seeds)),flush=True)
                torch.save(dict(values=value.cpu(),responses=response.cpu(),reference_values=v0.cpu(),reference_responses=r0.cpu(),seeds=shot_seeds,metrics=row),out/f'parent{state["index"]}-batch{batch}-{name}.pt')
                write_json(out/'validation.json',validation);write_json(out/'timings.json',timings)
    write_json(out/'complete.json',dict(passed=True,parents=b['parents'],batches=b['batch_sizes'],gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        precision='float64 states and observables; declared force arithmetic',noise='Existing full-cell-order streams',encoder_inputs=None,predictor_inputs=None))


def nve(c):
    archive=resolve_path(c['simulation_archive']);root=archive/c['nve_name']
    table=np.genfromtxt(root/'thermodynamics.csv',delimiter=',',names=True)
    reference=resolve_path(c['ancestor']).parent/'cpu-nve-control'
    a=np.load(root/'checkpoints/state-000001000.npz')['q'];b=np.load(reference/'checkpoints/state-000001000.npz')['q']
    row=dict(nve_energy_excursion_eV_per_atom=float(np.max(abs(table['total_eV_per_atom']-table['total_eV_per_atom'][0]))),
             q_rms_1ps_A=float(np.sqrt(np.mean((a-b)**2))))
    row['passed']=row['nve_energy_excursion_eV_per_atom']<c['gates']['nve_energy_excursion_eV_per_atom'] and row['q_rms_1ps_A']<1e-4
    if not row['passed']:raise ValueError(f'NVE gate failed: {row}')
    write_json(resolve_path(c['output'])/'analyses/numerical-v1/technical/nve.json',row);print(row,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['periodic','response','nve']);parser.add_argument('--config',required=True)
    a=parser.parse_args();torch.set_num_threads(1);globals()[a.stage](read(a.config))


if __name__=='__main__':main()
