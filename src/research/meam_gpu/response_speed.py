"""Same-GPU complete-query timings against the earlier fast float32 MEAM and MACE.

Historical float32 engines retain float32 integration/observables. The new
engine uses float64 integration/observables and float32 force arithmetic.
MACE is a runtime reference with a different potential, not an accuracy target.
"""
import argparse
import os
import time
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read,write_json
from src.research.response_meam.benchmark import inputs,query_data,features,make
from src.research.local_response.oracle import Teacher
from src.experiment_runner.metric_docs import write_metric_rows
from .potential import FastAl
from .oracle import FastOracle


def run(c):
    b=read(c['response_reference_config']);payload,norm,original=inputs(b)
    out=resolve_path(c['output'])/'analyses/fast-response-comparison-v1';(out/'technical').mkdir(parents=True,exist_ok=True)
    write_json(out/'technical/execution.json',dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,host=os.uname().nodename,allocation=os.environ.get('SLURM_JOB_ID'),config=c))
    files={k:resolve_path(v) for k,v in c['meam_files'].items()}
    teacher=Teacher(payload['oracle'],torch.float32);rows=[]
    for state in payload['states']:
        q,basis,atom_rows,seeds,_=query_data(b,state,original)
        for batch in b['batch_sizes']:
            names=['old-meam-f32','new-meam-mixed','mace-f32']
            oracles={'old-meam-f32':make(b,state,q,atom_rows,payload['oracle'],norm,'meam',torch.float32),
                     'mace-f32':make(b,state,q,atom_rows,payload['oracle'],norm,'mace',torch.float32,teacher),
                     'new-meam-mixed':FastOracle(FastAl(files,len(q),dtype=torch.float32),features(payload['oracle'],norm,torch.float64),payload['oracle'],atom_rows.cuda(),state['original_atoms'])}
            def query(name):
                dtype=torch.float64 if name=='new-meam-mixed' else torch.float32
                x=q.cuda().to(dtype).flatten()[None].repeat(batch,1);dq=basis.cuda().to(dtype).reshape(1,-1,2).repeat(batch,1,1)
                for start in range(0,len(seeds),batch):value,response=oracles[name].simulate(x,dq,seeds[start:start+batch])
                return value.shape[1],response.shape[-1]
            for name in names:query(name);torch.cuda.synchronize()
            for repeat in range(b['repeats']):
                for name in names[repeat%3:]+names[:repeat%3]:
                    torch.cuda.synchronize();start=time.perf_counter();features_count,directions=query(name);torch.cuda.synchronize()
                    row=dict(parent=state['index'],atoms=len(q),batch=batch,engine=name,repeat=repeat,shots=len(seeds),
                        seconds_per_shot=(time.perf_counter()-start)/len(seeds),features=features_count,directions=directions)
                    rows.append(row);write_json(out/'technical/timings.json',rows);print(row,flush=True)
    write_metric_rows(rows,out,family='meam_gpu_response_speed',name='timings')
    summary=[]
    for batch in b['batch_sizes']:
        for name in names:
            values=[np.median([r['seconds_per_shot'] for r in rows if r['batch']==batch and r['engine']==name and r['parent']==parent]) for parent in b['parents']]
            summary.append(dict(engine=name,batch=batch,seconds_per_shot=float(np.mean(values)),parents=len(values)))
    write_metric_rows(summary,out,family='meam_gpu_response_speed',name='summary')
    write_json(out/'technical/complete.json',dict(state='complete',gpu=torch.cuda.get_device_name(),torch=torch.__version__,summary=summary))
    print(summary,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    a=parser.parse_args();torch.set_num_threads(1);run(read(a.config))


if __name__=='__main__':main()
