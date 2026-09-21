"""Separate G0 audit and tiny real-data fixed-corruption overfit."""
import copy
import json
import subprocess
import sys
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from e3nn import o3
from src.experiment_runner.metric_docs import write_metric_table
from .runtime import load_data,identity,implementation_hashes
from .model import BCR
from .data import pack,corrupt
from .objective import per_environment


def verify(config,data,output,device):
    tech=Path(output)/'technical';tech.mkdir(parents=True,exist_ok=True)
    with (tech/'correctness-tests.log').open('w') as log:
        subprocess.run([sys.executable,'-m','pytest','tests/test_bcr.py','-q'],stdout=log,stderr=subprocess.STDOUT,check=True)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1);torch.manual_seed(27)
    patches,manifest=load_data(data);cfg=copy.deepcopy(config['training'])
    cfg['encoder'].update(d0=manifest['d0'],n_ref=manifest['n_ref'],radius=manifest['radius_A'])
    model=BCR(cfg).to(device);selected=[p for p,r in zip(patches,manifest['records']) if r['split']=='train'][:2]
    batch=pack(selected,device);noisy,eps,sigma,_=corrupt(batch,manifest['noise_levels'],manifest['d0'],torch.Generator().manual_seed(16))
    prediction,z=model(batch,noisy,sigma);R=o3.rand_matrix().to(device)
    rotated=dict(batch,positions=batch['positions']@R.T);yn=dict(noisy,positions=noisy['positions']@R.T)
    rp,rz=model(rotated,yn,sigma)
    torch.testing.assert_close(z,rz,atol=1e-6,rtol=1e-5);torch.testing.assert_close(rp,prediction@R.T,atol=1e-6,rtol=1e-5)
    repeated=model.encode(batch)
    torch.testing.assert_close(repeated,z,atol=1e-6,rtol=1e-5)
    loss=per_environment(prediction,eps,batch,manifest['radius_A']).mean();initial=float(loss.detach());loss.backward()
    grad=float(model.encoder.readout[-1].weight.grad.norm())
    if not np.isfinite(grad) or grad<=0:raise AssertionError('No finite reconstruction gradient into encoder')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=1e-5);begin=time.monotonic();curve=[]
    for step in range(config.get('overfit_updates',200)):
        optimizer.zero_grad();pred,_=model(batch,noisy,sigma);loss=per_environment(pred,eps,batch,manifest['radius_A']).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Overfit loss nonfinite')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();curve.append(float(loss.detach()))
    with torch.no_grad():final=float(per_environment(model(batch,noisy,sigma)[0],eps,batch,manifest['radius_A']).mean())
    passed=final<.8*initial
    receipt=dict(G0_pass=True,implementation_hashes=implementation_hashes(),correctness_suite='tests/test_bcr.py',
        correctness_suite_sha256=hashlib.sha256(Path('tests/test_bcr.py').read_bytes()).hexdigest(),
        decoder_contract=identity(cfg.get('decoder',{})),
        model_sha256=hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest(),real_overfit_pass=passed,model_contract=identity(cfg['encoder']),data_identity=manifest['identity'],
        encoder_gradient_norm=grad,initial_nmse=initial,final_nmse=final,updates=len(curve),seconds=time.monotonic()-begin,
        repeated_export_max_abs=float((repeated-z).detach().abs().max()),
        rotation_invariant_max_abs=float((z-rz).detach().abs().max()),rotation_equivariant_max_abs=float((rp-prediction@R.T).detach().abs().max()),
        note='Two real patches, fixed artificial corruption; optimization test only, not held-out quality')
    tech=Path(output)/'technical';tech.mkdir(parents=True,exist_ok=True)
    (tech/'gate.json').write_text(json.dumps(receipt,indent=2)+'\n');(tech/'overfit.json').write_text(json.dumps(curve)+'\n')
    torch.save(dict(model=model.state_dict(),config=cfg,data_identity=manifest['identity']),tech/'overfit.pt')
    write_metric_table(receipt,output,family='bcr',name='verification')
    print(json.dumps(receipt,indent=2),flush=True)
    if not passed:raise AssertionError('Real-data overfit did not lower fixed-noise loss by 20%')
