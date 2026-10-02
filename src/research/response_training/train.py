"""Paired native encoder fits with likelihood selection and exact optimizer resume."""
import math
import time

import numpy as np
import torch

from src.experiment_runner.checkpoints import TrainingState
from src.experiment_runner.execution import allocation_deadline
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.experiment_runner.wandb_tracking import online_training
from .common import FAMILY, root, read, write_json, sha, digest, metric_family, simulation_profile
from .data import load, save_pt
from .model import Predictor, initialize, responses


def location(c, arm, seed):
    return root(c)/'analyses'/f'{arm}-seed-{seed}'


@torch.no_grad()
def predict(model, q, batch):
    return torch.cat([model(x) for x in q.split(batch)])


def fit(c, arm, seed):
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    data = load(c)
    gate = read(root(c)/'technical/preflight.json')
    if gate['identity'] != data['identity'] or gate['state'] != 'complete':
        raise ValueError('Student/oracle preflight is missing or changed')
    contract = check_metric_docs(family=metric_family(c))[metric_family(c)]
    identity = digest(dict(data_identity=data['identity'], arm=arm, seed=seed, contract=contract))
    out = location(c,arm,seed); tech=out/'technical'; tech.mkdir(parents=True,exist_ok=True)
    if (tech/'complete.json').exists():
        done = read(tech/'complete.json')
        if done['identity'] != identity or sha(tech/'predictions.npz') != done['prediction_sha256']:
            raise ValueError('Completed fit changed')
        return
    cfg = c['training']; micro=cfg['microbatch']
    records = data['parents']
    ids = {role: np.array([i for i,r in enumerate(records) if r['role']==role]) for role in c['parent_roles']}
    if len(ids['train']) != cfg['batch_size']:
        raise ValueError('This pilot declares a full-parent-batch update')
    q = torch.stack([r['q'] for r in records]).float().cuda()
    basis = torch.stack([r['basis'] for r in records]).float().cuda()
    shots = 32 if arm == 'values32' else 8
    center,scale = data['center'],data['scale']
    # Shared validation always uses 32 independent shots, irrespective of arm.
    target = torch.stack([(r['values'][:shots if r['role']=='train' else 32].mean(0)-center)/scale
                          for r in records]).float().cuda()
    htarget = torch.stack([r['responses'].mean(0)/scale[:,None] for r in records if r['role']=='train']).float().cuda()
    response_scale = data['response_scale'].float().cuda()
    model = Predictor(c, records[0]['box']).float().cuda()
    optimizer = torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=cfg['encoder_lr']),
        dict(params=model.head.parameters(),lr=cfg['head_lr'])],weight_decay=cfg['weight_decay'])
    rng = np.random.default_rng(seed)
    history=[];best=math.inf;best_epoch=0;start=1;elapsed=0.
    if (tech/'last.pt').exists():
        state=TrainingState.read(tech/'last.pt',identity=identity,device='cuda')
        state.restore(model,optimizer,rng,restore_torch_rng=True)
        p=state.payload;history=p['history'];best=p['best'];best_epoch=p['best_epoch'];start=p['epoch']+1;elapsed=p['elapsed']
    else:
        initialize(model,q[ids['train']],micro)
    # Arm-specific oracle time is measured on actual value-only versus AD calls.
    field={'values8':'value8_seconds','values32':'value32_seconds','responses8':'response8_seconds'}[arm]
    profile=simulation_profile(c)
    if profile is not None:
        acquisition=sum(r['cost']['collection_seconds']+r['cost']['screen_seconds']
                        for r in records if r['role'] in ('train','selection'))
        cost_scope='actual shared train/selection bank including responses and audits, identical charge for all arms; not arm-specific counterfactual cost'
    else:
        acquisition=sum(r['cost'][field]+r['cost']['screen_seconds'] for r in records if r['role']=='train')
        acquisition+=sum(r['cost']['value32_seconds']+r['cost']['screen_seconds'] for r in records if r['role']=='selection')
        cost_scope='arm-specific train + shared validation oracle and screens; optimization/selection; common pilot, I/O and final evaluation recorded separately'
    context=dict(identity=identity,config=c,arm=arm,seed=seed,feature_dimensions=256,
        parameters=sum(p.numel() for p in model.parameters()),input_contract=read(root(c)/'technical/prediction-context.json'),
        selector=cfg['selector'],neural_precision='float32',oracle_precision=profile['dtype'] if profile else 'float64',
        acquisition_seconds=acquisition, cost_scope=cost_scope)
    write_json(tech/'identity.json',context)
    deadline=allocation_deadline(reserve_seconds=120)
    with online_training(c['wandb'],run_id='resp-al256-'+identity[:14],name=f'Al256 response | {arm} | {seed}',
            config=context,folder=tech,receipt_path=tech/'wandb.json',job_type='encoder',group=c['protocol'],
            receipt_fields=dict(identity=identity)) as run:
        def score():
            model.eval()
            prediction=predict(model,q[ids['selection']],micro)
            return float(.5*((prediction-target[ids['selection']]).square().mean()+math.log(2*math.pi)))
        if start==1:
            best=score()
            save_pt(tech/'best.pt',dict(identity=identity,model=model.state_dict(),epoch=0,selection_nll=best))
        for epoch in range(start,cfg['epochs']+1):
            if epoch-best_epoch>cfg['patience']:break
            if time.time()>=deadline:raise TimeoutError('Resume from the last complete optimizer update')
            began=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
            order=rng.permutation(ids['train']);value_loss=0.;response_loss=0.
            for begin in range(0,len(order),micro):
                ix=order[begin:begin+micro]
                prediction=model(q[ix]);lv=.5*(prediction-target[ix]).square().mean()
                ld=prediction.new_zeros(())
                if arm=='responses8':
                    derivative=responses(model,q[ix],basis[ix],create_graph=True)
                    # Training IDs are fixed 0:32, hence directly index htarget.
                    ld=.5*((derivative-htarget[ix])/response_scale).square().mean()
                loss=(lv+cfg['response_weight']*ld)*(len(ix)/len(order))
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite {arm} objective, epoch{epoch}')
                loss.backward();value_loss+=float(lv.detach())*len(ix)/len(order);response_loss+=float(ld.detach())*len(ix)/len(order)
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True))
            optimizer.step();selection=score();torch.cuda.synchronize();elapsed+=time.monotonic()-began
            if selection<best:
                best=selection;best_epoch=epoch
                save_pt(tech/'best.pt',dict(identity=identity,model=model.state_dict(),epoch=epoch,selection_nll=best))
            row=dict(epoch=epoch,train_value_nll=value_loss+.5*math.log(2*math.pi),
                train_response_half_mse=response_loss,selection_nll=selection,best_epoch=best_epoch,
                gradient_norm=norm,training_seconds=elapsed)
            history.append(row)
            TrainingState.capture(model,optimizer,rng,identity=identity,capture_torch_rng=True,
                history=history,best=best,best_epoch=best_epoch,epoch=epoch,elapsed=elapsed).save(tech/'last.pt')
            write_json(tech/'progress.json',dict(state='running',arm=arm,seed=seed,**row))
            run.log({k:v for k,v in row.items() if k!='training_seconds'},step=epoch)
            if epoch==1 or epoch%10==0:print(dict(arm=arm,seed=seed,**row),flush=True)
        selected=torch.load(tech/'best.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(selected['model']);model.eval()
        evaluation_start=time.monotonic()
        prediction=predict(model,q,micro).cpu().numpy()
        with torch.no_grad():z=torch.cat([model.encode(x) for x in q.split(micro)]).cpu().numpy()
        deriv=torch.cat([responses(model,q[i:i+micro],basis[i:i+micro]).detach().cpu()
                         for i in range(0,len(q),micro)]).numpy()
        np.savez_compressed(tech/'predictions.npz',normalized_prediction=prediction,
            normalized_response=deriv,z=z,parent=np.array([r['index'] for r in records]))
        write_metric_rows(history,out,family=metric_family(c),name='learning')
        evaluation_seconds=time.monotonic()-evaluation_start
        run.summary.update(dict(selected_epoch=best_epoch,selection_nll=best,acquisition_seconds=acquisition,
            training_seconds=elapsed,acquisition_plus_training_seconds=acquisition+elapsed,
            final_evaluation_seconds=evaluation_seconds))
    done=dict(state='complete',identity=identity,arm=arm,seed=seed,selected_epoch=best_epoch,
        selection_nll=best,acquisition_seconds=acquisition,training_seconds=elapsed,
        acquisition_plus_training_seconds=acquisition+elapsed,final_evaluation_seconds=evaluation_seconds,
        checkpoint_sha256=sha(tech/'best.pt'),prediction_sha256=sha(tech/'predictions.npz'))
    write_json(tech/'complete.json',done);write_json(tech/'progress.json',done)
