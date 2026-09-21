"""Held-out JEPA latent errors and decoded physical errors, each at its actual lag."""
import numpy as np
import torch
from src.data.structural_pretraining.batches import collate,move
from src.training_methods.structural_pretraining.objective import block_errors,PHYSICAL_BLOCKS,TDA_BLOCKS
from ..runtime import encode
from ..regularization.runtime import evaluate as original_evaluate
from ..v2.geometry import scaled_error


def feature_errors(prediction,target,blocks):
    """The repository block reducer takes matrices; retain horizon as a batch axis."""
    return block_errors(prediction.expand_as(target).flatten(0,1),target.flatten(0,1),blocks).mean(-1).reshape(target.shape[:2])


@torch.no_grad()
def evaluate(model,objective,data,config,baselines):
    metrics,arrays=original_evaluate(model,objective,data,config,baselines)
    rows=[];saved={name:[] for name in ('predicted','observed','valid','physical_target','physical_prediction','tda_target','tda_prediction')}
    for start in range(0,len(data.selection),16):
        samples=[data[i] for i in data.selection[start:start+16]]
        slots=[data.plan.slot(t,0) for t in (3,4,5)]
        views=[s['views'][slot] for s in samples for slot in slots]
        packed=[collate(views[i:i+config['microbatch']],'mace') for i in range(0,len(views),config['microbatch'])]
        actual=encode(model,packed,config['precision']).reshape(len(samples),3,248)
        current=torch.tensor(np.concatenate((arrays['invariant'][start:start+len(samples)],
                                             arrays['equivariant'][start:start+len(samples)]),axis=-1),device='cuda')
        target={name:torch.tensor(np.stack([s[name] for s in samples]),device='cuda') for name in
                ('future_physical','future_tda','future_valid','physical','tda','temperature_K')}
        target['temperature_K']=target['temperature_K'].float()
        inv,eq=model.future_embeddings(current,target['temperature_K'])
        groups=torch.zeros(len(samples)*3,dtype=torch.long,device='cuda')
        pp=model.physical(inv.flatten(0,1),groups).reshape(-1,3,85)
        tp=model.tda(inv.flatten(0,1),groups).reshape(-1,3,144)
        p=(target['future_physical']-objective.physical_mean)/objective.physical_std
        t=(target['future_tda']-objective.tda_mean)/objective.tda_std
        observed_p=(target['physical'][:,0,None]-objective.physical_mean)/objective.physical_std
        observed_t=(target['tda'][:,0,None]-objective.tda_mean)/objective.tda_std
        errors=torch.stack(((inv-actual[:,:,:128]).square().mean(-1),
            scaled_error(eq,actual[:,:,128:],model.encoder.geometry_scales).mean(-1),
            (current[:,None,:128]-actual[:,:,:128]).square().mean(-1),
            scaled_error(current[:,None,128:],actual[:,:,128:],model.encoder.geometry_scales).mean(-1),
            feature_errors(pp,p,PHYSICAL_BLOCKS),feature_errors(observed_p,p,PHYSICAL_BLOCKS),
            feature_errors(tp,t,TDA_BLOCKS),feature_errors(observed_t,t,TDA_BLOCKS)),-1)
        rows.append(errors.cpu().numpy())
        for name,value in dict(predicted=torch.cat((inv,eq),-1),observed=actual,valid=target['future_valid'],
                physical_target=p,physical_prediction=pp,tda_target=t,tda_prediction=tp).items():saved[name].append(value.cpu().numpy())
    error=np.concatenate(rows);saved={k:np.concatenate(v) for k,v in saved.items()};source=arrays['sources']
    names=['invariant_mse','equivariant_scaled_mse','invariant_persistence','equivariant_persistence',
           'physical','physical_persistence','tda','tda_persistence']
    metrics['horizons']={}
    for h,ps in enumerate(config['horizons_ps']):
        valid=saved['valid'][:,h]
        average=np.mean([error[valid&(source==s),h].mean(0) for s in np.unique(source[valid])],0)
        row=dict(zip(names,map(float,average)))
        row.update(windows=int(valid.sum()),sources=len(np.unique(source[valid])))
        for kind in ('invariant','equivariant'):
            numerator=row['invariant_mse' if kind=='invariant' else 'equivariant_scaled_mse']
            denominator=row[kind+'_persistence']
            row[kind+'_relative_to_persistence']=numerator/denominator if denominator>0 else None
        metrics['horizons'][f'{ps:g}']=row
    arrays.update({f'long_future_{k}':v for k,v in saved.items()},long_future_errors=error,long_future_error_names=np.array(names))
    return metrics,arrays
