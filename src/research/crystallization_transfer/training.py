"""Nested training populations and exact shuffled passes for scaling experiments."""
import math
import numpy as np
from src.data.structural_pretraining.prepare import digest


def configure_training(corpus,spec):
    """Restrict only training rows; all evaluation rows retain their global indices."""
    settings=spec['training'];seed=corpus.plan['config']['seed']
    full_count=len(corpus.splits['train']);all_groups=corpus.groups
    by_temperature={}
    for source in corpus.plan['sources']:
        if source['id'] in all_groups:
            by_temperature.setdefault(source['temperature_K'],[]).append(source['id'])
    count=settings['sources'];temperatures=sorted(by_temperature)
    if count%len(temperatures):raise ValueError('Training source count must balance temperatures exactly')
    per_temperature=count//len(temperatures);chosen=[]
    for temperature in temperatures:
        available=np.array(sorted(by_temperature[temperature]))
        if per_temperature>len(available):raise ValueError(f'Only {len(available)} training sources at {temperature} K')
        rng=np.random.default_rng(np.random.SeedSequence([seed,1701,int(temperature)]))
        chosen.extend(rng.permutation(available)[:per_temperature].tolist())
    fraction=settings['window_fraction']
    if not 0<fraction<=1:raise ValueError('Window fraction must be in (0,1]')
    groups={}
    for sid in sorted(chosen):
        ids=all_groups[sid];rng=np.random.default_rng(np.random.SeedSequence([seed,1702,int(sid)]))
        groups[sid]=np.sort(rng.permutation(ids)[:math.ceil(len(ids)*fraction)])
    corpus.groups=groups;corpus.splits['train']=sorted(np.concatenate(list(groups.values())).tolist())
    n=len(corpus.splits['train']);batch=corpus.plan['config']['batch_size']
    corpus.training_weights=np.zeros(len(corpus.rows),np.float32)
    for ids in groups.values():corpus.training_weights[ids]=n/(len(groups)*len(ids))
    per_epoch=math.ceil(n/batch)
    if settings['budget']=='epochs':updates=settings['epochs']*per_epoch
    elif settings['budget']=='full_data_epochs':updates=settings['epochs']*math.ceil(full_count/batch)
    else:raise ValueError(f'Unknown training budget: {settings["budget"]}')
    corpus.training_summary=dict(sources=len(groups),eligible_windows=n,full_training_windows=full_count,
        windows_by_source={str(s):len(ids) for s,ids in groups.items()},source_ids=sorted(groups),
        membership_sha256=digest(corpus.splits['train']),updates_per_epoch=per_epoch,updates=updates,
        complete_epochs=updates//per_epoch,partial_epoch_updates=updates%per_epoch,
        samples=sample_count(updates,n,batch),budget=settings,
        sampler='shuffled full passes; per-row weights N/(number_of_sources * source_window_count)')
    corpus._epoch=-1;corpus._permutation=None
    return corpus.training_summary


def sample_count(updates,windows,batch):
    full,remainder=divmod(updates,math.ceil(windows/batch))
    return full*windows+min(remainder*batch,windows)


def epoch_batch(corpus,step):
    batch=corpus.plan['config']['batch_size'];epoch,offset=divmod(step,corpus.training_summary['updates_per_epoch'])
    if corpus._epoch!=epoch:
        rng=np.random.default_rng(np.random.SeedSequence([corpus.plan['config']['seed'],1703,epoch]))
        corpus._permutation=rng.permutation(corpus.splits['train']);corpus._epoch=epoch
    return corpus._permutation[offset*batch:(offset+1)*batch].tolist()


def weighted_microbatch_loss(loss,weights,total_samples):
    """Accumulate exact sample means even when the final microbatch is short."""
    return (loss*weights).sum()/total_samples
