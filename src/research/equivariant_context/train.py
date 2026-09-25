"""Frozen-encoder context fits and held-out physical-event comparisons."""
import json
import time
from types import SimpleNamespace
import numpy as np
import torch
from src.research.structural_state.common import save_checkpoint, write_json, sha
from src.research.local_predictability.metrics import source_weights
from src.research.supervised_onset.data import sampling_distribution
from src.research.supervised_onset.tracking import (tracked_run, risk_diagnostics, validation_fields,
                                                  population_summary, final_fields)
from src.research.supervised_onset.evaluate import calibrate_risks, score_predictions
from src.experiment_runner.metric_docs import write_metric_table
from .data import ContextCorpus
from .model import ContextPredictor, context_fields


@torch.no_grad()
def predict(model,corpus,ids,chunk):
    model.eval()
    return torch.cat([model(corpus.batch(ids[s:s+chunk])) for s in range(0,len(ids),chunk)])


def fit(study,domain,variant,device,deadline):
    c=study.config;corpus=ContextCorpus(study,domain,variant,device)
    root=study.root/f'{domain}-{variant}';technical=root/'technical';technical.mkdir(parents=True,exist_ok=True)
    if (technical/'complete.json').exists():
        record=json.loads((technical/'complete.json').read_text())
        if record['identity']!=study.identity or record['cache_identity']!=corpus.cache_identity:
            raise ValueError('Completed run identity differs')
        return record
    torch.manual_seed(c['seed']);rng=np.random.default_rng(c['seed'])
    model=ContextPredictor(variant,**c['predictor']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['training']['learning_rate'],weight_decay=c['training']['weight_decay'])
    fit_ids=corpus.split['train'];selection=corpus.split['selection'];batch=c['batch_size']
    q,importance=sampling_distribution(corpus.pop['source'][fit_ids],corpus.pop['event'][fit_ids],.5)
    weights=torch.as_tensor(source_weights(corpus.pop['source'][selection]),dtype=torch.float32,device=device)
    start=0;best=float('inf');last=technical/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location=device,weights_only=False)
        if saved['identity']!=study.identity or saved['cache_identity']!=corpus.cache_identity:
            raise ValueError('Resume config/code/data differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['numpy_rng'];start=saved['update'];best=saved['best']
    epoch_stream=None
    if 'epochs' in c['training']:
        import math
        from src.research.encoder_context.epochs import batches,epoch_weights
        steps_per_epoch=math.ceil(len(fit_ids)/batch)
        if c['training']['updates']!=steps_per_epoch*c['training']['epochs']:
            raise ValueError('Predictor budget must cover its complete epochs')
        epoch_stream=iter(batches(np.arange(len(fit_ids)),batch,c['training']['epochs'],c['seed'],start))
        full_weights=epoch_weights(corpus.pop['source'][fit_ids])
    def save(update,path):
        save_checkpoint(path,dict(identity=study.identity,cache_identity=corpus.cache_identity,domain=domain,
            variant=variant,model=model.state_dict(),optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,
            update=update,best=best,scalers=corpus.scalers,config=c,input_fields=list(context_fields(variant))+['nominal']))
    labels=dict(symmetric_invariant='Symmetric invariant',vector_messages='Vector messages',
                tensor_attention='Tensor attention',harmonic_hierarchy='Harmonic hierarchy')
    cohort=c.get('experiment_label','Al64' if 'fixed_dataset' in c else 'Al16')
    tracked_config=dict(c,wandb=dict(c.get('wandb',{}),display_name=
        f"{cohort} | {'Observed' if domain=='hot' else 'Relaxed'} | {labels[variant]}"))
    tracking_study=SimpleNamespace(config=tracked_config,root=root,technical=technical,identity=study.identity)
    with tracked_run(tracking_study,f'{domain}-{variant}',job_type='predictor') as tracking:
        population_summary(tracking,corpus,c)
        tracking.summary['model/predictor_parameters']=sum(p.numel() for p in model.parameters())
        tracking.summary['model/input_domain']='observed' if domain=='hot' else 'relaxed'
        tracking.summary['training_log_semantics']='Source-weighted event likelihood; AP never used for optimization or selection'
        for update in range(start+1,c['training']['updates']+1):
            if time.time()>deadline-180:
                save(update-1,last)
                raise TimeoutError('Context fit checkpointed; resume the identical worker to finish the fixed update budget')
            if epoch_stream is None:
                drawn=rng.choice(len(fit_ids),batch,p=q);sample_weights=importance[drawn]
            else:
                step,drawn=next(epoch_stream)
                if step!=update-1:raise ValueError('Predictor epoch cursor differs')
                sample_weights=full_weights[drawn]
            ids=fit_ids[drawn]
            iw=torch.as_tensor(sample_weights,dtype=torch.float32,device=device)
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=torch.zeros((),device=device)
            for offset in range(0,len(ids),c['microbatch']):
                part=ids[offset:offset+c['microbatch']]
                logp=model(corpus.batch(part))
                value=-(iw[offset:offset+len(part)]*logp[torch.arange(len(part),device=device),corpus.events[part]]).sum()/len(ids)
                value.backward();loss+=value.detach()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            if update%c['training']['log_every']==0:
                record=dict(optimizer_update=update,**{'train/event_nll':float(loss.detach()),
                    'train/gradient_norm':float(norm),'train/learning_rate':optimizer.param_groups[0]['lr']})
                if epoch_stream is not None:record['train/epoch']=update/steps_per_epoch
                tracking.log(record)
                with (technical/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if update%c['training']['evaluate_every']==0 or update==c['training']['updates']:
                val=predict(model,corpus,selection,batch)
                nll=float(-(weights*val[torch.arange(len(selection),device=device),corpus.events[selection]]).sum())
                if not np.isfinite(nll):raise FloatingPointError(f'Nonfinite selection NLL at update {update}')
                minimum=c['training'].get('minimum_selection_epoch',0)*(steps_per_epoch if epoch_stream is not None else 1)
                if update>=minimum and nll<best:
                    best=nll;save(update,technical/'best.pt')
                    tracking.summary['checkpoint/selected_update']=update
                risks=val[:,:5].exp().cumsum(-1).clamp(0,1).cpu().numpy()
                scores=dict(nll=nll,**risk_diagnostics(corpus.pop['event'][selection],risks,corpus.pop['source'][selection]))
                record=dict(optimizer_update=update,**validation_fields(scores))
                tracking.log(record)
                with (technical/'validation.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                if np.isfinite(best):tracking.summary['checkpoint/validation_event_nll']=best
                save(update,last)
            elif update%c['training']['save_every']==0:save(update,last)
            if epoch_stream is not None and update==12*steps_per_epoch:save(update,technical/'epoch-012.pt')
        saved=torch.load(technical/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(saved['model']);logp=predict(model,corpus,np.arange(len(corpus.events)),batch).cpu().numpy()
        risk=logp[:,:5].astype(float);risk=np.exp(risk).cumsum(-1).clip(0,1)
        calibrated,calibration=calibrate_risks(corpus,risk)
        scores=score_predictions(corpus,risk,c['bootstrap'],c['seed'],calibrated)
        nll={role:float(-source_weights(corpus.pop['source'][ids])@logp[ids,corpus.pop['event'][ids]])
             for role,ids in corpus.split.items()}
        np.savez(technical/'predictions.npz',logp=logp,risks=risk,calibrated=calibrated,**corpus.pop)
        metrics=dict(event_nll=nll,horizons=scores)
        write_json(technical/'metrics.json',metrics)
        write_metric_table(metrics,root,family='equivariant_context')
        record=dict(identity=study.identity,cache_identity=corpus.cache_identity,domain=domain,variant=variant,
            input_fields=list(context_fields(variant))+['nominal'],
            updates=c['training']['updates'],selected_update=saved['update'],selection_nll=best,
            parameters=sum(p.numel() for p in model.parameters()),calibration=calibration,
            checkpoint_sha256=sha(technical/'best.pt'),predictions_sha256=sha(technical/'predictions.npz'),
            state='complete')
        if 'fixed_dataset' in c:
            from src.data.fixed_cohort.dataset import read_release
            from src.data.fixed_cohort.protocol import assert_prediction_rows
            fixed_root,_=read_release(c['fixed_dataset']['root'])
            with np.load(fixed_root/'benchmark/population.npz') as fixed:
                assert_prediction_rows(fixed['sample_id'],corpus.pop['sample_id'])
        write_json(technical/'complete.json',record)
        tracking.summary.update(final_fields(metrics,record))
        return record


def collect(study):
    """Paired whole-source uncertainty versus the matched symmetric baseline."""
    rows={};differences={}
    for domain in study.config['domains']:
        predictions={}
        for variant in study.config['variants']:
            path=study.root/f'{domain}-{variant}'/'technical'
            if not (path/'complete.json').exists():continue
            record=json.loads((path/'complete.json').read_text())
            if record['identity']!=study.identity:raise ValueError(f'Changed result: {path}')
            rows[f'{domain}-{variant}']=json.loads((path/'metrics.json').read_text())
            predictions[variant]=dict(np.load(path/'predictions.npz'))
        if 'symmetric_invariant' not in predictions:continue
        base=predictions['symmetric_invariant'];ids=np.flatnonzero(base['role']=='test')
        from sklearn.metrics import average_precision_score
        source=base['source'][ids];roots,inverse=np.unique(source,return_inverse=True);w=source_weights(source)
        for name,candidate in predictions.items():
            if name=='symmetric_invariant':continue
            for key in ('source','event','frame','atom','role'):np.testing.assert_array_equal(candidate[key],base[key])
            rng=np.random.default_rng(study.config['seed']);estimates=[]
            delta=-candidate['logp'][ids,base['event'][ids]]+base['logp'][ids,base['event'][ids]]
            def values(weight):
                result=[float(weight@delta)]
                for col in (1,2):
                    target=base['event'][ids]<=col;keep=weight>0
                    result.append(float(average_precision_score(target[keep],candidate['risks'][ids,col][keep],sample_weight=weight[keep])-
                        average_precision_score(target[keep],base['risks'][ids,col][keep],sample_weight=weight[keep])) if weight@target>0 else np.nan)
                return result
            for _ in range(study.config['bootstrap']):
                multiplicity=np.bincount(rng.integers(len(roots),size=len(roots)),minlength=len(roots))
                estimates.append(values(w*multiplicity[inverse]))
            a=np.array(estimates);point=values(w)
            differences[f'{domain}-{name}']={metric:dict(difference=point[k],ci95=np.nanquantile(a[:,k],[.025,.975]).tolist(),
                valid_draws=int(np.isfinite(a[:,k]).sum())) for k,metric in enumerate(('event_nll','AP3','AP6'))}
    metrics=dict(models=rows,paired_test_differences=differences)
    write_json(study.technical/'comparison.json',metrics)
    write_metric_table(metrics,study.root,family='equivariant_context',name='comparison')
    return dict(completed=len(rows),expected=len(study.config['domains'])*len(study.config['variants']))
