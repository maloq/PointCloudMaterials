"""Selection-only replay diagnostics before choosing forecasting follow-ups."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.data.structural_pretraining.prepare import save_json
from src.project_runtime.paths import resolve_path
from src.research.crystallization_transfer.runtime import setup
from src.research.local_predictability.metrics import source_weights
from src.experiment_runner.metric_docs import write_metric_table
from .data import ResidentPaths
from .model import Forecaster,block_loss,cdf_from_logits
from .runtime import selected_indices
from .metrics import dense_brier


@torch.no_grad()
def run(original,output):
    setup();root=resolve_path(original)/'technical';output=resolve_path(output);(output/'technical').mkdir(parents=True,exist_ok=True)
    plan=json.loads((root/'plan.json').read_text());spec=json.loads((root/'runs/direct-E12/spec.json').read_text())
    data=ResidentPaths(plan,spec);ids=selected_indices(data.corpus,'selection',8,plan['config']['seed'])
    observed=data.observed(ids);target=data.targets(ids);weights=source_weights(data.corpus.source_ids[ids]);results={}
    def brier(cdf):return float(weights@dense_brier(cdf.cpu().numpy(),target['event'].cpu().numpy()))
    for method in ('direct','ar_mse','ar_gaussian','mixture','diffusion'):
        candidates=[]
        for folder in (root/'runs').glob(f'{method}-E*'):
            rows=[json.loads(line) for line in (folder/'validation.jsonl').read_text().splitlines()]
            best=min(rows,key=lambda x:x['selection_brier']);candidates.append((best['selection_brier'],folder,best,rows[-1]))
        _,folder,best,last=min(candidates,key=lambda x:(x[0],str(x[1])))
        saved=torch.load(folder/'best.pt',map_location='cuda',weights_only=False);model=Forecaster(saved['spec']).cuda()
        model.load_state_dict(saved['model']);model.eval();context=model.encode(observed)
        result=dict(selected_fit=folder.name,best_epoch=best['step']/1717,
            best_selection_brier=best['selection_brier'],last_selection_brier=last['selection_brier'])
        torch.manual_seed(719)
        paths,cdf=model.forecast(observed,16,16);result['replay_selection_brier']=brier(cdf)
        result['state_block_mse']=float(block_loss((paths.mean(1)-target['state']).square()).mean())
        if method in ('ar_mse','ar_gaussian'):
            _,_,hazard,_=model.recurrent(context)
            result['mean_feedback_brier']=brier(cdf_from_logits(hazard))
            # Diagnostic oracle only: never a deployable forecast or selection candidate.
            model.train();_,_,hazard,_=model.recurrent(context,target['state'],1.)
            result['teacher_forced_oracle_brier']=brier(cdf_from_logits(hazard));model.eval()
        if method=='mixture':
            mean,logstd,hazard=model.parallel(context);p=model.mixing(context).softmax(-1)
            result['mean_mixture_weights']=p.mean(0).cpu().tolist()
            result['effective_components']=float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum(-1)).mean())
            result['between_component_mean_variance']=float(mean.var(1,unbiased=False).mean())
            result['within_component_variance']=float(logstd.mul(2).exp().mean())
        if method=='diffusion':
            clean=torch.cat((target['state'],2*target['occurred']-1),-1)
            noise=torch.randn_like(clean);a=model.alpha_bar[-1];x=a.sqrt()*clean+(1-a).sqrt()*noise
            pred=model.denoise(x,torch.full((len(x),),63,device='cuda'),context)
            result['terminal_alpha_bar']=float(a);result['noise_error_amplification']=float(a.rsqrt())
            result['input_noise_channels']=model.noisy.in_features;result['bottleneck_channels']=model.noisy.out_features
            result['terminal_noise_mse']=float((pred-noise).square().mean())
            result['terminal_clean_mse']=float(((x-(1-a).sqrt()*pred)/a.sqrt()-clean).square().mean())
            result['predicted_first_frame_event_probability']=float(cdf[:,0].mean())
            result['actual_first_frame_prevalence']=float((target['event']==0).float().mean())
            # A null-space perturbation is invisible to the denoiser's input projection.
            _,_,vh=torch.linalg.svd(model.noisy.weight.float(),full_matrices=True);delta=vh[-1]
            changed=model.denoise(x+delta,torch.full((len(x),),63,device='cuda'),context)
            result['null_perturbation_output_max_difference']=float((changed-pred).abs().max())
        results[method]=result;print(json.dumps({method:result}),flush=True)
    save_json(output/'technical/diagnostics.json',results)
    write_metric_table(results,output,family='crystallization_paths_diagnostics',name='diagnostics')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--original',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.original,args.output)
