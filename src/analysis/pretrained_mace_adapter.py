"""Fine-tuned MACE encoder bridge to the existing full static analysis."""
from pathlib import Path
from omegaconf import OmegaConf
import torch
from torch import nn
from src.models.encoders.pretrained_mace import PretrainedMACEEncoder
from src.data_utils.temporal_campaign import write_json


class PretrainedMACEAnalysis(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        if cfg.data.normalize:raise ValueError('MLIP MACE requires physical Angstrom offsets')
        if OmegaConf.select(cfg,'protocol') not in ('plain80','thermal80'):
            raise ValueError('Retired central/tapered MACE checkpoints require their recorded source snapshot; this encoder uses full 80-atom mean pooling.')
        self.encoder=PretrainedMACEEncoder(cfg.pretrained_checkpoint,performance=OmegaConf.to_container(cfg.performance,resolve=True))

    def forward(self,points):
        order=points.square().sum(-1).argsort(1)
        points=points.gather(1,order[:,:,None].expand(-1,-1,3))
        material=torch.zeros(len(points),device=points.device,dtype=torch.long)
        return self.encoder(points,material),None,None


def export_encoder(cfg):
    out=Path(cfg['output']);directory=out/'encoder';directory.mkdir(exist_ok=True);(directory/'.hydra').mkdir(exist_ok=True)
    analysis=OmegaConf.load(cfg['analysis_config'])
    data=OmegaConf.load(analysis.inputs.data_config)
    config=OmegaConf.create(dict(model_type='pretrained_mace_encoder',representation_source='encoder',pretrained_checkpoint=cfg['pretrained_checkpoint'],protocol=cfg['protocol'],performance=cfg['performance'],batch_size=cfg['microbatch_size'],num_workers=4,max_samples=0,split_seed=123,data=OmegaConf.to_container(data,resolve=True)))
    payload=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    weights={k:v for k,v in payload['model'].items() if k.startswith('encoder.')}
    torch.save(dict(state_dict=weights),directory/'encoder.ckpt');OmegaConf.save(config,directory/'.hydra/config.yaml')
    write_json(directory/'provenance.json',dict(training_checkpoint=str(out/'best.pt'),best_epoch=payload['epoch'],
        representation='256 scalar MACE channels mean-pooled over all 80 atoms; native interaction cutoff only; fixed train-only scaler; no TDA head',
        pretrained_url=cfg['pretrained_url']))


def saved_reference_spatial_rows(cfg,frames):
    """Reuse archived scores only on their exact grid and shared-center protocol."""
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree
    settings=cfg['saved_static_reference'];out=Path(cfg['output'])
    saved=pd.read_csv(settings['metrics_csv']);cache=np.load(out/'static_analysis/analysis_inference_cache.npz')
    rows=[];offset=0;rng=np.random.default_rng(20260907)
    for frame in frames:
        name=frame['output_name'];sl=slice(offset,offset+frame['num_samples']);coordinates=cache['coords'][sl]
        original=np.load(Path(settings['model_analysis'])/'snapshots'/name/'md_space/local_structure_coords_clusters.npz')['coords']
        np.testing.assert_array_equal(coordinates,original,err_msg=f'{name}: archived spatial scores require the exact original center grid and ordering')
        mask=np.ones(len(coordinates),dtype=bool)
        for root in settings['coordinate_roots']:
            reference=np.load(Path(root)/'snapshots'/name/'md_space/local_structure_coords_clusters.npz')['coords']
            distances,_=cKDTree(reference).query(coordinates);mask&=distances<1e-6
        selected=np.flatnonzero(mask);c=coordinates[selected]
        expected=saved[(saved.frame==name)&(saved.model=='pretrained_MACE_finetuned')].iloc[0]
        if len(c)!=expected.shared_centers:raise ValueError(f'{name}: {len(c)} shared centers differ from archived comparison {expected.shared_centers}')
        near=cKDTree(c).query(c,k=7,workers=4)[1][:,1:];sample=rng.choice(len(c),min(10000,len(c)),replace=False);random=rng.integers(len(c),size=(len(sample),6))
        z=cache['inv_latents'][sl][selected];scale=np.maximum(z.std(0),1e-3)
        local=np.square((z[sample,None]-z[near[sample]])/scale).mean();shuffled=np.square((z[sample,None]-z[random])/scale).mean()
        labels=np.load(out/'static_analysis/snapshots'/name/'md_space/local_structure_coords_clusters.npz')['clusters'][selected]
        a=np.broadcast_to(labels[:,None],near.shape).ravel();b=labels[near].ravel();chance=np.dot(np.bincount(a,minlength=7)/len(a),np.bincount(b,minlength=7)/len(b))
        rows.append(dict(frame=name,model='pretrained_MACE_finetuned',shared_centers=len(c),neighbor_mse_over_random=float(local/shuffled),adjusted_neighbor_cluster_agreement=float(((a==b).mean()-chance)/(1-chance))))
        for record in saved[saved.frame==name].to_dict('records'):
            if record['model']=='pretrained_MACE_finetuned':record['model']='previous_MACE'
            rows.append(record)
        offset+=frame['num_samples']
    return rows


def write_report(cfg):
    """Summarize the finished run and spatial geometry on exactly shared centers."""
    import json
    import pandas as pd
    out=Path(cfg['output']);training=json.loads((out/'training_summary.json').read_text());data=json.loads((out/'data_summary.json').read_text())
    metrics=json.loads((out/'static_analysis/analysis_metrics.json').read_text())
    frames=metrics['real_md_qualitative']['frames'];rows=saved_reference_spatial_rows(cfg,frames)
    pd.DataFrame(rows).to_csv(out/'static_spatial_comparison.csv',index=False)
    val=training['selected_validation'];selected_epoch=training['selected_epoch'];count=data['counts']['train'];fit=metrics['clustering']['cluster_fit_info_by_k']['7']
    write_json(out/'static_comparison_protocol.json',dict(reference_models='Evaluation references only; neither is used in MACE training.',aligned_centers=sum(r['shared_centers'] for r in rows if r['model']=='pretrained_MACE_finetuned'),neighbors=6,standardization='Per-model, per-frame feature standard deviations on shared centers.',limitations='Spatial coherence alone is not proof of physical validity. Seven clusters are imposed by the requested standard analysis. No PTM phase labels used as ground truth.'))
    write_json(out/'archived_spatial_reference.json',cfg['saved_static_reference'])
    lines=['# Pretrained small MACE: completed training and static analysis','',
        f"Stop: **{training['stop_reason']}**. Selected epoch: **{selected_epoch}** ({training['checkpoint_selection']} selection). Training anchor exposures: **{training['anchor_exposures']:,} anchors / {training['view_exposures']:,} views**.",
        '', 'The encoder starts from the official MACE-MP-0b2 small MLIP checkpoint. All used backbone weights are fine-tuned. There is no GeoFrame teacher, EMA model, PTM phase classification, or projector in the analyzed 256D encoder output.',
        '', '| Material | Training anchors | Distinct states in the three training views |','|---|---:|---:|']
    for name,v in count.items():lines.append(f"| {name} | {v['anchor_quadruplets']:,} | {v['distinct_neighborhood_states']:,} |")
    lines+=['','| Held-out validation | Value |','|---|---:|',f"| TDA whitened PCA MSE | {val['tda_mse']:.6f} |",f"| Spatial latent MSE | {val['spatial_mse']:.6f} |",f"| Short-time latent MSE | {val['temporal_mse']:.6f} |",'',
        'TDA PCA and feature scaling are fitted only on training data. These are in-domain validation metrics of learned embeddings; they do not establish predictive performance for arbitrary future times. '+data['limits'],'',
        '## Static spatial comparison','',
        'All six Al frames use the original sampling centers with the input point count and physical support recorded in `data_summary.json` and the exported encoder configuration. The full existing `configs/analysis/static.yaml` workflow runs on the encoder alone. Lower neighbor/random feature MSE means more spatial coherence; it is not itself a phase-identification score.','',
        '| Frame | Model | Neighbor / random MSE | Adjusted cluster agreement |','|---|---|---:|---:|']
    for r in rows:lines.append(f"| {r['frame']} | {r['model']} | {r['neighbor_mse_over_random']:.4f} | {r['adjusted_neighbor_cluster_agreement']:.4f} |")
    lines+=['',f"Standard clustering fit: `{json.dumps(fit)}`.",'',
        '[Static analysis](static_analysis/) · [Training curve](training.jsonl) · [Data counts](data_summary.json) · [Spatial comparison CSV](static_spatial_comparison.csv)','']
    (out/'RESULTS.md').write_text('\n'.join(lines))
