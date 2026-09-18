"""Small W&B dashboard; exhaustive per-update diagnostics stay in JSONL."""
from collections import defaultdict


def training_metrics(row):
    """Each loss component is its actual weighted contribution to loss/total."""
    values={'loss/total':row['loss'], 'loss/physical':row['physical'],
            'loss/tda':.25*row['instantaneous_tda']}
    if 'vicreg' in row:
        values['loss/vicreg']=row['vicreg_weighted']
        values.update({f'vicreg/{name}':row[key] for name,key in (
            ('total','vicreg'),('invariance','invariance'),('variance','variance'),('covariance','covariance'))})
        values['health/projector_std']=row['projector_std']
        values['health/projector_effective_rank']=row['projector_participation_ratio']
    else:
        values['loss/jepa']=.1*row['representation']
        values.update({f'jepa/{key}':row[key] for key in ('next_latent_mse','sigreg')})
    for key,name in (('physical_correlation_weighted','physical_correlation'),
                     ('bond_order_weighted','bond_order'),
                     ('backtracking_weighted','backtracking'),('future','future')):
        if key in row:values[f'loss/{name}']=row[key]
    return values


class Dashboard:
    """Average every update between emissions; axes and metadata get no panels."""
    def __init__(self,run):
        self.run=run;self.sums=defaultdict(float);self.counts=defaultdict(int)
        self.rows=0;self.latest=None;self.groups={}
        run.define_metric('training_step',hidden=True)
        run.define_metric('epoch',hidden=True)
        for prefix in ('loss','vicreg','jepa','health','optimization','validation'):
            run.define_metric(f'{prefix}/*',step_metric='epoch',summary='last')
        run.summary['logging']=dict(schema='compact_v1',
            training_aggregation='mean over all updates since previous emission',
            loss_components='weighted contributions; sum equals loss/total',
            diagnostics='technical/updates.jsonl and technical/validation.jsonl')

    def record(self,row):
        self.latest=row;self.rows+=1
        for key,value in training_metrics(row).items():
            self.sums[key]+=value;self.counts[key]+=1
        # Latest per-domain observations are summary metadata, never copied into
        # train/, train_by_group/, or separate spatial/temporal histories.
        for key,value in row.items():
            if key.startswith('groups/'):
                _,material,potential,metric=key.split('/')
                self.groups.setdefault(f'{material}/{potential}',{})[metric]=value

    def flush(self):
        if not self.rows:return
        # Absent future supervision on causal replay contributes zero to total.
        values={key:value/(self.rows if key.startswith('loss/') else self.counts[key])
                for key,value in self.sums.items()}
        self.run.log(dict(training_step=self.latest['step'],epoch=self.latest['epoch_equivalent'],
            **values,**{'optimization/head_learning_rate':self.latest['lr']}))
        if self.groups:
            self.run.summary['latest_group_diagnostics']=dict(step=self.latest['step'],groups=self.groups)
        self.sums.clear();self.counts.clear();self.rows=0;self.groups={}

    def validation(self,step,epoch,metrics):
        values={'validation/score':metrics['score'],
                'validation/present_physical':metrics['physical'],
                'validation/present_tda':metrics['instantaneous_tda']}
        if 'bond_order' in metrics:values['validation/bond_order']=metrics['bond_order']
        if 'future' in metrics:
            horizons=list(metrics['future'].values())
            for name in ('physical','tda'):
                values[f'validation/future_{name}']=sum(h[name] for h in horizons)/len(horizons)
        self.run.log(dict(training_step=step,epoch=epoch,**values))
        self.run.summary['validation_reference']=dict(rows=metrics['selection_rows'],
            sources=metrics['selection_sources'],present_group_mean_score=metrics['training_mean_score'])
        previous=self.run.summary.get('best_validation_score',float('inf'))
        if metrics['score']<previous:
            self.run.summary['best_validation_score']=metrics['score']
            self.run.summary['best_validation_step']=step
