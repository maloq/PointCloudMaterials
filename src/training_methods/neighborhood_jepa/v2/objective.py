"""Batch-independent data losses and explicitly sample-normalized SIGReg."""
import torch
from torch import nn
from lejepa.multivariate import SlicingUnivariateTest
from lejepa.univariate import EppsPulley
from src.training_methods.structural_pretraining.objective import block_errors, PHYSICAL_BLOCKS, TDA_BLOCKS
from .contracts import LAYOUT, RequiredViewPlan
from .geometry import scaled_error


class SIGReg(nn.Module):
    def __init__(self, mode='per_sample_discrepancy'):
        super().__init__()
        if mode not in ('per_sample_discrepancy', 'official_test_statistic'):
            raise ValueError(mode)
        self.mode = mode
        self.test = SlicingUnivariateTest(EppsPulley(t_max=3., n_points=17,
            integration='trapezoid'), num_slices=256)

    def forward(self, samples):
        if len(samples) < 2:
            raise ValueError('SIGReg requires >=2 independent anchors; inference does not')
        raw = self.test(samples)
        discrepancy = raw/len(samples)
        return (discrepancy if self.mode == 'per_sample_discrepancy' else raw), raw, discrepancy


class Objective(nn.Module):
    def __init__(self, manifest, spec, all_views=False):
        super().__init__()
        self.spec = spec
        self.plan = RequiredViewPlan.from_spec(spec, all_views)
        for name in ('physical','tda'):
            for field in ('mean','std'):
                self.register_buffer(f'{name}_{field}', torch.tensor(manifest['normalization'][name][field], dtype=torch.float32))
        self.sigreg = SIGReg(spec['sigreg_mode'])

    def physical_errors(self, model, z, target, group):
        return block_errors(model.physical(z,group), (target-self.physical_mean)/self.physical_std, PHYSICAL_BLOCKS).mean(-1)

    def tda_errors(self, model, z, target, group):
        return block_errors(model.tda(z,group), (target-self.tda_mean)/self.tda_std, TDA_BLOCKS).mean(-1)

    def forward(self, model, encoded, target):
        n = len(target['group'])
        z = encoded.reshape(n,len(self.plan.views),LAYOUT.packed_dim)
        center_slots = [self.plan.slot(t,0) for t in (1,2)]
        centers = z[:,center_slots,:LAYOUT.invariant_dim].flatten(0,1)
        groups = target['group'].repeat_interleave(2)
        physical = self.physical_errors(model,centers,target['physical'].flatten(0,1),groups).mean()
        tda = self.tda_errors(model,centers,target['tda'].flatten(0,1),groups).mean()
        geometry = scaled_error(z[:,center_slots,LAYOUT.invariant_dim:],
                                target['moments'][:,center_slots],model.encoder.geometry_scales).mean()
        zero = encoded.sum()*0
        invariant_prediction, equivariant_prediction, future = zero, zero, zero
        if self.plan.query_list:
            inv, eq = model.predictions(encoded,target,self.plan)
            for family,weight in self.spec['family_weights'].items():
                if weight == 0:
                    continue
                columns = [i for i,q in enumerate(self.plan.query_list) if q.family == family]
                slots = [self.plan.slot(self.plan.query_list[i].time_index,
                         self.plan.query_list[i].neighbor_index) for i in columns]
                actual = z[:,slots]
                invariant_prediction = invariant_prediction+weight*(inv[:,columns]-actual[...,:LAYOUT.invariant_dim]).square().mean()
                equivariant_prediction = equivariant_prediction+weight*scaled_error(eq[:,columns],actual[...,LAYOUT.invariant_dim:],model.encoder.geometry_scales).mean()
            if self.spec['future_weight']:
                j = next(i for i,q in enumerate(self.plan.query_list) if q.family == 'future_center')
                future = self.physical_errors(model,inv[:,j],target['physical'][:,1],target['group']).mean()
                future = future+.25*self.tda_errors(model,inv[:,j],target['tda'][:,1],target['group']).mean()
        current = z[:,self.plan.slot(1,0),:LAYOUT.invariant_dim]
        regularizer, raw, discrepancy = self.sigreg(model.projector(current))
        terms = dict(physical=physical,tda=.25*tda,geometry=self.spec['geometry_weight']*geometry,
            future=self.spec['future_weight']*future,
            prediction_invariant=self.spec['prediction_weight']*invariant_prediction,
            prediction_equivariant=self.spec['prediction_weight']*equivariant_prediction,
            regularizer=self.spec['sigreg_weight']*regularizer)
        self.diagnostics = dict(sigreg_raw=float(raw.detach()),sigreg_discrepancy=float(discrepancy.detach()),
            independent_anchors=n, group_mixture_weight=1., invariant_rms=float(current.detach().square().mean().sqrt()),
            equivariant_rms=float(z[...,LAYOUT.invariant_dim:].detach().square().mean().sqrt()))
        return sum(terms.values()),terms
