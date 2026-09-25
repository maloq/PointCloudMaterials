import json
from pathlib import Path
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.research.robust_onset.common import Study as BaseStudy,sha
from src.research.robust_onset.data import patches
from src.research.trajectory_stability.spectrum import source_weights
from .model import Model


class Study(BaseStudy):
    protocol='spatial_hierarchy_v1'
    queue_module='src.research.spatial_hierarchy.queue'
    metric_family='spatial_hierarchy'

    def __init__(self,path):
        super().__init__(path);self.context_cache=resolve_path(self.config['context_cache'])
        if any(a['input']!='observed' or a['noise'] or a['tensor_pool'] for a in self.config['arms']):
            raise ValueError('Hierarchy v1 holds observed focal geometry and noise-free training fixed')

    def prepare(self):
        from .data import prepare
        return prepare(self)

    def token_arrays(self):
        m=json.loads((self.context_cache/'manifest.json').read_text())
        path=self.context_cache/'tokens.npz'
        if sha(path)!=m['files'][path.name]:raise ValueError('Context token checksum changed')
        with np.load(path) as a:return dict(a)

    def make_model(self,encoder_config,arm,temperatures):
        from .data import RADII
        model=Model(encoder_config,temperatures,RADII[arm['context']],arm['early'])
        records=json.loads((self.cache/'records.json').read_text())
        fit=np.array([i for i,r in enumerate(records) if r['split']=='fit'])
        weights=source_weights([records[i]['root'] for i in fit])
        x=self.token_arrays()[arm['context']][fit].astype(float)
        mean=np.einsum('n,nsd->sd',weights,x);scale=np.sqrt(np.einsum('n,nsd->sd',weights,(x-mean)**2)).clip(1e-5)
        model.encoder.context_mean.copy_(torch.as_tensor(mean));model.encoder.context_scale.copy_(torch.as_tensor(scale))
        return model

    def graph_arrays(self,arm):
        return dict(super().graph_arrays(arm),context=self.token_arrays()[arm['context']])

    def noise_patches(self,arm,arrays):
        m=json.loads((self.context_cache/'manifest.json').read_text());p=self.context_cache/'patches.npz'
        if sha(p)!=m['files'][p.name]:raise ValueError('Noise input context changed')
        with np.load(p) as a:return patches(dict(a))

    def identity_extras(self):
        return dict(context=sha(self.context_cache/'manifest.json'),dense=sha(resolve_path(self.config['dense_inputs'])/'manifest.json'),
            implementation={str(p.relative_to(Path(__file__).resolve().parents[3])):sha(p) for p in Path(__file__).parent.glob('*.py')})

    def additional_diagnostics(self,model,corpus,features,device,deadline):
        from src.research.robust_onset.evaluate import infer_patches
        from src.research.trajectory_stability.spectrum import spectrum
        wide=self.noise_patches(None,None)
        source=np.array([r['root'] for r in corpus.records]);temperature=np.array([r['temperature_K'] for r in corpus.records])
        rng=np.random.default_rng(self.config['seed']+600);chosen=[]
        for root in np.unique(source[corpus.split['development']]):
            ix=corpus.split['development'][source[corpus.split['development']]==root]
            chosen.extend(rng.choice(ix,8,replace=False))
        chosen=np.array(chosen);donor=np.empty_like(chosen)
        for t in np.unique(temperature[chosen]):
            ix=np.flatnonzero(temperature[chosen]==t)
            donor[ix]=np.roll(chosen[ix],8)
        if np.any(source[donor]==source[chosen]):raise ValueError('Context intervention must change root within temperature')
        clean=[wide[i] for i in chosen];changed=[]
        for i,j in zip(chosen,donor,strict=True):
            local=wide[i][np.linalg.norm(wide[i],axis=1)<8.]
            outer=wide[j][np.linalg.norm(wide[j],axis=1)>=8.]
            changed.append(np.concatenate((local,outer)))
        chunk=self.config['training']['microbatch']
        a=infer_patches(model,clean,chunk,device,deadline);b=infer_patches(model,changed,chunk,device,deadline)
        ref=corpus.split['fit'];variance=spectrum(features[ref],source_weights(source[ref]))['total_energy']
        w=source_weights(source[chosen])
        response=float(np.sqrt(w@np.square(b.astype(float)-a).sum(1)/(2*variance)))
        replay=float(np.sqrt(w@np.square(a.astype(float)-features[chosen]).sum(1)/(2*variance)))
        return dict(rows=len(chosen),outer_context_swap_rms=response,clean_input_replay_rms=replay,
            definition='Keep coordinates inside 8 A exactly; replace 8-16 A atoms from another development root at the same temperature. Diagnostic only, never used for selection.',
            source_rows=chosen.tolist(),donor_rows=donor.tolist())
