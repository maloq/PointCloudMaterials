"""Immutable identities for predictive-information studies without AP tuning."""
import json
from pathlib import Path

from src.project_runtime.paths import resolve_path
from src.research.structural_state.common import sha, digest, write_json, save_checkpoint


class Study:
    def __init__(self, config):
        self.path = Path(config).resolve()
        self.config = json.loads(self.path.read_text())
        c = self.config
        if c['protocol'] != 'supervised_onset_information_v4' or c['branch'] != 'crystallization_supervised':
            raise ValueError('New runs require supervised_onset_information_v4 without AP tuning; '
                             'historical protocols must be inspected with their frozen source')
        forbidden = {'ranking_every', 'ap_temperature', 'ap_weight', 'ranking_horizon_weights'} & c['training'].keys()
        if forbidden or any('ranking' in a for a in c['arms']):
            raise ValueError(f'AP-specific tuning is prohibited: {sorted(forbidden)}; remove ranking settings')
        if c['objective'] != 'hazard_nll' or c['selection_metric'] != 'hazard_nll':
            raise ValueError('Supervised information studies require hazard_nll training and selection; AP is reporting only')
        if c['prediction_context']['external_inputs'] != []:
            raise ValueError('Temperature and explicit time inputs are disabled by user instruction')
        if (c['primary_horizon_ps'], c['secondary_horizon_ps']) != (3., 6.):
            raise ValueError('Report 3 ps and 6 ps diagnostics; checkpoint selection uses predictive NLL')
        c['training'] = dict(batch_size=256, microbatch=256) | c['training']
        c['encoder'] = dict(channels=128, code_dim=128) | c['encoder']
        from .tracking import DEFAULTS, require_online
        c['wandb'] = DEFAULTS | c.get('wandb', {})
        require_online(c['wandb'])
        self.root = resolve_path(c['output'])
        self.technical = self.root / 'technical'
        self.technical.mkdir(parents=True, exist_ok=True)
        self.cache = resolve_path(c['cache'])

    def arm(self, name):
        return next(a for a in self.config['arms'] if a['name'] == name)

    def bind(self):
        from src.experiment_runner.prediction_context import write_context
        base = Path(__file__).resolve().parents[3]
        paths = list((base / 'src/research/supervised_onset').glob('*.py'))
        paths += list((base/'src/data/fixed_cohort').glob('*.py'))
        paths += list((base/'src/research/encoder_context').glob('*.py'))
        paths += [base / p for p in (
            'src/research/structural_state/model.py', 'src/research/structural_state/data.py',
            'src/research/structural_state/common.py', 'src/training_methods/bcr/model.py',
            'src/training_methods/bcr/data.py', 'src/models/encoders/mace_causal.py',
            'src/models/encoders/mace_backend.py', 'src/research/robust_onset/metrics.py',
            'src/models/encoders/spatial_mace.py', 'src/models/encoders/graph_bank.py',
            'src/research/local_predictability/metrics.py', 'src/research/trajectory_stability/spectrum.py',
            'src/experiment_runner/prediction_context.py')]
        paths += [base / 'src/models/encoders/spatial_mace.py',
                  base / 'src/models/encoders/graph_bank.py']
        receipt = dict(config=self.config, data=sha(self.cache / 'manifest.json'),
                       implementation={str(p.relative_to(base)): sha(p) for p in paths})
        self.identity = digest(receipt)
        path = self.technical / 'identity.json'
        if path.exists() and json.loads(path.read_text()) != receipt:
            raise ValueError(f'Frozen supervised study changed: {path}; use a fresh output')
        write_json(path, receipt)
        write_context(self.root, dict(protocol=self.config['protocol'], identity=self.identity,
            branch=self.config['branch'], encoder_inputs=['centered local coordinates', 'center indicator', 'Al species'],
            predictor_inputs=[f"exported {self.config['encoder']['code_dim']}-dimensional state"], external_inputs=[],
            spatial=dict(maximum_atoms=80, radius_A=self.config['encoder']['radius'], halo=False,
                         edge_cutoff_A=self.config['encoder']['cutoff'], message_passing_layers=2,
                         channels=self.config['encoder']['channels'], export='projected residual'),
            parameter_budget=self.config['parameter_budget'],
            runtime=dict(self.config['runtime'], layout=self.config['encoder']['layout'],
                         conv_fusion=self.config['encoder']['conv_fusion'],
                         node_capacity=80*self.config['training']['microbatch'],
                         node_padding='disconnected zero-weight nodes; real edges unchanged'),
            history=dict(input_frames=1, history_ps=0, time_features=False), velocities=False,
            arms=[dict(name=a['name'], input=a['input'],
                       relaxation='full current periodic cell, then local crop' if a['input'] in ('cold','paired') else None,
                       training_teacher=self.config['teacher_arm'] if a['teacher'] else None)
                  for a in self.config['arms']],
            probes='Exported state only; no metadata covariates',
            controls=[] if 'fixed_dataset' in self.config else ['constant', 'observed descriptors', 'relaxed descriptors', 'paired descriptors'],
            fixed_dataset=self.config.get('fixed_dataset'),
            conditions_in_cache='Audit metadata are not model or readout inputs'))
        return self.identity
