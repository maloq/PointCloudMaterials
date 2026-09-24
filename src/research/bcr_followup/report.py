"""Export completed diagnostic stages without changing the original G1 outcome."""
import csv
import json

import numpy as np

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.training_methods.bcr.evaluate import paired_root_gain
from .common import write_json


def write_table(root, name, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (root/'tables'/f'{name}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def run(study):
    snapshot_metric_docs(study.root, 'bcr_followup')
    interventions = []
    for path in sorted((study.technical/'interventions').glob('*/complete.json')):
        record = json.loads(path.read_text())
        for level, populations in record['levels'].items():
            for population, values in populations.items():
                for name, result in values.items():
                    row = dict(step=record['encoder_step'], noise_over_d0=level, population=population, intervention=name,
                               **{k: v for k, v in result.items() if k != 'ci95'})
                    row.update(ci95_lower=result['ci95'][0] if result['ci95'] else None,
                               ci95_upper=result['ci95'][1] if result['ci95'] else None)
                    interventions.append(row)
    write_table(study.root, 'interventions', interventions)
    probes, groups, roots, temperatures = [], [], [], []
    for domain, folder in [('pilot', study.technical/'probes'), ('paired_relaxed', study.technical/'relaxed/probes')]:
        for path in sorted(folder.glob('**/complete.json')):
            record = json.loads(path.read_text())
            base = dict(domain=domain, step=record['step'] if domain == 'pilot' else record['encoder_step'],
                        input_domain=record.get('input_domain', 'melt'), target_domain=record.get('target_domain', 'melt'),
                        representation=record['representation'], family=record['family'], selected_step=record['selected_step'],
                        ridge_tuning_mse=record['ridge_tuning_mse'], selected_tuning_mse=record['selected_tuning_mse'])
            for probe, values in record['metrics'].items():
                for population in ('all', 'liquid'):
                    result = values[population]
                    probes.append(dict(**base, probe=probe, population=population,
                                       standardized_rmse=result['all']['standardized_rmse']))
                    roots += [dict(**base, probe=probe, population=population, root=root, standardized_rmse=value)
                              for root, value in result['per_root'].items()]
                    temperatures += [dict(**base, probe=probe, population=population, temperature=temperature,
                        standardized_rmse=value['standardized_rmse']) for temperature, value in result.items()
                        if temperature.startswith('T')]
                groups += [dict(**base, probe=probe, target_group=group,
                                standardized_rmse=value['all']['standardized_rmse']) for group, value in values['groups'].items()]
    write_table(study.root, 'probes', probes); write_table(study.root, 'probe_groups', groups); write_table(study.root, 'probe_roots', roots)
    write_table(study.root, 'probe_temperatures', temperatures)
    decoder_rows = []
    reference = study.technical/'fresh_decoders/0/errors.npz'
    if (reference.parent/'complete.json').exists():
        with np.load(reference) as data: reference_errors, root_ids, indices = data['errors'], data['roots'], data['indices']
        for path in sorted((study.technical/'fresh_decoders').glob('*/complete.json')):
            record = json.loads(path.read_text())
            with np.load(path.parent/'errors.npz') as data:
                np.testing.assert_array_equal(indices, data['indices']); np.testing.assert_array_equal(root_ids, data['roots'])
                for k, level in enumerate(study.manifest['noise_levels']):
                    score = paired_root_gain(data['errors'][k].mean(0), reference_errors[k].mean(0), root_ids)
                    decoder_rows.append(dict(encoder=record['encoder'], updates=record['decoder_updates'], noise_over_d0=level,
                        **{key: value for key, value in score.items() if key != 'ci95'},
                        ci95_lower=score['ci95'][0], ci95_upper=score['ci95'][1]))
    write_table(study.root, 'fresh_decoders', decoder_rows)
    progress = dict(pilot_probe_fits=len(probes)//4, relaxed_probe_fits=sum(r['domain']=='paired_relaxed' for r in probes)//4,
                    intervention_checkpoints=len({r['step'] for r in interventions}),
                    fresh_decoder_fits=len({r['encoder'] for r in decoder_rows}),
                    original_G1='Unchanged; no gate threshold revision', encoder_training=False)
    progress['pilot_probe_fits'] = sum(r['domain']=='pilot' for r in probes)//4
    write_json(study.technical/'report-status.json', progress)
    (study.root/'README.md').write_text('# BCR conditioning and relaxed-structure audit\n\n'
        'One seed; frozen BCR encoders. Eight-hour queue. Original G1 remains unchanged.\n\n'
        'Scientific protocol: [experiment](../../../experiments/bcr_followup_20260922/README.md). '
        'Definitions: [METRICS](tables/METRICS.md). Stage/checkpoint identities: `technical/identity.json`.\n\n'
        'Tables appear as stages complete: `interventions.csv`, `probes.csv`, `probe_groups.csv`, '
        '`probe_roots.csv`, `fresh_decoders.csv`. `technical/queue-status.json` records completion or time limit.\n\n'
        'The relaxed audit uses archived full-cell float16 geometry with complete local support; '
        'it is separate from the high-precision melt corruption experiment. No final test roots or new simulations.\n')
    return progress
