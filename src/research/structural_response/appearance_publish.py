"""Publish completed first-appearance diagnostics to existing training run IDs.

This is publication only: it reads exported metrics, never fits or recomputes.
"""
import argparse
import csv
import json
from pathlib import Path
import time

from src.experiment_runner.execution import recorded_stage
from src.experiment_runner.wandb_tracking import update_recorded_summary
from src.project_runtime.paths import resolve_path


def publish(c, wait=False):
    out=resolve_path(c['output']);tech=out/'technical'
    with recorded_stage(tech/'training-summary-publication.json',stage='waiting' if wait else 'publishing') as progress:
        launch=json.loads((tech/'launch.json').read_text())
        while not (tech/'complete.json').exists():
            if not wait:raise FileNotFoundError('First-appearance evaluation is not complete')
            command=Path(f'/proc/{launch["pid"]}/cmdline')
            if not command.exists() or b'src.research.structural_response.appearance' not in command.read_bytes():
                raise RuntimeError('Owned appearance producer stopped before completion; inspect coordinator log')
            time.sleep(30)
        progress.update(stage='publishing')
        with (out/'tables/scores.csv').open() as handle:rows=list(csv.DictReader(handle))
        parent=resolve_path(c['parent'])
        recipe=json.loads((parent/'technical/code/config.json').read_text())
        for name in sorted({r['model'] for r in rows}):
            if name=='prior' or name.startswith('physical-'):continue
            if name.endswith('-spatial_start'):
                method=name.removesuffix('-spatial_start')
                receipt=resolve_path(recipe['spatial_initialization']['output'])/'analyses'/('spatial-'+method)/'technical/wandb.json'
            else:receipt=parent/'analyses'/name/'technical/wandb.json'
            fields={f'first_appearance_v1/{r["target"]}/{r["space"]}/{float(r["horizon_ps"]):g}ps/{r["readout"]}/{r["metric"]}':float(r['value'])
                    for r in rows if r['model']==name}
            update_recorded_summary(receipt,fields,evaluation='first-appearance-20261007-v1',
                expected=dict(entity='teshbek',project='PointCloudMaterials'))
            print('published',name,len(fields),'metrics; no new online runs',flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--wait',action='store_true')
    a=p.parse_args();publish(json.loads(Path(a.config).read_text()),a.wait)


if __name__=='__main__':main()
