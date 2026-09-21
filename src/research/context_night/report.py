"""Current completion state and comparable physical/event results."""
import json
from src.project_runtime.paths import resolve_path


def report(c):
    root=resolve_path(c['output']);lines=['# Overnight context and representation study','',
        'One seed; historical source-held-out cohort. Controls and treatments use identical capacity, warm parents and budgets. '
        'Selections use development data only. The new-encoder bridge adds a current snapshot export to the same fixed original-encoder trajectory context and targets.', '',
        '| Task | Status |','|---|---|']
    plan=json.loads((root/'technical/plan.json').read_text())
    for task in plan['tasks']:
        p=root/'technical/tasks'/f'{task["kind"]}--{task["name"]}.json';state=json.loads(p.read_text())['state'] if p.exists() else 'pending'
        lines.append(f'| {task["kind"]}: {task["name"]} | {state} |')
    lines+=['','## Trajectory prediction','',
        '| Run | 96ps integrated Brier ↓ | 12ps AP ↑ | 12ps timing MAE, detected only ↓ | Missed/event windows | Physical path MSE ↓ |',
        '|---|---:|---:|---:|---|---:|']
    for p in sorted((resolve_path(c['path']['output'])/'technical/runs').glob('*/metrics.json')):
        m=json.loads(p.read_text());short=m['short_horizon'];t=short['timing']['12.0'];ap=short['classification']['12.0']['average_precision']
        lines.append(f'| {p.parent.name} | {m["dense_integrated_brier"]:.5f} | {ap} | {t["detected_timing_mae_ps"]} | {t["missed_windows"]}/{t["event_windows"]} | {m["path"]["standardized_mse_physical"]["all_times"]:.5f} |')
    lines+=['','All trajectory forecasts are evaluated without future target feedback. Timing MAE excludes misses; inspect both. '
        'Future physical trajectories and original-encoder latent targets are fixed across treatments. '
        'Reports distinguish a better local snapshot encoder from gains due to additional observed spatial/history context.',
        '', 'Detailed encoder probes: encoders/CRYSTALLIZATION.md and short_readouts/technical/fits/. '
        'Scientific protocol: experiments/context_night_20260921/README.md. Update after all queued tasks finish.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
