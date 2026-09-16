"""Prepare, validate, train and evaluate explicit MACE context variants."""

import argparse
from pathlib import Path
import sys

from src.project_runtime.paths import load_json
from src.experiment_runner.tracking import tracked_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', required=True, choices=['prepare', 'verify', 'frozen', 'train', 'labels', 'summarize', 'smoothness', 'recovery-readouts', 'recovery-verify', 'recovery-train', 'recovery-summarize', 'static-export', 'static-verify'])
    parser.add_argument('--mode', choices=['mean80', 'halo_mean80', 'halo_inner', 'halo_center'])
    parser.add_argument('--variant', choices=['dual_ssl', 'dual_physics'])
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    config = load_json(args.config)
    config['device'] = args.device
    root = Path(config['output'])
    if args.stage == 'smoothness':
        root = root.with_name(root.name+'-smoothness')
    output = root/'technical/executions'/f'{args.stage}-{args.variant or args.mode or "all"}'
    with tracked_run(output, kind='research', configs=[Path(args.config)],
                     command=[sys.executable, *sys.argv],
                     question='Does complete context remove MACE membership jumps without losing topology information?'):
        dispatch(args, config)


def dispatch(args, config):
    if args.stage == 'static-export':
        from .static import export
        export(config)
    elif args.stage == 'static-verify':
        from .static import verify
        verify(config)
    elif args.stage == 'recovery-readouts':
        from .recovery_readouts import run_cached
        run_cached(config)
    elif args.stage == 'recovery-verify':
        from .recovery_train import verify_joint
        verify_joint(config)
    elif args.stage == 'recovery-train':
        from .recovery_train import train_joint
        train_joint(config, args.variant)
    elif args.stage == 'recovery-summarize':
        from .recovery_readouts import summarize_recovery
        summarize_recovery(config)
    elif args.stage == 'prepare':
        from .data import prepare
        prepare(config)
    elif args.stage == 'verify':
        from .verify import verify
        verify(config)
    elif args.stage == 'frozen':
        from .evaluate import extract
        extract(config, args.mode)
    elif args.stage == 'train':
        from .train import train
        train(config, args.mode)
    elif args.stage == 'labels':
        from .labels import label_crossings
        label_crossings(config)
    elif args.stage == 'smoothness':
        from .smoothness import summarize_smoothness
        summarize_smoothness(config)
    else:
        from .evaluate import summarize
        summarize(config)


if __name__ == '__main__':
    main()
