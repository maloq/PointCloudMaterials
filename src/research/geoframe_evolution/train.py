"""Reuse the original trainer, retaining each epoch without changing its LR clock."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
from src.training_methods.trainer import train_model


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class EpochRecord(pl.Callback):
    def __init__(self, root, passes):
        self.root, self.passes = root, passes
        self.started = time.monotonic()

    def on_fit_start(self, trainer, model):
        dm = trainer.datamodule
        count = len(dm.train_dataset)
        batches = count // dm.batch_size
        if dm.batch_size != 16384 or count != 1009002 or batches != 61:
            raise RuntimeError(f'Historical dataset coverage changed: {count=}, {dm.batch_size=}, {batches=}')
        write_json(self.root / 'coverage.json', dict(train_samples=count,
            validation_samples=len(dm.val_dataset), batch_size=dm.batch_size,
            batches_per_epoch=batches, samples_per_pass=batches*dm.batch_size,
            drop_last_samples=count-batches*dm.batch_size, requested_passes=self.passes,
            scheduler_epochs=int(model.hparams.epochs), seed=int(model.hparams.seed_everything)))
        if trainer.ckpt_path is None:
            trainer.save_checkpoint(self.root / 'initial.ckpt')
        write_json(self.root / 'status.json', dict(state='running', requested_passes=self.passes))

    def on_train_epoch_end(self, trainer, model):
        epoch = int(trainer.current_epoch)
        scalars = {k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items()
                   if isinstance(v, torch.Tensor) and v.numel() == 1}
        write_json(self.root / f'epoch-{epoch:03d}.json', dict(epoch=epoch,
            completed_passes=epoch+1, lightning_global_step=int(trainer.global_step),
            model_updates=int(model.factor_vae_model_update_count),
            factor_gamma=model.factor_vae.effective_gamma(current_epoch=epoch),
            elapsed_seconds=time.monotonic()-self.started, metrics=scalars))
        if epoch+1 >= self.passes:
            trainer.should_stop = True


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--passes', type=int, default=35)
    parser.add_argument('--resume')
    args = parser.parse_args()
    if not 12 <= args.passes <= 160:
        raise ValueError('This protocol requires 12–160 complete training passes.')
    root = Path(args.output).resolve() / 'technical/training'
    root.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    if args.resume:
        cfg.resume_from_checkpoint = str(Path(args.resume).resolve())
    OmegaConf.save(cfg, root / 'config.yaml')
    checkpoint = ModelCheckpoint(dirpath=root, monitor=None, save_top_k=-1,
        filename='epoch-{epoch:03d}', auto_insert_metric_name=False,
        every_n_epochs=1, save_on_train_epoch_end=True, save_last=False)
    source = Path(__file__).resolve().parents[2]
    write_json(root / 'source-hashes.json', {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source.rglob('*.py'))})
    try:
        trainer, model, _, _ = train_model(cfg, VICRegModule, run_dir=str(root),
            checkpoint_callbacks=[EpochRecord(root, args.passes), checkpoint], run_test=False)
        final = torch.load(root / 'last.ckpt', map_location='cpu', weights_only=False)
        # Lightning advances current_epoch after fit; last.ckpt may refer to the
        # next epoch, whereas the periodic checkpoint stores the completed epoch.
        periodic = torch.load(root / f'epoch-{args.passes-1:03d}.ckpt', map_location='cpu', weights_only=False)
        if periodic['epoch'] != args.passes-1 or final['global_step'] != periodic['global_step']:
            raise RuntimeError('Final checkpoint does not match the last completed training epoch.')
        if int(final['state_dict']['factor_vae_model_update_count']) != args.passes*61:
            raise RuntimeError('Final model update count does not match full dataset coverage.')
        write_json(root / 'status.json', dict(state='complete', completed_passes=args.passes,
                   epoch=final['epoch'], global_step=final['global_step']))
    except BaseException as exc:
        write_json(root / 'status.json', dict(state='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
