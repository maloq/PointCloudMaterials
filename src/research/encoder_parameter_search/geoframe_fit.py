"""Full-pass GeoFrame factorial with common batches and explicit optimizer counts."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from src.research.encoder_screen.common import write, sha
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
from src.training_methods.trainer import train_model
from src.training_methods.shared_pretraining.queue import deadline_for_job


class Record(pl.Callback):
    def __init__(self, root, passes, seed, deadline):
        self.root, self.passes, self.seed, self.deadline = root, passes, seed, deadline
        self.started = time.monotonic()
        self.expired = False

    def on_fit_start(self, trainer, model):
        dm = trainer.datamodule
        if len(dm.train_dataset) != 1009002 or dm.batch_size != 16384:
            raise ValueError('Historical cache/split/batch coverage changed')
        original_loader = dm.train_dataloader
        def seeded_loader():
            loader = original_loader()
            loader.sampler.generator = torch.Generator().manual_seed(self.seed + 100000 * int(trainer.current_epoch))
            return loader
        dm.train_dataloader = seeded_loader
        state = {k: v.detach().cpu().contiguous().numpy().tobytes()
                 for k, v in model.encoder.state_dict().items()}
        write(self.root/'coverage.json', dict(train_samples=len(dm.train_dataset),
            validation_samples=len(dm.val_dataset), batches_per_pass=61, batch_size=dm.batch_size,
            requested_passes=self.passes, seed=self.seed,
            training_indices_sha256=hashlib.sha256(np.asarray(dm.train_dataset.indices, dtype=np.int64).tobytes()).hexdigest(),
            initial_encoder_sha256=hashlib.sha256(b''.join(state[k] for k in sorted(state))).hexdigest()))
        if trainer.ckpt_path is None:
            trainer.save_checkpoint(self.root/'initial.ckpt')

    def on_train_epoch_start(self, trainer, model):
        # An explicit sampler generator prevents head/discriminator initialization
        # from changing the per-epoch permutation across factorial treatments.
        seed = self.seed + 100000 * int(trainer.current_epoch)
        trainer.train_dataloader.sampler.generator = torch.Generator().manual_seed(seed)

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        seed = self.seed + 100000 * int(trainer.current_epoch) + int(batch_idx) + 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def on_train_epoch_end(self, trainer, model):
        passes = int(trainer.current_epoch)+1
        updates = int(model.factor_vae_model_update_count) if model.factor_vae.enabled else int(trainer.global_step)
        if updates != passes*61:
            raise ValueError(f'Incomplete pass: {passes=}, {updates=}')
        scalars = {k: float(v.detach().cpu()) for k,v in trainer.callback_metrics.items()
                   if isinstance(v,torch.Tensor) and v.numel()==1}
        write(self.root/f'epoch-{passes-1:03d}.json',dict(completed_passes=passes,
            model_updates=updates,lightning_global_step=int(trainer.global_step),
            elapsed_seconds=time.monotonic()-self.started,metrics=scalars))
        self.expired = time.time() > self.deadline-180
        if passes >= self.passes or self.expired:
            trainer.should_stop = True
        # Seed before the next iterator/prefetch is constructed as well.
        trainer.train_dataloader.sampler.generator = torch.Generator().manual_seed(self.seed + 100000 * passes)


def fit(config, output, passes):
    if passes < 12:
        raise ValueError('At least twelve complete passes are required')
    cfg=OmegaConf.load(config); root=Path(output)/'technical/training';root.mkdir(parents=True,exist_ok=True)
    if (root/'complete.json').exists():
        receipt=json.loads((root/'complete.json').read_text())
        if receipt['config_sha256'] != sha(config): raise ValueError('Completed training config changed')
        return
    final=root/f'epoch-{passes-1:03d}.ckpt'
    # Resume only periodic end-of-epoch checkpoints (Lightning's terminal last
    # checkpoint advances its epoch counter in the historical trainer).
    retained=sorted(root.glob('epoch-*.ckpt'))
    if retained: cfg.resume_from_checkpoint=str(retained[-1].resolve())
    OmegaConf.save(cfg,root/'config.yaml')
    record=Record(root,passes,int(cfg.seed_everything),deadline_for_job())
    periodic=ModelCheckpoint(dirpath=root,monitor=None,save_top_k=-1,filename='epoch-{epoch:03d}',
        auto_insert_metric_name=False,every_n_epochs=1,save_on_train_epoch_end=True,save_last=False)
    write(root/'status.json',dict(state='training',requested_passes=passes))
    if not final.exists():
        train_model(cfg,VICRegModule,run_dir=str(root),checkpoint_callbacks=[record,periodic],run_test=False)
    if not final.exists():
        write(root/'status.json',dict(state='checkpointed',reason='allocation deadline'))
        raise SystemExit(75)
    saved=torch.load(final,map_location='cpu',weights_only=False)
    if saved['epoch'] != passes-1: raise ValueError('Final epoch identity mismatch')
    write(root/'complete.json',dict(state='complete',passes=passes,model_updates=passes*61,
        checkpoint=str(final.resolve()),checkpoint_sha256=sha(final),config_sha256=sha(config)))
    write(root/'status.json',dict(state='complete',passes=passes))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--passes',type=int,default=35);a=p.parse_args();fit(a.config,a.output,a.passes)
