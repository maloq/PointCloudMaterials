"""Discarded real-GPU Lightning checks for normalized MEAM history integration."""

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
import pytorch_lightning as pl
import torch

from src.research.mace_original_vicreg.preflight import Measure
from src.data_utils.relaxed_histories import RelaxedHistoryDataModule
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule


def main(variant, output):
    overrides = []
    if variant != 'anchor':
        overrides = ['encoder.name=PretrainedMACEHistoryGeometry', 'data.input_mode=history',
            '+encoder.kwargs.frame_offsets_ps=[-3,-2.25,-1.5,-0.75,0]', f'+encoder.kwargs.fusion={variant}']
    with initialize_config_dir(version_base=None, config_dir=str(Path('configs').resolve())):
        cfg = compose(config_name='vicreg_mace_relaxed', overrides=overrides)
    pl.seed_everything(cfg.seed_everything, workers=True)
    dm = RelaxedHistoryDataModule(cfg)
    dm.setup()
    model = VICRegModule(cfg).cuda()
    callback = Measure()
    torch.cuda.reset_peak_memory_stats()
    trainer = pl.Trainer(accelerator='gpu', devices=[0], precision=cfg.precision,
        max_epochs=1, limit_train_batches=3, limit_val_batches=1, num_sanity_val_steps=0,
        logger=False, enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False,
        gradient_clip_val=cfg.gradient_clip_val, callbacks=[callback])
    trainer.fit(model, dm)
    history_gradients = None
    if variant == 'atom_temporal':
        model.cuda().eval()
        x = torch.stack([dm.train_dataset[i]['points'] for i in range(8)]).cuda().requires_grad_()
        model.tda_head(model(x)[0]).square().mean().backward()
        history_gradients = x.grad.abs().sum((0, 2, 3)).tolist()
        if not all(np.isfinite(history_gradients)) or not all(g>0 for g in history_gradients):
            raise AssertionError(f'Missing history gradients: {history_gradients}')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(state='passed', variant=variant, retained_updates=0, batch_size=cfg.batch_size,
        steps=callback.steps, peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        history_gradient_l1=history_gradients,
        trainable_backbone_parameters=sum(p.numel() for p in model.encoder.mace.parameters() if p.requires_grad))
    (output/f'{variant}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=['anchor','atom_temporal'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.variant, args.output)
