"""Exercise the original Lightning VICRegModule on real normalized 80-point views."""
import json
import argparse
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch

from src.data_utils.spatiotemporal_views import SpatiotemporalViewDataModule
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule


class Measure(pl.Callback):
    def __init__(self):
        self.steps = []

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        torch.cuda.synchronize()
        self.started = time.perf_counter()

    def on_before_optimizer_step(self, trainer, module, optimizer):
        self.gradient_norms = {}
        for name, part in (("encoder", module.encoder), ("projector", module.vicreg.projector)):
            grads = [p.grad for p in part.parameters() if p.grad is not None]
            norm = torch.stack([g.float().norm().square() for g in grads]).sum().sqrt()
            assert torch.isfinite(norm) and norm > 0, (name, norm)
            self.gradient_norms[name] = float(norm)
        if module.tda_head is not None:
            norm = torch.stack([p.grad.float().norm().square() for p in module.tda_head.parameters()]).sum().sqrt()
            assert torch.isfinite(norm) and norm > 0, ('tda_head', norm)
            self.gradient_norms['tda_head'] = float(norm)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        row = dict(step=trainer.global_step, seconds=time.perf_counter()-self.started,
                   loss=float(outputs['loss']), gradient_norms=self.gradient_norms)
        assert np.isfinite(row['loss']), row
        self.steps.append(row)
        print(json.dumps(row), flush=True)


def main(config, output):
    cfg = OmegaConf.load(config)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(cfg.seed_everything, workers=True)
    dm = SpatiotemporalViewDataModule(cfg)
    dm.setup('fit')
    item = dm.train_dataset[0]
    keys = ('points', 'spatial_points', 'temporal_points')
    expected_keys = set(keys)
    if OmegaConf.select(cfg, 'tda.enabled', default=False):
        expected_keys.add('tda_targets')
        assert item['tda_targets'].shape == (3, cfg.tda.components)
    assert set(item) == expected_keys
    assert all(item[k].shape == (80, 3) and item[k].dtype == torch.float32 for k in keys)
    model = VICRegModule(cfg).cuda()
    model.eval()
    points = torch.stack([dm.train_dataset[i]['points'] for i in range(8)]).cuda()
    with torch.no_grad():
        features = model.encoder(points)
        expected = model.encoder.mace.raw_features(points*cfg.encoder.kwargs.reference_radius_A,
                                                    torch.zeros(8, dtype=torch.long, device='cuda'))
        # CUDA scatter reductions differ by a few FP32 ulps between calls.
        torch.testing.assert_close(features, expected, rtol=1e-5, atol=3e-7)
        # Identical normalized geometry has identical features: the public API has no material argument.
        torch.testing.assert_close(model.encoder(points.clone()), features, rtol=1e-5, atol=3e-7)
        views = [torch.stack([dm.train_dataset[i][key] for i in range(8)]).cuda() for key in keys]
        encoded = model.encoder(torch.cat(views)).chunk(3)
        loss, _, _ = model.vicreg.compute_spatiotemporal_loss(features=encoded, temporal_weight=1.)
        projected = [model.vicreg.project_features(x) for x in encoded]
        reference = .5*(model.vicreg._loss(projected[0], projected[1])[0] +
                        model.vicreg._loss(projected[0], projected[2])[0])
        torch.testing.assert_close(loss, reference, rtol=0, atol=0)
    del views, encoded, projected, features, expected
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    callback = Measure()
    trainer = pl.Trainer(accelerator='gpu', devices=[0], precision=cfg.precision,
                         max_epochs=1, limit_train_batches=8, limit_val_batches=1,
                         num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False,
                         gradient_clip_val=cfg.gradient_clip_val, callbacks=[callback])
    trainer.fit(model, dm)
    report = dict(train_neighborhood_triplets=len(dm.train_dataset),
                  validation_neighborhood_triplets=len(dm.val_dataset), batch_size=cfg.batch_size,
                  trainable_encoder_parameters=sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
                  encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
                  trainable_projector_parameters=sum(p.numel() for p in model.vicreg.projector.parameters() if p.requires_grad),
                  peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                  median_warm_step_seconds=float(np.median([s['seconds'] for s in callback.steps[2:]])),
                  steps=callback.steps,
                  validation={k:float(v) for k,v in trainer.callback_metrics.items() if '/val' in k or k.startswith('val/')})
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/vicreg_pretrained_mace_geometry.yaml')
    parser.add_argument('--output', default='output/mace_original_vicreg_20260909/preflight')
    args = parser.parse_args()
    main(args.config, args.output)
