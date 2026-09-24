"""Disposable production-batch correctness gate; never a scientific fit."""
import json
from pathlib import Path

import torch

from src.data.structural_pretraining.batches import move
from src.research.encoder_screen.common import sha, write
from src.training_methods.neighborhood_jepa.execution import loader, training_step, prime_encoder
from src.training_methods.shared_pretraining.compilation import compile_encoder
from .data import Data, PassBatches
from .train import build, calibrate


def run(c, config_path):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = Data(c)
    items = [i for i in c['fits'] if i['seed']==c['fits'][0]['seed']]
    packed, target = next(iter(loader(data, PassBatches(data.train, c['batch_size'], items[0]['seed'], 0, 1), c['microbatch'])))
    reference, receipts = None, []
    for item in items:
        model, objective = build(c, item, data)
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if reference is None:
            reference = initial
        else:
            for k, v in reference.items():
                torch.testing.assert_close(initial[k], v, rtol=0, atol=0)
        if c['compile']:
            compile_encoder(model.encoder, move(packed[0], 'cuda'), c['precision'])
            prime_encoder(model.encoder, packed[0], c['precision'])
        calibrate(model, objective, data, c, item['seed'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=1e-4)
        losses = []
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss, terms, diagnostics = training_step(model, objective, packed, move(target, 'cuda'),
                c['precision'], diagnose=step==0, retain_chunks=0, gpu_cache=True)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            if float(norm) <= 0:
                raise ValueError(f'No encoder learning signal for {item["name"]}')
            optimizer.step()
            losses.append(dict(loss=loss, terms=terms, gradient_norm=float(norm), diagnostics=diagnostics))
        # Same producer/state schema as the existing neighborhood native extractor.
        from src.training_methods.neighborhood_jepa.regularization.model import Encoder
        restored = Encoder(c['encoder_channels'], 'raw').cuda()
        restored.load_state_dict(model.encoder.state_dict(), strict=True)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=c['precision']=='bf16'):
            encoded = restored(move(packed[0], 'cuda'))
        if encoded.shape != (c['microbatch'], 248) or not torch.isfinite(encoded).all():
            raise ValueError(f'Invalid native inference export: {encoded.shape}')
        receipts.append(dict(name=item['name'], losses=losses, native_shape=list(encoded.shape)))
        del model, restored, objective, optimizer, encoded
        torch.cuda.empty_cache()
    write(Path(c['output'])/'technical/preflight.json', dict(passed=True, config_sha256=sha(config_path),
        identical_initial_encoders=True, production_batch=c['batch_size'], train_anchors=data.train_size,
        gpu=torch.cuda.get_device_name(), treatments=receipts))
    print(json.dumps(dict(passed=True, treatments=[r['name'] for r in receipts])), flush=True)
