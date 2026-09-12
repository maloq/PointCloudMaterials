"""Compare retained checkpoint embeddings with compiled and eager radial layers."""
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from src.analysis.config import _apply_analysis_inference_overrides
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule


def main():
    out = Path('output/mace_original_vicreg_20260909/analysis_recovery')
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path('output/mace_original_vicreg_20260909/train/MACE-pretrained-original-VICReg-normalized-no-elements-epoch=13.ckpt')
    cfg = OmegaConf.load(checkpoint.parent/'.hydra/config.yaml')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    views = np.load(Path(cfg.data.cache_dir)/'Al_175ps_val.views.npy', mmap_mode='r')
    x = torch.tensor(views[:31, 0].astype(np.float32), device='cuda')
    compiled = VICRegModule(cfg).cuda().eval()
    compiled.load_state_dict(state, strict=True)
    with torch.inference_mode():
        reference = compiled(x)[0]
    cfg.data.kind = 'static'
    _apply_analysis_inference_overrides(cfg)
    eager = VICRegModule(cfg).cuda().eval()
    eager.load_state_dict(state, strict=True)
    assert all(layer.conv_tp_weights._compiled_call_impl is None for layer in eager.encoder.mace.backbone.interactions)
    with torch.inference_mode():
        actual = eager(x)[0]
        torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-5)
        errors = {}
        for count in (1, 2, 5, 7, 16, 31):
            z = eager(x[:count])[0]
            torch.testing.assert_close(z, actual[:count], rtol=2e-4, atol=2e-5)
            errors[count] = float((z-actual[:count]).abs().max())
    report = dict(checkpoint=str(checkpoint), compiled_eager_max_abs_error=float((reference-actual).abs().max()),
                  variable_batch_max_abs_errors=errors, output_shape=list(actual.shape))
    (out/'inference_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
