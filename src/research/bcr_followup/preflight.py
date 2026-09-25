"""Bounded correctness checks, separate from the scientific training queue."""
import copy
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from src.project_runtime.paths import resolve_path
from src.training_methods.bcr.data import pack
from src.training_methods.bcr.objective import per_environment
from src.training_methods.bcr.probes import descriptors
from .common import Study, fixed_batch, decode, write_json, file_hash
from .decoders import train_decoder
from .relaxed import freeze_selection, prepare


def run(config, device):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    study = Study(config)
    receipt = study.technical/'preflight.json'
    write_json(receipt, dict(passed=False, identity=study.identity, state='checking'))
    model = study.model(1000, device)
    codes = np.load(study.original/'technical/evaluations/001000/bcr-features.npy')
    indices = study.chosen[:16]
    patches = [study.patches[i] for i in indices]
    with torch.no_grad():
        pooled = model.encoder.pooled(pack(patches, device))
        z = model.encoder.readout(pooled)
        np.testing.assert_allclose(z.cpu().numpy(), codes[indices], atol=3e-5, rtol=2e-4)
        clean, noisy, epsilon, sigma = fixed_batch(patches, 0, 16, 3, 0,
            study.manifest['noise_levels'], study.manifest['d0'], device)
        prediction = decode(model.decoder, noisy, z, sigma, study.manifest['d0'])
        nmse = per_environment(prediction, epsilon, clean, study.manifest['radius_A'])
        assert torch.isfinite(nmse).all()
        # Replay both draws before comparison with the published per-anchor mean.
        clean, noisy, epsilon, sigma = fixed_batch(patches, 0, 16, 3, 1,
            study.manifest['noise_levels'], study.manifest['d0'], device)
        prediction = decode(model.decoder, noisy, z, sigma, study.manifest['d0'])
        nmse = (nmse+per_environment(prediction, epsilon, clean, study.manifest['radius_A']))/2
    original = json.loads((study.original/'technical/evaluations/001000/bcr-reconstruction.json').read_text())
    np.testing.assert_allclose(nmse.cpu().numpy(), original['levels']['0.08']['per_anchor_nmse'][:16], atol=3e-5, rtol=2e-4)
    # Full production architecture/batch; disposable decoder updates never enter a fit.
    initial = study.model(0, 'cpu').decoder.state_dict()
    before = copy.deepcopy(model.encoder.state_dict())
    with tempfile.TemporaryDirectory(prefix='preflight-decoder-', dir=study.technical) as scratch:
        train_decoder(model, initial, codes, study.patches, study.records, study.split['train'],
            study.manifest['noise_levels'], study.config['decoder'], study.config['seed'],
            Path(scratch), study.identity, device, stop_after=3)
    for key, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    selected = freeze_selection(study)
    cell = prepare(study, max_cells=1)[0]
    paired_file = resolve_path(study.config['relaxed']['cache'])/'cells'/cell['key']/'patches.npz'
    patch_counts = {}
    with np.load(paired_file) as data:
        for domain in ('observed', 'relaxed'):
            positions, offsets = data[domain+'_positions'], data[domain+'_offsets']
            paired = [positions[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]
            target, covariates = descriptors(paired, study.manifest['radius_A'])
            if not all(np.isfinite(y).all() for y in [*target.values(), covariates]):
                raise ValueError(f'Nonfinite relaxed structural targets: {cell["key"]}/{domain}')
            with torch.no_grad():
                if not torch.isfinite(model.encode(pack(paired, device))).all():
                    raise ValueError(f'Nonfinite paired features: {cell["key"]}/{domain}')
            patch_counts[domain] = np.diff(offsets).tolist()
    write_json(receipt, dict(passed=True, identity=study.identity, finished=time.time(),
        device=str(device), feature_replay=True, corruption_replay=True,
        full_batch_decoder_updates=3, encoder_unchanged=True,
        relaxed_roots=len(selected['sources']), relaxed_cells=len(selected['cells']),
        paired_cell=cell['key'], paired_atom_counts=patch_counts,
        relaxed_selection_sha256=file_hash(study.technical/'relaxed-selection.json')))
    print(json.dumps(json.loads(receipt.read_text()), indent=2), flush=True)
