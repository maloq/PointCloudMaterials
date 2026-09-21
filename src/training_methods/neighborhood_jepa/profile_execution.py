"""Standalone execution benchmark. Never invoked inside scientific training."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import numpy as np
import torch
from src.data.structural_pretraining.prepare import save_json, file_hash
from src.data.structural_pretraining.batches import move
from src.project_runtime.paths import resolve_path
from src.training_methods.shared_pretraining.compilation import compile_encoder, compilation_counters
from .execution import MACE_FIELDS, loader, prime_encoder, stage_inputs, training_step
from .v2.runtime import training_step as reference_step
from .v2.data import Batches
from .v2.parallel import configure_host


def run(args):
    configure_host(); torch.set_num_threads(1)
    config = json.loads(Path(args.config).read_text())
    if config['protocol'] == 'neighborhood_jepa_multihorizon_v1':
        from .multihorizon.data import Data, loader as old_loader
        from .multihorizon.model import Model
        from .multihorizon.objective import Objective
        from .multihorizon.specs import variants
    else:
        from .regularization.data import Data, loader as old_loader
        from .regularization.model import Model
        from .regularization.objective import Objective
        from .regularization.specs import variants
    spec = next(s for s in variants(config) if s['regularizer'] == 'vicreg' and s['projector'] == 'mlp_ln')
    data = Data(config, spec)
    torch.manual_seed(config['seed'])
    model = Model(config['encoder_channels'], spec, config['seed']).cuda().train()
    model.initialize(torch.load(resolve_path(config['warm_checkpoint']), map_location='cpu', weights_only=False))
    model.encoder.geometry_scales.copy_(torch.tensor(data.manifest['geometry_scales'], device='cuda'))
    objective = Objective(data.manifest, data.order_manifest, spec).cuda()
    make_loader = old_loader if args.reference else loader
    # Distinct training batches after warmup expose data wait separately.
    stream = iter(make_loader(data, Batches(data, args.anchors, config['seed'], 0, args.steps+1),
                             args.microbatch, config['loader_workers']))
    started = time.perf_counter(); batch, target = next(stream)
    initial_data_seconds = time.perf_counter()-started
    compile_encoder(model.encoder, move(batch[0], 'cuda'), config['precision'])
    prime_encoder(model.encoder, batch[0], config['precision'])
    memory_estimate = None
    if args.retain_chunks:
        # Offline safety estimate, not a training-time fallback or autotuner.
        example = stage_inputs(batch[:1], 'cuda')[0]
        baseline = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config['precision']=='bf16'):
            value = model.encoder(example).float()
        saved_bytes = torch.cuda.memory_allocated()-baseline
        value.sum().backward(); torch.cuda.synchronize()
        chunk_peak = torch.cuda.max_memory_allocated()-baseline
        model.zero_grad(set_to_none=True); del value, example
        free, _ = torch.cuda.mem_get_info()
        available = free+torch.cuda.memory_reserved()-torch.cuda.memory_allocated()
        input_bytes = sum(b[k].numel()*b[k].element_size() for b in batch for k in MACE_FIELDS)
        estimate = (chunk_peak+saved_bytes*args.retain_chunks+input_bytes)*1.15
        memory_estimate = dict(saved_GiB_per_chunk=saved_bytes/2**30,chunk_peak_GiB=chunk_peak/2**30,
                               estimated_GiB=estimate/2**30,available_GiB=available/2**30)
        print(json.dumps(dict(retention_memory_preflight=memory_estimate)), flush=True)
        if estimate > available:
            raise MemoryError(f'Retention benchmark exceeds available VRAM: {memory_estimate}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    fn = reference_step if args.reference else training_step
    kwargs = {} if args.reference else dict(retain_chunks=args.retain_chunks, gpu_cache=args.gpu_cache)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    # Capture first-batch loss/gradients before any update for cross-profile parity.
    started = time.perf_counter()
    loss, terms, diagnostics = fn(model, objective, batch, move(target, 'cuda'), config['precision'], **kwargs)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter()-started
    named = [(n,p.grad) for n,p in model.named_parameters() if p.grad is not None]
    np.savez(output.with_suffix('.gradients.npz'), names=np.array([n for n,_ in named]),
             gradient=torch.cat([p.flatten() for _,p in named]).cpu().numpy(), loss=loss)
    del named
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    rows=[]
    for index in range(args.steps):
        started = time.perf_counter(); batch, target = next(stream)
        data_seconds = time.perf_counter()-started
        started = time.perf_counter()
        loss, terms, diagnostics = fn(model, objective, batch, move(target, 'cuda'), config['precision'], **kwargs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize()
        compute_seconds = time.perf_counter()-started
        row = dict(step=index, data_wait_seconds=data_seconds, update_seconds=compute_seconds,
                   end_to_end_seconds=data_seconds+compute_seconds, loss=loss)
        rows.append(row); print(json.dumps(row), flush=True)
    result = dict(gpu=torch.cuda.get_device_name(), total_memory_GiB=torch.cuda.get_device_properties(0).total_memory/2**30,
        statistical_batch=args.anchors, views=len(data.plan.views), microbatch=args.microbatch,
        retain_chunks=args.retain_chunks, gpu_cache=args.gpu_cache, reference=args.reference,
        initial_data_seconds=initial_data_seconds, compile_warmup_seconds=warmup_seconds,
        median_update_seconds=statistics.median(r['update_seconds'] for r in rows),
        median_end_to_end_seconds=statistics.median(r['end_to_end_seconds'] for r in rows),
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
        diagnostics=diagnostics, steps=rows, compilation=compilation_counters(),memory_estimate=memory_estimate,
        config=str(Path(args.config).resolve()), data_identity=data.manifest['identity'],
        producers={str(p):file_hash(p) for p in (Path(__file__),Path(__file__).with_name('execution.py'))})
    save_json(output, result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--microbatch', type=int, required=True)
    parser.add_argument('--retain-chunks', type=int, default=0)
    parser.add_argument('--gpu-cache', action='store_true'); parser.add_argument('--reference', action='store_true')
    parser.add_argument('--steps', type=int, default=3); parser.add_argument('--anchors', type=int, default=512)
    args = parser.parse_args()
    code = 0
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        code = 1
    # PyTorch loader tracker cleanup can otherwise retain a completed CUDA job.
    sys.stdout.flush(); sys.stderr.flush(); os._exit(code)
