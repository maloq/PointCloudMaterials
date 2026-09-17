"""Explicit real-workload preflight; never an implicit training benchmark."""
import copy
import cProfile
import json
from pathlib import Path
import pstats
import time

import numpy as np
import torch
from src.data.predictive_memory.prepare import write_json
from src.experiment_runner.metric_docs import write_metric_table
from src.project_runtime.paths import resolve_path
from .model_factory import build_model
from .native_data import BoundedCache, small_set_indices
from .native_preflight import seed_all
from .native_runtime import ObservationPrefetcher


def profile(config, data, kind, resume=False):
    from .backbone_v2 import identity, update, evaluate, verify_gate, before_deadline
    expected = identity(config, data, kind, 'physical_means', 'snapshot')
    expected['gate_receipt_sha256'] = verify_gate(config, data, kind)
    root = resolve_path(config['output'])/'technical'/kind/'profile'
    root.mkdir(parents=True, exist_ok=True)
    for variant in ('snapshot', 'history12'):
        path = root/f'{variant}.json'
        if path.exists():
            if not resume or json.loads(path.read_text())['identity'] != expected:
                raise ValueError(f'Existing profile requires unchanged identity and explicit resume: {path}')
            continue
        before_deadline(config)
        batch = small_set_indices(data.windows.rows)[::4]
        # The original producer is retained, including its spatial graph. Measure
        # cold CPU assembly separately so GATr does not hide this unused edge cost.
        cold = copy.copy(data.windows); cold.device = torch.device('cpu'); cold.raw = {}
        cold.frames = BoundedCache(0); cold.observations = BoundedCache(0)
        profiler = cProfile.Profile(); started = time.monotonic()
        for i in batch:
            profiler.runcall(cold.observation, i, variant)
        preparation = time.monotonic()-started
        stats = pstats.Stats(profiler)
        cpu_functions = [dict(file=file, line=line, function=name, calls=calls, self_seconds=own, cumulative_seconds=cumulative)
            for (file, line, name), (_, calls, own, cumulative, _) in stats.stats.items()
            if name in ('frame_observation', 'assemble') or 'query_pairs' in name]
        profiler.dump_stats(str(root/f'{variant}-preparation.prof'))
        del cold
        data.windows.observations = BoundedCache(data.windows.observations.maximum_bytes)
        data.windows.frames = BoundedCache(data.windows.frames.maximum_bytes)
        torch.cuda.empty_cache()
        seed_all(); model = build_model(kind, 'physical_means', variant, config).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=.0001)
        micro = config['microbatch'][kind][variant]
        timings = []
        with ObservationPrefetcher(data.windows, config['prefetch_workers']) as inputs:
            future = inputs.submit(batch, variant)
            observed = inputs.take(future, batch)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            for k in range(config['profile_updates']+2):
                before_deadline(config)
                torch.cuda.synchronize(); started = time.monotonic()
                update(model, observed, batch, data, optimizer, micro)
                torch.cuda.synchronize()
                if k >= 2:
                    timings.append(time.monotonic()-started)
            wait = inputs.wait_seconds
        peak = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        metrics, _ = evaluate(model, data, batch, micro, config)
        # Inspect the actual masked production path, including backward. No
        # assumption that a dense/causal mask necessarily uses FlashAttention.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as trace:
            update(model, observed, batch, data, optimizer, micro)
            torch.cuda.synchronize()
        names = sorted({event.name for event in trace.events() if any(token in event.name.lower()
            for token in ('attention', 'fmha', 'flash', 'cutlass', 'softmax'))})
        trace.export_chrome_trace(str(root/f'{variant}-kernels.json'))
        result = dict(identity=expected, encoder_kind=kind, variant=variant,
            gpu=torch.cuda.get_device_name(), precision='float32', effective_batch=8, microbatch=micro,
            rows=batch, observation_shapes=[list(o.weights.shape) for o in observed],
            parameters=sum(p.numel() for p in model.parameters()),
            update_seconds=timings, windows_per_second=8/float(np.mean(timings)),
            cold_preparation_seconds=preparation, preparation_functions=cpu_functions,
            input_wait_seconds=wait, validation_seconds=metrics['validation_seconds'],
            validation_windows_per_second=8/metrics['validation_seconds'],
            validation_input_wait_seconds=metrics['input_wait_seconds'],
            actual_attention_kernels=names, **peak)
        write_json(path, result)
        write_metric_table({k: v for k, v in result.items() if k != 'identity'}, resolve_path(config['output']),
            family='local_predictability_backbone_v2', name=f'{kind}_{variant}_profile')
        print(json.dumps(dict(encoder=kind, stage='profile', variant=variant,
            windows_per_second=result['windows_per_second'], **peak)), flush=True)
        del observed, model, optimizer; torch.cuda.empty_cache()
