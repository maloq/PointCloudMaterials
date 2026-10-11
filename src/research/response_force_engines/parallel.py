"""Persistent CPU replicas for lower-latency common-noise response queries."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
import torch

from src.experiment_runner.metric_docs import write_metric_rows
from src.research.local_response.common import read, sha, write_json
from .benchmark import paths, error, publish_payload
from .engines import make_engine
from .oracle import ForceOracle, FrozenFeatures

_WORKER = None


def initialize_worker(spec, q, half_box, feature_payload, oracle_config, noises):
    global _WORKER
    torch.set_num_threads(1)
    engine = make_engine(spec, q, half_box)
    engine(q)
    _WORKER = (ForceOracle(engine, FrozenFeatures(feature_payload), oracle_config), noises)


def path_task(task):
    q, noise_index = task
    oracle, noises = _WORKER
    return os.getpid(), oracle.path_value(q, noises[noise_index]).numpy()


class ParallelForceOracle:
    """Pool owns one persistent engine per worker; all replicas share each tape."""
    def __init__(self, spec, q, half_box, features, oracle_config, noises, workers):
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
            initializer=initialize_worker,
            initargs=(spec, q, half_box, features, oracle_config, noises))

    def query(self, q, basis, noise_index, epsilon):
        points = [q]
        for j in range(basis.shape[-1]):
            points.extend([q + epsilon*basis[..., j], q - epsilon*basis[..., j]])
        start = time.monotonic()
        outputs = list(self.pool.map(path_task, [(p, noise_index) for p in points], chunksize=1))
        values = [torch.from_numpy(value) for _, value in outputs]
        response = torch.stack([(values[1+2*j] - values[2+2*j]) / (2*epsilon)
                                for j in range(basis.shape[-1])], -1)
        return dict(values=values[0], responses=response, seconds=time.monotonic()-start,
                    worker_pids=sorted(set(pid for pid, _ in outputs)))

    def close(self):
        self.pool.shutdown(wait=True)


def run(c, name, workers):
    out, _, archive = paths(c)
    binding = read(out / 'technical/binding.json')
    inputs = torch.load(archive / 'inputs.pt', weights_only=False)
    if sha(archive / 'inputs.pt') != binding['inputs_sha256']:
        raise ValueError('Prepared inputs changed')
    # This extension is a hardware benchmark on the first train parent. The
    # serial multi-parent numerical campaign remains the acceptance gate.
    state = inputs['states'][0]
    radius, epsilon = 24, 0.01
    mask = state['q'].norm(dim=-1) < radius
    q = state['q'][mask].numpy().copy()
    basis = np.zeros((*q.shape, 2)); basis[:80] = state['basis'].numpy()
    reference = torch.load(out / 'technical/reference' / f'parent{state["index"]}-radius{radius}.pt', weights_only=False)
    noises, serials = [], []
    for shot, seed in enumerate(reference['seeds']):
        path = f'noise/parent{state["index"]}-seed{seed}.pt'
        if sha(archive / path) != binding['noise_hashes'][path]:
            raise ValueError('Prepared noise changed')
        noises.append(torch.load(archive/path, weights_only=False)[:, mask].numpy())
        serials.append(torch.load(archive/name/f'parent{state["index"]}-radius{radius}-shot{shot}.pt', weights_only=False))
    for data in serials:
        if data['identity'] != binding['identity']:
            raise ValueError('Serial response identity changed')
    oracle = ParallelForceOracle(c['engines'][name], q, radius+c['vacuum_margin_A'],
                                inputs['features'], inputs['oracle'], noises, workers)
    rows, payloads = [], []
    try:
        cold = oracle.query(q, basis, 0, epsilon)
        print(dict(stage='parallel-warmup', engine=name, workers=workers, seconds=cold['seconds']), flush=True)
        for repeat in range(3):
            for shot in range(len(noises)):
                result = oracle.query(q, basis, shot, epsilon)
                serial = serials[shot]
                ve = error(result['values'], serial['values'])
                re = error(result['responses'], serial['responses'][str(epsilon)])
                if ve['rms'] > 1e-10 or re['relative'] > 1e-6:
                    raise ValueError(f'Parallel/serial numerical disagreement: {ve}, {re}')
                reference_seconds = reference['seconds'] / len(noises)
                rows.append(dict(engine=name, workers=workers, parent=state['index'], radius_A=radius,
                    epsilon_A=epsilon, repeat=repeat, shot=shot, seconds=result['seconds'],
                    serial_seconds=serial['bundle_seconds'][str(epsilon)],
                    serial_to_parallel_ratio=serial['bundle_seconds'][str(epsilon)]/result['seconds'],
                    mace_gpu_seconds=reference_seconds,
                    mace_gpu_to_parallel_cpu_ratio=reference_seconds/result['seconds'],
                    value_rms_to_serial=ve['rms'], response_relative_to_serial=re['relative']))
                payloads.append(result)
                print(rows[-1], flush=True)
    finally:
        oracle.close()
    publish_payload(c, f'parallel/{name}-{workers}workers.pt', dict(warmup=cold, queries=payloads, rows=rows))
    write_json(out/'technical'/f'parallel-{name}-{workers}.json', dict(
        serial_identity=binding['identity'], config=c, workers=workers, hostname=os.uname().nodename,
        allocation=os.environ.get('SLURM_JOB_ID'), warmup_seconds=cold['seconds'],
        implementation={str(p): sha(p) for p in [Path(__file__), Path(__file__).with_name('engines.py'),
                         Path(__file__).with_name('oracle.py')]},
        qualification='One train parent/two existing seeds; persistent pool, three warm repeats; both serial numerical gates and different-potential caveats still apply.'))
    write_metric_rows(rows, out/'analyses/parallel-v1', family='response_force_parallel', name=f'{name}-{workers}workers')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--engine', choices=['meam', 'unep'], required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 5:
        parser.error('Use1–5 allocated CPU workers for the five paths')
    torch.set_num_threads(1)
    run(read(args.config), args.engine, args.workers)


if __name__ == '__main__':
    main()
