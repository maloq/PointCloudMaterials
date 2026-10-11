"""Reuse reviewed Ni numerical audits across seeds, campaigns and allocations."""
import json
from pathlib import Path
from src.experiment_runner.artifacts import file_hash as sha, write_json
from src.project_runtime.paths import REPO, resolve_path


CATALOG = 'configs/simulation/ni_births_validation.json'
CODE_ROOT = Path(__file__).resolve().parents[4]


def verify(c, proof, *, code_root=CODE_ROOT, gpu=None):
    """Check recorded evidence without starting LAMMPS or a CUDA context."""
    if any(c[key] != value for key, value in proof['settings'].items()):
        raise ValueError('Ni numerical-validation settings changed; review or validate the new protocol')
    if sha(resolve_path(c['potential'])) != proof['settings']['potential_sha256']:
        raise ValueError('Validated Ni potential changed')
    for name, expected in proof['implementation_sha256'].items():
        actual = sha(Path(code_root)/name)
        if actual != expected:
            review = proof.get('implementation_equivalences', {}).get(name)
            if (review is None or review['validated_sha256'] != expected
                    or review['compatible_sha256'] != actual):
                raise ValueError(f'Validated Ni implementation changed: {name}')
            if sha(resolve_path(review['review'])) != review['review_sha256']:
                raise ValueError(f'Ni implementation-equivalence review changed: {name}')
    source = resolve_path(proof['validation'])
    if sha(source) != proof['validation_sha256']:
        raise ValueError(f'Ni numerical-validation receipt changed: {source}')
    result = json.loads(source.read_text())
    if (result['state'] != 'passed' or result['atoms'] != c['atoms']
            or result.get('observation_chunk_cleanup_passed') is not True
            or any(not result['metrics'][key] <= limit for key, limit in result['limits'].items())):
        raise ValueError(f'Ni numerical/publication validation did not pass: {source}')
    for name, expected in result['implementation_sha256'].items():
        if proof['implementation_sha256'][name] != expected:
            raise ValueError(f'Ni validation evidence disagrees about {name}')
    if gpu is not None and result['gpu'] != gpu:
        raise ValueError(f'Ni validation is for {result["gpu"]}, not {gpu}')
    return result


def reuse(c, *, code_root=CODE_ROOT, gpu=None, destination=None):
    catalog = json.loads((REPO/CATALOG).read_text())
    if catalog['schema_version'] != 1:
        raise ValueError('Unsupported Ni numerical-validation catalogue')
    failures = []
    for entry in catalog['audits']:
        proof = dict(entry, settings=catalog['settings'],
                     implementation_sha256=catalog['implementation_sha256'],
                     implementation_equivalences=catalog.get('implementation_equivalences', {}))
        try:
            result = verify(c, proof, code_root=code_root, gpu=gpu)
        except ValueError as error:
            failures.append(str(error))
            continue
        proof.update(state='reused', validated_gpu=result['gpu'], new_numerical_execution=False,
                     execution_implementation_sha256={name: sha(Path(code_root)/name)
                         for name in proof['implementation_sha256']})
        if destination is not None:
            write_json(destination, proof)
        return proof
    raise ValueError('No compatible reviewed Ni numerical validation. ' + '; '.join(failures))
