"""Full-cell PTM classifier and explicitly audited cache identity."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from src.experiment_runner.artifacts import file_hash

def full_labels(points, box, cutoff):
    from ovito.data import DataCollection, Particles, SimulationCell
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    data = DataCollection()
    particles = Particles(count=len(points))
    particles.create_property('Position', data=points)
    data.objects.append(particles)
    cell = SimulationCell(pbc=(True, True, True))
    cell[...] = np.column_stack((np.diag(box), np.zeros(3)))
    data.objects.append(cell)
    data.apply(PolyhedralTemplateMatchingModifier(rmsd_cutoff=cutoff))
    return np.asarray(data.particles['Structure Type']).astype(np.uint8)


@lru_cache(maxsize=1)
def producer_identity():
    """Reuse known equivalent labels, while unknown implementations get new keys."""
    actual = file_hash(Path(__file__))
    code_root = Path(__file__).resolve().parents[3]
    contract = json.loads((code_root / 'configs/postprocessing/producer_equivalence.json').read_text())
    return contract['ptm'].get(actual, actual)
