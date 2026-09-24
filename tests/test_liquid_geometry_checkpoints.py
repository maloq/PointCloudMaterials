"""Scientific invariants of the historical checkpoint assay (no GPU required)."""
import numpy as np
import pytest

from src.research.liquid_geometry.checkpoints import sampled_rows, summarize


def _manifest():
    sources, shards = [], []
    for i, split in enumerate(("train", "train", "selection", "selection")):
        sid = f"native_{i}"
        sources.append(dict(id=sid, lineage=f"root_{i}"))
        shards.append(dict(source=sid, material="Al", static=False, anchors=12,
                           task=dict(split=split)))
    # Static Al and other elements cannot enter the supposedly native-Al assay.
    shards.extend([dict(source="static", material="Al", static=True, anchors=100,
                        task=dict(split="train")),
                   dict(source="mg", material="Mg", static=False, anchors=100,
                        task=dict(split="train"))])
    return dict(sources=sources, shards=shards)


def test_sampling_is_outcome_blind_and_balanced():
    manifest = _manifest()
    first = sampled_rows(manifest, 10, 8, 20260922)
    # Outcome fields cannot affect sampling, including post-hoc pseudo-labels.
    for shard in manifest["shards"]:
        shard["unobserved_event_label"] = 1
    assert first == sampled_rows(manifest, 10, 8, 20260922)
    assert len(first["train"]) == 10
    assert len(first["selection"]) == 8
    assert sum(i < 12 for i in first["train"]) == 5
    assert max(first["train"]) < 24
    assert min(first["selection"]) >= 24
    assert max(first["selection"]) < 48


def test_same_ancestry_cannot_cross_probe_roles():
    manifest = _manifest()
    manifest["sources"][2]["lineage"] = "root_0"
    with pytest.raises(ValueError, match="simulation ancestry"):
        sampled_rows(manifest, 10, 8, 1)


def test_drift_alone_cannot_make_constant_encoder_look_informative():
    rng = np.random.default_rng(7)
    n, train_n = 180, 120
    physical = rng.normal(size=(n, 85))
    features = dict(role=np.array(["train"] * train_n + ["selection"] * (n-train_n)),
                    source=np.repeat(np.arange(6), 30), temperature=np.full(n, 500.),
                    encoder=np.ones((n, 128)), encoder_future=np.ones((n, 128)),
                    projector=np.ones((n, 64)), projector_future=np.ones((n, 64)),
                    physical=physical, future_physical=physical.copy())
    result = summarize(features)
    for space in ("encoder", "projector"):
        assert result[space]["same_atom_drift_squared"] == 0
        assert result[space]["normalized_same_atom_drift"] is None
        assert result[space]["selection_rank"]["rank"] == 0
        for metric in result[space]["physical"].values():
            assert abs(metric["skill"]) < 1e-12


def test_physical_information_is_measured_on_unseen_sources():
    rng = np.random.default_rng(9)
    n, train_n = 360, 240
    state = rng.normal(size=(n, 6))
    y = state @ rng.normal(size=(6, 85))
    z = state @ rng.normal(size=(6, 128))
    q = state @ rng.normal(size=(6, 64))
    features = dict(role=np.array(["train"] * train_n + ["selection"] * (n-train_n)),
                    source=np.repeat(np.arange(12), 30), temperature=np.full(n, 500.),
                    encoder=z, encoder_future=z, projector=q, projector_future=q,
                    physical=y, future_physical=y)
    measured = summarize(features)
    shuffled = dict(features, physical=np.array(y))
    shuffled["physical"][train_n:] = shuffled["physical"][train_n:][rng.permutation(n-train_n)]
    broken = summarize(shuffled)
    for space in ("encoder", "projector"):
        assert measured[space]["physical"]["angular"]["skill"] > .99
        assert broken[space]["physical"]["angular"]["skill"] < .1
