"""Scientific invariants of the partial-observation memory pilot."""
from dataclasses import replace
import copy
import numpy as np
import pytest
import torch
from src.data.predictive_memory.observations import assemble, frame_observation
from src.data.predictive_memory.targets import physical_packet
from src.models.encoders.predictive_memory.model import PredictiveMemoryEncoder, MaskedAttention
from e3nn import o3
from src.training_methods.predictive_memory.objective import PathHeads, joint_nll, fit_scaler


@pytest.fixture(autouse=True, scope='module')
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def arrays():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(3, 20, 3))*2
    x[:, 0] = 0
    x[:, -1] = [19, 0, 0]
    u = rng.normal(size=x.shape)
    return x, u, np.arange(20)+10


def observe(x=None, u=None, ids=None):
    old = arrays()
    x, u, ids = [b if a is None else a for a, b in zip((x, u, ids), old)]
    return assemble([frame_observation(a, b, np.ones(3)*80, ids, 10, 17, 5) for a, b in zip(x, u)],
                    [-1.5, -.75, 0], radius=17, cutoff=5)


def encoder(**kwargs):
    torch.manual_seed(12)
    return PredictiveMemoryEncoder(channels=2, output_dim=8, frame_chunk=2, **kwargs)


def test_rigid_transform_boost_permutation_and_outside_radius():
    model = encoder().eval()
    x, u, ids = arrays()
    rot, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))
    permutation = np.random.default_rng(3).permutation(len(ids))
    variants = [observe(x@rot, u@rot), observe(x+7, u+4),
                observe(x[:, permutation], u[:, permutation], ids[permutation])]
    changed = x.copy(); changed[:, -1] = [25, 0, 0]
    motion = u.copy(); motion[:, -1] = 1000
    variants.append(observe(changed, motion))
    with torch.no_grad():
        expected = model(observe())
        for observed in variants:
            torch.testing.assert_close(model(observed), expected, atol=2e-6, rtol=2e-5)
    # Atom IDs are correspondence keys, never numerical features.
    observed = observe()
    torch.testing.assert_close(model(replace(observed, atom_ids=observed.atom_ids*7)), model(observed))


def test_missing_atom_zero_influence_and_oldest_gradient():
    observed = observe()
    model = encoder().train()
    observed.positions.requires_grad_(); observed.velocities.requires_grad_()
    model(observed).square().sum().backward()
    assert (observed.positions.grad.abs().sum((1, 2)) > 0).all()
    assert (observed.velocities.grad.abs().sum((1, 2)) > 0).all()
    for module in [*model.interactions, *model.products, *model.temporal, *model.edge_motion, model.pool]:
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None) > 0
    # An atom absent in a frame cannot affect temporal values or normalization.
    attention = model.temporal[0]
    h = torch.randn(3, 4, 18)
    x = torch.randn(3, 4, 3)
    weight = torch.ones(3, 4); weight[0, -1] = 0
    y = attention(h, x, observed.offsets_ps.float(), weight)
    h2, x2 = h.clone(), x.clone(); h2[0, -1] = 50; x2[0, -1] = 50
    torch.testing.assert_close(attention(h2, x2, observed.offsets_ps.float(), weight), y)


def test_temporal_causality_and_optimized_equivariant_sum():
    torch.manual_seed(9)
    a = MaskedAttention(o3.Irreps('2x0e + 2x1o + 2x2e'), 2, 48, 5)
    h, x, times = torch.randn(3, 5, 18), torch.randn(3, 5, 3), torch.tensor([-48., -.75, 0.])
    support = torch.ones(3, 5)
    y = a(h, x, times, support)
    h2 = h.clone(); h2[-1] += 3
    x2 = x.clone(); x2[-1] += 2
    torch.testing.assert_close(a(h2, x2, times, support)[:-1], y[:-1], atol=0, rtol=0)
    # The accelerated value aggregation is exactly the explicit pairwise sum.
    weight = torch.softmax(torch.randn(3, 3, 5), 1)
    dr = (x[:, None]-x[None])/5
    sh = o3.spherical_harmonics([1, 2], dr, normalize=False, normalization='component')
    literal = (weight[..., None]*(a.value(h)[None]+a.displacement(sh))).sum(1)
    fast = torch.einsum('kjn,jnc->knc', weight, a.value(h))+a.displacement(torch.einsum('kjn,kjnd->knd', weight, sh))
    torch.testing.assert_close(fast, literal, atol=2e-6, rtol=2e-5)


def test_checkpointed_spatial_matches_outputs_and_gradients():
    checkpointed = encoder(activation_checkpoint=True).train()
    reference = copy.deepcopy(checkpointed); reference.activation_checkpoint = False
    for model in (checkpointed, reference):
        model(observe()).square().sum().backward()
    torch.testing.assert_close(checkpointed(observe()), reference(observe()), atol=0, rtol=0)
    for p, q in zip(checkpointed.parameters(), reference.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-5)


def test_target_packet_is_invariant_continuous_and_signed():
    x, u, _ = arrays()
    x, u = x[0], u[0]-u[0, 0]
    rot = o3.rand_matrix(dtype=torch.float64).numpy()
    y = physical_packet(x, u)
    np.testing.assert_allclose(physical_packet(x@rot, u@rot), y, atol=2e-6, rtol=2e-5)
    assert np.linalg.norm(physical_packet(x, -u)[96:]-y[96:]) > .001
    np.testing.assert_allclose(physical_packet(x[:-1], u[:-1]), y)
    boundary = np.r_[x, [[7., 0, 0]]]
    velocity = np.r_[u, [[1., 0, 0]]]
    boundary[-1, 0] -= 1e-4
    np.testing.assert_allclose(physical_packet(boundary, velocity), y, atol=1e-5, rtol=1e-5)


def test_joint_mixture_matches_full_covariance_and_has_gradients():
    torch.manual_seed(3)
    head = PathHeads(8, 2, components=2, rank=2)
    prediction = head(torch.randn(3, 8), torch.ones(3, 1))
    future = torch.randn(3, 2, 128)
    lowrank = joint_nll(prediction, future)
    factor = prediction['factor']
    covariance = torch.diag_embed(prediction['diagonal'])+factor@factor.transpose(-1, -2)
    logp = torch.distributions.MultivariateNormal(prediction['mean'], covariance_matrix=covariance).log_prob(future.flatten(1)[:, None])
    expected = -torch.logsumexp(prediction['logits'].log_softmax(-1)+logp, -1)/256
    torch.testing.assert_close(lowrank, expected, atol=1e-5, rtol=1e-5)
    lowrank.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in head.future.parameters())


def test_snapshot_positions_only_and_separate_target_object():
    model = encoder(use_history=False, use_velocity=False).eval()
    history = observe()
    snapshot = replace(history, positions=history.positions[-1:], velocities=history.velocities[-1:],
                       weights=history.weights[-1:], offsets_ps=history.offsets_ps[-1:], edges=history.edges[-1:])
    torch.testing.assert_close(model(snapshot), model(replace(snapshot, velocities=snapshot.velocities+10)))
    assert not {'future', 'present', 'labels', 'targets'} & vars(snapshot).keys()
    with pytest.raises(ValueError, match='snapshot'):
        model(history)


def test_train_only_scaler_reproducible():
    train = torch.arange(128.).repeat(4, 1)
    future = train[:, None].repeat(1, 2, 1)
    mean, scale = fit_scaler(train, future)
    torch.testing.assert_close(mean, train[0])
    torch.testing.assert_close(scale, torch.full((128,), 1e-4))


def test_resume_restores_next_sample_optimizer_and_randomness(tmp_path):
    from types import SimpleNamespace
    from src.training_methods.predictive_memory.train import save_checkpoint, restore_checkpoint
    torch.manual_seed(8)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = np.random.default_rng(13)
    dataset = SimpleNamespace(release_sha256='fixture-release')
    config, variant = {'steps': 3}, {'history_ps': 12.}
    def update():
        index = int(sampler.integers(10))
        x = torch.randn(3)+index
        optimizer.zero_grad(); model(x).square().sum().backward(); optimizer.step()
        return index
    update()
    save_checkpoint(tmp_path/'resume.pt', model, optimizer, 1, 2., config, variant,
                    (torch.zeros(2), torch.ones(2)), dataset, sampler)
    expected_index = update()
    expected = copy.deepcopy(model.state_dict())
    step, best, _ = restore_checkpoint(tmp_path/'resume.pt', model, optimizer, config, variant, dataset, sampler, 'cpu')
    assert (step, best) == (1, 2.) and update() == expected_index
    for key, value in expected.items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
    with pytest.raises(ValueError, match='identical scientific'):
        restore_checkpoint(tmp_path/'resume.pt', model, optimizer, {'steps': 4}, variant, dataset, sampler, 'cpu')


def test_known_hidden_velocity_memory_positive_and_full_state_negative_controls():
    # Analytic oscillator fixtures, not newly simulated physical data. Position
    # history identifies hidden velocity; it adds nothing to the full state.
    rng = np.random.default_rng(4)
    position, velocity = rng.normal(size=(2, 400))
    previous = np.cos(.4)*position-np.sin(.4)*velocity
    future = np.cos(.7)*position+np.sin(.7)*velocity
    def mse(features):
        features = np.column_stack([np.ones(400), *features])
        weights = np.linalg.lstsq(features[:200], future[:200], rcond=None)[0]
        return np.mean((features[200:]@weights-future[200:])**2)
    assert mse([position]) > .2
    assert mse([position, previous]) < 1e-25
    assert mse([position, velocity]) < 1e-25
    assert mse([position, velocity, previous]) < 1e-25


def test_windows_never_union_future_ids_or_accept_future_targets():
    from src.data.predictive_memory.windows import MemoryDataset
    sample = MemoryDataset.__new__(MemoryDataset)
    sample.history_ps = 0.
    sample.config = dict(radius_A=17., cutoff_A=5.)
    x, u, ids = arrays()
    frame = frame_observation(x[0], u[0], np.ones(3)*80, ids, 10, 17, 5)
    future_frame = copy.deepcopy(frame); future_frame['ids'] += 10000
    sample.shards = [dict(frames={0: frame, 1: future_frame}, times_ps=[0., .75])]
    sample.rows = [dict(source_index=0, anchor=0, future=torch.zeros(5, 128))]
    before = sample.observation(0)
    sample.observation.cache_clear()
    sample.rows[0]['future'] += 10000
    sample.shards[0]['frames'][1]['positions'] += 200
    after = sample.observation(0)
    assert torch.max(after.atom_ids) < 10000
    torch.testing.assert_close(before.positions, after.positions, atol=0, rtol=0)
