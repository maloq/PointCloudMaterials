"""Scientific contracts of early geometry/velocity/history MACE interaction."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data_utils.causal_history import build_history, physical_windows
from src.models.encoders.mace_causal import CausalMACEEncoder, SmoothMultiscalePooling, taper
from src.training_methods.mace_causal.objective import (
    PhysicalHeads, admissible, cumulative_risk, hazard_label, hazard_nll,
    target_normalization, targets, task_objective,
)


@pytest.fixture(scope='module', autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def arrays():
    rng = np.random.default_rng(43)
    base = rng.normal(size=(18, 3))*3
    base[0] = 0
    x = np.stack([base+rng.normal(size=base.shape)*.1 for _ in range(3)])
    v = rng.normal(size=x.shape)*2
    return x, v, np.full((3, 3), 50.), np.array([0., .75, 1.5]), np.arange(18)*3+7


def graph(x=None, v=None, boxes=None, times=None, ids=None, prune=True):
    a, b, c, d, e = arrays()
    x, v, boxes, times, ids = [old if new is None else new for new, old in
                              zip((x, v, boxes, times, ids), (a, b, c, d, e), strict=True)]
    return build_history(x, v, boxes, times, ids, np.full(len(ids), 13), 7,
                         cutoff_A=5., context_radius_A=9., spatial_layers=2, prune=prune)


@pytest.fixture
def model():
    torch.manual_seed(8)
    return CausalMACEEncoder(channels=3, output_dim=8, num_bessel=3, radial_width=8).eval()


def test_early_motion_history_gradients_and_causal_node_computation(model):
    history = graph()
    history.positions.requires_grad_(); history.velocities.requires_grad_()
    z = model(history)
    z.square().sum().backward()
    assert (history.positions.grad.abs().sum((1, 2)) > 0).all()
    assert (history.velocities.grad.abs().sum((1, 2)) > 0).all()
    for module in (model.interactions[0], model.interactions[1], model.temporal[0], model.edge_motion[0], model.motion_vectors[0]):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    with torch.no_grad():
        reverse = replace(history, velocities=-history.velocities)
        assert (model(reverse)-z).abs().max() > 1e-5
        assert (model(history.repeated_anchor())-z).abs().max() > 1e-5
        changed = history.positions.detach().clone(); changed[-1, 1:] *= 1.05
        # A change in a later frame cannot change any earlier atom features.
        torch.testing.assert_close(model.atom_features(replace(history, positions=changed))[:-1],
                                   model.atom_features(history)[:-1], atol=0, rtol=0)


def test_joint_o3_translation_boost_id_permutation_and_periodic_images(model):
    x, v, box, times, ids = arrays()
    rng = np.random.default_rng(9)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation[:, 0] *= -np.linalg.det(rotation)  # Explicit improper rotation.
    permutation = rng.permutation(len(ids))
    variants = [graph(x@rotation, v@rotation), graph(x+np.array([11, -3, 9]), v+7),
                graph(x[:, permutation], v[:, permutation], ids=ids[permutation]),
                graph(np.mod(x+np.array([24, 0, 0]), box[:, None]), v)]
    with torch.no_grad():
        expected = model(graph())
        for history in variants:
            torch.testing.assert_close(model(history), expected, atol=3e-5, rtol=2e-4)


def test_ancestor_pruning_agrees_with_complete_periodic_graph(model):
    rng = np.random.default_rng(32)
    x = rng.uniform(-24, 24, (3, 150, 3)); x[:, 0] = 0
    # Motion makes a previous spatial neighbor necessary outside a fixed halo.
    x[:, 1] = [[14, 0, 0], [9, 0, 0], [4, 0, 0]]
    x[:, 2] = [[18, 0, 0], [13, 0, 0], [8, 0, 0]]
    v = rng.normal(size=x.shape)
    ids = np.arange(150)+7
    small, full = graph(x, v, ids=ids), graph(x, v, ids=ids, prune=False)
    assert len(small.atom_ids) < len(full.atom_ids)
    with torch.no_grad():
        torch.testing.assert_close(model(small), model(full), atol=3e-5, rtol=2e-4)


def test_spatial_support_has_no_unweighted_softmax_or_count_jump():
    torch.manual_seed(3)
    pool = SmoothMultiscalePooling(2, [(0., 3.)], 5)
    h = torch.randn(3, 18)
    x = torch.tensor([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
    expected = pool(h[:2], x[:2])
    torch.testing.assert_close(pool(h, x), expected, atol=1e-7, rtol=1e-7)
    x[-1, 0] -= .001
    assert (pool(h, x)-expected).abs().max() < 1e-6
    t = torch.tensor([0., 2.25], requires_grad=True)
    w = taper(t, 0, 2.25)
    torch.testing.assert_close(w, torch.tensor([1., 0.]))
    w.sum().backward(); torch.testing.assert_close(t.grad, torch.zeros_like(t))


def test_age_boundary_removes_oldest_observation_smoothly(model):
    original = graph()
    history = replace(original, offsets_ps=torch.tensor([-2.25, -.75, 0.], dtype=torch.float64))
    n = history.positions.shape[1]
    keep = history.edges[0] >= n
    shorter = replace(history, positions=history.positions[1:], velocities=history.velocities[1:],
                      boxes=history.boxes[1:], offsets_ps=history.offsets_ps[1:], edges=history.edges[:, keep]-n)
    with torch.no_grad():
        torch.testing.assert_close(model(history), model(shorter), atol=2e-6, rtol=2e-5)
        near = replace(history, offsets_ps=torch.tensor([-2.249, -.75, 0.], dtype=torch.float64))
        torch.testing.assert_close(model(near), model(shorter), atol=2e-6, rtol=2e-5)


def test_snapshot_controls_ignore_history_and_geometry_control_ignores_motion(model):
    history = graph()
    model.use_history = False; model.use_velocity = False
    with torch.no_grad():
        expected = model(history)
        changed = replace(history, velocities=history.velocities*90, positions=history.positions.clone())
        changed.positions[:-1] *= 3
        torch.testing.assert_close(model(changed), expected, atol=0, rtol=0)


def test_repeated_anchor_is_an_encoder_policy_at_inference(model):
    history = graph()
    with torch.no_grad():
        expected = model(history.repeated_anchor())
        model.repeat_anchor = True
        torch.testing.assert_close(model(history), expected, atol=0, rtol=0)


def test_learns_opposite_futures_at_identical_geometry(model):
    # Expansion/contraction have identical present positions and opposite future
    # radius changes. A geometry-only snapshot cannot distinguish this pair.
    base = graph()
    model.use_history = False
    outward = replace(base, velocities=.3*base.positions)
    inward = replace(base, velocities=-.3*base.positions)
    geometry = copy.deepcopy(model); geometry.use_velocity = False
    with torch.no_grad():
        torch.testing.assert_close(geometry(outward), geometry(inward), atol=0, rtol=0)
    torch.manual_seed(10)
    readout = torch.nn.Linear(model.invariant_dim, 1)
    optimizer = torch.optim.Adam(list(model.parameters())+list(readout.parameters()), lr=.01)
    target = torch.tensor([[1.], [-1.]])
    errors = []
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        prediction = readout(torch.cat((model(outward), model(inward))))
        loss = (prediction-target).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        errors.append(float(loss.detach()))
    assert errors[-1] < .1 and errors[-1] < errors[0]/5


def test_time_matching_rejects_unavailable_future_and_causal_contract(model):
    times = np.arange(9)*.75
    windows = physical_windows(times, [-2.25, -1.5, -.75, 0], [.75, 1.5, 3])
    assert [w[0] for w in windows] == [3, 4]
    np.testing.assert_array_equal(windows[0][1], [0, 1, 2, 3])
    with pytest.raises(ValueError, match='No exact causal windows'):
        physical_windows(times, [-1.5, 0], [9])
    with pytest.raises(ValueError, match='No exact causal windows'):
        physical_windows(times, [-1.5, 0], [.8])
    with pytest.raises(ValueError, match='causal'):
        model(replace(graph(), offsets_ps=torch.tensor([-1., 0., .5])))


def test_followup_extension_uses_actual_timeline_and_keeps_retained_frames():
    from src.training_methods.mace_causal.data import extended_record
    record = dict(source=dict(id=3, timestep_fs=3.), frames=list(range(4, 13)))
    trajectory = SimpleNamespace(timesteps=np.arange(40)*250)
    result = extended_record(record, trajectory, 6.75)
    assert result['frames'] == list(range(4, 22))
    assert record['frames'] == list(range(4, 13))
    assert extended_record(record, trajectory, 0.) == record
    with pytest.raises(ValueError, match='exact follow-up'):
        extended_record(record, trajectory, .1)
    with pytest.raises(ValueError, match='last stored time'):
        extended_record(record, trajectory, 100.)


def test_hazard_confirmation_censoring_and_observed_only_risk_set():
    t = np.arange(7)*.75
    event = hazard_label([0, 0, 0, 1, 1, 0, 0], t, 2, [.75, 1.5, 3], 2)
    assert event == dict(event_bin=0, observed_bins=1, at_risk=True)
    # Last-frame crystal cannot be confirmed: censor at the previous frame.
    censor = hazard_label([0, 0, 0, 0, 0, 0, 1], t, 2, [.75, 1.5, 3], 2)
    assert censor == dict(event_bin=-1, observed_bins=2, at_risk=True)
    prior = hazard_label([1, 1, 0, 0, 0, 0, 0], t, 2, [.75, 1.5, 3], 2)
    assert not prior['at_risk']
    logits = torch.zeros(3, 3, requires_grad=True)
    value = hazard_nll(logits, torch.tensor([1, -1, -1]), torch.tensor([2, 3, 0]))
    torch.testing.assert_close(value, torch.log(torch.tensor(2.))*torch.tensor([2., 3., 0.]))
    torch.testing.assert_close(cumulative_risk(logits), torch.tensor([[.5, .75, .875]]).expand(3, -1))
    value.sum().backward(); assert torch.isfinite(logits.grad).all()


def samples():
    rng = np.random.default_rng(11)
    base = graph()
    result = []
    for sid, split in enumerate(('train', 'train', 'val', 'test')):
        for k in range(2):
            present = rng.normal(size=169).astype(np.float32)
            present[4] = .2
            future = present[None]+rng.normal(scale=.1, size=(2, 169)).astype(np.float32)
            result.append(dict(history=base, present=present, future=future,
                path=np.array([[.2, .01], [.21, .02]], dtype=np.float32), source_id=sid,
                split=split, lineage=f'lineage{sid}', center_atom_id=7, anchor_ps=k*.75,
                temperature_K=500., event_bin=-1, observed_bins=2, at_risk=True))
    return result


def test_fixed_target_gradient_and_no_heldout_normalization_leak(model):
    data = samples()
    norm = target_normalization(data)
    edited = copy.deepcopy(data)
    for sample in edited:
        if sample['split'] != 'train':
            sample['present'] += 1e5; sample['future'] *= 1000
    for key in ('mean', 'scale'):
        np.testing.assert_array_equal(target_normalization(edited)[key], norm[key])
    heads = PhysicalHeads(8, [.75, 3], hidden=8, probabilistic=True, event_bins_ps=[.75, 3])
    batch = data[:2]; y = targets(batch, norm, 'cpu')
    z = torch.cat([model(s['history']) for s in batch])
    value, _ = task_objective(heads(z, y['temperature']), y)
    value.backward()
    assert model.interactions[1].linear.weight.grad.norm() > 0
    assert all(torch.isfinite(p.grad).all() for p in heads.parameters() if p.grad is not None)
    assert admissible({'present': 1.04, 'future': .9}, {'present': 1., 'future': 1.}, .05, 0)
    assert not admissible({'present': 1.06, 'future': .1}, {'present': 1., 'future': 1.}, .05, 0)


def test_whole_lineage_split_guard():
    from src.training_methods.mace_causal.data import validate_splits
    records = [dict(source=dict(id=i, lineage=f'lineage{i}', split=s)) for i, s in enumerate(('train', 'val', 'test'))]
    validate_splits(records)
    records[1]['source']['lineage'] = records[0]['source']['lineage']
    with pytest.raises(ValueError, match='leaks'):
        validate_splits(records)


def test_jump_definition_and_threshold_do_not_use_liquid_crystal_separation():
    from src.training_methods.mace_causal.evaluate import aggregate, jump_metrics, jump_criterion
    data = samples()[:2]
    z = np.array([[0., 0.], [1., 0.]])
    summary = aggregate(jump_metrics(z, data, .75), 10, 2)
    # Cov trace is .25, squared increment is 1: J=sqrt(2).
    result = jump_criterion(summary, .75, .10)
    assert result['value'] == pytest.approx(np.sqrt(2)) and not result['met']
    assert jump_criterion([], .75, .10)['met'] is None


def test_event_thresholds_use_validation_and_misses_stay_in_timing_denominator():
    from src.training_methods.mace_causal.events import event_report, thresholds_from_validation
    data = [dict(split='val', source_id=i, at_risk=True, event_bin=event, observed_bins=1)
            for i, event in enumerate((0, 0, -1, -1))]
    logits = np.array([[3.], [-2.], [-1.], [-3.]])
    thresholds = thresholds_from_validation(logits, data, 0.)
    result = event_report(logits, data, [.75], thresholds)[0]
    assert result['detected_events'] == 1 and result['missed_events'] == 1
    assert result['recall'] == .5 and result['timed_within_first_bin_recall'] == .5
    assert result['false_alarm_rate'] == 0.
    highest_negative = np.array([[-2.], [-2.], [3.], [-3.]], dtype=np.float32)
    no_alarm = thresholds_from_validation(highest_negative, data, 0.)
    report = event_report(highest_negative, data, [.75], no_alarm)[0]
    assert report['false_alarm_rate'] == 0. and report['recall'] == 0.
    data[0]['split'] = 'test'
    with pytest.raises(ValueError, match='validation'):
        thresholds_from_validation(logits, data, 0.)


def test_training_checkpoint_export_and_strict_inference(tmp_path, monkeypatch):
    import json
    import src.training_methods.mace_causal.train as workflow
    config = json.loads(open('configs/mace_causal/pilot.json').read())
    config['output'] = str(tmp_path/'fit')
    config['future_lags_ps'] = [.75, 3.]
    config['encoder'].update(channels=3, output_dim=8, num_bessel=3, radial_width=8)
    config['heads']['hidden'] = 8
    config['events']['bin_edges_ps'] = [.75, 3.]
    config['training'].update(steps=2, validation_every=1, log_every=1)
    config['probes'].update(steps=1, validation_every=1, hidden=8)
    config['bootstrap_draws'] = 20
    data = samples()
    monkeypatch.setattr(workflow, 'load', lambda _: (data, dict(cache_sha256='test', plan_sha256='test')))
    args = SimpleNamespace(variant='D', device='cpu', resume=False)
    initial, _ = workflow.initialize(config, 'D', 'cpu')
    workflow.run(config, args)
    root = tmp_path/'fit-D'
    saved = torch.load(root/'technical/last.pt', weights_only=False)
    assert not torch.equal(saved['encoder_state']['interactions.1.linear.weight'], initial.interactions[1].linear.weight)
    restored, _ = workflow.load_encoder(root/'technical/last.pt')
    expected, _ = workflow.initialize(config, 'D', 'cpu')
    expected.load_state_dict(saved['encoder_state'])
    torch.testing.assert_close(restored(data[0]['history']), expected(data[0]['history']), atol=0, rtol=0)
    assert (root/'tables/METRICS.md').exists()
    assert (root/'tables/test.csv').exists()
    assert json.loads((root/'technical/status.json').read_text())['state'] == 'complete'
    with pytest.raises(FileExistsError):
        workflow.run(config, args)

    # Simulate interruption after a durable checkpoint, then verify the exact
    # optimizer and source/anchor sampling continuation against uninterrupted fit.
    interrupted = copy.deepcopy(config); interrupted['output'] = str(tmp_path/'interrupted')
    original_save = workflow.atomic_save
    def interrupt_after_checkpoint(path, payload):
        original_save(path, payload)
        if path.name == 'last.pt' and payload['step'] == 1:
            raise RuntimeError('simulated interruption')
    monkeypatch.setattr(workflow, 'atomic_save', interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match='simulated'):
        workflow.run(interrupted, args)
    monkeypatch.setattr(workflow, 'atomic_save', original_save)
    workflow.run(interrupted, SimpleNamespace(variant='D', device='cpu', resume=True))
    continued = torch.load(tmp_path/'interrupted-D/technical/last.pt', weights_only=False)
    for name in saved['encoder_state']:
        torch.testing.assert_close(continued['encoder_state'][name], saved['encoder_state'][name], atol=0, rtol=0)
    for name in saved['head_state']:
        torch.testing.assert_close(continued['head_state'][name], saved['head_state'][name], atol=0, rtol=0)

    # Exercise all frozen-state probes, including the matched raw-history pair.
    import src.training_methods.mace_causal.probes as probes
    monkeypatch.setattr(probes, 'load', lambda _: (data, dict(cache_sha256='test', plan_sha256='test')))
    probes.run(config, args)
    paired = [torch.load(tmp_path/f'fit-D-probe-{mode}/technical/best.pt', weights_only=False)
              for mode in ('state_constant', 'state_history')]
    assert paired[0]['parameter_count'] == paired[1]['parameter_count']
    assert (tmp_path/'fit-D-probe-state_history/tables/test.csv').exists()


@pytest.mark.parametrize('variant', ['A', 'B', 'C', 'D', 'repeated_anchor'])
def test_packed_variable_graphs_preserve_outputs_and_parameter_gradients(model, variant):
    from src.models.encoders.mace_causal_batch import pack_histories, validate_history
    model.use_history = variant in ('D', 'repeated_anchor')
    model.use_velocity = variant not in ('A', 'B')
    model.repeat_anchor = variant == 'repeated_anchor'
    x, v, box, times, ids = arrays()
    histories = [graph(), graph(x[:, :11], -v[:, :11], boxes=box+13, ids=ids[:11])]
    for h in histories:
        validate_history(h, model)
    packed_model = copy.deepcopy(model)
    expected = torch.cat([model(h) for h in histories])
    packed = pack_histories(histories)
    n = packed.positions.shape[1]
    assert torch.equal(packed.node_graph[packed.edges[0] % n], packed.node_graph[packed.edges[1] % n])
    actual = packed_model(packed)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    # An unequal per-example objective catches accidental cross-sample reduction.
    weight = torch.tensor([1., 3.])[:, None]
    (actual.square()*weight).sum().backward()
    (expected.square()*weight).sum().backward()
    for (name, p), (_, q) in zip(model.named_parameters(), packed_model.named_parameters(), strict=True):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(q.grad, p.grad, atol=8e-6, rtol=2e-4, msg=name)
    with torch.no_grad():
        reversed_batch = packed_model(pack_histories(histories[::-1]))
        torch.testing.assert_close(reversed_batch.flip(0), actual, atol=2e-6, rtol=2e-5)


def test_packing_requires_exact_time_and_support_and_preflight_rejects_bad_edges(model):
    from src.models.encoders.mace_causal_batch import pack_histories, validate_history
    h = graph()
    with pytest.raises(ValueError, match='physical offsets'):
        pack_histories([h, replace(h, offsets_ps=h.offsets_ps*1.01)])
    with pytest.raises(ValueError, match='spatial support'):
        pack_histories([h, replace(h, cutoff_A=4.)])
    edges = h.edges.clone(); edges[0, 0] += len(h.atom_ids)
    with pytest.raises(ValueError, match='cross time'):
        validate_history(replace(h, edges=edges), model)


def test_packed_minimum_images_use_each_examples_cell(model):
    from src.models.encoders.mace_causal_batch import pack_histories
    x, v, box, times, ids = arrays()
    x[:, 1] = [8.8, 0, 0]; x[:, 2] = [-8.8, 0, 0]
    histories = [graph(x, v, np.full_like(box, 20.)), graph(x, v, box)]
    with torch.no_grad():
        expected = torch.cat([model(h) for h in histories])
        torch.testing.assert_close(model(pack_histories(histories)), expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize('residency', ['host', 'device', 'train_device'])
def test_runtime_target_bank_and_chunked_extraction_match_reference(model, residency):
    from src.training_methods.mace_causal.runtime import CausalRuntime
    from src.training_methods.mace_causal.evaluate import extract
    data = samples(); norm = target_normalization(data)
    runtime = CausalRuntime(data, norm, model, 'cpu', dict(batch_size=3, residency=residency))
    selected = [data[i] for i in (7, 0, 4, 0)]
    for key, expected in targets(selected, norm, 'cpu').items():
        torch.testing.assert_close(runtime.target(selected)[key], expected, atol=0, rtol=0)
    heads = PhysicalHeads(8, [.75, 3], hidden=8, probabilistic=True, event_bins_ps=[.75, 3])
    expected = extract(model, heads, selected, norm, 'cpu')
    actual = runtime.extract(model, heads, selected)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-6, rtol=3e-5)


def test_constant_history_reuse_preserves_branch_gradients(model):
    reference = copy.deepcopy(model)
    weights = torch.arange(24).reshape(3, 8)/24
    (model(graph()).expand(3, -1)*weights).sum().backward()
    (torch.cat([reference(graph()) for _ in range(3)])*weights).sum().backward()
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=8e-6, rtol=2e-4, msg=name)
