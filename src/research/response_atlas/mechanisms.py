"""Known-law acquisition and matched value/response learning mechanisms."""
import copy
import math
import time

import numpy as np
import torch

from src.data.fixed_cohort.protocol import write_json
from src.experiment_runner.wandb_tracking import DEFAULTS, online_training
from .common import output, table
from .reference import (AutogradOracle, BAOABOracle, SmallAtlas, random_basis,
                        response_columns, training_loss, unbiased_gram, propose_direction,
                        verify_proposal)


def gaussian(q, seed):
    gen = torch.Generator(device=q.device).manual_seed(seed)
    noise = torch.randn((), generator=gen, dtype=q.dtype, device=q.device)
    y = q[0] + torch.exp(q[1] / 2) * noise
    return torch.stack((y, y.square()))


def exact(q):
    return torch.stack((q[..., 0], q[..., 0].square() + q[..., 1].exp()), -1)


def exact_response(q):
    out = q.new_zeros((*q.shape[:-1], 2, 2))
    out[..., 0, 0] = 1
    out[..., 1, 0] = 2 * q[..., 0]
    out[..., 1, 1] = q[..., 1].exp()
    return out


class MissingVariance(torch.nn.Module):
    def forward(self, q):
        return torch.stack((q[0], q[0].square() + q.new_tensor(1.)))


def numerical(c):
    """Local numerical gate on actual reference consumers, not a test suite."""
    torch.set_num_threads(2)
    torch.manual_seed(c['seed'])
    q = torch.tensor([.3, -.4], dtype=torch.float64)
    basis = torch.eye(2, dtype=q.dtype)
    oracle = AutogradOracle(gaussian)
    bundle = oracle.query(q, basis, range(100, 108))
    replay = oracle.query(q, basis, range(100, 108))
    if not torch.equal(bundle.responses, replay.responses):
        raise ValueError('Nonreproducible explicit branch stream')
    model = SmallAtlas(2, 2).double()
    loss, _ = training_loss(model, bundle)
    grads = torch.autograd.grad(loss, tuple(model.parameters()))
    response_loss, _ = training_loss(model, bundle)
    value_loss, _ = training_loss(model, bundle, derivative_weight=0)
    rg = torch.autograd.grad(response_loss - value_loss, tuple(model.parameters()), allow_unused=True)
    grad_norm = sum(float(g.square().sum()) for g in rg if g is not None)
    if grad_norm <= 0 or not all(torch.isfinite(g).all() for g in grads):
        raise ValueError('Response parameter differentiation is broken')
    # The supplied tangent integrator must include derivatives of initial anchors.
    core = BAOABOracle(lambda x: .5 * x.square().sum(), lambda initial, path: path[-1] - initial,
        mass=torch.ones_like(q), dt=.02, steps=20, kbt=.3, friction=.7, horizon_steps=[20])
    tangent = core.query(q, basis, [300, 301])
    errors = []
    for epsilon in (.01, .001, .0001):
        for j in range(2):
            plus = core.query(q + epsilon * basis[:, j], q.new_empty(2, 0), [300, 301])
            minus = core.query(q - epsilon * basis[:, j], q.new_empty(2, 0), [300, 301])
            errors.append(float(((plus.values - minus.values) / (2 * epsilon) - tangent.responses[:, :, j]).abs().max()))
    if max(errors) > 1e-8:
        raise ValueError(f'Harmonic discrete-path tangent mismatch: {errors}')
    # Establish the branchwise/mean derivative-regression gradient identity.
    parameter = torch.randn_like(bundle.responses[0], requires_grad=True)
    a = (parameter - bundle.responses).square().mean()
    b = (parameter - bundle.responses.mean(0)).square().mean()
    ga, = torch.autograd.grad(a, parameter); gb, = torch.autograd.grad(b, parameter)
    if not torch.allclose(ga, gb, rtol=1e-12, atol=1e-12):
        raise ValueError('Branchwise regression gradient differs from mean regression')
    proposal = propose_direction(MissingVariance(), bundle)
    verified = verify_proposal(MissingVariance(), oracle, q, proposal, .001, range(200, 216),
                               discovery_seeds=bundle.seeds)
    receipt = dict(passed=True, harmonic_max_abs_error=max(errors), response_parameter_gradient_squared=grad_norm,
                   fresh_verification=verified, tracking='local numerical gate')
    write_json(output(c) / 'technical/numerical.json', receipt)
    return receipt


def acquisition(c):
    rows = []
    oracle = AutogradOracle(gaussian)
    model = MissingVariance()
    for repeat in range(c['mechanism_repeats']):
        for qid, uv in enumerate(((0., -1.), (.7, 0.), (-.4, 1.))):
            q = torch.tensor(uv, dtype=torch.float64)
            basis = torch.eye(2, dtype=q.dtype)
            seed = 1_000_000 + repeat * 1000 + qid * 100
            discovery = oracle.query(q, basis, range(seed, seed + 8))
            _, predicted = response_columns(model, q, basis)
            error = discovery.responses - predicted
            corrected = unbiased_gram(error)
            naive = torch.einsum('bmr,bms->rs', error, error) / len(error)
            truth = exact_response(q) - predicted
            truth = truth.T @ truth
            for rule, gram in [('corrected', corrected), ('naive', naive)]:
                ev, vec = torch.linalg.eigh(gram)
                direction = vec[:, -1]
                from .reference import Proposal
                proposal = Proposal(basis @ direction, direction, float(ev[-1]), float((predicted @ direction).norm()))
                verified = verify_proposal(model, oracle, q, proposal, .001,
                    range(seed + 40, seed + 56), discovery_seeds=discovery.seeds)
                rows.append(dict(system='gaussian', repeat=repeat, query=qid, rule=rule,
                    discovery_score=float(ev[-1]), true_selected_error=float(direction @ truth @ direction),
                    verified_error=verified['squared_response_error'], true_max_error=float(torch.linalg.eigvalsh(truth)[-1]),
                    matrix_error=float((gram - truth).square().sum())))
        # Uniform phase makes the law independent of a, despite nonzero pathwise derivatives.
        gen = torch.Generator().manual_seed(3_000_000 + repeat)
        xi = 2 * math.pi * torch.rand(8, generator=gen, dtype=torch.float64)
        response = torch.cos(xi + .3)[:, None, None]
        fresh = 2 * math.pi * torch.rand(16, generator=gen, dtype=torch.float64)
        fd = ((torch.sin(fresh + .301) - torch.sin(fresh + .299)) / .002)[:, None, None]
        for rule, score in [('corrected', float(unbiased_gram(response)[0, 0])),
                            ('naive', float(response.square().mean()))]:
            rows.append(dict(system='cancellation', repeat=repeat, query=0, rule=rule,
                discovery_score=score, true_selected_error=0., verified_error=float(unbiased_gram(fd)[0, 0]),
                true_max_error=0., matrix_error=score * score))
    table(c, 'mechanisms-v1', 'acquisition', rows)


def basin(c):
    """Slow double-well coordinate plus fast harmonic nuisance coordinates."""
    rows = []
    for index, initial in enumerate((-.8, -.2, .2, .8)):
        q = torch.zeros(8, dtype=torch.float64); q[0] = initial
        basis = torch.eye(8, dtype=q.dtype)[:, [0, 2]]
        potential = lambda x: .25 * (x[0].square() - 1).square() + 2 * x[1:].square().sum()
        features = lambda initial, path: torch.cat((path[:, :2].flatten(), path[:, 0].square()))
        oracle = BAOABOracle(potential, features, mass=torch.ones_like(q), dt=.02, steps=100,
                            kbt=.3, friction=1., horizon_steps=[20, 100])
        discovery = oracle.query(q, basis, range(6_000_000 + index * 1000, 6_000_008 + index * 1000))
        reference = oracle.query(q, basis, range(6_000_100 + index * 1000, 6_000_228 + index * 1000))
        # Fixed zero-response surrogate is a declared missing-information control.
        corrected = unbiased_gram(discovery.responses)
        naive = torch.einsum('bmr,bms->rs', discovery.responses, discovery.responses) / 8
        reference_gram = unbiased_gram(reference.responses)
        for rule, gram in [('corrected', corrected), ('naive', naive)]:
            ev, vectors = torch.linalg.eigh(gram); a = vectors[:, -1]
            rows.append(dict(query=index, initial_slow_coordinate=initial, rule=rule,
                selected_slow_fraction=float(a[0].square()), discovery_score=float(ev[-1]),
                independent_128shot_score=float(a @ reference_gram @ a),
                force_calls=discovery.force_calls + reference.force_calls,
                hvp_calls=discovery.hvp_calls + reference.hvp_calls))
    table(c, 'mechanisms-v1', 'two-basin', rows)


def jacobian(model, x, create_graph=False):
    columns = []
    for j in range(2):
        tangent = torch.zeros_like(x); tangent[:, j] = 1
        columns.append(torch.autograd.functional.jvp(model, x, tangent, create_graph=create_graph)[1])
    return torch.stack(columns, -1)


def learn(c):
    """Scientific toy fits: Gaussian feature likelihood with auxiliary response loss."""
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(c['seed'])
    x = torch.rand(448, 2, generator=gen, dtype=torch.float64) * 2 - 1
    exact_y, exact_j = exact(x), exact_response(x)
    labels, derivatives = [], []
    oracle = AutogradOracle(gaussian)
    for i, q in enumerate(x[:192]):
        b = oracle.query(q, torch.eye(2, dtype=q.dtype), range(4_000_000 + i * 100, 4_000_008 + i * 100))
        labels.append(b.values.mean(0)); derivatives.append(b.responses.mean(0))
    labels, derivatives = torch.stack(labels), torch.stack(derivatives)
    value_scale = labels[:128].std(0).clamp_min(.1)
    response_scale = derivatives[:128].square().mean().sqrt().clamp_min(.1)
    base = output(c) / 'analyses/mechanisms-v1'
    base.mkdir(parents=True, exist_ok=True)
    torch.save(dict(x=x, feature_means=labels, response_means=derivatives,
                    value_scale=value_scale, response_scale=response_scale,
                    roles=['train'] * 128 + ['selection'] * 64 + ['test'] * 256), base / 'learning-data.pt')
    results = []
    for seed in c['fit_seeds']:
        for arm, weight in [('values', 0.), ('responses', 1.)]:
            torch.manual_seed(seed)
            model = SmallAtlas(2, 2, latent_dim=4, width=32).double()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
            root = base / f'{arm}-{seed}'; root.mkdir(exist_ok=True)
            began = time.monotonic(); best = math.inf
            def nll(prediction, target):
                return (.5 * ((prediction - target) / value_scale).square() + value_scale.log() + .5 * math.log(2 * math.pi)).mean()
            with online_training(DEFAULTS, run_id=f'response-toy-{seed}-{arm}', name=f'response-atlas-gaussian-{arm}-{seed}',
                    config=dict(protocol=c['protocol'], seed=seed, arm=arm, batch_size=128, microbatch=128,
                                objective='fixed-variance feature Gaussian NLL + scaled response MSE',
                                selector='selection feature NLL', scale_fit='train only', capacity='toy MLP 32/4'),
                    folder=root, receipt_path=root / 'wandb.json', job_type='predictor', group=c['protocol']) as run:
                for epoch in range(c['toy_epochs']):
                    model.train(); pred = model(x[:128])
                    loss = nll(pred, labels[:128])
                    if weight:
                        loss = loss + weight * ((jacobian(model, x[:128], True) - derivatives[:128]) / response_scale).square().mean()
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    with torch.no_grad():
                        selection = float(nll(model(x[128:192]), labels[128:192]))
                    run.log(dict(epoch=epoch + 1, train_objective=float(loss.detach()), selection_nll=selection))
                    if selection < best:
                        best = selection; state = copy.deepcopy(model.state_dict()); selected_epoch = epoch + 1
                model.load_state_dict(state); model.eval()
                pred = model(x[192:]).detach(); derivative = jacobian(model, x[192:]).detach()
                row = dict(arm=arm, seed=seed, selected_epoch=selected_epoch, selection_nll=best,
                    exact_value_mse=float((pred - exact_y[192:]).square().mean()),
                    exact_response_mse=float((derivative - exact_j[192:]).square().mean()),
                    seconds=time.monotonic() - began, scope='matched data; not equal total cost')
                torch.save(dict(model=state, predictions=pred, responses=derivative, row=row), root / 'checkpoint.pt')
                run.summary.update(row); write_json(root / 'complete.json', row); results.append(row)
    table(c, 'mechanisms-v1', 'learning', results)
