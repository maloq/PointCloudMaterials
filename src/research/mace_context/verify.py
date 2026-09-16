"""Compare pruned computation and gradients against complete unpruned graphs."""

from pathlib import Path
import time

import numpy as np
import torch

from src.experiment_runner.registry import write_json
from src.models.encoders.mace_context import context_features, inner_weights
from src.models.encoders.pretrained_mace import MACEGeometry, dense_matmul_precision
from .data import load_clouds
from .engine import encode, graph, load_model, loss_from_features, replay_step


def full_features(mace, cloud):
    x = torch.as_tensor(cloud, device=next(mace.parameters()).device)
    n = len(x)
    valid = (torch.cdist(x[None], x[None])[0] < 5.) & ~torch.eye(n, dtype=torch.bool, device=x.device)
    edges = valid.nonzero().T.contiguous()
    attrs = torch.nn.functional.one_hot(mace.element_indices[0].expand(n),len(mace.backbone.atomic_numbers)).float()
    vectors = x[edges[1]]-x[edges[0]]
    angular = mace.backbone.spherical_harmonics(vectors)
    radial, cutoff = mace.backbone.radial_embedding(vectors.norm(dim=-1,keepdim=True), attrs, edges, mace.backbone.atomic_numbers)
    g = MACEGeometry(1, n, attrs, edges, angular, radial, cutoff)
    with dense_matmul_precision(mace.dense_precision):
        return mace._learned_node_features(g)[0]


def relative_error(a, b):
    return float((a-b).square().mean()/b.square().mean())


def verify(config):
    model, _ = load_model(config)
    model.eval()
    clouds = load_clouds(Path(config['cache'])/'context-000.npz')[:24]
    results = {}
    with torch.no_grad():
        nodes = full_features(model.encoder.mace, clouds[0])
        weights = torch.as_tensor(inner_weights(np.linalg.norm(clouds[0],axis=1),
                       config['inner_radius_A'], config['outer_radius_A']),device=config['device'])
        expected = {'halo_center': nodes[0], 'halo_inner': (nodes*weights[:,None]).sum(0)/weights.sum(),
                    'halo_mean80': nodes[:80].mean(0),
                    'mean80': model.encoder(torch.as_tensor(clouds[0][:80],device=config['device'])[None]/model.encoder.reference_radius_A)[0]}
        for mode in config['modes']:
            actual = encode(config, model, [clouds[0]], mode)[0]
            error = relative_error(actual, expected[mode])
            if error > 1e-8:
                raise AssertionError(f'Pruned/full graph disagreement: {mode}: {error}')
            results[mode] = dict(full_graph_relative_mse=error)
        for mode in ['halo_inner', 'halo_center']:
            x = clouds[0]
            rng = np.random.default_rng(config['seed'])
            perm = np.r_[0, rng.permutation(np.arange(1,len(x)))]
            q, _ = np.linalg.qr(rng.normal(size=(3,3)))
            variants = [x[perm], (x@q).astype(np.float32),
                        np.concatenate([x, np.array([[19.,0,0],[0,19.,0]],dtype=np.float32)])]
            baseline = encode(config, model, [x], mode)[0]
            control = encode(config, model, variants, mode)
            errors = [relative_error(value,baseline) for value in control]
            if max(errors) > 1e-8:
                raise AssertionError(f'Context invariance failed: {mode}: {errors}')
            results[mode]['permutation_rotation_extra_halo_relative_mse'] = errors
    # Verify the full VICReg gradient, including projector BatchNorm, against replay.
    selected = [clouds[i*3+v] for v in range(3) for i in range(8)]
    saved = {key:value.detach().clone() for key,value in model.state_dict().items()}
    model.train()
    # Keep encoder microbatch shapes identical so this tests replay algebra,
    # independently of floating-point variation between different GEMM shapes.
    direct = torch.cat([context_features(model.encoder.mace, graph(config,selected[i:i+2],'halo_inner'))
                        for i in range(0,len(selected),2)])
    loss = loss_from_features(model,direct)[0]
    loss.backward()
    gradients = {name:p.grad.clone() for name,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    model.load_state_dict(saved,strict=True)
    replay_config = dict(config,micro_batch_size=2)
    replay_loss, _ = replay_step(replay_config, model, selected, 'halo_inner')
    numerator = denominator = 0.
    for name, p in model.named_parameters():
        if name in gradients:
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise AssertionError(f'Missing/nonfinite replay gradient: {name}')
            numerator += float((p.grad-gradients[name]).double().square().sum())
            denominator += float(gradients[name].double().square().sum())
    error = numerator/denominator
    # cuEquivariance reductions and compensated BF16 radial GEMMs retain a
    # numerical noise floor; require <0.1% relative L2 gradient disagreement.
    if error > 1e-6 or abs(float(loss.detach())-replay_loss) > 1e-3:
        raise AssertionError(f'Gradient replay mismatch: {error}, {float(loss.detach())}, {replay_loss}')
    results['gradient_replay'] = dict(relative_squared_error=error, direct_loss=float(loss.detach()), replay_loss=replay_loss)
    model.zero_grad(set_to_none=True)
    model.eval()
    for mode in ['mean80','halo_inner','halo_center']:
        torch.cuda.synchronize()
        started = time.monotonic()
        for _ in range(3):
            encode(config,model,clouds,mode)
        torch.cuda.synchronize()
        results[mode]['inference_seconds_per_cloud'] = (time.monotonic()-started)/(3*len(clouds))
    root = Path(config['output'])/'technical'
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/'verification.json',dict(state='complete',results=results))
    print(results,flush=True)
