"""Shared-backbone inner/center training with an explicit supervised control."""

from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch

from src.experiment_runner.registry import sha256, write_json
from src.models.encoders.mace_context import context_features
from .data import load_clouds, read_json, training_clouds
from .engine import augment, graph, load_model, loss_from_features
from .evaluate import crossing_clouds
from .recovery_data import GROUPS, PhysicalHeads, feature_arrays, inputs, publish, status
from .recovery_readouts import evaluate_features, score_predictions, summarize_recovery
from .train import view_major


def dual_encode(config, model, clouds, *, gradients=False):
    pieces = []
    with torch.set_grad_enabled(gradients):
        for start in range(0, len(clouds), config['micro_batch_size']):
            g = graph(config, clouds[start:start+config['micro_batch_size']], 'halo_inner')
            pieces.append(context_features(model.encoder.mace, g, return_center=True))
    z = torch.cat(pieces)
    if z.shape != (len(clouds), 512) or not torch.isfinite(z).all():
        raise FloatingPointError(f'Invalid combined context features: {z.shape}')
    return z


def physical_loss(heads, z, target, feature_mean, feature_scale, weights):
    prediction = heads((z-feature_mean)/feature_scale)
    difference = (prediction-target).square()*weights
    values = {name: difference[:, section].sum(dim=1).mean() for name, section in GROUPS.items()}
    return torch.stack(list(values.values())).mean(), values


def joint_loss(config, model, heads, z, target, mean, scale, weights, variant):
    n = len(target)
    if len(z) != 4*n:
        raise ValueError(f'Expected three SSL views plus unaugmented anchors: {z.shape}, targets {target.shape}')
    ssl = loss_from_features(model, z[:3*n, :256])[0]
    anchor = z[3*n:] if variant == 'dual_physics' else z[3*n:].detach()
    physics, values = physical_loss(heads, anchor, target, mean, scale, weights)
    # The control trains identical heads but detaches their inputs. Consequently,
    # only VICReg updates its encoder; both variants use the same validation rule.
    loss = ssl+config['joint']['physics_weight']*physics
    return loss, dict(ssl=float(ssl.detach()), physics=float(physics.detach()),
                      **{k: float(v.detach()) for k, v in values.items()})


def replay_joint(config, encoder_config, model, heads, clouds, target, mean, scale, weights, variant):
    z = dual_encode(encoder_config, model, clouds).detach().requires_grad_(True)
    loss, metrics = joint_loss(config, model, heads, z, target, mean, scale, weights, variant)
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Nonfinite joint loss: {metrics}')
    loss.backward()
    end = len(clouds) if variant == 'dual_physics' else 3*len(target)
    for start in range(0, end, encoder_config['micro_batch_size']):
        batch = clouds[start:start+encoder_config['micro_batch_size']]
        actual = dual_encode(encoder_config, model, batch, gradients=True)
        actual.backward(z.grad[start:start+len(batch)])
    return metrics


def warm_heads(config, data):
    pilot, _, _, ids, y, _, _, weights, _ = data
    features, hashes = feature_arrays(pilot, 'frozen', 'fusion')
    z = features['z'].astype(np.float64)
    mean, scale = z[ids['train']].mean(axis=0), z[ids['train']].std(axis=0)
    scale[scale == 0] = 1
    xx = torch.as_tensor((z-mean)/scale, device=config['device'], dtype=torch.float32)
    yy = torch.as_tensor(y, device=config['device'], dtype=torch.float32)
    ww = torch.as_tensor(weights, device=config['device'], dtype=torch.float32)
    torch.manual_seed(config['seed'])
    heads = PhysicalHeads(512, config['joint']['head_hidden_width']).to(config['device'])
    optimizer = torch.optim.AdamW(heads.parameters(), lr=config['joint']['head_learning_rate'], weight_decay=.0001)
    best, history = float('inf'), []
    for epoch in range(config['joint']['head_warmup_epochs']):
        heads.train()
        optimizer.zero_grad(set_to_none=True)
        loss = ((heads(xx[ids['train']])-yy[ids['train']]).square()*ww).sum(dim=1).mean()/len(GROUPS)
        loss.backward(); optimizer.step()
        heads.eval()
        with torch.no_grad():
            score = float(((heads(xx[ids['val']])-yy[ids['val']]).square()*ww).sum(dim=1).mean()/len(GROUPS))
        history.append(dict(epoch=epoch+1, train=float(loss.detach()), validation=score))
        if score < best:
            best, best_epoch = score, epoch+1
            state = {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()}
    return dict(head_state=state, feature_mean=mean, feature_scale=scale,
                best_validation=best, best_epoch=best_epoch, history=history, feature_inputs=hashes)


def verify_joint(config):
    data = inputs(config)
    pilot = dict(data[0], device=config['device'], cpu_threads=config['cpu_threads'], micro_batch_size=2)
    model, _ = load_model(pilot)
    root = Path(config['output'])/'technical/verification'
    root.mkdir(parents=True, exist_ok=True)
    if (root/'status.json').exists():
        raise FileExistsError(f'Preserve verification attempt: {root}')
    started = time.monotonic()
    status(config, 'verification', state='running')
    try:
        warm = warm_heads(config, data)
        torch.save(warm, root/'head-initialization.pt')
        clouds = load_clouds(Path(pilot['cache'])/'context-000.npz')[:6]
        model.eval()
        with torch.no_grad():
            z = dual_encode(pilot, model, clouds)
            expected = torch.cat([context_features(model.encoder.mace, graph(pilot, clouds, mode))
                                  for mode in ['halo_inner', 'halo_center']], dim=1)
            forward_error = float((z-expected).double().square().sum()/expected.double().square().sum())
            if forward_error > 1e-8:
                raise AssertionError(f'Combined/separate feature disagreement: {forward_error}')
            # Check input symmetries on a complete real context, with tracked ID 0.
            rng = np.random.default_rng(config['seed'])
            x = clouds[0]
            permutation = np.r_[0, rng.permutation(np.arange(1, len(x)))]
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            variants = [x[permutation], (x@rotation).astype(np.float32)]
            controls = dual_encode(pilot, model, variants)
            symmetry_errors = [float((v-z[0]).double().square().sum()/z[0].double().square().sum()) for v in controls]
            if max(symmetry_errors) > 1e-8:
                raise AssertionError(f'Combined readout symmetry errors: {symmetry_errors}')
        device = config['device']
        mean = torch.as_tensor(warm['feature_mean'], dtype=torch.float32, device=device)
        scale = torch.as_tensor(warm['feature_scale'], dtype=torch.float32, device=device)
        weights = torch.as_tensor(data[7], dtype=torch.float32, device=device)
        # context-000 contains anchor/spatial/temporal views for the first two rows.
        rows = read_json(Path(pilot['cache'])/'manifest.json')['records'][0]['rows'][:2]
        target = torch.as_tensor(data[4][rows], dtype=torch.float32, device=device)
        selected = [clouds[i*3+v] for v in range(3) for i in range(2)]+[clouds[0], clouds[3]]
        heads = PhysicalHeads(512, config['joint']['head_hidden_width']).to(device)
        heads.load_state_dict(warm['head_state'], strict=True)
        initial_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
        initial_heads = {k: v.detach().clone() for k, v in heads.state_dict().items()}
        errors = {}
        for variant in config['joint']['variants']:
            model.load_state_dict(initial_model, strict=True)
            heads.load_state_dict(initial_heads, strict=True)
            model.zero_grad(set_to_none=True); heads.zero_grad(set_to_none=True)
            model.train(); heads.train()
            direct = dual_encode(pilot, model, selected, gradients=True)
            loss, _ = joint_loss(config, model, heads, direct, target, mean, scale, weights, variant)
            loss.backward()
            parameters = lambda: [(f'encoder.{n}', p) for n, p in model.named_parameters()]+[(f'heads.{n}', p) for n, p in heads.named_parameters()]
            expected_grad = {n: p.grad.detach().clone() for n, p in parameters() if p.grad is not None}
            model.load_state_dict(initial_model, strict=True)
            heads.load_state_dict(initial_heads, strict=True)
            model.zero_grad(set_to_none=True); heads.zero_grad(set_to_none=True)
            replay_joint(config, pilot, model, heads, selected, target, mean, scale, weights, variant)
            numerator = denominator = 0.
            for name, parameter in parameters():
                if name in expected_grad:
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise AssertionError(f'Missing or nonfinite gradient: {variant}/{name}')
                    numerator += float((parameter.grad-expected_grad[name]).double().square().sum())
                    denominator += float(expected_grad[name].double().square().sum())
            error = numerator/denominator
            if error > 1e-6:
                raise AssertionError(f'Joint replay gradient disagreement: {variant}: {error}')
            errors[variant] = error
        write_json(root/'report.json', dict(forward_relative_squared_error=forward_error,
            symmetry_relative_squared_errors=symmetry_errors, replay_relative_squared_errors=errors,
            head_initialization_validation_mse=warm['best_validation'],
            head_initialization_sha256=sha256(root/'head-initialization.pt')))
        status(config, 'verification', state='complete', elapsed_seconds=time.monotonic()-started)
        publish(config, ['technical/verification/report.json'])
    except BaseException as error:
        status(config, 'verification', state='failed', error=repr(error))
        raise


def extract_joint(config, pilot, model, variant):
    root = Path(config['output'])/'technical'/f'train-{variant}'
    manifest = read_json(Path(pilot['cache'])/'manifest.json')
    z, temporal = np.full((5760, 512), np.nan, np.float32), np.full((144*17, 512), np.nan, np.float32)
    model.eval()
    for index, record in enumerate(manifest['records']):
        path = Path(pilot['cache'])/record['file']
        if sha256(path) != record['sha256']:
            raise ValueError(f'Changed context input: {path}')
        z[record['rows']] = dual_encode(pilot, model, load_clouds(path)[::3]).cpu().numpy()
        status(config, f'train-{variant}', state='extracting', context=index+1, total=90)
    boundaries = []
    for record in manifest['temporal']:
        path = Path(pilot['cache'])/record['file']
        if sha256(path) != record['sha256']:
            raise ValueError(f'Changed temporal input: {path}')
        clouds = load_clouds(path)
        temporal[record['rows']] = dual_encode(pilot, model, clouds).cpu().numpy()
        boundaries.extend(clouds[64:68])
    epsilons, crossings = [.1, .01, .001, .0001], []
    for epsilon in epsilons:
        clouds = [x for c in boundaries for x in crossing_clouds(c, epsilon)]
        crossings.append(dual_encode(pilot, model, clouds).cpu().numpy().reshape(72, 2, 512))
    if not np.isfinite(z).all() or not np.isfinite(temporal).all():
        raise ValueError(f'Incomplete joint feature extraction: {variant}')
    features = dict(z=z, temporal_z=temporal.reshape(144, 17, 512), crossing_z=np.stack(crossings), epsilons=np.asarray(epsilons))
    np.savez(root/'features.npz', **features)
    return features


def train_joint(config, variant):
    if variant not in config['joint']['variants']:
        raise ValueError('Select an explicit joint-training variant')
    root = Path(config['output'])/'technical'/f'train-{variant}'
    root.mkdir(parents=True, exist_ok=True)
    if (root/'status.json').exists():
        raise FileExistsError(f'Preserve previous training attempt: {root}')
    verification = Path(config['output'])/'technical/verification'
    if read_json(verification/'status.json')['state'] != 'complete':
        raise ValueError('Joint GPU verification must pass before training')
    data = inputs(config)
    pilot = dict(data[0], device=config['device'], cpu_threads=config['cpu_threads'],
                 micro_batch_size=config['joint']['micro_batch_size'])
    model, original_cfg = load_model(pilot)
    warm = torch.load(verification/'head-initialization.pt', map_location='cpu', weights_only=False)
    expected_hash = read_json(verification/'report.json')['head_initialization_sha256']
    if sha256(verification/'head-initialization.pt') != expected_hash:
        raise ValueError('Joint head initialization changed after verification')
    device = config['device']
    mean = torch.as_tensor(warm['feature_mean'], dtype=torch.float32, device=device)
    scale = torch.as_tensor(warm['feature_scale'], dtype=torch.float32, device=device)
    weights = torch.as_tensor(data[7], dtype=torch.float32, device=device)
    targets = torch.as_tensor(data[4], dtype=torch.float32, device=device)
    heads = PhysicalHeads(512, config['joint']['head_hidden_width']).to(device)
    heads.load_state_dict(warm['head_state'], strict=True)
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if p.requires_grad], lr=config['joint']['learning_rate']),
        dict(params=heads.parameters(), lr=config['joint']['head_learning_rate'])], weight_decay=float(original_cfg.decay_rate))
    rng = np.random.default_rng(config['seed'])
    ids = data[3]
    clouds = training_clouds(pilot)
    batch_size = config['joint']['batch_size']
    history, best = [], warm['best_validation']
    deadline = datetime.fromisoformat(config['joint']['deadline_utc']).timestamp()
    started = time.monotonic()
    source_hashes = {str(p): sha256(p) for p in [Path(__file__),
        Path('src/models/encoders/mace_context.py'), Path('src/research/mace_context/recovery_data.py'),
        Path('src/research/mace_context/engine.py'), Path(pilot['checkpoint']), Path(pilot['cache'])/'manifest.json']}

    def save(filename, epoch):
        payload = dict(protocol='mace_context_recovery_joint_v1', variant=variant, config=config, pilot_config=pilot,
            epoch=epoch, history=history, best_validation=best, model_state=model.state_dict(), head_state=heads.state_dict(),
            feature_mean=warm['feature_mean'], feature_scale=warm['feature_scale'],
            target_mean=data[5], target_scales=data[6], target_weights=data[7], optimizer_state=optimizer.state_dict(),
            numpy_rng_state=rng.bit_generator.state, torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state(), implementation_and_input_hashes=source_hashes)
        temporary = root/(filename+'.building')
        torch.save(payload, temporary); temporary.replace(root/filename)

    try:
        status(config, f'train-{variant}', state='initializing')
        write_json(root/'config.json', dict(config=config, pilot=pilot, variant=variant,
            encoder_supervision=variant == 'dual_physics', checkpoint_selection='minimum validation physical-head loss, including epoch zero',
            implementation_and_input_hashes=source_hashes))
        save('best.pt', 0)
        for epoch in range(1, config['joint']['epochs']+1):
            if time.time() >= deadline-600:
                raise TimeoutError('Stopping before allocation deadline; completed-epoch checkpoints are preserved')
            epoch_start = time.monotonic()
            model.train(); heads.train()
            order = rng.permutation(ids['train'])
            training = []
            for step, start in enumerate(range(0, len(order)-batch_size+1, batch_size)):
                rows = order[start:start+batch_size]
                views = augment(pilot, model, view_major(clouds, rows), 'halo_inner')
                views.extend([clouds[row][0] for row in rows])  # Physical labels match these unjittered anchors.
                optimizer.zero_grad(set_to_none=True)
                metrics = replay_joint(config, pilot, model, heads, views, targets[rows], mean, scale, weights, variant)
                # Head gradient size must not change encoder clipping in the SSL control.
                encoder_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(original_cfg.gradient_clip_val), error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(heads.parameters(), float(original_cfg.gradient_clip_val), error_if_nonfinite=True)
                optimizer.step()
                training.append(metrics)
                status(config, f'train-{variant}', state='training', epoch=epoch, epochs=config['joint']['epochs'],
                    step=step+1, steps=len(order)//batch_size, metrics=metrics, elapsed_seconds=time.monotonic()-started)
                print('JOINT_STEP', variant, epoch, step+1, metrics, 'gradient_norm', float(encoder_norm), flush=True)
            model.eval(); heads.eval()
            validation = []
            with torch.no_grad():
                for start in range(0, len(ids['val']), batch_size):
                    rows = ids['val'][start:start+batch_size]
                    z = dual_encode(pilot, model, [clouds[row][0] for row in rows])
                    loss, values = physical_loss(heads, z, targets[rows], mean, scale, weights)
                    validation.append((len(rows), float(loss), {k: float(v) for k, v in values.items()}))
            score = sum(n*v for n, v, _ in validation)/len(ids['val'])
            record = dict(epoch=epoch, train={k: float(np.mean([r[k] for r in training])) for k in training[0]},
                validation=score, validation_groups={k: sum(n*v[k] for n, _, v in validation)/len(ids['val']) for k in GROUPS},
                epoch_seconds=time.monotonic()-epoch_start, elapsed_seconds=time.monotonic()-started)
            history.append(record)
            if score < best:
                best = score
                save('best.pt', epoch)
            save('last.pt', epoch)
            write_json(root/'history.json', history)
            publish(config, [f'technical/train-{variant}/history.json'])
            print('JOINT_EPOCH', variant, record, flush=True)
        selected = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(selected['model_state'], strict=True)
        heads.load_state_dict(selected['head_state'], strict=True)
        features = extract_joint(config, pilot, model, variant)
        status(config, f'train-{variant}', state='evaluating', selected_epoch=selected['epoch'])
        directory = root/'evaluation'
        evaluate_features(config, variant, features, data, directory, nonlinear=False)
        zz = np.concatenate([features['z'], features['temporal_z'].reshape(-1, 512), features['crossing_z'].reshape(-1, 512)])
        heads.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(zz), 1024):
                predictions.append(heads((torch.as_tensor(zz[start:start+1024], device=device)-mean)/scale).cpu().numpy())
        score_predictions(variant, 'trained_head', np.concatenate(predictions).astype(np.float64), features, data,
                          dict(epoch=selected['epoch'], validation=selected['best_validation']), directory)
        status(config, f'train-{variant}', state='complete', selected_epoch=selected['epoch'], epochs=config['joint']['epochs'],
            elapsed_seconds=time.monotonic()-started, best_checkpoint_sha256=sha256(root/'best.pt'))
    except BaseException as error:
        status(config, f'train-{variant}', state='failed', error=repr(error), completed_epochs=len(history),
               elapsed_seconds=time.monotonic()-started)
        raise
