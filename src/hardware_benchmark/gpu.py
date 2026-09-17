"""Synthetic inputs, actual repository models, synchronized training-step timing."""
import time
from dataclasses import dataclass
from types import SimpleNamespace

from .common import command_info, summarize


@dataclass
class GPUSettings:
    workloads: tuple = ("pointnet", "mace", "forecast")
    points: int = 80
    pointnet_batch: int = 128
    mace_batch: int = 16
    forecast_batch: int = 512
    mace_channels: int = 64
    mace_accelerated: bool = False
    history_steps: int = 16
    future_steps: int = 16
    embedding_dim: int = 256
    forecast_width: int = 256
    warmup: int = 10
    steps: int = 30
    repeats: int = 3
    seed: int = 1729
    host_threads: int = 4
    matmul_precision: str = "highest"


def validate(settings):
    if not settings.workloads or len(set(settings.workloads)) != len(settings.workloads):
        raise ValueError("gpu.workloads must be a nonempty list without duplicates")
    unknown = set(settings.workloads) - {"pointnet", "mace", "forecast"}
    if unknown:
        raise ValueError(f"Unknown GPU workloads: {sorted(unknown)}")
    for name in ("points", "pointnet_batch", "mace_batch", "forecast_batch", "mace_channels",
                 "history_steps", "future_steps", "embedding_dim", "forecast_width", "warmup",
                 "steps", "repeats", "host_threads"):
        if type(getattr(settings, name)) is not int or getattr(settings, name) < 1:
            raise ValueError(f"gpu.{name} must be a positive integer")
    if min(settings.pointnet_batch, settings.mace_batch) < 2 or settings.points < 8:
        raise ValueError("VICReg requires batches >= 2; synthetic atom graphs require points >= 8")
    if settings.forecast_width % 4:
        raise ValueError("gpu.forecast_width must be divisible by 4 attention heads")
    if type(settings.mace_accelerated) is not bool:
        raise ValueError("gpu.mace_accelerated must be a boolean")
    if settings.matmul_precision not in ("highest", "high"):
        raise ValueError("gpu.matmul_precision must be highest or high (TF32 allowed)")


def _vicreg(input_dim):
    from src.training_methods.shared.vicreg import VICRegLoss
    return VICRegLoss.from_config(SimpleNamespace(
        vicreg_enabled=True, vicreg_weight=1.0, vicreg_embed_dim=128,
        vicreg_sim_coeff=25.0, vicreg_std_coeff=25.0, vicreg_cov_coeff=1.0,
        vicreg_drop_ratio=0.0, vicreg_jitter_mode="absolute"), input_dim=input_dim)


def radius_edges(clouds, cutoff):
    import torch
    distances = torch.cdist(clouds, clouds)
    # CUDA's matrix-multiply cdist path can round d(x, x) above zero.
    # Exclude identity edges by index, never by a floating-point distance test.
    diagonal = torch.eye(clouds.shape[1], device=clouds.device, dtype=torch.bool)
    masks = (distances < cutoff) & ~diagonal[None]
    counts = masks.sum((1, 2))
    edges = torch.nn.utils.rnn.pad_sequence([mask.nonzero() for mask in masks], batch_first=True)
    return edges, counts


def build_task(name, settings, device):
    """Build once outside timing. CPU is allowed here for correctness tests only."""
    import torch
    from torch import nn

    torch.manual_seed(settings.seed)
    if name == "forecast":
        from src.training_methods.embedding_forecast.model import EmbeddingForecaster, forecast_loss
        config = dict(architecture="transformer", target="trajectory", distribution="deterministic",
                      history_mode="real", width=settings.forecast_width, layers=2, heads=4, dropout=0.0)
        batch_size = settings.forecast_batch
        model = EmbeddingForecaster(settings.embedding_dim, settings.history_steps, 0.1,
                                    [settings.future_steps * 0.1], config).float().to(device)
        history = torch.randn(batch_size, settings.history_steps, settings.embedding_dim, device=device)
        future = history[:, -1:] + 0.05 * torch.randn(
            batch_size, settings.future_steps, settings.embedding_dim, device=device).cumsum(1)
        weights = dict(mse=1.0, bin_mse=0.0, increment_mse=0.0, nll=0.0)

        def loss_fn():
            return forecast_loss(model, model(history), future, history[:, -1], weights)[0]

        detail = dict(model="EmbeddingForecaster", config=config, input_shape=list(history.shape),
                      output_shape=list(future.shape), objective="production forecast_loss, MSE weight 1")
    else:
        if name == "pointnet":
            from src.models.encoders.pointnet import PointNetEncoder
            batch_size = settings.pointnet_batch
            encoder = PointNetEncoder(latent_size=128).float().to(device)
            cloud = torch.randn(batch_size, 3, settings.points, device=device)
            # Fixed synthetic paired views; augmentation and H2D are outside timing.
            views = torch.cat((cloud, cloud + 0.01 * torch.randn_like(cloud)))
            input_dim = 128

            def encode():
                return model["encoder"](views)

            detail = dict(model="PointNetEncoder/PnE_L", input_shape=list(views.shape), feature_transform=True)
        elif name == "mace":
            from src.models.encoders.atomic_graph import ReferenceMACEEncoder
            batch_size = settings.mace_batch
            encoder = ReferenceMACEEncoder(channels=settings.mace_channels, max_ell=2,
                                          correlation=3, cutoff_A=4.0,
                                          accelerated=settings.mace_accelerated).float().to(device)
            # A central site followed by its nearest simple-cubic grid sites.
            # Positions are Angstroms, material IDs exactly 0=Al, 1=Mg, 2=Ta.
            side = int(settings.points ** (1 / 3)) + 2
            axis = torch.arange(-side, side + 1, device=device, dtype=torch.float32)
            grid = torch.cartesian_prod(axis, axis, axis)
            grid = grid[grid.square().sum(-1).argsort(stable=True)[:settings.points]] * 2.4
            cloud = grid[None] + 0.02 * torch.randn(batch_size, settings.points, 3, device=device)
            cloud = cloud - cloud[:, :1]
            views = torch.cat((cloud, cloud + 0.01 * torch.randn_like(cloud)))
            views = views - views[:, :1]
            edges, counts = radius_edges(views, 4.0)
            material = torch.arange(batch_size, device=device).remainder(3).repeat(2)
            input_dim = 2 * settings.mace_channels

            def encode():
                return model["encoder"](views, material, edges, counts)

            detail = dict(model="ReferenceMACEEncoder", input_shape=list(views.shape),
                          channels=settings.mace_channels, max_ell=2, correlation=3,
                          cutoff_A=4.0, accelerated=settings.mace_accelerated,
                          directed_edges=int(counts.sum().item()), graph_construction_timed=False)
        else:
            raise ValueError(f"Unknown GPU workload {name}")
        model = nn.ModuleDict(dict(encoder=encoder, objective=_vicreg(input_dim).to(device)))

        def loss_fn():
            z = model["objective"].project_features(encode())
            # Repository's exact population-variance / sample-covariance protocol.
            return model["objective"]._loss(z[:batch_size], z[batch_size:])[0]

        detail["objective"] = "production VICReg, coefficients 25/25/1, projector width 128, two fused views"
    model.train()
    detail.update(batch_size=batch_size, dtype="float32", parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model, loss_fn, detail


def _check_training(model, losses, name):
    import torch
    if not torch.isfinite(torch.stack(losses)).all().item():
        raise RuntimeError(f"Nonfinite loss in GPU workload {name}")
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not all(torch.isfinite(g).all().item() for g in grads):
        raise RuntimeError(f"Missing or nonfinite gradients in GPU workload {name}")
    if not any(g.abs().max().item() > 0 for g in grads):
        raise RuntimeError(f"All gradients are zero in GPU workload {name}")
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError(f"Nonfinite parameters in GPU workload {name}")


def run(settings, device_name):
    import torch

    validate(settings)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requires CUDA; no CPU substitution is made")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError(f"GPU benchmark requires a CUDA device, got {device_name}")
    torch.cuda.set_device(device)
    torch.set_num_threads(settings.host_threads)
    torch.set_float32_matmul_precision(settings.matmul_precision)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    props = torch.cuda.get_device_properties(device)
    result = dict(device=str(device), name=props.name, total_memory_bytes=props.total_memory,
                  compute_capability=list(torch.cuda.get_device_capability(device)),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  cudnn_version=torch.backends.cudnn.version(),
                  matmul_precision=settings.matmul_precision, cudnn_tf32=False,
                  driver=command_info(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
                  nvidia_smi_before=command_info(["nvidia-smi"]), workloads={})
    for name in settings.workloads:
        print(f"gpu: building {name} with synthetic inputs", flush=True)
        setup_started = time.perf_counter()
        model, loss_fn, detail = build_task(name, settings, device)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4,
                                      weight_decay=1e-2, foreach=False, fused=False)

        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            optimizer.step()
            return loss.detach()

        torch.cuda.synchronize(device)
        detail["setup_seconds"] = time.perf_counter() - setup_started
        started = time.perf_counter()
        warm_losses = [step() for _ in range(settings.warmup)]
        torch.cuda.synchronize(device)
        detail["warmup_seconds"] = time.perf_counter() - started
        _check_training(model, warm_losses, name)
        del warm_losses
        torch.cuda.reset_peak_memory_stats(device)
        wall_samples, event_samples, last_losses = [], [], []
        for repeat in range(settings.repeats):
            print(f"gpu: {name}, trial {repeat + 1}/{settings.repeats}, {settings.steps} steps", flush=True)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            begin.record()
            losses = [step() for _ in range(settings.steps)]
            end.record()
            torch.cuda.synchronize(device)
            wall_samples.append(time.perf_counter() - started)
            event_samples.append(begin.elapsed_time(end) / 1000)
            _check_training(model, losses, name)
            last_losses.append(float(losses[-1].item()))
            del losses
        metrics = summarize(wall_samples, settings.steps * detail["batch_size"], "examples")
        metrics.update(step_ms=1000 * metrics["seconds_median"] / settings.steps,
                       steps_per_second=settings.steps / metrics["seconds_median"],
                       peak_allocated_MiB=torch.cuda.max_memory_allocated(device) / 2**20,
                       peak_reserved_MiB=torch.cuda.max_memory_reserved(device) / 2**20,
                       cuda_event_seconds_samples=event_samples, last_losses=last_losses)
        if name in ("pointnet", "mace"):
            metrics["cloud_views_per_second"] = 2 * metrics["examples_per_second"]
        result["workloads"][name] = dict(metrics=metrics, **detail)
        del model, optimizer, loss_fn, step
        torch.cuda.empty_cache()
    result["nvidia_smi_after"] = command_info(["nvidia-smi"])
    return result
