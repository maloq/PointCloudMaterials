import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# IDEC workflow example:
# 1) Pretrain AE with idec_enabled=False for N epochs
# 2) model.init_idec_centers_from_kmeans(trainer.datamodule.train_dataloader())
# 3) Finetune with idec_enabled=True, idec_gamma=0.1, idec_use_global_p=True, idec_update_interval=T


def resolve_latent_dim(cfg):
    if hasattr(cfg, "latent_size"):
        return int(cfg.latent_size)
    if hasattr(cfg, "encoder") and hasattr(cfg.encoder, "kwargs"):
        latent_size = cfg.encoder.kwargs.get("latent_size", None)
        if latent_size is not None:
            return int(latent_size)
    return None


class IDECClusteringHead(nn.Module):
    def __init__(self, num_clusters: int, embed_dim: int, eps: float = 1e-8):
        super().__init__()
        self.num_clusters = int(num_clusters)
        self.embed_dim = int(embed_dim)
        self.eps = float(eps)
        self.centers = nn.Parameter(torch.empty(self.num_clusters, self.embed_dim))
        nn.init.normal_(self.centers, std=0.02)

    def soft_assign_q(self, z: torch.Tensor) -> torch.Tensor:
        diff = z.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = (diff * diff).sum(dim=2)
        q = 1.0 / (1.0 + dist_sq)
        q = q / (q.sum(dim=1, keepdim=True) + self.eps)
        return q

    def target_distribution_p(self, q: torch.Tensor) -> torch.Tensor:
        f_j = q.sum(dim=0)
        numerator = (q * q) / (f_j + self.eps)
        p = numerator / (numerator.sum(dim=1, keepdim=True) + self.eps)
        return p

    def kl_pq(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p_safe = p.clamp(min=self.eps)
        q_safe = q.clamp(min=self.eps)
        return torch.mean(torch.sum(p_safe * torch.log(p_safe / q_safe), dim=1))


class IDECManager(nn.Module):
    def __init__(
        self,
        *,
        enabled: bool,
        num_clusters: int,
        gamma: float,
        update_interval: int,
        delta: float,
        max_iter,
        use_global_p: bool,
        normalize_z: bool,
        init_kmeans: bool,
        embed_dim,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.num_clusters = int(num_clusters)
        self.gamma = float(gamma)
        self.update_interval = int(update_interval)
        self.delta = float(delta)
        self.max_iter = max_iter
        self.use_global_p = bool(use_global_p)
        self.normalize_z = bool(normalize_z)
        self.init_kmeans = bool(init_kmeans)

        self.head = None
        if self.num_clusters > 0:
            if embed_dim is None:
                raise ValueError("IDEC requires latent_size to set clustering embed_dim")
            self.head = IDECClusteringHead(
                num_clusters=self.num_clusters,
                embed_dim=int(embed_dim),
            )
        elif self.enabled:
            raise ValueError("idec_num_clusters must be set when idec_enabled=True")

        self._p_cache = None
        self._p_cache_is_tensor = False
        self._p_cache_ready = False
        self._prev_labels = None
        self._last_p_update_step = None
        self._last_change_rate = None
        self._should_stop = False
        self._cached_train_loader = None
        self._global_p_invalid = False

    @classmethod
    def from_config(cls, cfg, *, embed_dim):
        return cls(
            enabled=bool(getattr(cfg, "idec_enabled", False)),
            num_clusters=int(getattr(cfg, "idec_num_clusters", 0) or 0),
            gamma=float(getattr(cfg, "idec_gamma", 0.1)),
            update_interval=int(getattr(cfg, "idec_update_interval", 200)),
            delta=float(getattr(cfg, "idec_delta", 0.001)),
            max_iter=getattr(cfg, "idec_max_iter", None),
            use_global_p=bool(getattr(cfg, "idec_use_global_p", True)),
            normalize_z=bool(getattr(cfg, "idec_normalize_z", True)),
            init_kmeans=bool(getattr(cfg, "idec_init_kmeans", True)),
            embed_dim=embed_dim,
        )

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.head is not None)

    def compute_loss(self, inv_z: torch.Tensor, meta: dict):
        if not self.active or inv_z is None or self.head is None:
            return None, None
        z = inv_z
        if self.normalize_z:
            z = F.normalize(z, dim=1)
        q = self.head.soft_assign_q(z).float()
        p = None
        if self.use_global_p and self._p_cache_ready:
            p = self._get_p_for_batch(meta.get("instance_id"), device=q.device)
        if p is None:
            p = self.head.target_distribution_p(q).detach()
        p = p.to(device=q.device, dtype=q.dtype)
        cluster_kl = self.head.kl_pq(p, q)
        q_entropy = (-q * q.clamp_min(self.head.eps).log()).sum(dim=1).mean()
        return cluster_kl, q_entropy

    def on_train_step_start(
        self,
        *,
        global_step: int,
        module,
        trainer,
        encoder,
        prepare_input,
        split_output,
        unpack_batch,
        device,
        dtype,
    ):
        if not self.active:
            return {}
        if self.max_iter is not None:
            try:
                max_iter = int(self.max_iter)
                if max_iter > 0 and int(global_step) >= max_iter:
                    self._should_stop = True
                    if trainer is not None:
                        trainer.should_stop = True
            except (TypeError, ValueError):
                pass
        change_rate = self._maybe_update_targets(
            global_step=global_step,
            module=module,
            trainer=trainer,
            encoder=encoder,
            prepare_input=prepare_input,
            split_output=split_output,
            unpack_batch=unpack_batch,
            device=device,
            dtype=dtype,
        )
        if change_rate is None:
            return {}
        return {"idec_label_change_rate": change_rate}

    def _maybe_update_targets(
        self,
        *,
        global_step: int,
        module,
        trainer,
        encoder,
        prepare_input,
        split_output,
        unpack_batch,
        device,
        dtype,
    ):
        if not self.active:
            return None
        if not self.use_global_p or self._global_p_invalid:
            return None
        if self.update_interval <= 0:
            return None
        if self._should_stop:
            return None
        step = int(global_step)
        if self._last_p_update_step is None or (step - self._last_p_update_step) >= self.update_interval:
            return self._update_p_cache(
                module=module,
                trainer=trainer,
                encoder=encoder,
                prepare_input=prepare_input,
                split_output=split_output,
                unpack_batch=unpack_batch,
                device=device,
                dtype=dtype,
                global_step=step,
            )
        return None

    @torch.no_grad()
    def _update_p_cache(
        self,
        *,
        module,
        trainer,
        encoder,
        prepare_input,
        split_output,
        unpack_batch,
        device,
        dtype,
        global_step: int,
    ):
        loader = self._get_train_dataloader(trainer)
        if loader is None:
            return None
        was_training = module.training if module is not None else encoder.training
        if module is not None:
            module.eval()
        else:
            encoder.eval()
        q_batches = []
        id_batches = []
        for batch in loader:
            pc, meta = unpack_batch(batch)
            pc = pc.to(device=device, dtype=dtype, non_blocking=True)
            enc_out = encoder(prepare_input(pc))
            inv_z, _ = split_output(enc_out)
            if inv_z is None:
                continue
            z = inv_z
            if self.normalize_z:
                z = F.normalize(z, dim=1)
            q = self.head.soft_assign_q(z)
            q_batches.append(q.detach().to(torch.float32).cpu())
            ids = meta.get("instance_id")
            if ids is None:
                ids = torch.arange(z.shape[0], dtype=torch.long)
            elif not torch.is_tensor(ids):
                ids = torch.as_tensor(ids, dtype=torch.long)
            ids = ids.view(-1).detach().cpu()
            id_batches.append(ids)
        if was_training:
            if module is not None:
                module.train()
            else:
                encoder.train()
        if not q_batches:
            self._p_cache_ready = False
            return None
        q_all = torch.cat(q_batches, dim=0)
        ids_all = torch.cat(id_batches, dim=0)
        if (ids_all < 0).any():
            self._global_p_invalid = True
            self._p_cache_ready = False
            return None
        p_all = self.head.target_distribution_p(q_all).detach()
        self._set_p_cache(ids_all, p_all)
        labels = torch.argmax(q_all, dim=1)
        change_rate = None
        if self._prev_labels is not None and self._prev_labels.numel() == labels.numel():
            change_rate = (self._prev_labels != labels).float().mean().item()
            self._last_change_rate = change_rate
            if self.delta > 0 and change_rate < self.delta:
                self._should_stop = True
                if trainer is not None:
                    trainer.should_stop = True
        self._prev_labels = labels
        self._last_p_update_step = int(global_step)
        return change_rate

    def _set_p_cache(self, instance_ids: torch.Tensor, p_all: torch.Tensor) -> None:
        instance_ids = instance_ids.to(torch.long).view(-1)
        p_all = p_all.to(torch.float32)
        if instance_ids.numel() == 0:
            self._p_cache = None
            self._p_cache_is_tensor = False
            self._p_cache_ready = False
            return
        unique_ids, _ = torch.unique(instance_ids, return_counts=True)
        has_duplicates = unique_ids.numel() != instance_ids.numel()
        min_id = int(unique_ids.min().item())
        max_id = int(unique_ids.max().item())
        contiguous = min_id >= 0 and (max_id + 1 == unique_ids.numel()) and not has_duplicates
        if contiguous:
            cache = torch.zeros((max_id + 1, p_all.shape[1]), dtype=torch.float32)
            cache[instance_ids] = p_all
            self._p_cache = cache
            self._p_cache_is_tensor = True
            self._p_cache_ready = True
            return
        sums = {}
        counts_map = {}
        for idx, key in enumerate(instance_ids.tolist()):
            key = int(key)
            if key not in sums:
                sums[key] = p_all[idx].clone()
                counts_map[key] = 1
            else:
                sums[key] += p_all[idx]
                counts_map[key] += 1
        cache = {key: sums[key] / float(counts_map[key]) for key in sums}
        self._p_cache = cache
        self._p_cache_is_tensor = False
        self._p_cache_ready = True

    def _get_p_for_batch(self, instance_ids, *, device):
        if instance_ids is None or self._p_cache is None or not self._p_cache_ready:
            return None
        if not torch.is_tensor(instance_ids):
            instance_ids = torch.as_tensor(instance_ids, dtype=torch.long)
        instance_ids = instance_ids.view(-1)
        if instance_ids.numel() == 0 or (instance_ids < 0).any():
            return None
        if self._p_cache_is_tensor:
            ids_cpu = instance_ids.to(torch.long).cpu()
            if ids_cpu.max().item() >= self._p_cache.shape[0]:
                return None
            p = self._p_cache[ids_cpu]
            return p.to(device=device)
        p_list = []
        for key in instance_ids.tolist():
            val = self._p_cache.get(int(key))
            if val is None:
                return None
            p_list.append(val)
        return torch.stack(p_list, dim=0).to(device=device)

    def _clone_dataloader_no_drop(self, loader):
        if not isinstance(loader, DataLoader):
            return loader
        kwargs = dict(
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            drop_last=False,
            collate_fn=loader.collate_fn,
            worker_init_fn=loader.worker_init_fn,
            timeout=loader.timeout,
            generator=loader.generator,
        )
        if loader.num_workers > 0:
            kwargs["persistent_workers"] = loader.persistent_workers
            prefetch_factor = getattr(loader, "prefetch_factor", None)
            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(loader.dataset, **kwargs)

    def _get_train_dataloader(self, trainer):
        if self._cached_train_loader is not None:
            return self._cached_train_loader
        if trainer is None:
            return None
        loader = None
        dm = getattr(trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "train_dataloader"):
            try:
                loader = dm.train_dataloader()
            except Exception:
                loader = None
        if loader is None:
            loader = getattr(trainer, "train_dataloader", None)
        if isinstance(loader, (list, tuple)):
            loader = loader[0] if loader else None
        if loader is None:
            return None
        loader = self._clone_dataloader_no_drop(loader)
        self._cached_train_loader = loader
        return loader

    def _run_kmeans(self, z: torch.Tensor, num_clusters: int, *, kmeans_kwargs=None) -> torch.Tensor:
        kmeans_kwargs = {} if kmeans_kwargs is None else dict(kmeans_kwargs)
        z_np = z.detach().cpu().numpy()
        try:
            from sklearn.cluster import KMeans

            n_init = kmeans_kwargs.pop("n_init", 10)
            random_state = kmeans_kwargs.pop("random_state", 0)
            kmeans = KMeans(
                n_clusters=num_clusters,
                n_init=n_init,
                random_state=random_state,
                **kmeans_kwargs,
            )
            kmeans.fit(z_np)
            centers = torch.from_numpy(kmeans.cluster_centers_).to(dtype=z.dtype)
            return centers
        except Exception as exc:
            print(f"Falling back to torch k-means: {exc}")
        return self._torch_kmeans(z, num_clusters)

    def _torch_kmeans(self, z: torch.Tensor, num_clusters: int, num_iters: int = 20) -> torch.Tensor:
        if z.shape[0] < num_clusters:
            raise ValueError("Not enough samples to initialize k-means centers")
        z = z.to(torch.float32)
        indices = torch.randperm(z.shape[0], device=z.device)[:num_clusters]
        centers = z[indices].clone()
        for _ in range(num_iters):
            diff = z.unsqueeze(1) - centers.unsqueeze(0)
            dist_sq = (diff * diff).sum(dim=2)
            labels = dist_sq.argmin(dim=1)
            new_centers = centers.clone()
            for k in range(num_clusters):
                mask = labels == k
                if mask.any():
                    new_centers[k] = z[mask].mean(dim=0)
            if torch.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers
        return centers

    @torch.no_grad()
    def init_centers_from_kmeans(
        self,
        *,
        module,
        encoder,
        prepare_input,
        split_output,
        unpack_batch,
        device,
        dtype,
        train_dataloader=None,
        trainer=None,
        max_samples=None,
        kmeans_kwargs=None,
    ):
        if self.head is None:
            raise ValueError("IDEC head is not initialized")
        loader = train_dataloader or self._get_train_dataloader(trainer)
        if loader is None:
            raise ValueError("Training dataloader is required for k-means initialization")
        was_training = module.training if module is not None else encoder.training
        if module is not None:
            module.eval()
        else:
            encoder.eval()
        z_batches = []
        total = 0
        for batch in loader:
            pc, meta = unpack_batch(batch)
            pc = pc.to(device=device, dtype=dtype, non_blocking=True)
            enc_out = encoder(prepare_input(pc))
            inv_z, _ = split_output(enc_out)
            if inv_z is None:
                continue
            z = inv_z
            if self.normalize_z:
                z = F.normalize(z, dim=1)
            z_batches.append(z.detach().to(torch.float32).cpu())
            total += int(z.shape[0])
            if max_samples is not None and total >= max_samples:
                break
        if was_training:
            if module is not None:
                module.train()
            else:
                encoder.train()
        if not z_batches:
            raise RuntimeError("No latents collected for k-means initialization")
        z_all = torch.cat(z_batches, dim=0)
        if max_samples is not None and z_all.shape[0] > max_samples:
            z_all = z_all[:max_samples]
        centers = self._run_kmeans(z_all, self.num_clusters, kmeans_kwargs=kmeans_kwargs)
        self.head.centers.data.copy_(centers.to(self.head.centers.device))
        return centers
