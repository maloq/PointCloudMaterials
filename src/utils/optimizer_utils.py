import torch

def get_optimizers_and_scheduler(module):
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=module.hparams.learning_rate,
        weight_decay=module.hparams.decay_rate
    )

    if module.hparams.enable_swa:
        epochs_before_swa = module.hparams.swa_epoch_start + 1
    else:
        epochs_before_swa = module.hparams.epochs

    scheduler_name = module.hparams.scheduler_name
    if scheduler_name == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100, 
            gamma=module.hparams.scheduler_gamma
        )
    elif scheduler_name == 'OneCycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=module.hparams.learning_rate, 
            total_steps=epochs_before_swa
        )
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20, 
            T_mult=3
        )
    elif scheduler_name == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs_before_swa,
            eta_min=0 
        )
    elif scheduler_name == 'chained':
        scheduler1 = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=module.hparams.learning_rate, 
            total_steps=int(epochs_before_swa/2)
        )
        scheduler2 = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=module.hparams.learning_rate, 
            total_steps=int(epochs_before_swa/2)
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
    
    else:
        raise ValueError(f"Scheduler {scheduler_name} not found")

    return [optimizer], [scheduler] 