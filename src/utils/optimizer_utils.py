import torch

def get_optimizers_and_scheduler(module):
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=module.hparams.learning_rate,
        weight_decay=module.hparams.decay_rate
    )
    
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
            total_steps=module.hparams.epochs
        )
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20, 
            T_mult=3
        )
    elif scheduler_name == 'chained':
        scheduler2 = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=module.hparams.learning_rate, 
            total_steps=module.hparams.epochs
        )
        scheduler1 = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=module.hparams.learning_rate, 
            total_steps=module.hparams.epochs
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
    else:
        raise ValueError(f"Scheduler {scheduler_name} not found")

    return [optimizer], [scheduler] 