from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
)


def get_scheduler(optimizer, CFG, loader):
    if CFG.scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, **CFG.reduce_params)
    elif CFG.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, **CFG.cosanneal_params)
    elif CFG.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, **CFG.cosanneal_res_params)
    elif CFG.scheduler == "OneCycleLR":
        scheduler = OneCycleLR(
            optimizer=optimizer,
            epochs=CFG.epochs,
            anneal_strategy="cos",
            pct_start=0.05,
            steps_per_epoch=len(loader),
            max_lr=CFG.lr,
            final_div_factor=100,
        )
    return scheduler
