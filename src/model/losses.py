from typing import Callable

import numpy as np
import torch
from torch import nn

from src.config import CFG


class KLDivLossWithLogits(nn.KLDivLoss):
    def __init__(self) -> None:
        super().__init__(reduction="batchmean")

    def forward(self, y: torch.tensor, t: torch.tensor) -> torch.tensor:
        y = nn.functional.log_softmax(y, dim=1)
        loss = super().forward(y, t)

        return loss


class KLDivBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha: float = 0.2) -> None:
        super(KLDivBCEWithLogitsLoss, self).__init__()
        self.alpha = alpha
        self.kl_div = KLDivLossWithLogits()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y: torch.tensor, t: torch.tensor) -> torch.tensor:
        kl_div_loss = self.kl_div(y, t)
        t_bin = torch.tensor(t > 0.1, dtype=torch.float32).to(CFG.device)
        bce_loss = self.bce(y, t_bin)
        loss = self.alpha * kl_div_loss + (1 - self.alpha) * bce_loss

        return loss


def mixup_data(
    X: torch.Tensor, y: torch.Tensor, config: CFG, device: str, alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if not config.USE_MIXUP:
        return X, y, torch.empty(0), 0.0

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)  # draw random number from beta distribution
    else:
        lam = 1.0

    batch_size = X.size()[0]
    index = torch.randperm(batch_size).to(
        device
    )  # torch tensor with shuffled numbers between 1:batch_size
    mixed_X = lam * X + (1 - lam) * X[index, :]  # perform mixup to the whole batch
    y_a, y_b = y, y[index]
    return mixed_X, y_a, y_b, lam


def get_criterion(config: CFG, criterion: Callable) -> Callable:
    """
    This function computes the criterion/loss depending
    whether MixUp augmentation was applied or not.
    If MixUp was applied it returns a weighted average of the loss
    averaging by the lambda parameter.
    Otherwise, it returns the regular loss as MixUp was not applied.
    :param config: configuration class with param to use mixup or not.
    :param criterion: loss function to use.
    """

    def mixup_criterion(
        pred: torch.tensor, y_a: float, y_b: float, lam: float
    ) -> torch.tensor:
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def single_criterion(
        pred: torch.tensor, y_a: float, y_b: float, lam: float
    ) -> torch.tensor:
        return criterion(pred, y_a)

    if config.USE_MIXUP:
        return mixup_criterion
    else:
        return single_criterion
