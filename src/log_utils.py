import math
import os
import time
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

import wandb
from src.config import CFG


def init_wandb(configs: CFG):
    if configs.wandb:
        os.environ["WANDB_SILENT"] = "true"
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project=configs.competition_name,
            config=configs,
            group=configs.exp_category,
            name=configs.exp_name,
            reinit=True,
            save_code=True,
            tags=[configs.model_name],
        )


def init_logger(log_file: str = "/kaggle/working/log/train.log") -> getLogger:
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s: int) -> str:
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))
