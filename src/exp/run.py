import os

import numpy as np
import pandas as pd
import wandb

from src.config import CFG
from src.data.load_preprocess import load_and_preprocess
from src.exp.train import train_loop
from src.log_utils import init_logger

if __name__ == "__main__":
    CFG = CFG()
    os.makedirs(CFG.POP_1_DIR, exist_ok=True)
    logger_path = os.path.join(CFG.POP_1_DIR, "train.log")
    LOGGER = init_logger(logger_path)
    if CFG.train:
        oof_df = pd.DataFrame()
        train = load_and_preprocess(CFG)
        scores = []
        for fold in range(CFG.n_fold):
            print(f"fold: {fold} start")
            if fold in CFG.trn_fold:
                _oof_df, score = train_loop(train, fold, CFG.POP_1_DIR, LOGGER, CFG)
                oof_df = pd.concat([oof_df, _oof_df])
                scores.append(score)
                LOGGER.info(f"========== fold: {fold} result ==========")
                LOGGER.info(f"Score with best loss weights stage1: {score}")
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info("========== CV ==========")
        LOGGER.info(f"Score with best loss weights stage1: {np.mean(scores)}")
        oof_df.to_csv(
            CFG.POP_1_DIR + f"{CFG.model_name}_oof_df_version{CFG.VERSION}_stage1.csv",
            index=False,
        )

    if CFG.wandb:
        wandb.finish()
