import os
from datetime import datetime

import numpy as np
import pandas as pd

import wandb
from src.config import CFG
from src.data.load_preprocess import load_and_preprocess
from src.exp.train import train_loop
from src.log_utils import init_logger, init_wandb

if __name__ == "__main__":
    CFG = CFG()
    output_dir = os.path.join(CFG.OUTPUT_DIR, CFG.exp_name)
    if os.path.exists(output_dir) and CFG.exp_name != "debug":
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = output_dir + f"_{now}"
    os.makedirs(output_dir, exist_ok=True)
    logger_path = os.path.join(output_dir, "train.log")
    LOGGER = init_logger(logger_path)
    if CFG.wandb:
        init_wandb(CFG)
    if CFG.train:
        oof_df = pd.DataFrame()
        train = load_and_preprocess(CFG)
        scores = []
        for fold in range(CFG.n_fold):
            print(f"fold: {fold} start")
            if fold in CFG.trn_fold:
                _oof_df, score = train_loop(train, fold, output_dir, LOGGER, CFG)
                oof_df = pd.concat([oof_df, _oof_df])
                scores.append(score)
                LOGGER.info(f"========== fold: {fold} result ==========")
                LOGGER.info(f"Score with best loss weights : {score}")
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info("========== CV ==========")
        for fold, _score in enumerate(scores):
            LOGGER.info(f"fold{fold} score: {_score}")
        LOGGER.info(f"Score mean: {np.mean(scores)}")

        oof_df_path = os.path.join(
            output_dir, f"{CFG.model_name}_oof_df_{CFG.exp_name}_stage1.csv"
        )
        oof_df.to_csv(oof_df_path, index=False)

    if CFG.wandb:
        wandb.finish()
