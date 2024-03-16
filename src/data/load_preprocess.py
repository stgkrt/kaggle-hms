import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import CFG


def split_cv(train: pd.DataFrame, CFG: CFG) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=CFG.n_fold)

    train["fold"] = -1

    for fold_id, (_, val_idx) in enumerate(
        gkf.split(train, y=train["target"], groups=train["patient_id"])
    ):
        train.loc[val_idx, "fold"] = fold_id

    return train


def load_and_preprocess(CFG: CFG) -> pd.DataFrame:
    train = pd.read_csv(CFG.train_csv)
    TARGETS = train.columns[-6:]
    print("Train shape:", train.shape)
    print("Targets", list(TARGETS))

    train["total_evaluators"] = train[
        ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
    ].sum(axis=1)

    print(f"There are {train.patient_id.nunique()} patients in the training data.")
    print(f"There are {train.eeg_id.nunique()} EEG IDs in the training data.")

    all_eegs = np.load(CFG.eggs_path, allow_pickle=True).item()
    train = train[train["label_id"].isin(all_eegs.keys())].copy()

    y_data = train[CFG.target_cols].values  # Regularization value
    y_data = y_data / y_data.sum(axis=1, keepdims=True)
    train[TARGETS] = y_data

    train["target"] = train["expert_consensus"]

    train = train.reset_index(drop=True)
    train = split_cv(train, CFG)

    return train


if __name__ == "__main__":
    from src.config import CFG

    config = CFG()
    train = load_and_preprocess(config)
    print(train.head())
    print(train.columns)
