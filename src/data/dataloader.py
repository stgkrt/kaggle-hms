import os
import sys

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from config import CFG


class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        CFG: CFG,
        mode: str = "train",
        augment: bool = False,
    ):
        self.df = df
        self.augment = augment
        self.mode = mode
        self.spectograms = np.load(CFG.specs_path, allow_pickle=True).item()
        self.eeg_spectograms = np.load(CFG.eggs_path, allow_pickle=True).item()
        self.CFG = CFG

    def __len__(self) -> int:
        """
        Denotes the number of batches per epoch.
        """
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.tensor]:
        """
        Generate one batch of data.
        """
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)
        return {
            "spectrogram": torch.tensor(X, dtype=torch.float32),
            "labels": torch.tensor(y, dtype=torch.float32),
        }

    def __data_generation(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates data containing batch_size samples.
        """
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        img = np.ones((128, 256), dtype="float32")
        row = self.df.iloc[index]
        if self.mode == "test":
            r = 0
        else:
            r = int(row["spectrogram_label_offset_seconds"] // 2)

        for region in range(4):
            img = self.spectograms[row.spectrogram_id][
                r : r + 300, region * 100 : (region + 1) * 100
            ].T

            # Log transform spectogram
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # Standarize per image
            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0
            img = self.eeg_spectograms[row.label_id]
            X[:, :, 4:] = img

            if self.mode != "test":
                y = row[CFG.target_cols].values.astype(np.float32)

        return X, y

    def __transform(self, img: np.ndarray) -> np.ndarray:
        params1 = {
            "num_masks_x": 1,
            "mask_x_length": (0, 20),  # This line changed from fixed  to a range
            "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),
        }
        params2 = {
            "num_masks_y": 1,
            "mask_y_length": (0, 20),
            "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),
        }
        params3 = {
            "num_masks_x": (2, 4),
            "num_masks_y": 5,
            "mask_y_length": 8,
            "mask_x_length": (10, 20),
            "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),
        }

        transforms = A.Compose(
            [
                # A.HorizontalFlip(p=0.5),
                A.XYMasking(**params1, p=0.5),
                A.XYMasking(**params2, p=0.5),
                A.XYMasking(**params3, p=0.5),
            ]
        )
        return transforms(image=img)["image"]


class SpecKaggleDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        CFG: CFG,
        mode: str = "train",
        augment: bool = False,
    ):
        self.df = df
        self.augment = augment
        self.mode = mode
        self.spectograms = np.load(CFG.specs_path, allow_pickle=True).item()
        self.CFG = CFG

    def __len__(self) -> int:
        """
        Denotes the number of batches per epoch.
        """
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.tensor]:
        """
        Generate one batch of data.
        """
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)
        return {
            "spectrogram": torch.tensor(X, dtype=torch.float32),
            "labels": torch.tensor(y, dtype=torch.float32),
        }

    def __data_generation(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates data containing batch_size samples.
        """
        X = np.zeros((128, 256, 4), dtype="float32")
        y = np.zeros(6, dtype="float32")
        img = np.ones((128, 256), dtype="float32")
        row = self.df.iloc[index]
        if self.mode == "test":
            r = 0
        else:
            r = int(row["spectrogram_label_offset_seconds"] // 2)

        for region in range(4):
            img = self.spectograms[row.spectrogram_id][
                r : r + 300, region * 100 : (region + 1) * 100
            ].T

            # Log transform spectogram
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # Standarize per image
            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0

            if self.mode != "test":
                y = row[CFG.target_cols].values.astype(np.float32)

        return X, y

    def __transform(self, img: np.ndarray) -> np.ndarray:
        params1 = {
            "num_masks_x": 1,
            "mask_x_length": (0, 20),  # This line changed from fixed  to a range
            "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),
        }
        params2 = {
            "num_masks_y": 1,
            "mask_y_length": (0, 20),
            "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),
        }
        params3 = {
            "num_masks_x": (2, 4),
            "num_masks_y": 5,
            "mask_y_length": 8,
            "mask_x_length": (10, 20),
            "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),
        }

        transforms = A.Compose(
            [
                # A.HorizontalFlip(p=0.5),
                A.XYMasking(**params1, p=0.5),
                A.XYMasking(**params2, p=0.5),
                A.XYMasking(**params3, p=0.5),
            ]
        )
        return transforms(image=img)["image"]


if __name__ == "__main__":
    from src.config import CFG

    config = CFG()
    train = pd.read_csv(config.train_csv)
    train = train[
        train["label_id"].isin(
            np.load(config.eggs_path, allow_pickle=True).item().keys()
        )
    ].copy()
    train = train.reset_index(drop=True)
    train = train.iloc[:10]
    dataset = CustomDataset(train, config, mode="train", augment=True)
    # for i in range(10):
    #     print(dataset[i])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0
    )
    for x in dataloader:
        print(x["spectrogram"].shape)
        print(x["labels"].shape)
        break
    print("Done")
