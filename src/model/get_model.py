import timm
import torch
from torch import nn

from src.config import CFG


class CustomModel(nn.Module):
    def __init__(
        self, config: CFG, num_classes: int = 6, pretrained: bool = True
    ) -> None:
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(
            config.model_name,
            pretrained=pretrained,
            in_chans=1,
        )
        if CFG.FREEZE:
            total_params = sum(p.numel() for p in self.model.parameters())
            percentage_to_freeze = 0.1
            for i, (name, param) in enumerate(
                list(self.model.named_parameters())[
                    0 : int(total_params * percentage_to_freeze)
                ]
            ):
                param.requires_grad = False

        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
        )

    def __reshape_input(self, x: torch.tensor) -> torch.tensor:
        """
        Reshapes input (128, 256, 8) -> (512, 512, 3) monotone image.
        """
        # === Get spectrograms ===
        spectrograms = [x[:, :, :, i : i + 1] for i in range(4)]
        spectrograms = torch.cat(spectrograms, dim=1)
        # 1stepずらしたものとdiffを取る
        # spectrograms_diff = spectrograms - torch.roll(spectrograms, shifts=1, dims=2)
        # spectrograms = torch.cat([spectrograms, spectrograms_diff], dim=-1)
        # # width方向にsplit num分割してmeanを取る
        # split_num = 16
        # window_size = spectrograms.shape[2] // split_num
        # spectrogram_windows = [
        #     spectrograms[:, :, i * window_size : (i + 1) * window_size, :]
        #     for i in range(split_num)
        # ]
        # spectrograms_mean = torch.stack(
        #     [torch.mean(x, dim=2) for x in spectrogram_windows], dim=2
        # )
        # # 元の波形とwindowのdiffを取る
        # spectrograms_mean_tile = spectrograms_mean.repeat(1, 1, split_num, 1)
        # spectrograms_diff = spectrograms_mean_tile - spectrograms
        # spectrograms = torch.cat([spectrograms, spectrograms_diff], dim=-1)

        # === Get EEG spectrograms ===
        eegs = [x[:, :, :, i : i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=1)
        # 1stepずらしたものとdiffを取る
        # eegs_diff = eegs - torch.roll(eegs, shifts=1, dims=2)
        # eegs = torch.cat([eegs, eegs_diff], dim=-1)
        # # width方向にsplit num分割してmeanを取る
        # eegs_windows = [
        #     eegs[:, :, i * window_size : (i + 1) * window_size, :]
        #     for i in range(split_num)
        # ]
        # eegs_mean = torch.stack([torch.mean(x, dim=2) for x in eegs_windows], dim=2)
        # # 元の波形とwindowのdiffを取る
        # eegs_mean_tile = eegs_mean.repeat(1, 1, split_num, 1)
        # eegs_diff = eegs_mean_tile - eegs
        # eegs = torch.cat([eegs, eegs_diff], dim=-1)

        # 時間方向にspectrogramとeegをconcat
        if self.USE_KAGGLE_SPECTROGRAMS & self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectrograms, eegs], dim=2)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eegs
        else:
            x = spectrograms

        # x = torch.cat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)
        return x

    def extract_features(self, x: torch.tensor) -> torch.tensor:
        x = self.__reshape_input(x)
        x = self.features(x)
        return x

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.extract_features(x)
        x = self.custom_layers(x)
        return x


class CustomDiffModel(nn.Module):
    def __init__(
        self, config: CFG, num_classes: int = 6, pretrained: bool = True
    ) -> None:
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(
            config.model_name,
            pretrained=pretrained,
            in_chans=1,
        )
        if CFG.FREEZE:
            total_params = sum(p.numel() for p in self.model.parameters())
            percentage_to_freeze = 0.1
            for i, (name, param) in enumerate(
                list(self.model.named_parameters())[
                    0 : int(total_params * percentage_to_freeze)
                ]
            ):
                param.requires_grad = False

        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
        )

    def __reshape_input(self, x: torch.tensor) -> torch.tensor:
        """
        Reshapes input (128, 256, 8) -> (512, 512, 3) monotone image.
        """
        # === Get spectrograms ===
        spectrograms = [x[:, :, :, i : i + 1] for i in range(4)]
        spectrograms = torch.cat(spectrograms, dim=1)

        # === Get EEG spectrograms ===
        eegs = [x[:, :, :, i : i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=1)

        # 時間方向にspectrogramとeegをconcat
        if self.USE_KAGGLE_SPECTROGRAMS & self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectrograms, eegs], dim=2)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eegs
        else:
            x = spectrograms

        # x = torch.cat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)
        return x

    def extract_features(self, x: torch.tensor) -> torch.tensor:
        x = self.__reshape_input(x)
        x = self.features(x)
        return x

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.extract_features(x)
        x = self.custom_layers(x)
        return x


if __name__ == "__main__":
    cfg = CFG()
    model = CustomModel(cfg)
    x = torch.rand((1, 128, 256, 8))
    output = model(x)
    print(output.shape)
