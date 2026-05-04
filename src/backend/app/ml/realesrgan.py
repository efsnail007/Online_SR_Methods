from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init as nn_init

MODEL_SCALE = 4
MODEL_NUM_BLOCKS = 23
MODEL_NUM_FEATURES = 64
MODEL_GROWTH_CHANNELS = 32


@torch.no_grad()
def initialize_weights(
    module_list: list[nn.Module] | nn.Module, scale: float = 1.0
) -> None:
    modules = module_list if isinstance(module_list, list) else [module_list]
    for module in modules:
        for submodule in module.modules():
            if isinstance(submodule, nn.Conv2d):
                nn_init.kaiming_normal_(submodule.weight)
                submodule.weight.mul_(scale)
                if submodule.bias is not None:
                    submodule.bias.zero_()
            elif isinstance(submodule, nn.Linear):
                nn_init.kaiming_normal_(submodule.weight)
                submodule.weight.mul_(scale)
                if submodule.bias is not None:
                    submodule.bias.zero_()


def make_layer(block: type[nn.Module], count: int, **kwargs) -> nn.Sequential:
    return nn.Sequential(*(block(**kwargs) for _ in range(count)))


def pixel_unshuffle(x: Tensor, scale: int) -> Tensor:
    batch, channels, height, width = x.size()
    output_channels = channels * (scale**2)
    reduced_height = height // scale
    reduced_width = width // scale
    reshaped = x.view(batch, channels, reduced_height, scale, reduced_width, scale)
    return reshaped.permute(0, 1, 3, 5, 2, 4).reshape(
        batch, output_channels, reduced_height, reduced_width
    )


class ResidualDenseBlock(nn.Module):
    def __init__(
        self,
        num_features: int = MODEL_NUM_FEATURES,
        growth_channels: int = MODEL_GROWTH_CHANNELS,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_features + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(
            num_features + 2 * growth_channels, growth_channels, 3, 1, 1
        )
        self.conv4 = nn.Conv2d(
            num_features + 3 * growth_channels, growth_channels, 3, 1, 1
        )
        self.conv5 = nn.Conv2d(
            num_features + 4 * growth_channels, num_features, 3, 1, 1
        )
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        initialize_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], scale=0.1
        )

    def forward(self, x: Tensor) -> Tensor:
        level1 = self.activation(self.conv1(x))
        level2 = self.activation(self.conv2(torch.cat((x, level1), dim=1)))
        level3 = self.activation(self.conv3(torch.cat((x, level1, level2), dim=1)))
        level4 = self.activation(
            self.conv4(torch.cat((x, level1, level2, level3), dim=1))
        )
        level5 = self.conv5(torch.cat((x, level1, level2, level3, level4), dim=1))
        return x + level5 * 0.2


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(
        self,
        num_features: int = MODEL_NUM_FEATURES,
        growth_channels: int = MODEL_GROWTH_CHANNELS,
    ) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_features, growth_channels)
        self.rdb2 = ResidualDenseBlock(num_features, growth_channels)
        self.rdb3 = ResidualDenseBlock(num_features, growth_channels)

    def forward(self, x: Tensor) -> Tensor:
        output = self.rdb1(x)
        output = self.rdb2(output)
        output = self.rdb3(output)
        return x + output * 0.2


class RRDBNet(nn.Module):
    def __init__(
        self,
        num_in_channels: int,
        num_out_channels: int,
        scale: int = MODEL_SCALE,
        num_features: int = MODEL_NUM_FEATURES,
        num_blocks: int = MODEL_NUM_BLOCKS,
        growth_channels: int = MODEL_GROWTH_CHANNELS,
    ) -> None:
        super().__init__()
        self.scale = scale
        first_layer_channels = num_in_channels
        if scale == 2:
            first_layer_channels *= 4
        elif scale == 1:
            first_layer_channels *= 16

        self.conv_first = nn.Conv2d(first_layer_channels, num_features, 3, 1, 1)
        self.body = make_layer(
            ResidualInResidualDenseBlock,
            num_blocks,
            num_features=num_features,
            growth_channels=growth_channels,
        )
        self.conv_body = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_features, num_out_channels, 3, 1, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.scale == 2:
            features = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            features = pixel_unshuffle(x, scale=4)
        else:
            features = x

        features = self.conv_first(features)
        body_features = self.conv_body(self.body(features))
        features = features + body_features
        features = self.activation(
            self.conv_up1(F.interpolate(features, scale_factor=2, mode="nearest"))
        )
        features = self.activation(
            self.conv_up2(F.interpolate(features, scale_factor=2, mode="nearest"))
        )
        features = self.activation(self.conv_hr(features))
        return self.conv_last(features)


def build_realesrgan_x4plus() -> RRDBNet:
    return RRDBNet(
        num_in_channels=3,
        num_out_channels=3,
        scale=MODEL_SCALE,
        num_features=MODEL_NUM_FEATURES,
        num_blocks=MODEL_NUM_BLOCKS,
        growth_channels=MODEL_GROWTH_CHANNELS,
    )


def extract_state_dict(checkpoint: object) -> tuple[dict[str, Tensor], str]:
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate, key
        if checkpoint and all(
            isinstance(value, Tensor) for value in checkpoint.values()
        ):
            return checkpoint, "raw"
    raise ValueError("Unsupported Real-ESRGAN checkpoint format.")


def normalize_state_dict_keys(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_realesrgan_x4plus(
    weights_path: Path,
    device: torch.device,
    use_half: bool,
    use_channels_last: bool,
) -> tuple[RRDBNet, str]:
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict, checkpoint_key = extract_state_dict(checkpoint)
    model = build_realesrgan_x4plus()
    model.load_state_dict(normalize_state_dict_keys(state_dict), strict=True)
    model.eval()
    model = model.to(device)
    if use_channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if use_half and device.type == "cuda":
        model = model.half()
    return model, checkpoint_key
