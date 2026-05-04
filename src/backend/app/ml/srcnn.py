from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

SRCNN_MODEL_SCALE = 4
SRCNN_NUM_CHANNELS = 3


class SRCNN(nn.Module):
    def __init__(self, num_channels: int = SRCNN_NUM_CHANNELS) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)


def normalize_state_dict_keys(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def extract_srcnn_checkpoint(checkpoint: object) -> tuple[dict[str, Tensor], dict[str, Any]]:
    if isinstance(checkpoint, dict):
        model_state = checkpoint.get("model_state")
        if isinstance(model_state, dict):
            return model_state, checkpoint
        if checkpoint and all(isinstance(value, Tensor) for value in checkpoint.values()):
            return checkpoint, {}
    raise ValueError("Unsupported SRCNN checkpoint format.")


def adapt_bgr_checkpoint_to_rgb_runtime(
    state_dict: dict[str, Tensor],
) -> dict[str, Tensor]:
    adapted = dict(state_dict)
    if adapted.get("conv1.weight") is not None:
        adapted["conv1.weight"] = adapted["conv1.weight"][:, [2, 1, 0], :, :]
    if adapted.get("conv3.weight") is not None:
        adapted["conv3.weight"] = adapted["conv3.weight"][[2, 1, 0], :, :, :]
    if adapted.get("conv3.bias") is not None:
        adapted["conv3.bias"] = adapted["conv3.bias"][[2, 1, 0]]
    return adapted


def load_srcnn_rgb(
    weights_path: Path,
    device: torch.device,
    use_half: bool,
    use_channels_last: bool,
) -> tuple[SRCNN, dict[str, Any]]:
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict, metadata = extract_srcnn_checkpoint(checkpoint)
    state_dict = normalize_state_dict_keys(state_dict)
    state_dict = adapt_bgr_checkpoint_to_rgb_runtime(state_dict)

    model = SRCNN(num_channels=SRCNN_NUM_CHANNELS)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)
    if use_channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if use_half and device.type == "cuda":
        model = model.half()

    return model, {
        "checkpoint_key": "model_state" if metadata else "raw",
        "predict_residual": bool(metadata.get("predict_residual", True)),
        "checkpoint_color_space": metadata.get("color_space", "BGR/RGB"),
        "runtime_color_space": "RGB",
        "network_scale": int(metadata.get("scale_factor", SRCNN_MODEL_SCALE)),
    }
