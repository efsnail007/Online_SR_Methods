from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F


def _bgr_u8_to_torch(image_bgr_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image_bgr_u8).to(device=device, dtype=torch.float32)
    return tensor.permute(2, 0, 1).unsqueeze(0)


def _torch_to_bgr_u8(image_bgr: torch.Tensor) -> np.ndarray:
    if image_bgr.dim() == 4:
        image_bgr = image_bgr[0]
    image = image_bgr.permute(1, 2, 0).contiguous()
    return torch.clamp(image, 0.0, 255.0).to(torch.uint8).cpu().numpy()


def _bgr_to_y_torch(image_bgr: torch.Tensor) -> torch.Tensor:
    blue = image_bgr[:, 0:1]
    green = image_bgr[:, 1:2]
    red = image_bgr[:, 2:3]
    return 0.114 * blue + 0.587 * green + 0.299 * red


def _bgr_to_ycrcb_torch(image_bgr: torch.Tensor) -> torch.Tensor:
    y = _bgr_to_y_torch(image_bgr)
    blue = image_bgr[:, 0:1]
    red = image_bgr[:, 2:3]
    cr = (red - y) * 0.713 + 128.0
    cb = (blue - y) * 0.564 + 128.0
    return torch.cat([y, cr, cb], dim=1)


def _ycrcb_to_bgr_torch(image_ycrcb: torch.Tensor) -> torch.Tensor:
    y = image_ycrcb[:, 0:1]
    cr = image_ycrcb[:, 1:2] - 128.0
    cb = image_ycrcb[:, 2:3] - 128.0
    red = y + 1.403 * cr
    blue = y + 1.773 * cb
    green = y - 0.714 * cr - 0.344 * cb
    return torch.cat([blue, green, red], dim=1)


@torch.inference_mode()
def run_bicubic_upscale(
    image_bgr_u8: np.ndarray,
    outscale: float,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    if outscale <= 0:
        raise ValueError("Outscale must be positive.")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    input_height, input_width = image_bgr_u8.shape[:2]
    target_height = max(1, int(round(input_height * outscale)))
    target_width = max(1, int(round(input_width * outscale)))
    target_hw = (target_height, target_width)

    lr = _bgr_u8_to_torch(image_bgr_u8, device)
    y_lr = _bgr_to_y_torch(lr)
    y_sr = F.interpolate(
        y_lr,
        size=target_hw,
        mode="bicubic",
        align_corners=False,
    )

    lr_up = F.interpolate(
        lr,
        size=target_hw,
        mode="bicubic",
        align_corners=False,
    )
    ycrcb = _bgr_to_ycrcb_torch(lr_up)
    ycrcb[:, 0:1] = torch.clamp(y_sr, 0.0, 255.0)
    sr_bgr = _ycrcb_to_bgr_torch(ycrcb)
    sr_bgr_u8 = _torch_to_bgr_u8(sr_bgr)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = float((time.perf_counter() - start_time) * 1000.0)
    return sr_bgr_u8, elapsed_ms
