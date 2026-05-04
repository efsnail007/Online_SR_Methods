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

    lr_bgr = _bgr_u8_to_torch(image_bgr_u8, device)
    sr_bgr = F.interpolate(
        lr_bgr,
        size=target_hw,
        mode="bicubic",
        align_corners=False,
    )
    sr_bgr_u8 = _torch_to_bgr_u8(sr_bgr)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = float((time.perf_counter() - start_time) * 1000.0)
    return sr_bgr_u8, elapsed_ms
