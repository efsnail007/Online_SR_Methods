from __future__ import annotations

import numpy as np
import torch
from app.ml.bicubic import run_bicubic_upscale


def test_bicubic_upscales_all_rgb_channels_independently() -> None:
    image_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    image_bgr[:, :, 0] = 10
    image_bgr[:, :, 1] = 80
    image_bgr[:, :, 2] = 220

    output_bgr, _ = run_bicubic_upscale(
        image_bgr,
        outscale=2.0,
        device=torch.device("cpu"),
    )

    assert output_bgr.shape == (4, 4, 3)
    assert np.all(output_bgr[:, :, 0] == 10)
    assert np.all(output_bgr[:, :, 1] == 80)
    assert np.all(output_bgr[:, :, 2] == 220)


def test_realesrgan_runtime_receives_rgb_tensor(monkeypatch) -> None:
    from app.core.config import ModelConfig, Settings
    from app.ml.model_runtime import RealESRGANTorchRuntime

    settings = Settings.from_env()
    settings.model_device = "cpu"
    runtime = RealESRGANTorchRuntime(
        ModelConfig(
            id="test",
            name="test",
            kind="torch",
            architecture="realesrgan_x4plus",
            scale=1.0,
        ),
        settings,
    )

    captured: dict[str, torch.Tensor] = {}

    class FakeModel:
        def __call__(self, tensor):
            captured["tensor"] = tensor.detach().cpu()
            return tensor

    runtime.model = FakeModel()
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    image_bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
    runtime._run_inference(image_bgr, outscale=1.0)

    assert torch.allclose(
        captured["tensor"][0, :, 0, 0],
        torch.tensor([30 / 255.0, 20 / 255.0, 10 / 255.0]),
    )


def test_srcnn_runtime_receives_rgb_tensor() -> None:
    from app.core.config import ModelConfig, Settings
    from app.ml.model_runtime import SRCNNRuntime

    settings = Settings.from_env()
    settings.model_device = "cpu"
    runtime = SRCNNRuntime(
        ModelConfig(
            id="test",
            name="test",
            kind="torch",
            architecture="srcnn_rgb",
            scale=1.0,
        ),
        settings,
    )

    captured: dict[str, torch.Tensor] = {}

    class FakeModel:
        def __call__(self, tensor):
            captured["tensor"] = tensor.detach().cpu()
            return tensor

    runtime.model = FakeModel()
    runtime.metadata = {"predict_residual": False}

    image_bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
    runtime._run_inference(image_bgr, outscale=1.0)

    assert torch.allclose(
        captured["tensor"][0, :, 0, 0],
        torch.tensor([30 / 255.0, 20 / 255.0, 10 / 255.0]),
    )
