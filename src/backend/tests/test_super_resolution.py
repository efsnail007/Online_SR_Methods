from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest
from app.core.config import ModelConfig, Settings
from app.ml.model_runtime import RuntimeResult
from app.services.super_resolution import SuperResolutionService


@pytest.fixture
def service_settings(tmp_path: Path) -> Settings:
    weights_path = tmp_path / "RealESRGAN_x4plus.pth"
    return Settings(
        app_name="Test Backend",
        environment="test",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False,
        api_prefix="/api/v1",
        cors_origins=[],
        cors_origin_regex=None,
        default_model_id="realesrgan_x4plus",
        startup_model_ids=["realesrgan_x4plus"],
        model_catalog_path=None,
        models=[
            ModelConfig(
                id="realesrgan_x4plus",
                name="RealESRGAN_x4plus",
                kind="torch",
                architecture="realesrgan_x4plus",
                weights_path=weights_path,
                scale=4.0,
            ),
            ModelConfig(
                id="bicubic",
                name="Bicubic",
                kind="bicubic",
                architecture="bicubic",
                scale=1.0,
            ),
        ],
        model_name="RealESRGAN_x4plus",
        model_weights_path=weights_path,
        model_device="cpu",
        use_half=True,
        use_channels_last=True,
        default_outscale=4.0,
        max_image_bytes=1024 * 1024,
        output_format="png",
        jpeg_quality=90,
        png_compression=3,
    )


def test_load_requires_existing_weights(service_settings: Settings) -> None:
    service = SuperResolutionService(service_settings)
    with pytest.raises(FileNotFoundError, match="Model weights not found"):
        service.load()


def test_load_uses_model_loader(monkeypatch, service_settings: Settings) -> None:
    service_settings.model_weights_path.write_bytes(b"weights")
    sentinel_model = object()

    def fake_loader(**kwargs):
        assert kwargs["weights_path"] == service_settings.model_weights_path
        assert kwargs["use_half"] is False
        assert kwargs["use_channels_last"] is False
        return sentinel_model, "params_ema"

    monkeypatch.setattr("app.ml.model_runtime.load_realesrgan_x4plus", fake_loader)

    service = SuperResolutionService(service_settings)
    service.load()
    runtime = service.registry.get_runtime("realesrgan_x4plus")

    assert runtime.model is sentinel_model
    assert runtime.checkpoint_key == "params_ema"


def test_process_raw_image_rejects_large_payload(service_settings: Settings) -> None:
    service_settings.max_image_bytes = 4
    service = SuperResolutionService(service_settings)

    with pytest.raises(ValueError, match="Image payload is too large"):
        service.process_raw_image(b"012345")


def test_process_image_requires_existing_default_weights(
    service_settings: Settings,
    sample_image,
) -> None:
    service = SuperResolutionService(service_settings)
    with pytest.raises(FileNotFoundError, match="Model weights not found"):
        service.process_image(sample_image)


def test_process_base64_image_delegates_to_raw_processing(
    monkeypatch,
    service_settings: Settings,
    sample_png_bytes: bytes,
) -> None:
    service = SuperResolutionService(service_settings)
    captured: dict[str, object] = {}

    def fake_process_raw_image(image_bytes, **kwargs):
        captured["image_bytes"] = image_bytes
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(service, "process_raw_image", fake_process_raw_image)

    result = service.process_base64_image(
        base64.b64encode(sample_png_bytes).decode("ascii"),
        outscale=2.0,
        model_id="bicubic",
        output_format="png",
        jpeg_quality=80,
        png_compression=1,
    )

    assert result == "ok"
    assert captured["image_bytes"] == sample_png_bytes
    assert captured["outscale"] == 2.0
    assert captured["model_id"] == "bicubic"
    assert captured["output_format"] == "png"


def test_process_image_returns_encoded_result(
    monkeypatch,
    service_settings: Settings,
    sample_image,
) -> None:
    service = SuperResolutionService(service_settings)

    class FakeRuntime:
        config = ModelConfig(
            id="realesrgan_x4plus",
            name="RealESRGAN_x4plus",
            kind="torch",
            architecture="realesrgan_x4plus",
            scale=4.0,
        )
        device = "cpu"

        def upscale(self, image_bgr, outscale):
            assert image_bgr.shape == (8, 8, 3)
            assert outscale == 2.0
            return RuntimeResult(sample_image.copy(), 7.5)

    def fake_get_runtime(model_id):
        assert model_id == "realesrgan_x4plus"
        return FakeRuntime()

    monkeypatch.setattr(service.registry, "get_runtime", fake_get_runtime)
    monkeypatch.setattr(
        "app.services.super_resolution.encode_image_bytes",
        lambda image_bgr, output_format, jpeg_quality, png_compression: (
            b"encoded-image",
            "image/png",
        ),
    )

    result = service.process_image(
        sample_image,
        outscale=2.0,
        output_format="png",
        jpeg_quality=75,
        png_compression=1,
    )

    assert result.image_bytes == b"encoded-image"
    assert result.content_type == "image/png"
    assert result.inference_time_ms == 7.5
    assert result.output_width == 8
    assert result.output_height == 8
    assert result.outscale == 2.0
    assert result.model_id == "realesrgan_x4plus"
    assert result.model_name == service_settings.model_name


def test_process_image_supports_bicubic_without_loaded_model(
    monkeypatch,
    service_settings: Settings,
    sample_image,
) -> None:
    service = SuperResolutionService(service_settings)

    monkeypatch.setattr(
        "app.ml.model_runtime.run_bicubic_upscale",
        lambda image_bgr, outscale, device: (
            np.full((16, 16, 3), 80, dtype=np.uint8),
            3.25,
        ),
    )
    monkeypatch.setattr(
        "app.services.super_resolution.encode_image_bytes",
        lambda image_bgr, output_format, jpeg_quality, png_compression: (
            b"bicubic-image",
            "image/png",
        ),
    )

    result = service.process_image(
        sample_image,
        outscale=2.0,
        model_id="bicubic",
        output_format="png",
        jpeg_quality=75,
        png_compression=1,
    )

    assert result.image_bytes == b"bicubic-image"
    assert result.content_type == "image/png"
    assert result.inference_time_ms == 3.25
    assert result.output_width == 16
    assert result.output_height == 16
    assert result.outscale == 2.0
    assert result.model_id == "bicubic"
    assert result.model_name == "Bicubic"
