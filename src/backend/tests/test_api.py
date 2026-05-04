from __future__ import annotations

import base64
from collections.abc import Callable

from app import main as app_main
from app.services.super_resolution import EncodedImageResult
from fastapi.testclient import TestClient


def build_client_factory(
    monkeypatch,
    image_bytes: bytes,
) -> Callable[..., TestClient]:
    def _factory(
        *,
        raw_error: Exception | None = None,
        base64_error: Exception | None = None,
    ) -> TestClient:
        class FakeSuperResolutionService:
            def __init__(self, settings) -> None:
                self.settings = settings

            def load(self) -> None:
                return None

            def unload(self) -> None:
                return None

            def model_info(
                self,
                model_id: str | None = None,
            ) -> dict[str, str | int | float | bool | None]:
                del model_id
                return {
                    "model_loaded": True,
                    "model_id": self.settings.default_model_id,
                    "model_name": self.settings.model_name,
                    "model_kind": "torch",
                    "architecture": "realesrgan_x4plus",
                    "checkpoint_key": "params_ema",
                    "weights_path": str(self.settings.model_weights_path),
                    "device": "cpu",
                    "use_half": False,
                    "use_channels_last": False,
                    "network_scale": 4,
                    "default_outscale": 4.0,
                    "scale": 4.0,
                    "description": None,
                    "tags": [],
                    "options": {},
                    "requested_providers": None,
                    "available_providers": None,
                    "active_providers": None,
                }

            def models_info(self) -> list[dict[str, object]]:
                return [
                    {
                        "id": self.settings.default_model_id,
                        "name": self.settings.model_name,
                        "kind": "torch",
                        "architecture": "realesrgan_x4plus",
                        "loaded": True,
                        "weights_path": str(self.settings.model_weights_path),
                        "device": "cpu",
                        "scale": 4.0,
                        "description": None,
                        "tags": [],
                        "options": {},
                    },
                    {
                        "id": "realesrgan_x4plus_onnx",
                        "name": "RealESRGAN x4plus ONNX",
                        "kind": "onnx",
                        "architecture": "generic-nchw-rgb",
                        "loaded": False,
                        "weights_path": "src/backend/assets/weights/RealESRGAN_x4plus.onnx",
                        "device": "cpu",
                        "scale": 4.0,
                        "description": "Bundled Real-ESRGAN x4plus ONNX export.",
                        "tags": ["onnx", "real-esrgan"],
                        "options": {},
                    },
                    {
                        "id": "bicubic",
                        "name": "Bicubic",
                        "kind": "bicubic",
                        "architecture": "bicubic",
                        "loaded": True,
                        "weights_path": None,
                        "device": "cpu",
                        "scale": 1.0,
                        "description": None,
                        "tags": [],
                        "options": {},
                    },
                ]

            def process_raw_image(
                self,
                payload: bytes,
                outscale: float | None = None,
                model_id: str | None = None,
                output_format: str | None = None,
                jpeg_quality: int | None = None,
                png_compression: int | None = None,
            ) -> EncodedImageResult:
                del payload, jpeg_quality, png_compression
                if raw_error is not None:
                    raise raw_error
                selected_outscale = outscale or 4.0
                selected_model_id = model_id or self.settings.default_model_id
                content_type = "image/png" if output_format == "png" else "image/jpeg"
                return EncodedImageResult(
                    image_bytes=image_bytes,
                    content_type=content_type,
                    inference_time_ms=12.5,
                    input_width=8,
                    input_height=8,
                    output_width=int(8 * selected_outscale),
                    output_height=int(8 * selected_outscale),
                    outscale=selected_outscale,
                    model_id=selected_model_id,
                    model_kind=(
                        "bicubic"
                        if selected_model_id == "bicubic"
                        else "onnx"
                        if selected_model_id == "realesrgan_x4plus_onnx"
                        else "torch"
                    ),
                    device="cpu",
                    model_name=(
                        self.settings.model_name
                        if selected_model_id == self.settings.default_model_id
                        else "RealESRGAN x4plus ONNX"
                        if selected_model_id == "realesrgan_x4plus_onnx"
                        else "Bicubic"
                    ),
                )

            def process_base64_image(
                self,
                image_base64: str,
                outscale: float | None = None,
                model_id: str | None = None,
                output_format: str | None = None,
                jpeg_quality: int | None = None,
                png_compression: int | None = None,
            ) -> EncodedImageResult:
                del image_base64, jpeg_quality, png_compression
                if base64_error is not None:
                    raise base64_error
                selected_outscale = outscale or 4.0
                selected_model_id = model_id or self.settings.default_model_id
                content_type = "image/png" if output_format == "png" else "image/jpeg"
                return EncodedImageResult(
                    image_bytes=image_bytes,
                    content_type=content_type,
                    inference_time_ms=15.0,
                    input_width=8,
                    input_height=8,
                    output_width=int(8 * selected_outscale),
                    output_height=int(8 * selected_outscale),
                    outscale=selected_outscale,
                    model_id=selected_model_id,
                    model_kind=(
                        "bicubic"
                        if selected_model_id == "bicubic"
                        else "onnx"
                        if selected_model_id == "realesrgan_x4plus_onnx"
                        else "torch"
                    ),
                    device="cpu",
                    model_name=(
                        self.settings.model_name
                        if selected_model_id == self.settings.default_model_id
                        else "RealESRGAN x4plus ONNX"
                        if selected_model_id == "realesrgan_x4plus_onnx"
                        else "Bicubic"
                    ),
                )

        monkeypatch.setattr(
            app_main, "SuperResolutionService", FakeSuperResolutionService
        )
        return TestClient(app_main.create_app())

    return _factory


def test_health_endpoint_returns_loaded_model(
    monkeypatch, sample_png_bytes: bytes
) -> None:
    client_factory = build_client_factory(monkeypatch, sample_png_bytes)
    with client_factory() as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True
    assert response.json()["model_id"] == "realesrgan_x4plus"


def test_model_endpoint_returns_backend_metadata(
    monkeypatch, sample_png_bytes: bytes
) -> None:
    client_factory = build_client_factory(monkeypatch, sample_png_bytes)
    with client_factory() as client:
        response = client.get("/api/v1/model")

    payload = response.json()
    assert response.status_code == 200
    assert payload["model_name"] == "RealESRGAN_x4plus"
    assert payload["device"] == "cpu"
    assert payload["network_scale"] == 4


def test_models_endpoint_returns_available_models(
    monkeypatch, sample_png_bytes: bytes
) -> None:
    client_factory = build_client_factory(monkeypatch, sample_png_bytes)
    with client_factory() as client:
        response = client.get("/api/v1/models")

    payload = response.json()
    assert response.status_code == 200
    assert payload["default_model_id"] == "realesrgan_x4plus"
    assert [model["id"] for model in payload["models"]] == [
        "realesrgan_x4plus",
        "realesrgan_x4plus_onnx",
        "bicubic",
    ]


def test_upscale_endpoint_returns_image_response(
    monkeypatch, sample_png_bytes: bytes
) -> None:
    client_factory = build_client_factory(monkeypatch, sample_png_bytes)
    with client_factory() as client:
        response = client.post(
            "/api/v1/upscale?output_format=png&outscale=2&model_id=bicubic",
            content=sample_png_bytes,
            headers={"content-type": "application/octet-stream"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-model-id"] == "bicubic"
    assert response.headers["x-output-width"] == "16"
    assert response.headers["x-output-height"] == "16"
    assert response.content == sample_png_bytes


def test_upscale_base64_endpoint_returns_json_payload(
    monkeypatch,
    sample_png_bytes: bytes,
) -> None:
    client_factory = build_client_factory(monkeypatch, sample_png_bytes)
    with client_factory() as client:
        response = client.post(
            "/api/v1/upscale/base64",
            json={
                "image_base64": base64.b64encode(sample_png_bytes).decode("ascii"),
                "model_id": "bicubic",
                "output_format": "png",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["content_type"] == "image/png"
    assert payload["model_id"] == "bicubic"
    assert payload["image_base64"] == base64.b64encode(sample_png_bytes).decode("ascii")


def test_upscale_endpoint_maps_value_error_to_bad_request(
    monkeypatch,
    sample_png_bytes: bytes,
) -> None:
    client_factory = build_client_factory(
        monkeypatch,
        sample_png_bytes,
    )
    with client_factory(raw_error=ValueError("bad image")) as client:
        response = client.post("/api/v1/upscale", content=sample_png_bytes)

    assert response.status_code == 400
    assert response.json()["detail"] == "bad image"


def test_upscale_base64_endpoint_maps_runtime_error_to_service_unavailable(
    monkeypatch,
    sample_png_bytes: bytes,
) -> None:
    client_factory = build_client_factory(
        monkeypatch,
        sample_png_bytes,
    )
    with client_factory(base64_error=RuntimeError("model offline")) as client:
        response = client.post(
            "/api/v1/upscale/base64",
            json={"image_base64": base64.b64encode(sample_png_bytes).decode("ascii")},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "model offline"
