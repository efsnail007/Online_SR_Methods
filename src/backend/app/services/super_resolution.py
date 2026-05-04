from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from app.core.config import Settings
from app.services.image_codec import (
    base64_to_bytes,
    decode_image_bytes,
    encode_image_bytes,
    normalize_output_format,
)
from app.services.model_registry import ModelRegistry


@dataclass(slots=True)
class EncodedImageResult:
    image_bytes: bytes
    content_type: str
    inference_time_ms: float
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    outscale: float
    model_id: str
    model_name: str
    model_kind: str
    device: str


class SuperResolutionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry = ModelRegistry(settings)

    def load(self) -> None:
        self.registry.load_startup_models()

    def unload(self) -> None:
        self.registry.unload_all()

    def models_info(self) -> list[dict]:
        return self.registry.list_models()

    def model_info(self, model_id: str | None = None) -> dict:
        info = self.registry.model_info(model_id)
        return {
            "model_loaded": info["loaded"],
            "model_id": info["id"],
            "model_name": info["name"],
            "model_kind": info["kind"],
            "architecture": info["architecture"],
            "checkpoint_key": info.get("checkpoint_key"),
            "weights_path": info["weights_path"],
            "device": info["device"],
            "use_half": info.get("use_half", False),
            "use_channels_last": info.get("use_channels_last", False),
            "network_scale": info.get("network_scale"),
            "default_outscale": self.settings.default_outscale,
            "scale": info["scale"],
            "description": info["description"],
            "tags": info["tags"],
            "options": info["options"],
            "requested_providers": info.get("requested_providers"),
            "available_providers": info.get("available_providers"),
            "active_providers": info.get("active_providers"),
        }

    def process_base64_image(
        self,
        image_base64: str,
        outscale: float | None,
        model_id: str | None,
        output_format: str | None,
        jpeg_quality: int | None,
        png_compression: int | None,
    ) -> EncodedImageResult:
        return self.process_raw_image(
            base64_to_bytes(image_base64),
            outscale=outscale,
            model_id=model_id,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            png_compression=png_compression,
        )

    def process_raw_image(
        self,
        image_bytes: bytes,
        outscale: float | None = None,
        model_id: str | None = None,
        output_format: str | None = None,
        jpeg_quality: int | None = None,
        png_compression: int | None = None,
    ) -> EncodedImageResult:
        if len(image_bytes) > self.settings.max_image_bytes:
            raise ValueError(
                f"Image payload is too large. Limit: {self.settings.max_image_bytes} bytes."
            )
        image_bgr = decode_image_bytes(image_bytes)
        return self.process_image(
            image_bgr=image_bgr,
            outscale=outscale,
            model_id=model_id,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            png_compression=png_compression,
        )

    def process_image(
        self,
        image_bgr: np.ndarray,
        outscale: float | None = None,
        model_id: str | None = None,
        output_format: str | None = None,
        jpeg_quality: int | None = None,
        png_compression: int | None = None,
    ) -> EncodedImageResult:
        selected_outscale = outscale or self.settings.default_outscale
        if selected_outscale <= 0:
            raise ValueError("Outscale must be positive.")
        selected_model_id = self.registry.resolve_model_id(model_id)
        selected_format = normalize_output_format(
            output_format, self.settings.output_format
        )
        selected_jpeg_quality = jpeg_quality or self.settings.jpeg_quality
        selected_png_compression = (
            self.settings.png_compression
            if png_compression is None
            else png_compression
        )

        input_height, input_width = image_bgr.shape[:2]
        runtime = self.registry.get_runtime(selected_model_id)
        runtime_result = runtime.upscale(image_bgr, selected_outscale)
        output_bgr = runtime_result.image_bgr
        output_height, output_width = output_bgr.shape[:2]
        encoded, content_type = encode_image_bytes(
            output_bgr,
            output_format=selected_format,
            jpeg_quality=selected_jpeg_quality,
            png_compression=selected_png_compression,
        )
        return EncodedImageResult(
            image_bytes=encoded,
            content_type=content_type,
            inference_time_ms=runtime_result.inference_time_ms,
            input_width=input_width,
            input_height=input_height,
            output_width=output_width,
            output_height=output_height,
            outscale=selected_outscale,
            model_id=selected_model_id,
            model_name=runtime.config.name,
            model_kind=runtime.config.kind,
            device=str(runtime.device),
        )
