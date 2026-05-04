from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, Field

UpscaleMethod: TypeAlias = Literal["realesrgan", "bicubic"]
DEFAULT_UPSCALE_METHOD: UpscaleMethod = "realesrgan"


class Base64UpscaleRequest(BaseModel):
    image_base64: str = Field(min_length=1)
    outscale: float | None = Field(default=None, gt=0.0, le=8.0)
    method: UpscaleMethod | None = None
    output_format: Literal["jpeg", "jpg", "png"] | None = None
    jpeg_quality: int | None = Field(default=None, ge=1, le=100)
    png_compression: int | None = Field(default=None, ge=0, le=9)


class Base64UpscaleResponse(BaseModel):
    image_base64: str
    content_type: str
    inference_time_ms: float
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    outscale: float
    method: UpscaleMethod
    device: str
    model_name: str
