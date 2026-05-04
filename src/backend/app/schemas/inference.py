from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Base64UpscaleRequest(BaseModel):
    image_base64: str = Field(min_length=1)
    outscale: float | None = Field(default=None, gt=0.0, le=8.0)
    model_id: str | None = Field(default=None, min_length=1)
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
    model_id: str
    model_kind: str
    device: str
    model_name: str


class ModelSummary(BaseModel):
    id: str
    name: str
    kind: str
    architecture: str | None
    loaded: bool
    weights_path: str | None
    device: str
    scale: float | None
    description: str | None
    tags: list[str]
    options: dict[str, Any]


class ModelsResponse(BaseModel):
    default_model_id: str
    models: list[ModelSummary]
