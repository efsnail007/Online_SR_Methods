from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    model_loaded: bool
    model_id: str
    model_name: str
    model_kind: str
    device: str
    weights_path: str | None


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    model_id: str
    model_name: str
    model_kind: str
    architecture: str | None
    checkpoint_key: str | None
    weights_path: str | None
    device: str
    use_half: bool
    use_channels_last: bool
    network_scale: int | None
    default_outscale: float
    scale: float | None
    description: str | None
    tags: list[str]
    options: dict
