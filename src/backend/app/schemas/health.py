from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    model_loaded: bool
    model_name: str
    device: str
    weights_path: str


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    model_name: str
    checkpoint_key: str | None
    weights_path: str
    device: str
    use_half: bool
    use_channels_last: bool
    network_scale: int
    default_outscale: float
