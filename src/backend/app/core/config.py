from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent.parent


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_project_path(value: str | None, fallback: Path) -> Path:
    raw_path = Path(value) if value else fallback
    if raw_path.is_absolute():
        return raw_path
    return (PROJECT_ROOT / raw_path).resolve()


def resolve_optional_project_path(value: str | None) -> Path | None:
    if not value:
        return None
    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path
    return (PROJECT_ROOT / raw_path).resolve()


@dataclass(slots=True)
class ModelConfig:
    id: str
    name: str
    kind: str
    architecture: str | None = None
    weights_path: Path | None = None
    scale: float | None = None
    device: str | None = None
    enabled: bool = True
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ModelConfig":
        model_id = str(payload.get("id", "")).strip()
        if not model_id:
            raise ValueError("Model config must include non-empty 'id'.")
        name = str(payload.get("name") or model_id)
        weights_path = resolve_optional_project_path(payload.get("weights_path"))
        raw_tags = payload.get("tags") or []
        if not isinstance(raw_tags, list):
            raise ValueError(f"Model '{model_id}' tags must be a list.")
        raw_options = payload.get("options") or {}
        if not isinstance(raw_options, dict):
            raise ValueError(f"Model '{model_id}' options must be an object.")
        return cls(
            id=model_id,
            name=name,
            kind=str(payload.get("kind") or payload.get("runtime") or "").lower(),
            architecture=payload.get("architecture"),
            weights_path=weights_path,
            scale=(
                None
                if payload.get("scale") is None
                else float(payload.get("scale"))
            ),
            device=payload.get("device"),
            enabled=parse_bool(str(payload.get("enabled")), True)
            if payload.get("enabled") is not None
            else True,
            description=payload.get("description"),
            tags=[str(tag) for tag in raw_tags],
            options=raw_options,
        )


def load_model_catalog(path: Path | None) -> list[ModelConfig]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    raw_models = payload.get("models", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_models, list):
        raise ValueError("Model catalog must be a list or an object with 'models'.")
    return [ModelConfig.from_mapping(item) for item in raw_models]


def merge_model_configs(
    defaults: list[ModelConfig],
    overrides: list[ModelConfig],
) -> list[ModelConfig]:
    by_id = {model.id: model for model in defaults}
    for model in overrides:
        by_id[model.id] = model
    return list(by_id.values())


@dataclass(slots=True)
class Settings:
    app_name: str
    environment: str
    host: str
    port: int
    log_level: str
    reload: bool
    api_prefix: str
    cors_origins: list[str]
    cors_origin_regex: str | None
    default_model_id: str
    startup_model_ids: list[str]
    model_catalog_path: Path | None
    models: list[ModelConfig]
    model_name: str
    model_weights_path: Path
    model_device: str
    use_half: bool
    use_channels_last: bool
    default_outscale: float
    max_image_bytes: int
    output_format: str
    jpeg_quality: int
    png_compression: int

    @classmethod
    def from_env(cls) -> "Settings":
        default_weights_path = (
            BACKEND_DIR / "assets" / "weights" / "RealESRGAN_x4plus.pth"
        )
        model_name = os.getenv("BACKEND_MODEL_NAME", "RealESRGAN_x4plus")
        model_weights_path = resolve_project_path(
            os.getenv("BACKEND_MODEL_WEIGHTS_PATH"),
            default_weights_path,
        )
        default_model_id = os.getenv("BACKEND_DEFAULT_MODEL_ID", "realesrgan_x4plus")
        model_catalog_path = resolve_optional_project_path(
            os.getenv("BACKEND_MODEL_CATALOG_PATH")
        )
        default_models = [
            ModelConfig(
                id=default_model_id,
                name=model_name,
                kind="torch",
                architecture="realesrgan_x4plus",
                weights_path=model_weights_path,
                scale=4.0,
                description="Bundled Real-ESRGAN x4plus PyTorch checkpoint.",
                tags=["torch", "real-esrgan"],
            ),
            ModelConfig(
                id="srcnn_rgb",
                name="SRCNN RGB",
                kind="torch",
                architecture="srcnn_rgb",
                weights_path=BACKEND_DIR / "assets" / "weights" / "srcnn_rgb_best.pth",
                scale=4.0,
                description="SRCNN x4 model trained on all RGB channels.",
                tags=["torch", "srcnn", "rgb"],
            ),
            ModelConfig(
                id="bicubic",
                name="Bicubic",
                kind="bicubic",
                architecture="bicubic",
                scale=1.0,
                description="Built-in bicubic RGB baseline.",
                tags=["builtin", "rgb"],
            ),
        ]
        catalog_models = load_model_catalog(model_catalog_path)
        return cls(
            app_name=os.getenv("BACKEND_APP_NAME", "Real-ESRGAN Backend"),
            environment=os.getenv("APP_ENV", "development"),
            host=os.getenv("BACKEND_HOST", "0.0.0.0"),
            port=int(os.getenv("BACKEND_PORT", "8000")),
            log_level=os.getenv("BACKEND_LOG_LEVEL", "info").lower(),
            reload=parse_bool(os.getenv("BACKEND_RELOAD"), False),
            api_prefix=os.getenv("BACKEND_API_PREFIX", "/api/v1"),
            cors_origins=parse_csv(os.getenv("BACKEND_CORS_ORIGINS")),
            cors_origin_regex=os.getenv("BACKEND_CORS_ORIGIN_REGEX") or None,
            default_model_id=default_model_id,
            startup_model_ids=parse_csv(os.getenv("BACKEND_STARTUP_MODEL_IDS"))
            or [default_model_id],
            model_catalog_path=model_catalog_path,
            models=merge_model_configs(default_models, catalog_models),
            model_name=model_name,
            model_weights_path=model_weights_path,
            model_device=os.getenv("BACKEND_MODEL_DEVICE", "auto").lower(),
            use_half=parse_bool(os.getenv("BACKEND_USE_HALF"), True),
            use_channels_last=parse_bool(os.getenv("BACKEND_USE_CHANNELS_LAST"), True),
            default_outscale=float(os.getenv("BACKEND_DEFAULT_OUTSCALE", "4.0")),
            max_image_bytes=int(
                os.getenv("BACKEND_MAX_IMAGE_BYTES", str(5 * 1024 * 1024))
            ),
            output_format=os.getenv("BACKEND_OUTPUT_FORMAT", "jpeg").lower(),
            jpeg_quality=int(os.getenv("BACKEND_JPEG_QUALITY", "90")),
            png_compression=int(os.getenv("BACKEND_PNG_COMPRESSION", "3")),
        )


settings = Settings.from_env()
