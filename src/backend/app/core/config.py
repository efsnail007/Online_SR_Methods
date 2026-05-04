from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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
            model_name=os.getenv("BACKEND_MODEL_NAME", "RealESRGAN_x4plus"),
            model_weights_path=resolve_project_path(
                os.getenv("BACKEND_MODEL_WEIGHTS_PATH"),
                default_weights_path,
            ),
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
