from __future__ import annotations

import base64

import cv2
import numpy as np

DATA_URL_SEPARATOR = ";base64,"


def strip_data_url_prefix(payload: str) -> str:
    if DATA_URL_SEPARATOR not in payload:
        return payload
    return payload.split(DATA_URL_SEPARATOR, maxsplit=1)[1]


def base64_to_bytes(payload: str) -> bytes:
    normalized = strip_data_url_prefix(payload.strip())
    try:
        return base64.b64decode(normalized, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("Invalid base64 image payload.") from exc


def bytes_to_base64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def decode_image_bytes(payload: bytes) -> np.ndarray:
    if not payload:
        raise ValueError("Image payload is empty.")
    buffer = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image payload.")
    return image


def normalize_output_format(value: str | None, default_format: str) -> str:
    selected = (value or default_format).strip().lower()
    if selected == "jpg":
        selected = "jpeg"
    if selected not in {"jpeg", "png"}:
        raise ValueError("Unsupported output format. Use jpeg or png.")
    return selected


def encode_image_bytes(
    image_bgr: np.ndarray,
    output_format: str,
    jpeg_quality: int,
    png_compression: int,
) -> tuple[bytes, str]:
    if output_format == "jpeg":
        extension = ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        content_type = "image/jpeg"
    else:
        extension = ".png"
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)]
        content_type = "image/png"

    success, encoded = cv2.imencode(extension, image_bgr, params)
    if not success:
        raise RuntimeError("Unable to encode image response.")
    return encoded.tobytes(), content_type
