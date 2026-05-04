from __future__ import annotations

import base64

import pytest
from app.services.image_codec import (
    base64_to_bytes,
    bytes_to_base64,
    decode_image_bytes,
    encode_image_bytes,
    normalize_output_format,
    strip_data_url_prefix,
)


def test_strip_data_url_prefix_removes_metadata() -> None:
    payload = "data:image/png;base64,Zm9v"
    assert strip_data_url_prefix(payload) == "Zm9v"


def test_base64_round_trip_preserves_payload(sample_png_bytes: bytes) -> None:
    encoded = bytes_to_base64(sample_png_bytes)
    decoded = base64_to_bytes(encoded)
    assert decoded == sample_png_bytes


def test_base64_to_bytes_accepts_data_url(sample_png_bytes: bytes) -> None:
    encoded = base64.b64encode(sample_png_bytes).decode("ascii")
    payload = f"data:image/png;base64,{encoded}"
    assert base64_to_bytes(payload) == sample_png_bytes


def test_base64_to_bytes_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="Invalid base64 image payload."):
        base64_to_bytes("not-base64")


def test_decode_image_bytes_rejects_empty_payload() -> None:
    with pytest.raises(ValueError, match="Image payload is empty."):
        decode_image_bytes(b"")


def test_decode_image_bytes_returns_bgr_image(sample_png_bytes: bytes) -> None:
    image = decode_image_bytes(sample_png_bytes)
    assert image.shape == (8, 8, 3)
    assert image.dtype.name == "uint8"


def test_normalize_output_format_supports_jpg_alias() -> None:
    assert normalize_output_format("jpg", "png") == "jpeg"


def test_normalize_output_format_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="Unsupported output format. Use jpeg or png."):
        normalize_output_format("webp", "png")


def test_encode_image_bytes_returns_png(sample_image) -> None:
    encoded, content_type = encode_image_bytes(
        sample_image,
        output_format="png",
        jpeg_quality=90,
        png_compression=3,
    )
    assert content_type == "image/png"
    assert encoded.startswith(b"\x89PNG")
