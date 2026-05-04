from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture
def sample_image() -> np.ndarray:
    return np.full((8, 8, 3), 127, dtype=np.uint8)


@pytest.fixture
def sample_png_bytes(sample_image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", sample_image)
    assert ok
    return buffer.tobytes()
