from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from app.core.config import Settings
from app.ml.bicubic import run_bicubic_upscale
from app.ml.realesrgan import MODEL_SCALE, RRDBNet, load_realesrgan_x4plus
from app.schemas.inference import DEFAULT_UPSCALE_METHOD, UpscaleMethod
from app.services.image_codec import (
    base64_to_bytes,
    decode_image_bytes,
    encode_image_bytes,
    normalize_output_format,
)


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
    method: UpscaleMethod
    device: str
    model_name: str


class SuperResolutionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = self._resolve_device(settings.model_device)
        self.use_half = settings.use_half and self.device.type == "cuda"
        self.use_channels_last = (
            settings.use_channels_last and self.device.type == "cuda"
        )
        self.model: RRDBNet | None = None
        self.checkpoint_key: str | None = None
        self._model_lock = threading.Lock()

    def _resolve_device(self, model_device: str) -> torch.device:
        if model_device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if model_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")
            return torch.device("cuda:0")
        if model_device == "cpu":
            return torch.device("cpu")
        raise ValueError("BACKEND_MODEL_DEVICE must be one of: auto, cpu, cuda.")

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.settings.model_weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.settings.model_weights_path}"
            )
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        self.model, self.checkpoint_key = load_realesrgan_x4plus(
            weights_path=self.settings.model_weights_path,
            device=self.device,
            use_half=self.use_half,
            use_channels_last=self.use_channels_last,
        )

    def unload(self) -> None:
        self.model = None
        self.checkpoint_key = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def model_info(self) -> dict[str, str | int | float | bool | None]:
        return {
            "model_loaded": self.model is not None,
            "model_name": self.settings.model_name,
            "checkpoint_key": self.checkpoint_key,
            "weights_path": str(self.settings.model_weights_path),
            "device": str(self.device),
            "use_half": self.use_half,
            "use_channels_last": self.use_channels_last,
            "network_scale": MODEL_SCALE,
            "default_outscale": self.settings.default_outscale,
        }

    def process_base64_image(
        self,
        image_base64: str,
        outscale: float | None,
        method: UpscaleMethod | None,
        output_format: str | None,
        jpeg_quality: int | None,
        png_compression: int | None,
    ) -> EncodedImageResult:
        return self.process_raw_image(
            base64_to_bytes(image_base64),
            outscale=outscale,
            method=method,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            png_compression=png_compression,
        )

    def process_raw_image(
        self,
        image_bytes: bytes,
        outscale: float | None = None,
        method: UpscaleMethod | None = None,
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
            method=method,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            png_compression=png_compression,
        )

    def process_image(
        self,
        image_bgr: np.ndarray,
        outscale: float | None = None,
        method: UpscaleMethod | None = None,
        output_format: str | None = None,
        jpeg_quality: int | None = None,
        png_compression: int | None = None,
    ) -> EncodedImageResult:
        selected_outscale = outscale or self.settings.default_outscale
        if selected_outscale <= 0:
            raise ValueError("Outscale must be positive.")
        selected_method = method or DEFAULT_UPSCALE_METHOD
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
        output_bgr, inference_time_ms, model_name = self._run_selected_method(
            image_bgr=image_bgr,
            outscale=selected_outscale,
            method=selected_method,
        )
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
            inference_time_ms=inference_time_ms,
            input_width=input_width,
            input_height=input_height,
            output_width=output_width,
            output_height=output_height,
            outscale=selected_outscale,
            method=selected_method,
            device=str(self.device),
            model_name=model_name,
        )

    def _run_selected_method(
        self,
        image_bgr: np.ndarray,
        outscale: float,
        method: UpscaleMethod,
    ) -> tuple[np.ndarray, float, str]:
        if method == "bicubic":
            output_bgr, inference_time_ms = run_bicubic_upscale(
                image_bgr,
                outscale=outscale,
                device=self.device,
            )
            return output_bgr, inference_time_ms, "bicubic"

        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        with self._model_lock:
            output_bgr, inference_time_ms = self._run_realesrgan_inference(
                image_bgr, outscale
            )
        return output_bgr, inference_time_ms, self.settings.model_name

    @torch.inference_mode()
    def _run_realesrgan_inference(
        self, image_bgr: np.ndarray, outscale: float
    ) -> tuple[np.ndarray, float]:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()

        image_tensor = torch.from_numpy(image_bgr).to(self.device, dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor[:, [2, 1, 0], :, :] / 255.0

        if self.use_channels_last:
            image_tensor = image_tensor.contiguous(memory_format=torch.channels_last)
        if self.use_half:
            image_tensor = image_tensor.half()

        output = self.model(image_tensor).clamp_(0.0, 1.0)
        if outscale != float(MODEL_SCALE):
            resize_factor = outscale / float(MODEL_SCALE)
            output = F.interpolate(
                output,
                scale_factor=resize_factor,
                mode="bicubic",
                align_corners=False,
            ).clamp_(0.0, 1.0)

        output_bgr = output[:, [2, 1, 0], :, :].float() * 255.0
        output_bgr_u8 = (
            output_bgr.squeeze(0)
            .permute(1, 2, 0)
            .contiguous()
            .round()
            .clamp_(0.0, 255.0)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = float((time.perf_counter() - start_time) * 1000.0)
        return output_bgr_u8, elapsed_ms
