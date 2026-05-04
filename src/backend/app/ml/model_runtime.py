from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from app.core.config import ModelConfig, Settings
from app.ml.bicubic import run_bicubic_upscale
from app.ml.realesrgan import MODEL_SCALE, RRDBNet, load_realesrgan_x4plus
from app.ml.srcnn import SRCNN, SRCNN_MODEL_SCALE, load_srcnn_rgb


@dataclass(slots=True)
class RuntimeResult:
    image_bgr: np.ndarray
    inference_time_ms: float


class BaseModelRuntime:
    def __init__(self, config: ModelConfig, settings: Settings) -> None:
        self.config = config
        self.settings = settings
        self.device = resolve_device(config.device or settings.model_device)
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def upscale(self, image_bgr: np.ndarray, outscale: float) -> RuntimeResult:
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        return {
            "id": self.config.id,
            "name": self.config.name,
            "kind": self.config.kind,
            "architecture": self.config.architecture,
            "loaded": self.loaded,
            "weights_path": (
                None
                if self.config.weights_path is None
                else str(self.config.weights_path)
            ),
            "device": str(self.device),
            "scale": self.config.scale,
            "description": self.config.description,
            "tags": self.config.tags,
            "options": self.config.options,
        }


class BicubicRuntime(BaseModelRuntime):
    def upscale(self, image_bgr: np.ndarray, outscale: float) -> RuntimeResult:
        output_bgr, inference_time_ms = run_bicubic_upscale(
            image_bgr,
            outscale=outscale,
            device=self.device,
        )
        return RuntimeResult(output_bgr, inference_time_ms)


class RealESRGANTorchRuntime(BaseModelRuntime):
    def __init__(self, config: ModelConfig, settings: Settings) -> None:
        super().__init__(config, settings)
        self.use_half = settings.use_half and self.device.type == "cuda"
        self.use_channels_last = (
            settings.use_channels_last and self.device.type == "cuda"
        )
        self.model: RRDBNet | None = None
        self.checkpoint_key: str | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.model is not None:
            return
        if self.config.weights_path is None:
            raise FileNotFoundError(
                f"Model weights path is not configured: {self.config.id}"
            )
        if not self.config.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.config.weights_path}"
            )
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        self.model, self.checkpoint_key = load_realesrgan_x4plus(
            weights_path=self.config.weights_path,
            device=self.device,
            use_half=self.use_half,
            use_channels_last=self.use_channels_last,
        )

    def unload(self) -> None:
        self.model = None
        self.checkpoint_key = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def upscale(self, image_bgr: np.ndarray, outscale: float) -> RuntimeResult:
        self.load()
        with self._lock:
            output_bgr, inference_time_ms = self._run_inference(image_bgr, outscale)
        return RuntimeResult(output_bgr, inference_time_ms)

    def info(self) -> dict[str, Any]:
        payload = super().info()
        payload.update(
            {
                "checkpoint_key": self.checkpoint_key,
                "use_half": self.use_half,
                "use_channels_last": self.use_channels_last,
                "network_scale": MODEL_SCALE,
            }
        )
        return payload

    @torch.inference_mode()
    def _run_inference(
        self,
        image_bgr: np.ndarray,
        outscale: float,
    ) -> tuple[np.ndarray, float]:
        if self.model is None:
            raise RuntimeError(f"Model is not loaded: {self.config.id}")

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
        model_scale = float(self.config.scale or MODEL_SCALE)
        if outscale != model_scale:
            output = F.interpolate(
                output,
                scale_factor=outscale / model_scale,
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


class SRCNNRuntime(BaseModelRuntime):
    def __init__(self, config: ModelConfig, settings: Settings) -> None:
        super().__init__(config, settings)
        self.use_half = settings.use_half and self.device.type == "cuda"
        self.use_channels_last = (
            settings.use_channels_last and self.device.type == "cuda"
        )
        self.model: SRCNN | None = None
        self.metadata: dict[str, Any] = {}

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.model is not None:
            return
        if self.config.weights_path is None:
            raise FileNotFoundError(
                f"Model weights path is not configured: {self.config.id}"
            )
        if not self.config.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.config.weights_path}"
            )
        self.model, self.metadata = load_srcnn_rgb(
            weights_path=self.config.weights_path,
            device=self.device,
            use_half=self.use_half,
            use_channels_last=self.use_channels_last,
        )

    def unload(self) -> None:
        self.model = None
        self.metadata = {}
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def upscale(self, image_bgr: np.ndarray, outscale: float) -> RuntimeResult:
        self.load()
        with self._lock:
            output_bgr, inference_time_ms = self._run_inference(image_bgr, outscale)
        return RuntimeResult(output_bgr, inference_time_ms)

    def info(self) -> dict[str, Any]:
        payload = super().info()
        payload.update(
            {
                "checkpoint_key": self.metadata.get("checkpoint_key"),
                "use_half": self.use_half,
                "use_channels_last": self.use_channels_last,
                "network_scale": self.metadata.get("network_scale", SRCNN_MODEL_SCALE),
                "predict_residual": self.metadata.get("predict_residual", True),
                "runtime_color_space": "RGB",
            }
        )
        return payload

    @torch.inference_mode()
    def _run_inference(
        self,
        image_bgr: np.ndarray,
        outscale: float,
    ) -> tuple[np.ndarray, float]:
        if self.model is None:
            raise RuntimeError(f"Model is not loaded: {self.config.id}")

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()

        image_tensor = torch.from_numpy(image_bgr).to(self.device, dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor[:, [2, 1, 0], :, :] / 255.0

        input_height, input_width = image_bgr.shape[:2]
        model_scale = float(self.config.scale or SRCNN_MODEL_SCALE)
        target_hw = (
            max(1, int(round(input_height * model_scale))),
            max(1, int(round(input_width * model_scale))),
        )
        input_tensor = F.interpolate(
            image_tensor,
            size=target_hw,
            mode="bicubic",
            align_corners=False,
        )

        if self.use_channels_last:
            input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
        if self.use_half:
            input_tensor = input_tensor.half()

        output = self.model(input_tensor)
        if self.metadata.get("predict_residual", True):
            output = input_tensor + output
        output = output.clamp_(0.0, 1.0)

        if outscale != model_scale:
            output = F.interpolate(
                output,
                scale_factor=outscale / model_scale,
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


class UnsupportedRuntime(BaseModelRuntime):
    def load(self) -> None:
        raise RuntimeError(
            f"Unsupported model kind '{self.config.kind}' for model '{self.config.id}'."
        )

    def upscale(self, image_bgr: np.ndarray, outscale: float) -> RuntimeResult:
        del image_bgr, outscale
        self.load()
        raise AssertionError("unreachable")


def resolve_device(model_device: str) -> torch.device:
    normalized = model_device.lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda:0")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError("BACKEND_MODEL_DEVICE must be one of: auto, cpu, cuda.")


def create_runtime(config: ModelConfig, settings: Settings) -> BaseModelRuntime:
    kind = config.kind.lower()
    architecture = (config.architecture or "").lower()
    if kind == "bicubic" or architecture == "bicubic":
        return BicubicRuntime(config, settings)
    if kind == "torch" and architecture == "realesrgan_x4plus":
        return RealESRGANTorchRuntime(config, settings)
    if kind == "torch" and architecture == "srcnn_rgb":
        return SRCNNRuntime(config, settings)
    return UnsupportedRuntime(config, settings)
