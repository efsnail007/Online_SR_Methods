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


class OnnxRuntime(BaseModelRuntime):
    def __init__(self, config: ModelConfig, settings: Settings) -> None:
        super().__init__(config, settings)
        self.session: Any | None = None
        self.input_name: str | None = None
        self.requested_providers: list[str] = []
        self.available_providers: list[str] = []
        self.active_providers: list[str] = []

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def load(self) -> None:
        if self.session is not None:
            return
        if self.config.weights_path is None:
            raise FileNotFoundError(
                f"ONNX model path is not configured: {self.config.id}"
            )
        if not self.config.weights_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.config.weights_path}"
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for ONNX models. Install the CPU or CUDA extra."
            ) from exc

        providers = self.config.options.get("providers")
        if providers is None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device.type == "cuda"
                else ["CPUExecutionProvider"]
            )
        self.requested_providers = list(providers)
        self.available_providers = list(ort.get_available_providers())
        missing_providers = [
            provider
            for provider in self.requested_providers
            if provider != "CPUExecutionProvider"
            and provider not in self.available_providers
        ]
        if missing_providers:
            raise RuntimeError(
                "Requested ONNX provider(s) are not available: "
                f"{', '.join(missing_providers)}. "
                "Install the CUDA extra and run the CUDA backend."
            )
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = ort.InferenceSession(
            str(self.config.weights_path),
            sess_options=session_options,
            providers=self.requested_providers,
        )
        self.active_providers = list(self.session.get_providers())
        if (
            self.device.type == "cuda"
            and "CUDAExecutionProvider" not in self.active_providers
        ):
            raise RuntimeError(
                "ONNX session was created without CUDAExecutionProvider. "
                f"Available providers: {self.available_providers}. "
                f"Active providers: {self.active_providers}."
            )
        self.input_name = self.session.get_inputs()[0].name

    def unload(self) -> None:
        self.session = None
        self.input_name = None
        self.active_providers = []

    def info(self) -> dict[str, Any]:
        payload = super().info()
        payload.update(
            {
                "requested_providers": self.requested_providers,
                "available_providers": self.available_providers,
                "active_providers": self.active_providers,
            }
        )
        return payload

    def upscale(self, image_bgr: np.ndarray, outscale: float) -> RuntimeResult:
        self.load()
        with self._lock:
            return self._run_inference(image_bgr, outscale)

    def _run_inference(self, image_bgr: np.ndarray, outscale: float) -> RuntimeResult:
        if self.session is None or self.input_name is None:
            raise RuntimeError(f"Model is not loaded: {self.config.id}")
        start_time = time.perf_counter()
        image_rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
        input_tensor = np.transpose(image_rgb, (2, 0, 1))[None, ...]
        outputs = self.session.run(None, {self.input_name: input_tensor})
        output = np.asarray(outputs[0])
        if output.ndim == 4:
            output = output[0]
        if output.shape[0] in {1, 3}:
            output = np.transpose(output, (1, 2, 0))
        if output.shape[2] == 1:
            output = np.repeat(output, 3, axis=2)
        output_bgr = np.clip(output[:, :, ::-1], 0.0, 1.0)
        model_scale = self.config.scale
        if model_scale and outscale != float(model_scale):
            target_hw = (
                max(1, int(round(image_bgr.shape[0] * outscale))),
                max(1, int(round(image_bgr.shape[1] * outscale))),
            )
            output_bgr = torch_resize_bgr(output_bgr, target_hw)
        output_bgr_u8 = np.clip(np.round(output_bgr * 255.0), 0, 255).astype(np.uint8)
        elapsed_ms = float((time.perf_counter() - start_time) * 1000.0)
        return RuntimeResult(output_bgr_u8, elapsed_ms)


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


def torch_resize_bgr(
    image_bgr_float: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    tensor = torch.from_numpy(image_bgr_float).permute(2, 0, 1).unsqueeze(0)
    output = F.interpolate(tensor, size=target_hw, mode="bicubic", align_corners=False)
    return output.squeeze(0).permute(1, 2, 0).clamp_(0.0, 1.0).numpy()


def create_runtime(config: ModelConfig, settings: Settings) -> BaseModelRuntime:
    kind = config.kind.lower()
    architecture = (config.architecture or "").lower()
    if kind == "bicubic" or architecture == "bicubic":
        return BicubicRuntime(config, settings)
    if kind == "torch" and architecture == "realesrgan_x4plus":
        return RealESRGANTorchRuntime(config, settings)
    if kind == "onnx":
        return OnnxRuntime(config, settings)
    return UnsupportedRuntime(config, settings)
