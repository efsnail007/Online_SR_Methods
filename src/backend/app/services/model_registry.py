from __future__ import annotations

from typing import Any

from app.core.config import ModelConfig, Settings
from app.ml.model_runtime import BaseModelRuntime, create_runtime


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._configs = {
            model.id: model for model in settings.models if model.enabled
        }
        if settings.default_model_id not in self._configs:
            raise ValueError(
                f"Default model '{settings.default_model_id}' is not configured."
            )
        self._runtimes: dict[str, BaseModelRuntime] = {}

    @property
    def default_model_id(self) -> str:
        return self.settings.default_model_id

    def resolve_model_id(self, model_id: str | None) -> str:
        return model_id or self.default_model_id

    def list_models(self) -> list[dict[str, Any]]:
        return [self._model_payload(config.id) for config in self._configs.values()]

    def get_runtime(self, model_id: str) -> BaseModelRuntime:
        if model_id not in self._configs:
            raise ValueError(f"Unknown model_id: {model_id}")
        if model_id not in self._runtimes:
            self._runtimes[model_id] = create_runtime(
                self._configs[model_id],
                self.settings,
            )
        return self._runtimes[model_id]

    def load(self, model_id: str) -> None:
        self.get_runtime(model_id).load()

    def load_startup_models(self) -> None:
        for model_id in self.settings.startup_model_ids:
            self.load(model_id)

    def unload_all(self) -> None:
        for runtime in self._runtimes.values():
            runtime.unload()

    def model_info(self, model_id: str | None = None) -> dict[str, Any]:
        resolved_model_id = model_id or self.default_model_id
        if resolved_model_id not in self._configs:
            raise ValueError(f"Unknown model_id: {resolved_model_id}")
        return self._model_payload(resolved_model_id)

    def _model_payload(self, model_id: str) -> dict[str, Any]:
        config = self._configs[model_id]
        runtime = self._runtimes.get(model_id)
        if runtime is None:
            return {
                "id": config.id,
                "name": config.name,
                "kind": config.kind,
                "architecture": config.architecture,
                "loaded": False,
                "weights_path": (
                    None if config.weights_path is None else str(config.weights_path)
                ),
                "device": config.device or self.settings.model_device,
                "scale": config.scale,
                "description": config.description,
                "tags": config.tags,
                "options": config.options,
            }
        return runtime.info()
