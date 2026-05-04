from __future__ import annotations

from functools import partial
from typing import Literal

from app.schemas.health import HealthResponse, ModelInfoResponse
from app.schemas.inference import (
    Base64UpscaleRequest,
    Base64UpscaleResponse,
    ModelsResponse,
)
from app.services.image_codec import bytes_to_base64
from app.services.super_resolution import EncodedImageResult, SuperResolutionService
from fastapi import APIRouter, HTTPException, Query, Request, Response
from starlette.concurrency import run_in_threadpool

router = APIRouter()


def get_service(request: Request) -> SuperResolutionService:
    service = getattr(request.app.state, "sr_service", None)
    if service is None:
        raise HTTPException(
            status_code=503, detail="Super-resolution service is not ready."
        )
    return service


def build_base64_response(result: EncodedImageResult) -> Base64UpscaleResponse:
    return Base64UpscaleResponse(
        image_base64=bytes_to_base64(result.image_bytes),
        content_type=result.content_type,
        inference_time_ms=result.inference_time_ms,
        input_width=result.input_width,
        input_height=result.input_height,
        output_width=result.output_width,
        output_height=result.output_height,
        outscale=result.outscale,
        model_id=result.model_id,
        model_kind=result.model_kind,
        device=result.device,
        model_name=result.model_name,
    )


@router.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {"message": "Real-ESRGAN backend is running."}


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    service = get_service(request)
    model_info = service.model_info()
    return HealthResponse(
        status="ok",
        app_name=request.app.title,
        model_loaded=model_info["model_loaded"],
        model_id=model_info["model_id"],
        model_name=model_info["model_name"],
        model_kind=model_info["model_kind"],
        device=model_info["device"],
        weights_path=model_info["weights_path"],
    )


@router.get("/model", response_model=ModelInfoResponse, tags=["model"])
async def model_info(
    request: Request,
    model_id: str | None = Query(default=None),
) -> ModelInfoResponse:
    service = get_service(request)
    try:
        return ModelInfoResponse.model_validate(service.model_info(model_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/models", response_model=ModelsResponse, tags=["model"])
async def models(request: Request) -> ModelsResponse:
    service = get_service(request)
    return ModelsResponse(
        default_model_id=service.settings.default_model_id,
        models=service.models_info(),
    )


@router.post(
    "/upscale/base64", response_model=Base64UpscaleResponse, tags=["inference"]
)
async def upscale_base64(
    payload: Base64UpscaleRequest,
    request: Request,
) -> Base64UpscaleResponse:
    service = get_service(request)
    try:
        result = await run_in_threadpool(
            partial(
                service.process_base64_image,
                payload.image_base64,
                payload.outscale,
                payload.model_id,
                payload.output_format,
                payload.jpeg_quality,
                payload.png_compression,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return build_base64_response(result)


@router.post(
    "/upscale",
    tags=["inference"],
    responses={200: {"content": {"image/jpeg": {}, "image/png": {}}}},
)
async def upscale_raw(
    request: Request,
    outscale: float | None = Query(default=None, gt=0.0, le=8.0),
    model_id: str | None = Query(default=None),
    output_format: Literal["jpeg", "jpg", "png"] | None = Query(default=None),
    jpeg_quality: int | None = Query(default=None, ge=1, le=100),
    png_compression: int | None = Query(default=None, ge=0, le=9),
) -> Response:
    service = get_service(request)
    payload = await request.body()
    try:
        result = await run_in_threadpool(
            partial(
                service.process_raw_image,
                payload,
                outscale,
                model_id,
                output_format,
                jpeg_quality,
                png_compression,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    headers = {
        "X-Inference-Time-Ms": f"{result.inference_time_ms:.3f}",
        "X-Model-Id": result.model_id,
        "X-Model-Name": result.model_name,
        "X-Model-Kind": result.model_kind,
        "X-Model-Device": result.device,
        "X-Outscale": f"{result.outscale:.3f}",
        "X-Output-Width": str(result.output_width),
        "X-Output-Height": str(result.output_height),
    }
    return Response(
        content=result.image_bytes, media_type=result.content_type, headers=headers
    )
