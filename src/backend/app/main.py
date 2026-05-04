from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from app.api import router as api_router
from app.core.config import settings
from app.services.super_resolution import SuperResolutionService
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = SuperResolutionService(settings)
    logger.info("Loading model from %s", settings.model_weights_path)
    await run_in_threadpool(service.load)
    app.state.sr_service = service
    try:
        yield
    finally:
        await run_in_threadpool(service.unload)


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[
                "X-Inference-Time-Ms",
                "X-Model-Name",
                "X-Model-Device",
                "X-Outscale",
                "X-Upscale-Method",
                "X-Output-Width",
                "X-Output-Height",
            ],
        )
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
