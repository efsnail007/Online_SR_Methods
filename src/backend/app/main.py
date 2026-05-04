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
    logger.info("Loading startup models: %s", ", ".join(settings.startup_model_ids))
    await run_in_threadpool(service.load)
    app.state.sr_service = service
    try:
        yield
    finally:
        await run_in_threadpool(service.unload)


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    if settings.cors_origins or settings.cors_origin_regex:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_origin_regex=settings.cors_origin_regex,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[
                "X-Inference-Time-Ms",
                "X-Model-Id",
                "X-Model-Name",
                "X-Model-Kind",
                "X-Model-Device",
                "X-Outscale",
                "X-Output-Width",
                "X-Output-Height",
            ],
        )
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
