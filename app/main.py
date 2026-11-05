"""FastAPI 应用工厂与入口。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routers import circle_fit, holes, judge, target_track, tracking, refinement
from app.core.config import settings
from app.core.exceptions import (
    ServiceError,
    service_error_handler,
    unhandled_exception_handler,
)
from app.core.logging import logger, setup_logging
from app.schemas.detection import HealthResponse
from app.services.model_manager import ModelManager

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("正在加载 YOLO 模型...")
    manager = ModelManager()
    manager.load_all()
    app.state.model_manager = manager
    logger.info("已配置的模型加载完成")
    yield
    logger.info("服务关闭")


def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
    )

    application.add_exception_handler(ServiceError, service_error_handler)
    application.add_exception_handler(Exception, unhandled_exception_handler)

    application.include_router(holes.router)
    application.include_router(circle_fit.router)
    application.include_router(judge.router)
    application.include_router(tracking.router)
    application.include_router(target_track.router)
    application.include_router(refinement.router)

    @application.get("/health", response_model=HealthResponse, tags=["健康检查"])
    async def health() -> HealthResponse:
        return HealthResponse()

    return application


app = create_app()
