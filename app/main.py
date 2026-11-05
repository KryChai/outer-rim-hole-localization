"""FastAPI 应用工厂与入口。"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api.routers import (
    circle_fit, circle_refinement, ellipse_refinement, holes, judge, target_track, tracking,
)
from app.core import audit
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

    @application.middleware("http")
    async def request_log_middleware(request: Request, call_next):
        if request.url.path == "/health" and not settings.log_health_check:
            return await call_next(request)
        request_id = audit.bind_request_id()
        query = f"?{request.url.query}" if request.url.query else ""
        size = request.headers.get("content-length", "-")
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.error("rid=%s | %s %s%s | 500 | %.1fms | %sB",
                         request_id, request.method, request.url.path, query,
                         (time.perf_counter() - start) * 1000, size)
            raise
        logger.info("rid=%s | %s %s%s | %s | %.1fms | %sB",
                    request_id, request.method, request.url.path, query,
                    response.status_code, (time.perf_counter() - start) * 1000, size)
        return response

    application.include_router(holes.router)
    application.include_router(circle_fit.router)
    application.include_router(judge.router)
    application.include_router(tracking.router)
    application.include_router(target_track.router)
    application.include_router(circle_refinement.router)
    application.include_router(ellipse_refinement.router)

    @application.get("/health", response_model=HealthResponse, tags=["健康检查"])
    async def health() -> HealthResponse:
        return HealthResponse()

    return application


app = create_app()
