"""自定义异常与全局异常处理，保证接口返回统一错误结构。"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.logging import logger


class ServiceError(Exception):
    """业务层可预期的异常，可指定 HTTP 状态码。"""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


async def service_error_handler(_: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.status_code, "error": exc.message},
    )


async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    # 完整堆栈只记录在服务端，避免泄露内部细节给客户端
    logger.exception("未处理异常：%s", exc)
    return JSONResponse(
        status_code=500,
        content={"code": 500, "error": "服务异常"},
    )
