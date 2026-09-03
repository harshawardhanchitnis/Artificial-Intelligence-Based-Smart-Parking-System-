from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time-Ms"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
logger = logging.getLogger("smart_parking.api")


def request_id_for(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def error_payload(
    request: Request,
    *,
    code: str,
    message: str,
    detail: object | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "detail": message,
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id_for(request),
        },
    }
    if detail is not None:
        payload["error"]["details"] = jsonable_encoder(detail)  # type: ignore[index]
    return payload


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    if isinstance(exc.detail, str):
        message = exc.detail
        details = None
    else:
        message = "Request failed"
        details = exc.detail
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(
            request,
            code=f"HTTP_{exc.status_code}",
            message=message,
            detail=details,
        ),
        headers=exc.headers,
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return JSONResponse(
        status_code=422,
        content=error_payload(
            request,
            code="VALIDATION_ERROR",
            message="Request validation failed",
            detail=exc.errors(),
        ),
    )


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        supplied_id = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = supplied_id if _REQUEST_ID_PATTERN.fullmatch(supplied_id) else uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "Unhandled request failure request_id=%s method=%s path=%s",
                request_id,
                request.method,
                request.url.path,
            )
            response = JSONResponse(
                status_code=500,
                content=error_payload(
                    request,
                    code="INTERNAL_SERVER_ERROR",
                    message="The local service could not complete the request",
                ),
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[PROCESS_TIME_HEADER] = f"{elapsed_ms:.2f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response
