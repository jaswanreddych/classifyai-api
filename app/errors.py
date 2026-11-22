"""Domain errors and the handler that turns them into a uniform JSON body.

Every error response has the same shape::

    {"error": {"code": "invalid_image", "message": "..."}}
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class PlantNotFoundError(AppError):
    status_code = 404
    code = "plant_not_found"


class InvalidImageError(AppError):
    status_code = 400
    code = "invalid_image"


class UnsupportedMediaTypeError(AppError):
    status_code = 415
    code = "unsupported_media_type"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"


class ModelUnavailableError(AppError):
    status_code = 503
    code = "model_unavailable"


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )
