from fastapi import Request, status
from fastapi.responses import JSONResponse
from common.exceptions import (
    IRSystemException,
    IndexNotFoundError,
    InvalidQueryError,
    ModelLoadError
)
import logging

logger = logging.getLogger(__name__)

async def ir_system_exception_handler(request: Request, exc: IRSystemException):
    logger.error(f"{exc.__class__.__name__}: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": exc.__class__.__name__,
            "detail": str(exc)
        }
    )

async def index_not_found_handler(request: Request, exc: IndexNotFoundError):
    logger.warning(f"Index not found: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "IndexNotFound",
            "detail": str(exc)
        }
    )

async def invalid_query_handler(request: Request, exc: InvalidQueryError):
    logger.warning(f"Invalid query: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "InvalidQuery",
            "detail": str(exc)
        }
    )

async def value_error_handler(request: Request, exc: ValueError):
    logger.warning(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "ValueError",
            "detail": str(exc)
        }
    )

async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    logger.error(f"FileNotFoundError: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "FileNotFound",
            "detail": str(exc)
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred"
        }
    )

async def model_load_error_handler(request: Request, exc: ModelLoadError):
    logger.error(f"Model load error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "ModelLoadError",
            "detail": str(exc)
        }
    )

def register_exception_handlers(app):
    app.add_exception_handler(IRSystemException, ir_system_exception_handler)
    app.add_exception_handler(IndexNotFoundError, index_not_found_handler)
    app.add_exception_handler(InvalidQueryError, invalid_query_handler)
    app.add_exception_handler(ModelLoadError, model_load_error_handler)
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(FileNotFoundError, file_not_found_handler)
    app.add_exception_handler(Exception, general_exception_handler)
