import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.logging_setup import request_id_var


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = str(uuid.uuid4())
        token = request_id_var.set(req_id)
        request.state.request_id = req_id
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-Id"] = req_id
        return response
