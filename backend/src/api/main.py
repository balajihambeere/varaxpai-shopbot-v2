from fastapi import FastAPI

from src.api.deps import RequestIdMiddleware
from src.api.routes import catalog, chat, health
from src.config import settings
from src.logging_setup import configure_logging

configure_logging(settings.log_level)

app = FastAPI(title="catalog-retrieval-assistant")
app.add_middleware(RequestIdMiddleware)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(catalog.router)
