"""Application factory.

``create_app`` takes optional overrides so tests can inject settings and a fake
classifier. Production code just uses the module-level ``app``.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import health, plants, predictions
from app.config import Settings, get_settings
from app.errors import register_error_handlers
from app.services.catalog import PlantCatalog
from app.services.classifier import Classifier, KerasClassifier

logger = logging.getLogger("app")

DESCRIPTION = """
Identify medicinal plants from a photo of a leaf.

* `POST /api/v1/predictions` uploads an image and returns the top matches
  plus full details of the best one.
* `GET /api/v1/plants` browses and searches the 78 supported plants.

Plant information is for general reference only and is not medical advice.
"""


def create_app(
    settings: Settings | None = None, classifier: Classifier | None = None
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.catalog = PlantCatalog.from_file(settings.data_dir / "plants.json")
        logger.info("Loaded %d plants", len(app.state.catalog))

        app.state.classifier = classifier
        if classifier is None and settings.load_model:
            try:
                app.state.classifier = await run_in_threadpool(
                    KerasClassifier.load,
                    settings.model_dir,
                    settings.data_dir / "labels.json",
                )
                logger.info("Model loaded from %s", settings.model_dir)
            except Exception:
                # Stay up so /health/ready can report the problem clearly.
                logger.exception(
                    "Could not load the model; predictions will return 503"
                )
        yield

    app = FastAPI(
        title="ClassifyAI API",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %d in %.0f ms [%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response

    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(plants.router, prefix="/api/v1")
    app.include_router(predictions.router, prefix="/api/v1")
    return app


app = create_app()
