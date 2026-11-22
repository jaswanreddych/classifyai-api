import logging
import time

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.deps import get_catalog, get_classifier, get_settings_dep
from app.config import Settings
from app.errors import PayloadTooLargeError, UnsupportedMediaTypeError
from app.schemas import ErrorResponse, Prediction, PredictionResponse
from app.services.catalog import PlantCatalog
from app.services.classifier import Classifier
from app.services.imaging import preprocess

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post(
    "",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Not a readable image"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Not JPEG, PNG or WebP"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    summary="Identify a plant from a leaf photo",
)
async def create_prediction(
    file: UploadFile = File(description="A JPEG, PNG or WebP photo of a leaf."),
    top_k: int = Query(
        default=3, ge=1, le=5, description="How many matches to return."
    ),
    classifier: Classifier = Depends(get_classifier),
    catalog: PlantCatalog = Depends(get_catalog),
    settings: Settings = Depends(get_settings_dep),
) -> PredictionResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise UnsupportedMediaTypeError("Upload a JPEG, PNG or WebP image.")

    # Read one byte past the limit so an oversized file is detected without
    # loading all of it into memory.
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise PayloadTooLargeError(
            f"The image is larger than {settings.max_upload_mb} MB."
        )

    # Decoding and inference are CPU-bound and blocking. Run them in the thread
    # pool so the event loop keeps serving other requests (health checks, etc.).
    started = time.perf_counter()
    image = await run_in_threadpool(preprocess, data)
    ranked = await run_in_threadpool(classifier.predict, image, top_k)
    inference_ms = round((time.perf_counter() - started) * 1000, 1)

    predictions = []
    for label, probability in ranked:
        plant = catalog.get(label)
        if plant is None:  # a model label with no catalog entry is a data bug
            logger.error("No catalog entry for model label %r", label)
            continue
        predictions.append(
            Prediction(
                plant_id=plant.id,
                name=plant.name,
                scientific_name=plant.scientific_name,
                confidence=round(probability, 4),
            )
        )

    best = catalog.get(ranked[0][0])
    return PredictionResponse(
        predictions=predictions,
        low_confidence=ranked[0][1] < settings.low_confidence_threshold,
        plant=best,
        model=classifier.name,
        inference_ms=inference_ms,
    )
