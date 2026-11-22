from fastapi import APIRouter, Depends, Request, Response, status

from app.api.deps import get_catalog
from app.schemas import ReadinessResponse
from app.services.catalog import PlantCatalog

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", summary="Liveness: the process is up")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Readiness: the model is loaded and the API can serve predictions",
    response_model=ReadinessResponse,
)
def ready(
    request: Request,
    response: Response,
    catalog: PlantCatalog = Depends(get_catalog),
) -> ReadinessResponse:
    model_loaded = request.app.state.classifier is not None
    if not model_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        plants=len(catalog),
    )
