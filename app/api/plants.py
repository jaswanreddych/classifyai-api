from fastapi import APIRouter, Depends, Query

from app.api.deps import get_catalog
from app.errors import PlantNotFoundError
from app.schemas import ErrorResponse, Plant, PlantSummary
from app.services.catalog import PlantCatalog

router = APIRouter(prefix="/plants", tags=["plants"])


@router.get("", response_model=list[PlantSummary], summary="List or search plants")
def list_plants(
    q: str | None = Query(
        default=None,
        max_length=100,
        description="Match against name, scientific name or any common name.",
    ),
    catalog: PlantCatalog = Depends(get_catalog),
) -> list[PlantSummary]:
    return catalog.search(q)


@router.get(
    "/{plant_id}",
    response_model=Plant,
    responses={404: {"model": ErrorResponse}},
    summary="Get one plant",
)
def get_plant(plant_id: str, catalog: PlantCatalog = Depends(get_catalog)) -> Plant:
    plant = catalog.get(plant_id)
    if plant is None:
        raise PlantNotFoundError(f"No plant with id '{plant_id}'.")
    return plant
