"""Request and response models. These also generate the OpenAPI docs."""

from pydantic import BaseModel, ConfigDict, Field


class PlantUses(BaseModel):
    medicinal: list[str]
    other: list[str]


class PlantSummary(BaseModel):
    id: str = Field(description="Class label used by the model, e.g. 'Aloevera'.")
    name: str
    scientific_name: str
    family: str


class Plant(PlantSummary):
    common_names: dict[str, list[str]] = Field(
        description="Common names keyed by language."
    )
    description: str
    native_to: str
    active_compounds: list[str]
    uses: PlantUses


class Prediction(BaseModel):
    plant_id: str
    name: str
    scientific_name: str
    confidence: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predictions: list[Prediction] = Field(description="Top matches, most likely first.")
    low_confidence: bool = Field(
        description="True when the best match is below the confidence threshold."
    )
    plant: Plant = Field(description="Full details of the best match.")
    model: str
    inference_ms: float


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    plants: int


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
