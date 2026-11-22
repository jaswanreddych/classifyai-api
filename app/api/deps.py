"""FastAPI dependencies. Routes ask for what they need instead of using globals."""

from fastapi import Request

from app.config import Settings
from app.errors import ModelUnavailableError
from app.services.catalog import PlantCatalog
from app.services.classifier import Classifier


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_catalog(request: Request) -> PlantCatalog:
    return request.app.state.catalog


def get_classifier(request: Request) -> Classifier:
    classifier = request.app.state.classifier
    if classifier is None:
        raise ModelUnavailableError(
            "The model is not loaded. Check the server logs and the models/ folder."
        )
    return classifier
