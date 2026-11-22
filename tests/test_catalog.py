import json
from pathlib import Path

import pytest

from app.config import BASE_DIR
from app.services.catalog import PlantCatalog

DATA_DIR = BASE_DIR / "app" / "data"


@pytest.fixture(scope="module")
def catalog() -> PlantCatalog:
    return PlantCatalog.from_file(DATA_DIR / "plants.json")


def test_every_model_label_has_a_plant(catalog):
    """The model can only return labels from labels.json. Each needs a catalog entry,
    otherwise a prediction for that class would have nothing to show."""
    labels = json.loads(Path(DATA_DIR / "labels.json").read_text(encoding="utf-8"))
    missing = [label for label in labels if catalog.get(label) is None]
    assert missing == []
    assert len(catalog) == len(labels) == 78


def test_labels_are_unique_and_sorted():
    labels = json.loads((DATA_DIR / "labels.json").read_text(encoding="utf-8"))
    assert len(set(labels)) == len(labels)
    # Keras assigns class indices by sorted folder name; the order must match training.
    assert labels == sorted(labels)


def test_get_returns_full_plant(catalog):
    plant = catalog.get("Tulsi")
    assert plant is not None
    assert plant.scientific_name
    assert plant.uses.medicinal


def test_get_unknown_returns_none(catalog):
    assert catalog.get("not-a-plant") is None


def test_search_matches_name_scientific_and_common_names(catalog):
    assert [p.id for p in catalog.search("tulsi")] == ["Tulsi"]
    assert "Tulsi" in [p.id for p in catalog.search(plant_scientific(catalog, "Tulsi"))]


def test_search_is_case_insensitive_and_empty_query_lists_all(catalog):
    assert catalog.search("TULSI") == catalog.search("tulsi")
    assert len(catalog.search("")) == len(catalog.search(None)) == 78


def test_search_with_no_match_is_empty(catalog):
    assert catalog.search("zzzz-no-such-plant") == []


def plant_scientific(catalog: PlantCatalog, plant_id: str) -> str:
    return catalog.get(plant_id).scientific_name
