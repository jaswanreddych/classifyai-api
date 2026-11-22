"""Read-only plant knowledge base, loaded once from app/data/plants.json."""

import json
from pathlib import Path

from app.schemas import Plant, PlantSummary


class PlantCatalog:
    def __init__(self, plants: list[Plant]) -> None:
        self._by_id = {plant.id: plant for plant in plants}
        if len(self._by_id) != len(plants):
            raise ValueError("plants.json contains duplicate ids")

    @classmethod
    def from_file(cls, path: Path) -> "PlantCatalog":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls([Plant.model_validate(item) for item in raw])

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, plant_id: str) -> Plant | None:
        return self._by_id.get(plant_id)

    def search(self, query: str | None = None) -> list[PlantSummary]:
        """List plants, optionally filtered by a case-insensitive text match.

        The query is matched against the name, scientific name and every
        common name, so 'tulsi', 'ocimum' and 'holy basil' all work.
        """
        needle = (query or "").strip().lower()
        matches = []
        for plant in self._by_id.values():
            if needle and not self._matches(plant, needle):
                continue
            matches.append(PlantSummary.model_validate(plant.model_dump()))
        return sorted(matches, key=lambda p: p.name.lower())

    @staticmethod
    def _matches(plant: Plant, needle: str) -> bool:
        haystack = [plant.name, plant.scientific_name]
        for names in plant.common_names.values():
            haystack.extend(names)
        return any(needle in text.lower() for text in haystack)
