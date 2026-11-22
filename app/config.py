"""Application settings, read from environment variables or a local .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Every setting can be overridden with a CLASSIFYAI_ prefixed variable.

    Example: ``CLASSIFYAI_MAX_UPLOAD_MB=10``.
    """

    model_config = SettingsConfigDict(
        env_prefix="CLASSIFYAI_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    model_dir: Path = BASE_DIR / "models"
    data_dir: Path = BASE_DIR / "app" / "data"

    # Load the TensorFlow model on startup. Tests turn this off.
    load_model: bool = True

    max_upload_mb: int = 5
    # Below this confidence the response is flagged as low confidence.
    low_confidence_threshold: float = 0.5

    log_level: str = "INFO"
    cors_origins: list[str] = []

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
