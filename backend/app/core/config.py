from pathlib import Path

from pydantic_settings import BaseSettings

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    disease_model_checkpoint: str = str(REPO_ROOT / "ml" / "checkpoints" / "disease_model.pt")
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Phase 4/5 — not wired up yet, kept here so env config is centralized.
    anthropic_api_key: str = ""
    openweather_api_key: str = ""

    class Config:
        env_file = str(REPO_ROOT / ".env")
        env_file_encoding = "utf-8"


settings = Settings()
