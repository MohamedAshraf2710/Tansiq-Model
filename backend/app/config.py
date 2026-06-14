"""
config.py — Decoupled configuration management using pydantic-settings.
All secrets are loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    DATABASE_URL: str = ""
    GEMINI_API_KEY: str = ""
    MODEL_NAME: str = "gemini-2.0-flash"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
