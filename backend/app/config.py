"""
config.py — Decoupled configuration management using pydantic-settings.
All secrets are loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    DATABASE_URL: str = ""
    GROQ_API_KEY: str = ""
    API_SECRET_KEY: str = ""
    MODEL_NAME: str = "llama-3.3-70b-versatile"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
