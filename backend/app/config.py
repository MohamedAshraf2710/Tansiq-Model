"""
config.py — Decoupled configuration management using pydantic-settings.
All secrets are loaded from environment variables / .env file.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    DATABASE_URL: str = Field(
        ...,
        env="DATABASE_URL",
        description="PostgreSQL connection URL via Supavisor pooler (port 6543)",
    )
    GEMINI_API_KEY: str = Field(
        ...,
        env="GEMINI_API_KEY",
        description="Google Gemini API key for LLM inference",
    )
    MODEL_NAME: str = "gemini-2.0-flash"

    class Config:
        env_file = ".env"


settings = Settings()
