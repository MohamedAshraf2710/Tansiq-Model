from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres.owggymklkuanjfcirbvc:hj&Am-8MNeeaCpL@aws-1-eu-central-1.pooler.supabase.com:6543/postgres?sslmode=require"
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    MODEL_NAME: str = "gemini-3-flash"
    
    class Config:
        env_file = ".env"

settings = Settings()
