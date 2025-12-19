import os
from pydantic_settings import BaseSettings
from typing import List
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # Application
    app_name: str = "Physical AI & Humanoid Robotics Interactive Book"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    version: str = "1.0.0"

    # Database
    database_url: str = os.getenv("DATABASE_URL", "postgresql://username:password@localhost:5432/physical_humanoid_robotics")

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Claude
    claude_api_key: str = os.getenv("CLAUDE_API_KEY", "")

    # JWT
    secret_key: str = os.getenv("SECRET_KEY", "your-default-secret-key-change-in-production")
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # CORS
    allowed_origins: List[str] = ["*"]  # In production, specify your frontend domain

    model_config = {"env_file": ".env"}


settings = Settings()