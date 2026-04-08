from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/volleyball_analytics"

    # JWT
    SECRET_KEY: str = "volleyball-analytics-super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # App
    APP_NAME: str = "Volleyball Analytics Platform"
    DEBUG: bool = True
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # File Upload
    UPLOAD_DIR: str = "uploads"
    RALLIES_DIR: str = "rallies"
    MAX_UPLOAD_SIZE_MB: int = 500

    # Admin seed
    ADMIN_EMAIL: str = "admin@volleyball.com"
    ADMIN_PASSWORD: str = "Admin@123456"
    ADMIN_USERNAME: str = "admin"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure upload directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RALLIES_DIR, exist_ok=True)
    