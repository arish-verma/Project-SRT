from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Project SRT"
    app_version: str = "0.1.0"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://srt:srt@localhost:5432/srt"
    redis_url: str = "redis://localhost:6379/0"
    model_path: str = "models"
    event_storage_path: str = "storage/events"
    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
