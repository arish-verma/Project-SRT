from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Project SRT"
    environment: str = "development"
    log_level: str = "INFO"
    api_port: int = 8000
    database_url: str = "postgresql://srt:srt@localhost:5432/srt"
    redis_url: str = "redis://localhost:6379/0"
    event_storage_path: str = "./storage/events"
    model_path: str = "./models"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
