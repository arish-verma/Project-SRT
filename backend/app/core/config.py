from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Project SRT"
    app_version: str = "0.2.0"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://srt:srt@localhost:5432/srt"
    redis_url: str = "redis://localhost:6379/0"
    model_path: str = "yolo11n.pt"
    model_confidence: float = 0.35
    # Run the general detector every N frames and reuse tracks between inference frames.
    detection_interval: int = 2
    # Unified aerial-object detection is intentionally less frequent than the base detector
    # because the specialist models are substantially heavier. Results are held on screen.
    drone_model_path: str = "QuincySorrentino/AeroYOLO"
    drone_model_confidence: float = 0.20
    drone_fallback_model_path: str = "sapoepsilon/yolov11s-drone-detector"
    drone_fallback_confidence: float = 0.12
    drone_scan_interval: int = 10
    # OCR is CPU-heavy. Vehicle classification remains live, while OCR is sampled and cached.
    anpr_scan_interval: int = 30
    face_scan_interval: int = 15
    event_storage_path: str = "storage/events"
    upload_storage_path: str = "storage/uploads"
    max_upload_size_mb: int = 500
    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
