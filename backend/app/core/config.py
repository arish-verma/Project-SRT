from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Project SRT"
    app_version: str = "0.3.0"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://srt:srt@localhost:5432/srt"
    redis_url: str = "redis://localhost:6379/0"
    model_path: str = "yolo11n.pt"
    model_confidence: float = 0.35
    detection_interval: int = 2

    # Dedicated flying-object classifier. AeroYOLO has exactly three classes:
    # aircraft, drone and helicopter. The application maps aircraft -> airplane
    # for a clearer operator-facing label.
    drone_model_path: str = "QuincySorrentino/AeroYOLO"
    drone_model_confidence: float = 0.25
    drone_fallback_model_path: str = "sapoepsilon/yolov11s-drone-detector"
    drone_fallback_confidence: float = 0.30
    drone_scan_interval: int = 6
    aerial_image_size: int = 960
    aerial_tiled_confidence: float = 0.20
    drone_max_area_ratio: float = 0.03
    drone_max_width_ratio: float = 0.15
    drone_max_height_ratio: float = 0.15
    aerial_confirmation_hits: int = 2
    aerial_track_ttl_seconds: float = 2.5
    drone_alert_cooldown_seconds: float = 30.0

    anpr_scan_interval: int = 30
    face_scan_interval: int = 15
    event_storage_path: str = "storage/events"
    upload_storage_path: str = "storage/uploads"
    max_upload_size_mb: int = 500
    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
