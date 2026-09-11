from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Project SRT"
    app_version: str = "0.3.0"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://srt:srt@localhost:5432/srt"
    redis_url: str = "redis://localhost:6379/0"

    # General CCTV detector. It handles people, vehicles and the rest of the
    # normal COCO classes; flying-object classification is deliberately kept
    # in a separate specialist model.
    model_path: str = "yolo11n.pt"
    model_confidence: float = 0.35
    detection_interval: int = 2

    # Specialist flying-object classifier/detector.
    # Javvanny's YOLOv8m model explicitly separates Drone/Airplane/Helicopter/Bird.
    drone_model_path: str = "Javvanny/yolov8m_flying_objects_detection"
    drone_model_file: str = "yolov8m/weights/best.pt"
    drone_model_confidence: float = 0.20

    # Smaller YOLO11 specialist used only if the primary model cannot be loaded
    # or inference fails. AeroYOLO exposes aircraft/drone/helicopter classes.
    drone_fallback_model_path: str = "QuincySorrentino/AeroYOLO"
    drone_fallback_model_file: str = "best.pt"
    drone_fallback_confidence: float = 0.20

    # Specialist inference is expensive on a 4 GB laptop GPU. Results are held
    # between scans so the live feed stays responsive.
    drone_scan_interval: int = 10
    drone_tiled_scan_every: int = 3
    drone_min_confidence: float = 0.28
    airplane_min_confidence: float = 0.38
    helicopter_min_confidence: float = 0.35
    bird_min_confidence: float = 0.50

    anpr_scan_interval: int = 30
    face_scan_interval: int = 15
    event_storage_path: str = "storage/events"
    upload_storage_path: str = "storage/uploads"
    max_upload_size_mb: int = 500
    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
