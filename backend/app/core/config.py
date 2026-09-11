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

    # Airborne-object specialist model. It supports Drone, Airplane,
    # Helicopter and Bird classes.
    drone_model_path: str = "Javvanny/yolov8m_flying_objects_detection"
    drone_model_file: str = "yolov8m/weights/best.pt"
    drone_model_confidence: float = 0.20

    drone_fallback_model_path: str = "QuincySorrentino/AeroYOLO"
    drone_fallback_model_file: str = "best.pt"
    drone_fallback_confidence: float = 0.20

    # Candidate thresholds are intentionally permissive; the specialist model
    # already limits the class vocabulary and temporal confirmation remains in
    # the event layer for high-risk alerts.
    drone_min_confidence: float = 0.38
    airplane_min_confidence: float = 0.38
    helicopter_min_confidence: float = 0.32
    bird_min_confidence: float = 0.55
    flying_model_imgsz: int = 960

    flying_min_box_px: int = 6
    flying_max_box_area_ratio: float = 0.45
    flying_max_center_y_ratio: float = 0.95

    flying_match_iou: float = 0.15
    flying_vote_window: int = 6
    flying_required_votes: int = 3
    flying_label_consensus_ratio: float = 0.67

    flying_min_motion_ratio: float = 0.008
    flying_hover_confidence: float = 0.78
    flying_hover_required_votes: int = 5
    flying_clear_after_misses: int = 2

    anpr_scan_interval: int = 30
    face_scan_interval: int = 15
    event_storage_path: str = "storage/events"
    upload_storage_path: str = "storage/uploads"
    max_upload_size_mb: int = 500
    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
