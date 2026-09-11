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
    # normal COCO classes; flying-object classification is kept separate.
    model_path: str = "yolo11n.pt"
    model_confidence: float = 0.35
    detection_interval: int = 2

    # Specialist flying-object model. One consistent model is used for all
    # airborne classes; mixing different class vocabularies caused instability.
    drone_model_path: str = "Javvanny/yolov8m_flying_objects_detection"
    drone_model_file: str = "yolov8m/weights/best.pt"
    drone_model_confidence: float = 0.45

    # Retained as configuration compatibility fields; the fallback model is no
    # longer mixed into live classification because it caused label oscillation.
    drone_fallback_model_path: str = "QuincySorrentino/AeroYOLO"
    drone_fallback_model_file: str = "best.pt"
    drone_fallback_confidence: float = 0.20

    # Precision-first airborne inference. A result must agree over multiple
    # specialist scans before it is shown as a live flying-object detection.
    drone_scan_interval: int = 5
    drone_tiled_scan_every: int = 1
    drone_min_confidence: float = 0.55
    airplane_min_confidence: float = 0.62
    helicopter_min_confidence: float = 0.60
    bird_min_confidence: float = 0.68
    flying_model_imgsz: int = 960
    flying_min_box_px: int = 10
    flying_max_box_area_ratio: float = 0.12
    flying_max_center_y_ratio: float = 0.90
    flying_match_iou: float = 0.25
    flying_vote_window: int = 5
    flying_required_votes: int = 3
    flying_clear_after_misses: int = 2

    anpr_scan_interval: int = 30
    face_scan_interval: int = 15
    event_storage_path: str = "storage/events"
    upload_storage_path: str = "storage/uploads"
    max_upload_size_mb: int = 500
    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
