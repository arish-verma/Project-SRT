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

    # One consistent specialist model for airborne-object detection.
    drone_model_path: str = "Javvanny/yolov8m_flying_objects_detection"
    drone_model_file: str = "yolov8m/weights/best.pt"
    # Low candidate threshold + strict post-filtering. This lets the model
    # surface small/oblique helicopters and drones without trusting them yet.
    drone_model_confidence: float = 0.25

    # Compatibility fields only. The fallback model is not mixed into live
    # inference because differing class vocabularies caused class oscillation.
    drone_fallback_model_path: str = "QuincySorrentino/AeroYOLO"
    drone_fallback_model_file: str = "best.pt"
    drone_fallback_confidence: float = 0.20

    # Precision-first airborne inference.
    drone_scan_interval: int = 5
    drone_tiled_scan_every: int = 1
    drone_min_confidence: float = 0.45
    airplane_min_confidence: float = 0.50
    helicopter_min_confidence: float = 0.45
    bird_min_confidence: float = 0.65
    flying_model_imgsz: int = 960

    # Geometry gate: allow a close helicopter/aircraft to occupy a meaningful
    # part of frame; temporal + motion gates provide the stronger false-positive
    # protection for ordinary webcam scenes.
    flying_min_box_px: int = 8
    flying_max_box_area_ratio: float = 0.18
    flying_max_center_y_ratio: float = 0.85

    # Temporal object lock: a class must agree across multiple specialist
    # scans and the same physical region must persist between scans.
    flying_match_iou: float = 0.15
    flying_vote_window: int = 6
    flying_required_votes: int = 3
    flying_label_consensus_ratio: float = 0.67

    # Moving airborne targets need measurable displacement. Stationary hover
    # is allowed only after a longer, high-confidence confirmation.
    flying_min_motion_ratio: float = 0.015
    flying_hover_confidence: float = 0.82
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
