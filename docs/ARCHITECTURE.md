# SRT Architecture

## Layers
1. Perception — raw video and frame handling.
2. Detection — people, vehicles and other supported objects.
3. Tracking — persistent track IDs and trajectories.
4. Context — zones, schedules, direction, dwell and scene/time context.
5. Event engine — converts context into normalized events and risk scores.
6. Alert/evidence — severity, lifecycle and evidence capture.
7. Operator/API — dashboard, REST and realtime event delivery.

## Dependency chain
`Camera → Video Ingestion → Frame Processing → Detection → Tracking → Zone Engine → Event Engine → Alert Engine → Evidence → Database → API → Dashboard`

## Module interfaces
Application code should depend on abstractions such as Detector, Tracker, FaceDetector, FaceRecognizer, PlateDetector, OCR, ActivityAnalyzer and EventEngine. Concrete models are registered separately.

## Universal event contract
```json
{
  "event_id": "EVT-001284",
  "camera_id": "CAM-04",
  "timestamp": "2026-09-10T02:14:31Z",
  "event_type": "INTRUSION",
  "severity": "HIGH",
  "confidence": 0.91,
  "risk_score": 82,
  "objects": [{"type": "person", "track_id": 17}],
  "zone": {"id": "ZONE-03", "name": "Restricted Border Area"},
  "location": {"camera_location": "North Gate"},
  "evidence": {"thumbnail": "...", "video_clip": "..."},
  "metadata": {"dwell_seconds": 46, "direction": "north-east"}
}
```

## Explainable risk scoring
Initial rules may add risk for restricted-zone entry, night presence, long dwell, unusual direction, multiple people, person/vehicle association and repeated entry. Scores are indicators, not claims of criminal intent.

## Storage
PostgreSQL stores metadata/events. Evidence files live on filesystem or later object storage; do not store video blobs directly in relational tables.

## Deployment
Start local GPU-first. Docker Compose is the initial reproducible deployment unit. Kubernetes and distributed workers are future options, not prerequisites.
