# Project SRT — Smart Recognition & Tracking

Project SRT is an AI-assisted intelligent video analytics platform for border-surveillance scenarios using existing CCTV/IP-camera infrastructure.

## End-to-end capabilities
- Webcam, uploaded sample video, local video path and RTSP/HTTP camera ingestion
- Browser-compatible MJPEG live stream
- Lazy YOLO person/vehicle detection with replaceable tracker boundary
- Dedicated flying-object detection and classification: **DRONE / AIRPLANE / HELICOPTER / BIRD**
- Drone-only security alerting with temporal confirmation; aircraft and helicopters do not trigger drone alerts
- Normalized polygon virtual fences and explainable intrusion reasoning
- Persistent SQLite event history and evidence JPEG capture
- Persistent operator alert queue with NEW / ACKNOWLEDGED / RESOLVED / DISMISSED states
- Real-time WebSocket feeds for events and alerts
- Analytics summary and flying-object model-health endpoints
- Command-center dashboard
- Deterministic natural-language-style event search
- Optional local face detection API; recognition remains a separate controlled integration
- Swappable ANPR/OCR service boundary with validation and confidence handling
- Explainable risk scoring and severity classification
- Docker Compose and GitHub Actions CI

## Flying-object intelligence

SRT keeps airborne-object perception separate from the general CCTV detector. The primary specialist is `Javvanny/yolov8m_flying_objects_detection`, whose model card documents Drone, Airplane, Helicopter and Bird classes. The weights are downloaded through Hugging Face Hub and cached locally.

AeroYOLO (`QuincySorrentino/AeroYOLO`) is the classification-preserving fallback and exposes aircraft, drone and helicopter classes.

The application never converts a person/face/vehicle detection into an aerial detection through a geometry heuristic. Only a supported specialist-model class can reach the flying-object layer. A security alert is generated **only for `drone`**, and the same classified target must survive two specialist scans before the alert is emitted.

Useful diagnostics:

- `GET /api/v1/analytics/flying-objects` — current classified flying objects + model health
- `GET /api/v1/analytics/summary` — includes flying objects and drone alert counts

## Architecture

`CAMERA → INGESTION → PERCEPTION → TRACKING → SPATIAL/TEMPORAL CONTEXT → EVENT → RISK → ALERT → EVIDENCE → OPERATOR`

The AI layer is adapter-based so detector, tracker, flying-object classifier, face, ANPR and activity models can be upgraded without rewriting the application layer.

## Quick start — Windows

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
$env:PYTHONPATH="."
uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`.

## Camera workflows
1. **WEBCAM** — enter camera index `0`, add, then Start. The backend must run on the machine hosting the webcam.
2. **VIDEO FILE** — select MP4/MOV/MKV/WebM; SRT stores it under `storage/uploads` and processes it as a source.
3. **IP / RTSP** — enter the RTSP URL. Keep credentials out of screenshots, commits and logs.
4. **LOCAL PATH** — enter a path visible to the backend process.

## Virtual fence
Create a zone with normalized polygon points (`0..1`) through the API. Example:

```json
{
  "camera_id": "CAM-...",
  "name": "Restricted Border Area",
  "zone_type": "RESTRICTED",
  "polygon": [[0.10,0.10],[0.90,0.10],[0.90,0.90],[0.10,0.90]],
  "severity": "HIGH",
  "enabled": true
}
```

## API surface

`/api/v1/cameras` · `/api/v1/zones` · `/api/v1/events` · `/api/v1/alerts` · `/api/v1/analytics/summary` · `/api/v1/analytics/flying-objects` · `/api/v1/faces/detect` · `/api/v1/uploads/video` · `/api/v1/search` · `/ws/events` · `/ws/alerts`

## Validation note
The repository is implemented through the connected GitHub workflow, but the current environment cannot execute the project's local NVIDIA/CUDA, webcam or RTSP stack. GitHub Actions validates code-level tests/builds; final inference speed, GPU compatibility, camera permissions and a real RTSP source must be tested on the demo laptop.

## Safety and claims
SRT is an assistive surveillance analytics prototype. Risk scores are explainable indicators, not proof of criminal intent. Accuracy depends on camera resolution, lighting, scene geometry, model quality and hardware. Facial recognition and ANPR integrations require appropriate authorization, data governance and human oversight.
