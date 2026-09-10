# Project SRT — Smart Recognition & Tracking

Project SRT is an AI-assisted intelligent video analytics platform for border-surveillance scenarios using existing CCTV/IP-camera infrastructure.

## What works in this branch
- Webcam input (`0` or another OpenCV camera index)
- Local video path evaluation
- Browser video upload to server storage
- RTSP / HTTP camera source ingestion
- Live browser-compatible MJPEG stream
- Person/vehicle detection through a lazy YOLO adapter
- Stable object track IDs through a replaceable tracking boundary
- Normalized polygon virtual fences
- Explainable intrusion events and risk scores
- Evidence JPEG capture for generated events
- In-memory event history API
- Deterministic natural-language-style event search
- React/TypeScript command-center dashboard
- Docker Compose and GitHub Actions CI

## Architecture

`CAMERA → INGESTION → PERCEPTION → TRACKING → SPATIAL/TEMPORAL CONTEXT → EVENT → RISK → EVIDENCE → OPERATOR`

The AI layer is deliberately adapter-based so detector, tracker, face, ANPR and activity models can be upgraded without rewriting the application layer.

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
1. **WEBCAM** — enter camera index `0`, add, then Start. This expects the backend to run on the machine hosting the webcam.
2. **VIDEO FILE** — select an MP4/MOV/MKV/WebM sample; SRT stores it under `storage/uploads` and can process it as a camera source.
3. **IP / RTSP** — enter the RTSP URL. Keep credentials out of screenshots, commits and logs.
4. **LOCAL PATH** — enter a path visible to the backend process.

## Virtual fence
Create a zone with normalized polygon points (`0..1`) using the API. Example:

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

## Important validation note
The repository can be edited and reviewed through the connected GitHub workflow, but this environment cannot execute the project's local GPU/RTSP stack. GitHub Actions will validate Python imports/tests and the frontend build. Final inference speed, NVIDIA/CUDA compatibility, webcam permissions and a real RTSP camera must be validated on the demo laptop.

## Safety and claims
SRT is an assistive surveillance analytics prototype. Risk scores are explainable indicators, not proof of criminal intent. Accuracy depends on camera resolution, lighting, scene geometry, model quality and hardware. Facial recognition, ANPR and advanced behavior models remain controlled extension points rather than claims of perfect recognition.
