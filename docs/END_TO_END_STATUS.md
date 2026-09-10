# End-to-End Build Status

## Implemented
- Multi-source ingestion: webcam, local file, uploaded video, RTSP/HTTP
- Server-side upload storage
- YOLO detector adapter with lazy model loading
- Stable track-ID adapter behind Tracker interface
- Thread-safe latest-frame store
- Browser MJPEG stream endpoint
- Normalized virtual-fence zones
- Restricted-zone intrusion event generation
- Event debounce to reduce duplicate alerts
- Explainable risk scoring
- Evidence JPEG persistence
- Event history and deterministic natural-language search APIs
- React command-center dashboard with explicit source modes
- Docker Compose and CI workflow

## Intentionally controlled / extension points
- Full ByteTrack implementation can replace the current lightweight tracker adapter.
- ANPR, face recognition, cross-camera identity and learned anomaly models require dedicated validated weights/datasets and are not represented as perfect capabilities.
- PostgreSQL/Redis services are architectural targets; the current demo event/camera/zone state is in-memory so the laptop demo has minimal setup.

## Demo acceptance path
1. Start backend and frontend.
2. Select WEBCAM and add source `0`, or upload a sample MP4.
3. Start the camera and verify the live MJPEG feed.
4. Create a RESTRICTED zone through `/docs`.
5. When a detected person enters the polygon, verify an INTRUSION event, risk score and evidence image.
6. Use `/api/v1/search?q=people in restricted areas` to demonstrate structured natural-language filtering.

## Validation boundary
GitHub Actions is configured for automated backend tests and frontend build. Real NVIDIA GPU inference, webcam access and RTSP interoperability must still be tested on the demo machine.
