# Project SRT — Current Status

## Phase 0 — Foundation
Completed: FastAPI application, versioned API prefix, Pydantic Settings, logging, canonical SRT event contracts, swappable AI interfaces, OpenCV video source abstraction, tests, and environment/git hygiene.

## Phase 1 — Camera & Video Processing
Implemented on `feat/phase1-camera-video`:
- Runtime camera registry with create/list/get/update/delete operations.
- Automatic source classification for local files, webcam indices, RTSP and HTTP sources.
- Camera lifecycle states: OFFLINE, CONNECTING, ONLINE, ERROR.
- Per-camera daemon worker with clean stop and application-shutdown handling.
- Frame ingestion through the existing `VideoSource` abstraction.
- Runtime FPS, frame count, last-frame timestamp and error tracking.
- Model-agnostic `process_frame` extension point for the detection/tracking phase.
- Versioned REST camera endpoints under `/api/v1/cameras`.
- API tests covering camera CRUD, source classification and 404 behavior.

## Important design boundary
Phase 1 deliberately does **not** select or embed a computer-vision model. Detection, tracking, event generation and persistence remain replaceable downstream components.

## Persistence
Camera state is currently in-memory. PostgreSQL/Redis persistence and distributed workers are intentionally deferred until the core processing pipeline is proven.

## Validation
The GitHub connector can inspect and commit source but cannot execute the repository's Python runtime. Run the test suite locally/CI before calling this phase fully validated.

## Next milestone
Phase 2: concrete person/vehicle detection and multi-object tracking adapters, using the approved model registry and preserving the existing abstraction boundaries.
