# Phase 0 status

## Completed in `feat/phase0-foundation`
- FastAPI application with versioned API prefix.
- Environment configuration through Pydantic Settings and `.env` template.
- Structured application logging setup.
- Canonical SRT event, severity, alert and object contracts.
- Replaceable `Detector`, `Tracker`, `EventEngine` and `VideoSource` interfaces.
- OpenCV video-source implementation supporting local files, webcam indices and RTSP/HTTP-style sources.
- Root and health endpoint tests.
- Python dependency manifest and Git ignore rules.

## Validation
The repository connector can create and inspect source files but cannot execute the Python environment. Tests are therefore committed and must be run locally/CI before calling the foundation validated.

## Next milestone
Camera management + video processing service, followed by model research/selection and concrete detection/tracking adapters.
