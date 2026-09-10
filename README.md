# Project SRT — Smart Recognition & Tracking

AI-based intelligent video analytics platform for border surveillance using existing CCTV/IP-camera infrastructure.

## Phase 0 foundation
This branch establishes the backend application contract, configuration, logging, domain schemas, replaceable AI interfaces, video-source abstraction, Docker support, and tests. Concrete CV models are intentionally not selected here.

## Core philosophy
**SRT reasons about events, not merely objects.**

`PERCEPTION → DETECTION → TRACKING → CONTEXT → EVENT → ALERT → EVIDENCE → OPERATOR`

## Stack
- Python 3.11+
- FastAPI + Pydantic
- OpenCV-compatible video sources
- PyTorch/model adapters
- PostgreSQL-ready persistence boundary
- React/TypeScript frontend (next implementation stage)
- Docker / Compose

## Quick start
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Health: `GET /api/v1/system/health`
Docs: `/docs`

## Safety
SRT is an assistive surveillance analytics prototype. It does not guarantee detection, recognition, crime prediction, or prevention. Human operators remain responsible for decisions.
