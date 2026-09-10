# Project SRT — Smart Recognition & Tracking

AI-based intelligent video analytics platform for border surveillance using existing CCTV/IP-camera infrastructure.

## Core philosophy

**SRT reasons about events, not merely objects.**

Canonical pipeline:

`PERCEPTION → DETECTION → TRACKING → CONTEXT → EVENT → ALERT → EVIDENCE → OPERATOR`

## Foundation stack

- Python 3.11+
- FastAPI
- Pydantic
- OpenCV
- PyTorch/model adapters
- PostgreSQL
- Redis (optional in local foundation)
- React + TypeScript (frontend scaffold)
- Docker / Compose

## Repository structure

```text
backend/        API and application services
ai/             model adapters and analytics modules
frontend/       operator dashboard
infrastructure/ deployment and monitoring
scripts/        developer utilities
docs/           architecture, status, contracts and QA
storage/        local evidence storage (gitignored contents)
tests/          integration-level tests
```

## Current status

Phase 0 foundation initialized. Model selection is intentionally abstracted behind interfaces and will be recorded in `docs/MODEL_REGISTRY.md` after evaluation.

## Safety / scope

SRT is an assistive surveillance analytics prototype. It does not guarantee detection accuracy, identity accuracy, crime prediction, or prevention. Human operators remain responsible for decisions.

## Quick start

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Health check: `GET http://127.0.0.1:8000/api/v1/system/health`

Interactive API docs: `http://127.0.0.1:8000/docs`
