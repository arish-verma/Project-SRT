# AI Context

## Product
Project SRT (Smart Recognition & Tracking) is an AI-based video analytics platform for border surveillance using existing CCTV/IP-camera infrastructure.

## Objective
Turn standard video feeds into actionable, explainable events: detect objects, maintain tracks, reason about zones/time/context, generate alerts, preserve evidence, and expose an operator dashboard/API.

## Canonical pipeline
`PERCEPTION → DETECTION → TRACKING → CONTEXT → EVENT → ALERT → EVIDENCE → OPERATOR`

## Core features
Camera management, local video, RTSP architecture, person/vehicle detection, tracking, virtual fences, intrusion events, risk/severity, alerts, evidence frames, event history, dashboard, camera status and REST API.

## Planned extensions
ANPR/OCR, vehicle classification, face detection, night movement, loitering, suspicious-activity scoring, event clips, WebSockets, facial recognition, natural-language search, cross-camera tracking and advanced anomaly models.

## Engineering constraints
Python/FastAPI backend; React/TypeScript frontend; PostgreSQL; OpenCV/PyTorch; Docker. Keep dependencies practical and model adapters swappable. Do not introduce distributed infrastructure without evidence of need.

## Trust model
Rules and scores must be explainable. Low-confidence recognition/OCR must remain uncertain. SRT assists authorized human operators and does not replace human decision-making.
